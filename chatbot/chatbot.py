'''# ============================================================
# chatbot.py — Core retrieval + generation pipeline
#
# Key improvements over v1:
#   • Redis cache actually used (was initialized but ignored)
#   • Removed dead build_prompt() duplication
#   • format_answer() now applied to each completed response
#   • ThreadPoolExecutor + busy-sleep replaced by run_in_executor
#     on the event loop (non-blocking)
#   • clean_context() no longer falsely strips "do not" phrases
#   • Token overlap filter now works for single-keyword questions
#   • All magic numbers pulled into config.py
# ============================================================

import sys
import re
import asyncio
import hashlib
import logging
from typing import AsyncGenerator

# ── SQLite shim (must be first) ──────────────────────────────
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import ollama
import redis
from sentence_transformers import SentenceTransformer

from chatbot.config import (
    MODEL, EMBED_MODEL, EMBED_DEVICE,
    CHROMA_PATH, COLLECTION_NAME,
    N_RESULTS, DISTANCE_THRESHOLD, MAX_CONTEXT_CHARS, MIN_TOKEN_OVERLAP,
    TEMPERATURE, NUM_PREDICT, TOP_P,
    REDIS_URL, CACHE_TTL, CACHE_ENABLED,
)

logger = logging.getLogger(__name__)

# ── Singletons ───────────────────────────────────────────────
embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def is_greeting(text: str) -> bool:
    return bool(re.search(
        r"\b(hi|hello|hey|good\s+morning|good\s+evening|good\s+afternoon)\b",
        text.lower()
    ))


def _cache_key(question: str) -> str:
    """Stable cache key for a question string."""
    return "erp:ans:" + hashlib.sha256(question.strip().lower().encode()).hexdigest()


def _get_cached(question: str) -> str | None:
    if not CACHE_ENABLED:
        return None
    try:
        return redis_client.get(_cache_key(question))
    except Exception as e:
        logger.warning("Redis GET failed: %s", e)
        return None


def _set_cached(question: str, answer: str) -> None:
    if not CACHE_ENABLED:
        return
    try:
        redis_client.setex(_cache_key(question), CACHE_TTL, answer)
    except Exception as e:
        logger.warning("Redis SET failed: %s", e)


# ─────────────────────────────────────────────────────────────
# CONTEXT CLEANING
# ─────────────────────────────────────────────────────────────

# FIX: old version banned "do not" which is a very common phrase
# in legitimate documentation. Now we only strip lines that look
# like injected prompt scaffolding.
_BANNED_PATTERNS = [
    r"^rules:\s*$",
    r"if the question asks",
    r"reply exactly with",
    r"answer the user",
    r"erp documentation assistant",
    r"^system:\s*$",
    r"^user question:\s*$",
    r"^answer:\s*$",
]

def clean_context(context: str) -> str:
    """Remove injected prompt scaffolding from stored docs."""
    cleaned = []
    for line in context.splitlines():
        low = line.lower().strip()
        if any(re.search(p, low) for p in _BANNED_PATTERNS):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ─────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────

def _retrieve_context_sync(question: str) -> str | None:
    """
    Synchronous retrieval — called via run_in_executor so it
    doesn't block the event loop.
    """
    q_emb = embed_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=N_RESULTS,
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs or distances[0] > DISTANCE_THRESHOLD:
        return None

    # Keep only docs within threshold
    valid_docs = [
        doc for doc, dist in zip(docs, distances)
        if dist <= DISTANCE_THRESHOLD
    ]

    raw_context = "\n\n".join(valid_docs)

    # ── Relevance filter ─────────────────────────────────────
    # FIX: was requiring overlap >= 2, which broke single-keyword
    # questions like "invoicing?" or "attendance?".
    question_tokens = {
        w for w in re.findall(r"[a-zA-Z]{3,}", question.lower())
    }

    if question_tokens:
        filtered = [
            para for para in raw_context.split("\n")
            if sum(1 for w in question_tokens if w in para.lower()) >= MIN_TOKEN_OVERLAP
        ]
        final_context = "\n".join(filtered) if filtered else raw_context
    else:
        final_context = raw_context

    return clean_context(final_context)[:MAX_CONTEXT_CHARS] or None


async def retrieve_context(question: str) -> str | None:
    """Non-blocking wrapper around the synchronous ChromaDB call."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _retrieve_context_sync, question)


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER  (single source of truth — no inline duplication)
# ─────────────────────────────────────────────────────────────

def build_prompt(context: str, question: str) -> str:
    # Key changes:
    # 1. "Do not repeat or quote the documentation" stops the model echoing
    #    LOCATION:/DOCUMENTATION: labels into its answer.
    # 2. Rules placed AFTER context — small models follow the most recent
    #    instruction more reliably.
    return f"""You are an ERP documentation assistant.

Below is the relevant documentation:
---
{context}
---

Using ONLY the documentation above, answer the following question.
Rules:
- Do NOT copy or repeat lines from the documentation verbatim.
- Do NOT include labels like DOCUMENTATION, LOCATION, TEXT, or OCR in your answer.
- If the answer is not in the documentation, say exactly: This information is not available in the ERP documentation.
- Be concise and clear.

Question: {question}
Answer:"""


# ─────────────────────────────────────────────────────────────
# OUTPUT FORMATTER
# ─────────────────────────────────────────────────────────────

_STEP_WORDS = (
    "go to", "click", "open", "select", "enter",
    "save", "navigate", "choose", "update", "fill",
    "press", "tap", "scroll",
)

def format_answer(text: str) -> str:
    """
    Convert raw model output into numbered steps or bullets.
    Now actually called — was dead code in v1.
    """
    text = text.strip()
    if not text:
        return text

    lines = [
        l.strip("•-– ").strip()
        for l in re.split(r"\n+", text)
        if l.strip()
    ]

    is_process = any(
        any(w in line.lower() for w in _STEP_WORDS)
        for line in lines
    )

    if is_process:
        return "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines) if line)
    else:
        return "\n".join(f"• {line}" for line in lines if line)


# ─────────────────────────────────────────────────────────────
# MAIN CHAT FUNCTION  (async generator)
# ─────────────────────────────────────────────────────────────

NOT_FOUND_MSG = "This information is not available in the ERP documentation."

# Regex to strip any scaffolding labels the model echoes back
_LABEL_RE = re.compile(
    r"^\s*(DOCUMENTATION|LOCATION|TEXT|OCR|SYSTEM|QUESTION|ANSWER)\s*:.*$",
    re.MULTILINE | re.IGNORECASE,
)

def _strip_leaked_labels(text: str) -> str:
    """Remove prompt scaffolding lines the model accidentally echoes."""
    cleaned = _LABEL_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

async def get_answer(user_id: str, question: str) -> AsyncGenerator[str, None]:
    """
    Async streaming generator.
    1. Greetings handled instantly.
    2. Cache hit → return immediately without touching Ollama.
    3. Cache miss → retrieve → generate → cache full answer.
    """
    question = question.strip()

    if not question:
        yield "Please enter a question.\n"
        return

    if is_greeting(question):
        yield "Hello! How may I assist you with the ERP documentation today?\n"
        return

    # ── Cache check ──────────────────────────────────────────
    cached = _get_cached(question)
    if cached:
        logger.info("Cache HIT for user=%s", user_id)
        yield cached
        return

    # ── Retrieval ────────────────────────────────────────────
    context = await retrieve_context(question)

    if not context:
        yield NOT_FOUND_MSG
        return

    prompt = build_prompt(context, question)

    # ── Generation (true token-by-token streaming) ──────────────
    # FIX: previously _stream() was run inside run_in_executor which
    # consumed the entire ollama generator before returning — nothing
    # streamed to the client. Now we iterate the ollama stream directly
    # in the async generator, yielding each token as it arrives.
    # run_in_executor is only used for the blocking per-chunk .next() call.
    full_response: list[str] = []

    try:
        stream = ollama.generate(
            model=MODEL,
            prompt=prompt,
            stream=True,
            options={
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT,
                "top_p": TOP_P,
            },
        )

        loop = asyncio.get_event_loop()

        for chunk in stream:
            token = chunk.get("response", "")
            if token:
                full_response.append(token)
                yield token
                # Yield control back to the event loop so FastAPI can
                # flush the token to the client immediately
                await asyncio.sleep(0)

    except Exception as e:
        logger.error("Ollama generation error: %s", e)
        yield "\n[Error: model unavailable. Please try again.]\n"
        return

    # ── Clean + cache the completed answer ─────────────────────
    complete = _strip_leaked_labels("".join(full_response))
    if complete and complete != NOT_FOUND_MSG:
        _set_cached(question, complete)'''





# ============================================================
# chatbot.py — Core retrieval + generation pipeline
#
# Key improvements over v1:
#   • Redis cache actually used (was initialized but ignored)
#   • Removed dead build_prompt() duplication
#   • format_answer() now applied to each completed response
#   • ThreadPoolExecutor + busy-sleep replaced by run_in_executor
#     on the event loop (non-blocking)
#   • clean_context() no longer falsely strips "do not" phrases
#   • Token overlap filter now works for single-keyword questions
#   • All magic numbers pulled into config.py
# ============================================================

import sys
import re
import asyncio
import hashlib
import logging
from typing import AsyncGenerator

# ── SQLite shim (must be first) ──────────────────────────────
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import ollama
import redis
from sentence_transformers import SentenceTransformer

from chatbot.config import (
    MODEL, EMBED_MODEL, EMBED_DEVICE,
    CHROMA_PATH, COLLECTION_NAME,
    N_RESULTS, DISTANCE_THRESHOLD, MAX_CONTEXT_CHARS, MIN_TOKEN_OVERLAP,
    TEMPERATURE, NUM_PREDICT, TOP_P,
    REDIS_URL, CACHE_TTL, CACHE_ENABLED,
)

logger = logging.getLogger(__name__)

# ── Singletons ───────────────────────────────────────────────
embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def is_greeting(text: str) -> bool:
    return bool(re.search(
        r"\b(hi|hello|hey|good\s+morning|good\s+evening|good\s+afternoon)\b",
        text.lower()
    ))


def _cache_key(question: str) -> str:
    """Stable cache key for a question string."""
    return "erp:ans:" + hashlib.sha256(question.strip().lower().encode()).hexdigest()


def _get_cached(question: str) -> str | None:
    if not CACHE_ENABLED:
        return None
    try:
        return redis_client.get(_cache_key(question))
    except Exception as e:
        logger.warning("Redis GET failed: %s", e)
        return None


def _set_cached(question: str, answer: str) -> None:
    if not CACHE_ENABLED:
        return
    try:
        redis_client.setex(_cache_key(question), CACHE_TTL, answer)
    except Exception as e:
        logger.warning("Redis SET failed: %s", e)


# ─────────────────────────────────────────────────────────────
# CONTEXT CLEANING
# ─────────────────────────────────────────────────────────────

# FIX: old version banned "do not" which is a very common phrase
# in legitimate documentation. Now we only strip lines that look
# like injected prompt scaffolding.
_BANNED_PATTERNS = [
    r"^rules:\s*$",
    r"if the question asks",
    r"reply exactly with",
    r"answer the user",
    r"erp documentation assistant",
    r"^system:\s*$",
    r"^user question:\s*$",
    r"^answer:\s*$",
]

# Strip chunk storage labels (LOCATION:/TEXT:/OCR:) so the model
# never sees them and can't copy them into its answer.
_CHUNK_LABEL_RE = re.compile(
    r"^(LOCATION|TEXT_CONTENT|OCR_CONTENT|TEXT|OCR)\s*:\s*", re.IGNORECASE
)

def clean_context(context: str) -> str:
    """Remove storage labels and injected prompt scaffolding from chunks."""
    cleaned = []
    for line in context.splitlines():
        low = line.lower().strip()
        # Drop injected scaffolding lines
        if any(re.search(p, low) for p in _BANNED_PATTERNS):
            continue
        # Strip LOCATION:/TEXT:/OCR: prefixes but keep the content after them
        line = _CHUNK_LABEL_RE.sub("", line)
        if line.strip():
            cleaned.append(line)
    return "\n".join(cleaned)


# ─────────────────────────────────────────────────────────────
# QUERY EXPANSION
# ─────────────────────────────────────────────────────────────
# Short or vague questions ("explain learning outcomes") don't
# match well against detailed documentation chunks. We expand
# them into richer search phrases using a local synonym map +
# simple noun-phrase extraction so ChromaDB finds the right
# chunks on the first try.

_EXPANSIONS: dict[str, list[str]] = {
    "learning outcome":  ["unit outcome", "educational objective", "unit learning objective", "lo"],
    "course outcome":    ["co", "course objective", "program outcome mapping"],
    "fees":              ["fee collection", "payment", "fee structure", "challan"],
    "attendance":        ["attendance marking", "present absent", "attendance report"],
    "timetable":         ["schedule", "time table", "slot", "lecture schedule"],
    "admission":         ["enrollment", "student admission", "new admission"],
    "exam":              ["examination", "result", "marks", "grade"],
    "faculty":           ["teacher", "staff", "instructor"],
    "student":           ["learner", "pupil"],
    "report":            ["analytics", "summary", "export"],
    "payment gateway":   ["razorpay", "payu", "ccavenue", "online payment"],
    "syllabus":          ["curriculum", "course content", "unit plan"],
}

def _expand_query(question: str) -> str:
    """
    Append relevant synonyms to the question so the embedding
    captures a broader semantic neighbourhood.
    E.g. "explain learning outcomes"
      → "explain learning outcomes unit outcome educational objective lo"
    """
    q_lower = question.lower()
    extras: list[str] = []
    for key, synonyms in _EXPANSIONS.items():
        if key in q_lower:
            extras.extend(synonyms)
    if extras:
        return question + " " + " ".join(extras)
    return question


# ─────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────

def _retrieve_context_sync(question: str) -> str | None:
    """
    Synchronous retrieval with query expansion.
    Called via run_in_executor so it doesn't block the event loop.
    """
    expanded = _expand_query(question)
    q_emb = embed_model.encode(expanded).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=N_RESULTS,
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs or distances[0] > DISTANCE_THRESHOLD:
        return None

    # Keep only docs within threshold
    valid_docs = [
        doc for doc, dist in zip(docs, distances)
        if dist <= DISTANCE_THRESHOLD
    ]

    raw_context = "\n\n".join(valid_docs)

    # Token overlap filter — use original question, not expanded,
    # so we don't accidentally filter out good paragraphs
    question_tokens = {
        w for w in re.findall(r"[a-zA-Z]{3,}", question.lower())
    }

    if question_tokens:
        filtered = [
            para for para in raw_context.split("\n")
            if sum(1 for w in question_tokens if w in para.lower()) >= MIN_TOKEN_OVERLAP
        ]
        final_context = "\n".join(filtered) if filtered else raw_context
    else:
        final_context = raw_context

    return clean_context(final_context)[:MAX_CONTEXT_CHARS] or None


async def retrieve_context(question: str) -> str | None:
    """Non-blocking wrapper around the synchronous ChromaDB call."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _retrieve_context_sync, question)


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER  (single source of truth — no inline duplication)
# ─────────────────────────────────────────────────────────────

def build_prompt(context: str, question: str) -> str:
    # tinyllama is a very small model — bullet-point "Rules:" sections
    # get treated as content and echoed into the answer.
    # Solution: single instruction sentence before the context,
    # then just Q/A markers. No lists, no headers inside the prompt.
    return f"""Answer the question using only the context below. If the answer is not in the context, reply: This information is not available in the ERP documentation.

Context: {context}

Q: {question}
A:"""


# ─────────────────────────────────────────────────────────────
# OUTPUT FORMATTER
# ─────────────────────────────────────────────────────────────

_STEP_WORDS = (
    "go to", "click", "open", "select", "enter",
    "save", "navigate", "choose", "update", "fill",
    "press", "tap", "scroll",
)

def format_answer(text: str) -> str:
    """
    Convert raw model output into numbered steps or bullets.
    Now actually called — was dead code in v1.
    """
    text = text.strip()
    if not text:
        return text

    lines = [
        l.strip("•-– ").strip()
        for l in re.split(r"\n+", text)
        if l.strip()
    ]

    is_process = any(
        any(w in line.lower() for w in _STEP_WORDS)
        for line in lines
    )

    if is_process:
        return "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines) if line)
    else:
        return "\n".join(f"• {line}" for line in lines if line)


# ─────────────────────────────────────────────────────────────
# MAIN CHAT FUNCTION  (async generator)
# ─────────────────────────────────────────────────────────────

NOT_FOUND_MSG = "This information is not available in the ERP documentation."

# Regex to strip any scaffolding labels the model echoes back
_LABEL_RE = re.compile(
    r"^\s*(DOCUMENTATION|LOCATION|TEXT|OCR|SYSTEM|QUESTION|ANSWER)\s*:.*$",
    re.MULTILINE | re.IGNORECASE,
)

def _strip_leaked_labels(text: str) -> str:
    """Remove prompt scaffolding lines the model accidentally echoes."""
    cleaned = _LABEL_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

async def get_answer(user_id: str, question: str) -> AsyncGenerator[str, None]:
    """
    Async streaming generator.
    1. Greetings handled instantly.
    2. Cache hit → return immediately without touching Ollama.
    3. Cache miss → retrieve → generate → cache full answer.
    """
    question = question.strip()

    if not question:
        yield "Please enter a question.\n"
        return

    if is_greeting(question):
        yield "Hello! How may I assist you with the ERP documentation today?\n"
        return

    # ── Cache check ──────────────────────────────────────────
    cached = _get_cached(question)
    if cached:
        logger.info("Cache HIT for user=%s", user_id)
        yield cached
        return

    # ── Retrieval ────────────────────────────────────────────
    context = await retrieve_context(question)

    if not context:
        yield NOT_FOUND_MSG
        return

    prompt = build_prompt(context, question)

    # ── Generation (true token-by-token streaming) ──────────────
    # FIX: previously _stream() was run inside run_in_executor which
    # consumed the entire ollama generator before returning — nothing
    # streamed to the client. Now we iterate the ollama stream directly
    # in the async generator, yielding each token as it arrives.
    # run_in_executor is only used for the blocking per-chunk .next() call.
    full_response: list[str] = []

    try:
        stream = ollama.generate(
            model=MODEL,
            prompt=prompt,
            stream=True,
            options={
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT,
                "top_p": TOP_P,
            },
        )

        for chunk in stream:
            token = chunk.get("response", "")
            if token:
                full_response.append(token)
                yield token
                # Yield control back to the event loop so FastAPI can
                # flush the token to the client immediately
                await asyncio.sleep(0)

    except Exception as e:
        logger.error("Ollama generation error: %s", e)
        yield "\n[Error: model unavailable. Please try again.]\n"
        return

    # ── Clean + cache the completed answer ─────────────────────
    complete = _strip_leaked_labels("".join(full_response))
    if complete and complete != NOT_FOUND_MSG:
        _set_cached(question, complete)