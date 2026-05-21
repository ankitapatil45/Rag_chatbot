import sys
import re
import os
import time
from typing import Generator
from concurrent.futures import ThreadPoolExecutor

# ---------------- SQLite compatibility ----------------
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import ollama
import redis
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
MODEL = "tinyllama"
DISTANCE_THRESHOLD = 1.2
MAX_CONTEXT_CHARS = 2500
MEMORY_TTL = 3600

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ---------------- THREAD POOL ----------------
executor = ThreadPoolExecutor(max_workers=2)

# ---------------- INIT ----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
client = chromadb.PersistentClient(path=".chroma")
collection = client.get_or_create_collection("erp_docs")

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True
)

# ---------------- BASIC HELPERS ----------------
def is_greeting(text: str) -> bool:
    return bool(re.search(r"\b(hi|hello|hey|good morning|good evening)\b", text.lower()))

# ---------------- CONTEXT CLEANING ----------------
def clean_context(context: str) -> str:
    """
    Removes instruction-like or chatbot-related lines
    accidentally stored in documentation.
    """
    banned_patterns = [
        r"do not",
        r"rules:",
        r"if the question asks",
        r"reply exactly",
        r"answer the user",
        r"erp documentation assistant",
        r"system:",
        r"user question:",
        r"answer:"
    ]

    cleaned = []
    for line in context.splitlines():
        low = line.lower()
        if any(re.search(p, low) for p in banned_patterns):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)

# ---------------- RETRIEVAL ----------------
def retrieve_context(question: str):
    q_emb = embed_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3   # slightly better recall
    )

    # ❌ No results or weak match → reject
    if (
        not results.get("documents")
        or not results["documents"][0]
        or results["distances"][0][0] > DISTANCE_THRESHOLD
    ):
        return None

    # Join top documents
    raw_context = "\n".join(results["documents"][0])

    # ---------------- STRICT RELEVANCE FILTER ----------------
    question_tokens = {
        w for w in re.findall(r"[a-zA-Z]{3,}", question.lower())
    }

    filtered_paragraphs = []
    for para in raw_context.split("\n"):
        para_lower = para.lower()

        # paragraph must share at least 2 meaningful tokens
        overlap = sum(1 for w in question_tokens if w in para_lower)

        if overlap >= 2:
            filtered_paragraphs.append(para)

    # Fallback: if filtering removes everything, use raw context
    final_context = (
        "\n".join(filtered_paragraphs)
        if filtered_paragraphs
        else raw_context
    )

    return final_context[:MAX_CONTEXT_CHARS]


# ---------------- PROMPT BUILDER ----------------
def build_prompt(context: str, question: str) -> str:
    return f"""
SYSTEM:
Answer the question using only the software documentation provided below.

If the answer is not present in the documentation, respond exactly with:
This information is not available in the ERP documentation.

DOCUMENTATION:
{context}

QUESTION:
{question}

ANSWER:
"""

# ---------------- OUTPUT FORMATTER ----------------
def format_answer(text: str) -> str:
    """
    Converts raw model output into clean bullets or steps.
    """
    text = text.strip()
    if not text:
        return text

    # Split into logical chunks
    lines = [
        l.strip("•- ").strip()
        for l in re.split(r"\n+|\. ", text)
        if l.strip()
    ]

    step_words = (
        "go to", "click", "open", "select", "enter",
        "save", "navigate", "choose", "update"
    )

    is_process = any(
        any(w in line.lower() for w in step_words)
        for line in lines
    )

    if is_process:
        return "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
    else:
        return "\n".join(f"• {line}" for line in lines)

# ---------------- MAIN CHAT FUNCTION ----------------
def get_answer(user_id: str, question: str) -> Generator[str, None, None]:

    # Greeting
    if is_greeting(question):
        yield "Hello. How may I assist you today?\n"
        return

    # 🔥 Start streaming instantly
    yield ""

    # Retrieve context (non-blocking feel)
    future = executor.submit(retrieve_context, question)

    while not future.done():
        time.sleep(0.02)

    context = future.result()

    if not context:
        yield "This information is not available in the ERP documentation."
        return

    # Clean context (VERY IMPORTANT)
    context = clean_context(context)

    # Build minimal safe prompt
    prompt = f"""
SYSTEM:
Answer the question using only the software documentation provided below.

If the answer is not present in the documentation, respond exactly with:
This information is not available in the ERP documentation.

DOCUMENTATION:
{context}

QUESTION:
{question}

ANSWER:
"""

    # 🔥 REAL STREAMING
    stream = ollama.generate(
        model=MODEL,
        prompt=prompt,
        stream=True,
        options={
            "temperature": 0.0,
            "num_predict": 300,
            "top_p": 0.9
        }
    )

    # Stream tokens as they come
    for chunk in stream:
        token = chunk.get("response", "")
        if token:
            yield token
