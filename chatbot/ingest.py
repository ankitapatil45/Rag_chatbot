# ============================================================
# ingest.py — Fetch, chunk, deduplicate and index ERP docs
#
# Key improvements over v1:
#   • crawl() extracted as a proper module-level function
#     (was a nested closure mutating outer scope — messy & untestable)
#   • Text chunking added — large blobs are split so retrieval
#     returns focused paragraphs instead of 2000-char walls
#   • SHA-256 deduplication — identical text won't be re-indexed
#   • OCR calls parallelised with ThreadPoolExecutor
#   • Bare `except:` replaced with `except Exception as e:`
#   • All magic numbers imported from config.py
# ============================================================

import re
import hashlib
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from chatbot.utils import encode_data
from chatbot.ocr_utils import extract_text_from_image
from chatbot.config import (
    API_URL,
    INGEST_BATCH_SIZE,
    MIN_TEXT_LENGTH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)

IMG_REGEX = re.compile(
    r'(https?://[^"\'> ]+\.(png|jpg|jpeg))',
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split long text into overlapping chunks for better retrieval.
    Single sentences shorter than `size` are returned as-is.

    Example with size=400, overlap=80:
      "...chunk 1 end [80 char tail]" becomes the start of chunk 2,
      so context is never lost at boundaries.
    """
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += size - overlap

    return chunks


def _doc_id(text: str) -> str:
    """Stable SHA-256 ID so identical chunks are never duplicated."""
    return hashlib.sha256(text.encode()).hexdigest()[:24]


# ─────────────────────────────────────────────────────────────
# OCR HELPERS
# ─────────────────────────────────────────────────────────────

def _extract_image_urls(raw_text: str) -> list[str]:
    return [url for url, _ in IMG_REGEX.findall(raw_text)]


def _parallel_ocr(urls: list[str], max_workers: int = 4) -> str:
    """
    Run OCR on multiple images in parallel.
    In v1 this was sequential — very slow for doc-heavy pages.
    """
    if not urls:
        return ""

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(extract_text_from_image, url): url for url in urls}
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                results[url] = future.result() or ""
            except Exception as e:
                logger.debug("OCR failed for %s: %s", url, e)
                results[url] = ""

    # Return in original URL order for deterministic output
    return " ".join(results.get(u, "") for u in urls).strip()


# ─────────────────────────────────────────────────────────────
# CRAWLER  (was a nested closure — now a proper function)
# ─────────────────────────────────────────────────────────────

def crawl(obj, path: str, module_name: str, seen_ids: set, out: list) -> None:
    """
    Recursively walk the API response tree.
    Extracts text + OCR, chunks, deduplicates, and appends to `out`.

    Parameters
    ----------
    obj        : dict | list | scalar — current node
    path       : breadcrumb string for the LOCATION field
    module_name: root module name for metadata
    seen_ids   : mutable set of already-seen chunk hashes
    out        : mutable list to append results to
    """
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, (dict, list)):
                crawl(v, path, module_name, seen_ids, out)
            elif isinstance(v, str):
                _process_string(v, path, module_name, seen_ids, out)

    elif isinstance(obj, list):
        for item in obj:
            crawl(item, path, module_name, seen_ids, out)


def _process_string(
    raw: str,
    path: str,
    module_name: str,
    seen_ids: set,
    out: list,
) -> None:
    if not raw or len(raw.strip()) < MIN_TEXT_LENGTH:
        return

    cleaned = clean_text(raw)

    # Parallelised OCR
    image_urls = _extract_image_urls(raw)
    ocr_text = _parallel_ocr(image_urls) if image_urls else ""

    combined = f"LOCATION: {path}\nTEXT:\n{cleaned}"
    if ocr_text:
        combined += f"\nOCR:\n{ocr_text}"

    # Chunk + deduplicate
    for chunk in chunk_text(combined):
        cid = _doc_id(chunk)
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        out.append({
            "id": cid,
            "text": chunk,
            "module": module_name,
            "path": path,
            "has_ocr": bool(ocr_text),
        })


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def collect_documents() -> list[dict]:
    """
    Fetch all ERP documentation from the remote API,
    crawl, chunk, and return deduplicated doc list.
    """
    final_docs: list[dict] = []
    seen_ids: set[str] = set()

    # Step 1 — fetch module list
    payload = {
        "GetPMInstituteModuleList": "true",
        "data": encode_data({}),
    }

    logger.info("Connecting to API for module list…")
    try:
        resp = requests.post(API_URL, data=payload, timeout=20)
        resp.raise_for_status()
        modules_data = resp.json().get("data", {})
    except Exception as e:
        logger.error("Failed to fetch module list: %s", e)
        return []

    logger.info("Found %d root modules", len(modules_data))

    # Step 2 — fetch and crawl each module
    for _, module in modules_data.items():
        module_name = module.get("module_name", "Unknown")
        module_id = module.get("moduleId")

        try:
            p = {
                "HelpDeskGetModuleWiseDemoPointList": "true",
                "data": encode_data({"moduleId": module_id}),
            }
            res = requests.post(API_URL, data=p, timeout=15)
            res.raise_for_status()
            deep_data = res.json().get("data", {})
        except Exception as e:
            logger.warning("Skipping module %s: %s", module_name, e)
            continue

        before = len(final_docs)
        crawl(deep_data, module_name, module_name, seen_ids, final_docs)
        logger.info("  %-30s → %d chunks", module_name, len(final_docs) - before)

    logger.info("Total deduplicated chunks collected: %d", len(final_docs))
    return final_docs