# ============================================================
# embed.py — Build / rebuild the ChromaDB vector store
#
# Key improvements over v1:
#   • Uses doc["id"] (SHA-256 hash) as ChromaDB ID instead of
#     sequential "id_N" — enables safe incremental upserts
#   • upsert() instead of add() — re-running won't crash on
#     existing IDs or create duplicates
#   • Progress reporting uses logger, not bare print()
#   • All constants from config.py
# ============================================================

import sys
import logging

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from sentence_transformers import SentenceTransformer

from chatbot.ingest import collect_documents
from chatbot.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    EMBED_DEVICE,
    INGEST_BATCH_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def run_ingestion(reset: bool = False) -> None:
    """
    Build (or rebuild) the ChromaDB vector store.

    Parameters
    ----------
    reset : bool
        If True, drop and recreate the collection before indexing.
        Use False for incremental updates (safe to run repeatedly).
    """
    logger.info("Initialising ChromaDB at %s …", CHROMA_PATH)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        logger.warning("reset=True — dropping existing collection")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
    logger.info("Collection '%s' has %d existing items", COLLECTION_NAME, collection.count())

    logger.info("Fetching documents from API…")
    raw_docs = collect_documents()

    if not raw_docs:
        logger.error("No documents returned — aborting.")
        return

    logger.info("Processing %d chunks…", len(raw_docs))

    embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

    texts     = [d["text"]   for d in raw_docs]
    metadatas = [{"module": d["module"], "path": d["path"], "has_ocr": d["has_ocr"]}
                 for d in raw_docs]
    ids       = [d["id"]     for d in raw_docs]   # stable SHA-256 hash

    logger.info("Generating embeddings (this may take a while)…")
    embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()

    logger.info("Upserting into ChromaDB in batches of %d…", INGEST_BATCH_SIZE)
    for i in range(0, len(texts), INGEST_BATCH_SIZE):
        end = i + INGEST_BATCH_SIZE
        collection.upsert(              # upsert = safe re-run
            documents=texts[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
            embeddings=embeddings[i:end],
        )
        logger.info("  Upserted %d – %d", i, min(end, len(texts)))

    logger.info("Done. Collection now contains %d items.", collection.count())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build ERP vector store")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate the collection before indexing")
    args = parser.parse_args()
    run_ingestion(reset=args.reset)