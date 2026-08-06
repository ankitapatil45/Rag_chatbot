# ============================================================
# config.py — Central configuration for the ERP Chatbot
# All tuneable constants live here. Never scatter magic
# numbers across files.
# ============================================================

import os

# ── Model ────────────────────────────────────────────────────
MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DEVICE = "cpu"

# ── Retrieval ────────────────────────────────────────────────
CHROMA_PATH = ".chroma"
COLLECTION_NAME = "erp_docs"
N_RESULTS = 5                  # fetch more, then filter down
DISTANCE_THRESHOLD = 1.0       # tighter than 1.2; lower = stricter
MAX_CONTEXT_CHARS = 3000
MIN_TOKEN_OVERLAP = 1          # was 2 — single-keyword queries now work

# ── Generation ───────────────────────────────────────────────
TEMPERATURE = 0.0
NUM_PREDICT = 400              # slightly more room for longer answers
TOP_P = 0.9

# ── Redis / Cache ────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 3600               # seconds — cache answers for 1 hour
CACHE_ENABLED = False          # flip to False to disable during dev

# ── Ingestion ────────────────────────────────────────────────
API_URL = "https://preprod.vmedulife.com/api/helpDesk/documentationPublicData.php"
INGEST_BATCH_SIZE = 5000
MIN_TEXT_LENGTH = 30           # ignore tiny fragments
CHUNK_SIZE = 400               # chars per chunk after splitting
CHUNK_OVERLAP = 80             # overlap between consecutive chunks

# ── OCR ──────────────────────────────────────────────────────
OCR_MAX_IMAGE_DIM = 2000       # resize if larger (speeds up Tesseract)
OCR_TIMEOUT = 7                # HTTP timeout for image download