# ============================================================
# chatbot/main.py — APIRouter (mounted by root main.py)
#
# Exports `router` so the root main.py can do:
#   from chatbot.main import router as chatbot_router
#   app.include_router(chatbot_router, prefix="/chatbot")
# ============================================================

import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from chatbot.chatbot import get_answer
from chatbot.config import CHROMA_PATH, COLLECTION_NAME

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request schema ───────────────────────────────────────────
class ChatRequest(BaseModel):
    user_id: str
    question: str

    @field_validator("user_id")
    @classmethod
    def user_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id must not be empty")
        return v.strip()

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        if len(v) > 1000:
            raise ValueError("question must be under 1000 characters")
        return v.strip()


# ── Routes ───────────────────────────────────────────────────
@router.get("/health")
async def health_check():
    """Reports DB item count — useful as a readiness probe."""
    try:
        import sys
        try:
            __import__("pysqlite3")
            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        except ImportError:
            pass
        import chromadb
        c = chromadb.PersistentClient(path=CHROMA_PATH)
        count = c.get_or_create_collection(COLLECTION_NAME).count()
    except Exception:
        count = -1
    return {"status": "online", "indexed_docs": count}


@router.post("/ask")
async def ask(request: ChatRequest):
    """Stream the chatbot answer token-by-token."""
    logger.info("user=%s  q=%r", request.user_id, request.question[:80])
    return StreamingResponse(
        get_answer(request.user_id, request.question),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )