from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot.chatbot import get_answer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

router_app = FastAPI(title="ERP Documentation Bot")

class ChatRequest(BaseModel):
    user_id: str
    question: str

from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "Online"}

@router.post("/ask")
async def ask(request: ChatRequest):
    if not request.user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    return StreamingResponse(
        get_answer(request.user_id, request.question),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
