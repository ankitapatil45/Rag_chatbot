from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from chatbot.main import router as chatbot_router
from semantic.main import router as semantic_router

app = FastAPI(title="AI Services")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

app.include_router(chatbot_router, prefix="/chatbot", tags=["Chatbot"])
app.include_router(semantic_router, prefix="/semantic", tags=["Semantic"])

@app.get("/")
def root():
    return {
        "status": "Online",
        "chatbot_api": "/chatbot/ask",
        "semantic_api": "/semantic/search/similar",
        "docs": "/docs"
    }
