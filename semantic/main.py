from fastapi import APIRouter
from semantic.routes import search

router = APIRouter()
router.include_router(search.router, prefix="/search", tags=["Semantic Search"])
