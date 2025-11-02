"""
API v2 Router
"""
from fastapi import APIRouter
from app.api.v2.endpoints import similarity, upload

api_router = APIRouter()
api_router.include_router(similarity.router)
api_router.include_router(upload.router)
