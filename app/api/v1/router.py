"""
API v1 Router
"""
from fastapi import APIRouter
from app.api.v1.endpoints import similarity

api_router = APIRouter()
api_router.include_router(similarity.router)
