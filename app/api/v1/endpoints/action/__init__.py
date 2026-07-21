from .update_face_engine import router as update_face_engine_router
from fastapi import APIRouter

router = APIRouter()

router.include_router(update_face_engine_router)
