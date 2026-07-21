from fastapi import APIRouter
from .endpoints.detect import router as detect_router
from .endpoints.analyze import router as analyze_router
from .endpoints.action import router as action_router

router = APIRouter()
router.include_router(detect_router, tags=["detect"])
router.include_router(analyze_router, tags=["analyze"])
router.include_router(action_router, tags=["action"])
