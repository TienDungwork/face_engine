from .detect import router as detect_router
from .compare import router as compare_router
from .search_face import router as search_face_router
from .search_face_feature import router as search_face_feature_router
from .search_face_events import router as search_face_events_router
from fastapi import APIRouter

router = APIRouter()

router.include_router(detect_router)
router.include_router(compare_router)
router.include_router(search_face_router)
router.include_router(search_face_feature_router)
router.include_router(search_face_events_router)
