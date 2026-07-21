from fastapi import APIRouter, HTTPException

from app.schemas.analyze.search_face_events import (
    SearchFaceEventsRequest,
    SearchFaceEventsResponse,
)
from app.services.analyze.search_face_events import search_face_events

router = APIRouter()


@router.post("/searchFaceEvents", response_model=SearchFaceEventsResponse)
async def search_face_events_endpoint(request: SearchFaceEventsRequest):
    """Tìm event trong smf_face_events theo ảnh (similarity), có khoảng thời gian."""
    try:
        return await search_face_events(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
