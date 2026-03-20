from fastapi import APIRouter, HTTPException
from app.schemas.analyze.search_face import (
    FaceSearchRequest,
    FaceSearchResponse,
)
from app.services.analyze.search_face import search_face

router = APIRouter()


@router.post("/searchFace", response_model=FaceSearchResponse)
async def search_face_endpoint(request: FaceSearchRequest):
    """Endpoint for face detection and embedding"""
    try:
        return await search_face(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Prevent recursion by not wrapping the error multiple times
        raise HTTPException(status_code=500, detail=str(e))
