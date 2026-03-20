from fastapi import APIRouter, HTTPException
from app.schemas.analyze.search_face_feature import (
    FaceSearchFeatureRequest,
    FaceSearchFeatureResponse,
)
from app.services.analyze.search_face_feature import search_face_feature

router = APIRouter()


@router.post("/searchFaceFeature", response_model=FaceSearchFeatureResponse)
async def search_face_feature_endpoint(request: FaceSearchFeatureRequest):
    """Endpoint for face detection and embedding"""
    try:
        return await search_face_feature(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Prevent recursion by not wrapping the error multiple times
        raise HTTPException(status_code=500, detail=str(e))
