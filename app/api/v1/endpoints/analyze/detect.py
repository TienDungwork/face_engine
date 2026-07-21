from fastapi import APIRouter, HTTPException
from app.schemas.analyze.detect import (
    FaceDetectRequest,
    FaceDetectResponse,
)
from app.services.analyze.detect import detect_face

router = APIRouter()


@router.post("/detect", response_model=FaceDetectResponse)
async def detect(request: FaceDetectRequest):
    """Endpoint for face detection and embedding"""
    try:
        return await detect_face(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
