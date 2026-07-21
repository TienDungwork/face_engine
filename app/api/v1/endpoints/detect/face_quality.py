from fastapi import APIRouter, HTTPException
from app.schemas.detect.face_quality import (
    FaceQualityRequest,
    FaceQualityResponse,
)
from app.services.detect.face_quality import predict_face_quality

router = APIRouter()


@router.post("/quality", response_model=FaceQualityResponse)
async def detect(request: FaceQualityRequest):
    """Endpoint for face quality detection"""
    try:
        return await predict_face_quality(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
