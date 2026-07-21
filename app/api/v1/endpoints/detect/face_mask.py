from fastapi import APIRouter, HTTPException
from app.schemas.detect.face_mask import (
    FaceMaskRequest,
    FaceMaskResponse,
)
from app.services.detect.face_mask import predict_face_mask

router = APIRouter()


@router.post("/mask", response_model=FaceMaskResponse)
async def detect(request: FaceMaskRequest):
    """Endpoint for face mask detection"""
    try:
        return await predict_face_mask(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
