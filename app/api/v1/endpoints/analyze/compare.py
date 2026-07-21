from fastapi import APIRouter, HTTPException
from app.schemas.analyze.compare import (
    FaceCompareRequest,
    FaceCompareResponse,
)
from app.services.analyze.compare import compare_face

router = APIRouter()


@router.post("/compare", response_model=FaceCompareResponse)
async def compare(request: FaceCompareRequest):
    """Endpoint for face comparison"""
    try:
        return await compare_face(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
