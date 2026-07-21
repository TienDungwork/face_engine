from fastapi import APIRouter, HTTPException
from app.schemas.action.update_face_engine import (
    UpdateFaceEngineRequest,
    UpdateFaceEngineResponse,
)
from app.services.action.update_face_engine_service import FaceEngineService

router = APIRouter()


@router.post("/update-face-engine", response_model=UpdateFaceEngineResponse)
async def update_face_engine(request: UpdateFaceEngineRequest):
    """Endpoint for face mask detection"""
    try:
        return await FaceEngineService.update_face_engine(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
