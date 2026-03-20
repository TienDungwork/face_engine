from typing import Optional
from datetime import datetime
from app.schemas.action.update_face_engine import UpdateFaceEngineRequest, UpdateFaceEngineResponse
from app.core.database_manager import database_manager


class FaceEngineService:
    @staticmethod
    async def update_face_engine(request: UpdateFaceEngineRequest) -> UpdateFaceEngineResponse:
        try:
            await database_manager.load_persons()
            return UpdateFaceEngineResponse(
                timestamp=datetime.utcnow(),
                status=200,
                error="SUCCESS",
                data=None
            )
        except Exception as e:
            return UpdateFaceEngineResponse(
                timestamp=datetime.utcnow(),
                status=500,
                error=str(e),
                data=None
            )
