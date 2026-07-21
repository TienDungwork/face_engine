from datetime import datetime
from app.schemas.action.update_face_engine import UpdateFaceEngineRequest, UpdateFaceEngineResponse


class FaceEngineService:
    @staticmethod
    async def update_face_engine(request: UpdateFaceEngineRequest) -> UpdateFaceEngineResponse:
        try:
            # Sync từ backend paging API → SQLite → load memory (giống sample periodic sync)
            from app.core.local_db import local_db_manager
            await local_db_manager.sync_persons()
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
