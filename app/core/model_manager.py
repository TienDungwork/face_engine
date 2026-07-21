from app.core.config import app_config
from app.core.architecture import FaceRecognize
import asyncio


class BaseModelManager:
    def __init__(self, pool_size: int = 4):
        self.model_pool = []
        self.pool_size = pool_size
        self._current = 0
        self._lock = asyncio.Lock()
        self.device = None

    async def get_model(self):
        async with self._lock:
            if not self.model_pool:
                return None
            model = self.model_pool[self._current]
            self._current = (self._current + 1) % self.pool_size
            return model

    async def close_model(self):
        for model in self.model_pool:
            del model
        self.model_pool = []


class InsightFaceManager(BaseModelManager):
    def __init__(self):
        super().__init__(pool_size=2)

    async def load_model(self):
        config = app_config.insightface
        for _ in range(self.pool_size):
            model = FaceRecognize(
                device_id=config.device_id, det_size=config.det_size)
            model.load_model()
            model.warm_up()
            self.model_pool.append(model)
        print(f"Loaded {len(self.model_pool)} insight face models")


insightface_model_manager = InsightFaceManager()
