from insightface.app import FaceAnalysis
import numpy as np


class FaceRecognize:
    def __init__(self, device_id: int = -1, det_size: tuple[int, int] = (640, 640)):
        self.device_id = device_id
        self.det_size = det_size
        if device_id != -1:
            self.providers = [('CUDAExecutionProvider', {
                'device_id': device_id,
                'use_tf32': '1',
            }), 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']

    def load_model(self):
        self.app = FaceAnalysis(providers=self.providers, allowed_modules=[
                                "detection", "recognition"])
        self.app.prepare(ctx_id=0, det_size=self.det_size)

    def warm_up(self):
        for _ in range(10):
            dummy_image = np.zeros((320, 320, 3), dtype=np.uint8)
            self.detect(dummy_image)

    def detect(self, image: np.ndarray) -> bool:
        return self.app.get(image)
