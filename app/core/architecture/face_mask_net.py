import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch
from app.utils.tools import transform_image


class FaceMaskNet(nn.Module):
    def __init__(self, device: str = "cpu"):
        super(FaceMaskNet, self).__init__()
        self.device = device
        self.features = models.mobilenet_v2().features
        self.classifier = models.mobilenet_v2().classifier
        self.classifier[1] = nn.Linear(self.classifier[1].in_features, 2)

    def load_model(self, weight: str):
        state_dict = torch.load(
            weight, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)
        self.to(self.device)
        self.eval()

    def warm_up(self):
        for _ in range(10):
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            self.detect(dummy_image)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def detect(self, image: np.ndarray) -> bool:
        image = transform_image(image).unsqueeze(0).to(self.device)
        output = self.forward(image)
        _, pred = torch.max(output, 1)
        return bool(pred.item())
