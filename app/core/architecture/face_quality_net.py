import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch
from app.utils.tools import transform_image


class FaceQualityNet(nn.Module):
    def __init__(self, device="cpu"):
        super(FaceQualityNet, self).__init__()
        self.device = device
        self.mobilenet = models.mobilenet_v2()
        num_ftrs = self.mobilenet.classifier[1].in_features

        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 1)
        )

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
        return self.mobilenet(x).squeeze()

    def detect(self, image: np.ndarray):
        image = transform_image(image).unsqueeze(0).to(self.device)
        return self.forward(image)
