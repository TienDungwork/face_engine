import numpy as np
from torchvision import transforms
import torch


def transform_image(image: np.ndarray) -> torch.Tensor:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])(image)
