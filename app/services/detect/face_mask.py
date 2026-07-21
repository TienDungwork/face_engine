import numpy as np
import torch
from app.schemas.detect.face_mask import FaceMaskRequest, FaceMaskResponse
from app.utils.helpers import base64_to_image
from app.core.model_manager import face_mask_model_manager


async def predict_face_mask(request: FaceMaskRequest) -> FaceMaskResponse:
    """
    Detect if a face is wearing a mask.

    Args:
        request (FaceDetectRequest): Request containing base64 image

    Returns:
        FaceDetectResponse: Response containing mask detection result
    """
    try:
        # Get image from request
        if request.base64_image:
            image = await base64_to_image(request.base64_image)
        else:
            return FaceMaskResponse(
                status_code=400,
                message="Need base64 image",
                is_mask=None
            )

        if image is None:
            return FaceMaskResponse(
                status_code=400,
                message="Invalid image format",
                is_mask=None
            )

        wear_mask = await detect_mask(image)

        return FaceMaskResponse(
            status_code=200,
            message="Success",
            is_mask=wear_mask
        )

    except Exception as e:
        return FaceMaskResponse(
            status_code=500,
            message=str(e),
            is_mask=None
        )


async def detect_mask(image: np.ndarray) -> bool:
    mask_model = await face_mask_model_manager.get_model()
    return mask_model.detect(image)
