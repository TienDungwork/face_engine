import numpy as np
from app.schemas.detect.face_quality import FaceQualityRequest, FaceQualityResponse
from app.utils.helpers import base64_to_image
from app.core.model_manager import face_quality_model_manager


async def predict_face_quality(request: FaceQualityRequest) -> FaceQualityResponse:
    """
    Detect face quality.

    Args:
        request (FaceQualityRequest): Request containing base64 image

    Returns:
        FaceQualityResponse: Response containing face quality detection result
    """
    try:
        # Get image from request
        if request.base64_image:
            image = await base64_to_image(request.base64_image)
        else:
            return FaceQualityResponse(
                status_code=400,
                message="Need base64 image",
                quality=None
            )

        if image is None:
            return FaceQualityResponse(
                status_code=400,
                message="Invalid image format",
                quality=None
            )

        quality = await detect_quality(image)

        return FaceQualityResponse(
            status_code=200,
            message="Success",
            quality=quality
        )

    except Exception as e:
        return FaceQualityResponse(
            status_code=500,
            message=str(e),
            quality=None
        )


async def detect_quality(image: np.ndarray) -> float:
    quality_model = await face_quality_model_manager.get_model()
    return quality_model.detect(image)
