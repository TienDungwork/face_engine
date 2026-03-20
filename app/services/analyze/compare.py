import numpy as np
from app.schemas.analyze.compare import FaceCompareRequest, FaceCompareResponse
from app.utils.helpers import base64_to_image, download_image
from app.core.model_manager import insightface_model_manager
from app.utils.helpers import cosine_similarity


async def compare_face(request: FaceCompareRequest) -> FaceCompareResponse:
    """
    Compare two faces and calculate similarity score.

    Args:
        request (FaceDetectRequest): Request containing either base64_image or url_image

    Returns:
        FaceDetectResponse: Response containing face detection results
    """
    try:
        # Get and validate image
        image_1 = await _get_valid_image(request.base64_1, request.url_1)
        image_2 = await _get_valid_image(request.base64_2, request.url_2)
        if isinstance(image_1, FaceCompareResponse) or isinstance(image_2, FaceCompareResponse):
            return image_1 or image_2

        # Detect faces
        faces_1 = await _detect_faces(image_1)
        faces_2 = await _detect_faces(image_2)
        if isinstance(faces_1, FaceCompareResponse) or isinstance(faces_2, FaceCompareResponse):
            return faces_1 or faces_2

        # Process largest face
        largest_face_1 = max(faces_1, key=lambda x:
                             (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        largest_face_2 = max(faces_2, key=lambda x:
                             (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        similarity = _compare_faces(
            largest_face_1.embedding, largest_face_2.embedding)
        similarity = min(similarity * 2, 1)
        return FaceCompareResponse(
            status_code=200,
            message="Success",
            data={
                "similarity": similarity,
            }
        )

    except Exception as e:
        return FaceCompareResponse(
            status_code=500,
            message=str(e),
            data=None
        )


async def _get_valid_image(base64_image: str, url_image: str):
    """Get and validate image from request."""
    if base64_image:
        image = await base64_to_image(base64_image)
    elif url_image:
        image = await download_image(url_image)
    else:
        return FaceCompareResponse(
            status_code=400,
            message="Need a valid image",
            data=None
        )

    if image is None:
        return FaceCompareResponse(
            status_code=400,
            message="Invalid image format",
            data=None
        )

    return image


async def _detect_faces(image):
    """Detect faces in image using InsightFace asynchronously."""
    model = await insightface_model_manager.get_model()
    faces = model.detect(image)

    if not faces:
        return FaceCompareResponse(
            status_code=400,
            message="No face detected",
            data=None
        )

    return faces


def _compare_faces(embedding_1: np.ndarray, embedding_2: np.ndarray) -> float:
    """Calculate face similarity score asynchronously."""
    similarity = cosine_similarity(embedding_1, embedding_2)
    # Convert numpy.float32 to Python float
    return float(similarity)
