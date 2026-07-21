import numpy as np
import torch
import cv2
from app.schemas.analyze.detect import (
    FaceDetectRequest,
    FaceDetectResponse
)
from app.utils.helpers import base64_to_image, download_image
from app.core.model_manager import (
    insightface_model_manager
)
from app.utils.helpers import encode_embedding
from datetime import datetime
from app.utils.helpers import expand_image


async def detect_face(request: FaceDetectRequest) -> FaceDetectResponse:
    """
    Detect face and extract embedding.

    Args:
        request (FaceDetectRequest): Request containing either base64_image or url_image

    Returns:
        FaceDetectResponse: Response containing face detection results
    """
    try:
        # Get and validate image
        image = await _get_valid_image(request)
        if isinstance(image, FaceDetectResponse):
            return image

        # Detect faces
        faces = await _detect_faces(image)
        if isinstance(faces, FaceDetectResponse):
            return faces

        # Process largest face
        largest_face = max(faces, key=lambda x:
                           (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        # Get face rectangle and crop
        face_rectangle = [int(x) for x in largest_face.bbox]
        expanded_face_rectangle = expand_image(face_rectangle, image)
        face_crop = image[face_rectangle[1]:face_rectangle[3],
                          face_rectangle[0]:face_rectangle[2]]

        # Get embedding vector
        embedding_vector = largest_face.embedding

        return FaceDetectResponse(
            timestamp=datetime.now().isoformat(),
            status=200,
            error="Success",
            data={
                "thumbnail": request.img_base64,
                "feature": encode_embedding(embedding_vector),
                "face_rectangle": expanded_face_rectangle,
                "quality": 1.0,  # Default quality score
                "wearmask": 0,   # Default mask status
                "decode_time": 0,
                "process_time": 0
            }
        )

    except Exception as e:
        return FaceDetectResponse(
            timestamp=datetime.now().isoformat(),
            status=500,
            error=str(e),
            data=None
        )


async def _get_valid_image(request: FaceDetectRequest):
    """Get and validate image from request."""
    if request.img_base64:
        image = await base64_to_image(request.img_base64)
    elif request.img_url:
        image = await download_image(request.img_url)
    else:
        return FaceDetectResponse(
            timestamp=datetime.now().isoformat(),
            status=400,
            error="Need a valid image",
            data=None
        )

    if image is None:
        return FaceDetectResponse(
            timestamp=datetime.now().isoformat(),
            status=400,
            error="Invalid image format",
            data=None
        )

    return image


async def _detect_faces(image):
    """Detect faces in image using InsightFace asynchronously."""
    model = await insightface_model_manager.get_model()
    faces = model.detect(image)

    if not faces:
        return FaceDetectResponse(
            timestamp=datetime.now().isoformat(),
            status=400,
            error="No face detected",
            data=None
        )

    return faces
