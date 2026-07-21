import cv2
import numpy as np
from datetime import datetime

from app.core.database_manager import database_manager
from app.core.model_manager import insightface_model_manager
from app.schemas.analyze.search_event import EventSearchRequest, EventSearchResponse
from app.utils.helpers import base64_to_image


async def search_event(request: EventSearchRequest) -> EventSearchResponse:
    """Search events by face embedding from input base64 image."""
    try:
        image = await _get_valid_image(request)
        if isinstance(image, EventSearchResponse):
            return image

        faces = await _detect_faces(image)
        if isinstance(faces, EventSearchResponse):
            return faces

        largest_face = max(
            faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )

        face_rectangle = [int(x) for x in largest_face.bbox]
        face_crop = image[
            face_rectangle[1] : face_rectangle[3], face_rectangle[0] : face_rectangle[2]
        ]
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        embedding_vector = largest_face.embedding

        wear_mask = await _detect_mask(face_crop_rgb)
        events = database_manager.search_event(
            embedding_vector,
            company_ids=request.company_ids,
            threshold=request.threshold,
            num_result=100,
            wear_mask=wear_mask,
            from_date=request.fromDate,
            to_date=request.toDate,
        )

        return EventSearchResponse(
            timestamp=datetime.now().isoformat(),
            status=200,
            error="Success",
            data=events,
        )
    except Exception as e:
        return EventSearchResponse(
            timestamp=datetime.now().isoformat(),
            status=500,
            error=str(e),
            data=[],
        )


async def _get_valid_image(request: EventSearchRequest):
    """Decode and validate image from request (base64 only)."""
    if request.img_base64:
        image = await base64_to_image(request.img_base64)
    else:
        return EventSearchResponse(
            timestamp=datetime.now().isoformat(),
            status=400,
            error="Need img_base64",
            data=[],
        )

    if image is None:
        return EventSearchResponse(
            timestamp=datetime.now().isoformat(),
            status=400,
            error="Invalid image format",
            data=[],
        )
    return image


async def _detect_faces(image):
    """Detect faces in image."""
    model = await insightface_model_manager.get_model()
    faces = model.detect(image)
    if not faces:
        return EventSearchResponse(
            timestamp=datetime.now().isoformat(),
            status=400,
            error="Face not found in this image",
            data=[],
        )
    return faces


async def _detect_mask(face_crop: np.ndarray) -> int:
    """Mask model has been removed; default to no mask."""
    return 0
