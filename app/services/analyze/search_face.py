from app.schemas.analyze.search_face import (
    FaceSearchRequest,
    FaceSearchResponse
)
from app.utils.helpers import base64_to_image, download_image
from app.core.model_manager import (
    insightface_model_manager
)
from app.utils.helpers import encode_embedding
from datetime import datetime
from app.core.database_manager import database_manager


async def search_face(request: FaceSearchRequest) -> FaceSearchResponse:
    """
    Detect face and extract embedding for search.

    Args:
        request (FaceSearchRequest): Request containing either base64_image or url_image

    Returns:
        FaceSearchResponse: Response containing face search results
    """
    try:
        # Get and validate image
        image = await _get_valid_image(request)
        if isinstance(image, FaceSearchResponse):
            return image

        # Detect faces
        faces = await _detect_faces(image)
        if isinstance(faces, FaceSearchResponse):
            return faces

        # Process largest face
        largest_face = max(faces, key=lambda x:
                           (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        # Get face rectangle and crop
        face_rectangle = [int(x) for x in largest_face.bbox]

        # Get embedding vector
        embedding_vector = largest_face.embedding

        # Search for persons in the database
        persons, similar_person = await database_manager.search_person(
            embedding_vector, threshold=request.threshold, company_ids=request.company_ids)
        # Return the results

        return FaceSearchResponse(
            timestamp=datetime.now().isoformat(),
            status=200,
            error="Success",
            data={
                "input_data": {
                    "featureBase64": encode_embedding(embedding_vector),
                    "face_rectangle": face_rectangle,
                    "quality": 0.5,  # Default quality score
                    "wearmask": 0,   # Default mask status
                    "decode_time": 0,
                    "process_time": 0,
                    "gender": 0,
                    "age": 0
                },
                "result": persons,
                "similar": similar_person
            }
        )

    except Exception as e:
        return FaceSearchResponse(
            timestamp=datetime.now().isoformat(),
            status=500,
            error=str(e),
            data=None
        )


async def _get_valid_image(request: FaceSearchRequest):
    """Get and validate image from request."""
    if request.img_base64:
        image = await base64_to_image(request.img_base64)
    elif request.img_url:
        image = await download_image(request.img_url)
    else:
        return FaceSearchResponse(
            timestamp=datetime.now().isoformat(),
            status=400,
            error="Need a valid image",
            data=None
        )

    if image is None:
        return FaceSearchResponse(
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
        return FaceSearchResponse(
            timestamp=datetime.now().isoformat(),
            status=400,
            error="Face not found in this image",
            data=None
        )

    return faces
