from app.schemas.analyze.search_face_feature import (
    FaceSearchFeatureRequest,
    FaceSearchFeatureResponse
)
from app.utils.helpers import decode_embedding, cosine_similarity
from datetime import datetime


async def search_face_feature(request: FaceSearchFeatureRequest) -> FaceSearchFeatureResponse:
    """
    Search for persons in the database using face features.

    Args:
        request (FaceSearchFeatureRequest): Request containing face features to search for.
            Can be either a single feature or a list of features.

    Returns:
        FaceSearchFeatureResponse: Response containing face search results with cosine similarities
    """
    try:
        # Validate and decode base feature
        if not isinstance(request.featureInput, str):
            return _create_error_response(400, "Invalid feature format")

        try:
            base_feature = decode_embedding(request.featureInput)
        except Exception as e:
            return _create_error_response(400, f"Invalid feature format: {e}")

        # Validate and decode search features
        if not isinstance(request.featureSearch, list):
            return _create_error_response(400, "Invalid feature format")

        features = []
        for feature in request.featureSearch:
            if not feature:
                continue
            try:
                decoded_feature = decode_embedding(feature)
                features.append(decoded_feature)
            except Exception as e:
                return _create_error_response(400, f"Invalid feature format: {e}")

        # Calculate similarities
        similarities = [
            cosine_similarity(base_feature, feature)
            for feature in features
        ]

        return FaceSearchFeatureResponse(
            timestamp=datetime.now().isoformat(),
            status=200,
            error="Success",
            data=similarities
        )

    except Exception as e:
        return _create_error_response(500, str(e))


def _create_error_response(status: int, error: str) -> FaceSearchFeatureResponse:
    """Helper function to create error responses."""
    return FaceSearchFeatureResponse(
        timestamp=datetime.now().isoformat(),
        status=status,
        error=error,
        data=[]
    )
