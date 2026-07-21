from pydantic import BaseModel
from typing import Optional


class FaceQualityRequest(BaseModel):
    """Request model for face quality detection."""
    base64_image: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class FaceQualityResponse(BaseModel):
    """Response model for face quality detection."""
    status_code: int
    message: str
    quality: Optional[float]

    class Config:
        arbitrary_types_allowed = True
