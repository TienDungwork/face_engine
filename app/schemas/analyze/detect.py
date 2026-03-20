from pydantic import BaseModel
from typing import Optional, List


class FaceDetectRequest(BaseModel):
    """Request model for face detection."""
    img_base64: Optional[str] = None
    img_url: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class FaceDetectResponse(BaseModel):
    """Response model for face detection."""
    timestamp: str
    status: int
    error: str
    data: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True


class FaceDetectData(BaseModel):
    """Data model for face detection response."""
    feature: str
    face_rectangle: List[int]
