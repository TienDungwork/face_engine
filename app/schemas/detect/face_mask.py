from pydantic import BaseModel
from typing import Optional


class FaceMaskRequest(BaseModel):
    """Request model for face detection."""
    base64_image: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class FaceMaskResponse(BaseModel):
    """Response model for face detection."""
    status_code: int
    message: str
    is_mask: Optional[bool]

    class Config:
        arbitrary_types_allowed = True
