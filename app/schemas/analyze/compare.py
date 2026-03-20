from pydantic import BaseModel
from typing import Optional


class FaceCompareRequest(BaseModel):
    """Request model for face comparison."""
    base64_1: Optional[str] = None
    base64_2: Optional[str] = None
    url_1: Optional[str] = None
    url_2: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class FaceCompareResponse(BaseModel):
    """Response model for face comparison."""
    status_code: int
    message: str
    data: Optional[dict]

    class Config:
        arbitrary_types_allowed = True
