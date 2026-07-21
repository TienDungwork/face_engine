from pydantic import BaseModel
from typing import Optional, List, Union, Any


class FaceSearchFeatureRequest(BaseModel):
    """Request model for face feature search."""
    featureInput: str
    featureSearch: List[str]

    class Config:
        arbitrary_types_allowed = True


class FaceSearchFeatureResponse(BaseModel):
    """Response model for face feature search."""
    timestamp: str
    status: int
    error: str
    data: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True
