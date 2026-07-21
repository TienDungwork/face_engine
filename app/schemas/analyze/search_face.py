from pydantic import BaseModel, Field, AliasChoices
from typing import Optional, List, Dict, Any


class FaceSearchRequest(BaseModel):
    """Request model for face search."""
    img_base64: Optional[str] = None
    img_url: Optional[str] = None
    quality: Optional[float] = None
    company_ids: Optional[List[int]] = Field(
        default=None,
        validation_alias=AliasChoices(
            "company_ids", "companyIds", "compIds", "comIds"
        ),
    )
    threshold: float = 0.8

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True


class FaceSearchResponse(BaseModel):
    """Response model for face search."""
    timestamp: str
    status: int
    error: str
    data: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


class FaceData(BaseModel):
    """Data model for face search response."""
    featureBase64: str
    face_rectangle: List[int]
    gender: int = 0
    age: int = 0


class FaceResult(BaseModel):
    """Data model for face search response."""
    personId: str
    personCode: str
    fullname: str
    department_id: Optional[str] = None
    person_type: Optional[int] = None
    scoreMatching: float
