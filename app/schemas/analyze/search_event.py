from datetime import datetime
from typing import List, Optional

from pydantic import AliasChoices, BaseModel, Field


class EventSearchRequest(BaseModel):
    """Request model for event search."""

    img_base64: Optional[str] = None
    img_url: Optional[str] = None
    company_ids: List[int] = Field(
        validation_alias=AliasChoices(
            "company_ids", "companyIds", "compIds", "comIds"
        ),
    )
    threshold: float = 0.85
    quality: float = 0.35
    num_result: int = 100
    fromDate: Optional[datetime] = None
    toDate: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
        populate_by_name = True


class EventResult(BaseModel):
    """Data model for event search response."""

    eventId: str
    score: float


class EventSearchResponse(BaseModel):
    """Response model for event search."""

    timestamp: str
    status: int
    error: str
    data: List[EventResult] = []

    class Config:
        arbitrary_types_allowed = True
