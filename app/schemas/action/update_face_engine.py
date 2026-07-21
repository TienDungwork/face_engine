from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class UpdateFaceEngineRequest(BaseModel):
    personId: str = Field(..., description="ID of the person to update")
    action: int = Field(..., description="Action to perform (1 for update)")
    url: Optional[str] = Field(None, description="URL of the face image")


class UpdateFaceEngineResponse(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: int = Field(200, description="HTTP status code")
    error: str = Field("SUCCESS", description="Error message or status")
    data: Optional[dict] = Field(None, description="Response data")
