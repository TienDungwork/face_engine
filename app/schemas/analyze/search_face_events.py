from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class SearchFaceEventsRequest(BaseModel):
    img_base64: Optional[str] = Field(
        default=None,
        description="Base64 ảnh (JPEG/PNG). Có thể kèm prefix data:image/jpeg;base64,",
    )
    img_url: Optional[str] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    camera_id: Optional[Union[int, str]] = Field(
        default=None,
        description="Không lọc: bỏ field hoặc null. Giá trị 0 bị bỏ qua (Swagger hay gửi 0).",
    )
    company_id: Optional[Union[int, str]] = Field(
        default=None,
        description="Tương tự camera_id; 0 = không lọc.",
    )
    department_id: Optional[Union[int, str]] = Field(
        default=None,
        description="Tương tự camera_id; 0 = không lọc.",
    )
    direction: Optional[Union[int, str]] = Field(
        default=None,
        description="Tương tự; 0 = không lọc (tránh direction::text = '0' loại hết bản ghi).",
    )
    limit: int = Field(default=5000, ge=1, le=100000)

    class Config:
        arbitrary_types_allowed = True


class FaceEventSearchItem(BaseModel):
    id: int
    event_id: Optional[str] = None
    access_time: Optional[str] = None
    camera_id: Optional[str] = None
    camera_code: Optional[str] = None
    camera_name: Optional[str] = None
    user_code: Optional[str] = None
    user_name: Optional[str] = None
    department_name: Optional[str] = None
    image: Optional[str] = None
    direction: Optional[str] = None
    score_match: Optional[float] = None
    similarity: float


class SearchFaceEventsResponse(BaseModel):
    timestamp: str
    status: int
    error: str
    data: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
