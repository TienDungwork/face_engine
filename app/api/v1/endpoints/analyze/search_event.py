from fastapi import APIRouter, HTTPException

from app.schemas.analyze.search_event import EventSearchRequest, EventSearchResponse
from app.services.analyze.search_event import search_event

router = APIRouter()


@router.post("/searchEvent", response_model=EventSearchResponse)
async def search_event_endpoint(request: EventSearchRequest):
    """Endpoint for event search by face."""
    try:
        return await search_event(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
