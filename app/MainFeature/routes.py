from fastapi import APIRouter
from .schemas import (
    InvisibilityStartRequest,
    InvisibilitySendRequest,
    InvisibilityEndRequest,
    InvisibilityResponse,
)
from . import service

router = APIRouter(prefix="/invisibility", tags=["Invisibility"])

@router.post("/start", response_model=InvisibilityResponse)
def start_session(req: InvisibilityStartRequest):
    result = service.start_invisibility_session(req.user_id)
    return InvisibilityResponse(**result)

@router.post("/send", response_model=InvisibilityResponse)
def send_response(req: InvisibilitySendRequest):
    result = service.send_invisible_response(req.user_id, req.response)
    return InvisibilityResponse(**result)

@router.post("/end", response_model=InvisibilityResponse)
def end_session(req: InvisibilityEndRequest):
    result = service.end_invisibility_session(req.user_id)
    return InvisibilityResponse(**result)

# Optional: frontend can poll for stored responses
@router.get("/messages/{user_id}")
def get_messages(user_id: str):
    return {"messages": service.get_invisible_responses(user_id)}
