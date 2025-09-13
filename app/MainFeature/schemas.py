from pydantic import BaseModel

class InvisibilityStartRequest(BaseModel):
    user_id: str

class InvisibilitySendRequest(BaseModel):
    user_id: str
    response: str

class InvisibilityEndRequest(BaseModel):
    user_id: str

class InvisibilityResponse(BaseModel):
    status: str
    message: str
