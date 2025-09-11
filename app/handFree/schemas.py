
from pydantic import BaseModel
from typing import Optional

# Request schema
class HandsFreeRequest(BaseModel):
    audio_base64: str   # base64 encoded audio from frontend
    expression: Optional[str] = None  # "neutral", "confused", "stressed" etc.

# Response schema
class HandsFreeResponse(BaseModel):
    transcript: str               # what interviewer said
    triggered: bool               # whether AI should reply
    quick_reply: Optional[dict]   # AI reply if triggered
