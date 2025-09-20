from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- Original Requests ---
class UserRequest(BaseModel):
    user_id: str

class ImageBase64Request(BaseModel):
    image_base64: str

# --- New Screen Capture Requests ---
class ScreenCaptureRequest(BaseModel):
    region: Optional[str] = None
    analysis_type: str = "general"
    include_text: bool = True

class CameraCaptureRequest(BaseModel):
    camera_id: int = 0
    analysis_type: str = "emotion"
    duration: Optional[int] = None

# --- Original Responses ---
class StreamResponse(BaseModel):
    status: str
    user: str

class EmotionResponse(BaseModel):
    label: str = None
    confidence: float = None
    advice: str = None

class AnalyzeScreenResponse(BaseModel):
    result: str

class AnalyzeCameraResponse(BaseModel):
    emotion: str
    advice: str

# --- New Screen Capture Responses ---
class ScreenCaptureResponse(BaseModel):
    analysis: str
    timestamp: float
    region_captured: Optional[str] = None

class CameraCaptureResponse(BaseModel):
    analysis: str
    timestamp: float
    camera_used: int