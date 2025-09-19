from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# --- Basic Requests ---
class UserRequest(BaseModel):
    user_id: str

class ImageBase64Request(BaseModel):
    image_base64: str

# --- Screen Recording Requests ---
class StartRecordingRequest(BaseModel):
    user_id: str
    permissions_granted: Optional[bool] = True

class StopRecordingRequest(BaseModel):
    user_id: str

class PermissionRequest(BaseModel):
    user_id: str

class AnalyzeRecordingRequest(BaseModel):
    user_id: str
    question: Optional[str] = ""

# --- Basic Responses ---
class StreamResponse(BaseModel):
    status: str
    user: str

class EmotionResponse(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    advice: Optional[str] = None

class AnalyzeScreenResponse(BaseModel):
    result: str

class AnalyzeCameraResponse(BaseModel):
    emotion: str
    advice: str

# --- Screen Recording Responses ---
class RecordingStatusResponse(BaseModel):
    status: str
    user: str
    message: Optional[str] = None
    error: Optional[str] = None
    duration: Optional[float] = None
    frame_count: Optional[int] = None

class PermissionResponse(BaseModel):
    user: str
    permissions_granted: bool
    message: str

class RecordingAnalysisResponse(BaseModel):
    user: str
    analysis: List[str]
    ai_response: str
    frames_analyzed: int
    error: Optional[str] = None

# --- Enhanced Analysis Responses ---
class ScreenAnalysisResponse(BaseModel):
    description: Optional[str] = None
    error: Optional[str] = None

class CameraAnalysisResponse(BaseModel):
    emotion: str
    advice: str

# --- Recording Control Responses ---
class RecordingControlResponse(BaseModel):
    status: str
    user: str
    action: str
    message: Optional[str] = None

# --- Extended User Request with Options ---
class ExtendedUserRequest(BaseModel):
    user_id: str
    options: Optional[Dict[str, Any]] = {}

# --- Batch Analysis Request ---
class BatchAnalysisRequest(BaseModel):
    user_id: str
    images: List[str]  # List of base64 images
    analysis_type: str  # "screen" or "camera"

class BatchAnalysisResponse(BaseModel):
    user_id: str
    results: List[Dict[str, Any]]
    total_analyzed: int
    errors: Optional[List[str]] = []