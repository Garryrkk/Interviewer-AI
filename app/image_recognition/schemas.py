from pydantic import BaseModel
<<<<<<< HEAD
from typing import Optional, Dict, Any

# --- Original Requests ---
=======
from typing import Optional, List, Dict, Any

# --- Basic Requests ---
>>>>>>> c4e4e634ed651b11369f4907425872c4b1129e87
class UserRequest(BaseModel):
    user_id: str

class ImageBase64Request(BaseModel):
    image_base64: str

<<<<<<< HEAD
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
=======
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
>>>>>>> c4e4e634ed651b11369f4907425872c4b1129e87
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

<<<<<<< HEAD
# --- New Screen Capture Responses ---
class ScreenCaptureResponse(BaseModel):
    analysis: str
    timestamp: float
    region_captured: Optional[str] = None

class CameraCaptureResponse(BaseModel):
    analysis: str
    timestamp: float
    camera_used: int
=======
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
>>>>>>> c4e4e634ed651b11369f4907425872c4b1129e87
