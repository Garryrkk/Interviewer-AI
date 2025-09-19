from pydantic import BaseModel

# --- Requests ---
class UserRequest(BaseModel):
    user_id: str

class ImageBase64Request(BaseModel):
    image_base64: str

# --- Responses ---
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
