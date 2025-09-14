from pydantic import BaseModel

# --- Screen Capture ---
class ScreenCaptureRequest(BaseModel):
    image_base64: str  # screenshot sent as base64 string

class ScreenCaptureResponse(BaseModel):
    description: str   # AI's interpretation of the screen


# --- Camera Capture ---
class CameraCaptureRequest(BaseModel):
    image_base64: str  # camera frame as base64 string

class CameraCaptureResponse(BaseModel):
    emotion: str       # e.g. "stressed", "confused", "neutral"
    advice: str        # suggestion or simplified answer
   