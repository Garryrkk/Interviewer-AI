from fastapi import APIRouter
from .schemas import (
    ScreenCaptureRequest,
    ScreenCaptureResponse,
    CameraCaptureRequest,
    CameraCaptureResponse,
)
from .service import analyze_screen, analyze_camera

router = APIRouter()

# --- Screen Capture Route ---
@router.post("/analyze-screen", response_model=ScreenCaptureResponse)
def analyze_screen_route(request: ScreenCaptureRequest):
    description = analyze_screen(request.image_base64)
    return ScreenCaptureResponse(description=description)


# --- Camera Capture Route ---
@router.post("/analyze-camera", response_model=CameraCaptureResponse)
def analyze_camera_route(request: CameraCaptureRequest):
    result = analyze_camera(request.image_base64)
    return CameraCaptureResponse(**result)
