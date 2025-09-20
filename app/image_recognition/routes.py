import os
import random
import base64
import logging
from io import BytesIO
from PIL import Image
import requests
from fastapi import APIRouter, HTTPException, Body
from typing import Optional

# Import schemas
from .schemas import (
    UserRequest, StreamResponse, ImageBase64Request,
    EmotionResponse, AnalyzeScreenResponse, AnalyzeCameraResponse,
    StartRecordingRequest, StopRecordingRequest, PermissionRequest,
    AnalyzeRecordingRequest, RecordingStatusResponse, PermissionResponse,
    RecordingAnalysisResponse, ScreenAnalysisResponse, CameraAnalysisResponse,
    RecordingControlResponse, ExtendedUserRequest, BatchAnalysisRequest,
    BatchAnalysisResponse
)

# Import service functions
from .service import (
    analyze_screen, analyze_camera, start_screen_recording,
    pause_screen_recording, resume_screen_recording, stop_screen_recording,
    get_recording_status, check_screen_permissions, analyze_screen_recording
)

router = APIRouter()

# --- Logger ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Active users for stream ---
active_users = set()

# --- Stream routes ---
@router.post("/start_stream", response_model=StreamResponse)  
async def start_stream(request: UserRequest):
    """Start a fake camera stream for a user."""
    active_users.add(request.user_id)
    return StreamResponse(status="started", user=request.user_id)

@router.post("/stop_stream", response_model=StreamResponse)
async def stop_stream(request: UserRequest):
    """Stop a fake camera stream for a user."""
    active_users.discard(request.user_id)
    return StreamResponse(status="stopped", user=request.user_id)

# --- Screen Recording Routes ---
@router.post("/screen_recording/permissions", response_model=PermissionResponse)
async def check_permissions(request: PermissionRequest):
    """Check screen recording permissions for a user."""
    result = check_screen_permissions(request.user_id)
    return PermissionResponse(**result)

@router.post("/screen_recording/start", response_model=RecordingStatusResponse)
async def start_recording(request: StartRecordingRequest):
    """Start screen recording for a user."""
    result = start_screen_recording(request.user_id, request.permissions_granted)
    return RecordingStatusResponse(**result)

@router.post("/screen_recording/pause", response_model=RecordingControlResponse)
async def pause_recording(request: UserRequest):
    """Pause screen recording for a user."""
    result = pause_screen_recording(request.user_id)
    return RecordingControlResponse(
        status=result["status"],
        user=result["user"],
        action="pause"
    )

@router.post("/screen_recording/resume", response_model=RecordingControlResponse)
async def resume_recording(request: UserRequest):
    """Resume screen recording for a user."""
    result = resume_screen_recording(request.user_id)
    return RecordingControlResponse(
        status=result["status"],
        user=result["user"],
        action="resume"
    )

@router.post("/screen_recording/stop", response_model=RecordingStatusResponse)
async def stop_recording(request: UserRequest):
    """Stop screen recording for a user."""
    result = stop_screen_recording(request.user_id)
    return RecordingStatusResponse(**result)

@router.get("/screen_recording/status/{user_id}", response_model=RecordingStatusResponse)
async def get_recording_status_route(user_id: str):
    """Get current recording status for a user."""
    result = get_recording_status(user_id)
    return RecordingStatusResponse(**result)

@router.post("/screen_recording/analyze", response_model=RecordingAnalysisResponse)
async def analyze_recording(request: AnalyzeRecordingRequest):
    """Analyze recorded screen content and get AI response."""
    result = analyze_screen_recording(request.user_id, request.question)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return RecordingAnalysisResponse(**result)

# --- Facial expression detection ---
@router.post("/detect_expression", response_model=EmotionResponse)
async def detect_expression(image_base64: str = Body(..., embed=True)):
    """Fake facial-expression detection."""
    emotions = ["neutral", "happy", "confused", "stressed"]
    emotion = random.choice(emotions)
    confidence = round(random.uniform(0.55, 0.98), 2)

    advice = ""
    if emotion == "confused":
        advice = "User looks confused. Simplify explanation."
    elif emotion == "stressed":
        advice = "User looks stressed. Provide a calm response."
    else:
        advice = "All good. Continue normally."

    return EmotionResponse(label=emotion, confidence=confidence, advice=advice)

# --- Screen analysis ---
@router.post("/analyze_screen", response_model=ScreenAnalysisResponse)
async def analyze_screen_route(image_base64: str = Body(..., embed=True)):
    """Send screenshot to LLaVA model (mock)."""
    try:
        response = requests.post(
            os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            json={
                "model": "llava",
                "prompt": "Describe the content of this screenshot in detail.",
                "images": [image_base64]
            },
            timeout=120
        )
        response.raise_for_status()
        description = response.json().get("response", "").strip()
        return ScreenAnalysisResponse(description=description)
    except (requests.RequestException, ValueError) as e:
        logger.exception("LLaVA request failed")
        return ScreenAnalysisResponse(error=f"Error analyzing screen: {e}")

# --- Camera capture analysis ---
@router.post("/analyze_camera", response_model=CameraAnalysisResponse)
async def analyze_camera_route(image_base64: str = Body(..., embed=True)):
    """Fake facial emotion detection."""
    possible_emotions = ["neutral", "stressed", "confused", "happy"]
    emotion = random.choice(possible_emotions)

    advice = ""
    if emotion == "confused":
        advice = "User seems confused. AI should simplify the explanation."
    elif emotion == "stressed":
        advice = "User looks stressed. AI should answer calmly and slowly."
    else:
        advice = "No issues detected. Continue normally."

    return CameraAnalysisResponse(emotion=emotion, advice=advice)

# --- Enhanced Analysis Routes ---
@router.post("/analyze_with_recording", response_model=RecordingAnalysisResponse)
async def analyze_with_recording_context(request: AnalyzeRecordingRequest):
    """Analyze current screen with context from recording history."""
    result = analyze_screen_recording(request.user_id, request.question)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return RecordingAnalysisResponse(**result)

@router.post("/batch_analyze", response_model=BatchAnalysisResponse)
async def batch_analyze_images(request: BatchAnalysisRequest):
    """Analyze multiple images in batch."""
    results = []
    errors = []
    
    for i, image_base64 in enumerate(request.images):
        try:
            if request.analysis_type == "camera":
                result = analyze_camera(image_base64)
                results.append({"index": i, "type": "camera", "result": result})
            elif request.analysis_type == "screen":
                # Mock screen analysis
                result = {"description": f"Screen analysis for image {i}"}
                results.append({"index": i, "type": "screen", "result": result})
            else:
                errors.append(f"Unknown analysis type for image {i}: {request.analysis_type}")
        except Exception as e:
            errors.append(f"Error analyzing image {i}: {str(e)}")
    
    return BatchAnalysisResponse(
        user_id=request.user_id,
        results=results,
        total_analyzed=len(results),
        errors=errors if errors else None
    )

# --- Recording Management Routes ---
@router.get("/users/{user_id}/recording_history")
async def get_recording_history(user_id: str):
    """Get recording history for a user."""
    # Mock implementation - replace with actual data storage
    return {
        "user_id": user_id,
        "recordings": [
            {"id": "rec_1", "date": "2025-09-20", "duration": 120.5, "frames": 245},
            {"id": "rec_2", "date": "2025-09-19", "duration": 89.2, "frames": 178}
        ]
    }

@router.delete("/users/{user_id}/recordings/{recording_id}")
async def delete_recording(user_id: str, recording_id: str):
    """Delete a specific recording."""
    # Mock implementation
    return {
        "message": f"Recording {recording_id} deleted for user {user_id}",
        "status": "success"
    }

# --- System Status Routes ---
@router.get("/system/recording_stats")
async def get_recording_stats():
    """Get system-wide recording statistics."""
    return {
        "active_recordings": len(recording_users),
        "total_active_users": len(active_users),
        "system_status": "operational"
    }

# --- Utility ---
def base64_to_pil(img_b64: str) -> Image.Image:
    img_data = base64.b64decode(img_b64)
    return Image.open(BytesIO(img_data)).convert("RGB")