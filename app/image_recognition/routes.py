from fastapi import APIRouter
from .schemas import (
    UserRequest, StreamResponse, ImageBase64Request,
    EmotionResponse, AnalyzeScreenResponse, AnalyzeCameraResponse
)
from .service import analyze_screen, analyze_camera

# routes.py
import os
import random
import base64
import logging
from io import BytesIO
from PIL import Image
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()

# --- Logger ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Active users for stream ---
active_users = set()

# --- Stream routes ---
@router.post("/start_stream")
async def start_stream(user_id: str):
    """Start a fake camera stream for a user."""
    active_users.add(user_id)
    return {"status": "started", "user": user_id}


@router.post("/stop_stream")
async def stop_stream(user_id: str):
    """Stop a fake camera stream for a user."""
    active_users.discard(user_id)
    return {"status": "stopped", "user": user_id}


# --- Facial expression detection ---
@router.post("/detect_expression")
async def detect_expression(image_base64: str):
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

    return {"label": emotion, "confidence": confidence, "advice": advice}


# --- Screen analysis ---
@router.post("/analyze_screen")
async def analyze_screen(image_base64: str):
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
        return {"description": response.json().get("response", "").strip()}
    except (requests.RequestException, ValueError) as e:
        logger.exception("LLaVA request failed")
        raise HTTPException(status_code=500, detail=f"Error analyzing screen: {e}")


# --- Camera capture analysis ---
@router.post("/analyze_camera")
async def analyze_camera(image_base64: str):
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

    return {"emotion": emotion, "advice": advice}


# --- Utility ---
def base64_to_pil(img_b64: str) -> Image.Image:
    img_data = base64.b64decode(img_b64)
    return Image.open(BytesIO(img_data)).convert("RGB")
