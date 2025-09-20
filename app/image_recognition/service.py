import random
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
import requests
from fer import FER  # pip install fer
import numpy as np
import cv2

# --- Screen Capture Service ---
active_users = set()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

streams = {}

def startStream(user_id: str) -> dict:
    cap = cv2.VideoCapture(0)
    streams[user_id] = cap
    return {"status": "started", "user": user_id}

def stopStream(user_id: str) -> dict:
    cap = streams.get(user_id)
    if cap:
        cap.release()
        streams.pop(user_id)
    active_users.discard(user_id)
    return {"status": "stopped", "user": user_id}

# --- Fake Expression Detection ---
def detectFromCamera_real(image_base64: str) -> dict:
    img = base64_to_pil(image_base64)
    img_np = np.array(img)
    detector = FER(mtcnn=True)  # mtcnn=True uses better face detection
    result = detector.top_emotion(img_np)
    if result:
        emotion, confidence = result
        advice = ""
        if emotion in ["angry", "fear", "sad"]:
            advice = "User seems stressed. Respond calmly."
        elif emotion == "surprise":
            advice = "User seems confused. Simplify explanation."
        else:
            advice = "All good. Continue normally."
        return {"label": emotion, "confidence": round(confidence, 2), "advice": advice}
    return {"label": "neutral", "confidence": 0.9, "advice": "All good."}

def analyze_screen_real(image_base64: str, focus: str = "general") -> dict:
    try:
        response = requests.post(
            os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            json={
                "model": "llava",
                "prompt": f"Describe the content of this screenshot in detail. Focus on {focus}.",
                "images": [image_base64]
            },
            timeout=120
        )
        response.raise_for_status()
        return {"analysis": response.json().get("response", "").strip()}
    except (requests.RequestException, ValueError) as e:
        logger.exception("LLaVA request failed")
        return {"error": str(e)}
    
# --- Camera Capture Service ---
def analyze_camera(image_base64: str) -> dict:
    """
    Fake facial emotion detection.
    Replace with real model later.
    """
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

def base64_to_pil(img_b64: str) -> Image.Image:
    img_data = base64.b64decode(img_b64)
    return Image.open(BytesIO(img_data)).convert("RGB")