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
import logging
import requests
import os
import threading
import time
from typing import Dict, Optional

# --- Screen Capture Service ---
active_users = set()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

streams = {}

recording_users = set()
screen_recordings = {}  # Store recording data
recording_threads = {}  # Store recording thread references

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


# --- Screen Recording Service ---
def start_screen_recording(user_id: str, permissions_granted: bool = True) -> dict:
    """Start screen recording for a user"""
    if not permissions_granted:
        return {
            "status": "failed", 
            "user": user_id, 
            "error": "Screen recording permissions denied"
        }
    
    if user_id in recording_users:
        return {
            "status": "already_recording", 
            "user": user_id,
            "message": "Screen recording already active"
        }
    
    recording_users.add(user_id)
    screen_recordings[user_id] = {
        "start_time": time.time(),
        "frames": [],
        "status": "recording"
    }
    
    # Start recording thread
    thread = threading.Thread(target=_record_screen_loop, args=(user_id,))
    recording_threads[user_id] = thread
    thread.daemon = True
    thread.start()
    
    logger.info(f"Started screen recording for user: {user_id}")
    return {"status": "recording_started", "user": user_id}

def pause_screen_recording(user_id: str) -> dict:
    """Pause screen recording for a user"""
    if user_id not in recording_users:
        return {"status": "not_recording", "user": user_id}
    
    if user_id in screen_recordings:
        screen_recordings[user_id]["status"] = "paused"
    
    logger.info(f"Paused screen recording for user: {user_id}")
    return {"status": "recording_paused", "user": user_id}

def resume_screen_recording(user_id: str) -> dict:
    """Resume screen recording for a user"""
    if user_id not in recording_users:
        return {"status": "not_recording", "user": user_id}
    
    if user_id in screen_recordings:
        screen_recordings[user_id]["status"] = "recording"
    
    logger.info(f"Resumed screen recording for user: {user_id}")
    return {"status": "recording_resumed", "user": user_id}

def stop_screen_recording(user_id: str) -> dict:
    """Stop screen recording for a user"""
    if user_id not in recording_users:
        return {"status": "not_recording", "user": user_id}
    
    recording_users.discard(user_id)
    
    # Stop recording thread
    if user_id in recording_threads:
        # Signal thread to stop by updating status
        if user_id in screen_recordings:
            screen_recordings[user_id]["status"] = "stopped"
        recording_threads.pop(user_id)
    
    recording_data = screen_recordings.get(user_id, {})
    duration = time.time() - recording_data.get("start_time", 0)
    frame_count = len(recording_data.get("frames", []))
    
    logger.info(f"Stopped screen recording for user: {user_id}")
    return {
        "status": "recording_stopped", 
        "user": user_id,
        "duration": round(duration, 2),
        "frame_count": frame_count
    }

def get_recording_status(user_id: str) -> dict:
    """Get current recording status for a user"""
    if user_id not in recording_users:
        return {"status": "not_recording", "user": user_id}
    
    recording_data = screen_recordings.get(user_id, {})
    duration = time.time() - recording_data.get("start_time", 0)
    
    return {
        "status": recording_data.get("status", "unknown"),
        "user": user_id,
        "duration": round(duration, 2),
        "frame_count": len(recording_data.get("frames", []))
    }

def _record_screen_loop(user_id: str):
    """Background thread function for recording screen"""
    while user_id in recording_users:
        recording_data = screen_recordings.get(user_id, {})
        
        if recording_data.get("status") == "stopped":
            break
        
        if recording_data.get("status") == "recording":
            # Simulate screen capture (replace with actual screen capture)
            frame_data = {
                "timestamp": time.time(),
                "frame": f"mock_screen_frame_{len(recording_data.get('frames', []))}"
            }
            recording_data.setdefault("frames", []).append(frame_data)
        
        time.sleep(1)  # Capture frame every second

def check_screen_permissions(user_id: str) -> dict:
    """Check if user has granted screen recording permissions"""
    # Simulate permission check (replace with actual permission check)
    has_permission = random.choice([True, False])  # Mock permission status
    
    return {
        "user": user_id,
        "permissions_granted": has_permission,
        "message": "Screen recording permissions granted" if has_permission else "Screen recording permissions denied"
    }


def analyze_screen_recording(user_id: str, question: str = "") -> dict:
    """Analyze recorded screen content and provide AI answer"""
    if user_id not in screen_recordings:
        return {"error": "No recording found for user", "user": user_id}
    
    recording_data = screen_recordings[user_id]
    frames = recording_data.get("frames", [])
    
    if not frames:
        return {"error": "No frames recorded", "user": user_id}
    
    # Simulate analysis of recorded content
    analysis_results = []
    for i, frame in enumerate(frames[-5:]):  # Analyze last 5 frames
        # Mock analysis - replace with actual AI analysis
        mock_analysis = f"Frame {i}: User was viewing application interface"
        analysis_results.append(mock_analysis)
    
    # Generate AI response based on analysis
    if question:
        ai_response = f"Based on your screen recording analysis: {question}. I can see from the recorded frames that you were interacting with various interface elements."
    else:
        ai_response = "Screen recording analysis complete. The recorded content shows typical application usage patterns."
    
    return {
        "user": user_id,
        "analysis": analysis_results,
        "ai_response": ai_response,
        "frames_analyzed": len(analysis_results)
    }