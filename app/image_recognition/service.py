import random

# --- Screen Capture Service ---
def analyze_screen(image_base64: str) -> str:
    """
    Fake image recognition for screen capture.
    Replace this with actual CV/ML model later.
    """
    possible_outputs = [
        "This looks like a bar chart with sales data.",
        "It seems to be a code editor window.",
        "This appears to be a PowerPoint slide presentation.",
        "Looks like a browser screen showing a dashboard."
    ]
    return random.choice(possible_outputs)


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
