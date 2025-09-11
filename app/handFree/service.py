
import base64
import tempfile
import speech_recognition as sr
from typing import Optional, Dict

# Import QuickRespond service
from .service_quickrespond import generate_quick_reply  

def process_handsfree(audio_base64: str, expression: Optional[str]) -> Dict:
    """
    Convert speech to text, analyze expression, and optionally trigger QuickRespond.
    """

    # Step 1: Decode audio
    audio_bytes = base64.b64decode(audio_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes)
        tmp_path = tmpfile.name

    # Step 2: Speech-to-Text (using SpeechRecognition lib with Google API as fallback)
    recognizer = sr.Recognizer()
    transcript = ""
    try:
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)
    except Exception as e:
        transcript = f"[Error transcribing audio: {str(e)}]"

    # Step 3: Decide if AI should reply
    triggered = True if transcript else False
    ai_reply = None

    if triggered:
        reply_text = generate_quick_reply(transcript)

        # If user looks confused → simplify the reply
        if expression and expression.lower() == "confused":
            reply_text = simplify_reply(reply_text)

        ai_reply = {"reply": reply_text}

    return {
        "transcript": transcript,
        "triggered": triggered,
        "quick_reply": ai_reply,
    }


def simplify_reply(reply: str) -> str:
    """
    Make AI response simpler for confused expression.
    """
    if not reply:
        return reply
    # Just return a shorter first sentence for demo
    if "." in reply:
        return reply.split(".")[0] + "."
    return reply
