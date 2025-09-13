import base64
import pytest
from fastapi.testclient import TestClient
from main import app  
client = TestClient(app)

FAKE_AUDIO_BYTES = b"this_is_fake_audio_data"
FAKE_AUDIO_B64 = base64.b64encode(FAKE_AUDIO_BYTES).decode("utf-8")


@pytest.fixture
def start_session():
    """Start a voice session and return the session_id."""
    response = client.post("/api/v1/voice/start-session")
    assert response.status_code == 200
    data = response.json()
    return data["session_id"]


def test_start_session():
    """Test that a voice session starts successfully."""
    response = client.post("/api/v1/voice/start-session")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert isinstance(data["session_id"], str)


def test_send_audio_chunk_valid(start_session):
    """Send a valid audio chunk and check if transcript updates."""
    response = client.post(
        "/api/v1/voice/recognize",
        json={"session_id": start_session, "audio_chunk": FAKE_AUDIO_B64},
    )
    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert isinstance(data["transcript"], str)


def test_get_transcript_status(start_session):
    """Check transcript/status retrieval."""
    response = client.get(f"/api/v1/voice/status?session_id={start_session}")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] == start_session
    assert "status" in data


def test_end_session(start_session):
    """End the session and verify closure."""
    response = client.post("/api/v1/voice/end-session", json={"session_id": start_session})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Session ended"


def test_invalid_session_id():
    """Using a fake session ID should return 404."""
    response = client.post(
        "/api/v1/voice/recognize",
        json={"session_id": "invalid_id", "audio_chunk": FAKE_AUDIO_B64},
    )
    assert response.status_code == 404


def test_audio_before_start_session():
    """Trying to send audio before starting a session should return 400."""
    response = client.post(
        "/api/v1/voice/recognize",
        json={"session_id": "", "audio_chunk": FAKE_AUDIO_B64},
    )
    assert response.status_code == 400


def test_corrupted_audio_chunk(start_session):
    """Send non-base64 audio and expect 422."""
    response = client.post(
        "/api/v1/voice/recognize",
        json={"session_id": start_session, "audio_chunk": "not_base64!!"},
    )
    assert response.status_code == 422
