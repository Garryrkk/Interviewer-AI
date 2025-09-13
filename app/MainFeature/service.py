from typing import Dict

# In-memory sessions (replace with Redis/DB later for production)
_sessions: Dict[str, list] = {}

def start_invisibility_session(user_id: str) -> dict:
    if user_id not in _sessions:
        _sessions[user_id] = []
    return {"status": "success", "message": f"Session started for {user_id}"}

def send_invisible_response(user_id: str, response: str) -> dict:
    if user_id not in _sessions:
        return {"status": "error", "message": "Session not found"}

    # Append response to user's session cache
    _sessions[user_id].append(response)
    return {"status": "success", "message": "Response sent invisibly"}

def get_invisible_responses(user_id: str) -> list:
    """Optional: frontend polls for new invisible messages."""
    return _sessions.get(user_id, [])

def end_invisibility_session(user_id: str) -> dict:
    if user_id in _sessions:
        del _sessions[user_id]
        return {"status": "success", "message": f"Session ended for {user_id}"}
    return {"status": "error", "message": "No active session found"}
