# backend/app/websocket_manager.py
import asyncio
from typing import Dict, Set
from fastapi import WebSocket
import aioredis  # optional if using Redis pub/sub

class ConnectionManager:
    def __init__(self):
        # session_id -> set of websockets (usually 1)
        self.active: Dict[str, Set[WebSocket]] = {}

    def connect(self, session_id: str, websocket: WebSocket):
        sockets = self.active.setdefault(session_id, set())
        sockets.add(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        sockets = self.active.get(session_id)
        if not sockets:
            return
        sockets.discard(websocket)
        if not sockets:
            self.active.pop(session_id, None)

    async def send_to_session(self, session_id: str, message: str):
        sockets = self.active.get(session_id, set())
        coros = [ws.send_text(message) for ws in sockets]
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

# single manager instance
manager = ConnectionManager()
