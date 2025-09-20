from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.responses import JSONResponse
import asyncio
import json
import base64
from typing import Dict, List, Optional
import logging

from .services import VoiceRecognitionService
from .schemas import (
    PreopConfig,   
    PreopResponse, 
    AudioChunk, 
    ProcessingResponse, 
    RecognitionSession, 
    RecognitionResponse,
    SessionStatus
)

from fastapi import APIRouter, UploadFile, File
from schemas import TranscriptionResponse
from service import transcribe_audio_file

router = APIRouter()

@router.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    transcript = await transcribe_audio_file(audio)
    return {"transcript": transcript}


# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/voice", tags=["voice"])

# Initialize service
voice_service = VoiceRecognitionService()   
  
# Active WebSocket connections for real-time processing
active_connections: Dict[str, WebSocket] = {}

@router.post("/preop", response_model=PreopResponse)
async def voice_prep_check(config: PreopConfig):
    try:
        logger.info(f"Preop check requested for device: {config.device}")
        
        # Use service to perform preop checks
        result = await voice_service.preop_check(config)
        
        if result.status == "ready":
            logger.info("Preop check successful")
            return result
        else:
            logger.warning(f"Preop check failed: {result.message}")
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
        logger.error(f"Preop check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preop check failed: {str(e)}")


@router.get("/preop/devices")
async def get_available_devices():
    try:
        devices = await voice_service.get_available_devices()
        return {"devices": devices}
    except Exception as e:
        logger.error(f"Failed to get devices: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audio devices")

@router.post("/process", response_model=ProcessingResponse)
async def process_audio_chunk(audio_chunk: AudioChunk):
    try:
        logger.debug(f"Processing audio chunk at timestamp: {audio_chunk.timestamp}")
        
        # Decode base64 audio data
        try:
            audio_data = base64.b64decode(audio_chunk.data)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        
        # Process through service
        result = await voice_service.process_audio_chunk(audio_data, audio_chunk.timestamp)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
    
@router.websocket("/process/stream")
async def websocket_audio_stream(websocket: WebSocket):
    await websocket.accept()
    connection_id = f"conn_{id(websocket)}"
    active_connections[connection_id] = websocket
    
    logger.info(f"WebSocket connection established: {connection_id}")
    
    try:
        while True:
            # Receive audio data from frontend
            data = await websocket.receive_text()
            
            try:
                # Parse incoming message
                message = json.loads(data)
                
                if message.get("type") == "audio_chunk":
                    # Decode and process audio
                    audio_data = base64.b64decode(message["data"])
                    timestamp = message.get("timestamp", 0)
                    
                    # Process through service
                    result = await voice_service.process_audio_chunk(audio_data, timestamp)
                    
                    # Send response back
                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "transcript": result.transcript,
                        "confidence": result.confidence,
                        "timestamp": timestamp,
                        "is_final": result.is_final
                    }))
                    
                elif message.get("type") == "ping":
                    # Health check
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"WebSocket processing error: {str(e)}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Cleanup connection
        if connection_id in active_connections:
            del active_connections[connection_id]

@router.post("/recognize/start", response_model=RecognitionResponse)
async def start_recognition_session(session: RecognitionSession):
    try:
        logger.info(f"Starting recognition session: {session.session_id}")
        
        result = await voice_service.start_recognition_session(session)
        
        if result.session_id:
            logger.info(f"Recognition session started successfully: {result.session_id}")
            return result
        else:
            raise HTTPException(status_code=400, detail="Failed to start recognition session")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session start error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")
    
@router.post("/recognize/stop/{session_id}", response_model=RecognitionResponse)
async def stop_recognition_session(session_id: str):
    try:
        logger.info(f"Stopping recognition session: {session_id}")
        
        result = await voice_service.stop_recognition_session(session_id)
        
        if result:
            logger.info(f"Recognition session stopped: {session_id}")
            return result
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session stop error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop session: {str(e)}")
    
@router.get("/recognize/status/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str):  
    try:
        status = await voice_service.get_session_status(session_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")

@router.post("/recognize/process/{session_id}")
async def add_audio_to_session(session_id: str, audio_file: UploadFile = File(...)):
    try:
        logger.info(f"Adding audio to session: {session_id}")
        
        # Read audio file
        audio_data = await audio_file.read()
        
        # Process through service
        result = await voice_service.add_audio_to_session(session_id, audio_data)
        
        return {"message": "Audio added successfully", "status": result}
        
    except Exception as e:
        logger.error(f"Audio addition error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add audio: {str(e)}")

@router.get("/health")
async def health_check():
    try:
        # Check service health
        health_status = await voice_service.health_check()
        
        return {
            "status": "healthy" if health_status else "unhealthy",
            "service": "voice_recognition",
            "active_connections": len(active_connections),
            "timestamp": voice_service.get_current_timestamp()
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": voice_service.get_current_timestamp()
        }

@router.get("/sessions")
async def list_active_sessions():
    """
    List all active recognition sessions.
    Admin/debug endpoint.
    """
    try:
        sessions = await voice_service.list_active_sessions()
        return {"active_sessions": sessions}
    except Exception as e:
        logger.error(f"Session list error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@router.delete("/sessions/{session_id}")
async def force_close_session(session_id: str):
    """
    Force close a recognition session.
    Admin/cleanup endpoint.
    """
    try:
        result = await voice_service.force_close_session(session_id)
        
        if result:
            return {"message": f"Session {session_id} closed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Force close error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to close session: {str(e)}")


async def general_exception_handler(request, exc):
    """
    General exception handler for voice routes.
    """
    logger.error(f"Unhandled exception in voice routes: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error in voice recognition module"}
    )