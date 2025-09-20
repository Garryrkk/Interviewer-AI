from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import asyncio
import json
import logging
from datetime import datetime

from .schemas import (
    VoiceSessionRequest,
    VoiceSessionResponse,
    MicrophoneStatusRequest,
    MicrophoneStatusResponse,
    DeviceListResponse,
    DeviceSelectionRequest,
    AudioProcessingRequest,
    AudioProcessingResponse,
    TranscriptionResponse,
    AIResponseRequest,
    AIResponseResponse,
    VoiceAnalysisResponse,
    SimplifiedAnswerRequest
)
from .service import VoiceProcessingService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice-processing"])
voice_service = VoiceProcessingService()

@router.post("/session/start", response_model=VoiceSessionResponse)
async def start_voice_session(request: VoiceSessionRequest):
    """
    Start a new voice recognition session
    """
    try:
        session_id = await voice_service.create_session(request.user_id)
        
        return VoiceSessionResponse(
            success=True,
            session_id=session_id,
            message="Voice session started successfully",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to start voice session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@router.get("/microphone/status/{session_id}", response_model=MicrophoneStatusResponse)
async def check_microphone_status(session_id: str):
    """
    Check microphone status before and during meeting
    """
    try:
        mic_status = await voice_service.check_microphone_status(session_id)
        
        return MicrophoneStatusResponse(
            success=True,
            is_available=mic_status["is_available"],
            is_enabled=mic_status["is_enabled"],
            current_device=mic_status.get("current_device"),
            message=mic_status["message"],
            alert_required=not mic_status["is_enabled"]
        )
    except Exception as e:
        logger.error(f"Failed to check microphone status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Microphone check failed: {str(e)}")

@router.get("/devices/list/{session_id}", response_model=DeviceListResponse)
async def get_audio_devices(session_id: str):
    """
    Get list of available audio input devices
    """
    try:
        devices = await voice_service.get_audio_devices(session_id)
        
        return DeviceListResponse(
            success=True,
            devices=devices,
            default_device=devices[0] if devices else None,
            message="Audio devices retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Failed to get audio devices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get devices: {str(e)}")

@router.post("/device/select", response_model=JSONResponse)
async def select_audio_device(request: DeviceSelectionRequest):
    """
    Select and connect to a specific audio device
    """
    try:
        result = await voice_service.select_audio_device(
            request.session_id, 
            request.device_id
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "connected_device": result["device_name"],
                "device_id": result["device_id"],
                "message": "Device connected successfully"
            }
        )
    except Exception as e:
        logger.error(f"Failed to select audio device: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Device selection failed: {str(e)}")

@router.post("/microphone/toggle")
async def toggle_microphone(request: MicrophoneStatusRequest):
    """
    Turn microphone on/off and handle device connection
    """
    try:
        if request.turn_on:
            # Check if mic is available, if not prompt user
            mic_status = await voice_service.check_microphone_status(request.session_id)
            
            if not mic_status["is_available"]:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": False,
                        "message": "Please enable microphone access in your browser settings",
                        "action_required": "enable_permissions"
                    }
                )
            
            # Get devices and connect to default or user-selected
            devices = await voice_service.get_audio_devices(request.session_id)
            if not devices:
                raise HTTPException(status_code=404, detail="No audio devices found")
            
            # Connect to default device if no specific device requested
            device_to_use = request.device_id if hasattr(request, 'device_id') and request.device_id else devices[0]["id"]
            result = await voice_service.select_audio_device(request.session_id, device_to_use)
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "microphone_on": True,
                    "connected_device": result["device_name"],
                    "message": "Microphone turned on and connected"
                }
            )
        else:
            await voice_service.disable_microphone(request.session_id)
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "microphone_on": False,
                    "message": "Microphone turned off"
                }
            )
            
    except Exception as e:
        logger.error(f"Failed to toggle microphone: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Microphone toggle failed: {str(e)}")

@router.post("/audio/process", response_model=AudioProcessingResponse)
async def process_audio(
    session_id: str,
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Process uploaded audio: analyze, transcribe, and prepare for AI response
    """
    try:
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Process audio in background
        processing_result = await voice_service.process_audio(
            session_id=session_id,
            audio_data=audio_data,
            filename=audio_file.filename
        )
        
        return AudioProcessingResponse(
            success=True,
            transcription=processing_result["transcription"],
            confidence_score=processing_result["confidence"],
            audio_quality=processing_result["audio_quality"],
            processing_time=processing_result["processing_time"],
            message="Audio processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: AudioProcessingRequest):
    """
    Transcribe audio to text
    """
    try:
        transcription_result = await voice_service.transcribe_audio(
            request.session_id,
            request.audio_data
        )
        
        return TranscriptionResponse(
            success=True,
            text=transcription_result["text"],
            confidence=transcription_result["confidence"],
            language=transcription_result.get("language", "en"),
            duration=transcription_result.get("duration", 0),
            message="Transcription completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@router.post("/ai/respond", response_model=AIResponseResponse)
async def get_ai_response(request: AIResponseRequest):
    """
    Get AI response using Ollama Nous Hermes model
    """
    try:
        ai_response = await voice_service.generate_ai_response(
            session_id=request.session_id,
            question=request.question,
            response_format=request.response_format,
            context=request.context
        )
        
        return AIResponseResponse(
            success=True,
            response=ai_response["response"],
            format_type=request.response_format,
            confidence=ai_response["confidence"],
            processing_time=ai_response["processing_time"],
            message="AI response generated successfully"
        )
        
    except Exception as e:
        logger.error(f"AI response generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI response failed: {str(e)}")

@router.post("/analyze/voice", response_model=VoiceAnalysisResponse)
async def analyze_voice_confidence(request: AudioProcessingRequest):
    """
    Analyze voice confidence and provide situational tips
    """
    try:
        analysis_result = await voice_service.analyze_voice_characteristics(
            request.session_id,
            request.audio_data
        )
        
        return VoiceAnalysisResponse(
            success=True,
            confidence_rating=analysis_result["confidence_rating"],
            voice_characteristics=analysis_result["characteristics"],
            situational_tips=analysis_result["tips"],
            recommendations=analysis_result["recommendations"],
            analysis_summary=analysis_result["summary"]
        )
        
    except Exception as e:
        logger.error(f"Voice analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")

@router.post("/simplify", response_model=AIResponseResponse)
async def get_simplified_answer(request: SimplifiedAnswerRequest):
    """
    Get a more simplified version of the AI response
    """
    try:
        simplified_response = await voice_service.simplify_response(
            session_id=request.session_id,
            original_response=request.original_response,
            simplification_level=request.simplification_level
        )
        
        return AIResponseResponse(
            success=True,
            response=simplified_response["response"],
            format_type="simplified",
            confidence=simplified_response["confidence"],
            processing_time=simplified_response["processing_time"],
            message="Simplified response generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Response simplification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simplification failed: {str(e)}")

@router.delete("/session/{session_id}")
async def end_voice_session(session_id: str):
    """
    End voice session and cleanup resources
    """
    try:
        await voice_service.end_session(session_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Voice session ended successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to end session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session cleanup failed: {str(e)}")

@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """
    Get current session status and statistics
    """
    try:
        status = await voice_service.get_session_status(session_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "session_active": status["active"],
                "microphone_connected": status["mic_connected"],
                "current_device": status.get("current_device"),
                "session_duration": status.get("duration"),
                "total_interactions": status.get("interactions", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get session status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")