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
    SimplifiedAnswerRequest,
    CalibrationRequest, 
    CalibrationResponse, 
    TranscriptionResponse,
    AudioTestRequest,
    AudioTestResponse,
    ErrorResponse
)
from .services import VoiceProcessingService
from .services import AudioService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])
router = APIRouter(prefix="/api/voice", tags=["voice-processing"])
voice_service = VoiceProcessingService()
audio_service = AudioService()

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
    
@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_audio(request: CalibrationRequest):
    """
    Audio Calibration endpoint - measures background noise and sets optimal levels
    for clear voice recognition.
    """
    try:
        logger.info(f"Starting audio calibration with duration: {request.duration}s")
        
        # Perform calibration
        result = await audio_service.calibrate_audio(
            duration=request.duration,
            sample_rate=request.sample_rate,
            channels=request.channels
        )
        
        logger.info(f"Calibration completed. Noise level: {result.noise_level}dB")
        return result
        
    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio calibration failed: {str(e)}"
        )

@router.post("/test-record", response_model=AudioTestResponse)
async def test_audio_recording(request: AudioTestRequest):
    """
    Test audio recording endpoint - records a test clip and validates audio quality
    before transcription.
    """
    try:
        logger.info(f"Starting test recording with duration: {request.duration}s")
        
        # Record test audio
        result = await audio_service.test_recording(
            duration=request.duration,
            sample_rate=request.sample_rate,
            channels=request.channels,
            apply_calibration=request.apply_calibration
        )
        
        logger.info("Test recording completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Test recording failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test recording failed: {str(e)}"
        )

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = "auto",
    model_size: Optional[str] = "base"
):
    """
    Speech-to-text transcription endpoint - converts uploaded audio to text.
    """
    try:
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        logger.info(f"Starting transcription for file: {audio_file.filename}")
        logger.info(f"File size: {audio_file.size} bytes, Content type: {audio_file.content_type}")
        
        # Read audio file
        audio_data = await audio_file.read()
        
        # Perform transcription
        result = await audio_service.transcribe_audio(
            audio_data=audio_data,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
            language=language,
            model_size=model_size
        )
        
        logger.info("Transcription completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@router.post("/transcribe-test", response_model=TranscriptionResponse)
async def transcribe_test_recording():
    """
    Transcribe the most recent test recording.
    """
    try:
        logger.info("Transcribing latest test recording")
        
        result = await audio_service.transcribe_latest_test()
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail="No test recording found. Please record a test clip first."
            )
        
        logger.info("Test recording transcription completed")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test transcription failed: {str(e)}"
        )

@router.get("/calibration-status")
async def get_calibration_status():
    """
    Get current audio calibration status and settings.
    """
    try:
        status = await audio_service.get_calibration_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Failed to get calibration status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get calibration status: {str(e)}"
        )

@router.delete("/reset-calibration")
async def reset_calibration():
    """
    Reset audio calibration settings to defaults.
    """
    try:
        await audio_service.reset_calibration()
        return JSONResponse(content={"message": "Calibration reset successfully"})
        
    except Exception as e:
        logger.error(f"Failed to reset calibration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset calibration: {str(e)}"
        )

@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get list of supported audio formats for upload.
    """
    return JSONResponse(content={
        "supported_formats": [
            "audio/wav",
            "audio/mp3", 
            "audio/flac",
            "audio/ogg",
            "audio/m4a",
            "audio/aac"
        ],
        "recommended_format": "audio/wav",
        "max_file_size_mb": 50
    })

# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check endpoint for audio service.
    """
    try:
        health_status = await audio_service.health_check()
        return JSONResponse(content=health_status)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )