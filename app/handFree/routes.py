from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from typing import List, Optional
import asyncio
import json
import base64
from datetime import datetime

from .schemas import (
    HandsFreeSessionRequest, HandsFreeSessionResponse, 
    InterviewResponseRequest, FacialAnalysisRequest,
    ConfidenceTipResponse, SessionStatus, AudioStreamData,
    RealTimeAnalysisResponse, SessionSettings
)
from .service import HandsFreeService

router = APIRouter(prefix="/hands-free", tags=["hands-free"])
hands_free_service = HandsFreeService()

# Session Management Routes
@router.post("/interview-response/", response_model=InterviewResponse)
async def create_interview_response(response: InterviewResponse):
    # Add your logic to save or process response
    return response

@router.get("/interview-response/", response_model=List[InterviewResponse])
async def get_interview_responses():
    # Add your logic to fetch all responses
    return []

# --- FacialAnalysis ---
@router.post("/facial-analysis/", response_model=FacialAnalysis)
async def analyze_facial(data: FacialAnalysisRequest):
    # Add logic to analyze facial data
    return FacialAnalysis()

@router.get("/facial-analysis/", response_model=List[FacialAnalysis])
async def get_facial_analyses():
    return []

# --- ConfidenceTipResponse ---
@router.post("/confidence-tip/", response_model=ConfidenceTipResponse)
async def create_confidence_tip(tip: ConfidenceTipResponse):
    return tip

@router.get("/confidence-tip/", response_model=List[ConfidenceTipResponse])
async def get_confidence_tips():
    return []

# --- AudioStreamResult ---
@router.post("/audio-stream-result/", response_model=AudioStreamResult)
async def create_audio_stream_result(result: AudioStreamResult):
    return result

@router.get("/audio-stream-result/", response_model=List[AudioStreamResult])
async def get_audio_stream_results():
    return []

@router.post("/session/start", response_model=HandsFreeSessionResponse)
async def start_hands_free_session(request: HandsFreeSessionRequest):
    """
    Start a new hands-free interview session
    Automatically configures default mic and initializes AI systems
    """
    try:
        session_id = await hands_free_service.create_session(
            user_id=request.user_id,
            default_mic_id=request.default_mic_id,
            interview_type=request.interview_type,
            company_info=request.company_info,
            job_role=request.job_role
        )
        
        # Auto-configure microphone and systems
        await hands_free_service.configure_audio_input(session_id, request.default_mic_id)
        await hands_free_service.initialize_ai_systems(session_id)
        
        return HandsFreeSessionResponse(
            session_id=session_id,
            status="active",
            message="Hands-free session started successfully. All systems automated.",
            mic_configured=True,
            ai_ready=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@router.post("/session/{session_id}/activate")
async def activate_hands_free_mode(session_id: str):
    """
    Activate hands-free mode - everything becomes fully automated
    """
    try:
        await hands_free_service.activate_hands_free_mode(session_id)
        return {"message": "Hands-free mode activated. System is now fully automated.", "status": "hands_free_active"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to activate hands-free mode: {str(e)}")

@router.get("/session/{session_id}/status", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """Get current session status and system health"""
    try:
        status = await hands_free_service.get_session_status(session_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {str(e)}")

# Real-time Audio Processing Routes
@router.websocket("/session/{session_id}/audio-stream")
async def audio_stream_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time audio streaming and automated response generation
    Handles continuous audio input and provides automated responses
    """
    await websocket.accept()
    
    try:
        # Verify session exists and is active
        await hands_free_service.verify_session(session_id)
        
        while True:
            # Receive audio data
            audio_data = await websocket.receive_bytes()
            
            # Process audio automatically
            result = await hands_free_service.process_audio_stream(
                session_id=session_id,
                audio_data=audio_data
            )
            
            # Send automated response if question detected
            if result.question_detected:
                response = await hands_free_service.generate_automated_response(
                    session_id=session_id,
                    question=result.detected_question,
                    context=result.context
                )
                
                await websocket.send_json({
                    "type": "automated_response",
                    "question": result.detected_question,
                    "response": response.response_text,
                    "key_insights": response.key_insights,
                    "confidence_score": response.confidence_score,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Send real-time status updates
            await websocket.send_json({
                "type": "status_update",
                "listening": result.is_listening,
                "processing": result.is_processing,
                "audio_level": result.audio_level
            })
            
    except WebSocketDisconnect:
        await hands_free_service.cleanup_session(session_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Audio processing error: {str(e)}"
        })

# Facial Analysis Routes
@router.websocket("/session/{session_id}/video-analysis")
async def video_analysis_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time facial analysis and confidence coaching
    Automatically analyzes facial expressions and provides tips
    """
    await websocket.accept()
    
    try:
        await hands_free_service.verify_session(session_id)
        
        while True:
            # Receive video frame
            frame_data = await websocket.receive_bytes()
            
            # Automated facial analysis
            analysis = await hands_free_service.analyze_facial_expression(
                session_id=session_id,
                frame_data=frame_data
            )
            
            # Generate confidence tips automatically
            tips = await hands_free_service.generate_confidence_tips(
                session_id=session_id,
                analysis_result=analysis
            )
            
            response = RealTimeAnalysisResponse(
                facial_analysis=analysis,
                confidence_tips=tips,
                overall_score=analysis.confidence_score,
                timestamp=datetime.utcnow().isoformat()
            )
            
            await websocket.send_json(response.dict())
            
    except WebSocketDisconnect:
        await hands_free_service.cleanup_video_analysis(session_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Video analysis error: {str(e)}"
        })

# Automated Response Generation Routes
@router.post("/session/{session_id}/manual-response")
async def generate_manual_response(session_id: str, request: InterviewResponseRequest):
    """
    Fallback route for manual response generation (rarely used in hands-free mode)
    """
    try:
        response = await hands_free_service.generate_interview_response(
            session_id=session_id,
            question=request.question,
            context=request.context,
            response_type=request.response_type
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

# Settings and Configuration Routes
@router.put("/session/{session_id}/settings")
async def update_session_settings(session_id: str, settings: SessionSettings):
    """Update session settings for automated responses"""
    try:
        await hands_free_service.update_session_settings(session_id, settings)
        return {"message": "Settings updated successfully", "settings": settings.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@router.get("/session/{session_id}/insights")
async def get_session_insights(session_id: str):
    """Get comprehensive insights from the hands-free session"""
    try:
        insights = await hands_free_service.get_session_insights(session_id)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

# System Health Routes
@router.get("/system/health")
async def system_health_check():
    """Check if all automated systems are working properly"""
    try:
        health = await hands_free_service.system_health_check()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System health check failed: {str(e)}")

@router.post("/session/{session_id}/stop")
async def stop_hands_free_session(session_id: str):
    """Stop hands-free session and cleanup resources"""
    try:
        summary = await hands_free_service.stop_session(session_id)
        return {
            "message": "Hands-free session stopped successfully",
            "session_summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop session: {str(e)}")

# Emergency Routes
@router.post("/session/{session_id}/emergency-pause")
async def emergency_pause(session_id: str):
    """Emergency pause for hands-free mode"""
    try:
        await hands_free_service.emergency_pause(session_id)
        return {"message": "System paused. User can manually resume.", "status": "paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency pause failed: {str(e)}")

@router.post("/session/{session_id}/resume")
async def resume_hands_free(session_id: str):
    """Resume hands-free mode after pause"""
    try:
        await hands_free_service.resume_hands_free(session_id)
        return {"message": "Hands-free mode resumed. System is automated again.", "status": "active"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume: {str(e)}")