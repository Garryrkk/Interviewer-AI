from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import asyncio


from .schemas import (
    InvisibilityModeRequest,
    InvisibilityModeResponse,
    RecordingSessionRequest,
    RecordingSessionResponse,
    UIVisibilityRequest,
    UIVisibilityResponse,
    SessionStatusResponse,
    InsightGenerationRequest,
    InsightGenerationResponse,
    SecurityStatusResponse,
    HideModeEnum,
    RecordingTypeEnum,
    RecordingData,
    RecordingConfig,
    UIComponentEnum,
    InsightTypeEnum,
    InsightData,
    InvisibilityError,
    UIConfig,
    UIState,
    SecurityConfig,
    SecurityStatus,
    SessionData,
    SystemConfig,
    PerformanceMetrics,
    

)
from .service import InvisibilityService

router = APIRouter(prefix="/api/v1/invisibility", tags=["invisibility"])
invisibility_service = InvisibilityService()

@router.post("/mode/enable", response_model=InvisibilityModeResponse)
async def enable_invisibility_mode(
    request: InvisibilityModeRequest,
    background_tasks: BackgroundTasks
):
    """
    Enable invisibility mode for the interview AI assistant.
    Starts background recording and hides UI components.
    """
    try:
        session_id = str(uuid.uuid4())
        
        # Enable invisibility mode
        result = await invisibility_service.enable_invisibility_mode(
            session_id=session_id,
            recording_config=request.recording_config,
            ui_config=request.ui_config,
            security_config=request.security_config
        )
        
        # Start background tasks
        background_tasks.add_task(
            invisibility_service.start_background_recording,
            session_id,
            request.recording_config
        )
        
        return InvisibilityModeResponse(
            success=True,
            session_id=session_id,
            message="Invisibility mode enabled successfully",
            ui_state=result.get("ui_state"),
            recording_state=result.get("recording_state"),
            security_status=result.get("security_status")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable invisibility mode: {str(e)}")
    
@router.post("/recording/start", response_model=RecordingData)
async def start_recording(config: RecordingConfig):
    """
    Start a new recording with the given configuration.
    """
    return RecordingData(
        type=config.type,
        started=True,
        message="Recording started successfully",
    )


@router.get("/recording/{recording_type}", response_model=RecordingData)
async def get_recording(recording_type: RecordingTypeEnum):
    """
    Fetch info for a specific recording type.
    """
    return RecordingData(type=recording_type, started=False, message="Stub response")


# -------------------- UI --------------------
@router.get("/ui/state", response_model=UIState)
async def get_ui_state():
    """
    Get current UI state/config.
    """
    return UIState(active_component=UIComponentEnum.PLAYER, config=UIConfig())


@router.put("/ui/config", response_model=UIConfig)
async def update_ui_config(config: UIConfig):
    """
    Update UI configuration.
    """
    return config


# -------------------- Insights --------------------
@router.post("/insights", response_model=InsightData)
async def generate_insight(insight_type: InsightTypeEnum):
    """
    Generate insight of a specific type.
    """
    return InsightData(type=insight_type, result="Sample insight")


# -------------------- Security --------------------
@router.get("/security/status", response_model=SecurityStatusResponse)
async def get_security_status():
    """
    Get current security status of the system.
    """
    return SecurityStatusResponse(status=SecurityStatus.OK)


@router.post("/security/config", response_model=SecurityConfig)
async def update_security_config(config: SecurityConfig):
    """
    Update security configuration.
    """
    return config


# -------------------- System --------------------
@router.get("/system/config", response_model=SystemConfig)
async def get_system_config():
    """
    Retrieve overall system configuration.
    """
    return SystemConfig()


@router.get("/session", response_model=SessionData)
async def get_session_data():
    """
    Return session-related information.
    """
    return SessionData(session_id="dummy", user="placeholder")

@router.post("/mode/disable", response_model=InvisibilityModeResponse)
async def disable_invisibility_mode(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """
    Disable invisibility mode and restore normal UI visibility.
    Stops background recording and processes final insights.
    """
    try:
        result = await invisibility_service.disable_invisibility_mode(session_id)
        
        # Process final insights in background
        background_tasks.add_task(
            invisibility_service.generate_final_insights,
            session_id
        )
        
        return InvisibilityModeResponse(
            success=True,
            session_id=session_id,
            message="Invisibility mode disabled successfully",
            ui_state=result.get("ui_state"),
            recording_state=result.get("recording_state"),
            final_insights_url=result.get("insights_url")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable invisibility mode: {str(e)}")

@router.post("/recording/start", response_model=RecordingSessionResponse)
async def start_invisible_recording(
    request: RecordingSessionRequest,
    background_tasks: BackgroundTasks
):
    """
    Start invisible recording session with specified configuration.
    Recording happens in background without UI indication.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        result = await invisibility_service.start_invisible_recording(
            session_id=session_id,
            screen_recording=request.screen_recording,
            voice_recording=request.voice_recording,
            auto_notes=request.auto_notes,
            real_time_insights=request.real_time_insights
        )
        
        return RecordingSessionResponse(
            success=True,
            session_id=session_id,
            recording_started=True,
            message="Invisible recording started successfully",
            recording_config=result.get("config"),
            estimated_duration=request.estimated_duration
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start invisible recording: {str(e)}")

@router.post("/recording/stop", response_model=RecordingSessionResponse)
async def stop_invisible_recording(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """
    Stop invisible recording and begin processing captured data.
    """
    try:
        result = await invisibility_service.stop_invisible_recording(session_id)
        
        # Process recording in background
        background_tasks.add_task(
            invisibility_service.process_recording_data,
            session_id
        )
        
        return RecordingSessionResponse(
            success=True,
            session_id=session_id,
            recording_started=False,
            message="Invisible recording stopped successfully",
            recording_duration=result.get("duration"),
            data_size=result.get("data_size"),
            processing_status="started"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop invisible recording: {str(e)}")

@router.post("/ui/hide", response_model=UIVisibilityResponse)
async def hide_ui_components(request: UIVisibilityRequest):
    """
    Hide specified UI components to maintain invisibility during screen sharing.
    """
    try:
        result = await invisibility_service.hide_ui_components(
            session_id=request.session_id,
            components_to_hide=request.components_to_hide,
            hide_mode=request.hide_mode
        )
        
        return UIVisibilityResponse(
            success=True,
            session_id=request.session_id,
            hidden_components=result.get("hidden_components"),
            ui_state="hidden",
            message="UI components hidden successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to hide UI components: {str(e)}")

@router.post("/ui/show", response_model=UIVisibilityResponse)
async def show_ui_components(request: UIVisibilityRequest):
    """
    Restore visibility of UI components when screen sharing ends.
    """
    try:
        result = await invisibility_service.show_ui_components(
            session_id=request.session_id,
            components_to_show=request.components_to_show
        )
        
        return UIVisibilityResponse(
            success=True,
            session_id=request.session_id,
            visible_components=result.get("visible_components"),
            ui_state="visible",
            message="UI components restored successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to show UI components: {str(e)}")

@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    Get current status of invisibility session including recording and UI state.
    """
    try:
        status = await invisibility_service.get_session_status(session_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionStatusResponse(
            session_id=session_id,
            is_active=status.get("is_active", False),
            invisibility_enabled=status.get("invisibility_enabled", False),
            recording_status=status.get("recording_status"),
            ui_state=status.get("ui_state"),
            start_time=status.get("start_time"),
            duration=status.get("duration"),
            data_captured=status.get("data_captured", {}),
            security_status=status.get("security_status")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")

@router.post("/insights/generate", response_model=InsightGenerationResponse)
async def generate_background_insights(
    request: InsightGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate AI insights from captured data without showing progress to interviewer.
    """
    try:
        # Start insight generation in background
        background_tasks.add_task(
            invisibility_service.generate_insights,
            request.session_id,
            request.insight_types,
            request.processing_options
        )
        
        return InsightGenerationResponse(
            success=True,
            session_id=request.session_id,
            generation_started=True,
            message="Background insight generation started",
            estimated_completion_time=request.processing_options.get("estimated_time"),
            insight_types=request.insight_types
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start insight generation: {str(e)}")

@router.get("/insights/{session_id}", response_model=Dict[str, Any])
async def get_generated_insights(session_id: str):
    """
    Retrieve generated insights for a completed invisibility session.
    """
    try:
        insights = await invisibility_service.get_session_insights(session_id)
        
        if not insights:
            raise HTTPException(status_code=404, detail="No insights found for this session")
        
        return {
            "session_id": session_id,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve insights: {str(e)}")

@router.get("/security/status/{session_id}", response_model=SecurityStatusResponse)
async def get_security_status(session_id: str):
    """
    Check security status to ensure no data leakage to interviewer.
    """
    try:
        security_status = await invisibility_service.get_security_status(session_id)
        
        return SecurityStatusResponse(
            session_id=session_id,
            data_encrypted=security_status.get("data_encrypted", False),
            local_processing=security_status.get("local_processing", False),
            no_external_leaks=security_status.get("no_external_leaks", False),
            secure_storage=security_status.get("secure_storage", False),
            privacy_compliant=security_status.get("privacy_compliant", False),
            security_score=security_status.get("security_score", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security status: {str(e)}")

@router.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up session data and remove all traces of invisibility session.
    """
    try:
        result = await invisibility_service.cleanup_session(session_id)
        
        return JSONResponse(
            content={
                "success": True,
                "session_id": session_id,
                "message": "Session cleaned up successfully",
                "data_removed": result.get("data_removed", [])
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup session: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check endpoint for invisibility service.
    """
    try:
        health = await invisibility_service.health_check()
        return JSONResponse(content=health)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )
    
# ============================
# HideModeEnum Routes
# ============================

@router.get("/hide-modes", response_model=List[HideModeEnum])
async def list_hide_modes():
    """
    Return all available HideModeEnum options.
    """
    return [mode for mode in HideModeEnum]

@router.get("/hide-modes/{mode}", response_model=HideModeEnum)
async def get_hide_mode(mode: HideModeEnum):
    """
    Validate and return a single HideModeEnum value.
    """
    return mode


# ============================
# InvisibilityError Routes
# ============================

@router.post("/invisibility-error", response_model=InvisibilityError)
async def create_invisibility_error(error: InvisibilityError):
    """
    Store or log an invisibility-related error.
    For demo, simply echoes the request.
    """
    return error

@router.get("/invisibility-error/{error_code}", response_model=InvisibilityError)
async def get_invisibility_error(error_code: str):
    """
    Fetch an example InvisibilityError by code.
    Replace with DB lookup in production.
    """
    return InvisibilityError(
        error_code=error_code,
        message="Sample error message",
        session_id="session_123",
        details={"info": "Example details"},
        timestamp=datetime.utcnow()
    )


# ============================
# PerformanceMetrics Routes
# ============================

_fake_metrics_store: List[PerformanceMetrics] = []  # In-memory demo storage

@router.post("/performance-metrics", response_model=PerformanceMetrics)
async def create_performance_metrics(metrics: PerformanceMetrics):
    """
    Submit performance metrics for a given session.
    """
    _fake_metrics_store.append(metrics)
    return metrics

@router.get("/performance-metrics/{session_id}", response_model=List[PerformanceMetrics])
async def get_performance_metrics(session_id: str):
    """
    Retrieve all performance metrics for a specific session_id.
    """
    return [m for m in _fake_metrics_store if m.session_id == session_id]

@router.get("/performance-metrics", response_model=List[PerformanceMetrics])
async def list_all_performance_metrics(
    limit: int = Query(50, ge=1, le=500, description="Max number of records to return")
):
    """
    List all collected performance metrics (demo).
    """
    return _fake_metrics_store[:limit]

@router.websocket("/ws")
async def invisibility_ws(websocket: WebSocket):
    """WebSocket endpoint for real-time invisibility updates"""
    await websocket.accept()
    try:
        while True:
            # (Optional) receive data
            try:
                await websocket.receive_text()
            except Exception:
                pass

            # send a sample heartbeat / status
            await websocket.send_json({
                "type": "heartbeat",
                "message": "Connection alive",
                "timestamp": datetime.utcnow().isoformat()
            })
            await asyncio.sleep(2)

    except WebSocketDisconnect:
        return
