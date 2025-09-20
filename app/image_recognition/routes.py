from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import List, Optional
import json
import asyncio
from datetime import datetime

from .schemas import (
    CameraDevice, CameraSessionRequest, CameraSessionResponse,
    ExpressionDetectionRequest, ExpressionDetectionResponse,
    ChatMessage, CameraStatusResponse, SimplifyMessageRequest,
    RecordingStatusResponse,
    StartRecordingRequest,
    StartRecordingResponse,
    StopRecordingResponse,
    AnalysisRequest,
    AnalysisResponse,
    RecordingInfo,
    HealthCheckResponse,
    PermissionStatusResponse,
    ScreenshotAnalysisRequest,
    ScreenshotAnalysisResponse
)
from .service import CameraService, ExpressionDetectionService, ChatService, ScreenRecordingService, AIAnalysisService, PermissionService

# Initialize routers
camera_router = APIRouter(prefix="/camera", tags=["camera"])
expression_router = APIRouter(prefix="/expression", tags=["expression"])
chat_router = APIRouter(prefix="/chat", tags=["chat"])


async def get_recording_service():
    return ScreenRecordingService()

async def get_ai_service():
    return AIAnalysisService()

async def get_permission_service():
    return PermissionService()

# Initialize services
camera_service = CameraService()
expression_service = ExpressionDetectionService()
chat_service = ChatService()

# Store active WebSocket connections
active_connections: List[WebSocket] = []

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# CAMERA ROUTES
@camera_router.get("/devices", response_model=List[CameraDevice])
async def get_available_cameras():
    """Get list of available camera devices"""
    try:
        devices = await camera_service.get_available_cameras()
        return devices
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cameras: {str(e)}")

@camera_router.post("/session/start", response_model=CameraSessionResponse)
async def start_camera_session(request: CameraSessionRequest):
    """Start a camera session with specified device"""
    try:
        session = await camera_service.start_session(
            device_id=request.device_id,
            resolution=request.resolution,
            fps=request.fps
        )
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start camera session: {str(e)}")

@camera_router.post("/session/stop/{session_id}")
async def stop_camera_session(session_id: str):
    """Stop an active camera session"""
    try:
        success = await camera_service.stop_session(session_id)
        if success:
            return {"message": "Camera session stopped successfully"}
        else:
            raise HTTPException(status_code=404, detail="Camera session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop camera session: {str(e)}")

@camera_router.get("/session/status/{session_id}", response_model=CameraStatusResponse)
async def get_camera_status(session_id: str):
    """Get status of a camera session"""
    try:
        status = await camera_service.get_session_status(session_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Camera session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get camera status: {str(e)}")

@camera_router.get("/stream/{session_id}")
async def get_camera_stream(session_id: str):
    """Get camera stream for a session"""
    try:
        stream = camera_service.get_video_stream(session_id)
        return StreamingResponse(
            stream,
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get camera stream: {str(e)}")

@camera_router.post("/test/{session_id}")
async def test_camera_connection(session_id: str):
    """Test camera connection and capture a test frame"""
    try:
        test_result = await camera_service.test_camera_connection(session_id)
        return test_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera test failed: {str(e)}")

# EXPRESSION DETECTION ROUTES
@expression_router.post("/detect", response_model=ExpressionDetectionResponse)
async def detect_expression(request: ExpressionDetectionRequest):
    """Detect facial expression from camera frame"""
    try:
        result = await expression_service.detect_expression(
            session_id=request.session_id,
            frame_data=request.frame_data,
            confidence_threshold=request.confidence_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expression detection failed: {str(e)}")

@expression_router.post("/start-monitoring/{session_id}")
async def start_expression_monitoring(session_id: str, interval_seconds: int = 2):
    """Start continuous expression monitoring for a camera session"""
    try:
        monitoring_id = await expression_service.start_monitoring(
            session_id=session_id,
            interval_seconds=interval_seconds
        )
        return {"monitoring_id": monitoring_id, "message": "Expression monitoring started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@expression_router.post("/stop-monitoring/{monitoring_id}")
async def stop_expression_monitoring(monitoring_id: str):
    """Stop expression monitoring"""
    try:
        success = await expression_service.stop_monitoring(monitoring_id)
        if success:
            return {"message": "Expression monitoring stopped"}
        else:
            raise HTTPException(status_code=404, detail="Monitoring session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

# CHAT ROUTES
@chat_router.get("/messages/{session_id}", response_model=List[ChatMessage])
async def get_chat_messages(session_id: str, limit: int = 50):
    """Get chat messages for a session"""
    try:
        messages = await chat_service.get_messages(session_id, limit)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@chat_router.post("/message/{session_id}")
async def add_chat_message(session_id: str, message: ChatMessage):
    """Add a new chat message"""
    try:
        saved_message = await chat_service.add_message(session_id, message)
        
        # Broadcast to WebSocket connections
        await manager.broadcast(json.dumps({
            "type": "new_message",
            "session_id": session_id,
            "message": saved_message.dict()
        }))
        
        return saved_message
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@chat_router.post("/simplify")
async def simplify_last_message(request: SimplifyMessageRequest):
    """Simplify the last AI message and send follow-up"""
    try:
        simplified_response = await chat_service.simplify_last_ai_message(
            session_id=request.session_id,
            original_message_id=request.original_message_id,
            confusion_confidence=request.confusion_confidence
        )
        
        # Broadcast simplified message
        await manager.broadcast(json.dumps({
            "type": "simplified_message",
            "session_id": request.session_id,
            "message": simplified_response.dict()
        }))
        
        return simplified_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to simplify message: {str(e)}")

# WEBSOCKET ROUTES
@camera_router.websocket("/ws/{session_id}")
async def camera_websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time camera and expression updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "start_monitoring":
                # Start expression monitoring
                monitoring_id = await expression_service.start_monitoring(
                    session_id=session_id,
                    interval_seconds=message_data.get("interval", 2)
                )
                await websocket.send_text(json.dumps({
                    "type": "monitoring_started",
                    "monitoring_id": monitoring_id
                }))
            
            elif message_data.get("type") == "camera_frame":
                # Process camera frame for expression detection
                result = await expression_service.detect_expression(
                    session_id=session_id,
                    frame_data=message_data.get("frame_data"),
                    confidence_threshold=0.7
                )
                
                # Send expression result back
                await websocket.send_text(json.dumps({
                    "type": "expression_detected",
                    "result": result.dict()
                }))
                
                # If confused expression detected, trigger simplification
                if result.primary_expression == "confused" and result.confidence > 0.8:
                    # Get last AI message and simplify it
                    messages = await chat_service.get_messages(session_id, limit=1)
                    if messages and messages[0].sender == "ai":
                        simplified = await chat_service.simplify_last_ai_message(
                            session_id=session_id,
                            original_message_id=messages[0].id,
                            confusion_confidence=result.confidence
                        )
                        
                        await websocket.send_text(json.dumps({
                            "type": "auto_simplified",
                            "message": simplified.dict()
                        }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))
        manager.disconnect(websocket)

# HEALTH CHECK
@camera_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "camera_service": camera_service.is_healthy(),
            "expression_service": expression_service.is_healthy(),
            "chat_service": chat_service.is_healthy()
        }
    }
@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check if the backend service is alive and reachable"""
    try:
        return HealthCheckResponse(
            status="connected",
            message="Backend service is running",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="disconnected",
            message=f"Service error: {str(e)}",
            timestamp=datetime.utcnow()
        )


@router.get("/permissions", response_model=PermissionStatusResponse)
async def check_permissions(
    permission_service: PermissionService = Depends(get_permission_service)
):
    """Check screen recording permissions status"""
    try:
        status = await permission_service.check_screen_recording_permission()
        return PermissionStatusResponse(
            has_permission=status.get("granted", False),
            permission_type=status.get("type", "unknown"),
            message=status.get("message", "Permission check completed"),
            needs_user_action=status.get("needs_action", False)
        )
    except Exception as e:
        logger.error(f"Permission check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check permissions: {str(e)}")


@router.post("/permissions/request")
async def request_permissions(
    permission_service: PermissionService = Depends(get_permission_service)
):
    """Request screen recording permissions from user"""
    try:
        result = await permission_service.request_screen_recording_permission()
        return {
            "success": result.get("granted", False),
            "message": result.get("message", "Permission request completed"),
            "action_required": result.get("action_required", False)
        }
    except Exception as e:
        logger.error(f"Permission request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to request permissions: {str(e)}")


@router.get("/status", response_model=RecordingStatusResponse)
async def get_recording_status(
    recording_service: ScreenRecordingService = Depends(get_recording_service)
):
    """Get current recording status"""
    try:
        status = await recording_service.get_recording_status()
        return RecordingStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get recording status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recording status: {str(e)}")


@router.post("/start", response_model=StartRecordingResponse)
async def start_recording(
    request: StartRecordingRequest,
    background_tasks: BackgroundTasks,
    recording_service: ScreenRecordingService = Depends(get_recording_service)
):
    """Start screen recording"""
    try:
        # Check permissions first
        permission_service = PermissionService()
        permission_status = await permission_service.check_screen_recording_permission()
        
        if not permission_status.get("granted", False):
            raise HTTPException(
                status_code=403, 
                detail="Screen recording permission not granted. Please enable permissions first."
            )
        
        # Start recording
        result = await recording_service.start_recording(
            quality=request.quality,
            include_audio=request.include_audio,
            capture_mouse=request.capture_mouse,
            frame_rate=request.frame_rate
        )
        
        # Start background screenshot capture if enabled
        if request.enable_screenshot_analysis:
            background_tasks.add_task(
                recording_service.start_screenshot_capture,
                result["recording_id"],
                request.screenshot_interval
            )
        
        return StartRecordingResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {str(e)}")


@router.post("/stop", response_model=StopRecordingResponse)
async def stop_recording(
    recording_service: ScreenRecordingService = Depends(get_recording_service)
):
    """Stop current recording"""
    try:
        result = await recording_service.stop_recording()
        return StopRecordingResponse(**result)
    except Exception as e:
        logger.error(f"Failed to stop recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop recording: {str(e)}")


@router.get("/list", response_model=List[RecordingInfo])
async def list_recordings(
    limit: Optional[int] = 50,
    offset: Optional[int] = 0,
    recording_service: ScreenRecordingService = Depends(get_recording_service)
):
    """List all available recordings"""
    try:
        recordings = await recording_service.list_recordings(limit=limit, offset=offset)
        return [RecordingInfo(**recording) for recording in recordings]
    except Exception as e:
        logger.error(f"Failed to list recordings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list recordings: {str(e)}")


@router.delete("/delete/{recording_id}")
async def delete_recording(
    recording_id: str,
    recording_service: ScreenRecordingService = Depends(get_recording_service)
):
    """Delete a specific recording"""
    try:
        result = await recording_service.delete_recording(recording_id)
        return {"success": result, "message": "Recording deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete recording {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete recording: {str(e)}")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_recording(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    ai_service: AIAnalysisService = Depends(get_ai_service),
    recording_service: ScreenRecordingService = Depends(get_recording_service)
):
    """Analyze recording with AI based on user question"""
    try:
        # Verify recording exists
        recording_info = await recording_service.get_recording_info(request.recording_id)
        if not recording_info:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        # Start AI analysis
        analysis_result = await ai_service.analyze_recording(
            recording_id=request.recording_id,
            question=request.question,
            analysis_type=request.analysis_type,
            time_range=request.time_range,
            include_screenshots=request.include_screenshots
        )
        
        return AnalysisResponse(**analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze recording {request.recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-screenshot", response_model=ScreenshotAnalysisResponse)
async def analyze_screenshot(
    request: ScreenshotAnalysisRequest,
    ai_service: AIAnalysisService = Depends(get_ai_service)
):
    """Analyze a specific screenshot or presentation slide"""
    try:
        result = await ai_service.analyze_screenshot(
            screenshot_data=request.screenshot_data,
            question=request.question,
            context=request.context,
            analysis_focus=request.analysis_focus
        )
        
        return ScreenshotAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to analyze screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Screenshot analysis failed: {str(e)}")


@router.post("/upload-screenshot")
async def upload_screenshot(
    file: UploadFile = File(...),
    question: Optional[str] = None,
    ai_service: AIAnalysisService = Depends(get_ai_service)
):
    """Upload and analyze a screenshot file"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await file.read()
        
        # Analyze the uploaded screenshot
        result = await ai_service.analyze_screenshot(
            screenshot_data=file_content,
            question=question or "What do you see in this image?",
            context="uploaded_screenshot"
        )
        
        return {
            "filename": file.filename,
            "analysis": result,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload analysis failed: {str(e)}")


@router.get("/analysis-history/{recording_id}")
async def get_analysis_history(
    recording_id: str,
    ai_service: AIAnalysisService = Depends(get_ai_service)
):
    """Get all analysis history for a specific recording"""
    try:
        history = await ai_service.get_analysis_history(recording_id)
        return {"recording_id": recording_id, "analyses": history}
    except Exception as e:
        logger.error(f"Failed to get analysis history for {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis history: {str(e)}")


@router.websocket("/ws/status")
async def websocket_status(websocket):
    """WebSocket endpoint for real-time status updates"""
    await websocket.accept()
    recording_service = ScreenRecordingService()
    
    try:
        while True:
            status = await recording_service.get_recording_status()
            await websocket.send_json(status)
            await asyncio.sleep(1)  # Send updates every second
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()


@router.websocket("/ws/analysis/{recording_id}")
async def websocket_analysis_progress(websocket, recording_id: str):
    """WebSocket endpoint for real-time analysis progress"""
    await websocket.accept()
    ai_service = AIAnalysisService()
    
    try:
        async for progress_update in ai_service.get_analysis_progress(recording_id):
            await websocket.send_json(progress_update)
    except Exception as e:
        logger.error(f"Analysis WebSocket error: {str(e)}")
        await websocket.close()