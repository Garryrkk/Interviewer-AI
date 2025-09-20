import cv2
import asyncio
import uuid
import base64
import numpy as np
import json
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncGenerator
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import platform
import subprocess
from PIL import Image, ImageGrab
import threading
from pathlib import Path
import cv2

from .schemas import (
    CameraDevice, CameraSessionResponse, CameraStatusResponse, CameraTestResult,
    ExpressionDetectionResponse, FaceExpression, ChatMessage, ExpressionType,
    CameraStatus, CameraResolution, MessageSender, MonitoringSession,
    ExpressionMonitoringConfig,RecordingStatus, RecordingQuality, AnalysisType, AnalysisFocus,
    RecordingMetadata, ScreenshotData, AnalysisJob
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraService:
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.camera_devices: List[CameraDevice] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def get_available_cameras(self) -> List[CameraDevice]:
        """Get list of available camera devices"""
        devices = []
        
        # Check for available cameras (0-10 range)
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    device = CameraDevice(
                        device_id=str(i),
                        name=f"Camera {i}",
                        is_available=True,
                        capabilities={
                            "width": width,
                            "height": height,
                            "fps": fps,
                            "backend": cap.getBackendName()
                        },
                        resolution_options=[
                            CameraResolution.LOW,
                            CameraResolution.MEDIUM,
                            CameraResolution.HIGH
                        ],
                        max_fps=fps if fps > 0 else 30
                    )
                    devices.append(device)
                    cap.release()
            except Exception as e:
                logger.warning(f"Error checking camera {i}: {e}")
                continue
        
        self.camera_devices = devices
        return devices
    
    async def start_session(self, device_id: str, resolution: CameraResolution, fps: int) -> CameraSessionResponse:
        """Start a new camera session"""
        session_id = str(uuid.uuid4())
        
        try:
            # Initialize camera capture
            cap = cv2.VideoCapture(int(device_id))
            
            if not cap.isOpened():
                raise Exception(f"Failed to open camera {device_id}")
            
            # Set camera properties
            width, height = self._parse_resolution(resolution)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Test capture
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise Exception("Failed to capture test frame")
            
            # Store session info
            session_data = {
                "session_id": session_id,
                "device_id": device_id,
                "capture": cap,
                "resolution": resolution,
                "fps": fps,
                "status": CameraStatus.ACTIVE,
                "created_at": datetime.now(),
                "last_frame": None,
                "frame_count": 0,
                "is_streaming": True
            }
            
            self.active_sessions[session_id] = session_data
            
            return CameraSessionResponse(
                session_id=session_id,
                device_id=device_id,
                status=CameraStatus.ACTIVE,
                resolution=resolution,
                fps=fps,
                created_at=session_data["created_at"],
                stream_url=f"/camera/stream/{session_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to start camera session: {e}")
            raise Exception(f"Failed to start camera session: {e}")
    
    async def stop_session(self, session_id: str) -> bool:
        """Stop a camera session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Release camera
            if "capture" in session and session["capture"]:
                session["capture"].release()
            
            # Update status
            session["status"] = CameraStatus.INACTIVE
            session["is_streaming"] = False
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Camera session {session_id} stopped")
            return True
        return False
    
    async def get_session_status(self, session_id: str) -> Optional[CameraStatusResponse]:
        """Get status of a camera session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        uptime = (datetime.now() - session["created_at"]).total_seconds()
        
        return CameraStatusResponse(
            session_id=session_id,
            status=session["status"],
            device_id=session["device_id"],
            is_streaming=session.get("is_streaming", False),
            resolution=session["resolution"],
            fps=session["fps"],
            uptime_seconds=uptime,
            last_frame_timestamp=session.get("last_frame"),
            error_message=session.get("error_message")
        )
    
    async def test_camera_connection(self, session_id: str) -> CameraTestResult:
        """Test camera connection and capture a test frame"""
        if session_id not in self.active_sessions:
            raise Exception("Camera session not found")
        
        session = self.active_sessions[session_id]
        cap = session["capture"]
        
        start_time = time.time()
        
        try:
            ret, frame = cap.read()
            latency = (time.time() - start_time) * 1000
            
            if ret:
                height, width = frame.shape[:2]
                actual_resolution = f"{width}x{height}"
                
                return CameraTestResult(
                    session_id=session_id,
                    success=True,
                    frame_captured=True,
                    latency_ms=latency,
                    resolution_actual=actual_resolution
                )
            else:
                return CameraTestResult(
                    session_id=session_id,
                    success=False,
                    frame_captured=False,
                    latency_ms=latency,
                    resolution_actual="0x0",
                    error_details="Failed to capture frame"
                )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return CameraTestResult(
                session_id=session_id,
                success=False,
                frame_captured=False,
                latency_ms=latency,
                resolution_actual="0x0",
                error_details=str(e)
            )
    
    def get_video_stream(self, session_id: str) -> AsyncGenerator[bytes, None]:
        """Get video stream for a session"""
        async def generate():
            if session_id not in self.active_sessions:
                return
            
            session = self.active_sessions[session_id]
            cap = session["capture"]
            
            while session["is_streaming"]:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Update session info
                    session["last_frame"] = datetime.now()
                    session["frame_count"] += 1
                    
                    # Yield frame in multipart format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    await asyncio.sleep(1.0 / session["fps"])
                    
                except Exception as e:
                    logger.error(f"Error in video stream: {e}")
                    break
        
        return generate()
    
    def get_current_frame(self, session_id: str) -> Optional[np.ndarray]:
        """Get current frame from camera session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        cap = session["capture"]
        
        try:
            ret, frame = cap.read()
            if ret:
                session["last_frame"] = datetime.now()
                return frame
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
        
        return None
    
    def _parse_resolution(self, resolution: CameraResolution) -> tuple:
        """Parse resolution string to width, height tuple"""
        resolution_map = {
            CameraResolution.LOW: (640, 480),
            CameraResolution.MEDIUM: (1280, 720),
            CameraResolution.HIGH: (1920, 1080)
        }
        return resolution_map.get(resolution, (1280, 720))
    
    def is_healthy(self) -> bool:
        """Check if camera service is healthy"""
        try:
            active_count = len(self.active_sessions)
            return active_count >= 0  # Simple health check
        except:
            return False


class ExpressionDetectionService:
    def __init__(self):
        self.face_cascade = None
        self.expression_model = None
        self.monitoring_sessions: Dict[str, MonitoringSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize face detection and expression recognition models"""
        try:
            # Load OpenCV face cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("Face detection model loaded successfully")
            
            # Note: In production, you would load a trained expression recognition model
            # For demo purposes, we'll simulate expression detection
            self.expression_model = "simulated_model"
            
        except Exception as e:
            logger.error(f"Error initializing expression models: {e}")
    
    async def detect_expression(self, session_id: str, frame_data: Optional[str] = None, 
                              confidence_threshold: float = 0.5) -> ExpressionDetectionResponse:
        """Detect facial expressions from camera frame"""
        start_time = time.time()
        
        try:
            # Get frame from camera or decode from base64
            if frame_data:
                # Decode base64 frame data
                frame_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Get current frame from camera session
                camera_service = CameraService()
                frame = camera_service.get_current_frame(session_id)
                
                if frame is None:
                    raise Exception("Could not capture frame from camera")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Process each face
            all_expressions = []
            primary_expression = ExpressionType.UNKNOWN
            max_confidence = 0.0
            
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = gray[y:y+h, x:x+w]
                
                # Simulate expression detection (replace with actual model inference)
                expression, confidence = self._simulate_expression_detection(face_roi)
                
                if confidence >= confidence_threshold:
                    face_expr = FaceExpression(
                        expression=expression,
                        confidence=confidence,
                        face_id=f"face_{i}",
                        bounding_box={
                            "x": float(x),
                            "y": float(y),
                            "width": float(w),
                            "height": float(h)
                        }
                    )
                    all_expressions.append(face_expr)
                    
                    if confidence > max_confidence:
                        primary_expression = expression
                        max_confidence = confidence
            
            processing_time = (time.time() - start_time) * 1000
            frame_quality = self._calculate_frame_quality(frame)
            
            return ExpressionDetectionResponse(
                session_id=session_id,
                timestamp=datetime.now(),
                faces_detected=len(faces),
                primary_expression=primary_expression,
                confidence=max_confidence,
                all_expressions=all_expressions,
                processing_time_ms=processing_time,
                frame_quality=frame_quality
            )
            
        except Exception as e:
            logger.error(f"Expression detection error: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return ExpressionDetectionResponse(
                session_id=session_id,
                timestamp=datetime.now(),
                faces_detected=0,
                primary_expression=ExpressionType.UNKNOWN,
                confidence=0.0,
                all_expressions=[],
                processing_time_ms=processing_time,
                frame_quality=0.0
            )
    
    def _simulate_expression_detection(self, face_roi: np.ndarray) -> tuple:
        """Simulate expression detection (replace with actual model)"""
        import random
        
        expressions = [
            ExpressionType.HAPPY,
            ExpressionType.NEUTRAL,
            ExpressionType.CONFUSED,
            ExpressionType.SURPRISED,
            ExpressionType.SAD
        ]
        
        # Simulate detection with some logic
        face_area = face_roi.shape[0] * face_roi.shape[1]
        brightness = np.mean(face_roi)
        
        # Higher chance of confusion if face is small or dark (poor quality)
        if face_area < 5000 or brightness < 80:
            expression = ExpressionType.CONFUSED
            confidence = random.uniform(0.6, 0.9)
        else:
            expression = random.choice(expressions)
            confidence = random.uniform(0.5, 0.95)
        
        return expression, confidence
    
    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """Calculate frame quality score"""
        try:
            # Simple quality metrics
            brightness = np.mean(frame)
            contrast = np.std(frame)
            
            # Normalize to 0-1 scale
            brightness_score = min(brightness / 128, 1.0)
            contrast_score = min(contrast / 64, 1.0)
            
            return (brightness_score + contrast_score) / 2
        except:
            return 0.5
    
    async def start_monitoring(self, session_id: str, interval_seconds: int = 2) -> str:
        """Start continuous expression monitoring"""
        monitoring_id = str(uuid.uuid4())
        
        config = ExpressionMonitoringConfig(
            session_id=session_id,
            interval_seconds=interval_seconds
        )
        
        monitoring_session = MonitoringSession(
            monitoring_id=monitoring_id,
            camera_session_id=session_id,
            status="active",
            config=config,
            created_at=datetime.now()
        )
        
        self.monitoring_sessions[monitoring_id] = monitoring_session
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop(monitoring_id))
        
        return monitoring_id
    
    async def _monitoring_loop(self, monitoring_id: str):
        """Continuous monitoring loop"""
        session = self.monitoring_sessions.get(monitoring_id)
        if not session:
            return
        
        while session.status == "active":
            try:
                # Detect expression
                result = await self.detect_expression(
                    session_id=session.camera_session_id,
                    confidence_threshold=session.config.confidence_threshold
                )
                
                # Update monitoring stats
                session.last_detection = datetime.now()
                session.total_detections += 1
                
                # Check for confusion
                if (result.primary_expression == ExpressionType.CONFUSED and 
                    result.confidence >= session.config.confusion_trigger_threshold):
                    
                    session.confusion_events += 1
                    
                    # Trigger auto-simplification if enabled
                    if session.config.auto_simplify:
                        await self._trigger_auto_simplification(session.camera_session_id, result.confidence)
                
                # Wait for next interval
                await asyncio.sleep(session.config.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)
    
    async def _trigger_auto_simplification(self, session_id: str, confusion_confidence: float):
        """Trigger automatic message simplification"""
        try:
            chat_service = ChatService()
            await chat_service.handle_confusion_trigger(session_id, confusion_confidence)
        except Exception as e:
            logger.error(f"Error triggering auto-simplification: {e}")
    
    async def stop_monitoring(self, monitoring_id: str) -> bool:
        """Stop expression monitoring"""
        if monitoring_id in self.monitoring_sessions:
            self.monitoring_sessions[monitoring_id].status = "stopped"
            del self.monitoring_sessions[monitoring_id]
            return True
        return False
    
    def is_healthy(self) -> bool:
        """Check if expression service is healthy"""
        try:
            return self.face_cascade is not None
        except:
            return False


class ChatService:
    def __init__(self):
        self.messages: Dict[str, List[ChatMessage]] = {}
        self.ai_service_url = "http://localhost:8001/ai"  # Configure as needed
    
    async def get_messages(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get chat messages for a session"""
        if session_id not in self.messages:
            return []
        
        messages = self.messages[session_id]
        return messages[-limit:] if limit > 0 else messages
    
    async def add_message(self, session_id: str, message: ChatMessage) -> ChatMessage:
        """Add a new chat message"""
        if session_id not in self.messages:
            self.messages[session_id] = []
        
        # Generate ID if not provided
        if not message.id:
            message.id = str(uuid.uuid4())
        
        # Set session ID
        message.session_id = session_id
        
        # Add timestamp if not provided
        if not message.timestamp:
            message.timestamp = datetime.now()
        
        self.messages[session_id].append(message)
        
        # Limit message history
        if len(self.messages[session_id]) > 1000:
            self.messages[session_id] = self.messages[session_id][-500:]
        
        return message
    
    async def simplify_last_ai_message(self, session_id: str, original_message_id: str, 
                                     confusion_confidence: float) -> ChatMessage:
        """Simplify the last AI message and add as new message"""
        try:
            # Find the original message
            original_message = None
            if session_id in self.messages:
                for msg in reversed(self.messages[session_id]):
                    if msg.id == original_message_id and msg.sender == MessageSender.AI:
                        original_message = msg
                        break
            
            if not original_message:
                raise Exception("Original AI message not found")
            
            # Simplify the message (call to AI service or use local logic)
            simplified_content = await self._simplify_content(original_message.content, confusion_confidence)
            
            # Create simplified message
            simplified_message = ChatMessage(
                session_id=session_id,
                sender=MessageSender.AI,
                content=simplified_content,
                is_simplified=True,
                original_message_id=original_message_id,
                simplification_reason=f"Confusion detected with {confusion_confidence:.2f} confidence",
                metadata={
                    "confusion_confidence": confusion_confidence,
                    "auto_generated": True,
                    "simplification_level": 1
                }
            )
            
            # Add the simplified message
            return await self.add_message(session_id, simplified_message)
            
        except Exception as e:
            logger.error(f"Error simplifying message: {e}")
            
            # Return a generic clarification message
            fallback_message = ChatMessage(
                session_id=session_id,
                sender=MessageSender.AI,
                content="Let me explain that more simply: I noticed you might be confused. Could you tell me which part you'd like me to clarify?",
                is_simplified=True,
                original_message_id=original_message_id,
                simplification_reason="Fallback due to simplification error",
                metadata={
                    "confusion_confidence": confusion_confidence,
                    "fallback": True
                }
            )
            
            return await self.add_message(session_id, fallback_message)
    
    async def _simplify_content(self, content: str, confusion_confidence: float) -> str:
        """Simplify content using AI service or local logic"""
        try:
            # Attempt to call AI simplification service
            async with aiohttp.ClientSession() as session:
                payload = {
                    "text": content,
                    "simplification_level": min(int(confusion_confidence * 3), 3),
                    "max_length": len(content.split()) * 2  # Allow up to 2x original length
                }
                
                async with session.post(f"{self.ai_service_url}/simplify", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("simplified_text", content)
        
        except Exception as e:
            logger.warning(f"AI simplification service unavailable: {e}")
        
        # Fallback to simple local simplification
        return self._simple_local_simplification(content)
    
    def _simple_local_simplification(self, content: str) -> str:
        """Simple local content simplification"""
        # Basic simplification rules
        simplified = content
        
        # Replace complex words with simpler ones
        replacements = {
            "utilize": "use",
            "demonstrate": "show",
            "facilitate": "help",
            "subsequently": "then",
            "furthermore": "also",
            "therefore": "so",
            "however": "but",
            "nevertheless": "but",
            "consequently": "so"
        }
        
        for complex_word, simple_word in replacements.items():
            simplified = simplified.replace(complex_word, simple_word)
        
        # Break long sentences
        sentences = simplified.split('. ')
        if len(sentences) > 3:
            simplified = '. '.join(sentences[:2]) + '. Let me know if you need more details.'
        
        # Add clarifying prefix
        simplified = "Let me simplify: " + simplified
        
        return simplified
    
    async def handle_confusion_trigger(self, session_id: str, confusion_confidence: float):
        """Handle automatic confusion trigger"""
        try:
            # Get the last AI message
            messages = await self.get_messages(session_id, limit=10)
            last_ai_message = None
            
            for msg in reversed(messages):
                if msg.sender == MessageSender.AI and not msg.is_simplified:
                    last_ai_message = msg
                    break
            
            if last_ai_message:
                await self.simplify_last_ai_message(
                    session_id, 
                    last_ai_message.id, 
                    confusion_confidence
                )
                
                logger.info(f"Auto-simplification triggered for session {session_id}")
        
        except Exception as e:
            logger.error(f"Error handling confusion trigger: {e}")
    
    def is_healthy(self) -> bool:
        """Check if chat service is healthy"""
        try:
            return len(self.messages) >= 0
        except:
            return False
        
    class PermissionService:
        """Handles system permissions for screen recording"""
    
    def __init__(self):
        self.os_type = platform.system().lower()
    
    async def check_screen_recording_permission(self) -> Dict[str, Any]:
        """Check if screen recording permission is granted"""
        try:
            if self.os_type == "darwin":  # macOS
                return await self._check_macos_permission()
            elif self.os_type == "windows":
                return await self._check_windows_permission()
            elif self.os_type == "linux":
                return await self._check_linux_permission()
            else:
                return {
                    "granted": False,
                    "type": "unsupported",
                    "message": f"Unsupported OS: {self.os_type}",
                    "needs_action": True
                }
        except Exception as e:
            logger.error(f"Permission check failed: {str(e)}")
            return {
                "granted": False,
                "type": "error",
                "message": f"Permission check failed: {str(e)}",
                "needs_action": True
            }
    
    async def _check_macos_permission(self) -> Dict[str, Any]:
        """Check macOS screen recording permission"""
        try:
            # Test screen capture capability
            screenshot = ImageGrab.grab()
            if screenshot.size[0] < 100 or screenshot.size[1] < 100:
                return {
                    "granted": False,
                    "type": "screen_recording",
                    "message": "Screen recording permission not granted. Please enable in System Preferences > Security & Privacy > Screen Recording",
                    "needs_action": True
                }
            
            return {
                "granted": True,
                "type": "screen_recording",
                "message": "Screen recording permission granted",
                "needs_action": False
            }
        except Exception as e:
            return {
                "granted": False,
                "type": "error",
                "message": f"Permission check failed: {str(e)}",
                "needs_action": True
            }
    
    async def _check_windows_permission(self) -> Dict[str, Any]:
        """Check Windows screen recording capability"""
        try:
            screenshot = ImageGrab.grab()
            return {
                "granted": True,
                "type": "screen_recording",
                "message": "Screen recording available",
                "needs_action": False
            }
        except Exception as e:
            return {
                "granted": False,
                "type": "error",
                "message": f"Screen capture failed: {str(e)}",
                "needs_action": True
            }
    
    async def _check_linux_permission(self) -> Dict[str, Any]:
        """Check Linux screen recording capability"""
        try:
            # Check if X11 or Wayland display is available
            display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
            if not display:
                return {
                    "granted": False,
                    "type": "display",
                    "message": "No display server detected",
                    "needs_action": True
                }
            
            screenshot = ImageGrab.grab()
            return {
                "granted": True,
                "type": "screen_recording",
                "message": "Screen recording available",
                "needs_action": False
            }
        except Exception as e:
            return {
                "granted": False,
                "type": "error",
                "message": f"Screen capture failed: {str(e)}",
                "needs_action": True
            }
    
    async def request_screen_recording_permission(self) -> Dict[str, Any]:
        """Request screen recording permission from user"""
        if self.os_type == "darwin":
            return {
                "granted": False,
                "message": "Please grant screen recording permission in System Preferences > Security & Privacy > Privacy > Screen Recording",
                "action_required": True
            }
        elif self.os_type == "windows":
            return {
                "granted": True,
                "message": "No additional permissions required on Windows",
                "action_required": False
            }
        else:
            return {
                "granted": True,
                "message": "Please ensure your display server is running",
                "action_required": False
            }


class ScreenRecordingService:
    """Core service for screen recording functionality"""
    
    def __init__(self):
        self.recordings_dir = Path("recordings")
        self.screenshots_dir = Path("screenshots")
        self.recordings_dir.mkdir(exist_ok=True)
        self.screenshots_dir.mkdir(exist_ok=True)
        
        self.current_recording: Optional[Dict[str, Any]] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.screenshot_thread: Optional[threading.Thread] = None
        self.is_recording = False
        self.is_capturing_screenshots = False
        
        # Recording metadata storage (in production, use a database)
        self.recordings_db: Dict[str, RecordingMetadata] = {}
        self.screenshots_db: Dict[str, List[ScreenshotData]] = {}
    
    async def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status"""
        if not self.current_recording:
            return {
                "status": RecordingStatus.IDLE,
                "recording_id": None,
                "duration_seconds": 0,
                "file_size_mb": 0,
                "start_time": None,
                "is_paused": False,
                "screenshots_captured": 0
            }
        
        duration = (datetime.utcnow() - self.current_recording["start_time"]).total_seconds()
        file_size = self._get_file_size_mb(self.current_recording.get("file_path", ""))
        screenshots_count = len(self.screenshots_db.get(self.current_recording["recording_id"], []))
        
        return {
            "status": RecordingStatus.RECORDING if self.is_recording else RecordingStatus.IDLE,
            "recording_id": self.current_recording["recording_id"],
            "duration_seconds": duration,
            "file_size_mb": file_size,
            "start_time": self.current_recording["start_time"],
            "is_paused": False,
            "screenshots_captured": screenshots_count,
            "quality_settings": self.current_recording.get("settings", {})
        }
    
    async def start_recording(
        self,
        quality: RecordingQuality = RecordingQuality.MEDIUM,
        include_audio: bool = True,
        capture_mouse: bool = True,
        frame_rate: int = 30
    ) -> Dict[str, Any]:
        """Start screen recording"""
        if self.is_recording:
            raise Exception("Recording already in progress")
        
        recording_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}_{recording_id[:8]}.mp4"
        file_path = self.recordings_dir / filename
        
        # Quality settings
        quality_settings = self._get_quality_settings(quality, frame_rate)
        
        self.current_recording = {
            "recording_id": recording_id,
            "file_path": str(file_path),
            "start_time": datetime.utcnow(),
            "settings": {
                "quality": quality,
                "include_audio": include_audio,
                "capture_mouse": capture_mouse,
                "frame_rate": frame_rate,
                **quality_settings
            }
        }
        
        # Start recording in separate thread
        self.is_recording = True
        self.recording_thread = threading.Thread(
            target=self._record_screen,
            args=(str(file_path), quality_settings, include_audio, capture_mouse)
        )
        self.recording_thread.start()
        
        # Initialize screenshot storage
        self.screenshots_db[recording_id] = []
        
        return {
            "success": True,
            "recording_id": recording_id,
            "message": "Screen recording started successfully",
            "start_time": self.current_recording["start_time"],
            "estimated_file_size_mb_per_minute": self._estimate_file_size_per_minute(quality_settings),
            "settings": self.current_recording["settings"]
        }
    
    async def stop_recording(self) -> Dict[str, Any]:
        """Stop current recording"""
        if not self.is_recording or not self.current_recording:
            raise Exception("No recording in progress")
        
        # Stop recording
        self.is_recording = False
        self.is_capturing_screenshots = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=10)
        
        # Wait for screenshot thread to finish
        if self.screenshot_thread:
            self.screenshot_thread.join(timeout=5)
        
        end_time = datetime.utcnow()
        duration = (end_time - self.current_recording["start_time"]).total_seconds()
        file_size = self._get_file_size_mb(self.current_recording["file_path"])
        screenshots_count = len(self.screenshots_db.get(self.current_recording["recording_id"], []))
        
        # Save recording metadata
        recording_metadata = RecordingMetadata(
            recording_id=self.current_recording["recording_id"],
            file_path=self.current_recording["file_path"],
            duration_seconds=duration,
            frame_count=int(duration * self.current_recording["settings"]["frame_rate"]),
            resolution={"width": 1920, "height": 1080},  # Should be detected
            codec="h264",
            bitrate_kbps=self.current_recording["settings"].get("bitrate", 5000),
            created_at=self.current_recording["start_time"],
            updated_at=end_time
        )
        
        self.recordings_db[self.current_recording["recording_id"]] = recording_metadata
        
        result = {
            "success": True,
            "recording_id": self.current_recording["recording_id"],
            "message": "Recording stopped successfully",
            "end_time": end_time,
            "total_duration_seconds": duration,
            "final_file_size_mb": file_size,
            "file_path": self.current_recording["file_path"],
            "screenshots_captured": screenshots_count
        }
        
        self.current_recording = None
        return result
    
    async def start_screenshot_capture(self, recording_id: str, interval_seconds: int = 5):
        """Start capturing screenshots during recording"""
        if not self.is_recording:
            return
        
        self.is_capturing_screenshots = True
        self.screenshot_thread = threading.Thread(
            target=self._capture_screenshots,
            args=(recording_id, interval_seconds)
        )
        self.screenshot_thread.start()
    
    def _capture_screenshots(self, recording_id: str, interval_seconds: int):
        """Capture screenshots at regular intervals"""
        screenshot_count = 0
        start_time = datetime.utcnow()
        
        while self.is_capturing_screenshots and self.is_recording:
            try:
                # Capture screenshot
                screenshot = ImageGrab.grab()
                
                # Calculate timestamp relative to recording start
                current_time = datetime.utcnow()
                timestamp = (current_time - start_time).total_seconds()
                
                # Save screenshot
                screenshot_id = f"{recording_id}_{screenshot_count:04d}"
                filename = f"screenshot_{screenshot_id}.png"
                file_path = self.screenshots_dir / filename
                
                screenshot.save(file_path)
                
                # Create thumbnail
                thumbnail = screenshot.copy()
                thumbnail.thumbnail((320, 240), Image.Resampling.LANCZOS)
                thumbnail_path = self.screenshots_dir / f"thumb_{screenshot_id}.png"
                thumbnail.save(thumbnail_path)
                
                # Store screenshot data
                screenshot_data = ScreenshotData(
                    screenshot_id=screenshot_id,
                    recording_id=recording_id,
                    timestamp=timestamp,
                    file_path=str(file_path),
                    thumbnail_path=str(thumbnail_path),
                    width=screenshot.width,
                    height=screenshot.height,
                    file_size_bytes=file_path.stat().st_size,
                    captured_at=current_time
                )
                
                if recording_id not in self.screenshots_db:
                    self.screenshots_db[recording_id] = []
                
                self.screenshots_db[recording_id].append(screenshot_data)
                screenshot_count += 1
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Screenshot capture error: {str(e)}")
                time.sleep(1)
    
    def _record_screen(self, output_path: str, quality_settings: Dict, include_audio: bool, capture_mouse: bool):
        """Record screen using OpenCV (basic implementation)"""
        try:
            # Get screen dimensions
            screen = ImageGrab.grab()
            width, height = screen.size
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = quality_settings.get('frame_rate', 30)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            start_time = time.time()
            frame_count = 0
            
            while self.is_recording:
                # Capture frame
                screenshot = ImageGrab.grab()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(frame)
                frame_count += 1
                
                # Control frame rate
                elapsed = time.time() - start_time
                expected_time = frame_count / fps
                if elapsed < expected_time:
                    time.sleep(expected_time - elapsed)
            
            out.release()
            logger.info(f"Recording saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Recording error: {str(e)}")
            self.is_recording = False
    
    def _get_quality_settings(self, quality: RecordingQuality, frame_rate: int) -> Dict[str, Any]:
        """Get quality-specific encoding settings"""
        settings = {
            RecordingQuality.LOW: {"bitrate": 1000, "resolution_scale": 0.5},
            RecordingQuality.MEDIUM: {"bitrate": 3000, "resolution_scale": 0.75},
            RecordingQuality.HIGH: {"bitrate": 6000, "resolution_scale": 1.0},
            RecordingQuality.ULTRA: {"bitrate": 10000, "resolution_scale": 1.0}
        }
        
        config = settings.get(quality, settings[RecordingQuality.MEDIUM])
        config["frame_rate"] = frame_rate
        return config
    
    def _estimate_file_size_per_minute(self, quality_settings: Dict) -> float:
        """Estimate file size per minute in MB"""
        bitrate_kbps = quality_settings.get("bitrate", 3000)
        # Convert to MB per minute: (bitrate_kbps * 60 seconds) / (8 bits/byte * 1024 KB/MB)
        return (bitrate_kbps * 60) / (8 * 1024)
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            if os.path.exists(file_path):
                return os.path.getsize(file_path) / (1024 * 1024)
        except:
            pass
        return 0.0
    
    async def list_recordings(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List all recordings with metadata"""
        recordings = []
        
        for recording_id, metadata in list(self.recordings_db.items())[offset:offset + limit]:
            screenshots_count = len(self.screenshots_db.get(recording_id, []))
            
            recordings.append({
                "recording_id": recording_id,
                "file_name": Path(metadata.file_path).name,
                "file_path": metadata.file_path,
                "duration_seconds": metadata.duration_seconds,
                "file_size_mb": self._get_file_size_mb(metadata.file_path),
                "created_at": metadata.created_at,
                "quality": "medium",  # Should be stored in metadata
                "has_audio": True,    # Should be stored in metadata
                "frame_rate": metadata.bitrate_kbps // 100,  # Approximation
                "screenshots_count": screenshots_count,
                "analysis_count": 0   # Should be tracked
            })
        
        return recordings
    
    async def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording and its associated files"""
        try:
            if recording_id not in self.recordings_db:
                return False
            
            metadata = self.recordings_db[recording_id]
            
            # Delete video file
            if os.path.exists(metadata.file_path):
                os.remove(metadata.file_path)
            
            # Delete screenshots
            screenshots = self.screenshots_db.get(recording_id, [])
            for screenshot in screenshots:
                if os.path.exists(screenshot.file_path):
                    os.remove(screenshot.file_path)
                if screenshot.thumbnail_path and os.path.exists(screenshot.thumbnail_path):
                    os.remove(screenshot.thumbnail_path)
            
            # Remove from databases
            del self.recordings_db[recording_id]
            if recording_id in self.screenshots_db:
                del self.screenshots_db[recording_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete recording {recording_id}: {str(e)}")
            return False
    
    async def get_recording_info(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific recording"""
        if recording_id not in self.recordings_db:
            return None
        
        metadata = self.recordings_db[recording_id]
        screenshots_count = len(self.screenshots_db.get(recording_id, []))
        
        return {
            "recording_id": recording_id,
            "metadata": metadata.dict(),
            "screenshots_count": screenshots_count,
            "current_file_size_mb": self._get_file_size_mb(metadata.file_path),
            "exists": os.path.exists(metadata.file_path)
        }


class AIAnalysisService:
    """Service for AI-powered analysis of recordings and screenshots"""
    
    def __init__(self):
        self.analysis_jobs: Dict[str, AnalysisJob] = {}
        self.analysis_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def analyze_recording(
        self,
        recording_id: str,
        question: str,
        analysis_type: AnalysisType = AnalysisType.SPECIFIC_QUESTION,
        time_range: Optional[Dict[str, float]] = None,
        include_screenshots: bool = True
    ) -> Dict[str, Any]:
        """Analyze a recording with AI"""
        
        analysis_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Create analysis job
            job = AnalysisJob(
                job_id=analysis_id,
                recording_id=recording_id,
                question=question,
                analysis_type=analysis_type,
                status="processing",
                created_at=datetime.utcnow(),
                started_at=datetime.utcnow()
            )
            
            self.analysis_jobs[analysis_id] = job
            
            # Get recording service to access screenshots
            recording_service = ScreenRecordingService()
            screenshots = recording_service.screenshots_db.get(recording_id, [])
            
            # Filter screenshots by time range if provided
            if time_range:
                screenshots = [
                    s for s in screenshots 
                    if time_range["start"] <= s.timestamp <= time_range["end"]
                ]
            
            # Perform analysis based on type
            if analysis_type == AnalysisType.SUMMARY:
                answer = await self._generate_summary(screenshots, question)
            elif analysis_type == AnalysisType.KEY_INSIGHTS:
                answer = await self._extract_key_insights(screenshots, question)
            elif analysis_type == AnalysisType.PRESENTATION_ANALYSIS:
                answer = await self._analyze_presentation(screenshots, question)
            elif analysis_type == AnalysisType.MEETING_HIGHLIGHTS:
                answer = await self._extract_meeting_highlights(screenshots, question)
            elif analysis_type == AnalysisType.ACTION_ITEMS:
                answer = await self._extract_action_items(screenshots, question)
            else:
                answer = await self._answer_specific_question(screenshots, question)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            result = {
                "success": True,
                "analysis_id": analysis_id,
                "recording_id": recording_id,
                "question": question,
                "answer": answer,
                "confidence_score": 0.85,  # Should be calculated based on analysis
                "analysis_type": analysis_type,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.utcnow(),
                "relevant_screenshots": [
                    {
                        "screenshot_id": s.screenshot_id,
                        "timestamp": s.timestamp,
                        "thumbnail_path": s.thumbnail_path,
                        "relevance_score": 0.8  # Should be calculated
                    }
                    for s in screenshots[:5]  # Top 5 relevant screenshots
                ],
                "key_findings": self._extract_key_findings(answer),
                "time_references": self._extract_time_references(answer, screenshots),
                "metadata": {
                    "total_screenshots_analyzed": len(screenshots),
                    "analysis_method": "vision_ai",
                    "model_version": "1.0"
                }
            }
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.result = result
            
            # Store in history
            if recording_id not in self.analysis_history:
                self.analysis_history[recording_id] = []
            self.analysis_history[recording_id].append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for recording {recording_id}: {str(e)}")
            
            # Update job with error
            if analysis_id in self.analysis_jobs:
                self.analysis_jobs[analysis_id].status = "failed"
                self.analysis_jobs[analysis_id].error_message = str(e)
            
            raise Exception(f"Analysis failed: {str(e)}")
    
    async def analyze_screenshot(
        self,
        screenshot_data: bytes,
        question: str,
        context: Optional[str] = None,
        analysis_focus: AnalysisFocus = AnalysisFocus.GENERAL
    ) -> Dict[str, Any]:
        """Analyze a single screenshot"""
        
        analysis_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(screenshot_data))
            
            # Perform different types of analysis based on focus
            if analysis_focus == AnalysisFocus.TEXT_EXTRACTION:
                answer, extracted_text = await self._extract_text_from_image(image, question)
            elif analysis_focus == AnalysisFocus.UI_ELEMENTS:
                answer, detected_elements = await self._detect_ui_elements(image, question)
            elif analysis_focus == AnalysisFocus.CHARTS_GRAPHS:
                answer, chart_data = await self._analyze_charts(image, question)
            elif analysis_focus == AnalysisFocus.PRESENTATION_SLIDES:
                answer, slide_content = await self._analyze_slide_content(image, question)
            elif analysis_focus == AnalysisFocus.CODE_ANALYSIS:
                answer, code_elements = await self._analyze_code_screenshot(image, question)
            else:
                answer = await self._analyze_image_general(image, question)
                detected_elements = []
                extracted_text = None
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "analysis_id": analysis_id,
                "question": question,
                "answer": answer,
                "confidence_score": 0.80,
                "detected_elements": detected_elements if 'detected_elements' in locals() else [],
                "extracted_text": extracted_text if 'extracted_text' in locals() else None,
                "identified_objects": self._identify_objects(answer),
                "processing_time_seconds": processing_time,
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "image_size": f"{image.width}x{image.height}",
                    "analysis_focus": analysis_focus,
                    "context": context
                }
            }
            
        except Exception as e:
            logger.error(f"Screenshot analysis failed: {str(e)}")
            raise Exception(f"Screenshot analysis failed: {str(e)}")
    
    async def get_analysis_history(self, recording_id: str) -> List[Dict[str, Any]]:
        """Get all analyses performed on a recording"""
        return self.analysis_history.get(recording_id, [])
    
    async def get_analysis_progress(self, recording_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream analysis progress updates"""
        # Find active analysis job for this recording
        active_job = None
        for job in self.analysis_jobs.values():
            if job.recording_id == recording_id and job.status == "processing":
                active_job = job
                break
        
        if not active_job:
            yield {"error": "No active analysis found for this recording"}
            return
        
        # Simulate progress updates
        for progress in range(0, 101, 10):
            if active_job.status != "processing":
                break
                
            yield {
                "analysis_id": active_job.job_id,
                "progress_percentage": progress,
                "current_stage": self._get_analysis_stage(progress),
                "estimated_completion_seconds": max(0, 30 - (progress * 0.3)),
                "processed_screenshots": int(progress * 0.1),
                "total_screenshots": 10
            }
            
            await asyncio.sleep(1)
    
    # AI Analysis Methods (Mock implementations - replace with actual AI service calls)
    
    async def _generate_summary(self, screenshots: List[ScreenshotData], question: str) -> str:
        """Generate a summary of the recording"""
        # Mock implementation - replace with actual AI analysis
        return f"Based on analysis of {len(screenshots)} screenshots, this appears to be a {self._detect_content_type(screenshots)}. Key activities observed include screen interactions, application usage, and content review. The session lasted approximately {screenshots[-1].timestamp if screenshots else 0:.1f} seconds with various UI elements and content visible throughout."
    
    async def _extract_key_insights(self, screenshots: List[ScreenshotData], question: str) -> str:
        """Extract key insights from the recording"""
        return f"Key insights from the session:\n1. Primary application used: Desktop/Browser\n2. Content type: Mixed digital content\n3. User interaction patterns: Regular screen activity\n4. Duration: {len(screenshots) * 5} seconds of captured activity\n5. Visual elements: Various UI components and content areas detected"
    
    async def _analyze_presentation(self, screenshots: List[ScreenshotData], question: str) -> str:
        """Analyze presentation content"""
        return f"Presentation analysis results:\n- Detected {len(screenshots)} slide transitions/content changes\n- Content appears to include text, images, and structured information\n- Presentation flow shows logical progression\n- Visual elements suggest professional presentation format"
    
    async def _extract_meeting_highlights(self, screenshots: List[ScreenshotData], question: str) -> str:
        """Extract meeting highlights"""
        return f"Meeting highlights identified:\n- Session duration: {len(screenshots) * 5} seconds\n- Screen sharing detected with various content types\n- Multiple application windows and interfaces observed\n- Key discussion points visible through screen content changes"
    
    async def _extract_action_items(self, screenshots: List[ScreenshotData], question: str) -> str:
        """Extract action items from the recording"""
        return "Action items identified:\n1. Follow up on discussed topics\n2. Review shared documents/presentations\n3. Complete tasks mentioned during screen sharing\n4. Schedule next meeting/session"
    
    async def _answer_specific_question(self, screenshots: List[ScreenshotData], question: str) -> str:
        """Answer a specific question about the recording"""
        return f"Based on your question '{question}' and analysis of {len(screenshots)} screenshots, I can see various screen activities and content. The recording shows typical desktop/application usage with multiple visual elements and content areas. For more specific insights, please provide more targeted questions about particular aspects of the recording."
    
    async def _extract_text_from_image(self, image: Image.Image, question: str) -> tuple:
        """Extract text from screenshot using OCR"""
        # Mock OCR implementation
        extracted_text = "Sample text extracted from image - replace with actual OCR"
        answer = f"Text extraction results for your question '{question}': {extracted_text}"
        return answer, extracted_text
    
    async def _detect_ui_elements(self, image: Image.Image, question: str) -> tuple:
        """Detect UI elements in screenshot"""
        detected_elements = [
            {"type": "button", "position": {"x": 100, "y": 200}, "confidence": 0.9},
            {"type": "text_field", "position": {"x": 150, "y": 250}, "confidence": 0.8},
            {"type": "menu", "position": {"x": 50, "y": 100}, "confidence": 0.7}
        ]
        answer = f"UI elements detected: {len(detected_elements)} elements including buttons, text fields, and menus"
        return answer, detected_elements
    
    async def _analyze_charts(self, image: Image.Image, question: str) -> tuple:
        """Analyze charts and graphs in screenshot"""
        chart_data = {"type": "bar_chart", "data_points": 5, "trend": "increasing"}
        answer = f"Chart analysis: Detected bar chart with 5 data points showing an increasing trend"
        return answer, chart_data
    
    async def _analyze_slide_content(self, image: Image.Image, question: str) -> tuple:
        """Analyze presentation slide content"""
        slide_content = {"title": "Sample Slide", "bullet_points": 3, "images": 1}
        answer = f"Slide content: Title slide with 3 bullet points and 1 image"
        return answer, slide_content
    
    async def _analyze_code_screenshot(self, image: Image.Image, question: str) -> tuple:
        """Analyze code in screenshot"""
        code_elements = {"language": "python", "functions": 2, "lines": 25}
        answer = f"Code analysis: Python code with 2 functions spanning 25 lines"
        return answer, code_elements
    
    async def _analyze_image_general(self, image: Image.Image, question: str) -> str:
        """General image analysis"""
        return f"General image analysis for question '{question}': The screenshot shows a typical desktop interface with various visual elements, text content, and UI components. The image appears to be {image.width}x{image.height} pixels and contains multiple areas of interest."
    
    # Helper methods
    
    def _detect_content_type(self, screenshots: List[ScreenshotData]) -> str:
        """Detect the type of content based on screenshots"""
        if len(screenshots) > 20:
            return "presentation or extended session"
        elif len(screenshots) > 10:
            return "meeting or demonstration"
        else:
            return "brief activity or task"
    
    def _extract_key_findings(self, answer: str) -> List[str]:
        """Extract key findings from analysis answer"""
        # Simple keyword-based extraction - replace with better NLP
        findings = []
        if "presentation" in answer.lower():
            findings.append("Presentation content detected")
        if "meeting" in answer.lower():
            findings.append("Meeting activity identified")
        if "code" in answer.lower():
            findings.append("Code or development content found")
        return findings or ["General screen activity recorded"]
    
    def _extract_time_references(self, answer: str, screenshots: List[ScreenshotData]) -> List[Dict[str, float]]:
        """Extract time references from analysis"""
        time_refs = []
        for i, screenshot in enumerate(screenshots[:3]):  # First 3 as examples
            time_refs.append({
                "timestamp": screenshot.timestamp,
                "description": f"Key activity at {screenshot.timestamp:.1f}s"
            })
        return time_refs
    
    def _identify_objects(self, answer: str) -> List[str]:
        """Identify objects mentioned in the analysis"""
        # Simple keyword extraction
        objects = []
        keywords = ["button", "menu", "window", "text", "image", "chart", "graph", "slide"]
        for keyword in keywords:
            if keyword in answer.lower():
                objects.append(keyword)
        return objects or ["screen", "interface"]
    
    def _get_analysis_stage(self, progress: int) -> str:
        """Get current analysis stage based on progress"""
        if progress < 20:
            return "Loading screenshots"
        elif progress < 40:
            return "Analyzing visual content"
        elif progress < 60:
            return "Processing text and UI elements"
        elif progress < 80:
            return "Generating insights"
        else:
            return "Finalizing analysis"


# Additional utility functions
import io

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))