from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# ENUMS
class CameraStatus(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    CONNECTING = "connecting"

class ExpressionType(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    CONFUSED = "confused"
    ANGRY = "angry"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

class MessageSender(str, Enum):
    USER = "user"
    AI = "ai"
    SYSTEM = "system"

class CameraResolution(str, Enum):
    LOW = "640x480"
    MEDIUM = "1280x720"
    HIGH = "1920x1080"

# CAMERA MODELS
class CameraDevice(BaseModel):
    device_id: str = Field(..., description="Unique identifier for the camera device")
    name: str = Field(..., description="Human-readable name of the camera")
    is_available: bool = Field(..., description="Whether the camera is available for use")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Camera capabilities")
    resolution_options: List[CameraResolution] = Field(default_factory=list)
    max_fps: int = Field(default=30, description="Maximum supported FPS")

class CameraSessionRequest(BaseModel):
    device_id: str = Field(..., description="ID of the camera device to use")
    resolution: CameraResolution = Field(default=CameraResolution.MEDIUM, description="Desired resolution")
    fps: int = Field(default=30, ge=1, le=60, description="Desired frames per second")
    auto_start_monitoring: bool = Field(default=True, description="Start expression monitoring automatically")

class CameraSessionResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    device_id: str = Field(..., description="Camera device ID")
    status: CameraStatus = Field(..., description="Current session status")
    resolution: CameraResolution = Field(..., description="Active resolution")
    fps: int = Field(..., description="Active FPS")
    created_at: datetime = Field(..., description="Session creation timestamp")
    stream_url: Optional[str] = Field(None, description="URL for video stream")

class CameraStatusResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    status: CameraStatus = Field(..., description="Current camera status")
    device_id: str = Field(..., description="Camera device ID")
    is_streaming: bool = Field(..., description="Whether camera is actively streaming")
    resolution: CameraResolution = Field(..., description="Current resolution")
    fps: int = Field(..., description="Current FPS")
    uptime_seconds: float = Field(..., description="Session uptime in seconds")
    last_frame_timestamp: Optional[datetime] = Field(None, description="Timestamp of last captured frame")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")

class CameraTestResult(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    success: bool = Field(..., description="Whether test was successful")
    frame_captured: bool = Field(..., description="Whether a test frame was captured")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    resolution_actual: str = Field(..., description="Actual resolution captured")
    error_details: Optional[str] = Field(None, description="Error details if test failed")

# EXPRESSION DETECTION MODELS
class ExpressionDetectionRequest(BaseModel):
    session_id: str = Field(..., description="Camera session ID")
    frame_data: Optional[str] = Field(None, description="Base64 encoded frame data (optional if using session)")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for detection")
    detect_multiple_faces: bool = Field(default=False, description="Detect expressions for multiple faces")

class FaceExpression(BaseModel):
    expression: ExpressionType = Field(..., description="Detected expression type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    face_id: Optional[str] = Field(None, description="Unique identifier for the face")
    bounding_box: Dict[str, float] = Field(default_factory=dict, description="Face bounding box coordinates")

class ExpressionDetectionResponse(BaseModel):
    session_id: str = Field(..., description="Camera session ID")
    timestamp: datetime = Field(..., description="Detection timestamp")
    faces_detected: int = Field(..., description="Number of faces detected")
    primary_expression: ExpressionType = Field(..., description="Most confident expression")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of primary expression")
    all_expressions: List[FaceExpression] = Field(default_factory=list, description="All detected expressions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    frame_quality: float = Field(..., ge=0.0, le=1.0, description="Quality score of the analyzed frame")

class ExpressionMonitoringConfig(BaseModel):
    session_id: str = Field(..., description="Camera session ID")
    interval_seconds: int = Field(default=2, ge=1, le=60, description="Detection interval in seconds")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    confusion_trigger_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    auto_simplify: bool = Field(default=True, description="Automatically simplify AI messages on confusion")

# CHAT MODELS
class ChatMessage(BaseModel):
    id: Optional[str] = Field(None, description="Unique message identifier")
    session_id: str = Field(..., description="Session identifier")
    sender: MessageSender = Field(..., description="Message sender type")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    is_simplified: bool = Field(default=False, description="Whether this is a simplified version")
    original_message_id: Optional[str] = Field(None, description="ID of original message if simplified")
    simplification_reason: Optional[str] = Field(None, description="Reason for simplification")

class SimplifyMessageRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    original_message_id: str = Field(..., description="ID of message to simplify")
    confusion_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of confusion detection")
    simplification_level: int = Field(default=1, ge=1, le=3, description="Level of simplification (1=mild, 3=maximum)")

class ChatSessionSummary(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    total_messages: int = Field(..., description="Total number of messages")
    ai_messages: int = Field(..., description="Number of AI messages")
    user_messages: int = Field(..., description="Number of user messages")
    system_messages: int = Field(..., description="Number of system messages")
    simplified_messages: int = Field(..., description="Number of simplified messages")
    confusion_triggers: int = Field(..., description="Number of times confusion triggered simplification")
    session_duration_minutes: float = Field(..., description="Session duration in minutes")
    last_activity: datetime = Field(..., description="Last message timestamp")

# MONITORING MODELS
class MonitoringSession(BaseModel):
    monitoring_id: str = Field(..., description="Unique monitoring session ID")
    camera_session_id: str = Field(..., description="Associated camera session ID")
    status: str = Field(..., description="Monitoring status")
    config: ExpressionMonitoringConfig = Field(..., description="Monitoring configuration")
    created_at: datetime = Field(..., description="Monitoring session creation time")
    last_detection: Optional[datetime] = Field(None, description="Last successful detection timestamp")
    total_detections: int = Field(default=0, description="Total number of detections performed")
    confusion_events: int = Field(default=0, description="Number of confusion events detected")

# SYSTEM MODELS
class SystemHealthStatus(BaseModel):
    service_name: str = Field(..., description="Name of the service")
    status: str = Field(..., description="Health status")
    last_check: datetime = Field(..., description="Last health check timestamp")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    active_sessions: int = Field(default=0, description="Number of active sessions")
    error_count: int = Field(default=0, description="Number of recent errors")

class APIResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    error_code: Optional[str] = Field(None, description="Error code if applicable")

# WEBSOCKET MODELS
class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type")
    session_id: str = Field(..., description="Session identifier")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")

class CameraFrameMessage(WebSocketMessage):
    type: str = Field(default="camera_frame", description="Message type")
    frame_data: str = Field(..., description="Base64 encoded frame data")
    frame_index: int = Field(..., description="Frame sequence number")

class ExpressionDetectedMessage(WebSocketMessage):
    type: str = Field(default="expression_detected", description="Message type")
    expression: ExpressionType = Field(..., description="Detected expression")
    confidence: float = Field(..., description="Detection confidence")
    auto_simplified: bool = Field(default=False, description="Whether auto-simplification was triggered")

# CONFIGURATION MODELS
class CameraConfig(BaseModel):
    max_concurrent_sessions: int = Field(default=10, description="Maximum concurrent camera sessions")
    default_resolution: CameraResolution = Field(default=CameraResolution.MEDIUM)
    default_fps: int = Field(default=30)
    session_timeout_minutes: int = Field(default=60, description="Session timeout in minutes")
    frame_buffer_size: int = Field(default=10, description="Number of frames to buffer")
    
class ExpressionConfig(BaseModel):
    model_path: str = Field(..., description="Path to expression detection model")
    confidence_threshold: float = Field(default=0.7)
    processing_timeout_seconds: int = Field(default=5)
    max_faces_per_frame: int = Field(default=5)
    enable_face_tracking: bool = Field(default=True)

class ChatConfig(BaseModel):
    max_messages_per_session: int = Field(default=1000)
    message_retention_days: int = Field(default=30)
    auto_simplification_enabled: bool = Field(default=True)
    simplification_api_endpoint: str = Field(..., description="Endpoint for AI simplification service")

class RecordingQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class AnalysisType(str, Enum):
    SUMMARY = "summary"
    KEY_INSIGHTS = "key_insights"
    SPECIFIC_QUESTION = "specific_question"
    PRESENTATION_ANALYSIS = "presentation_analysis"
    MEETING_HIGHLIGHTS = "meeting_highlights"
    ACTION_ITEMS = "action_items"


class RecordingStatus(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    PROCESSING = "processing"
    ERROR = "error"


class ConnectionStatus(str, Enum):
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"


class AnalysisFocus(str, Enum):
    GENERAL = "general"
    TEXT_EXTRACTION = "text_extraction"
    UI_ELEMENTS = "ui_elements"
    CHARTS_GRAPHS = "charts_graphs"
    PRESENTATION_SLIDES = "presentation_slides"
    CODE_ANALYSIS = "code_analysis"


# Request Models
class StartRecordingRequest(BaseModel):
    quality: RecordingQuality = RecordingQuality.MEDIUM
    include_audio: bool = True
    capture_mouse: bool = True
    frame_rate: int = Field(default=30, ge=15, le=60)
    enable_screenshot_analysis: bool = True
    screenshot_interval: int = Field(default=5, ge=1, le=30)  # seconds
    
    @validator('frame_rate')
    def validate_frame_rate(cls, v):
        if v not in [15, 24, 30, 60]:
            raise ValueError('Frame rate must be 15, 24, 30, or 60')
        return v


class AnalysisRequest(BaseModel):
    recording_id: str
    question: str = Field(..., min_length=1, max_length=500)
    analysis_type: AnalysisType = AnalysisType.SPECIFIC_QUESTION
    time_range: Optional[Dict[str, float]] = None  # {"start": 0, "end": 300}
    include_screenshots: bool = True
    language: str = "en"
    
    @validator('time_range')
    def validate_time_range(cls, v):
        if v is not None:
            if 'start' not in v or 'end' not in v:
                raise ValueError('time_range must contain start and end keys')
            if v['start'] < 0 or v['end'] < 0:
                raise ValueError('time_range values must be non-negative')
            if v['start'] >= v['end']:
                raise ValueError('start time must be less than end time')
        return v


class ScreenshotAnalysisRequest(BaseModel):
    screenshot_data: bytes = Field(..., description="Base64 encoded screenshot data")
    question: str = Field(..., min_length=1, max_length=500)
    context: Optional[str] = None
    analysis_focus: AnalysisFocus = AnalysisFocus.GENERAL


# Response Models
class HealthCheckResponse(BaseModel):
    status: ConnectionStatus
    message: str
    timestamp: datetime
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None


class PermissionStatusResponse(BaseModel):
    has_permission: bool
    permission_type: str
    message: str
    needs_user_action: bool
    os_type: Optional[str] = None


class RecordingStatusResponse(BaseModel):
    status: RecordingStatus
    recording_id: Optional[str] = None
    duration_seconds: Optional[float] = None
    file_size_mb: Optional[float] = None
    start_time: Optional[datetime] = None
    is_paused: bool = False
    error_message: Optional[str] = None
    screenshots_captured: int = 0
    quality_settings: Optional[Dict[str, Any]] = None


class StartRecordingResponse(BaseModel):
    success: bool
    recording_id: str
    message: str
    start_time: datetime
    estimated_file_size_mb_per_minute: float
    settings: Dict[str, Any]


class StopRecordingResponse(BaseModel):
    success: bool
    recording_id: str
    message: str
    end_time: datetime
    total_duration_seconds: float
    final_file_size_mb: float
    file_path: str
    screenshots_captured: int


class RecordingInfo(BaseModel):
    recording_id: str
    file_name: str
    file_path: str
    duration_seconds: float
    file_size_mb: float
    created_at: datetime
    quality: RecordingQuality
    has_audio: bool
    frame_rate: int
    screenshots_count: int = 0
    thumbnail_path: Optional[str] = None
    analysis_count: int = 0


class AnalysisResponse(BaseModel):
    success: bool
    analysis_id: str
    recording_id: str
    question: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    analysis_type: AnalysisType
    processing_time_seconds: float
    timestamp: datetime
    relevant_screenshots: List[Dict[str, Any]] = []
    key_findings: List[str] = []
    time_references: List[Dict[str, float]] = []  # [{"timestamp": 45.2, "description": "Key moment"}]
    metadata: Dict[str, Any] = {}


class ScreenshotAnalysisResponse(BaseModel):
    success: bool
    analysis_id: str
    question: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    detected_elements: List[Dict[str, Any]] = []
    extracted_text: Optional[str] = None
    identified_objects: List[str] = []
    processing_time_seconds: float
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class AnalysisProgress(BaseModel):
    analysis_id: str
    progress_percentage: float = Field(ge=0.0, le=100.0)
    current_stage: str
    estimated_completion_seconds: Optional[float] = None
    processed_screenshots: int = 0
    total_screenshots: int = 0


class ErrorResponse(BaseModel):
    error: bool = True
    error_code: str
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


# Internal Models
class ScreenshotData(BaseModel):
    screenshot_id: str
    recording_id: str
    timestamp: float
    file_path: str
    thumbnail_path: Optional[str] = None
    width: int
    height: int
    file_size_bytes: int
    captured_at: datetime
    analysis_results: List[Dict[str, Any]] = []


class RecordingMetadata(BaseModel):
    recording_id: str
    file_path: str
    duration_seconds: float
    frame_count: int
    resolution: Dict[str, int]  # {"width": 1920, "height": 1080}
    codec: str
    bitrate_kbps: int
    created_at: datetime
    updated_at: datetime
    tags: List[str] = []


class AnalysisJob(BaseModel):
    job_id: str
    recording_id: str
    question: str
    analysis_type: AnalysisType
    status: Literal["queued", "processing", "completed", "failed"]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    result: Optional[Dict[str, Any]] = None


# Configuration Models
class RecordingConfig(BaseModel):
    max_recording_duration_minutes: int = 120
    max_file_size_mb: int = 1000
    supported_formats: List[str] = ["mp4", "webm", "mov"]
    screenshot_formats: List[str] = ["png", "jpg"]
    max_screenshots_per_recording: int = 1000
    cleanup_after_days: int = 30


class AIConfig(BaseModel):
    model_name: str = "gpt-4-vision-preview"
    max_tokens: int = 4000
    temperature: float = 0.3
    max_screenshot_analysis_batch: int = 10
    analysis_timeout_seconds: int = 300
    supported_languages: List[str] = ["en", "es", "fr", "de", "ja", "zh"]


# WebSocket Message Models
class WSMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatusUpdate(WSMessage):
    type: Literal["status_update"] = "status_update"


class AnalysisUpdate(WSMessage):
    type: Literal["analysis_progress"] = "analysis_progress"


class ErrorUpdate(WSMessage):
    type: Literal["error"] = "error"


# Batch Processing Models
class BatchAnalysisRequest(BaseModel):
    recording_ids: List[str]
    questions: List[str]
    analysis_type: AnalysisType = AnalysisType.SUMMARY
    priority: Literal["low", "normal", "high"] = "normal"


class BatchAnalysisResponse(BaseModel):
    batch_id: str
    total_jobs: int
    estimated_completion_minutes: float
    created_at: datetime
    status: Literal["queued", "processing", "completed", "failed"]