from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class HideModeEnum(str, Enum):
    MINIMIZE = "minimize"
    HIDE_WINDOW = "hide_window"
    BACKGROUND_TAB = "background_tab"
    SEPARATE_DISPLAY = "separate_display"

class RecordingTypeEnum(str, Enum):
    SCREEN = "screen"
    VOICE = "voice" 
    COMBINED = "combined"
    NOTES_ONLY = "notes_only"

class UIComponentEnum(str, Enum):
    MAIN_WINDOW = "main_window"
    RECORDING_INDICATOR = "recording_indicator"
    AI_INSIGHTS_PANEL = "ai_insights_panel"
    NOTES_SECTION = "notes_section"
    CONTROLS_BAR = "controls_bar"
    STATUS_INDICATORS = "status_indicators"
    ALL_COMPONENTS = "all_components"

class InsightTypeEnum(str, Enum):
    CONVERSATION_ANALYSIS = "conversation_analysis"
    SENTIMENT_TRACKING = "sentiment_tracking"
    KEY_MOMENTS = "key_moments"
    PERFORMANCE_METRICS = "performance_metrics"
    AUTO_SUMMARY = "auto_summary"
    QUESTION_ANALYSIS = "question_analysis"

# Request Models
class RecordingConfig(BaseModel):
    screen_recording: bool = Field(True, description="Enable screen recording")
    voice_recording: bool = Field(True, description="Enable voice recording")
    auto_notes: bool = Field(True, description="Enable automatic note-taking")
    real_time_insights: bool = Field(False, description="Generate insights in real-time")
    recording_quality: str = Field("medium", description="Recording quality: low, medium, high")
    audio_format: str = Field("mp3", description="Audio format for voice recording")
    video_format: str = Field("mp4", description="Video format for screen recording")
    max_duration: Optional[int] = Field(None, description="Maximum recording duration in minutes")

class UIConfig(BaseModel):
    hide_mode: HideModeEnum = Field(HideModeEnum.MINIMIZE, description="How to hide the UI")
    components_to_hide: List[UIComponentEnum] = Field(default_factory=list)
    keep_separate_window: bool = Field(False, description="Keep AI in separate window")
    minimize_to_tray: bool = Field(True, description="Minimize to system tray")
    show_discrete_indicator: bool = Field(False, description="Show small discrete recording indicator")

class SecurityConfig(BaseModel):
    local_processing_only: bool = Field(True, description="Process all data locally")
    encrypt_data: bool = Field(True, description="Encrypt captured data")
    auto_delete_after: Optional[int] = Field(None, description="Auto-delete data after N hours")
    no_cloud_upload: bool = Field(True, description="Prevent any cloud uploads")
    secure_storage_path: Optional[str] = Field(None, description="Custom secure storage location")

class InvisibilityModeRequest(BaseModel):
    recording_config: RecordingConfig
    ui_config: UIConfig
    security_config: SecurityConfig
    session_name: Optional[str] = Field(None, description="Optional session name")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class RecordingSessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Existing session ID or create new")
    screen_recording: bool = Field(True)
    voice_recording: bool = Field(True)
    auto_notes: bool = Field(True)
    real_time_insights: bool = Field(False)
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")

class UIVisibilityRequest(BaseModel):
    session_id: str
    components_to_hide: Optional[List[UIComponentEnum]] = Field(default_factory=list)
    components_to_show: Optional[List[UIComponentEnum]] = Field(default_factory=list)
    hide_mode: Optional[HideModeEnum] = Field(None)

class InsightGenerationRequest(BaseModel):
    session_id: str
    insight_types: List[InsightTypeEnum]
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    priority: str = Field("normal", description="Processing priority: low, normal, high")

# Response Models
class InvisibilityModeResponse(BaseModel):
    success: bool
    session_id: str
    message: str
    ui_state: Optional[Dict[str, Any]] = Field(None)
    recording_state: Optional[Dict[str, Any]] = Field(None)
    security_status: Optional[Dict[str, Any]] = Field(None)
    final_insights_url: Optional[str] = Field(None)

class RecordingSessionResponse(BaseModel):
    success: bool
    session_id: str
    recording_started: bool
    message: str
    recording_config: Optional[Dict[str, Any]] = Field(None)
    estimated_duration: Optional[int] = Field(None)
    recording_duration: Optional[int] = Field(None)
    data_size: Optional[str] = Field(None)
    processing_status: Optional[str] = Field(None)

class UIVisibilityResponse(BaseModel):
    success: bool
    session_id: str
    hidden_components: Optional[List[str]] = Field(None)
    visible_components: Optional[List[str]] = Field(None)
    ui_state: str
    message: str

class SessionStatusResponse(BaseModel):
    session_id: str
    is_active: bool
    invisibility_enabled: bool
    recording_status: Optional[str] = Field(None)
    ui_state: Optional[str] = Field(None)
    start_time: Optional[datetime] = Field(None)
    duration: Optional[int] = Field(None)
    data_captured: Dict[str, Any] = Field(default_factory=dict)
    security_status: Optional[Dict[str, Any]] = Field(None)

class InsightGenerationResponse(BaseModel):
    success: bool
    session_id: str
    generation_started: bool
    message: str
    estimated_completion_time: Optional[str] = Field(None)
    insight_types: List[InsightTypeEnum]

class SecurityStatusResponse(BaseModel):
    session_id: str
    data_encrypted: bool
    local_processing: bool
    no_external_leaks: bool
    secure_storage: bool
    privacy_compliant: bool
    security_score: int = Field(ge=0, le=100, description="Security score out of 100")

# Internal Data Models
class SessionData(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    invisibility_enabled: bool
    recording_config: RecordingConfig
    ui_config: UIConfig
    security_config: SecurityConfig
    current_state: Dict[str, Any] = Field(default_factory=dict)
    captured_data: Dict[str, Any] = Field(default_factory=dict)
    insights_generated: Dict[str, Any] = Field(default_factory=dict)

class RecordingData(BaseModel):
    session_id: str
    recording_id: str
    recording_type: RecordingTypeEnum
    start_time: datetime
    end_time: Optional[datetime] = Field(None)
    file_path: Optional[str] = Field(None)
    file_size: Optional[int] = Field(None)
    duration: Optional[int] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_status: str = Field("pending")

class UIState(BaseModel):
    session_id: str
    is_hidden: bool
    hidden_components: List[UIComponentEnum] = Field(default_factory=list)
    hide_mode: Optional[HideModeEnum] = Field(None)
    window_positions: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime

class SecurityStatus(BaseModel):
    session_id: str
    encryption_enabled: bool
    local_processing_only: bool
    data_isolation: bool
    no_network_leaks: bool
    secure_deletion: bool
    compliance_status: Dict[str, bool] = Field(default_factory=dict)
    last_security_check: datetime

class InsightData(BaseModel):
    session_id: str
    insight_id: str
    insight_type: InsightTypeEnum
    generated_at: datetime
    content: Dict[str, Any]
    confidence_score: float = Field(ge=0, le=1)
    processing_time: Optional[float] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Error Models
class InvisibilityError(BaseModel):
    error_code: str
    message: str
    session_id: Optional[str] = Field(None)
    details: Optional[Dict[str, Any]] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Configuration Models
class SystemConfig(BaseModel):
    max_concurrent_sessions: int = Field(5)
    max_recording_duration: int = Field(180)  # minutes
    default_storage_path: str
    encryption_key_rotation: int = Field(24)  # hours
    cleanup_interval: int = Field(1)  # hours
    security_audit_interval: int = Field(6)  # hours

class PerformanceMetrics(BaseModel):
    session_id: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_activity: Dict[str, float] = Field(default_factory=dict)
    recording_quality_score: Optional[float] = Field(None)
    processing_latency: Optional[float] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)