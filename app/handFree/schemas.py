from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Enums for various states and types
class SessionStatusEnum(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    HANDS_FREE_ACTIVE = "hands_free_active"
    PAUSED = "paused"
    PROCESSING = "processing"
    ERROR = "error"

class InterviewType(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    CASE_STUDY = "case_study"
    GENERAL = "general"
    MIXED = "mixed"

class ResponseType(str, Enum):
    DETAILED = "detailed"
    CONCISE = "concise"
    KEY_INSIGHTS = "key_insights"
    STORYTELLING = "storytelling"

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXCELLENT = "excellent" 

class EmotionState(str, Enum):
    NERVOUS = "nervous"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    CALM = "calm"
    EXCITED = "excited"
    FOCUSED = "focused"

# Request Models
class HandsFreeSessionRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    default_mic_id: str = Field(..., description="Default microphone device ID")
    interview_type: InterviewType = Field(default=InterviewType.GENERAL)
    company_info: Optional[str] = Field(None, description="Company information for context")
    job_role: Optional[str] = Field(None, description="Job role being interviewed for")
    auto_start: bool = Field(True, description="Automatically start hands-free mode")

class InterviewResponseRequest(BaseModel):
    question: str = Field(..., description="Interview question to respond to")
    context: Optional[str] = Field(None, description="Additional context")
    response_type: ResponseType = Field(default=ResponseType.KEY_INSIGHTS)
    max_length: Optional[int] = Field(150, description="Maximum response length in words")

class FacialAnalysisRequest(BaseModel):
    frame_data: str = Field(..., description="Base64 encoded video frame")
    analysis_type: str = Field(default="confidence", description="Type of analysis to perform")

class SessionSettings(BaseModel):
    auto_response_enabled: bool = Field(True, description="Enable automatic response generation")
    response_delay: float = Field(2.0, description="Delay before responding in seconds")
    confidence_coaching_enabled: bool = Field(True, description="Enable real-time confidence tips")
    facial_analysis_enabled: bool = Field(True, description="Enable facial expression analysis")
    key_insights_only: bool = Field(True, description="Focus on key insights format")
    voice_feedback_enabled: bool = Field(False, description="Enable voice feedback")
    sensitivity_level: float = Field(0.7, description="Audio sensitivity level")

# Response Models
class HandsFreeSessionResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    status: SessionStatusEnum
    message: str
    mic_configured: bool = Field(..., description="Microphone successfully configured")
    ai_ready: bool = Field(..., description="AI systems initialized and ready")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class KeyInsight(BaseModel):
    point: str = Field(..., description="Key insight point")
    elaboration: Optional[str] = Field(None, description="Brief elaboration if needed")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in this insight")

class InterviewResponse(BaseModel):
    response_text: str = Field(..., description="Generated response text")
    key_insights: List[KeyInsight] = Field(..., description="Key insights in structured format")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence score")
    response_time: float = Field(..., description="Time taken to generate response")
    word_count: int = Field(..., description="Number of words in response")
    suggested_improvements: List[str] = Field(default=[], description="Suggestions for improvement")

class FacialAnalysis(BaseModel):
    confidence_score: float = Field(..., ge=0, le=1, description="Detected confidence level")
    primary_emotion: EmotionState = Field(..., description="Primary detected emotion")
    secondary_emotions: List[EmotionState] = Field(default=[], description="Secondary emotions")
    eye_contact_score: float = Field(..., ge=0, le=1, description="Eye contact quality score")
    posture_score: float = Field(..., ge=0, le=1, description="Posture quality score")
    facial_expression_score: float = Field(..., ge=0, le=1, description="Facial expression score")
    energy_level: float = Field(..., ge=0, le=1, description="Detected energy level")
    stress_indicators: List[str] = Field(default=[], description="Signs of stress or nervousness")

class ConfidenceTip(BaseModel):
    tip_type: str = Field(..., description="Type of tip (posture, expression, etc.)")
    message: str = Field(..., description="Actionable tip message")
    priority: str = Field(..., description="Priority level: high, medium, low")
    immediate_action: bool = Field(..., description="Whether this needs immediate attention")

class ConfidenceTipResponse(BaseModel):
    tips: List[ConfidenceTip] = Field(..., description="List of confidence tips")
    overall_assessment: str = Field(..., description="Overall assessment message")
    compliment: Optional[str] = Field(None, description="Compliment if performance is good")
    improvement_areas: List[str] = Field(default=[], description="Areas for improvement")

class AudioStreamResult(BaseModel):
    question_detected: bool = Field(..., description="Whether a question was detected")
    detected_question: Optional[str] = Field(None, description="The detected question text")
    context: Optional[str] = Field(None, description="Context around the question")
    is_listening: bool = Field(..., description="System is actively listening")
    is_processing: bool = Field(..., description="System is processing audio")
    audio_level: float = Field(..., ge=0, le=1, description="Current audio input level")
    speech_clarity: float = Field(..., ge=0, le=1, description="Clarity of detected speech")

class RealTimeAnalysisResponse(BaseModel):
    facial_analysis: FacialAnalysis
    confidence_tips: ConfidenceTipResponse
    overall_score: float = Field(..., ge=0, le=1, description="Combined performance score")
    timestamp: str = Field(..., description="Analysis timestamp")
    recommendations: List[str] = Field(default=[], description="Real-time recommendations")

class SessionStatus(BaseModel):
    session_id: str
    status: SessionStatusEnum
    hands_free_active: bool = Field(..., description="Hands-free mode is active")
    mic_status: str = Field(..., description="Microphone status")
    ai_systems_status: str = Field(..., description="AI systems status")
    uptime: float = Field(..., description="Session uptime in minutes")
    questions_answered: int = Field(default=0, description="Number of questions answered")
    avg_response_time: float = Field(default=0.0, description="Average response time")
    confidence_trend: List[float] = Field(default=[], description="Confidence scores over time")
    last_activity: datetime = Field(default_factory=datetime.utcnow)

class AudioStreamData(BaseModel):
    session_id: str
    audio_chunk: bytes = Field(..., description="Raw audio data chunk")
    chunk_index: int = Field(..., description="Sequence number of this chunk")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SystemHealthCheck(BaseModel):
    overall_status: str = Field(..., description="Overall system status")
    microphone_service: bool = Field(..., description="Microphone service health")
    speech_recognition_service: bool = Field(..., description="Speech recognition health")
    ai_response_service: bool = Field(..., description="AI response generation health")
    facial_analysis_service: bool = Field(..., description="Facial analysis service health")
    websocket_connections: int = Field(..., description="Active WebSocket connections")
    average_response_time: float = Field(..., description="System average response time")
    error_count: int = Field(default=0, description="Recent error count")
    last_health_check: datetime = Field(default_factory=datetime.utcnow)

class SessionInsights(BaseModel):
    session_id: str
    total_duration: float = Field(..., description="Session duration in minutes")
    questions_answered: int = Field(..., description="Total questions answered")
    average_confidence: float = Field(..., ge=0, le=1, description="Average confidence score")
    best_responses: List[str] = Field(..., description="Top performing responses")
    improvement_areas: List[str] = Field(..., description="Areas needing improvement")
    confidence_progression: List[float] = Field(..., description="Confidence over time")
    facial_analysis_summary: Dict[str, Any] = Field(..., description="Facial analysis summary")
    key_strengths: List[str] = Field(..., description="Identified strengths")
    recommended_practice_areas: List[str] = Field(..., description="Practice recommendations")
    overall_score: float = Field(..., ge=0, le=1, description="Overall session score")

class SessionSummary(BaseModel):
    session_id: str
    start_time: datetime
    end_time: datetime
    total_duration: float = Field(..., description="Duration in minutes")
    hands_free_uptime: float = Field(..., description="Time in hands-free mode")
    automation_efficiency: float = Field(..., ge=0, le=1, description="How well automation worked")
    questions_handled: int
    average_response_quality: float = Field(..., ge=0, le=1)
    confidence_improvement: float = Field(..., description="Confidence improvement during session")
    technical_performance: Dict[str, Any] = Field(..., description="Technical metrics")
    user_satisfaction_indicators: List[str] = Field(default=[], description="Satisfaction indicators")

# WebSocket Message Models
class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default={}, description="Message payload")

class AutomatedResponseMessage(WebSocketMessage):
    question: str = Field(..., description="Original question")
    response: str = Field(..., description="Generated response")
    key_insights: List[KeyInsight] = Field(..., description="Key insights")
    confidence_score: float = Field(..., ge=0, le=1)
    processing_time: float = Field(..., description="Time taken to process")

class FacialAnalysisMessage(WebSocketMessage):
    analysis: FacialAnalysis = Field(..., description="Facial analysis results")
    tips: List[ConfidenceTip] = Field(..., description="Generated tips")
    immediate_feedback: Optional[str] = Field(None, description="Immediate feedback message")

class SystemStatusMessage(WebSocketMessage):
    status: SessionStatusEnum = Field(..., description="Current system status")
    is_listening: bool = Field(..., description="Audio input active")
    is_processing: bool = Field(..., description="Processing in progress")
    hands_free_active: bool = Field(..., description="Hands-free mode status")
    error_message: Optional[str] = Field(None, description="Error message if any")