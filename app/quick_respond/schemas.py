from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class UrgencyLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class AnalysisType(str, Enum):
    KEY_INSIGHTS = "key_insights"
    FULL_ANALYSIS = "full_analysis"
    QUICK_SUMMARY = "quick_summary"

class MeetingType(str, Enum):
    STANDUP = "standup"
    PRESENTATION = "presentation"
    BRAINSTORMING = "brainstorming"
    REVIEW = "review"
    CLIENT_MEETING = "client_meeting"
    GENERAL = "general"

class MeetingStatus(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEEDS_ATTENTION = "needs_attention"

class KeyInsight(BaseModel):
    """
    Individual key insight from meeting analysis
    """
    insight: str = Field(..., description="The key insight or observation")
    urgency: UrgencyLevel = Field(default=UrgencyLevel.MEDIUM, description="Urgency level of this insight")
    context: str = Field(default="", description="Additional context for the insight")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this insight was generated")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence score for this insight")
    
    class Config:
        use_enum_values = True

class QuickRespondRequest(BaseModel):
    """
    Request model for quick respond analysis
    """
    screenshot_data: bytes = Field(..., description="Screenshot image data")
    meeting_context: Optional[str] = Field(None, description="Additional meeting context")
    audio_transcript: Optional[str] = Field(None, description="Recent audio transcript")
    analysis_type: AnalysisType = Field(default=AnalysisType.KEY_INSIGHTS, description="Type of analysis requested")
    urgency_filter: Optional[UrgencyLevel] = Field(None, description="Filter insights by urgency level")
    
    class Config:
        use_enum_values = True

class QuickRespondResponse(BaseModel):
    """
    Response model for quick respond analysis
    """
    key_insights: List[KeyInsight] = Field(default_factory=list, description="List of key insights")
    full_analysis: str = Field(..., description="Complete analysis text")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    confidence_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Overall confidence in analysis")
    can_simplify: bool = Field(default=True, description="Whether this analysis can be simplified")
    session_id: str = Field(..., description="Unique session identifier")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        use_enum_values = True

class SimplifyRequest(BaseModel):
    """
    Request to simplify a complex analysis
    """
    original_analysis: str = Field(..., description="Original analysis text to simplify")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    simplification_level: int = Field(default=1, ge=1, le=3, description="Level of simplification (1=light, 3=very simple)")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on in simplification")

class SimplifyResponse(BaseModel):
    """
    Response with simplified analysis
    """
    simplified_text: str = Field(..., description="Simplified analysis text")
    simple_points: List[str] = Field(default_factory=list, description="Key points in bullet format")
    actions_needed: List[str] = Field(default_factory=list, description="Immediate actions required")
    meeting_status: MeetingStatus = Field(default=MeetingStatus.NEUTRAL, description="Overall meeting status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Simplification timestamp")
    original_length: int = Field(default=0, description="Length of original text")
    simplified_length: int = Field(default=0, description="Length of simplified text")
    
    class Config:
        use_enum_values = True

class MeetingContext(BaseModel):
    """
    Meeting context information
    """
    meeting_title: str = Field(..., description="Title of the meeting")
    participants: List[str] = Field(default_factory=list, description="List of meeting participants")
    agenda: Optional[str] = Field(None, description="Meeting agenda or topics")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Meeting start time")
    meeting_type: MeetingType = Field(default=MeetingType.GENERAL, description="Type of meeting")
    expected_duration_minutes: Optional[int] = Field(None, description="Expected meeting duration")
    is_recorded: bool = Field(default=False, description="Whether meeting is being recorded")
    
    class Config:
        use_enum_values = True

class ParticipantInfo(BaseModel):
    """
    Individual participant information
    """
    name: str = Field(..., description="Participant name")
    role: Optional[str] = Field(None, description="Participant role or title")
    is_presenter: bool = Field(default=False, description="Whether this person is currently presenting")
    engagement_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement level (0-1)")
    last_spoke_timestamp: Optional[datetime] = Field(None, description="When they last spoke")

class ScreenContent(BaseModel):
    """
    Analysis of screen shared content
    """
    content_type: str = Field(..., description="Type of content being shared")
    has_charts: bool = Field(default=False, description="Whether charts/graphs are visible")
    has_text: bool = Field(default=False, description="Whether text content is visible")
    has_code: bool = Field(default=False, description="Whether code is visible")
    key_elements: List[str] = Field(default_factory=list, description="Key visual elements identified")
    text_snippets: List[str] = Field(default_factory=list, description="Important text snippets extracted")

class MeetingMetrics(BaseModel):
    """
    Meeting performance metrics
    """
    participation_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Overall participation score")
    engagement_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Average engagement level")
    content_clarity: float = Field(default=0.5, ge=0.0, le=1.0, description="Clarity of shared content")
    meeting_pace: str = Field(default="normal", description="Pace of the meeting (slow/normal/fast)")
    technical_issues: int = Field(default=0, description="Number of technical issues detected")

class StreamingResponse(BaseModel):
    """
    Response for streaming analysis
    """
    type: str = Field(..., description="Type of streaming data")
    data: Dict[str, Any] = Field(..., description="Streaming data payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of this chunk")
    is_complete: bool = Field(default=False, description="Whether this is the final chunk")
    session_id: str = Field(..., description="Session identifier")

class BatchAnalysisRequest(BaseModel):
    """
    Request for batch analysis of multiple screenshots
    """
    screenshots: List[bytes] = Field(..., description="List of screenshot data")
    meeting_context: Optional[MeetingContext] = Field(None, description="Meeting context")
    time_intervals: Optional[List[datetime]] = Field(None, description="Timestamps for each screenshot")
    analysis_type: AnalysisType = Field(default=AnalysisType.KEY_INSIGHTS, description="Type of analysis")

class BatchAnalysisResponse(BaseModel):
    """
    Response for batch analysis
    """
    individual_analyses: List[QuickRespondResponse] = Field(..., description="Analysis for each screenshot")
    summary_insights: List[KeyInsight] = Field(..., description="Overall summary insights")
    meeting_progression: List[str] = Field(..., description="How the meeting progressed")
    key_moments: List[Dict[str, Any]] = Field(..., description="Key moments identified")
    overall_metrics: MeetingMetrics = Field(..., description="Overall meeting metrics")
    batch_session_id: str = Field(..., description="Batch processing session ID")

class ErrorResponse(BaseModel):
    """
    Error response model
    """
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    session_id: Optional[str] = Field(None, description="Session ID if available")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retrying")

class HealthCheckResponse(BaseModel):
    """
    Health check response
    """
    status: str = Field(..., description="Service status")
    ollama: bool = Field(..., description="Ollama service availability")
    llava_model: bool = Field(..., description="LLAVA model availability")
    llama_model: bool = Field(..., description="Llama model availability")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    version: Optional[str] = Field(None, description="Service version")

class WebSocketMessage(BaseModel):
    """
    WebSocket message format for real-time updates
    """
    message_type: str = Field(..., description="Type of WebSocket message")
    data: Dict[str, Any] = Field(..., description="Message data")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    sequence_number: Optional[int] = Field(None, description="Message sequence number")

# Configuration models
class OllamaConfig(BaseModel):
    """
    Ollama service configuration
    """
    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    llava_model: str = Field(default="llava:latest", description="LLAVA model name")
    llama_model: str = Field(default="llama2:latest", description="Llama model name")
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    stream_timeout: int = Field(default=60, description="Streaming timeout in seconds")

class QuickRespondConfig(BaseModel):
    """
    Quick respond feature configuration
    """
    max_screenshot_size_mb: float = Field(default=10.0, description="Maximum screenshot size in MB")
    max_insights_per_response: int = Field(default=5, description="Maximum insights per analysis")
    default_confidence_threshold: float = Field(default=0.6, description="Default confidence threshold")
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")
    cache_analyses: bool = Field(default=True, description="Cache analysis results")
    cache_ttl_minutes: int = Field(default=30, description="Cache TTL in minutes")

class ModelPrompts(BaseModel):
    """
    Model prompts configuration
    """
    key_insights_prompt: str = Field(
        default="""Analyze this meeting screenshot for key insights. Focus on urgent items, decisions, and action points. Format as KEY_INSIGHT, URGENCY, CONTEXT.""",
        description="Prompt for key insights analysis"
    )
    simplify_prompt: str = Field(
        default="""Simplify this meeting analysis into 3 bullet points, immediate actions, and overall status.""",
        description="Prompt for simplification"
    )
    batch_summary_prompt: str = Field(
        default="""Summarize the progression of this meeting based on multiple screenshots. Identify key moments and overall flow.""",
        description="Prompt for batch analysis summary"
    )

class APIResponse(BaseModel):
    """
    Generic API response wrapper
    """
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[ErrorResponse] = Field(None, description="Error information if request failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
class PaginatedResponse(BaseModel):
    """
    Paginated response for list endpoints
    """
    items: List[Dict[str, Any]] = Field(..., description="List of items")
    total_count: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")

# Request/Response models for advanced features
class AdvancedAnalysisRequest(BaseModel):
    """
    Advanced analysis request with more options
    """
    screenshot_data: bytes = Field(..., description="Screenshot image data")
    meeting_context: Optional[MeetingContext] = Field(None, description="Meeting context")
    audio_transcript: Optional[str] = Field(None, description="Audio transcript")
    previous_insights: Optional[List[KeyInsight]] = Field(None, description="Previous session insights")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus analysis on")
    custom_prompts: Optional[Dict[str, str]] = Field(None, description="Custom analysis prompts")
    output_format: str = Field(default="structured", description="Output format (structured/narrative/bullet)")
    include_metrics: bool = Field(default=True, description="Include performance metrics")
    sentiment_analysis: bool = Field(default=True, description="Include sentiment analysis")

class AdvancedAnalysisResponse(BaseModel):
    """
    Advanced analysis response with detailed information
    """
    key_insights: List[KeyInsight] = Field(..., description="Key insights")
    full_analysis: str = Field(..., description="Complete analysis")
    screen_content: Optional[ScreenContent] = Field(None, description="Screen content analysis")
    participants: Optional[List[ParticipantInfo]] = Field(None, description="Participant information")
    meeting_metrics: Optional[MeetingMetrics] = Field(None, description="Meeting metrics")
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Overall sentiment score")
    recommendations: List[str] = Field(default_factory=list, description="Action recommendations")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    session_id: str = Field(..., description="Session identifier")
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")

# Analytics and reporting models
class SessionAnalytics(BaseModel):
    """
    Analytics for a meeting session
    """
    session_id: str = Field(..., description="Session identifier")
    total_analyses: int = Field(default=0, description="Total number of analyses performed")
    average_confidence: float = Field(default=0.0, description="Average confidence score")
    high_urgency_insights: int = Field(default=0, description="Number of high urgency insights")
    session_duration_minutes: Optional[float] = Field(None, description="Session duration")
    most_common_insights: List[str] = Field(default_factory=list, description="Most common insight types")
    participant_engagement: Optional[Dict[str, float]] = Field(None, description="Participant engagement scores")
    technical_issues_count: int = Field(default=0, description="Number of technical issues detected")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")

class UsageMetrics(BaseModel):
    """
    System usage metrics
    """
    total_requests: int = Field(default=0, description="Total API requests")
    successful_analyses: int = Field(default=0, description="Successful analyses")
    failed_analyses: int = Field(default=0, description="Failed analyses")
    average_response_time_ms: float = Field(default=0.0, description="Average response time")
    peak_concurrent_requests: int = Field(default=0, description="Peak concurrent requests")
    model_usage: Dict[str, int] = Field(default_factory=dict, description="Usage by model")
    error_types: Dict[str, int] = Field(default_factory=dict, description="Error type counts")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")

# Webhook models for integrations
class WebhookEvent(BaseModel):
    """
    Webhook event for external integrations
    """
    event_type: str = Field(..., description="Type of event")
    session_id: str = Field(..., description="Session identifier")
    payload: Dict[str, Any] = Field(..., description="Event payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    webhook_url: str = Field(..., description="Webhook URL")
    retry_count: int = Field(default=0, description="Number of retry attempts")

class WebhookResponse(BaseModel):
    """
    Webhook response status
    """
    webhook_id: str = Field(..., description="Webhook identifier")
    status: str = Field(..., description="Delivery status")
    response_code: Optional[int] = Field(None, description="HTTP response code")
    response_body: Optional[str] = Field(None, description="Response body")
    delivery_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Delivery timestamp")
    next_retry: Optional[datetime] = Field(None, description="Next retry timestamp if failed")

# Export all models for easy importing
__all__ = [
    # Enums
    "UrgencyLevel",
    "AnalysisType", 
    "MeetingType",
    "MeetingStatus",
    
    # Core models
    "KeyInsight",
    "QuickRespondRequest",
    "QuickRespondResponse", 
    "SimplifyRequest",
    "SimplifyResponse",
    "MeetingContext",
    "ParticipantInfo",
    "ScreenContent",
    "MeetingMetrics",
    
    # Streaming and batch
    "StreamingResponse",
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    
    # Advanced features
    "AdvancedAnalysisRequest",
    "AdvancedAnalysisResponse",
    
    # Configuration
    "OllamaConfig",
    "QuickRespondConfig", 
    "ModelPrompts",
    
    # Utility models
    "ErrorResponse",
    "HealthCheckResponse",
    "WebSocketMessage",
    "APIResponse",
    "PaginatedResponse",
    
    # Analytics
    "SessionAnalytics",
    "UsageMetrics",
    
    # Webhooks
    "WebhookEvent",
    "WebhookResponse"
]