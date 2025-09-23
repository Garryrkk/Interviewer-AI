from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SummaryType(str, Enum):
    """Types of summaries that can be generated"""
    BRIEF = "brief"
    DETAILED = "detailed"
    ACTION_ITEMS = "action_items"
    KEY_POINTS = "key_points"
    FULL_TRANSCRIPT = "full_transcript"


class AnalysisType(str, Enum):
    """Types of meeting analysis"""
    REAL_TIME = "real_time"
    POST_MEETING = "post_meeting"
    CONTINUOUS = "continuous"


class AudioUploadResponse(BaseModel):
    """Response model for audio upload"""
    file_id: str = Field(..., description="Unique identifier for uploaded file")
    file_path: str = Field(..., description="Path to stored audio file")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    file_size: int = Field(..., description="File size in bytes")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="uploaded", description="Upload status")


class MeetingAnalysisRequest(BaseModel):
    """Request model for meeting analysis"""
    audio_file_path: str = Field(..., description="Path to audio file for analysis")
    meeting_context: Optional[str] = Field(None, description="Additional context about the meeting")
    analysis_type: AnalysisType = Field(default=AnalysisType.POST_MEETING, description="Type of analysis to perform")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_speakers: bool = Field(default=True, description="Identify different speakers")
    
    @validator('audio_file_path')
    def validate_audio_path(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Audio file path must be a non-empty string')
        return v


class ActionItem(BaseModel):
    """Model for action items extracted from meeting"""
    task: str = Field(..., description="Description of the action item")
    assignee: Optional[str] = Field(None, description="Person assigned to the task")
    deadline: Optional[str] = Field(None, description="Deadline for the task")
    priority: str = Field(default="medium", description="Priority level (low, medium, high)")
    status: str = Field(default="pending", description="Status of the action item")


class KeyPoint(BaseModel):
    """Model for key points from meeting"""
    point: str = Field(..., description="Key point or insight")
    category: Optional[str] = Field(None, description="Category of the key point")
    timestamp: Optional[str] = Field(None, description="Timestamp in the meeting")
    importance: str = Field(default="medium", description="Importance level")


class MeetingAnalysisResponse(BaseModel):
    """Response model for meeting analysis"""
    analysis_id: str = Field(..., description="Unique identifier for analysis")
    meeting_id: Optional[str] = Field(None, description="Associated meeting ID")
    summary: str = Field(..., description="Brief summary of the meeting")
    key_points: List[KeyPoint] = Field(default_factory=list, description="Key points from meeting")
    action_items: List[ActionItem] = Field(default_factory=list, description="Action items identified")
    sentiment_analysis: Optional[Dict[str, Any]] = Field(None, description="Overall sentiment analysis")
    speaker_insights: Optional[Dict[str, Any]] = Field(None, description="Speaker-specific insights")
    recommendations: List[str] = Field(default_factory=list, description="AI-generated recommendations")
    confidence_score: float = Field(default=0.0, description="Confidence score of analysis")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = Field(None, description="Time taken to process in seconds")


class SummarizationRequest(BaseModel):
    """Request model for creating summaries"""
    content: str = Field(..., description="Content to summarize")
    summary_type: SummaryType = Field(default=SummaryType.BRIEF, description="Type of summary to generate")
    meeting_id: Optional[str] = Field(None, description="Associated meeting ID")
    include_action_items: bool = Field(default=True, description="Include action items in summary")
    max_length: Optional[int] = Field(None, description="Maximum length of summary")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Content must be at least 10 characters long')
        return v.strip()


class SummarizationResponse(BaseModel):
    """Response model for summaries"""
    summary_id: str = Field(..., description="Unique identifier for summary")
    meeting_id: Optional[str] = Field(None, description="Associated meeting ID")
    summary_type: SummaryType = Field(..., description="Type of summary")
    summary_text: str = Field(..., description="The generated summary")
    key_points: List[str] = Field(default_factory=list, description="Key points in bullet format")
    action_items: List[ActionItem] = Field(default_factory=list, description="Extracted action items")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    participants: Optional[List[str]] = Field(None, description="Meeting participants")
    topics_discussed: List[str] = Field(default_factory=list, description="Main topics discussed")
    decisions_made: List[str] = Field(default_factory=list, description="Decisions made during meeting")
    questions_raised: List[str] = Field(default_factory=list, description="Questions raised but not answered")
    meeting_effectiveness_score: Optional[float] = Field(None, description="Score rating meeting effectiveness")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None)
    word_count: int = Field(default=0, description="Word count of original content")
    summary_ratio: Optional[float] = Field(None, description="Ratio of summary to original content")


class SummaryUpdateRequest(BaseModel):
    """Request model for updating summaries"""
    summary_text: Optional[str] = Field(None, description="Updated summary text")
    key_points: Optional[List[str]] = Field(None, description="Updated key points")
    action_items: Optional[List[ActionItem]] = Field(None, description="Updated action items")
    next_steps: Optional[List[str]] = Field(None, description="Updated next steps")


class MeetingContext(BaseModel):
    """Context information about a meeting"""
    meeting_title: Optional[str] = Field(None, description="Title of the meeting")
    meeting_type: Optional[str] = Field(None, description="Type of meeting (standup, review, etc.)")
    attendees: Optional[List[str]] = Field(None, description="List of attendees")
    agenda: Optional[List[str]] = Field(None, description="Meeting agenda items")
    duration: Optional[int] = Field(None, description="Expected or actual duration in minutes")
    meeting_date: Optional[datetime] = Field(None, description="Date and time of meeting")
    department: Optional[str] = Field(None, description="Department or team")
    project: Optional[str] = Field(None, description="Associated project")


class LLAVAAnalysisConfig(BaseModel):
    """Configuration for LLAVA model analysis"""
    model_version: str = Field(default="llava-v1.5", description="LLAVA model version")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=2048, description="Maximum tokens for response")
    include_visual_analysis: bool = Field(default=False, description="Include visual component analysis")
    audio_chunk_size: int = Field(default=30, description="Audio chunk size in seconds")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence threshold")


class RealTimeAnalysisUpdate(BaseModel):
    """Model for real-time analysis updates"""
    update_id: str = Field(..., description="Unique update identifier")
    meeting_id: str = Field(..., description="Meeting identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    current_topic: Optional[str] = Field(None, description="Current topic being discussed")
    speaker_change: Optional[str] = Field(None, description="New speaker detected")
    key_insight: Optional[str] = Field(None, description="Real-time key insight")
    action_item_detected: Optional[ActionItem] = Field(None, description="New action item detected")
    sentiment_shift: Optional[str] = Field(None, description="Change in meeting sentiment")
    engagement_level: Optional[str] = Field(None, description="Current engagement level")


class BatchSummarizationRequest(BaseModel):
    """Request for batch processing multiple meetings"""
    meeting_ids: List[str] = Field(..., description="List of meeting IDs to process")
    summary_type: SummaryType = Field(default=SummaryType.BRIEF)
    include_comparative_analysis: bool = Field(default=False, description="Include comparison across meetings")
    
    @validator('meeting_ids')
    def validate_meeting_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one meeting ID is required')
        if len(v) > 50:
            raise ValueError('Maximum 50 meetings can be processed at once')
        return v
    
# ----------------- SessionSummary -----------------
class SessionSummary(BaseModel):
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    summary_text: Optional[str]


# ----------------- WebSocketMessage -----------------
class WebSocketMessage(BaseModel):
    message_id: str
    sender_id: str
    receiver_id: str
    content: str
    timestamp: datetime
    is_read: bool = False


# ----------------- AutomatedResponseMessage -----------------
class AutomatedResponseMessage(BaseModel):
    response_id: str
    user_id: str
    query_text: str
    response_text: str
    created_at: datetime


# ----------------- FacialAnalysisMessage -----------------
class FacialAnalysisMessage(BaseModel):
    analysis_id: str
    user_id: str
    image_url: Optional[str]
    emotion_detected: Optional[str]
    confidence_score: Optional[float]
    analyzed_at: datetime


# ----------------- SystemStatusMessage -----------------
class SystemStatusMessage(BaseModel):
    status_id: str
    system_name: str
    status: str  # e.g., "online", "offline", "error"
    message: Optional[str]
    updated_at: datetime