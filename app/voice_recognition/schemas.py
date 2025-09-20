from pydantic import BaseModel, Field
from typing import Optional, List, Union
from datetime import datetime
from pydantic import BaseModel

class TranscriptionResponse(BaseModel):
    transcript: str


class PreopConfig(BaseModel):
    device: str = Field(..., description="Audio input device identifier")
    noise_cancellation: bool = Field(default=False, description="Enable noise cancelation")
    sample_rate: Optional[int] = Field(default=16000, description="Audio sample rate in Hz")
    language: Optional[str] = Field(default="en-US", description="Recognition language")

class PreopResponse(BaseModel):
    status: str = Field(..., description="Configuration status")
    message: str = Field(..., description="Status message")
    config_validated: bool = Field(..., description="Whether config is valid")
    ai_service_ready: bool = Field(..., description="Whether AI service is ready")

class AudioChunk(BaseModel):
    data: str = Field(..., description="Base64 encoded audio data")
    timestamp: float = Field(..., description="Timestamp of audio chunk")
    sequence_id: Optional[int] = Field(default=None, description="Sequence number for ordering")
    chunk_size: Optional[int] = Field(default=None, description="Size of audio chunk in bytes")

class ProcessingResponse(BaseModel):
    partial_transcript: str = Field(..., description="Partial transcription")
    is_final: bool = Field(default=False, description="Whether this is final transcript")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    timestamp: float = Field(..., description="Processing timestamp")  

class RecognitionSession(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    device: str = Field(..., description="Audio input device")
    noise_cancellation: bool = Field(default=False)
    language: str = Field(default="en-US")
    enable_ai_insights: bool = Field(default=True, description="Enable AI-powered insights")
    max_duration: Optional[int] = Field(default=3600, description="Max session duration in seconds")

class Insight(BaseModel):
    """AI-generated insight from transcript"""
    type: str = Field(..., description="Type of insight (summary, action_item, key_point)")
    content: str = Field(..., description="Insight content")
    confidence: float = Field(..., description="Confidence score")
    timestamp: Optional[float] = Field(default=None, description="Related timestamp in audio")
   
class RecognitionResponse(BaseModel):
    """Complete voice recognition response"""
    session_id: str = Field(..., description="Session identifier")
    transcript: str = Field(..., description="Complete transcript")
    confidence: float = Field(..., description="Overall confidence score")
    duration: float = Field(..., description="Audio duration in seconds")
    insights: Optional[List[Insight]] = Field(default=None, description="AI-generated insights")
    language_detected: Optional[str] = Field(default=None, description="Detected language")
    speaker_count: Optional[int] = Field(default=None, description="Number of speakers detected")
    created_at: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class SessionControl(BaseModel):
    """Session control commands"""
    action: str = Field(..., description="Action: start, stop, pause, resume")
    session_id: Optional[str] = Field(default=None, description="Session ID for control")

class SessionStatus(BaseModel):
    """Current session status"""
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Session status: active, paused, stopped")
    duration: float = Field(..., description="Current session duration")
    transcript_length: int = Field(..., description="Current transcript character count")
    last_activity: datetime = Field(..., description="Last activity timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")                         