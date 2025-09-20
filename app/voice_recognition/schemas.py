from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ResponseFormat(str, Enum):
    """Available response formats for AI responses"""
    SUMMARY = "summary"
    KEY_INSIGHTS = "key_insights"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"

class SimplificationLevel(str, Enum):
    """Levels of response simplification"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class AudioQuality(str, Enum):
    """Audio quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

# Request Models
class VoiceSessionRequest(BaseModel):
    """Request to start a new voice session"""
    user_id: str = Field(..., description="Unique identifier for the user")
    meeting_id: Optional[str] = Field(None, description="Optional meeting identifier")
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")

class MicrophoneStatusRequest(BaseModel):
    """Request to check or toggle microphone status"""
    session_id: str = Field(..., description="Active session identifier")
    turn_on: bool = Field(..., description="Whether to turn microphone on or off")
    device_id: Optional[str] = Field(None, description="Specific device to connect to")

class DeviceSelectionRequest(BaseModel):
    """Request to select a specific audio device"""
    session_id: str = Field(..., description="Active session identifier")
    device_id: str = Field(..., description="ID of the device to select")
    device_name: Optional[str] = Field(None, description="Human-readable device name")

class AudioProcessingRequest(BaseModel):
    """Request to process audio data"""
    session_id: str = Field(..., description="Active session identifier")
    audio_data: Union[str, bytes] = Field(..., description="Base64 encoded audio data or raw bytes")
    format: Optional[str] = Field("wav", description="Audio format")
    sample_rate: Optional[int] = Field(16000, description="Audio sample rate")

class AIResponseRequest(BaseModel):
    """Request for AI response generation"""
    session_id: str = Field(..., description="Active session identifier")
    question: str = Field(..., description="Transcribed question or text")
    response_format: ResponseFormat = Field(ResponseFormat.SUMMARY, description="Desired response format")
    context: Optional[str] = Field(None, description="Additional context for the AI")
    max_length: Optional[int] = Field(500, description="Maximum response length")

class SimplifiedAnswerRequest(BaseModel):
    """Request for simplified version of AI response"""
    session_id: str = Field(..., description="Active session identifier")
    original_response: str = Field(..., description="Original AI response to simplify")
    simplification_level: SimplificationLevel = Field(SimplificationLevel.BASIC, description="Level of simplification")
    target_audience: Optional[str] = Field(None, description="Target audience for simplification")

# Response Models
class AudioDevice(BaseModel):
    """Audio device information"""
    id: str = Field(..., description="Unique device identifier")
    name: str = Field(..., description="Human-readable device name")
    is_default: bool = Field(False, description="Whether this is the default device")
    is_available: bool = Field(True, description="Whether device is currently available")
    device_type: str = Field("microphone", description="Type of audio device")

class VoiceSessionResponse(BaseModel):
    """Response for voice session creation"""
    success: bool = Field(..., description="Whether the operation was successful")
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="Human-readable status message")
    timestamp: datetime = Field(..., description="Session creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Session expiration time")

class MicrophoneStatusResponse(BaseModel):
    """Response for microphone status check"""
    success: bool = Field(..., description="Whether the check was successful")
    is_available: bool = Field(..., description="Whether microphone is available")
    is_enabled: bool = Field(..., description="Whether microphone is currently enabled")
    current_device: Optional[AudioDevice] = Field(None, description="Currently selected device")
    message: str = Field(..., description="Status message for user")
    alert_required: bool = Field(False, description="Whether user alert is needed")

class DeviceListResponse(BaseModel):
    """Response containing list of audio devices"""
    success: bool = Field(..., description="Whether the operation was successful")
    devices: List[AudioDevice] = Field(..., description="List of available audio devices")
    default_device: Optional[AudioDevice] = Field(None, description="Default audio device")
    message: str = Field(..., description="Operation status message")

class TranscriptionResponse(BaseModel):
    """Response for audio transcription"""
    success: bool = Field(..., description="Whether transcription was successful")
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Transcription confidence score")
    language: str = Field("en", description="Detected language")
    duration: float = Field(..., ge=0.0, description="Audio duration in seconds")
    message: str = Field(..., description="Operation status message")

class AudioProcessingResponse(BaseModel):
    """Response for audio processing operations"""
    success: bool = Field(..., description="Whether processing was successful")
    transcription: str = Field(..., description="Transcribed text from audio")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    audio_quality: AudioQuality = Field(..., description="Assessed audio quality")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    message: str = Field(..., description="Operation status message")

class VoiceCharacteristics(BaseModel):
    """Voice analysis characteristics"""
    pitch_average: float = Field(..., description="Average pitch in Hz")
    volume_level: float = Field(..., ge=0.0, le=1.0, description="Average volume level")
    speech_rate: float = Field(..., description="Words per minute")
    clarity_score: float = Field(..., ge=0.0, le=1.0, description="Speech clarity score")
    emotion_indicators: Dict[str, float] = Field(default_factory=dict, description="Emotional indicators")

class VoiceAnalysisResponse(BaseModel):
    """Response for voice confidence analysis"""
    success: bool = Field(..., description="Whether analysis was successful")
    confidence_rating: float = Field(..., ge=0.0, le=10.0, description="Voice confidence rating (1-10)")
    voice_characteristics: VoiceCharacteristics = Field(..., description="Detailed voice characteristics")
    situational_tips: List[str] = Field(..., description="Tips based on current situation")
    recommendations: List[str] = Field(..., description="Recommendations for improvement")
    analysis_summary: str = Field(..., description="Summary of voice analysis")

class AIResponseResponse(BaseModel):
    """Response containing AI-generated answer"""
    success: bool = Field(..., description="Whether AI response generation was successful")
    response: str = Field(..., description="Generated AI response")
    format_type: str = Field(..., description="Format of the response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="AI response confidence")
    processing_time: float = Field(..., ge=0.0, description="Response generation time")
    message: str = Field(..., description="Operation status message")
    related_topics: Optional[List[str]] = Field(None, description="Related topics for further exploration")

# Internal Models for Service Layer
class VoiceSession(BaseModel):
    """Internal model for managing voice sessions"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    is_active: bool = True
    current_device: Optional[AudioDevice] = None
    microphone_enabled: bool = False
    total_interactions: int = 0
    session_settings: Dict[str, Any] = Field(default_factory=dict)

class ProcessingResult(BaseModel):
    """Internal model for audio processing results"""
    transcription: str
    confidence: float
    audio_quality: AudioQuality
    processing_time: float
    audio_duration: float
    detected_language: str = "en"
    voice_characteristics: Optional[VoiceCharacteristics] = None

class OllamaRequest(BaseModel):
    """Request model for Ollama API"""
    model: str = Field("nous-hermes", description="Ollama model to use")
    prompt: str = Field(..., description="Prompt for the model")
    stream: bool = Field(False, description="Whether to stream the response")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional model options")

class OllamaResponse(BaseModel):
    """Response model from Ollama API"""
    response: str = Field(..., description="Generated response")
    done: bool = Field(..., description="Whether generation is complete")
    context: Optional[List[int]] = Field(None, description="Context tokens")
    total_duration: Optional[int] = Field(None, description="Total processing duration")
    load_duration: Optional[int] = Field(None, description="Model loading duration")

# Validation methods
@validator('confidence', 'confidence_score')
def validate_confidence_score(cls, v):
    """Validate confidence scores are between 0 and 1"""
    if not 0.0 <= v <= 1.0:
        raise ValueError('Confidence score must be between 0.0 and 1.0')
    return v

@validator('confidence_rating')
def validate_confidence_rating(cls, v):
    """Validate confidence rating is between 0 and 10"""
    if not 0.0 <= v <= 10.0:
        raise ValueError('Confidence rating must be between 0.0 and 10.0')
    return v

# Configuration Models
class VoiceProcessingConfig(BaseModel):
    """Configuration for voice processing services"""
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama API base URL")
    ollama_model: str = Field("nous-hermes", description="Default Ollama model")
    max_session_duration: int = Field(7200, description="Maximum session duration in seconds")
    audio_chunk_size: int = Field(1024, description="Audio processing chunk size")
    transcription_timeout: int = Field(30, description="Transcription timeout in seconds")
    max_audio_duration: int = Field(300, description="Maximum audio duration in seconds")
    supported_audio_formats: List[str] = Field(
        default=["wav", "mp3", "m4a", "ogg", "webm"],
        description="Supported audio formats"
    )
    voice_analysis_enabled: bool = Field(True, description="Whether to enable voice analysis")
    ai_response_max_tokens: int = Field(1000, description="Maximum tokens for AI responses")

# Error Models
class VoiceProcessingError(BaseModel):
    """Standard error response model"""
    success: bool = Field(False, description="Always false for errors")
    error_code: str = Field(..., description="Specific error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    AAC = "aac"

class CalibrationStatus(str, Enum):
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    FAILED = "failed"

class TranscriptionModel(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

# Request Models
class CalibrationRequest(BaseModel):
    duration: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Calibration duration in seconds"
    )
    sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description="Audio sample rate in Hz"
    )
    channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of audio channels (1=mono, 2=stereo)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "duration": 3,
                "sample_rate": 16000,
                "channels": 1
            }
        }

class AudioTestRequest(BaseModel):
    duration: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Test recording duration in seconds"
    )
    sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description="Audio sample rate in Hz"
    )
    channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of audio channels"
    )
    apply_calibration: bool = Field(
        default=True,
        description="Whether to apply calibration settings"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "duration": 5,
                "sample_rate": 16000,
                "channels": 1,
                "apply_calibration": True
            }
        }

# Response Models
class CalibrationResponse(BaseModel):
    status: CalibrationStatus
    noise_level: float = Field(description="Background noise level in dB")
    recommended_threshold: float = Field(description="Recommended voice detection threshold")
    sample_rate: int
    channels: int
    calibration_time: datetime
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Audio environment quality score (0-1)"
    )
    recommendations: List[str] = Field(
        description="List of recommendations for better audio quality"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "calibrated",
                "noise_level": -45.2,
                "recommended_threshold": -30.0,
                "sample_rate": 16000,
                "channels": 1,
                "calibration_time": "2024-01-15T10:30:00Z",
                "quality_score": 0.85,
                "recommendations": [
                    "Good audio environment detected",
                    "Consider using a microphone closer to your mouth for better clarity"
                ]
            }
        }

class AudioTestResponse(BaseModel):
    success: bool
    duration: float = Field(description="Actual recording duration")
    file_size: int = Field(description="Size of recorded audio in bytes")
    audio_quality: Dict[str, Any] = Field(description="Audio quality metrics")
    peak_amplitude: float = Field(description="Peak audio amplitude")
    average_amplitude: float = Field(description="Average audio amplitude")
    signal_to_noise_ratio: Optional[float] = Field(description="SNR if calibration available")
    recommendations: List[str]
    audio_preview_url: Optional[str] = Field(description="URL to preview the recorded audio")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "duration": 5.02,
                "file_size": 160640,
                "audio_quality": {
                    "bit_rate": 16,
                    "sample_rate": 16000,
                    "channels": 1
                },
                "peak_amplitude": 0.78,
                "average_amplitude": 0.23,
                "signal_to_noise_ratio": 15.3,
                "recommendations": [
                    "Audio quality is good for transcription",
                    "Clear voice detected"
                ],
                "audio_preview_url": "/api/v1/audio/preview/test_recording_123.wav"
            }
        }

class TranscriptionSegment(BaseModel):
    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds") 
    text: str = Field(description="Transcribed text for this segment")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this segment"
    )

class TranscriptionResponse(BaseModel):
    success: bool
    text: str = Field(description="Complete transcribed text")
    language: str = Field(description="Detected or specified language")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall transcription confidence"
    )
    duration: float = Field(description="Audio duration in seconds")
    segments: List[TranscriptionSegment] = Field(
        description="Detailed transcription segments with timestamps"
    )
    word_count: int = Field(description="Number of words transcribed")
    processing_time: float = Field(description="Time taken to process in seconds")
    model_used: str = Field(description="Transcription model used")
    audio_quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Audio quality assessment"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "text": "Hello, this is a test recording for speech to text transcription.",
                "language": "en",
                "confidence": 0.92,
                "duration": 5.02,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.5,
                        "text": "Hello, this is a test recording",
                        "confidence": 0.95
                    },
                    {
                        "start": 2.5,
                        "end": 5.02,
                        "text": "for speech to text transcription.",
                        "confidence": 0.89
                    }
                ],
                "word_count": 12,
                "processing_time": 1.23,
                "model_used": "base",
                "audio_quality_score": 0.88
            }
        }

# Error Response Model
class ErrorResponse(BaseModel):
    error: bool = True
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "error": True,
                "message": "Audio file format not supported",
                "error_code": "INVALID_FORMAT",
                "details": {
                    "supported_formats": ["wav", "mp3", "flac"],
                    "received_format": "txt"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

# Status Models
class CalibrationStatus(BaseModel):
    is_calibrated: bool
    last_calibration: Optional[datetime]
    current_settings: Optional[Dict[str, Any]]
    noise_level: Optional[float]
    quality_score: Optional[float]

class ServiceHealth(BaseModel):
    status: str = Field(description="Service health status")
    version: str = Field(description="Service version")
    uptime: float = Field(description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(description="Status of service dependencies")
    last_check: datetime = Field(default_factory=datetime.now)

# Validation helpers
class AudioValidationMixin:
    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f'Sample rate must be one of {valid_rates}')
        return v
    
    @validator('channels')
    def validate_channels(cls, v):
        if v not in [1, 2]:
            raise ValueError('Channels must be 1 (mono) or 2 (stereo)')
        return v

# Apply validation mixin to request models
CalibrationRequest.__bases__ += (AudioValidationMixin,)
AudioTestRequest.__bases__ += (AudioValidationMixin,)