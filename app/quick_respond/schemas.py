from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone


class QuickResponseRequest(BaseModel):
    """Request model for quick response generation"""
    prompt: str = Field(
        description="The prompt/text to generate a response for",
        min_length=1, 
        max_length=4000,
        examples=["Tell me about artificial intelligence"]
    )
    simplify: Optional[bool] = Field(
        default=False, 
        description="If true, produce a simpler/lower-reading-level reply"
    )
    max_tokens: Optional[int] = Field(
        default=80, 
        description="Optional max tokens / length hint for the model"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "What is the capital of France?",
                "simplify": False,
                "max_tokens": 80
            }
        }
    }


class QuickResponse(BaseModel):
    """Response model for quick response generation"""
    response: str = Field(
        description="The generated response",
        examples=["The capital of France is Paris."]
    )
    suggestions: Optional[List[str]] = Field(
        default=None, 
        description="Optional short tips or response variants"
    )
    source: str = Field(
        default="openai", 
        description="Source used for generation (mock | openai | custom)"
    )
    success: bool = Field(
        default=True,
        description="Indicates if the request was successful"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp (UTC, timezone-aware)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "The capital of France is Paris, a beautiful city known for its art, culture, and history.",
                "suggestions": ["Tell me more about Paris", "What are other French cities?"],
                "source": "openai",
                "success": True,
                "timestamp": "2025-09-11T12:00:00Z"
            }
        }
    }


class QuickResponseError(BaseModel):
    """Error response model for quick response generation"""
    error: str = Field(
        description="Error message describing what went wrong"
    )
    error_code: str = Field(
        description="Error code for frontend handling (e.g., 'INVALID_INPUT', 'API_ERROR')"
    )
    success: bool = Field(
        default=False, 
        description="Always false for error responses"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp (UTC, timezone-aware)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Prompt exceeds maximum length limit",
                "error_code": "INVALID_INPUT",
                "success": False,
                "timestamp": "2025-09-11T12:00:00Z"
            }
        }
    }


# Legacy aliases for backward compatibility
QuickRequest = QuickResponseRequest  # Alias for the original QuickRequest