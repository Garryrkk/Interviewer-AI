from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone


class HandsFreeRequest(BaseModel):
    """Request model for hands-free mode"""
    transcript: str = Field(
        description="The spoken text from speech recognition",
        min_length=1,
        max_length=2000,
        examples=["What is machine learning?"]
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context like 'interview' or 'casual'"
    )
    simplify: Optional[bool] = Field(
        default=False,
        description="If true, reply will be simplified"
    )


class HandsFreeResponse(BaseModel):
    """Response model for hands-free mode"""
    response: str = Field(
        description="AI's generated reply",
        examples=["Machine learning is when computers learn patterns from data."]
    )
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Optional follow-up suggestions"
    )
    source: str = Field(
        default="handsfree-service",
        description="Source used for generation"
    )
    success: bool = Field(
        default=True,
        description="Indicates success"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp"
    )


class HandsFreeError(BaseModel):
    """Error model for hands-free mode"""
    error: str = Field(description="Error message")
    error_code: str = Field(description="Short error code")
    success: bool = Field(default=False)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
