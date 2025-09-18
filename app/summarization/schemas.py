from pydantic import BaseModel, Field  
from typing import Optional
from datetime import datetime, timezone


class SummarizeRequest(BaseModel):
    """Request model for summarization"""
    text: str = Field(
        description="The input text to summarize",
        min_length=1,
        max_length=8000,
        examples=["Artificial intelligence is the simulation of human intelligence..."]
    )
    style: Optional[str] = Field(
        default="concise",
        description="Style of summarization: concise | detailed | bullet"
    )
    simplify: Optional[bool] = Field(
        default=False,
        description="If true, summary is made simpler and easier to read"
    )


class SummarizeResponse(BaseModel):
    """Response model for summarization"""
    summary: str = Field(
        description="The generated summary",
        examples=["AI is when machines simulate human intelligence, learning, and decision making."]
    )
    source: str = Field(
        default="summarization-service",
        description="Source of generation"
    )
    success: bool = Field(
        default=True,
        description="Indicates success"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp"
    )


class SummarizeError(BaseModel):
    """Error response for summarization"""
    error: str = Field(description="Error message")
    error_code: str = Field(description="Short error code (e.g., INVALID_INPUT)")
    success: bool = Field(default=False)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
