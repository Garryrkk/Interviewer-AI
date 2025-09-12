from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class InsightType(str, Enum):
    DECISION = "decision"
    ACTION_ITEM = "action_item"
    KEY_POINT = "key_point"
    RISK = "risk"
    OPPORTUNITY = "opportunity"
    QUESTION = "question"

class KeyInsight(BaseModel):
    id: Optional[str] = None
    content: str = Field(..., min_length=1, max_length=500, description="The insight text")
    type: InsightType = Field(default=InsightType.KEY_POINT, description="Type of insight")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="AI confidence in this insight")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    source_section: Optional[str] = Field(default=None, description="Which part of transcript this came from")

class KeyInsightRequest(BaseModel):
    transcript: str = Field(..., min_length=10, description="Meeting transcript or text to analyze")
    meeting_id: Optional[str] = Field(default=None, description="Optional meeting identifier")
    extract_types: Optional[List[InsightType]] = Field(
        default=None, 
        description="Specific types of insights to extract. If None, extracts all types"
    )
    max_insights: Optional[int] = Field(default=10, ge=1, le=50, description="Maximum number of insights to return")

class KeyInsightResponse(BaseModel):
    insights: List[KeyInsight] = Field(description="List of extracted insights")
    total_insights: int = Field(description="Total number of insights found")
    processing_time: Optional[float] = Field(default=None, description="Time taken to process in seconds")
    meeting_id: Optional[str] = Field(default=None, description="Meeting identifier if provided")
    summary: Optional[str] = Field(default=None, description="Brief summary of the meeting")

class ErrorResponse(BaseModel):
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code for debugging")
    timestamp: datetime = Field(default_factory=datetime.utcnow)