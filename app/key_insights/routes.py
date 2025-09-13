from fastapi import APIRouter, HTTPException, Body, Query, status
from fastapi.responses import JSONResponse
from typing import Optional, List
import logging
from datetime import datetime

from .services import insights_service
from .schemas import (
    KeyInsightRequest, 
    KeyInsightResponse, 
    ErrorResponse,
    InsightType,
    KeyInsight
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/insights",
    tags=["Key Insights"],
    responses={404: {"description": "Not found"}}
)

@router.post(
    "/extract",
    response_model=KeyInsightResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Key Insights",
    description="Extract key insights from meeting transcript or text content"
)
async def extract_insights(request: KeyInsightRequest = Body(...)):
    try:
        logger.info(f"Processing insight extraction request for meeting: {request.meeting_id}")
        
        if len(request.transcript.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transcript is too short. Minimum 10 characters required."
            )
        result = insights_service.extract_key_insights(request)
        
        response = KeyInsightResponse(**result)
        
        logger.info(f"Successfully extracted {response.total_insights} insights")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during insight extraction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during insight extraction"
        )

@router.get(
    "/types",
    response_model=List[str],
    summary="Get Available Insight Types",
    description="Get list of all available insight types"
)
async def get_insight_types():
    try:
        return [insight_type.value for insight_type in InsightType]
    except Exception as e:
        logger.error(f"Error getting insight types: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving insight types"
        )

@router.post(
    "/extract/simple",
    response_model=List[str],
    summary="Simple Insight Extraction",
    description="Simple endpoint that returns just the insight text (for backward compatibility)"
)
async def extract_insights_simple(
    transcript: str = Body(..., embed=True),
    max_insights: Optional[int] = Query(default=10, ge=1, le=50)
):
    try:
        if len(transcript.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transcript is too short. Minimum 10 characters required."
            )
        
        request = KeyInsightRequest(
            transcript=transcript,
            max_insights=max_insights
        )
        
        result = insights_service.extract_key_insights(request)
        
        return [insight.content for insight in result["insights"]]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in simple insight extraction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during insight extraction"
        )

@router.get(
    "/health",
    summary="Health Check",
    description="Check if the insights service is healthy"
)
async def health_check():
    try:
        test_request = KeyInsightRequest(
            transcript="This is a test meeting transcript to verify the service is working.",
            max_insights=1
        )
        
        insights_service.extract_key_insights(test_request)
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "key_insights",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "key_insights",
                "error": str(e)
            }
        )

@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions and return structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions and return structured error response."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR"
        ).dict()
    )