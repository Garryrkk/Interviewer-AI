from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import asyncio
import logging
from datetime import datetime

from .services import KeyInsightsService
from .schemas import (
    KeyInsightsRequest,
    KeyInsightsResponse,
    SimplifiedInsightsRequest,
    SimplifiedInsightsResponse,
    AnalysisStatusResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/key-insights", tags=["Key Insights"])

# Initialize service
insights_service = KeyInsightsService()


@router.post("/analyze", response_model=KeyInsightsResponse)
async def generate_key_insights(
    request: KeyInsightsRequest,
    image_file: Optional[UploadFile] = File(None)
):
    """
    Generate key insights from meeting context with optional image analysis
    for facial expressions and body language.
    """
    try:
        logger.info(f"Generating key insights for meeting: {request.meeting_id}")
        
        # Validate input
        if not request.meeting_context and not image_file:
            raise HTTPException(
                status_code=400,
                detail="Either meeting context or image file must be provided"
            )
        
        # Process image if provided
        image_data = None
        if image_file:
            # Validate image file
            if not image_file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Please upload an image file."
                )
            
            image_data = await image_file.read()
            logger.info(f"Image uploaded: {image_file.filename}, Size: {len(image_data)} bytes")
        
        # Generate insights
        insights_response = await insights_service.generate_insights(
            meeting_context=request.meeting_context,
            meeting_id=request.meeting_id,
            participants=request.participants,
            image_data=image_data,
            analysis_focus=request.analysis_focus
        )
        
        logger.info(f"Successfully generated insights for meeting: {request.meeting_id}")
        return insights_response
        
    except Exception as e:
        logger.error(f"Error generating key insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.post("/simplify", response_model=SimplifiedInsightsResponse)
async def get_simplified_insights(request: SimplifiedInsightsRequest):
    """
    Generate a more simplified version of the key insights.
    """
    try:
        logger.info(f"Generating simplified insights for insight ID: {request.original_insight_id}")
        
        simplified_response = await insights_service.simplify_insights(
            original_insights=request.original_insights,
            original_tips=request.original_tips,
            simplification_level=request.simplification_level,
            original_insight_id=request.original_insight_id
        )
        
        logger.info(f"Successfully generated simplified insights")
        return simplified_response
        
    except Exception as e:
        logger.error(f"Error generating simplified insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to simplify insights: {str(e)}")


@router.get("/status/{insight_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(insight_id: str):
    """
    Get the status of an ongoing insight analysis.
    """
    try:
        status = await insights_service.get_analysis_status(insight_id)
        return status
        
    except Exception as e:
        logger.error(f"Error getting analysis status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/history/{meeting_id}")
async def get_insights_history(meeting_id: str):
    """
    Get all insights generated for a specific meeting.
    """
    try:
        history = await insights_service.get_insights_history(meeting_id)
        return {"meeting_id": meeting_id, "insights_history": history}
        
    except Exception as e:
        logger.error(f"Error getting insights history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights history: {str(e)}")


@router.delete("/insights/{insight_id}")
async def delete_insights(insight_id: str):
    """
    Delete specific insights.
    """
    try:
        success = await insights_service.delete_insights(insight_id)
        if not success:
            raise HTTPException(status_code=404, detail="Insights not found")
        
        return {"message": "Insights deleted successfully", "insight_id": insight_id}
        
    except Exception as e:
        logger.error(f"Error deleting insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete insights: {str(e)}")


@router.post("/batch-analyze")
async def batch_analyze_insights(
    meeting_contexts: List[str],
    meeting_ids: List[str],
    image_files: Optional[List[UploadFile]] = File(None)
):
    """
    Generate insights for multiple meetings in batch.
    """
    try:
        if len(meeting_contexts) != len(meeting_ids):
            raise HTTPException(
                status_code=400,
                detail="Number of meeting contexts and meeting IDs must match"
            )
        
        # Process batch requests
        batch_results = []
        for i, (context, meeting_id) in enumerate(zip(meeting_contexts, meeting_ids)):
            image_data = None
            if image_files and i < len(image_files):
                image_data = await image_files[i].read()
            
            try:
                result = await insights_service.generate_insights(
                    meeting_context=context,
                    meeting_id=meeting_id,
                    image_data=image_data
                )
                batch_results.append({
                    "meeting_id": meeting_id,
                    "status": "success",
                    "insights": result
                })
            except Exception as e:
                batch_results.append({
                    "meeting_id": meeting_id,
                    "status": "error",
                    "error": str(e)
                })
        
        return {"batch_results": batch_results}
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")