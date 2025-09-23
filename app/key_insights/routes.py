from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import asyncio
import logging
from datetime import datetime

from .services import KeyInsightsService
from .schemas import (
    # New schema imports (now fully used)
    InsightType,
    KeyInsight,
    KeyInsightRequest,
    KeyInsightResponse,
    ErrorResponse,
    SimplifiedInsightRequest,
    SimplifiedInsightResponse,
    AnalysisStatusResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/key-insights", tags=["Key Insights"])

insights_service = KeyInsightsService()

# ---------------------------------------------------------------------
# ✅ NEW: Insight Types Endpoint (uses InsightType)
# ---------------------------------------------------------------------
@router.get("/types", response_model=List[str])
async def list_insight_types():
    """
    Get all available InsightType values.
    Useful for UI dropdowns or validation on the frontend.
    """
    return [t.value for t in InsightType]


# ---------------------------------------------------------------------
# ✅ NEW: Sample Insight Endpoint (uses KeyInsight)
# ---------------------------------------------------------------------
@router.get("/sample", response_model=KeyInsight)
async def get_sample_key_insight():
    """
    Returns a sample KeyInsight object for testing or UI prototyping.
    """
    sample = KeyInsight(
        id="sample-123",
        content="This is a sample key point generated for testing.",
        type=InsightType.KEY_POINT,
        confidence_score=0.95,
        timestamp=datetime.utcnow(),
        source_section="introduction"
    )
    return sample


# ---------------------------------------------------------------------
# Existing Endpoints (unchanged except imports)
# ---------------------------------------------------------------------
@router.post("/analyze", response_model=KeyInsightResponse, responses={500: {"model": ErrorResponse}})
async def generate_key_insights(
    request: KeyInsightRequest,
    image_file: Optional[UploadFile] = File(None)
):
    """
    Generate key insights from a meeting transcript, with optional image analysis.
    """
    try:
        logger.info(f"Generating key insights for meeting: {request.meeting_id}")

        if not request.transcript and not image_file:
            raise HTTPException(
                status_code=400,
                detail="Either transcript or image file must be provided"
            )

        image_data = None
        if image_file:
            if not image_file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Please upload an image."
                )
            image_data = await image_file.read()
            logger.info(f"Image uploaded: {image_file.filename}, Size: {len(image_data)} bytes")

        insights_response = await insights_service.generate_insights(
            meeting_context=request.transcript,
            meeting_id=request.meeting_id,
            extract_types=request.extract_types,
            max_insights=request.max_insights,
            image_data=image_data
        )

        return insights_response

    except Exception as e:
        logger.error(f"Error generating key insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.post("/simplify", response_model=SimplifiedInsightResponse, responses={500: {"model": ErrorResponse}})
async def get_simplified_insights(request: SimplifiedInsightRequest):
    """
    Generate a simplified version of existing key insights.
    """
    try:
        logger.info(f"Simplifying insights for insight ID: {request.original_insight_id}")

        simplified_response = await insights_service.simplify_insights(
            original_insights=request.original_insights,
            original_tips=request.original_tips,
            simplification_level=request.simplification_level,
            original_insight_id=request.original_insight_id
        )

        return simplified_response

    except Exception as e:
        logger.error(f"Error generating simplified insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to simplify insights: {str(e)}")


@router.get("/status/{insight_id}", response_model=AnalysisStatusResponse, responses={500: {"model": ErrorResponse}})
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


@router.get("/history/{meeting_id}", responses={500: {"model": ErrorResponse}})
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


@router.delete("/insights/{insight_id}", responses={500: {"model": ErrorResponse}})
async def delete_insights(insight_id: str):
    """
    Delete specific insights by ID.
    """
    try:
        success = await insights_service.delete_insights(insight_id)
        if not success:
            raise HTTPException(status_code=404, detail="Insights not found")

        return {"message": "Insights deleted successfully", "insight_id": insight_id}

    except Exception as e:
        logger.error(f"Error deleting insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete insights: {str(e)}")


@router.post("/batch-analyze", responses={500: {"model": ErrorResponse}})
async def batch_analyze_insights(
    meeting_contexts: List[str],
    meeting_ids: List[str],
    image_files: Optional[List[UploadFile]] = File(None)
):
    """
    Generate insights for multiple meetings in a single request.
    """
    try:
        if len(meeting_contexts) != len(meeting_ids):
            raise HTTPException(
                status_code=400,
                detail="Number of meeting contexts and meeting IDs must match"
            )

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
