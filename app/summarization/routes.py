from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any
from .schemas import SummarizeRequest, SummarizeResponse, SummarizeError
from .service import summarization_service, generate_summary_async

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/summarize",
    tags=["Summarization"],
    responses={404: {"description": "Not found"}}
)


@router.post("/", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """
    Generate a summary of the given text (e.g., interview transcripts).
    """
    try:
        if not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )

        logger.info(f"Summarization request received, length: {len(request.text)} chars")

        # Generate summary
        summary = await generate_summary_async(
            text=request.text,
            style=request.style,
            simplify=request.simplify
        )

        logger.info("Summary generated successfully")

        return SummarizeResponse(
            summary=summary,
            source="summarization-service",
            success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=SummarizeError(
                error="Failed to generate summary",
                error_code="INTERNAL_ERROR"
            ).model_dump()
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check for Summarization service
    """
    return {
        "status": "healthy",
        "service": "summarization",
        "message": "Summarization service is running"
    }
