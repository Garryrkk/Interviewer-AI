from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any
from .schemas import HandsFreeRequest, HandsFreeResponse, HandsFreeError
from .service import handsfree_service, generate_handsfree_response_async

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/handsfree",
    tags=["Hands-Free Mode"],
    responses={404: {"description": "Not found"}}
)


@router.post("/", response_model=HandsFreeResponse)
async def handsfree_reply(request: HandsFreeRequest) -> HandsFreeResponse:
    """
    Generate AI reply for spoken transcript in hands-free mode.
    """
    try:
        if not request.transcript.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Transcript cannot be empty"
            )

        logger.info(f"Hands-free transcript received: {request.transcript[:50]}...")

        # Generate AI response
        ai_text = await generate_handsfree_response_async(
            transcript=request.transcript,
            context=request.context,
            simplify=request.simplify
        )

        logger.info("Hands-free response generated successfully")

        return HandsFreeResponse(
            response=ai_text,
            source="handsfree-service",
            success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in hands-free mode: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=HandsFreeError(
                error="Failed to generate hands-free response",
                error_code="INTERNAL_ERROR"
            ).model_dump()
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check for Hands-Free Mode service
    """
    return {
        "status": "healthy",
        "service": "handsfree",
        "message": "Hands-Free Mode service is running"
    }
