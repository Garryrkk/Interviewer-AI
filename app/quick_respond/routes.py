from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
import json
from typing import Optional
from .schemas import (
    QuickRespondRequest, 
    QuickRespondResponse, 
    SimplifyRequest,
    SimplifyResponse,
    MeetingContext
)
from .services import QuickRespondService

router = APIRouter(prefix="/api/quick-respond", tags=["quick-respond"])
quick_respond_service = QuickRespondService()

@router.post("/analyze-screenshot", response_model=QuickRespondResponse)
async def analyze_meeting_screenshot(
    screenshot: UploadFile = File(...),
    meeting_context: Optional[str] = None,
    audio_transcript: Optional[str] = None
):
    """
    Analyze live meeting screenshot and provide key insights
    """
    try:
        # Validate file type
        if not screenshot.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read screenshot data
        screenshot_data = await screenshot.read()
        
        # Create request object
        request_data = QuickRespondRequest(
            screenshot_data=screenshot_data,
            meeting_context=meeting_context,
            audio_transcript=audio_transcript,
            analysis_type="key_insights"
        )
        
        # Get analysis from service
        response = await quick_respond_service.analyze_meeting_content(request_data)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze-screenshot/stream")
async def analyze_meeting_screenshot_stream(
    screenshot: UploadFile = File(...),
    meeting_context: Optional[str] = None,
    audio_transcript: Optional[str] = None
):
    """
    Stream real-time analysis of meeting screenshot
    """
    try:
        if not screenshot.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        screenshot_data = await screenshot.read()
        
        request_data = QuickRespondRequest(
            screenshot_data=screenshot_data,
            meeting_context=meeting_context,
            audio_transcript=audio_transcript,
            analysis_type="key_insights"
        )
        
        async def generate_stream():
            async for chunk in quick_respond_service.analyze_meeting_content_stream(request_data):
                yield f"data: {json.dumps(chunk)}\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming analysis failed: {str(e)}")

@router.post("/simplify", response_model=SimplifyResponse)
async def simplify_response(request: SimplifyRequest):
    """
    Simplify a complex analysis response
    """
    try:
        response = await quick_respond_service.simplify_analysis(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simplification failed: {str(e)}")

@router.post("/batch-analyze")
async def batch_analyze_screenshots(
    screenshots: list[UploadFile] = File(...),
    meeting_context: Optional[str] = None
):
    """
    Analyze multiple screenshots from a meeting session
    """
    try:
        results = []
        
        for screenshot in screenshots:
            if not screenshot.content_type.startswith('image/'):
                continue
                
            screenshot_data = await screenshot.read()
            
            request_data = QuickRespondRequest(
                screenshot_data=screenshot_data,
                meeting_context=meeting_context,
                analysis_type="key_insights"
            )
            
            result = await quick_respond_service.analyze_meeting_content(request_data)
            results.append({
                "filename": screenshot.filename,
                "analysis": result
            })
        
        return {"batch_results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Check if LLAVA/Ollama services are available
    """
    try:
        health_status = await quick_respond_service.check_service_health()
        return health_status
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@router.post("/context/update")
async def update_meeting_context(context: MeetingContext):
    """
    Update meeting context for better analysis
    """
    try:
        await quick_respond_service.update_meeting_context(context)
        return {"status": "success", "message": "Meeting context updated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context update failed: {str(e)}")

@router.delete("/context/clear")
async def clear_meeting_context():
    """
    Clear stored meeting context
    """
    try:
        await quick_respond_service.clear_meeting_context()
        return {"status": "success", "message": "Meeting context cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context clearing failed: {str(e)}")