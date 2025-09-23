from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
import json
from typing import Optional
from typing import List
from fastapi import Query
from datetime import datetime
from .schemas import UrgencyLevel, PaginatedResponse
from .schemas import (
    QuickRespondRequest, 
    QuickRespondResponse, 
    SimplifyRequest,
    SimplifyResponse,
    MeetingContext,
    MeetingStatus,
    ParticipantInfo,
    ScreenContent,
    MeetingMetrics,
    OllamaConfig,
    QuickRespondConfig,
    ModelPrompts,
    QuickRespondRequest, 
    UrgencyLevel, 
    BatchAnalysisRequest, 
    PaginatedResponse,
    HealthCheckResponse,
    AdvancedAnalysisRequest,
    AdvancedAnalysisResponse,
    BatchAnalysisResponse
)
from .services import QuickRespondService

router = APIRouter(prefix="/api/quick-respond", tags=["quick-respond"])
quick_respond_service = QuickRespondService()


@router.post("/webhook-event", response_model=WebhookResponse)
async def create_webhook_event(event: WebhookEvent):
    # Replace this with your actual logic
    return WebhookResponse(message=f"Received event: {event.event_name}")

# ---------------- OllamaConfig Routes ----------------
@router.post("/ollama-config", response_model=OllamaConfig)
async def create_ollama_config(config: OllamaConfig):
    # Add save logic
    return config

@router.get("/ollama-config", response_model=List[OllamaConfig])
async def get_all_ollama_configs():
    # Replace with fetch logic
    return []

# ---------------- QuickRespondConfig Routes ----------------
@router.post("/quick-respond-config", response_model=QuickRespondConfig)
async def create_quick_respond_config(config: QuickRespondConfig):
    return config

@router.get("/quick-respond-config", response_model=List[QuickRespondConfig])
async def get_all_quick_respond_configs():
    return []

# ---------------- ModelPrompts Routes ----------------
@router.post("/model-prompts", response_model=ModelPrompts)
async def create_model_prompt(prompt: ModelPrompts):
    return prompt

@router.get("/model-prompts", response_model=List[ModelPrompts])
async def get_all_model_prompts():
    return []

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
    
@router.post("/meeting_status/", response_model=MeetingStatus)
async def create_meeting_status(meeting_status: MeetingStatus):
    # Add your DB logic here
    return meeting_status

@router.get("/meeting_status/{id}", response_model=MeetingStatus)
async def get_meeting_status(id: int):
    # Fetch from DB
    return {"id": id, "status": "active"}  # Example

@router.get("/meeting_status/", response_model=List[MeetingStatus])
async def list_meeting_statuses():
    # Fetch all from DB
    return []

@router.put("/meeting_status/{id}", response_model=MeetingStatus)
async def update_meeting_status(id: int, meeting_status: MeetingStatus):
    # Update DB
    return meeting_status

@router.delete("/meeting_status/{id}")
async def delete_meeting_status(id: int):
    # Delete from DB
    return {"message": f"MeetingStatus {id} deleted"}

# ============================
# ParticipantInfo Routes
# ============================

@router.post("/participant_info/", response_model=ParticipantInfo)
async def create_participant_info(participant: ParticipantInfo):
    return participant

@router.get("/participant_info/{id}", response_model=ParticipantInfo)
async def get_participant_info(id: int):
    return {"id": id, "name": "John Doe"}  # Example

@router.get("/participant_info/", response_model=List[ParticipantInfo])
async def list_participants():
    return []

@router.put("/participant_info/{id}", response_model=ParticipantInfo)
async def update_participant_info(id: int, participant: ParticipantInfo):
    return participant

@router.delete("/participant_info/{id}")
async def delete_participant_info(id: int):
    return {"message": f"ParticipantInfo {id} deleted"}

# ============================
# ScreenContent Routes
# ============================

@router.post("/screen_content/", response_model=ScreenContent)
async def create_screen_content(screen: ScreenContent):
    return screen

@router.get("/screen_content/{id}", response_model=ScreenContent)
async def get_screen_content(id: int):
    return {"id": id, "content": "Example content"}  # Example

@router.get("/screen_content/", response_model=List[ScreenContent])
async def list_screen_content():
    return []

@router.put("/screen_content/{id}", response_model=ScreenContent)
async def update_screen_content(id: int, screen: ScreenContent):
    return screen

@router.delete("/screen_content/{id}")
async def delete_screen_content(id: int):
    return {"message": f"ScreenContent {id} deleted"}

# ============================
# MeetingMetrics Routes
# ============================

@router.post("/meeting_metrics/", response_model=MeetingMetrics)
async def create_meeting_metrics(metrics: MeetingMetrics):
    return metrics

@router.get("/meeting_metrics/{id}", response_model=MeetingMetrics)
async def get_meeting_metrics(id: int):
    return {"id": id, "metrics": {}}  # Example

@router.get("/meeting_metrics/", response_model=List[MeetingMetrics])
async def list_meeting_metrics():
    return []

@router.put("/meeting_metrics/{id}", response_model=MeetingMetrics)
async def update_meeting_metrics(id: int, metrics: MeetingMetrics):
    return metrics

@router.delete("/meeting_metrics/{id}")
async def delete_meeting_metrics(id: int):
    return {"message": f"MeetingMetrics {id} deleted"}  

@router.post("/quick-respond", response_model=QuickRespondResponse)
async def quick_respond(request: QuickRespondRequest):
    """
    Generate quick insights from a screenshot and optional audio transcript.
    """
    # TODO: Implement analysis logic
    return QuickRespondResponse(
        key_insights=[],
        full_analysis="Analysis placeholder",
        timestamp=datetime.utcnow(),
        confidence_score=0.7,
        can_simplify=True,
        session_id="session_123"
    )

# Simplify analysis endpoint
@router.post("/simplify", response_model=SimplifyResponse)
async def simplify_analysis(request: SimplifyRequest):
    """
    Simplify a complex analysis text into bullet points and actions.
    """
    # TODO: Implement simplification logic
    return SimplifyResponse(
        simplified_text="Simplified placeholder",
        simple_points=["Point 1", "Point 2"],
        actions_needed=["Action 1"],
        meeting_status="neutral",
        timestamp=datetime.utcnow(),
        original_length=len(request.original_analysis),
        simplified_length=50
    )

# Batch analysis endpoint
@router.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analysis(request: BatchAnalysisRequest):
    """
    Perform batch analysis on multiple screenshots.
    """
    # TODO: Implement batch analysis
    return BatchAnalysisResponse(
        individual_analyses=[],
        summary_insights=[],
        meeting_progression=[],
        key_moments=[],
        overall_metrics=None,
        batch_session_id="batch_123"
    )

# Advanced analysis endpoint
@router.post("/advanced", response_model=AdvancedAnalysisResponse)
async def advanced_analysis(request: AdvancedAnalysisRequest):
    """
    Perform detailed analysis with optional custom prompts, metrics, and sentiment.
    """
    # TODO: Implement advanced analysis
    return AdvancedAnalysisResponse(
        key_insights=[],
        full_analysis="Advanced analysis placeholder",
        session_id="adv_session_123",
        timestamp=datetime.utcnow(),
        recommendations=[]
    )

# Health check endpoint
@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check service and model health.
    """
    # TODO: Implement real health check
    return HealthCheckResponse(
        status="ok",
        ollama=True,
        llava_model=True,
        llama_model=True,
        timestamp=datetime.utcnow(),
        response_time_ms=50
    )

@router.get("/urgency-levels", response_model=List[UrgencyLevel])
async def list_urgency_levels():
    """
    Get all possible urgency levels (enum values).
    """
    return [level for level in UrgencyLevel]

@router.get("/urgency-levels/{level}", response_model=UrgencyLevel)
async def get_urgency_level(level: UrgencyLevel):
    """
    Validate and return a specific urgency level.
    If an invalid level is passed, FastAPI will auto-throw 422.
    """
    return level


# ============================
# PaginatedResponse Demo Route
# ============================

@router.get("/items", response_model=PaginatedResponse)
async def list_items(
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page")
):
    """
    Example endpoint to demonstrate PaginatedResponse usage.
    Replace with actual DB query in production.
    """
    # Fake dataset for demo
    total_items = 55
    all_items = [f"Item {i}" for i in range(1, total_items + 1)]

    # Pagination math
    start = (page - 1) * page_size
    end = start + page_size
    paged_items = all_items[start:end]

    return PaginatedResponse(
        items=paged_items,
        total=total_items,
        page=page,
        page_size=page_size,
        timestamp=datetime.utcnow()
    )