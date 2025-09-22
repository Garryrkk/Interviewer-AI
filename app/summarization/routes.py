from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from .schemas import (
    SummarizationRequest,
    SummarizationResponse,
    MeetingAnalysisRequest,
    MeetingAnalysisResponse,
    AudioUploadResponse,
    SummaryUpdateRequest,
    MeetingContext,
    LLAVAAnalysisConfig,
    RealTimeAnalysisUpdate,
    ActionItem, KeyPoint, SummaryType, AnalysisType,
    SummaryUpdateRequest,
    MeetingContext,
    LLAVAAnalysisConfig,
    RealTimeAnalysisUpdate,
    SessionSummary,
    WebSocketMessage,
    AutomatedResponseMessage,
    FacialAnalysisMessage,
    SystemStatusMessage
)
from .service import SummarizationService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/summarization", tags=["summarization"])

# Initialize service
summarization_service = SummarizationService()


@router.post("/upload-audio", response_model=AudioUploadResponse)
async def upload_meeting_audio(
    audio_file: UploadFile = File(...),
    meeting_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Upload audio file for meeting analysis
    """
    try:
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        # Process audio upload
        result = await summarization_service.process_audio_upload(
            audio_file=audio_file,
            user_id=current_user.id,
            meeting_id=meeting_id
        )
        
        return AudioUploadResponse(**result)
    
    except Exception as e:
        logger.error(f"Error uploading audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload audio: {str(e)}"
        )


@router.post("/analyze-meeting", response_model=MeetingAnalysisResponse)
async def analyze_meeting_audio(
    request: MeetingAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze meeting audio and provide summarization with actionable points
    """
    try:
        # Analyze the meeting using LLAVA model
        analysis_result = await summarization_service.analyze_meeting_audio(
            audio_file_path=request.audio_file_path,
            meeting_context=request.meeting_context,
            user_id=current_user.id,
            analysis_type=request.analysis_type
        )
        
        return MeetingAnalysisResponse(**analysis_result)
    
    except Exception as e:
        logger.error(f"Error analyzing meeting: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze meeting: {str(e)}"
        )


@router.post("/summarize", response_model=SummarizationResponse)
async def create_summary(
    request: SummarizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create summarization from transcribed text or audio analysis
    """
    try:
        # Generate summary using LLAVA
        summary_result = await summarization_service.generate_summary(
            content=request.content,
            summary_type=request.summary_type,
            user_id=current_user.id,
            meeting_id=request.meeting_id,
            include_action_items=request.include_action_items
        )
        
        return SummarizationResponse(**summary_result)
    
    except Exception as e:
        logger.error(f"Error creating summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create summary: {str(e)}"
        )


@router.get("/meeting/{meeting_id}/summary", response_model=SummarizationResponse)
async def get_meeting_summary(
    meeting_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get existing summary for a meeting
    """
    try:
        summary = await summarization_service.get_meeting_summary(
            meeting_id=meeting_id,
            user_id=current_user.id
        )
        
        if not summary:
            raise HTTPException(
                status_code=404,
                detail="Summary not found for this meeting"
            )
        
        return SummarizationResponse(**summary)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting meeting summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get meeting summary: {str(e)}"
        )


@router.get("/user/summaries", response_model=List[SummarizationResponse])
async def get_user_summaries(
    current_user: User = Depends(get_current_user),
    limit: int = 10,
    offset: int = 0
):
    """
    Get all summaries for the current user
    """
    try:
        summaries = await summarization_service.get_user_summaries(
            user_id=current_user.id,
            limit=limit,
            offset=offset
        )
        
        return [SummarizationResponse(**summary) for summary in summaries]
    
    except Exception as e:
        logger.error(f"Error getting user summaries: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user summaries: {str(e)}"
        )


@router.delete("/meeting/{meeting_id}/summary")
async def delete_meeting_summary(
    meeting_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete summary for a specific meeting
    """
    try:
        success = await summarization_service.delete_meeting_summary(
            meeting_id=meeting_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Summary not found or access denied"
            )
        
        return JSONResponse(
            status_code=200,
            content={"message": "Summary deleted successfully"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete summary: {str(e)}"
        )


@router.post("/real-time-analysis", response_model=MeetingAnalysisResponse)
async def real_time_meeting_analysis(
    request: MeetingAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Real-time analysis of ongoing meeting audio
    """
    try:
        # Process real-time audio for immediate insights
        analysis = await summarization_service.real_time_audio_analysis(
            audio_chunk_path=request.audio_file_path,
            meeting_context=request.meeting_context,
            user_id=current_user.id
        )
        
        return MeetingAnalysisResponse(**analysis)
    
    except Exception as e:
        logger.error(f"Error in real-time analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform real-time analysis: {str(e)}"
        )
    
@router.post("/summary", response_model=SummaryUpdateRequest)
async def create_summary(summary: SummaryUpdateRequest):
    summaries.append(summary)
    return summary

@router.get("/summary", response_model=List[SummaryUpdateRequest])
async def get_summaries():
    return summaries

# ---------------- MeetingContext Routes ----------------
@router.post("/meeting", response_model=MeetingContext)
async def create_meeting(meeting: MeetingContext):
    meetings.append(meeting)
    return meeting

@router.get("/meeting", response_model=List[MeetingContext])
async def get_meetings():
    return meetings

# ---------------- LLAVAAnalysisConfig Routes ----------------
@router.post("/llava-config", response_model=LLAVAAnalysisConfig)
async def create_llava_config(config: LLAVAAnalysisConfig):
    llava_configs.append(config)
    return config

@router.get("/llava-config", response_model=List[LLAVAAnalysisConfig])
async def get_llava_configs():
    return llava_configs

# ---------------- RealTimeAnalysisUpdate Routes ----------------
@router.post("/realtime-update", response_model=RealTimeAnalysisUpdate)
async def create_realtime_update(update: RealTimeAnalysisUpdate):
    real_time_updates.append(update)
    return update

@router.get("/realtime-update", response_model=List[RealTimeAnalysisUpdate])
async def get_realtime_updates():
    return real_time_update

@router.post("/actionitems/", response_model=ActionItem)
async def create_action_item(item: ActionItem):
    # logic to save item
    return item

@router.get("/actionitems/", response_model=List[ActionItem])
async def get_action_items():
    # logic to retrieve all action items
    return []

@router.get("/actionitems/{item_id}", response_model=ActionItem)
async def get_action_item(item_id: int):
    # logic to retrieve item by ID
    return {"id": item_id, "name": "example"}

@router.put("/actionitems/{item_id}", response_model=ActionItem)
async def update_action_item(item_id: int, item: ActionItem):
    # logic to update item
    return item

@router.delete("/actionitems/{item_id}")
async def delete_action_item(item_id: int):
    # logic to delete item
    return {"message": f"ActionItem {item_id} deleted"}


# ----- KeyPoint CRUD -----
@router.post("/keypoints/", response_model=KeyPoint)
async def create_key_point(point: KeyPoint):
    return point

@router.get("/keypoints/", response_model=List[KeyPoint])
async def get_key_points():
    return []

@router.get("/keypoints/{point_id}", response_model=KeyPoint)
async def get_key_point(point_id: int):
    return {"id": point_id, "name": "example"}

@router.put("/keypoints/{point_id}", response_model=KeyPoint)
async def update_key_point(point_id: int, point: KeyPoint):
    return point

@router.delete("/keypoints/{point_id}")
async def delete_key_point(point_id: int):
    return {"message": f"KeyPoint {point_id} deleted"}


# ----- SummaryType CRUD -----
@router.post("/summarytypes/", response_model=SummaryType)
async def create_summary_type(summary: SummaryType):
    return summary

@router.get("/summarytypes/", response_model=List[SummaryType])
async def get_summary_types():
    return []

@router.get("/summarytypes/{summary_id}", response_model=SummaryType)
async def get_summary_type(summary_id: int):
    return {"id": summary_id, "name": "example"}

@router.put("/summarytypes/{summary_id}", response_model=SummaryType)
async def update_summary_type(summary_id: int, summary: SummaryType):
    return summary

@router.delete("/summarytypes/{summary_id}")
async def delete_summary_type(summary_id: int):
    return {"message": f"SummaryType {summary_id} deleted"}


# ----- AnalysisType CRUD -----
@router.post("/analysistypes/", response_model=AnalysisType)
async def create_analysis_type(analysis: AnalysisType):
    return analysis

@router.get("/analysistypes/", response_model=List[AnalysisType])
async def get_analysis_types():
    return []

@router.get("/analysistypes/{analysis_id}", response_model=AnalysisType)
async def get_analysis_type(analysis_id: int):
    return {"id": analysis_id, "name": "example"}

@router.put("/analysistypes/{analysis_id}", response_model=AnalysisType)
async def update_analysis_type(analysis_id: int, analysis: AnalysisType):
    return analysis

@router.delete("/analysistypes/{analysis_id}")
async def delete_analysis_type(analysis_id: int):
    return {"message": f"AnalysisType {analysis_id} deleted"}

session_summaries: List[SessionSummary] = []
websocket_messages: List[WebSocketMessage] = []
automated_responses: List[AutomatedResponseMessage] = []
facial_analyses: List[FacialAnalysisMessage] = []
system_statuses: List[SystemStatusMessage] = []

# ---------- SessionSummary Routes ----------
@router.post("/sessions", response_model=SessionSummary)
async def create_session(session: SessionSummary):
    session_summaries.append(session)
    return session

@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    return session_summaries

# ---------- WebSocketMessage Routes ----------
@router.post("/ws-messages", response_model=WebSocketMessage)
async def create_ws_message(message: WebSocketMessage):
    websocket_messages.append(message)
    return message

@router.get("/ws-messages", response_model=List[WebSocketMessage])
async def list_ws_messages():
    return websocket_messages

# ---------- AutomatedResponseMessage Routes ----------
@router.post("/automated-responses", response_model=AutomatedResponseMessage)
async def create_automated_response(response: AutomatedResponseMessage):
    automated_responses.append(response)
    return response

@router.get("/automated-responses", response_model=List[AutomatedResponseMessage])
async def list_automated_responses():
    return automated_responses

# ---------- FacialAnalysisMessage Routes ----------
@router.post("/facial-analyses", response_model=FacialAnalysisMessage)
async def create_facial_analysis(analysis: FacialAnalysisMessage):
    facial_analyses.append(analysis)
    return analysis

@router.get("/facial-analyses", response_model=List[FacialAnalysisMessage])
async def list_facial_analyses():
    return facial_analyses

# ---------- SystemStatusMessage Routes ----------
@router.post("/system-statuses", response_model=SystemStatusMessage)
async def create_system_status(status: SystemStatusMessage):
    system_statuses.append(status)
    return status

@router.get("/system-statuses", response_model=List[SystemStatusMessage])
async def list_system_statuses():
    return system_statuses