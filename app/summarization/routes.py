from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from .schemas import (
    SummarizationRequest,
    SummarizationResponse,
    MeetingAnalysisRequest,
    MeetingAnalysisResponse,
    AudioUploadResponse
)
from .service import SummarizationService
from ..auth.dependencies import get_current_user
from ..models.user import User

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