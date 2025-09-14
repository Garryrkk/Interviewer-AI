import asyncio
import base64
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator
import io
import wave

import aiohttp  
import numpy as np
from pydantic import ValidationError

from .schemas import (
    PreopConfig, PreopResponse, AudioChunk, ProcessingResponse,
    RecognitionSession, RecognitionResponse, Insight, SessionStatus,
    ErrorResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceRecognitionService:
    def __init__(self, ollama_host: str = "http://localhost:11434", model_name: str = "nous-hermes2"):
        self.active_sessions: Dict[str, dict] = {}
        self.ollama_host = ollama_host.rstrip('/')
        self.model_name = model_name
        self.supported_formats = ['wav', 'mp3', 'flac', 'm4a']
        self.max_chunk_size = 1024 * 1024  
        self.initialize = False  # Initialize attribute added

    async def initialize_service(self) -> bool:
        """Initialize the voice recognition service"""
        try:
            # Check if Ollama service is available
            ollama_ready = await self._check_ollama_service()
            
            if ollama_ready:
                self.initialize = True
                logger.info("VoiceRecognitionService initialized successfully")
            else:
                logger.warning("VoiceRecognitionService initialization failed - Ollama service not ready")
            
            return self.initialize
            
        except Exception as e:
            logger.error(f"Service initialization error: {str(e)}")
            self.initialize = False
            return False

    def is_initialized(self) -> bool:
        """Check if the service is initialized"""
        return self.initialize

    async def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        if not self.initialize:
            logger.warning("Service not initialized. Call initialize_service() first.")
            return ""
            
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 4096
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "").strip()
                    else:
                        logger.error(f"Ollama API error: {response.status}")
                        return ""
                        
        except Exception as e:
            logger.error(f"Ollama API call error: {str(e)}")
            return ""

    async def validate_preop_config(self, config: PreopConfig) -> PreopResponse:
        try:
            logger.info(f"Validating preop config: {config.dict()}")
            
            config_validated = await self._validate_audio_device(config.device)
            
            ai_service_ready = await self._check_ollama_service()
            
            if config.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
                return PreopResponse(
                    status="error",
                    message="Unsupported sample rate",
                    config_validated=False,
                    ai_service_ready=ai_service_ready
                )
            
            status = "success" if config_validated and ai_service_ready else "warning"
            message = "Configuration validated successfully"
            
            if not config_validated:
                message = "Audio device validation failed"
            elif not ai_service_ready:
                message = "Ollama service not ready"
            
            return PreopResponse(
                status=status,
                message=message,
                config_validated=config_validated,
                ai_service_ready=ai_service_ready
            )
            
        except Exception as e:
            logger.error(f"Preop validation error: {str(e)}")
            return PreopResponse(
                status="error",
                message=f"Configuration validation failed: {str(e)}",
                config_validated=False,
                ai_service_ready=False
            )

    async def _validate_audio_device(self, device: str) -> bool:
        try:
            valid_devices = ["default", "microphone", "system_audio", "line_in"]
            return device in valid_devices
        except Exception as e:
            logger.error(f"Device validation error: {str(e)}")
            return False
    
    async def _check_ollama_service(self) -> bool:
        """Check if Ollama service is ready"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if our model is available
                        models = [model.get("name", "") for model in data.get("models", [])]
                        return any(self.model_name in model for model in models)
                    return False
        except Exception as e:
            logger.error(f"Ollama service check error: {str(e)}")
            return False

    async def process_audio_chunk(self, chunk: AudioChunk) -> ProcessingResponse:
        try:
            audio_data = base64.b64decode(chunk.data)
            
            if len(audio_data) > self.max_chunk_size:
                raise ValueError(f"Chunk size exceeds maximum ({self.max_chunk_size} bytes)")
            
            processed_audio = await self._preprocess_audio(audio_data)
            
            partial_transcript = await self._transcribe_chunk(processed_audio)
            
            confidence = await self._calculate_confidence(partial_transcript)
            
            return ProcessingResponse(
                partial_transcript=partial_transcript,
                is_final=False,  
                confidence=confidence,
                timestamp=chunk.timestamp
            )
            
        except Exception as e:
            logger.error(f"Audio chunk processing error: {str(e)}")
            return ProcessingResponse(
                partial_transcript="",
                is_final=False,
                confidence=0.0,
                timestamp=chunk.timestamp
            )

    async def _preprocess_audio(self, audio_data: bytes) -> bytes:
        try:
            return audio_data
        except Exception as e:
            logger.error(f"Audio preprocessing error: {str(e)}")
            return audio_data
    
    async def _transcribe_chunk(self, audio_data: bytes) -> str:
        try:
            transcript = "Sample partial transcript..."
            
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""
    
    async def _calculate_confidence(self, text: str) -> float:
        if not text:
            return 0.0
        return min(0.95, len(text) / 100.0)
    
    async def start_recognition_session(self, session: RecognitionSession) -> SessionStatus:
        if not self.initialize:
            raise RuntimeError("Service not initialized. Call initialize_service() first.")
            
        try:
            session_id = session.session_id or str(uuid.uuid4())
            
            self.active_sessions[session_id] = {
                "config": session,
                "transcript": "",
                "start_time": datetime.now(),
                "last_activity": datetime.now(),
                "status": "active",
                "audio_chunks": []
            }
            
            logger.info(f"Started recognition session: {session_id}")
            
            return SessionStatus(
                session_id=session_id,
                status="active",
                duration=0.0,
                transcript_length=0,
                last_activity=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Session start error: {str(e)}")
            raise
    
    async def stop_recognition_session(self, session_id: str) -> RecognitionResponse:
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session_data = self.active_sessions[session_id]
            session_data["status"] = "stopped"
            
            start_time = session_data["start_time"]
            duration = (datetime.now() - start_time).total_seconds()
            
            transcript = session_data.get("transcript", "")
            
            insights = None
            if session_data["config"].enable_ai_insights and transcript:
                insights = await self._generate_insights(transcript)
            
            confidence = await self._calculate_confidence(transcript)
            
            del self.active_sessions[session_id]
            
            return RecognitionResponse(
                session_id=session_id,
                transcript=transcript,
                confidence=confidence,
                duration=duration,
                insights=insights,
                language_detected=session_data["config"].language,
                speaker_count=1  
            )
            
        except Exception as e:
            logger.error(f"Session stop error: {str(e)}")
            raise
        
    async def _generate_insights(self, transcript: str) -> List[Insight]:
        """Generate AI-powered insights from transcript using Ollama"""
        try:
            insights = []
            
            summary = await self._generate_summary(transcript)
            if summary:
                insights.append(Insight(
                    type="summary",
                    content=summary,
                    confidence=0.85
                ))
            
            key_points = await self._extract_key_points(transcript)
            for point in key_points:
                insights.append(Insight(
                    type="key_point",
                    content=point,
                    confidence=0.8
                ))
            
            action_items = await self._identify_action_items(transcript)
            for item in action_items:
                insights.append(Insight(
                    type="action_item",
                    content=item,
                    confidence=0.75
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Insight generation error: {str(e)}")
            return []
    
    async def _generate_summary(self, transcript: str) -> str:
        """Generate summary from transcript using Ollama"""
        try:
            if len(transcript) < 50:
                return ""
            
            system_prompt = "You are an expert at creating concise, informative summaries of conversations and text."
            prompt = f"""Please provide a brief, clear summary of the following transcript in 2-3 sentences:

Transcript:
{transcript}

Summary:"""
            
            summary = await self._call_ollama(prompt, system_prompt)
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return ""
    
    async def _extract_key_points(self, transcript: str) -> List[str]:
        """Extract key points from transcript using Ollama"""
        try:
            system_prompt = "You are an expert at identifying the most important points from conversations and text."
            prompt = f"""Please identify the 3-5 most important key points from the following transcript. List each point on a separate line:

Transcript:
{transcript}

Key Points:"""
            
            response = await self._call_ollama(prompt, system_prompt)
            if response:
                points = [point.strip() for point in response.split('\n') if point.strip()]
                cleaned_points = []
                for point in points:
                    cleaned_point = point
                    for prefix in ['- ', '• ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                        if cleaned_point.startswith(prefix):
                            cleaned_point = cleaned_point[len(prefix):].strip()
                            break
                    if cleaned_point and len(cleaned_point) > 10:  # Filter out very short points
                        cleaned_points.append(cleaned_point)
                return cleaned_points[:5]  # Limit to 5 points
            
            return []
            
        except Exception as e:
            logger.error(f"Key point extraction error: {str(e)}")
            return []

    async def _identify_action_items(self, transcript: str) -> List[str]:
        """Identify action items from transcript using Ollama"""
        try:
            system_prompt = "You are an expert at identifying actionable tasks and commitments from conversations."
            prompt = f"""Please identify any action items, tasks, or commitments mentioned in the following transcript. List each action item on a separate line:

Transcript:
{transcript}

Action Items:"""
            
            response = await self._call_ollama(prompt, system_prompt)
            if response:
                items = [item.strip() for item in response.split('\n') if item.strip()]
                cleaned_items = []
                for item in items:
                    cleaned_item = item
                    for prefix in ['- ', '• ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                        if cleaned_item.startswith(prefix):
                            cleaned_item = cleaned_item[len(prefix):].strip()
                            break
                    if cleaned_item and len(cleaned_item) > 10:  
                        cleaned_items.append(cleaned_item)
                return cleaned_items[:5]  
            
            return []
            
        except Exception as e:
            logger.error(f"Action item identification error: {str(e)}")
            return []
    
    async def get_session_status(self, session_id: str) -> SessionStatus:
        """Get current session status"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_data = self.active_sessions[session_id]
        start_time = session_data["start_time"]
        duration = (datetime.now() - start_time).total_seconds()
        
        return SessionStatus(
            session_id=session_id,
            status=session_data["status"],
            duration=duration,
            transcript_length=len(session_data.get("transcript", "")),
            last_activity=session_data["last_activity"]
        )
    
    async def update_session_transcript(self, session_id: str, text: str):
        """Update session transcript with new text"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["transcript"] += " " + text
            self.active_sessions[session_id]["last_activity"] = datetime.now()
    
    async def cleanup_inactive_sessions(self, max_inactive_minutes: int = 60):
        """Clean up inactive sessions"""
        cutoff_time = datetime.now() - timedelta(minutes=max_inactive_minutes)
        inactive_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            if session_data["last_activity"] < cutoff_time:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            logger.info(f"Cleaning up inactive session: {session_id}")
            del self.active_sessions[session_id]


voice_service = VoiceRecognitionService(
    ollama_host="http://localhost:11434",  
    model_name="nous-hermes2" 
)