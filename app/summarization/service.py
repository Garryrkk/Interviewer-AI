import os
import json
import uuid
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import aiofiles
import speech_recognition as sr
import cv2
import mediapipe as mp
import numpy as np
import wave
import subprocess
import requests
import openai
from transformers import pipeline
import torch

from database import get_db
from config import settings
from .schemas import (
    ActionItem, 
    KeyPoint, 
    MeetingAnalysisResponse, 
    SummarizationResponse,
    LLAVAAnalysisConfig,
    SummaryType
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationService:
    """Service class for handling meeting summarization using LLAVA model"""
    
    def __init__(self):
        self.llava_config = LLAVAAnalysisConfig()
        self.audio_storage_path = Path(settings.AUDIO_STORAGE_PATH)
        self.audio_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize MediaPipe for audio processing
        self.mp_audio = mp.solutions.audio
        
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # LLAVA API configuration
        self.llava_api_url = settings.LLAVA_API_URL
        self.llava_api_key = settings.LLAVA_API_KEY

    def _get_audio_duration_cv2(self, audio_file_path: str) -> float:
        """Get audio duration using OpenCV and subprocess"""
        try:
            # Use ffprobe to get audio duration
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', audio_file_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                # Fallback: try to read as wave file
                try:
                    with wave.open(audio_file_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        return frames / sample_rate
                except:
                    # Last resort: estimate based on file size (rough approximation)
                    file_size = os.path.getsize(audio_file_path)
                    # Assuming average bitrate of 128 kbps
                    return file_size * 8 / (128 * 1024)
                    
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {str(e)}")
            return 0.0

    def _convert_audio_to_wav_cv2(self, input_path: str, output_path: str) -> bool:
        """Convert audio file to WAV format using subprocess/ffmpeg"""
        try:
            # Use ffmpeg for audio conversion
            result = subprocess.run([
                'ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', 
                '-ar', '44100', '-ac', '1', '-y', output_path
            ], capture_output=True, stderr=subprocess.PIPE)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error converting audio: {str(e)}")
            return False

    async def process_audio_upload(
        self, 
        audio_file, 
        user_id: str, 
        meeting_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process uploaded audio file"""
        try:
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            
            # Create file path
            file_extension = Path(audio_file.filename).suffix
            file_path = self.audio_storage_path / f"{file_id}{file_extension}"
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)
            
            # Get audio duration using CV2-based method
            duration = self._get_audio_duration_cv2(str(file_path))
            
            # Store metadata in database
            async with get_db() as conn:
                await conn.execute("""
                    INSERT INTO audio_files (id, user_id, meeting_id, file_path, duration, file_size, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (file_id, user_id, meeting_id, str(file_path), duration, len(content), datetime.utcnow()))
                await conn.commit()
            
            return {
                "file_id": file_id,
                "file_path": str(file_path),
                "duration": duration,
                "file_size": len(content),
                "upload_timestamp": datetime.utcnow(),
                "status": "uploaded"
            }
            
        except Exception as e:
            logger.error(f"Error processing audio upload: {str(e)}")
            raise

    async def analyze_meeting_audio(
        self,
        audio_file_path: str,
        meeting_context: Optional[str] = None,
        user_id: str = None,
        analysis_type: str = "post_meeting"
    ) -> Dict[str, Any]:
        """Analyze meeting audio using LLAVA model"""
        try:
            analysis_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            # Step 1: Transcribe audio
            logger.info("Starting audio transcription...")
            transcript = await self._transcribe_audio(audio_file_path)
            
            # Step 2: Analyze with LLAVA model
            logger.info("Analyzing transcript with LLAVA...")
            llava_analysis = await self._analyze_with_llava(
                transcript, 
                meeting_context,
                analysis_type
            )
            
            # Step 3: Extract structured data
            key_points = self._extract_key_points(llava_analysis)
            action_items = self._extract_action_items(llava_analysis)
            
            # Step 4: Sentiment analysis
            sentiment = self._analyze_sentiment(transcript)
            
            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(llava_analysis, sentiment)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Prepare response
            analysis_result = {
                "analysis_id": analysis_id,
                "summary": llava_analysis.get("summary", ""),
                "key_points": key_points,
                "action_items": action_items,
                "sentiment_analysis": sentiment,
                "speaker_insights": llava_analysis.get("speaker_insights", {}),
                "recommendations": recommendations,
                "confidence_score": llava_analysis.get("confidence_score", 0.8),
                "analysis_timestamp": datetime.utcnow(),
                "processing_time": processing_time
            }
            
            # Store analysis results
            await self._store_analysis_results(analysis_result, user_id)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing meeting audio: {str(e)}")
            raise

    async def generate_summary(
        self,
        content: str,
        summary_type: SummaryType,
        user_id: str,
        meeting_id: Optional[str] = None,
        include_action_items: bool = True
    ) -> Dict[str, Any]:
        """Generate summary using LLAVA model"""
        try:
            summary_id = str(uuid.uuid4())
            
            # Prepare prompt for LLAVA
            prompt = self._build_summary_prompt(content, summary_type, include_action_items)
            
            # Generate summary with LLAVA
            llava_response = await self._call_llava_api(prompt, content)
            
            # Parse structured response
            summary_data = self._parse_llava_summary_response(llava_response)
            
            # Calculate metrics
            word_count = len(content.split())
            summary_ratio = len(summary_data["summary_text"].split()) / word_count if word_count > 0 else 0
            
            # Prepare response
            summary_result = {
                "summary_id": summary_id,
                "meeting_id": meeting_id,
                "summary_type": summary_type,
                "summary_text": summary_data["summary_text"],
                "key_points": summary_data.get("key_points", []),
                "action_items": summary_data.get("action_items", []),
                "next_steps": summary_data.get("next_steps", []),
                "participants": summary_data.get("participants", []),
                "topics_discussed": summary_data.get("topics_discussed", []),
                "decisions_made": summary_data.get("decisions_made", []),
                "questions_raised": summary_data.get("questions_raised", []),
                "meeting_effectiveness_score": summary_data.get("effectiveness_score"),
                "created_at": datetime.utcnow(),
                "word_count": word_count,
                "summary_ratio": summary_ratio
            }
            
            # Store summary
            await self._store_summary(summary_result, user_id)
            
            return summary_result
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

    async def real_time_audio_analysis(
        self,
        audio_chunk_path: str,
        meeting_context: Optional[str] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Real-time analysis of audio chunks during ongoing meeting"""
        try:
            # Transcribe audio chunk
            chunk_transcript = await self._transcribe_audio_chunk(audio_chunk_path)
            
            # Quick analysis for real-time insights
            quick_analysis = await self._quick_llava_analysis(chunk_transcript, meeting_context)
            
            # Generate real-time insights
            insights = {
                "current_topic": quick_analysis.get("current_topic"),
                "key_insight": quick_analysis.get("key_insight"),
                "action_item_detected": quick_analysis.get("new_action_item"),
                "sentiment_shift": quick_analysis.get("sentiment_change"),
                "engagement_level": quick_analysis.get("engagement_level"),
                "suggested_response": quick_analysis.get("suggested_response", [])
            }
            
            return {
                "analysis_id": str(uuid.uuid4()),
                "summary": quick_analysis.get("brief_summary", ""),
                "key_points": [KeyPoint(point=insights["key_insight"], category="real-time")] if insights["key_insight"] else [],
                "action_items": [insights["action_item_detected"]] if insights["action_item_detected"] else [],
                "sentiment_analysis": {"current_sentiment": insights["sentiment_shift"]},
                "recommendations": insights.get("suggested_response", []),
                "confidence_score": quick_analysis.get("confidence", 0.7),
                "analysis_timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error in real-time analysis: {str(e)}")
            raise

    async def _transcribe_audio(self, audio_file_path: str) -> str:
        """Transcribe audio file to text"""
        try:
            # Convert audio to wav format for better compatibility using CV2-based method
            wav_path = audio_file_path.replace(Path(audio_file_path).suffix, '.wav')
            
            # Convert using subprocess/ffmpeg instead of pydub
            if not self._convert_audio_to_wav_cv2(audio_file_path, wav_path):
                # If conversion fails, try to use the original file
                wav_path = audio_file_path
            
            # Transcribe using speech recognition
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                transcript = self.recognizer.recognize_google(audio_data)
            
            # Clean up temporary file if it was created
            if wav_path != audio_file_path and os.path.exists(wav_path):
                os.remove(wav_path)
            
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise

    async def _analyze_with_llava(
        self, 
        transcript: str, 
        context: Optional[str] = None,
        analysis_type: str = "post_meeting"
    ) -> Dict[str, Any]:
        """Analyze transcript using LLAVA model"""
        try:
            prompt = f"""
            Analyze this meeting transcript and provide structured insights:
            
            Context: {context or 'General meeting analysis'}
            Analysis Type: {analysis_type}
            
            Transcript: {transcript}
            
            Please provide:
            1. A brief summary (2-3 sentences)
            2. Key discussion points
            3. Action items with assignees if mentioned
            4. Decisions made
            5. Questions raised
            6. Speaker insights (if multiple speakers detected)
            7. Overall meeting sentiment
            8. Confidence score (0.0-1.0)
            
            Format your response as structured JSON.
            """
            
            response = await self._call_llava_api(prompt, transcript)
            return response
            
        except Exception as e:
            logger.error(f"Error with LLAVA analysis: {str(e)}")
            raise

    async def _call_llava_api(self, prompt: str, content: str) -> Dict[str, Any]:
        """Make API call to LLAVA model"""
        try:
            headers = {
                "Authorization": f"Bearer {self.llava_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.llava_config.model_version,
                "prompt": prompt,
                "content": content,
                "temperature": self.llava_config.temperature,
                "max_tokens": self.llava_config.max_tokens
            }
            
            response = requests.post(
                self.llava_api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error calling LLAVA API: {str(e)}")
            # Fallback to local processing if API fails
            return await self._fallback_analysis(content)

    async def _fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback analysis when LLAVA API is unavailable"""
        try:
            # Simple keyword-based analysis as fallback
            sentences = content.split('.')
            summary = '. '.join(sentences[:3]) + '.'
            
            # Extract potential action items
            action_words = ['will', 'should', 'need to', 'action', 'todo', 'task']
            action_items = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in action_words):
                    action_items.append({
                        "task": sentence.strip(),
                        "assignee": None,
                        "deadline": None,
                        "priority": "medium"
                    })
            
            return {
                "summary": summary,
                "key_points": sentences[:5],
                "action_items": action_items[:3],
                "confidence_score": 0.6,
                "speaker_insights": {},
                "decisions_made": [],
                "questions_raised": []
            }
            
        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}")
            return {
                "summary": "Analysis unavailable",
                "key_points": [],
                "action_items": [],
                "confidence_score": 0.0
            }

    def _extract_key_points(self, analysis: Dict[str, Any]) -> List[KeyPoint]:
        """Extract key points from LLAVA analysis"""
        key_points = []
        
        try:
            raw_points = analysis.get("key_points", [])
            for i, point in enumerate(raw_points):
                if isinstance(point, str):
                    key_points.append(KeyPoint(
                        point=point,
                        category="discussion",
                        importance="medium"
                    ))
                elif isinstance(point, dict):
                    key_points.append(KeyPoint(**point))
        except Exception as e:
            logger.error(f"Error extracting key points: {str(e)}")
        
        return key_points

    def _extract_action_items(self, analysis: Dict[str, Any]) -> List[ActionItem]:
        """Extract action items from LLAVA analysis"""
        action_items = []
        
        try:
            raw_items = analysis.get("action_items", [])
            for item in raw_items:
                if isinstance(item, str):
                    action_items.append(ActionItem(
                        task=item,
                        priority="medium",
                        status="pending"
                    ))
                elif isinstance(item, dict):
                    action_items.append(ActionItem(**item))
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
        
        return action_items

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the meeting"""
        try:
            # Split text into chunks for analysis
            chunks = [text[i:i+512] for i in range(0, len(text), 512)]
            sentiments = []
            
            for chunk in chunks[:5]:  # Analyze first 5 chunks
                result = self.sentiment_analyzer(chunk)[0]
                sentiments.append(result)
            
            # Calculate overall sentiment
            positive_count = sum(1 for s in sentiments if s['label'] == 'LABEL_2')
            negative_count = sum(1 for s in sentiments if s['label'] == 'LABEL_0')
            neutral_count = sum(1 for s in sentiments if s['label'] == 'LABEL_1')
            
            total = len(sentiments)
            overall_sentiment = "neutral"
            if positive_count / total > 0.6:
                overall_sentiment = "positive"
            elif negative_count / total > 0.6:
                overall_sentiment = "negative"
            
            return {
                "overall_sentiment": overall_sentiment,
                "positive_ratio": positive_count / total,
                "negative_ratio": negative_count / total,
                "neutral_ratio": neutral_count / total,
                "sentiment_details": sentiments
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"overall_sentiment": "neutral", "error": str(e)}

    def _generate_recommendations(self, analysis: Dict[str, Any], sentiment: Dict[str, Any]) -> List[str]:
        """Generate AI recommendations based on analysis"""
        recommendations = []
        
        try:
            # Sentiment-based recommendations
            if sentiment.get("overall_sentiment") == "negative":
                recommendations.append("• Consider addressing concerns raised during the meeting")
                recommendations.append("• Follow up with participants to ensure clarity on decisions")
            
            # Action items recommendations
            action_items = analysis.get("action_items", [])
            if len(action_items) > 5:
                recommendations.append("• Consider breaking down complex action items into smaller tasks")
            
            if not action_items:
                recommendations.append("• Define clear action items and assign ownership")
            
            # Meeting effectiveness recommendations
            confidence = analysis.get("confidence_score", 0)
            if confidence < 0.7:
                recommendations.append("• Improve meeting structure for better clarity")
                recommendations.append("• Ensure all participants are actively engaged")
            
            # Default recommendations
            recommendations.extend([
                "• Send meeting summary to all participants within 24 hours",
                "• Schedule follow-up meetings for unresolved items",
                "• Update project tracking tools with new action items"
            ])
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def _build_summary_prompt(self, content: str, summary_type: SummaryType, include_action_items: bool) -> str:
        """Build prompt for LLAVA summary generation"""
        base_prompt = f"""
        Create a {summary_type.value} summary of this meeting content:
        
        Requirements:
        - Provide key points in bullet format
        - Include main decisions made
        - Highlight important discussions
        """
        
        if include_action_items:
            base_prompt += "\n- Extract and list all action items with assignees if mentioned"
        
        if summary_type == SummaryType.DETAILED:
            base_prompt += "\n- Provide comprehensive coverage of all topics discussed"
        elif summary_type == SummaryType.BRIEF:
            base_prompt += "\n- Keep summary concise and focus on most important points"
        elif summary_type == SummaryType.ACTION_ITEMS:
            base_prompt += "\n- Focus primarily on action items and next steps"
        
        base_prompt += "\n\nFormat response as structured JSON with clear sections."
        
        return base_prompt

    def _parse_llava_summary_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLAVA response into structured summary data"""
        try:
            # Handle different response formats
            if "choices" in response:
                content = response["choices"][0]["message"]["content"]
            else:
                content = response.get("content", str(response))
            
            # Try to parse as JSON first
            try:
                if isinstance(content, str):
                    parsed = json.loads(content)
                else:
                    parsed = content
            except json.JSONDecodeError:
                # Fallback to text parsing
                parsed = self._parse_text_response(content)
            
            return {
                "summary_text": parsed.get("summary", content if isinstance(content, str) else ""),
                "key_points": parsed.get("key_points", []),
                "action_items": self._format_action_items(parsed.get("action_items", [])),
                "next_steps": parsed.get("next_steps", []),
                "participants": parsed.get("participants", []),
                "topics_discussed": parsed.get("topics_discussed", []),
                "decisions_made": parsed.get("decisions_made", []),
                "questions_raised": parsed.get("questions_raised", []),
                "effectiveness_score": parsed.get("effectiveness_score")
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLAVA response: {str(e)}")
            return {"summary_text": "Error parsing response", "key_points": []}

    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse plain text response when JSON parsing fails"""
        lines = text.split('\n')
        result = {"summary": "", "key_points": [], "action_items": []}
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "summary" in line.lower():
                current_section = "summary"
            elif "key point" in line.lower() or "bullet" in line.lower():
                current_section = "key_points"
            elif "action" in line.lower():
                current_section = "action_items"
            elif line.startswith('•') or line.startswith('-'):
                if current_section == "key_points":
                    result["key_points"].append(line[1:].strip())
                elif current_section == "action_items":
                    result["action_items"].append(line[1:].strip())
            else:
                if current_section == "summary":
                    result["summary"] += line + " "
        
        return result

    def _format_action_items(self, raw_items: List[Any]) -> List[ActionItem]:
        """Format raw action items into ActionItem objects"""
        formatted_items = []
        
        for item in raw_items:
            if isinstance(item, str):
                formatted_items.append(ActionItem(
                    task=item,
                    priority="medium",
                    status="pending"
                ))
            elif isinstance(item, dict):
                formatted_items.append(ActionItem(
                    task=item.get("task", ""),
                    assignee=item.get("assignee"),
                    deadline=item.get("deadline"),
                    priority=item.get("priority", "medium"),
                    status=item.get("status", "pending")
                ))
        
        return formatted_items

    def _trim_audio_chunk_cv2(self, audio_file_path: str, max_duration: int = 30) -> str:
        """Trim audio chunk to specified duration using subprocess/ffmpeg"""
        try:
            output_path = audio_file_path.replace('.', f'_trimmed.')
            
            result = subprocess.run([
                'ffmpeg', '-i', audio_file_path, '-t', str(max_duration),
                '-acodec', 'copy', '-y', output_path
            ], capture_output=True, stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                return output_path
            else:
                return audio_file_path
                
        except Exception as e:
            logger.warning(f"Error trimming audio chunk: {str(e)}")
            return audio_file_path

    async def _transcribe_audio_chunk(self, audio_chunk_path: str) -> str:
        """Transcribe smaller audio chunk for real-time analysis"""
        try:
            # Trim chunk for faster processing using CV2-based method
            trimmed_path = self._trim_audio_chunk_cv2(audio_chunk_path, 30)  # 30 seconds max
            
            # Convert to wav if needed
            wav_path = trimmed_path.replace(Path(trimmed_path).suffix, '_chunk.wav')
            
            if not self._convert_audio_to_wav_cv2(trimmed_path, wav_path):
                wav_path = trimmed_path
            
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                transcript = self.recognizer.recognize_google(audio_data)
            
            # Clean up temporary files
            if wav_path != trimmed_path and os.path.exists(wav_path):
                os.remove(wav_path)
            if trimmed_path != audio_chunk_path and os.path.exists(trimmed_path):
                os.remove(trimmed_path)
                
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio chunk: {str(e)}")
            return ""

    async def _quick_llava_analysis(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Quick LLAVA analysis for real-time insights"""
        try:
            prompt = f"""
            Provide quick real-time insights from this meeting segment:
            Context: {context or 'Ongoing meeting'}
            
            Text: {text}
            
            Provide brief insights on:
            1. Current topic being discussed
            2. Any new action items mentioned
            3. Sentiment change
            4. Engagement level
            5. Suggested responses or questions
            
            Keep response concise for real-time processing.
            """
            
            # Use faster, simpler processing for real-time
            response = await self._call_llava_api(prompt, text[:500])  # Limit text for speed
            return response
            
        except Exception as e:
            logger.error(f"Error in quick LLAVA analysis: {str(e)}")
            return {"brief_summary": "Real-time analysis unavailable"}

    async def _store_analysis_results(self, results: Dict[str, Any], user_id: str):
        """Store analysis results in database"""
        try:
            async with get_db() as conn:
                await conn.execute("""
                    INSERT INTO meeting_analysis (
                        analysis_id, user_id, summary, key_points, action_items,
                        sentiment_analysis, recommendations, confidence_score,
                        created_at, processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    results["analysis_id"],
                    user_id,
                    results["summary"],
                    json.dumps([kp.dict() for kp in results["key_points"]]),
                    json.dumps([ai.dict() for ai in results["action_items"]]),
                    json.dumps(results["sentiment_analysis"]),
                    json.dumps(results["recommendations"]),
                    results["confidence_score"],
                    results["analysis_timestamp"],
                    results["processing_time"]
                ))
                await conn.commit()
        except Exception as e:
            logger.error(f"Error storing analysis results: {str(e)}")

    async def _store_summary(self, summary: Dict[str, Any], user_id: str):
        """Store summary in database"""
        try:
            async with get_db() as conn:
                await conn.execute("""
                    INSERT INTO meeting_summaries (
                        summary_id, meeting_id, user_id, summary_type, summary_text,
                        key_points, action_items, next_steps, topics_discussed,
                        decisions_made, word_count, summary_ratio, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    summary["summary_id"],
                    summary.get("meeting_id"),
                    user_id,
                    summary["summary_type"].value,
                    summary["summary_text"],
                    json.dumps(summary["key_points"]),
                    json.dumps([ai.dict() for ai in summary["action_items"]]),
                    json.dumps(summary["next_steps"]),
                    json.dumps(summary["topics_discussed"]),
                    json.dumps(summary["decisions_made"]),
                    summary["word_count"],
                    summary["summary_ratio"],
                    summary["created_at"]
                ))
                await conn.commit()
        except Exception as e:
            logger.error(f"Error storing summary: {str(e)}")

    async def get_meeting_summary(self, meeting_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get existing summary for a meeting"""
        try:
            async with get_db() as conn:
                cursor = await conn.execute("""
                    SELECT * FROM meeting_summaries 
                    WHERE meeting_id = ? AND user_id = ?
                    ORDER BY created_at DESC LIMIT 1
                """, (meeting_id, user_id))
                
                row = await cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting meeting summary: {str(e)}")
            return None

    async def get_user_summaries(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all summaries for a user"""
        try:
            async with get_db() as conn:
                cursor = await conn.execute("""
                    SELECT * FROM meeting_summaries 
                    WHERE user_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (user_id, limit, offset))
                
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting user summaries: {str(e)}")
            return []

    async def delete_meeting_summary(self, meeting_id: str, user_id: str) -> bool:
        """Delete summary for a meeting"""
        try:
            async with get_db() as conn:
                cursor = await conn.execute("""
                    DELETE FROM meeting_summaries 
                    WHERE meeting_id = ? AND user_id = ?
                """, (meeting_id, user_id))
                await conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting summary: {str(e)}")
            return False