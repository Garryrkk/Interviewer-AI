import asyncio
import json
import logging
import uuid
import base64
import io
import wave
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import httpx
import speech_recognition as sr
import pyaudio
import numpy as np
import librosa
from pydub import AudioSegment
import tempfile
import os

from .schemas import (
    VoiceSession,
    AudioDevice,
    ProcessingResult,
    VoiceCharacteristics,
    AudioQuality,
    OllamaRequest,
    OllamaResponse,
    ResponseFormat,
    SimplificationLevel,
    VoiceProcessingConfig
)

logger = logging.getLogger(__name__)

class VoiceProcessingService:
    """Service class for handling voice recognition and processing operations"""
    
    def __init__(self, config: Optional[VoiceProcessingConfig] = None):
        self.config = config or VoiceProcessingConfig()
        self.active_sessions: Dict[str, VoiceSession] = {}
        self.recognizer = sr.Recognizer()
        self.microphone_instances: Dict[str, sr.Microphone] = {}
        
        # Initialize PyAudio for device enumeration
        self.audio = pyaudio.PyAudio()
        
        # HTTP client for Ollama API
        self.http_client = httpx.AsyncClient(
            base_url=self.config.ollama_base_url,
            timeout=60.0
        )
        
    async def create_session(self, user_id: str, meeting_id: Optional[str] = None) -> str:
        """Create a new voice processing session"""
        session_id = str(uuid.uuid4())
        
        session = VoiceSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            session_settings={
                "meeting_id": meeting_id,
                "auto_device_selection": True,
                "noise_reduction": True,
                "voice_activation": False
            }
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"Created voice session {session_id} for user {user_id}")
        
        # Schedule session cleanup
        asyncio.create_task(self._schedule_session_cleanup(session_id))
        
        return session_id
    
    async def check_microphone_status(self, session_id: str) -> Dict[str, Any]:
        """Check if microphone is available and enabled"""
        session = self._get_session(session_id)
        
        try:
            # Check if microphone access is available
            devices = await self.get_audio_devices(session_id)
            is_available = len(devices) > 0
            
            # Check current connection status
            is_enabled = session.microphone_enabled and session.current_device is not None
            
            message = ""
            if not is_available:
                message = "No microphone devices found. Please connect a microphone and refresh."
            elif not is_enabled:
                message = "Microphone is available but not connected. Click to enable."
            else:
                message = f"Microphone connected: {session.current_device.name}"
            
            return {
                "is_available": is_available,
                "is_enabled": is_enabled,
                "current_device": session.current_device,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Failed to check microphone status: {str(e)}")
            return {
                "is_available": False,
                "is_enabled": False,
                "current_device": None,
                "message": f"Error checking microphone: {str(e)}"
            }
    
    async def get_audio_devices(self, session_id: str) -> List[AudioDevice]:
        """Get list of available audio input devices"""
        self._get_session(session_id)  # Validate session
        
        devices = []
        
        try:
            # Get system default device info
            default_device_info = self.audio.get_default_input_device_info()
            default_device_id = str(default_device_info['index'])
            
            # Enumerate all input devices
            device_count = self.audio.get_device_count()
            
            for i in range(device_count):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    
                    # Only include input devices
                    if device_info['maxInputChannels'] > 0:
                        device = AudioDevice(
                            id=str(i),
                            name=device_info['name'],
                            is_default=(str(i) == default_device_id),
                            is_available=True,
                            device_type="microphone"
                        )
                        devices.append(device)
                        
                except Exception as e:
                    logger.warning(f"Could not get info for device {i}: {str(e)}")
                    continue
            
            # Sort devices with default first
            devices.sort(key=lambda x: (not x.is_default, x.name))
            
            logger.info(f"Found {len(devices)} audio input devices")
            return devices
            
        except Exception as e:
            logger.error(f"Failed to enumerate audio devices: {str(e)}")
            return []
    
    async def select_audio_device(self, session_id: str, device_id: str) -> Dict[str, Any]:
        """Select and connect to a specific audio device"""
        session = self._get_session(session_id)
        
        try:
            # Get device info
            device_index = int(device_id)
            device_info = self.audio.get_device_info_by_index(device_index)
            
            # Create AudioDevice object
            device = AudioDevice(
                id=device_id,
                name=device_info['name'],
                is_default=(device_index == self.audio.get_default_input_device_info()['index']),
                is_available=True,
                device_type="microphone"
            )
            
            # Test device connection
            try:
                test_mic = sr.Microphone(device_index=device_index)
                with test_mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Store microphone instance
                self.microphone_instances[session_id] = test_mic
                
                # Update session
                session.current_device = device
                session.microphone_enabled = True
                session.last_activity = datetime.now()
                
                logger.info(f"Connected to audio device: {device.name}")
                
                return {
                    "device_id": device_id,
                    "device_name": device.name,
                    "connection_status": "connected"
                }
                
            except Exception as device_error:
                logger.error(f"Failed to connect to device {device_id}: {str(device_error)}")
                raise Exception(f"Cannot connect to device '{device.name}': {str(device_error)}")
                
        except Exception as e:
            logger.error(f"Device selection failed: {str(e)}")
            raise Exception(f"Failed to select audio device: {str(e)}")
    
    async def disable_microphone(self, session_id: str):
        """Disable microphone for the session"""
        session = self._get_session(session_id)
        
        # Clean up microphone instance
        if session_id in self.microphone_instances:
            del self.microphone_instances[session_id]
        
        # Update session
        session.microphone_enabled = False
        session.current_device = None
        session.last_activity = datetime.now()
        
        logger.info(f"Microphone disabled for session {session_id}")
    
    async def process_audio(self, session_id: str, audio_data: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
        """Process audio data: analyze quality, transcribe, and prepare for AI response"""
        session = self._get_session(session_id)
        start_time = time.time()
        
        try:
            # Convert audio data to a temporary file
            temp_audio_path = await self._save_temp_audio(audio_data, filename)
            
            # Analyze audio quality
            audio_quality = await self._analyze_audio_quality(temp_audio_path)
            
            # Transcribe audio
            transcription_result = await self._transcribe_audio_file(temp_audio_path)
            
            # Clean up temporary file
            os.unlink(temp_audio_path)
            
            # Update session stats
            session.total_interactions += 1
            session.last_activity = datetime.now()
            
            processing_time = time.time() - start_time
            
            return {
                "transcription": transcription_result["text"],
                "confidence": transcription_result["confidence"],
                "audio_quality": audio_quality,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise Exception(f"Failed to process audio: {str(e)}")
    
    async def transcribe_audio(self, session_id: str, audio_data: Union[str, bytes]) -> Dict[str, Any]:
        """Transcribe audio to text using speech recognition"""
        self._get_session(session_id)  # Validate session
        
        try:
            # Handle base64 encoded audio
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data)
            
            # Save to temporary file
            temp_audio_path = await self._save_temp_audio(audio_data)
            
            # Transcribe
            result = await self._transcribe_audio_file(temp_audio_path)
            
            # Clean up
            os.unlink(temp_audio_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    async def generate_ai_response(self, session_id: str, question: str, response_format: ResponseFormat, context: Optional[str] = None) -> Dict[str, Any]:
        """Generate AI response using Ollama Nous Hermes model"""
        self._get_session(session_id)  # Validate session
        start_time = time.time()
        
        try:
            # Build prompt based on response format
            prompt = await self._build_ai_prompt(question, response_format, context)
            
            # Make request to Ollama
            response = await self._call_ollama_api(prompt)
            
            processing_time = time.time() - start_time
            
            return {
                "response": response["response"],
                "confidence": 0.85,  # Default confidence for Ollama responses
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"AI response generation failed: {str(e)}")
            raise Exception(f"Failed to generate AI response: {str(e)}")
    
    async def analyze_voice_characteristics(self, session_id: str, audio_data: Union[str, bytes]) -> Dict[str, Any]:
        """Analyze voice characteristics and provide confidence rating"""
        self._get_session(session_id)  # Validate session
        
        try:
            # Handle base64 encoded audio
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data)
            
            # Save to temporary file for analysis
            temp_audio_path = await self._save_temp_audio(audio_data)
            
            # Load audio for analysis
            y, sr = librosa.load(temp_audio_path, sr=None)
            
            # Extract voice characteristics
            characteristics = await self._extract_voice_features(y, sr)
            
            # Calculate confidence rating
            confidence_rating = await self._calculate_confidence_rating(characteristics)
            
            # Generate tips and recommendations
            tips = await self._generate_voice_tips(characteristics, confidence_rating)
            recommendations = await self._generate_recommendations(characteristics)
            
            # Clean up
            os.unlink(temp_audio_path)
            
            return {
                "confidence_rating": confidence_rating,
                "characteristics": characteristics,
                "tips": tips,
                "recommendations": recommendations,
                "summary": f"Voice confidence rating: {confidence_rating:.1f}/10. {tips[0] if tips else 'Good voice quality detected.'}"
            }
            
        except Exception as e:
            logger.error(f"Voice analysis failed: {str(e)}")
            raise Exception(f"Voice analysis failed: {str(e)}")
    
    async def simplify_response(self, session_id: str, original_response: str, simplification_level: SimplificationLevel) -> Dict[str, Any]:
        """Generate a simplified version of the AI response"""
        self._get_session(session_id)  # Validate session
        start_time = time.time()
        
        try:
            # Build simplification prompt
            prompt = await self._build_simplification_prompt(original_response, simplification_level)
            
            # Call Ollama API
            response = await self._call_ollama_api(prompt)
            
            processing_time = time.time() - start_time
            
            return {
                "response": response["response"],
                "confidence": 0.80,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Response simplification failed: {str(e)}")
            raise Exception(f"Failed to simplify response: {str(e)}")
    
    async def end_session(self, session_id: str):
        """End voice session and cleanup resources"""
        if session_id in self.active_sessions:
            # Clean up microphone instance
            if session_id in self.microphone_instances:
                del self.microphone_instances[session_id]
            
            # Remove session
            del self.active_sessions[session_id]
            logger.info(f"Ended voice session {session_id}")
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current session status"""
        session = self._get_session(session_id)
        
        duration = (datetime.now() - session.created_at).total_seconds()
        
        return {
            "active": session.is_active,
            "mic_connected": session.microphone_enabled,
            "current_device": session.current_device.dict() if session.current_device else None,
            "duration": duration,
            "interactions": session.total_interactions
        }
    
    # Private helper methods
    
    def _get_session(self, session_id: str) -> VoiceSession:
        """Get session by ID or raise exception"""
        if session_id not in self.active_sessions:
            raise Exception(f"Session {session_id} not found or expired")
        
        session = self.active_sessions[session_id]
        session.last_activity = datetime.now()
        return session
    
    async def _save_temp_audio(self, audio_data: bytes, filename: Optional[str] = None) -> str:
        """Save audio data to temporary file"""
        suffix = '.wav'
        if filename:
            _, ext = os.path.splitext(filename.lower())
            if ext in ['.mp3', '.m4a', '.ogg', '.webm']:
                suffix = ext
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    
    async def _analyze_audio_quality(self, audio_path: str) -> AudioQuality:
        """Analyze audio quality"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate quality metrics
            rms_energy = np.sqrt(np.mean(y**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Simple quality assessment
            if rms_energy > 0.01 and zero_crossing_rate < 0.3:
                return AudioQuality.EXCELLENT
            elif rms_energy > 0.005:
                return AudioQuality.GOOD
            elif rms_energy > 0.001:
                return AudioQuality.FAIR
            else:
                return AudioQuality.POOR
                
        except Exception:
            return AudioQuality.FAIR
    
    async def _transcribe_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file to text"""
        try:
            # Convert to WAV if necessary
            if not audio_path.lower().endswith('.wav'):
                audio = AudioSegment.from_file(audio_path)
                wav_path = audio_path.replace(os.path.splitext(audio_path)[1], '.wav')
                audio.export(wav_path, format='wav')
                audio_path = wav_path
            
            # Use speech recognition
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            
            try:
                # Try Google Speech Recognition first
                text = self.recognizer.recognize_google(audio)
                confidence = 0.85
            except sr.UnknownValueError:
                # Fallback to other engines or return empty
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    confidence = 0.70
                except:
                    text = "Could not understand audio"
                    confidence = 0.10
            except sr.RequestError:
                # Network error, try offline
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    confidence = 0.70
                except:
                    text = "Speech recognition service unavailable"
                    confidence = 0.10
            
            return {
                "text": text,
                "confidence": confidence,
                "language": "en",
                "duration": librosa.get_duration(filename=audio_path)
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return {
                "text": "Transcription failed",
                "confidence": 0.0,
                "language": "en",
                "duration": 0.0
            }
    
    async def _extract_voice_features(self, y: np.ndarray, sr: int) -> VoiceCharacteristics:
        """Extract voice characteristics from audio"""
        try:
            # Pitch/fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Volume/energy
            rms = librosa.feature.rms(y=y)[0]
            volume_mean = np.mean(rms)
            
            # Speech rate (approximate)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            speech_rate = len(onset_frames) * 60 / librosa.get_duration(y=y, sr=sr)  # per minute
            
            # Clarity (spectral centroid as proxy)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            clarity_score = min(np.mean(spectral_centroids) / 4000, 1.0)  # Normalize
            
            # Basic emotion indicators (simplified)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            energy_variance = np.var(rms)
            
            emotion_indicators = {
                "energy": float(volume_mean),
                "stability": float(1.0 - energy_variance),
                "expressiveness": float(np.var(pitch_mean) / 1000 if pitch_mean > 0 else 0)
            }
            
            return VoiceCharacteristics(
                pitch_average=float(pitch_mean),
                volume_level=float(min(volume_mean * 10, 1.0)),
                speech_rate=float(speech_rate),
                clarity_score=float(clarity_score),
                emotion_indicators=emotion_indicators
            )
            
        except Exception as e:
            logger.error(f"Voice feature extraction failed: {str(e)}")
            # Return default characteristics
            return VoiceCharacteristics(
                pitch_average=150.0,
                volume_level=0.5,
                speech_rate=120.0,
                clarity_score=0.7,
                emotion_indicators={"energy": 0.5, "stability": 0.7, "expressiveness": 0.6}
            )
    
    async def _calculate_confidence_rating(self, characteristics: VoiceCharacteristics) -> float:
        """Calculate voice confidence rating from characteristics"""
        try:
            # Scoring factors
            volume_score = min(characteristics.volume_level * 2, 1.0) * 2.5
            clarity_score = characteristics.clarity_score * 2.5
            stability_score = characteristics.emotion_indicators.get("stability", 0.5) * 2.0
            speech_rate_score = 3.0 if 80 <= characteristics.speech_rate <= 180 else 1.5
            
            total_score = volume_score + clarity_score + stability_score + speech_rate_score
            confidence_rating = min(total_score, 10.0)
            
            return round(confidence_rating, 1)
            
        except Exception:
            return 6.5  # Default moderate confidence
    
    async def _generate_voice_tips(self, characteristics: VoiceCharacteristics, rating: float) -> List[str]:
        """Generate situational tips based on voice analysis"""
        tips = []
        
        if characteristics.volume_level < 0.3:
            tips.append("Speak a bit louder to ensure your voice is clearly heard by others.")
        elif characteristics.volume_level > 0.8:
            tips.append("Consider speaking slightly softer for better audio balance.")
        
        if characteristics.speech_rate > 180:
            tips.append("Try speaking a bit slower to improve clarity and comprehension.")
        elif characteristics.speech_rate < 100:
            tips.append("You can speak a bit faster to maintain engagement.")
        
        if characteristics.clarity_score < 0.6:
            tips.append("Focus on clear articulation and proper pronunciation.")
        
        if rating < 5.0:
            tips.append("Take deep breaths and speak with confidence. You've got this!")
        elif rating > 8.0:
            tips.append("Great voice confidence! Keep maintaining this excellent level.")
        
        if not tips:
            tips.append("Your voice sounds clear and confident. Keep up the good work!")
        
        return tips[:3]  # Limit to top 3 tips
    
    async def _generate_recommendations(self, characteristics: VoiceCharacteristics) -> List[str]:
        """Generate recommendations for voice improvement"""
        recommendations = [
            "Practice deep breathing exercises before important conversations",
            "Stay hydrated to maintain vocal clarity",
            "Record yourself speaking to monitor your progress"
        ]
        
        if characteristics.volume_level < 0.4:
            recommendations.append("Practice projecting your voice from your diaphragm")
        
        if characteristics.speech_rate > 160:
            recommendations.append("Practice reading aloud at a measured pace")
        
        return recommendations[:4]
    
    async def _build_ai_prompt(self, question: str, response_format: ResponseFormat, context: Optional[str] = None) -> str:
        """Build prompt for AI response based on format"""
        base_prompt = f"Question: {question}\n\n"
        
        if context:
            base_prompt += f"Context: {context}\n\n"
        
        format_instructions = {
            ResponseFormat.SUMMARY: "Provide a concise summary response in 2-3 sentences.",
            ResponseFormat.KEY_INSIGHTS: "Provide the key insights as bullet points with brief explanations.",
            ResponseFormat.DETAILED: "Provide a comprehensive, detailed response with examples where helpful.",
            ResponseFormat.BULLET_POINTS: "Structure your response as clear, actionable bullet points."
        }
        
        instruction = format_instructions.get(response_format, format_instructions[ResponseFormat.SUMMARY])
        
        return base_prompt + instruction
    
    async def _build_simplification_prompt(self, original_response: str, level: SimplificationLevel) -> str:
        """Build prompt for response simplification"""
        level_instructions = {
            SimplificationLevel.BASIC: "Rewrite this in very simple terms that a beginner could easily understand. Use common words and short sentences.",
            SimplificationLevel.INTERMEDIATE: "Rewrite this in clearer, more accessible language while keeping the main points.",
            SimplificationLevel.ADVANCED: "Refine this response to be more concise while maintaining technical accuracy."
        }
        
        instruction = level_instructions.get(level, level_instructions[SimplificationLevel.BASIC])
        
        return f"Original response: {original_response}\n\nTask: {instruction}\n\nSimplified response:"
    
    async def _call_ollama_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to Ollama"""
        try:
            request_data = {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": self.config.ai_response_max_tokens
                }
            }
            
            response = await self.http_client.post("/api/generate", json=request_data)
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get("done", False):
                raise Exception("Ollama response incomplete")
            
            return {
                "response": result.get("response", "").strip(),
                "processing_time": result.get("total_duration", 0) / 1e9  # Convert to seconds
            }
            
        except httpx.RequestError as e:
            logger.error(f"Ollama API request failed: {str(e)}")
            raise Exception(f"AI service unavailable: {str(e)}")
        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            raise Exception(f"AI response generation failed: {str(e)}")
    
    async def _schedule_session_cleanup(self, session_id: str):
        """Schedule automatic session cleanup"""
        await asyncio.sleep(self.config.max_session_duration)
        
        if session_id in self.active_sessions:
            logger.info(f"Auto-cleaning up expired session {session_id}")
            await self.end_session(session_id)
    
    def __del__(self):
        """Cleanup PyAudio resources"""
        try:
            if hasattr(self, 'audio'):
                self.audio.terminate()
            if hasattr(self, 'http_client'):
                asyncio.create_task(self.http_client.aclose())
        except:
            pass