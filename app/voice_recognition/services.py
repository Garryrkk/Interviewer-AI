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
from pathlib import Path
import httpx
import speech_recognition as sr
import pyaudio
import numpy as np
import librosa
from pydub import AudioSegment
import tempfile
import os

try:
    import sounddevice as sd
    import soundfile as sf
    import librosa
    import whisper
    from scipy import signal
    from scipy.io import wavfile
except ImportError as e:
    logging.warning(f"Some audio libraries not available: {e}")


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
    VoiceProcessingConfig,
    CalibrationResponse,
    AudioTestResponse, 
    TranscriptionResponse,
    TranscriptionSegment,
    CalibrationStatus,
    ServiceHealth
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

    
class AudioService:
    """
    Service class handling audio calibration, recording, and transcription operations.
    """
    
    def __init__(self):
        self.calibration_data = {}
        self.test_recordings = {}
        self.whisper_model = None
        self.temp_dir = tempfile.mkdtemp()
        self.service_start_time = time.time()
        
        # Default audio settings
        self.default_sample_rate = 16000
        self.default_channels = 1
        self.max_recording_duration = 300  # 5 minutes
        
        # Initialize Whisper model lazily
        self._model_loaded = False
        
        logger.info(f"AudioService initialized. Temp dir: {self.temp_dir}")

    async def _load_whisper_model(self, model_size: str = "base"):
        """Load Whisper model if not already loaded."""
        if not self._model_loaded or (self.whisper_model and model_size not in str(self.whisper_model)):
            try:
                logger.info(f"Loading Whisper model: {model_size}")
                self.whisper_model = whisper.load_model(model_size)
                self._model_loaded = True
                logger.info(f"Whisper model {model_size} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise Exception(f"Failed to load speech recognition model: {e}")

    async def calibrate_audio(
        self,
        duration: int = 3,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> CalibrationResponse:
        """
        Perform audio calibration by measuring background noise levels.
        """
        try:
            logger.info(f"Starting calibration: {duration}s, {sample_rate}Hz, {channels}ch")
            
            # Record ambient noise
            logger.info("Recording ambient noise for calibration...")
            audio_data = sd.rec(
                int(duration * sample_rate), 
                samplerate=sample_rate, 
                channels=channels,
                dtype='float32'
            )
            sd.wait()  # Wait for recording to complete
            
            # Analyze audio data
            audio_flat = audio_data.flatten()
            
            # Calculate noise metrics
            rms = np.sqrt(np.mean(audio_flat ** 2))
            noise_level_db = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)
            
            # Calculate spectral characteristics
            freqs, psd = signal.welch(audio_flat, sample_rate, nperseg=1024)
            dominant_freq = freqs[np.argmax(psd)]
            
            # Determine quality score and recommendations
            quality_score = self._calculate_quality_score(noise_level_db, psd)
            recommendations = self._generate_recommendations(noise_level_db, quality_score)
            
            # Set recommended threshold (typically 10-15 dB above noise floor)
            recommended_threshold = noise_level_db + 12.0
            
            # Store calibration data
            calibration_data = {
                'noise_level': noise_level_db,
                'recommended_threshold': recommended_threshold,
                'sample_rate': sample_rate,
                'channels': channels,
                'dominant_frequency': dominant_freq,
                'quality_score': quality_score,
                'calibration_time': datetime.now(),
                'audio_statistics': {
                    'rms': float(rms),
                    'peak': float(np.max(np.abs(audio_flat))),
                    'mean': float(np.mean(audio_flat)),
                    'std': float(np.std(audio_flat))
                }
            }
            
            self.calibration_data = calibration_data
            
            logger.info(f"Calibration completed. Noise level: {noise_level_db:.2f}dB")
            
            return CalibrationResponse(
                status=CalibrationStatus.CALIBRATED,
                noise_level=noise_level_db,
                recommended_threshold=recommended_threshold,
                sample_rate=sample_rate,
                channels=channels,
                calibration_time=calibration_data['calibration_time'],
                quality_score=quality_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise Exception(f"Audio calibration failed: {e}")

    def _calculate_quality_score(self, noise_level_db: float, psd: np.ndarray) -> float:
        """Calculate audio environment quality score (0-1)."""
        # Better score for lower noise levels
        noise_score = max(0, min(1, (noise_level_db + 60) / 40))  # -60dB = 1.0, -20dB = 0.0
        
        # Check for consistent spectrum (avoid harsh frequency spikes)
        psd_normalized = psd / np.max(psd)
        spectral_flatness = np.exp(np.mean(np.log(psd_normalized + 1e-10))) / np.mean(psd_normalized)
        spectral_score = min(1.0, spectral_flatness * 2)  # Higher flatness = better
        
        # Combine scores
        quality_score = (noise_score * 0.7 + spectral_score * 0.3)
        return float(quality_score)

    def _generate_recommendations(self, noise_level_db: float, quality_score: float) -> List[str]:
        """Generate recommendations based on calibration results."""
        recommendations = []
        
        if noise_level_db > -30:
            recommendations.append("High background noise detected. Consider moving to a quieter environment.")
        elif noise_level_db > -45:
            recommendations.append("Moderate background noise. Audio quality may be affected.")
        else:
            recommendations.append("Good audio environment detected.")
            
        if quality_score < 0.5:
            recommendations.append("Consider using a higher quality microphone.")
            recommendations.append("Ensure microphone is positioned close to your mouth.")
        elif quality_score < 0.8:
            recommendations.append("Consider using a microphone closer to your mouth for better clarity.")
        else:
            recommendations.append("Excellent audio setup for speech recognition.")
            
        return recommendations

    async def test_recording(
        self,
        duration: int = 5,
        sample_rate: int = 16000,
        channels: int = 1,
        apply_calibration: bool = True
    ) -> AudioTestResponse:
        """
        Record a test audio clip and analyze its quality.
        """
        try:
            logger.info(f"Starting test recording: {duration}s")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='float32'
            )
            sd.wait()
            
            # Save to temporary file
            test_filename = f"test_recording_{int(time.time())}.wav"
            test_path = os.path.join(self.temp_dir, test_filename)
            
            sf.write(test_path, audio_data, sample_rate)
            actual_duration = len(audio_data) / sample_rate
            file_size = os.path.getsize(test_path)
            
            # Analyze audio quality
            audio_flat = audio_data.flatten()
            peak_amplitude = float(np.max(np.abs(audio_flat)))
            average_amplitude = float(np.mean(np.abs(audio_flat)))
            
            # Calculate SNR if calibration available
            snr = None
            if apply_calibration and self.calibration_data:
                noise_rms = np.sqrt(self.calibration_data.get('audio_statistics', {}).get('rms', 0.01) ** 2)
                signal_rms = np.sqrt(np.mean(audio_flat ** 2))
                if noise_rms > 0:
                    snr = 20 * np.log10(signal_rms / noise_rms)
            
            # Generate quality recommendations
            recommendations = []
            if peak_amplitude > 0.95:
                recommendations.append("Audio may be clipping. Reduce input volume.")
            elif peak_amplitude < 0.1:
                recommendations.append("Audio level is low. Speak louder or move closer to microphone.")
            else:
                recommendations.append("Audio quality is good for transcription.")
                
            if average_amplitude > 0.1:
                recommendations.append("Clear voice detected.")
            else:
                recommendations.append("Voice level is low. Ensure you're speaking clearly.")
            
            # Store test recording info
            self.test_recordings[test_filename] = {
                'path': test_path,
                'duration': actual_duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'timestamp': datetime.now()
            }
            
            audio_quality = {
                'bit_rate': 16,  # Using float32, equivalent to 16-bit
                'sample_rate': sample_rate,
                'channels': channels,
                'format': 'WAV'
            }
            
            return AudioTestResponse(
                success=True,
                duration=actual_duration,
                file_size=file_size,
                audio_quality=audio_quality,
                peak_amplitude=peak_amplitude,
                average_amplitude=average_amplitude,
                signal_to_noise_ratio=snr,
                recommendations=recommendations,
                audio_preview_url=f"/api/v1/audio/preview/{test_filename}"
            )
            
        except Exception as e:
            logger.error(f"Test recording failed: {e}")
            raise Exception(f"Test recording failed: {e}")

    async def transcribe_audio(
        self,
        audio_data: bytes,
        filename: str,
        content_type: str,
        language: str = "auto",
        model_size: str = "base"
    ) -> TranscriptionResponse:
        """
        Transcribe uploaded audio file to text.
        """
        try:
            start_time = time.time()
            
            # Load Whisper model
            await self._load_whisper_model(model_size)
            
            # Save uploaded file temporarily
            temp_filename = f"upload_{int(time.time())}_{filename}"
            temp_path = os.path.join(self.temp_dir, temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # Load and preprocess audio
            audio, sample_rate = librosa.load(temp_path, sr=16000)
            duration = len(audio) / sample_rate
            
            # Perform transcription
            logger.info("Starting Whisper transcription...")
            
            # Set language parameter
            language_param = None if language == "auto" else language
            
            result = self.whisper_model.transcribe(
                temp_path,
                language=language_param,
                word_timestamps=True,
                verbose=False
            )
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get('segments', []):
                segments.append(TranscriptionSegment(
                    start=segment['start'],
                    end=segment['end'],
                    text=segment['text'].strip(),
                    confidence=segment.get('avg_logprob', 0.0)  # Convert log prob to confidence-like score
                ))
            
            # Calculate overall confidence (average of segment confidences)
            overall_confidence = np.mean([seg.confidence for seg in segments]) if segments else 0.0
            overall_confidence = max(0.0, min(1.0, (overall_confidence + 1.0) / 2.0))  # Normalize log prob
            
            # Calculate audio quality score
            audio_quality_score = self._assess_audio_quality(audio, sample_rate)
            
            # Count words
            word_count = len(result['text'].split())
            
            processing_time = time.time() - start_time
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            
            return TranscriptionResponse(
                success=True,
                text=result['text'].strip(),
                language=result.get('language', language if language != "auto" else "unknown"),
                confidence=overall_confidence,
                duration=duration,
                segments=segments,
                word_count=word_count,
                processing_time=processing_time,
                model_used=model_size,
                audio_quality_score=audio_quality_score
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise Exception(f"Transcription failed: {e}")

    def _assess_audio_quality(self, audio: np.ndarray, sample_rate: int) -> float:
        """Assess audio quality for transcription purposes."""
        # Calculate various quality metrics
        
        # 1. Signal level
        rms = np.sqrt(np.mean(audio ** 2))
        signal_level_score = min(1.0, max(0.0, (rms - 0.01) / 0.1))
        
        # 2. Dynamic range
        dynamic_range = np.max(audio) - np.min(audio)
        dynamic_score = min(1.0, dynamic_range / 0.5)
        
        # 3. Spectral characteristics
        freqs, psd = signal.welch(audio, sample_rate, nperseg=1024)
        # Focus on speech frequency range (85-255 Hz fundamental, 2-4kHz formants)
        speech_band = (freqs >= 85) & (freqs <= 4000)
        speech_energy = np.sum(psd[speech_band])
        total_energy = np.sum(psd)
        speech_ratio = speech_energy / (total_energy + 1e-10)
        
        # 4. Clipping detection
        clipping_score = 1.0 - (np.sum(np.abs(audio) > 0.99) / len(audio))
        
        # Combine scores
        quality_score = (
            signal_level_score * 0.3 +
            dynamic_score * 0.2 +
            speech_ratio * 0.3 +
            clipping_score * 0.2
        )
        
        return float(min(1.0, max(0.0, quality_score)))

    async def transcribe_latest_test(self) -> Optional[TranscriptionResponse]:
        """
        Transcribe the most recent test recording.
        """
        if not self.test_recordings:
            return None
        
        # Get most recent test recording
        latest_test = max(
            self.test_recordings.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        test_filename, test_info = latest_test
        
        try:
            # Read the test file
            with open(test_info['path'], 'rb') as f:
                audio_data = f.read()
            
            # Transcribe using the existing method
            return await self.transcribe_audio(
                audio_data=audio_data,
                filename=test_filename,
                content_type="audio/wav",
                language="auto",
                model_size="base"
            )
            
        except Exception as e:
            logger.error(f"Failed to transcribe test recording: {e}")
            raise Exception(f"Failed to transcribe test recording: {e}")

    async def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get current calibration status and settings.
        """
        if not self.calibration_data:
            return {
                "is_calibrated": False,
                "last_calibration": None,
                "current_settings": None,
                "noise_level": None,
                "quality_score": None
            }
        
        return {
            "is_calibrated": True,
            "last_calibration": self.calibration_data['calibration_time'].isoformat(),
            "current_settings": {
                "sample_rate": self.calibration_data['sample_rate'],
                "channels": self.calibration_data['channels'],
                "recommended_threshold": self.calibration_data['recommended_threshold']
            },
            "noise_level": self.calibration_data['noise_level'],
            "quality_score": self.calibration_data['quality_score']
        }

    async def reset_calibration(self):
        """
        Reset calibration settings to defaults.
        """
        self.calibration_data = {}
        logger.info("Calibration settings reset")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the audio service.
        """
        uptime = time.time() - self.service_start_time
        
        # Check dependencies
        dependencies = {}
        
        # Check audio device availability
        try:
            devices = sd.query_devices()
            dependencies["audio_devices"] = "healthy" if len(devices) > 0 else "no_devices"
        except Exception:
            dependencies["audio_devices"] = "error"
        
        # Check Whisper model
        dependencies["whisper_model"] = "loaded" if self._model_loaded else "not_loaded"
        
        # Check temp directory
        dependencies["temp_directory"] = "healthy" if os.path.exists(self.temp_dir) else "error"
        
        # Overall status
        status = "healthy" if all(v in ["healthy", "loaded", "not_loaded"] for v in dependencies.values()) else "degraded"
        
        return ServiceHealth(
            status=status,
            version="1.0.0",
            uptime=uptime,
            dependencies=dependencies,
            last_check=datetime.now()
        ).dict()

    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Clean up old temporary files.
        """
        try:
            current_time = time.time()
            
            # Clean up test recordings
            for filename, info in list(self.test_recordings.items()):
                file_age = (datetime.now() - info['timestamp']).total_seconds() / 3600
                
                if file_age > max_age_hours:
                    try:
                        if os.path.exists(info['path']):
                            os.remove(info['path'])
                        del self.test_recordings[filename]
                        logger.info(f"Cleaned up old test recording: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {filename}: {e}")
            
            # Clean up other temp files
            if os.path.exists(self.temp_dir):
                for file_path in Path(self.temp_dir).glob("*"):
                    try:
                        file_age = (current_time - file_path.stat().st_mtime) / 3600
                        if file_age > max_age_hours:
                            file_path.unlink()
                            logger.info(f"Cleaned up temp file: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def get_audio_devices(self) -> Dict[str, Any]:
        """
        Get available audio input devices.
        """
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            
            return {
                'devices': input_devices,
                'default_device': sd.default.device[0] if sd.default.device[0] is not None else -1
            }
            
        except Exception as e:
            logger.error(f"Failed to get audio devices: {e}")
            return {'devices': [], 'default_device': -1, 'error': str(e)}

    async def set_audio_device(self, device_id: int):
        """
        Set the default audio input device.
        """
        try:
            devices = sd.query_devices()
            if 0 <= device_id < len(devices):
                sd.default.device[0] = device_id
                logger.info(f"Audio input device set to: {devices[device_id]['name']}")
                return True
            else:
                raise ValueError(f"Invalid device ID: {device_id}")
                
        except Exception as e:
            logger.error(f"Failed to set audio device: {e}")
            raise Exception(f"Failed to set audio device: {e}")

    def __del__(self):
        """
        Cleanup when service is destroyed.
        """
        try:
            # Clean up all temporary files
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info("Temporary directory cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")