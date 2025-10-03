import asyncio
import uuid
import json
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import speech_recognition as sr
try:
    import openai
except ImportError:
    openai = None
from transformers import pipeline
import mediapipe as mp
try:
    import librosa
except ImportError:
    librosa = None
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

# MediaPipe is the primary facial analysis library
# No need for face_recognition since we're using MediaPipe

from .schemas import *

logger = logging.getLogger(__name__)

class HandsFreeService:
    def __init__(self):
        # Active sessions storage
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # AI Services initialization
        self.speech_recognizer = sr.Recognizer()
        
        try:
            self.emotion_analyzer = pipeline("text-classification", 
                                            model="j-hartmann/emotion-english-distilroberta-base")
        except Exception as e:
            logger.warning(f"Could not load emotion analyzer: {e}")
            self.emotion_analyzer = None
            
        try:
            self.confidence_analyzer = pipeline("text-classification",
                                               model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception as e:
            logger.warning(f"Could not load confidence analyzer: {e}")
            self.confidence_analyzer = None
        
        # Audio processing
        self.audio_buffer_size = 4096
        self.sample_rate = 16000
        
        # Facial analysis - Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe drawing utilities for debugging (optional)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Fallback to Haar cascades for basic face detection if needed
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            logger.warning(f"Could not load face cascade: {e}")
            self.face_cascade = None
        
        # Thread pool for intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Interview response templates and insights
        self.response_templates = self._load_response_templates()
        self.confidence_tips_database = self._load_confidence_tips()
        
    async def create_session(self, user_id: str, default_mic_id: str, 
                           interview_type: InterviewType, company_info: Optional[str] = None,
                           job_role: Optional[str] = None) -> str:
        """Create a new hands-free interview session"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'default_mic_id': default_mic_id,
            'interview_type': interview_type,
            'company_info': company_info,
            'job_role': job_role,
            'status': SessionStatusEnum.INACTIVE,
            'created_at': datetime.utcnow(),
            'hands_free_active': False,
            'mic_configured': False,
            'ai_initialized': False,
            'audio_buffer': deque(maxlen=100),
            'conversation_history': [],
            'confidence_history': [],
            'facial_analysis_history': [],
            'response_times': [],
            'settings': SessionSettings(),
            'real_time_data': {
                'is_listening': False,
                'is_processing': False,
                'current_audio_level': 0.0,
                'last_question': None,
                'last_response': None
            }
        }
        
        self.active_sessions[session_id] = session_data
        logger.info(f"Created new session: {session_id} for user: {user_id}")
        
        return session_id
    
    async def configure_audio_input(self, session_id: str, mic_id: str) -> bool:
        """Automatically configure the default microphone"""
        async with self.session_locks[session_id]:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError("Session not found")
            
            try:
                # Configure speech recognizer with the specific microphone
                mic = sr.Microphone(device_index=int(mic_id) if mic_id.isdigit() else None)
                
                # Calibrate for ambient noise
                with mic as source:
                    self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
                
                session['mic_configured'] = True
                session['microphone'] = mic
                session['mic_id'] = mic_id
                
                logger.info(f"Microphone configured for session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to configure microphone: {str(e)}")
                session['mic_configured'] = False
                return False
    
    async def initialize_ai_systems(self, session_id: str) -> bool:
        """Initialize all AI systems for automated operation"""
        async with self.session_locks[session_id]:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError("Session not found")
            
            try:
                # Initialize response generation context
                session['ai_context'] = {
                    'interview_type': session['interview_type'],
                    'company_info': session['company_info'],
                    'job_role': session['job_role'],
                    'response_style': 'key_insights',
                    'confidence_threshold': 0.7
                }
                
                # Pre-load common interview patterns
                session['question_patterns'] = await self._load_question_patterns(
                    session['interview_type']
                )
                
                # Initialize facial analysis components
                session['facial_analyzer_ready'] = True
                
                session['ai_initialized'] = True
                logger.info(f"AI systems initialized for session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize AI systems: {str(e)}")
                session['ai_initialized'] = False
                return False
    
    async def activate_hands_free_mode(self, session_id: str) -> None:
        """Activate full hands-free automation"""
        async with self.session_locks[session_id]:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError("Session not found")
            
            if not session['mic_configured'] or not session['ai_initialized']:
                raise ValueError("Session not properly configured")
            
            session['hands_free_active'] = True
            session['status'] = SessionStatusEnum.HANDS_FREE_ACTIVE
            session['real_time_data']['is_listening'] = True
            
            # Start background monitoring tasks
            asyncio.create_task(self._monitor_audio_continuously(session_id))
            asyncio.create_task(self._monitor_confidence_continuously(session_id))
            
            logger.info(f"Hands-free mode activated for session {session_id}")
    
    async def process_audio_stream(self, session_id: str, audio_data: bytes) -> AudioStreamResult:
        """Process incoming audio stream and detect questions automatically"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        
        # Update audio level
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_level = np.abs(audio_array).mean() / 32768.0
        session['real_time_data']['current_audio_level'] = float(audio_level)
        
        # Add to audio buffer
        session['audio_buffer'].append(audio_data)
        
        result = AudioStreamResult(
            question_detected=False,
            is_listening=session['real_time_data']['is_listening'],
            is_processing=False,
            audio_level=audio_level,
            speech_clarity=0.0
        )
        
        # Process audio if enough data accumulated
        if len(session['audio_buffer']) >= 10:  # ~1 second of audio
            try:
                session['real_time_data']['is_processing'] = True
                
                # Combine audio chunks
                combined_audio = b''.join(list(session['audio_buffer']))
                
                # Convert to speech
                text = await self._audio_to_text(combined_audio)
                
                if text and len(text.strip()) > 10:
                    # Analyze if this is a question
                    is_question = await self._detect_question(text)
                    
                    if is_question:
                        result.question_detected = True
                        result.detected_question = text.strip()
                        result.context = await self._extract_context(text, session)
                        result.speech_clarity = await self._calculate_speech_clarity(text)
                        
                        # Store for response generation
                        session['real_time_data']['last_question'] = text.strip()
                        
                # Clear processed buffer
                session['audio_buffer'].clear()
                
            except Exception as e:
                logger.error(f"Audio processing error: {str(e)}")
            finally:
                session['real_time_data']['is_processing'] = False
        
        return result
    
    async def generate_automated_response(self, session_id: str, question: str, 
                                        context: Optional[str] = None) -> InterviewResponse:
        """Generate automated response in key insights format"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        
        start_time = datetime.utcnow()
        
        try:
            # Get response context
            ai_context = session['ai_context']
            conversation_history = session['conversation_history']
            
            # Generate response using AI
            response_text = await self._generate_ai_response(
                question=question,
                context=context,
                ai_context=ai_context,
                history=conversation_history[-5:]  # Last 5 exchanges
            )
            
            # Extract key insights from response
            key_insights = await self._extract_key_insights(response_text, question)
            
            # Calculate confidence score
            confidence_score = await self._calculate_response_confidence(
                response_text, question, ai_context
            )
            
            # Generate improvement suggestions
            suggestions = await self._generate_improvement_suggestions(
                response_text, confidence_score
            )
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            word_count = len(response_text.split())
            
            # Store in session history
            session['conversation_history'].append({
                'question': question,
                'response': response_text,
                'key_insights': [insight.dict() for insight in key_insights],
                'confidence_score': confidence_score,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            session['response_times'].append(response_time)
            session['real_time_data']['last_response'] = response_text
            
            return InterviewResponse(
                response_text=response_text,
                key_insights=key_insights,
                confidence_score=confidence_score,
                response_time=response_time,
                word_count=word_count,
                suggested_improvements=suggestions
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            # Fallback response
            return InterviewResponse(
                response_text="I need a moment to process that question. Could you please repeat it?",
                key_insights=[],
                confidence_score=0.5,
                response_time=0.1,
                word_count=12,
                suggested_improvements=["Technical issue occurred, please retry"]
            )
    
    async def analyze_facial_expression(self, session_id: str, frame_data: bytes) -> FacialAnalysis:
        """Automatically analyze facial expressions for confidence coaching"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        
        try:
            # Decode frame data
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Invalid frame data")
            
            # Try MediaPipe first, fallback to Haar cascades
            analysis = await self._analyze_facial_features(frame)
            
            # Store in session history
            session['facial_analysis_history'].append({
                'analysis': analysis.dict(),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Facial analysis error: {str(e)}")
            return FacialAnalysis(
                confidence_score=0.5,
                primary_emotion=EmotionState.CALM,
                eye_contact_score=0.5,
                posture_score=0.5,
                facial_expression_score=0.5,
                energy_level=0.5,
                stress_indicators=["Analysis error occurred"]
            )
    
    async def generate_confidence_tips(self, session_id: str, 
                                     analysis_result: FacialAnalysis) -> ConfidenceTipResponse:
        """Generate real-time confidence tips based on facial analysis"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        
        tips = []
        improvement_areas = []
        compliment = None
        
        # Analyze confidence level and generate appropriate tips
        confidence = analysis_result.confidence_score
        
        if confidence >= 0.8:
            compliment = "Excellent! You're showing great confidence and composure. Keep it up!"
            
            # Minor tips for already good performance
            if analysis_result.eye_contact_score < 0.9:
                tips.append(ConfidenceTip(
                    tip_type="eye_contact",
                    message="Great job! Try maintaining eye contact just a bit longer for maximum impact.",
                    priority="low",
                    immediate_action=False
                ))
                
        elif confidence >= 0.6:
            tips.extend([
                ConfidenceTip(
                    tip_type="confidence",
                    message="You're doing well! Take a deep breath and speak with conviction.",
                    priority="medium",
                    immediate_action=False
                ),
                ConfidenceTip(
                    tip_type="posture",
                    message="Straighten your shoulders slightly - good posture boosts confidence.",
                    priority="medium",
                    immediate_action=True
                )
            ])
            improvement_areas.append("Maintain steady confidence levels")
            
        else:  # confidence < 0.6
            tips.extend([
                ConfidenceTip(
                    tip_type="breathing",
                    message="Take a slow, deep breath. You've got this!",
                    priority="high",
                    immediate_action=True
                ),
                ConfidenceTip(
                    tip_type="posture",
                    message="Sit up straight and lean slightly forward to show engagement.",
                    priority="high",
                    immediate_action=True
                ),
                ConfidenceTip(
                    tip_type="expression",
                    message="Relax your facial muscles and let your natural personality shine.",
                    priority="medium",
                    immediate_action=False
                )
            ])
            improvement_areas.extend(["Confidence building", "Stress management"])
        
        # Eye contact specific tips
        if analysis_result.eye_contact_score < 0.5:
            tips.append(ConfidenceTip(
                tip_type="eye_contact",
                message="Look directly at the camera/interviewer - it shows confidence and engagement.",
                priority="high",
                immediate_action=True
            ))
            improvement_areas.append("Eye contact")
        
        # Energy level tips
        if analysis_result.energy_level < 0.4:
            tips.append(ConfidenceTip(
                tip_type="energy",
                message="Show more enthusiasm in your voice and expressions - energy is contagious!",
                priority="medium",
                immediate_action=False
            ))
            improvement_areas.append("Energy and enthusiasm")
        
        # Stress indicator handling
        if analysis_result.stress_indicators:
            tips.append(ConfidenceTip(
                tip_type="stress_management",
                message="I notice some tension. Remember, the interviewer wants you to succeed!",
                priority="high",
                immediate_action=True
            ))
        
        # Overall assessment
        if confidence >= 0.8:
            assessment = "Outstanding performance! You're demonstrating excellent confidence."
        elif confidence >= 0.6:
            assessment = "Good job! You're showing solid confidence with room for minor improvements."
        else:
            assessment = "You have great potential! Focus on the tips above to boost your confidence."
        
        return ConfidenceTipResponse(
            tips=tips,
            overall_assessment=assessment,
            compliment=compliment,
            improvement_areas=improvement_areas
        )
    
    async def get_session_status(self, session_id: str) -> SessionStatus:
        """Get comprehensive session status"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        
        uptime = (datetime.utcnow() - session['created_at']).total_seconds() / 60
        avg_response_time = np.mean(session['response_times']) if session['response_times'] else 0.0
        
        # Extract confidence trend from history
        confidence_scores = [
            entry.get('confidence_score', 0.5) 
            for entry in session['conversation_history']
        ]
        
        return SessionStatus(
            session_id=session_id,
            status=session['status'],
            hands_free_active=session['hands_free_active'],
            mic_status="active" if session['mic_configured'] else "inactive",
            ai_systems_status="ready" if session['ai_initialized'] else "not_ready",
            uptime=uptime,
            questions_answered=len(session['conversation_history']),
            avg_response_time=avg_response_time,
            confidence_trend=confidence_scores[-10:],  # Last 10 scores
            last_activity=datetime.utcnow()
        )
    
    async def update_session_settings(self, session_id: str, settings: SessionSettings) -> None:
        """Update session settings for automation"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        
        session['settings'] = settings
        
        # Apply settings immediately
        if not settings.auto_response_enabled:
            session['real_time_data']['is_listening'] = False
        
        logger.info(f"Updated settings for session {session_id}")
    
    async def get_session_insights(self, session_id: str) -> SessionInsights:
        """Get comprehensive session insights and analytics"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        
        duration = (datetime.utcnow() - session['created_at']).total_seconds() / 60
        conversation_history = session['conversation_history']
        facial_history = session['facial_analysis_history']
        
        # Calculate metrics
        avg_confidence = np.mean([
            entry.get('confidence_score', 0.5) 
            for entry in conversation_history
        ]) if conversation_history else 0.5
        
        confidence_progression = [
            entry.get('confidence_score', 0.5) 
            for entry in conversation_history
        ]
        
        # Identify best responses
        best_responses = sorted(
            conversation_history, 
            key=lambda x: x.get('confidence_score', 0), 
            reverse=True
        )[:3]
        
        best_response_texts = [resp['response'][:100] + "..." for resp in best_responses]
        
        # Analyze improvement areas
        improvement_areas = []
        if avg_confidence < 0.7:
            improvement_areas.append("Overall confidence building")
        if len([r for r in conversation_history if r.get('confidence_score', 0) < 0.6]) > len(conversation_history) * 0.3:
            improvement_areas.append("Consistency in responses")
        
        # Facial analysis summary
        facial_summary = {}
        if facial_history:
            facial_summary = {
                'avg_confidence': np.mean([f['analysis']['confidence_score'] for f in facial_history]),
                'avg_eye_contact': np.mean([f['analysis']['eye_contact_score'] for f in facial_history]),
                'most_common_emotion': max(set([f['analysis']['primary_emotion'] for f in facial_history]), key=[f['analysis']['primary_emotion'] for f in facial_history].count)
            }
        
        # Identify strengths
        strengths = []
        if avg_confidence >= 0.8:
            strengths.append("High confidence levels")
        if facial_summary.get('avg_eye_contact', 0) >= 0.7:
            strengths.append("Good eye contact")
        if len(conversation_history) >= 5:
            strengths.append("Strong engagement and participation")
        
        # Practice recommendations
        recommendations = []
        if avg_confidence < 0.7:
            recommendations.append("Practice responses to common interview questions")
        if facial_summary.get('avg_eye_contact', 0) < 0.6:
            recommendations.append("Practice maintaining eye contact during conversations")
        
        return SessionInsights(
            session_id=session_id,
            total_duration=duration,
            questions_answered=len(conversation_history),
            average_confidence=avg_confidence,
            best_responses=best_response_texts,
            improvement_areas=improvement_areas,
            confidence_progression=confidence_progression,
            facial_analysis_summary=facial_summary,
            key_strengths=strengths,
            recommended_practice_areas=recommendations,
            overall_score=avg_confidence
        )
    
    async def system_health_check(self) -> SystemHealthCheck:
        """Check overall system health for hands-free operation"""
        try:
            # Test microphone service
            mic_health = await self._test_microphone_service()
            
            # Test speech recognition
            speech_health = await self._test_speech_recognition()
            
            # Test AI response generation
            ai_health = await self._test_ai_response_service()
            
            # Test facial analysis
            facial_health = await self._test_facial_analysis()
            
            # Calculate metrics
            active_connections = len(self.active_sessions)
            avg_response_time = np.mean([
                np.mean(session.get('response_times', [0.5]))
                for session in self.active_sessions.values()
            ]) if self.active_sessions else 0.5
            
            overall_status = "healthy" if all([
                mic_health, speech_health, ai_health, facial_health
            ]) else "degraded"
            
            return SystemHealthCheck(
                overall_status=overall_status,
                microphone_service=mic_health,
                speech_recognition_service=speech_health,
                ai_response_service=ai_health,
                facial_analysis_service=facial_health,
                websocket_connections=active_connections,
                average_response_time=avg_response_time,
                error_count=0,
                last_health_check=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return SystemHealthCheck(
                overall_status="error",
                microphone_service=False,
                speech_recognition_service=False,
                ai_response_service=False,
                facial_analysis_service=False,
                websocket_connections=0,
                average_response_time=0.0,
                error_count=1,
                last_health_check=datetime.utcnow()
            )
    
    async def stop_session(self, session_id: str) -> SessionSummary:
        """Stop session and generate comprehensive summary"""
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError("Session not found")
        
        end_time = datetime.utcnow()
        start_time = session['created_at']
        total_duration = (end_time - start_time).total_seconds() / 60
        
        # Calculate hands-free uptime
        hands_free_uptime = total_duration  # Simplified - could track actual hands-free time
        
        # Calculate automation efficiency
        automation_efficiency = 0.9  # Simplified - could calculate based on manual interventions
        
        conversation_history = session['conversation_history']
        avg_quality = np.mean([
            entry.get('confidence_score', 0.5) 
            for entry in conversation_history
        ]) if conversation_history else 0.5
        
        # Confidence improvement
        confidence_scores = [entry.get('confidence_score', 0.5) for entry in conversation_history]
        confidence_improvement = 0.0
        if len(confidence_scores) >= 2:
            first_half = np.mean(confidence_scores[:len(confidence_scores)//2])
            second_half = np.mean(confidence_scores[len(confidence_scores)//2:])
            confidence_improvement = second_half - first_half
        
        summary = SessionSummary(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            hands_free_uptime=hands_free_uptime,
            automation_efficiency=automation_efficiency,
            questions_handled=len(conversation_history),
            average_response_quality=avg_quality,
            confidence_improvement=confidence_improvement,
            technical_performance={
                'avg_response_time': np.mean(session['response_times']) if session['response_times'] else 0.0,
                'total_audio_processed': len(session['audio_buffer']),
                'facial_analysis_count': len(session['facial_analysis_history'])
            },
            user_satisfaction_indicators=[]
        )
        
        # Cleanup session
        del self.active_sessions[session_id]
        if session_id in self.session_locks:
            del self.session_locks[session_id]
        
        logger.info(f"Session {session_id} stopped successfully")
        return summary
    
    # Helper methods
    async def _audio_to_text(self, audio_data: bytes) -> str:
        """Convert audio bytes to text using speech recognition"""
        try:
            # Convert bytes to audio data
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Use speech recognition
            audio_data_sr = sr.AudioData(audio_data, self.sample_rate, 2)
            text = self.speech_recognizer.recognize_google(audio_data_sr)
            
            return text
        except Exception as e:
            logger.debug(f"Speech recognition failed: {str(e)}")
            return ""
    
    async def _detect_question(self, text: str) -> bool:
        """Detect if text contains a question"""
        question_indicators = [
            '?', 'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'can you', 'could you', 'would you', 'tell me', 'explain'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators)
    
    async def _extract_context(self, text: str, session: Dict) -> str:
        """Extract relevant context for response generation"""
        context_parts = []
        
        # Add interview type context
        context_parts.append(f"Interview type: {session['interview_type']}")
        
        # Add company info if available
        if session['company_info']:
            context_parts.append(f"Company: {session['company_info']}")
        
        # Add job role if available
        if session['job_role']:
            context_parts.append(f"Role: {session['job_role']}")
        
        return " | ".join(context_parts)
    
    async def _calculate_speech_clarity(self, text: str) -> float:
        """Calculate speech clarity score based on text quality"""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        # Simple heuristic based on text length and coherence
        clarity_score = min(len(text.split()) / 10.0, 1.0)
        return clarity_score
    
    async def _generate_ai_response(self, question: str, context: Optional[str],
                                  ai_context: Dict, history: List[Dict]) -> str:
        """Generate AI response using OpenAI or similar service"""
        # This is a simplified version - in real implementation, use OpenAI API
        
        # Build prompt
        prompt_parts = [
            f"You are helping with a {ai_context['interview_type']} interview.",
            f"Question: {question}",
            "Generate a professional response in key insights format (2-3 main points)."
        ]
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        # Simplified response generation (replace with actual AI service)
        response_templates = {
            'technical': "Based on my experience: 1) I approach this systematically by analyzing requirements, 2) I implement best practices and clean code principles, 3) I ensure thorough testing and documentation.",
            'behavioral': "In my experience: 1) I believe in clear communication and collaboration, 2) I take ownership of challenges and learn from them, 3) I focus on delivering value while maintaining team harmony.",
            'general': "I'd highlight three key points: 1) My relevant experience aligns well with this role, 2) I'm passionate about contributing to meaningful projects, 3) I'm committed to continuous learning and growth."
        }
        
        return response_templates.get(ai_context['interview_type'], response_templates['general'])
    
    async def _extract_key_insights(self, response: str, question: str) -> List[KeyInsight]:
        """Extract key insights from generated response"""
        # Simple extraction based on numbered points
        insights = []
        sentences = response.split('.')
        
        for i, sentence in enumerate(sentences):
            if any(marker in sentence for marker in ['1)', '2)', '3)', '•', '-']):
                insights.append(KeyInsight(
                    point=sentence.strip(),
                    elaboration=None,
                    confidence_score=0.8
                ))
        
        # Fallback if no structured points found
        if not insights and response:
            insights.append(KeyInsight(
                point=response[:100] + "..." if len(response) > 100 else response,
                elaboration=None,
                confidence_score=0.7
            ))
        
        return insights[:3]  # Limit to 3 key insights
    
    async def _calculate_response_confidence(self, response: str, question: str, 
                                           ai_context: Dict) -> float:
        """Calculate confidence score for generated response"""
        # Simplified confidence calculation
        factors = []
        
        # Length factor (good responses are usually substantial)
        length_factor = min(len(response.split()) / 50.0, 1.0)
        factors.append(length_factor)
        
        # Structure factor (key insights format)
        structure_factor = 0.9 if any(marker in response for marker in ['1)', '2)', '3)']) else 0.6
        factors.append(structure_factor)
        
        # Relevance factor (simplified)
        relevance_factor = 0.8  # Would use NLP similarity in real implementation
        factors.append(relevance_factor)
        
        return np.mean(factors)
    
    async def _generate_improvement_suggestions(self, response: str, confidence_score: float) -> List[str]:
        """Generate suggestions for improving the response"""
        suggestions = []
        
        if confidence_score < 0.6:
            suggestions.append("Consider adding more specific examples")
            suggestions.append("Structure your response with clear key points")
        elif confidence_score < 0.8:
            suggestions.append("Great response! Consider adding a brief example to strengthen it")
        else:
            suggestions.append("Excellent response with clear structure and insights!")
        
        return suggestions
    
    async def _analyze_facial_features(self, frame, full_frame=None):
        """Analyze facial features using MediaPipe Face Mesh"""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark
                
                # Get frame dimensions
                h, w, _ = frame.shape
                
                # Extract key facial landmarks for analysis
                # Eye landmarks (MediaPipe face mesh indices)
                left_eye_center = landmarks[468]  # Left eye center
                right_eye_center = landmarks[473]  # Right eye center
                nose_tip = landmarks[1]  # Nose tip
                mouth_center = landmarks[13]  # Mouth center
                
                # Calculate eye contact score based on eye positioning
                # More centered eyes indicate better eye contact
                left_eye_x = left_eye_center.x * w
                right_eye_x = right_eye_center.x * w
                eye_center_x = (left_eye_x + right_eye_x) / 2
                frame_center_x = w / 2
                
                # Eye contact score (closer to center = better score)
                eye_deviation = abs(eye_center_x - frame_center_x) / (w / 2)
                eye_contact_score = max(0.0, 1.0 - eye_deviation)
                
                # Calculate facial expression score based on mouth positioning
                mouth_y = mouth_center.y * h
                nose_y = nose_tip.y * h
                mouth_nose_distance = abs(mouth_y - nose_y)
                
                # Normalize facial expression score
                facial_expression_score = min(1.0, mouth_nose_distance / (h * 0.1))
                
                # Calculate confidence score based on multiple factors
                eye_distance = abs(left_eye_x - right_eye_x)
                face_width_ratio = eye_distance / w
                
                # Confidence heuristic: better posture and positioning = higher confidence
                confidence_score = (eye_contact_score * 0.4 + 
                                  facial_expression_score * 0.3 + 
                                  face_width_ratio * 2.0 * 0.3)
                confidence_score = min(1.0, max(0.0, confidence_score))
                
                # Determine primary emotion based on facial features
                # This is simplified - in production, you'd use emotion detection models
                if facial_expression_score > 0.7:
                    primary_emotion = EmotionState.HAPPY
                elif confidence_score < 0.4:
                    primary_emotion = EmotionState.NERVOUS
                elif eye_contact_score > 0.7:
                    primary_emotion = EmotionState.FOCUSED
                else:
                    primary_emotion = EmotionState.CALM
                
                # Calculate energy level based on facial positioning
                energy_level = min(1.0, (confidence_score + eye_contact_score) / 2.0)
                
                # Detect stress indicators
                stress_indicators = []
                if eye_contact_score < 0.3:
                    stress_indicators.append("Poor eye contact")
                if confidence_score < 0.4:
                    stress_indicators.append("Low confidence posture")
                if face_width_ratio < 0.15:  # Face too small/far from camera
                    stress_indicators.append("Positioning issues")
                
                return FacialAnalysis(
                    confidence_score=confidence_score,
                    primary_emotion=primary_emotion,
                    eye_contact_score=eye_contact_score,
                    posture_score=min(1.0, face_width_ratio * 3.0),  # Based on face positioning
                    facial_expression_score=facial_expression_score,
                    energy_level=energy_level,
                    stress_indicators=stress_indicators
                )
                
        except Exception as e:
            logger.debug(f"MediaPipe facial analysis failed: {e}")
        
        # Fallback to basic face detection with Haar cascades
        try:
            if self.face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Basic analysis based on face detection
                    (x, y, w, h) = faces[0]
                    frame_h, frame_w, _ = frame.shape
                    
                    # Simple heuristics based on face size and position
                    face_size_ratio = (w * h) / (frame_w * frame_h)
                    center_deviation = abs((x + w/2) - frame_w/2) / (frame_w/2)
                    
                    confidence_score = min(1.0, face_size_ratio * 10.0)  # Larger face = more confident
                    eye_contact_score = max(0.0, 1.0 - center_deviation)  # Centered = better eye contact
                    
                    return FacialAnalysis(
                        confidence_score=confidence_score,
                        primary_emotion=EmotionState.CALM,
                        eye_contact_score=eye_contact_score,
                        posture_score=confidence_score,
                        facial_expression_score=0.6,
                        energy_level=0.6,
                        stress_indicators=[] if confidence_score > 0.5 else ["Low confidence detected"]
                    )
                    
        except Exception as e:
            logger.debug(f"Haar cascade analysis failed: {e}")
        
        # Final fallback - no face detected
        return FacialAnalysis(
            confidence_score=0.3,
            primary_emotion=EmotionState.CALM,
            eye_contact_score=0.3,
            posture_score=0.3,
            facial_expression_score=0.3,
            energy_level=0.3,
            stress_indicators=["No face detected or analysis failed"]
        )
    
    # Additional helper methods
    async def verify_session(self, session_id: str) -> bool:
        """Verify session exists and is active"""
        return session_id in self.active_sessions
    
    async def cleanup_session(self, session_id: str) -> None:
        """Cleanup session resources"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    async def emergency_pause(self, session_id: str) -> None:
        """Emergency pause hands-free mode"""
        session = self.active_sessions.get(session_id)
        if session:
            session['hands_free_active'] = False
            session['status'] = SessionStatusEnum.PAUSED
            session['real_time_data']['is_listening'] = False
    
    async def resume_hands_free(self, session_id: str) -> None:
        """Resume hands-free mode"""
        session = self.active_sessions.get(session_id)
        if session:
            session['hands_free_active'] = True
            session['status'] = SessionStatusEnum.HANDS_FREE_ACTIVE
            session['real_time_data']['is_listening'] = True
    
    def _load_response_templates(self) -> Dict:
        """Load response templates for different interview types"""
        return {
            'technical': [
                "Based on my experience with {technology}, I would approach this by...",
                "In my previous role, I handled similar challenges by...",
                "The key considerations for this problem would be..."
            ],
            'behavioral': [
                "In a similar situation, I demonstrated...",
                "My approach to team collaboration involves...",
                "When faced with challenges, I typically..."
            ],
            'general': [
                "I believe my experience in {field} makes me well-suited because...",
                "My career goal aligns with this role by...",
                "I'm particularly excited about this opportunity because..."
            ]
        }
    
    def _load_confidence_tips(self) -> Dict:
        """Load confidence tips database"""
        return {
            'posture': [
                "Sit up straight with shoulders back",
                "Lean slightly forward to show engagement",
                "Keep your feet flat on the floor"
            ],
            'eye_contact': [
                "Look directly at the camera/interviewer",
                "Maintain steady eye contact while speaking",
                "Avoid looking away frequently"
            ],
            'breathing': [
                "Take slow, deep breaths before responding",
                "Pause briefly to collect your thoughts",
                "Speak at a measured pace"
            ],
            'expression': [
                "Maintain a pleasant, professional expression",
                "Show enthusiasm through facial expressions",
                "Relax your facial muscles"
            ]
        }
    
    async def _load_question_patterns(self, interview_type: InterviewType) -> List:
        """Load common question patterns for the interview type"""
        patterns = {
            'technical': [
                "how would you implement",
                "explain the difference between",
                "what is your experience with",
                "walk me through",
                "design a system"
            ],
            'behavioral': [
                "tell me about a time",
                "describe a situation",
                "how do you handle",
                "give me an example",
                "what would you do if"
            ],
            'general': [
                "why do you want",
                "what interests you",
                "where do you see yourself",
                "what are your strengths",
                "why should we hire you"
            ]
        }
        return patterns.get(interview_type, patterns['general'])
    
    async def _monitor_audio_continuously(self, session_id: str) -> None:
        """Background task to monitor audio continuously"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        try:
            while session.get('hands_free_active', False):
                # Monitor audio levels and quality
                current_level = session['real_time_data'].get('current_audio_level', 0.0)
                
                # Log audio monitoring (simplified)
                if current_level > 0.1:  # Some audio detected
                    logger.debug(f"Audio level: {current_level:.2f} for session {session_id}")
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
        except Exception as e:
            logger.error(f"Audio monitoring error for session {session_id}: {e}")
    
    async def _monitor_confidence_continuously(self, session_id: str) -> None:
        """Background task to monitor confidence levels"""
        session = self.active_sessions.get(session_id)
        if not session:
            return
        
        try:
            while session.get('hands_free_active', False):
                # Monitor confidence trends
                history = session.get('confidence_history', [])
                
                # Add current confidence if available
                if session['conversation_history']:
                    latest = session['conversation_history'][-1]
                    confidence = latest.get('confidence_score', 0.5)
                    history.append({
                        'confidence': confidence,
                        'timestamp': datetime.utcnow()
                    })
                    
                    # Keep only recent history (last hour)
                    cutoff = datetime.utcnow() - timedelta(hours=1)
                    session['confidence_history'] = [
                        h for h in history 
                        if h['timestamp'] > cutoff
                    ]
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            logger.error(f"Confidence monitoring error for session {session_id}: {e}")
    
    async def _test_microphone_service(self) -> bool:
        """Test microphone service health"""
        try:
            # Try to list available microphones
            mic_list = sr.Microphone.list_microphone_names()
            return len(mic_list) > 0
        except Exception as e:
            logger.error(f"Microphone test failed: {e}")
            return False
    
    async def _test_speech_recognition(self) -> bool:
        """Test speech recognition service"""
        try:
            # Test speech recognizer initialization
            recognizer = sr.Recognizer()
            return recognizer is not None
        except Exception as e:
            logger.error(f"Speech recognition test failed: {e}")
            return False
    
    async def _test_ai_response_service(self) -> bool:
        """Test AI response generation service"""
        try:
            # Test response generation
            test_response = await self._generate_ai_response(
                question="Test question",
                context=None,
                ai_context={'interview_type': 'general'},
                history=[]
            )
            return len(test_response) > 0
        except Exception as e:
            logger.error(f"AI response test failed: {e}")
            return False
    
    async def _test_facial_analysis(self) -> bool:
        """Test facial analysis service"""
        try:
            # Test MediaPipe initialization
            return self.mp_face_mesh is not None
        except Exception as e:
            logger.error(f"Facial analysis test failed: {e}")
            return False