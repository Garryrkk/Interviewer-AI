import asyncio
import json
import base64
import aiohttp
import logging
from datetime import datetime
from typing import Optional, AsyncGenerator, Dict, Any
from .schemas import (
    QuickRespondRequest, 
    QuickRespondResponse, 
    SimplifyRequest,
    SimplifyResponse,
    MeetingContext,
    KeyInsight
)

logger = logging.getLogger(__name__)

class QuickRespondService:
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
        self.llava_model = "llava:latest"
        self.llama_model = "llama2:latest"
        self.meeting_context = {}
        self.session_insights = []
        
    async def analyze_meeting_content(self, request: QuickRespondRequest) -> QuickRespondResponse:
        """
        Analyze meeting screenshot using LLAVA and provide key insights
        """
        try:
            # Encode screenshot to base64
            screenshot_b64 = base64.b64encode(request.screenshot_data).decode('utf-8')
            
            # Build comprehensive prompt
            prompt = self._build_analysis_prompt(request)
            
            # Send to LLAVA via Ollama
            analysis_result = await self._call_llava(prompt, screenshot_b64)
            
            # Parse and structure the response
            key_insights = self._parse_insights(analysis_result)
            
            # Store insights for session context
            self.session_insights.extend(key_insights)
            
            response = QuickRespondResponse(
                key_insights=key_insights,
                full_analysis=analysis_result,
                timestamp=datetime.utcnow(),
                confidence_score=self._calculate_confidence(analysis_result),
                can_simplify=True,
                session_id=self._generate_session_id()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise Exception(f"Failed to analyze meeting content: {str(e)}")
    
    async def analyze_meeting_content_stream(self, request: QuickRespondRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real-time analysis results
        """
        try:
            screenshot_b64 = base64.b64encode(request.screenshot_data).decode('utf-8')
            prompt = self._build_analysis_prompt(request)
            
            async for chunk in self._call_llava_stream(prompt, screenshot_b64):
                # Parse partial insights as they come
                partial_insights = self._parse_partial_insights(chunk)
                
                yield {
                    "type": "partial_insight",
                    "data": partial_insights,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            yield {
                "type": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def simplify_analysis(self, request: SimplifyRequest) -> SimplifyResponse:
        """
        Simplify complex analysis using Llama model
        """
        try:
            simplify_prompt = f"""
            Please simplify this meeting analysis for quick understanding:
            
            Original Analysis: {request.original_analysis}
            
            Provide:
            1. 3 most important points in simple bullet points
            2. Any immediate actions needed
            3. Overall meeting status/sentiment
            
            Keep it concise and actionable.
            """
            
            simplified_text = await self._call_llama(simplify_prompt)
            
            # Parse simplified response
            simple_points = self._extract_simple_points(simplified_text)
            actions_needed = self._extract_actions(simplified_text)
            meeting_status = self._extract_status(simplified_text)
            
            response = SimplifyResponse(
                simplified_text=simplified_text,
                simple_points=simple_points,
                actions_needed=actions_needed,
                meeting_status=meeting_status,
                timestamp=datetime.utcnow()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Simplification failed: {str(e)}")
            raise Exception(f"Failed to simplify analysis: {str(e)}")
    
    async def check_service_health(self) -> Dict[str, Any]:
        """
        Check if Ollama and models are available
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Check Ollama service
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status != 200:
                        return {"status": "unhealthy", "ollama": False}
                
                # Check LLAVA model
                llava_available = await self._check_model_availability(self.llava_model)
                
                # Check Llama model
                llama_available = await self._check_model_availability(self.llama_model)
                
                return {
                    "status": "healthy" if llava_available and llama_available else "partial",
                    "ollama": True,
                    "llava_model": llava_available,
                    "llama_model": llama_available,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def update_meeting_context(self, context: MeetingContext):
        """
        Update meeting context for better analysis
        """
        self.meeting_context = {
            "meeting_title": context.meeting_title,
            "participants": context.participants,
            "agenda": context.agenda,
            "start_time": context.start_time,
            "meeting_type": context.meeting_type,
            "updated_at": datetime.utcnow()
        }
    
    async def clear_meeting_context(self):
        """
        Clear stored meeting context and insights
        """
        self.meeting_context = {}
        self.session_insights = []
    
    def _build_analysis_prompt(self, request: QuickRespondRequest) -> str:
        """
        Build comprehensive analysis prompt for LLAVA
        """
        base_prompt = """
        Analyze this live meeting screenshot and provide key insights ASAP. Focus on:
        
        1. **URGENT ACTIONS**: Any immediate actions or decisions visible
        2. **KEY POINTS**: Main discussion topics or presentations shown
        3. **PARTICIPANT STATUS**: Who's speaking, engagement level, reactions
        4. **SCREEN CONTENT**: Charts, documents, or data being shared
        5. **MEETING FLOW**: Current stage of meeting, next steps
        
        Provide response in this format:
        KEY_INSIGHT: [Brief actionable insight]
        URGENCY: [HIGH/MEDIUM/LOW]
        CONTEXT: [What's happening now]
        
        Be concise but comprehensive for quick decision making.
        """
        
        # Add meeting context if available
        if self.meeting_context:
            context_info = f"""
            
            MEETING CONTEXT:
            - Title: {self.meeting_context.get('meeting_title', 'N/A')}
            - Type: {self.meeting_context.get('meeting_type', 'N/A')}
            - Participants: {', '.join(self.meeting_context.get('participants', []))}
            - Agenda: {self.meeting_context.get('agenda', 'N/A')}
            """
            base_prompt += context_info
        
        # Add audio transcript context
        if request.audio_transcript:
            base_prompt += f"\n\nAUDIO CONTEXT: {request.audio_transcript}"
        
        # Add additional context
        if request.meeting_context:
            base_prompt += f"\n\nADDITIONAL CONTEXT: {request.meeting_context}"
        
        return base_prompt
    
    async def _call_llava(self, prompt: str, image_b64: str) -> str:
        """
        Call LLAVA model via Ollama API
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.llava_model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }
            
            async with session.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise Exception(f"LLAVA API error: {response.status}")
                
                result = await response.json()
                return result.get('response', '')
    
    async def _call_llava_stream(self, prompt: str, image_b64: str) -> AsyncGenerator[str, None]:
        """
        Stream LLAVA model responses
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.llava_model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": True
            }
            
            async with session.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                yield chunk['response']
                        except json.JSONDecodeError:
                            continue
    
    async def _call_llama(self, prompt: str) -> str:
        """
        Call Llama model for text simplification
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.llama_model,
                "prompt": prompt,
                "stream": False
            }
            
            async with session.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Llama API error: {response.status}")
                
                result = await response.json()
                return result.get('response', '')
    
    async def _check_model_availability(self, model_name: str) -> bool:
        """
        Check if a specific model is available
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        return model_name in models
            return False
        except:
            return False
    
    def _parse_insights(self, analysis_text: str) -> list[KeyInsight]:
        """
        Parse analysis text into structured key insights
        """
        insights = []
        lines = analysis_text.split('\n')
        
        current_insight = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('KEY_INSIGHT:'):
                if current_insight:
                    insights.append(current_insight)
                current_insight = KeyInsight(
                    insight=line.replace('KEY_INSIGHT:', '').strip(),
                    urgency='MEDIUM',
                    context='',
                    timestamp=datetime.utcnow()
                )
            elif line.startswith('URGENCY:') and current_insight:
                current_insight.urgency = line.replace('URGENCY:', '').strip()
            elif line.startswith('CONTEXT:') and current_insight:
                current_insight.context = line.replace('CONTEXT:', '').strip()
        
        if current_insight:
            insights.append(current_insight)
        
        # If no structured format found, create general insights
        if not insights:
            insights.append(KeyInsight(
                insight=analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text,
                urgency='MEDIUM',
                context='General meeting analysis',
                timestamp=datetime.utcnow()
            ))
        
        return insights
    
    def _parse_partial_insights(self, chunk: str) -> Dict[str, Any]:
        """
        Parse partial streaming insights
        """
        return {
            "partial_text": chunk,
            "processed": True
        }
    
    def _calculate_confidence(self, analysis: str) -> float:
        """
        Calculate confidence score based on analysis quality
        """
        # Simple heuristic - can be improved with more sophisticated methods
        if len(analysis) > 100 and any(keyword in analysis.lower() 
                                     for keyword in ['urgent', 'action', 'decision', 'important']):
            return 0.9
        elif len(analysis) > 50:
            return 0.7
        else:
            return 0.5
    
    def _generate_session_id(self) -> str:
        """
        Generate unique session ID
        """
        return f"session_{int(datetime.utcnow().timestamp())}"
    
    def _extract_simple_points(self, text: str) -> list[str]:
        """
        Extract simple bullet points from simplified text
        """
        lines = text.split('\n')
        points = []
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                points.append(line)
        return points[:3]  # Limit to 3 points
    
    def _extract_actions(self, text: str) -> list[str]:
        """
        Extract action items from simplified text
        """
        actions = []
        lines = text.split('\n')
        for line in lines:
            if any(action_word in line.lower() for action_word in ['action', 'todo', 'follow up', 'next step']):
                actions.append(line.strip())
        return actions
    
    def _extract_status(self, text: str) -> str:
        """
        Extract meeting status from simplified text
        """
        if any(word in text.lower() for word in ['positive', 'good', 'progress']):
            return 'positive'
        elif any(word in text.lower() for word in ['concern', 'issue', 'problem']):
            return 'needs_attention'
        else:
            return 'neutral'