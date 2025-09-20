import asyncio
import base64
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import aiohttp
import logging
from io import BytesIO
from PIL import Image

from .schemas import (
    KeyInsightsResponse,
    SimplifiedInsightsResponse,
    AnalysisStatusResponse,
    InsightPoint,
    TipPoint
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeyInsightsService:
    """
    Service class for generating key insights using LLAVA and Ollama
    """
    
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"  # Default Ollama URL
        self.llava_model = "llava:latest"
        self.text_model = "llama3.1:latest"  # Fallback text model
        self.analysis_status = {}  # In-memory status tracking
        self.insights_cache = {}   # Simple cache for insights
        
    async def generate_insights(
        self,
        meeting_context: Optional[str] = None,
        meeting_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        image_data: Optional[bytes] = None,
        analysis_focus: Optional[str] = None
    ) -> KeyInsightsResponse:
        """
        Generate key insights from meeting context and optional image analysis
        """
        insight_id = str(uuid.uuid4())
        
        try:
            # Update status
            self.analysis_status[insight_id] = {
                "status": "processing",
                "started_at": datetime.utcnow(),
                "progress": 10
            }
            
            # Analyze image if provided (facial expressions and body language)
            visual_analysis = None
            if image_data:
                logger.info("Analyzing facial expressions and body language...")
                visual_analysis = await self._analyze_visual_cues(image_data)
                self.analysis_status[insight_id]["progress"] = 40
            
            # Generate key insights from context
            logger.info("Generating key insights from meeting context...")
            insights = await self._generate_context_insights(
                meeting_context, participants, analysis_focus
            )
            self.analysis_status[insight_id]["progress"] = 70
            
            # Generate tips based on visual analysis and context
            tips = await self._generate_situation_tips(
                meeting_context, visual_analysis, participants
            )
            self.analysis_status[insight_id]["progress"] = 90
            
            # Create response
            response = KeyInsightsResponse(
                insight_id=insight_id,
                meeting_id=meeting_id or str(uuid.uuid4()),
                generated_at=datetime.utcnow(),
                key_insights=insights,
                situation_tips=tips,
                visual_analysis_included=image_data is not None,
                participants_analyzed=participants or [],
                confidence_score=0.85  # Base confidence score
            )
            
            # Cache the results
            self.insights_cache[insight_id] = response
            
            # Update status to completed
            self.analysis_status[insight_id] = {
                "status": "completed",
                "started_at": self.analysis_status[insight_id]["started_at"],
                "completed_at": datetime.utcnow(),
                "progress": 100
            }
            
            return response
            
        except Exception as e:
            # Update status to failed
            self.analysis_status[insight_id] = {
                "status": "failed",
                "started_at": self.analysis_status.get(insight_id, {}).get("started_at", datetime.utcnow()),
                "error": str(e),
                "progress": 0
            }
            raise e
    
    async def _analyze_visual_cues(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze facial expressions and body language using LLAVA
        """
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare prompt for visual analysis
            visual_prompt = """
            Analyze this image focusing on facial expressions and body language of people in what appears to be a meeting or professional setting. 

            Provide analysis in the following format:
            1. Facial Expressions: [describe emotions, engagement levels, stress indicators]
            2. Body Language: [describe posture, gestures, attention levels]
            3. Overall Mood: [describe the general atmosphere]
            4. Engagement Level: [rate from 1-10 and explain]
            5. Stress Indicators: [identify any signs of stress or discomfort]
            6. Communication Dynamics: [describe interaction patterns if multiple people visible]

            Keep each point concise and actionable.
            """
            
            payload = {
                "model": self.llava_model,
                "prompt": visual_prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        visual_analysis = result.get('response', '')
                        
                        # Parse the analysis into structured format
                        return {
                            "raw_analysis": visual_analysis,
                            "engagement_level": self._extract_engagement_level(visual_analysis),
                            "stress_indicators": self._extract_stress_indicators(visual_analysis),
                            "mood_assessment": self._extract_mood_assessment(visual_analysis)
                        }
                    else:
                        logger.error(f"LLAVA analysis failed: {response.status}")
                        return {"error": "Visual analysis failed", "fallback": True}
                        
        except Exception as e:
            logger.error(f"Error in visual analysis: {str(e)}")
            return {"error": str(e), "fallback": True}
    
    async def _generate_context_insights(
        self,
        meeting_context: Optional[str],
        participants: Optional[List[str]],
        analysis_focus: Optional[str]
    ) -> List[InsightPoint]:
        """
        Generate key insights from meeting context
        """
        try:
            # Prepare context prompt
            context_prompt = f"""
            Based on the following meeting context, generate 5-7 key insights in bullet point format. 
            Each insight should be concise, actionable, and valuable for the participants.

            Meeting Context: {meeting_context or "General meeting discussion"}
            Participants: {', '.join(participants) if participants else "Multiple participants"}
            Focus Area: {analysis_focus or "General insights"}

            Format your response as numbered points:
            1. [Key insight about decision made]
            2. [Key insight about action items]
            3. [Key insight about challenges identified]
            4. [Key insight about opportunities]
            5. [Key insight about next steps]
            6. [Key insight about team dynamics]
            7. [Key insight about outcomes]

            Make each point specific, actionable, and under 25 words.
            """
            
            insights_text = await self._call_ollama_text_model(context_prompt)
            insights = self._parse_insights_from_text(insights_text)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating context insights: {str(e)}")
            # Return fallback insights
            return [
                InsightPoint(
                    point="Meeting analysis unavailable - technical error occurred",
                    category="system",
                    priority="low",
                    confidence=0.1
                )
            ]
    
    async def _generate_situation_tips(
        self,
        meeting_context: Optional[str],
        visual_analysis: Optional[Dict],
        participants: Optional[List[str]]
    ) -> List[TipPoint]:
        """
        Generate situational tips based on context and visual analysis
        """
        try:
            # Build prompt with visual analysis if available
            visual_context = ""
            if visual_analysis and not visual_analysis.get("error"):
                visual_context = f"""
                Visual Analysis Results:
                - Engagement Level: {visual_analysis.get('engagement_level', 'Not assessed')}
                - Stress Indicators: {visual_analysis.get('stress_indicators', 'None detected')}
                - Mood Assessment: {visual_analysis.get('mood_assessment', 'Neutral')}
                """
            
            tips_prompt = f"""
            Based on the meeting context and visual analysis (if available), provide 4-5 practical tips 
            for improving the meeting effectiveness and participant engagement.

            Meeting Context: {meeting_context or "General meeting"}
            {visual_context}
            Participants: {', '.join(participants) if participants else "Team members"}

            Provide tips in this format:
            1. [Tip for immediate action]
            2. [Tip for engagement improvement]
            3. [Tip for stress reduction]
            4. [Tip for better communication]
            5. [Tip for follow-up actions]

            Make each tip actionable and under 30 words. Focus on practical advice.
            """
            
            tips_text = await self._call_ollama_text_model(tips_prompt)
            tips = self._parse_tips_from_text(tips_text)
            
            return tips
            
        except Exception as e:
            logger.error(f"Error generating situation tips: {str(e)}")
            # Return fallback tips
            return [
                TipPoint(
                    tip="Ensure all participants have a chance to speak",
                    category="engagement",
                    actionability="high"
                ),
                TipPoint(
                    tip="Take regular breaks to maintain focus",
                    category="wellness",
                    actionability="high"
                )
            ]
    
    async def simplify_insights(
        self,
        original_insights: List[InsightPoint],
        original_tips: List[TipPoint],
        simplification_level: str = "moderate",
        original_insight_id: Optional[str] = None
    ) -> SimplifiedInsightsResponse:
        """
        Generate simplified version of insights
        """
        try:
            # Prepare simplification prompt
            insights_text = "\n".join([f"• {insight.point}" for insight in original_insights])
            tips_text = "\n".join([f"• {tip.tip}" for tip in original_tips])
            
            simplification_prompt = f"""
            Simplify the following insights and tips for easier understanding. 
            Simplification level: {simplification_level}

            Original Insights:
            {insights_text}

            Original Tips:
            {tips_text}

            For {simplification_level} simplification:
            - Use simpler words and shorter sentences
            - Combine related points where possible
            - Make the language more conversational
            - Keep the core meaning intact

            Provide simplified insights (3-4 points) and simplified tips (3-4 points):

            SIMPLIFIED INSIGHTS:
            1. [simplified insight]
            2. [simplified insight]
            3. [simplified insight]

            SIMPLIFIED TIPS:
            1. [simplified tip]
            2. [simplified tip]
            3. [simplified tip]
            """
            
            simplified_text = await self._call_ollama_text_model(simplification_prompt)
            
            # Parse simplified results
            simplified_insights, simplified_tips = self._parse_simplified_response(simplified_text)
            
            return SimplifiedInsightsResponse(
                simplified_insight_id=str(uuid.uuid4()),
                original_insight_id=original_insight_id,
                simplified_insights=simplified_insights,
                simplified_tips=simplified_tips,
                simplification_level=simplification_level,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error simplifying insights: {str(e)}")
            raise e
    
    async def _call_ollama_text_model(self, prompt: str) -> str:
        """
        Call Ollama text model for generating insights
        """
        try:
            payload = {
                "model": self.text_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        logger.error(f"Ollama API error: {response.status}")
                        raise Exception(f"Ollama API returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise e
    
    def _parse_insights_from_text(self, text: str) -> List[InsightPoint]:
        """Parse insights from generated text"""
        insights = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                # Clean up the line
                clean_line = line.lstrip('0123456789.•- ').strip()
                if clean_line:
                    insights.append(InsightPoint(
                        point=clean_line,
                        category="general",
                        priority="medium",
                        confidence=0.8
                    ))
        
        return insights[:7]  # Limit to 7 insights
    
    def _parse_tips_from_text(self, text: str) -> List[TipPoint]:
        """Parse tips from generated text"""
        tips = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                clean_line = line.lstrip('0123456789.•- ').strip()
                if clean_line:
                    tips.append(TipPoint(
                        tip=clean_line,
                        category="general",
                        actionability="medium"
                    ))
        
        return tips[:5]  # Limit to 5 tips
    
    def _parse_simplified_response(self, text: str) -> tuple:
        """Parse simplified insights and tips from response"""
        lines = text.split('\n')
        simplified_insights = []
        simplified_tips = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'SIMPLIFIED INSIGHTS:' in line.upper():
                current_section = 'insights'
                continue
            elif 'SIMPLIFIED TIPS:' in line.upper():
                current_section = 'tips'
                continue
            
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                clean_line = line.lstrip('0123456789.•- ').strip()
                if clean_line:
                    if current_section == 'insights':
                        simplified_insights.append(InsightPoint(
                            point=clean_line,
                            category="simplified",
                            priority="medium",
                            confidence=0.8
                        ))
                    elif current_section == 'tips':
                        simplified_tips.append(TipPoint(
                            tip=clean_line,
                            category="simplified",
                            actionability="high"
                        ))
        
        return simplified_insights, simplified_tips
    
    def _extract_engagement_level(self, analysis: str) -> str:
        """Extract engagement level from visual analysis"""
        analysis_lower = analysis.lower()
        if 'high engagement' in analysis_lower or 'very engaged' in analysis_lower:
            return "high"
        elif 'low engagement' in analysis_lower or 'disengaged' in analysis_lower:
            return "low"
        else:
            return "moderate"
    
    def _extract_stress_indicators(self, analysis: str) -> str:
        """Extract stress indicators from visual analysis"""
        analysis_lower = analysis.lower()
        stress_words = ['stress', 'tension', 'anxious', 'worried', 'concerned', 'frustrated']
        
        for word in stress_words:
            if word in analysis_lower:
                return "detected"
        return "minimal"
    
    def _extract_mood_assessment(self, analysis: str) -> str:
        """Extract mood assessment from visual analysis"""
        analysis_lower = analysis.lower()
        if 'positive' in analysis_lower or 'happy' in analysis_lower or 'upbeat' in analysis_lower:
            return "positive"
        elif 'negative' in analysis_lower or 'frustrated' in analysis_lower or 'upset' in analysis_lower:
            return "negative"
        else:
            return "neutral"
    
    async def get_analysis_status(self, insight_id: str) -> AnalysisStatusResponse:
        """Get the status of an insight analysis"""
        status_data = self.analysis_status.get(insight_id)
        
        if not status_data:
            return AnalysisStatusResponse(
                insight_id=insight_id,
                status="not_found",
                progress=0
            )
        
        return AnalysisStatusResponse(
            insight_id=insight_id,
            status=status_data["status"],
            progress=status_data.get("progress", 0),
            started_at=status_data.get("started_at"),
            completed_at=status_data.get("completed_at"),
            error=status_data.get("error")
        )
    
    async def get_insights_history(self, meeting_id: str) -> List[Dict]:
        """Get insights history for a meeting"""
        # This would typically query a database
        # For now, return cached insights that match the meeting_id
        history = []
        for insight_id, insights in self.insights_cache.items():
            if insights.meeting_id == meeting_id:
                history.append({
                    "insight_id": insight_id,
                    "generated_at": insights.generated_at,
                    "visual_analysis_included": insights.visual_analysis_included,
                    "insights_count": len(insights.key_insights),
                    "tips_count": len(insights.situation_tips)
                })
        
        return sorted(history, key=lambda x: x["generated_at"], reverse=True)
    
    async def delete_insights(self, insight_id: str) -> bool:
        """Delete specific insights"""
        if insight_id in self.insights_cache:
            del self.insights_cache[insight_id]
            return True
        return False