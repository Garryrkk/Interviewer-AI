import base64
import json
import uuid
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import aiohttp
import logging
from io import BytesIO
from PIL import Image
import re

from .schemas import (
    KeyInsight,
    KeyInsightRequest,
    KeyInsightResponse,
    ErrorResponse,
    InsightType
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
        self.text_model = "llama3.1:latest"  # Primary text model for analysis
        self.insights_cache = {}   # Simple cache for insights
        
    async def generate_insights(
        self,
        request: KeyInsightRequest,
        image_data: Optional[bytes] = None
    ) -> KeyInsightResponse:
        """
        Generate key insights from meeting transcript and optional image analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting insight generation for meeting: {request.meeting_id}")
            
            # Analyze image if provided for additional context
            visual_context = None
            if image_data:
                logger.info("Analyzing visual context from image...")
                visual_context = await self._analyze_visual_context(image_data)
            
            # Generate insights from transcript
            logger.info("Extracting key insights from transcript...")
            insights = await self._extract_insights_from_transcript(
                transcript=request.transcript,
                extract_types=request.extract_types,
                max_insights=request.max_insights,
                visual_context=visual_context
            )
            
            # Generate meeting summary
            summary = await self._generate_meeting_summary(request.transcript)
            
            processing_time = time.time() - start_time
            
            # Create response
            response = KeyInsightResponse(
                insights=insights,
                total_insights=len(insights),
                processing_time=processing_time,
                meeting_id=request.meeting_id,
                summary=summary
            )
            
            # Cache the results
            if request.meeting_id:
                self.insights_cache[request.meeting_id] = response
            
            logger.info(f"Generated {len(insights)} insights in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            raise e
    
    async def _analyze_visual_context(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze image for meeting context using LLAVA
        """
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare prompt for visual context analysis
            visual_prompt = """
            Analyze this image to understand the meeting context. Focus on:
            1. Meeting setting and environment
            2. Number of participants and their engagement
            3. Any visible presentations, whiteboards, or materials
            4. Overall meeting atmosphere and mood
            5. Any body language or facial expressions that indicate meeting dynamics
            
            Provide a concise analysis that could help understand the meeting better.
            Keep the response under 200 words and focus on actionable observations.
            """
            
            payload = {
                "model": self.llava_model,
                "prompt": visual_prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.6,
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
                        
                        return {
                            "analysis": visual_analysis,
                            "has_visual_context": True
                        }
                    else:
                        logger.error(f"LLAVA analysis failed: {response.status}")
                        return {"has_visual_context": False, "error": "Visual analysis failed"}
                        
        except Exception as e:
            logger.error(f"Error in visual analysis: {str(e)}")
            return {"has_visual_context": False, "error": str(e)}
    
    async def _extract_insights_from_transcript(
        self,
        transcript: str,
        extract_types: Optional[List[InsightType]] = None,
        max_insights: int = 10,
        visual_context: Optional[Dict] = None
    ) -> List[KeyInsight]:
        """
        Extract key insights from meeting transcript
        """
        try:
            # Prepare types filter
            types_filter = extract_types if extract_types else list(InsightType)
            types_str = ", ".join([t.value for t in types_filter])
            
            # Add visual context if available
            context_addition = ""
            if visual_context and visual_context.get("has_visual_context"):
                context_addition = f"\n\nAdditional Visual Context: {visual_context.get('analysis', '')}"
            
            # Prepare insight extraction prompt
            insight_prompt = f"""
            Analyze the following meeting transcript and extract key insights. Focus on extracting the following types of insights: {types_str}

            Extract up to {max_insights} insights and categorize each one as:
            - DECISION: Important decisions that were made
            - ACTION_ITEM: Specific tasks or actions assigned
            - KEY_POINT: Important discussion points or conclusions
            - RISK: Potential risks or concerns identified
            - OPPORTUNITY: Opportunities or positive developments discussed
            - QUESTION: Important unresolved questions or issues

            Meeting Transcript:
            {transcript}
            {context_addition}

            For each insight, provide:
            1. The insight content (clear and concise)
            2. The type (from the categories above)
            3. A confidence score (0.0-1.0) based on how clear and significant the insight is
            4. The source section (brief quote or reference to where this came from)

            Format your response as a JSON-like structure:
            INSIGHT_1: [TYPE] | [CONTENT] | [CONFIDENCE] | [SOURCE_SECTION]
            INSIGHT_2: [TYPE] | [CONTENT] | [CONFIDENCE] | [SOURCE_SECTION]
            ...

            Example:
            INSIGHT_1: DECISION | Team agreed to implement new CRM system by Q2 | 0.95 | "We've decided to move forward with Salesforce implementation"
            INSIGHT_2: ACTION_ITEM | John will prepare budget proposal by Friday | 0.90 | "John, can you have the budget ready by end of week?"

            Keep each insight content under 100 characters and make them actionable and specific.
            """
            
            insights_text = await self._call_ollama_text_model(insight_prompt)
            insights = self._parse_insights_from_response(insights_text)
            
            return insights[:max_insights]
            
        except Exception as e:
            logger.error(f"Error extracting insights from transcript: {str(e)}")
            # Return a basic error insight
            return [
                KeyInsight(
                    id=str(uuid.uuid4()),
                    content="Unable to extract insights due to processing error",
                    type=InsightType.KEY_POINT,
                    confidence_score=0.1,
                    timestamp=datetime.utcnow(),
                    source_section="system_error"
                )
            ]
    
    async def _generate_meeting_summary(self, transcript: str) -> str:
        """
        Generate a brief summary of the meeting
        """
        try:
            summary_prompt = f"""
            Provide a concise summary of this meeting transcript in 2-3 sentences. 
            Focus on the main topics discussed, key outcomes, and overall purpose of the meeting.
            
            Transcript:
            {transcript}
            
            Summary:
            """
            
            summary = await self._call_ollama_text_model(summary_prompt)
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating meeting summary: {str(e)}")
            return "Meeting summary unavailable due to processing error."
    
    async def _call_ollama_text_model(self, prompt: str) -> str:
        """
        Call Ollama text model for generating insights and analysis
        """
        try:
            payload = {
                "model": self.text_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
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
    
    def _parse_insights_from_response(self, response_text: str) -> List[KeyInsight]:
        """
        Parse insights from the Ollama response text
        """
        insights = []
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('INSIGHT_') and '|' in line:
                try:
                    # Parse the structured response
                    parts = line.split('|')
                    if len(parts) >= 4:
                        # Extract components
                        type_part = parts[1].strip()
                        content_part = parts[2].strip()
                        confidence_part = parts[3].strip()
                        source_part = parts[4].strip() if len(parts) > 4 else "transcript"
                        
                        # Map type string to InsightType
                        insight_type = self._map_type_string(type_part)
                        
                        # Parse confidence score
                        try:
                            confidence_score = float(confidence_part)
                            confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to 0-1
                        except (ValueError, TypeError):
                            confidence_score = 0.7  # Default confidence
                        
                        # Create insight
                        insight = KeyInsight(
                            id=str(uuid.uuid4()),
                            content=content_part[:500],  # Ensure max length
                            type=insight_type,
                            confidence_score=confidence_score,
                            timestamp=datetime.utcnow(),
                            source_section=source_part[:200] if source_part else None
                        )
                        
                        insights.append(insight)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse insight line: {line}, error: {str(e)}")
                    continue
        
        # If no structured insights found, try to extract from unstructured text
        if not insights:
            insights = self._extract_fallback_insights(response_text)
        
        return insights
    
    def _map_type_string(self, type_str: str) -> InsightType:
        """
        Map type string to InsightType enum
        """
        type_mapping = {
            'DECISION': InsightType.DECISION,
            'ACTION_ITEM': InsightType.ACTION_ITEM,
            'KEY_POINT': InsightType.KEY_POINT,
            'RISK': InsightType.RISK,
            'OPPORTUNITY': InsightType.OPPORTUNITY,
            'QUESTION': InsightType.QUESTION
        }
        
        return type_mapping.get(type_str.upper(), InsightType.KEY_POINT)
    
    def _extract_fallback_insights(self, text: str) -> List[KeyInsight]:
        """
        Extract insights from unstructured text as fallback
        """
        insights = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                # Clean up the line
                clean_line = re.sub(r'^[\d\.\-•\s]+', '', line).strip()
                if clean_line and len(clean_line) > 10:  # Ensure meaningful content
                    insight = KeyInsight(
                        id=str(uuid.uuid4()),
                        content=clean_line[:500],
                        type=InsightType.KEY_POINT,
                        confidence_score=0.6,  # Lower confidence for fallback
                        timestamp=datetime.utcnow(),
                        source_section="extracted_from_response"
                    )
                    insights.append(insight)
        
        return insights[:10]  # Limit to 10 insights
    
    async def generate_insights_safe(
        self,
        request: KeyInsightRequest,
        image_data: Optional[bytes] = None
    ) -> KeyInsightResponse | ErrorResponse:
        """
        Generate key insights with proper error response handling
        """
        try:
            return await self.generate_insights(request, image_data)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return ErrorResponse(
                error=f"Invalid request data: {str(e)}",
                error_code="VALIDATION_ERROR",
                timestamp=datetime.utcnow()
            )
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {str(e)}")
            return ErrorResponse(
                error="Unable to connect to AI model service",
                error_code="AI_SERVICE_UNAVAILABLE",
                timestamp=datetime.utcnow()
            )
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            return ErrorResponse(
                error="Request timed out while processing",
                error_code="PROCESSING_TIMEOUT",
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return ErrorResponse(
                error="An unexpected error occurred during processing",
                error_code="INTERNAL_SERVER_ERROR",
                timestamp=datetime.utcnow()
            )

    async def get_cached_insights_safe(
        self, 
        meeting_id: str
    ) -> KeyInsightResponse | ErrorResponse:
        """
        Get cached insights with error handling
        """
        try:
            if not meeting_id or not meeting_id.strip():
                return ErrorResponse(
                    error="Meeting ID is required and cannot be empty",
                    error_code="INVALID_MEETING_ID",
                    timestamp=datetime.utcnow()
                )
            
            cached_response = await self.get_cached_insights(meeting_id)
            if not cached_response:
                return ErrorResponse(
                    error=f"No insights found for meeting ID: {meeting_id}",
                    error_code="INSIGHTS_NOT_FOUND",
                    timestamp=datetime.utcnow()
                )
            
            return cached_response
            
        except Exception as e:
            logger.error(f"Error retrieving cached insights: {str(e)}")
            return ErrorResponse(
                error="Failed to retrieve cached insights",
                error_code="CACHE_RETRIEVAL_ERROR",
                timestamp=datetime.utcnow()
            )

    async def get_insights_by_type_safe(
        self, 
        meeting_id: str, 
        insight_type: InsightType
    ) -> List[KeyInsight] | ErrorResponse:
        """
        Get insights by type with error handling
        """
        try:
            if not meeting_id or not meeting_id.strip():
                return ErrorResponse(
                    error="Meeting ID is required and cannot be empty",
                    error_code="INVALID_MEETING_ID",
                    timestamp=datetime.utcnow()
                )
            
            insights = await self.get_insights_by_type(meeting_id, insight_type)
            return insights
            
        except Exception as e:
            logger.error(f"Error getting insights by type: {str(e)}")
            return ErrorResponse(
                error=f"Failed to retrieve {insight_type.value} insights",
                error_code="INSIGHTS_RETRIEVAL_ERROR",
                timestamp=datetime.utcnow()
            )

    async def validate_request(self, request: KeyInsightRequest) -> Optional[ErrorResponse]:
        """
        Validate the insight request and return error if invalid
        """
        try:
            # Check transcript length
            if len(request.transcript.strip()) < 10:
                return ErrorResponse(
                    error="Transcript is too short. Minimum 10 characters required.",
                    error_code="TRANSCRIPT_TOO_SHORT",
                    timestamp=datetime.utcnow()
                )
            
            # Check max insights range
            if request.max_insights < 1 or request.max_insights > 50:
                return ErrorResponse(
                    error="max_insights must be between 1 and 50",
                    error_code="INVALID_MAX_INSIGHTS",
                    timestamp=datetime.utcnow()
                )
            
            # Validate extract_types if provided
            if request.extract_types:
                invalid_types = [
                    t for t in request.extract_types 
                    if t not in InsightType
                ]
                if invalid_types:
                    return ErrorResponse(
                        error=f"Invalid insight types: {invalid_types}",
                        error_code="INVALID_INSIGHT_TYPES",
                        timestamp=datetime.utcnow()
                    )
            
            return None  # No errors
            
        except Exception as e:
            logger.error(f"Error validating request: {str(e)}")
            return ErrorResponse(
                error="Request validation failed",
                error_code="VALIDATION_ERROR",
                timestamp=datetime.utcnow()
            )
    
    async def get_cached_insights(self, meeting_id: str) -> Optional[KeyInsightResponse]:
        """
        Get cached insights for a meeting
        """
        return self.insights_cache.get(meeting_id)
    
    async def clear_cache_safe(self, meeting_id: Optional[str] = None) -> bool | ErrorResponse:
        """
        Clear insights cache with error handling
        """
        try:
            result = await self.clear_cache(meeting_id)
            return result
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return ErrorResponse(
                error="Failed to clear cache",
                error_code="CACHE_CLEAR_ERROR",
                timestamp=datetime.utcnow()
            )

    async def health_check_safe(self) -> Dict[str, Any] | ErrorResponse:
        """
        Health check with error handling
        """
        try:
            health_data = await self.health_check()
            return health_data
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return ErrorResponse(
                error="Health check failed",
                error_code="HEALTH_CHECK_ERROR",
                timestamp=datetime.utcnow()
            )
    
    async def get_insights_by_type(
        self, 
        meeting_id: str, 
        insight_type: InsightType
    ) -> List[KeyInsight]:
        """
        Get insights filtered by type for a specific meeting
        """
        cached_response = await self.get_cached_insights(meeting_id)
        if not cached_response:
            return []
        
        return [
            insight for insight in cached_response.insights 
            if insight.type == insight_type
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the service and Ollama connection
        """
        try:
            # Test Ollama connection
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_base_url}/api/version",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    ollama_healthy = response.status == 200
            
            return {
                "service_healthy": True,
                "ollama_connection": ollama_healthy,
                "cached_meetings": len(self.insights_cache),
                "timestamp": datetime.utcnow(),
                "models": {
                    "text_model": self.text_model,
                    "vision_model": self.llava_model
                }
            }
            
        except Exception as e:
            return {
                "service_healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
    