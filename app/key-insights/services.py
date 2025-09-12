import re
import time
from typing import List, Optional, Dict
from datetime import datetime
import logging
from .schemas import KeyInsight, InsightType, KeyInsightRequest

logger = logging.getLogger(__name__)

class KeyInsightsService:

    def __init__(self):
        # Keywords for different insight types
        self.insight_patterns = {
            InsightType.DECISION: [
                r"(?:we (?:decided|agreed|concluded)|decision|final decision|we'll go with)",
                r"(?:it's decided|settled|final)",
                r"(?:chosen|selected|picked)"
            ],
            InsightType.ACTION_ITEM: [
                r"(?:action item|todo|to do|task|assignment)",
                r"(?:will|should|need to|must|have to) (?:follow up|contact|send|prepare|complete)",
                r"(?:next steps?|follow[- ]?up|deadline)"
            ],
            InsightType.KEY_POINT: [
                r"(?:important|key|main|primary|crucial|significant|notable)",
                r"(?:the point is|main takeaway|key insight|summary)"
            ],
            InsightType.RISK: [
                r"(?:risk|concern|worry|problem|issue|challenge|obstacle)",
                r"(?:might fail|could go wrong|potential problem)"
            ],
            InsightType.OPPORTUNITY: [
                r"(?:opportunity|potential|possibility|chance)",
                r"(?:could help|might benefit|advantage)"
            ],
            InsightType.QUESTION: [
                r"(?:question|need to ask|wonder|clarify|unclear)",
                r"(?:\?|what about|how about|should we)"
            ]
        }

    def extract_key_insights(self, request: KeyInsightRequest) -> Dict:
        start_time = time.time()
        
        try:
            # Clean and prepare transcript
            cleaned_transcript = self._clean_transcript(request.transcript)
            
            # Split into sentences
            sentences = self._split_into_sentences(cleaned_transcript)
            
            # Extract insights based on patterns and rules
            insights = self._extract_insights_from_sentences(
                sentences, 
                request.extract_types, 
                request.max_insights
            )
            
            # Generate summary
            summary = self._generate_summary(cleaned_transcript, insights)
            
            processing_time = time.time() - start_time
            
            return {
                "insights": insights,
                "total_insights": len(insights),
                "processing_time": round(processing_time, 3),
                "meeting_id": request.meeting_id,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            raise e

    def _clean_transcript(self, transcript: str) -> str:
        cleaned = re.sub(r'\s+', ' ', transcript.strip())
        
        # Remove speaker labels like "Speaker 1:", "John:", etc.
        cleaned = re.sub(r'^[A-Za-z\s]+\d*:\s*', '', cleaned, flags=re.MULTILINE)
        
        # Remove timestamps like [00:15:30]
        cleaned = re.sub(r'\[\d{1,2}:\d{2}:\d{2}\]', '', cleaned)
        
        # Remove filler words and hesitations
        cleaned = re.sub(r'\b(?:um|uh|ah|like|you know|sort of|kind of)\b', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_insights_from_sentences(
        self, 
        sentences: List[str], 
        extract_types: Optional[List[InsightType]], 
        max_insights: int
    ) -> List[KeyInsight]:
        """Extract insights from sentences using pattern matching."""
        insights = []
        
        # Default to all types if none specified
        target_types = extract_types or list(InsightType)
        
        for sentence in sentences:
            if len(insights) >= max_insights:
                break
                
            for insight_type in target_types:
                if len(insights) >= max_insights:
                    break
                    
                if self._matches_insight_pattern(sentence, insight_type):
                    insight = KeyInsight(
                        content=sentence.strip(),
                        type=insight_type,
                        confidence_score=self._calculate_confidence(sentence, insight_type),
                        source_section=sentence[:50] + "..." if len(sentence) > 50 else sentence
                    )
                    insights.append(insight)
                    break  # One insight per sentence
        
        # Sort by confidence score (descending)
        insights.sort(key=lambda x: x.confidence_score or 0, reverse=True)
        
        return insights[:max_insights]

    def _matches_insight_pattern(self, sentence: str, insight_type: InsightType) -> bool:
        """Check if sentence matches patterns for given insight type."""
        patterns = self.insight_patterns.get(insight_type, [])
        
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False

    def _calculate_confidence(self, sentence: str, insight_type: InsightType) -> float:
        """Calculate confidence score for an insight (0.0 to 1.0)."""
        base_score = 0.5
        
        # Boost confidence based on sentence length (longer = more context)
        length_factor = min(len(sentence) / 100, 1.0) * 0.2
        
        # Boost confidence based on keyword matches
        patterns = self.insight_patterns.get(insight_type, [])
        match_count = sum(1 for pattern in patterns if re.search(pattern, sentence, re.IGNORECASE))
        match_factor = min(match_count * 0.15, 0.3)
        
        # Boost confidence for specific insight types
        type_factor = {
            InsightType.DECISION: 0.1,
            InsightType.ACTION_ITEM: 0.15,
            InsightType.RISK: 0.05,
            InsightType.OPPORTUNITY: 0.05,
            InsightType.KEY_POINT: 0.0,
            InsightType.QUESTION: 0.05
        }.get(insight_type, 0.0)
        
        confidence = base_score + length_factor + match_factor + type_factor
        return min(confidence, 1.0)

    def _generate_summary(self, transcript: str, insights: List[KeyInsight]) -> str:
        """Generate a brief summary of the meeting."""
        if not insights:
            return "No significant insights extracted from the meeting."
        
        # Count insights by type
        type_counts = {}
        for insight in insights:
            type_counts[insight.type] = type_counts.get(insight.type, 0) + 1
        
        # Create summary based on insight types
        summary_parts = []
        
        if InsightType.DECISION in type_counts:
            summary_parts.append(f"{type_counts[InsightType.DECISION]} key decision(s) made")
        
        if InsightType.ACTION_ITEM in type_counts:
            summary_parts.append(f"{type_counts[InsightType.ACTION_ITEM]} action item(s) identified")
        
        if InsightType.RISK in type_counts:
            summary_parts.append(f"{type_counts[InsightType.RISK]} risk(s) discussed")
        
        if InsightType.OPPORTUNITY in type_counts:
            summary_parts.append(f"{type_counts[InsightType.OPPORTUNITY]} opportunity(s) identified")
        
        if summary_parts:
            return f"Meeting summary: {', '.join(summary_parts)}."
        else:
            return f"Meeting covered {len(insights)} key points and insights."


# Service instance
insights_service = KeyInsightsService()

# Convenience function for backward compatibility
def extract_key_insights(transcript: str, max_insights: int = 10) -> List[str]:
    """
    Simplified function for extracting insights (returns just the content strings).
    This maintains backward compatibility with existing code.
    """
    request = KeyInsightRequest(transcript=transcript, max_insights=max_insights)
    result = insights_service.extract_key_insights(request)
    return [insight.content for insight in result["insights"]]