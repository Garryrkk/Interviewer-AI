import logging
import asyncio
import random
from typing import Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickResponseService:

    def __init__(self):
        self.response_templates = {
            "professional": [
                "Thank you for your question. Based on my analysis, {}",
                "I appreciate your inquiry. Here's my professional assessment: {}",
                "After careful consideration, I believe {}",
            ],
            "casual": [
                "Hey! So from what I can see, {}",
                "Good question! I think {}",
                "Hmm, here's what I'm thinking: {}",
            ],
            "technical": [
                "From a technical perspective, {}",
                "The technical analysis shows that {}",
                "Based on the specifications, {}",
            ],
            "general": [
                "Here's my response: {}",
                "Based on your input, {}",
                "My suggestion would be: {}",
            ]
        }
    
    def preprocess_prompt(self, prompt: str) -> str:
        prompt = re.sub(r'\s+', ' ', prompt.strip())
        prompt = re.sub(r'[<>{}[\]\\]', '', prompt)
        if len(prompt) > 2000:
            prompt = prompt[:2000] + "..."
        
        return prompt
    
    def postprocess_response(self, response: str, max_length: int = 200) -> str:
        response = response.strip()

        if response and not response.endswith(('.', '!', '?')):
            response += '.'

        if len(response) > max_length:
            sentences = response.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= max_length:
                    truncated += sentence + '. '
                else:
                    break
            response = truncated.rstrip() or response[:max_length-3] + "..."
        
        return response
    
    def get_response_template(self, response_type: str) -> str:
        templates = self.response_templates.get(response_type, self.response_templates["general"])
        return random.choice(templates)
    
    async def generate_quick_response_async(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        response_type: str = "general",
        max_length: int = 200
    ) -> str:
        return await asyncio.to_thread(
            self.generate_quick_response, 
            prompt, 
            context, 
            response_type, 
            max_length
        )
    
    def generate_quick_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        response_type: str = "general",
        max_length: int = 200
    ) -> str:
        try:
            logger.info(f"Generating response for prompt: {prompt[:50]}...")
            clean_prompt = self.preprocess_prompt(prompt)
            response_content = self._generate_mock_response(clean_prompt, context, response_type)
            template = self.get_response_template(response_type)
            formatted_response = template.format(response_content)
            final_response = self.postprocess_response(formatted_response, max_length)

            logger.info(f"Successfully generated response of length: {len(final_response)}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating quick response: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def _generate_mock_response(self, prompt: str, context: Optional[str], response_type: str) -> str:
        prompt_lower = prompt.lower()
        
        if context:
            context_lower = context.lower()
            if "meeting" in context_lower or "presentation" in context_lower:
                if "question" in prompt_lower:
                    return "I understand your question. Let me provide a clear answer that addresses your concerns"
                elif "problem" in prompt_lower or "issue" in prompt_lower:
                    return "I see the challenge you're facing. Here's a practical solution we can implement"
                else:
                    return "Thank you for bringing this up. This is definitely worth discussing further"
        
        if "how" in prompt_lower:
            return "there are several approaches to consider. The most effective method would be to start with the fundamentals"
        elif "what" in prompt_lower:
            return "this is an important topic that requires careful consideration of multiple factors"
        elif "why" in prompt_lower:
            return "the underlying reasons involve several key principles that I can explain"
        elif "when" in prompt_lower:
            return "the timing depends on various factors, but generally speaking, sooner is better than later"
        elif "where" in prompt_lower:
            return "the location or context matters significantly for this particular situation"
        elif any(word in prompt_lower for word in ["problem", "issue", "error", "bug"]):
            return "I can help you troubleshoot this. Let's break down the problem systematically"
        elif any(word in prompt_lower for word in ["thank", "thanks", "appreciate"]):
            return "you're very welcome! I'm glad I could assist you with this"
        else:
            return "this is an interesting point that deserves a thoughtful response"


# Create a global instance
quick_response_service = QuickResponseService()

def generate_quick_response(
    prompt: str, 
    context: Optional[str] = None,
    response_type: str = "general",
    max_length: int = 200
) -> str:
    """
    Generate a quick response - synchronous version
    """
    return quick_response_service.generate_quick_response(prompt, context, response_type, max_length)

async def generate_quick_response_async(
    prompt: str, 
    context: Optional[str] = None,
    response_type: str = "general",
    max_length: int = 200
) -> str:
    """
    Generate a quick response - asynchronous version
    """
    return await quick_response_service.generate_quick_response_async(prompt, context, response_type, max_length)