import logging
import asyncio
import random
import re
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HandsFreeService:

    def __init__(self):
        self.templates = {
            "interview": [
                "That's a great question. {}",
                "Based on my understanding, {}",
                "I think the main point is that {}",
            ],
            "casual": [
                "Oh, that's interesting! {}",
                "Well, from what I know, {}",
                "Let me explain simply: {}",
            ],
            "general": [
                "Here's my response: {}",
                "Based on what you said, {}",
                "My suggestion would be: {}",
            ]
        }

    def preprocess(self, transcript: str) -> str:
        transcript = re.sub(r"\s+", " ", transcript.strip())
        transcript = re.sub(r"[<>{}\[\]\\]", "", transcript)
        return transcript[:2000]

    def simplify_text(self, text: str) -> str:
        """Basic simplifier for confused users"""
        text = re.sub(r"\b(utilize|commence|terminate)\b",
                      lambda m: {"utilize": "use", "commence": "start", "terminate": "end"}[m.group()],
                      text)
        sentences = text.split(". ")
        return ". ".join(sentences[:2]) + "."

    def generate_response(self, transcript: str, context: Optional[str], simplify: bool) -> str:
        logger.info(f"Generating hands-free response for: {transcript[:50]}...")

        clean = self.preprocess(transcript)
        base_reply = self._mock_logic(clean)

        ctx = context or "general"
        template = random.choice(self.templates.get(ctx, self.templates["general"]))
        reply = template.format(base_reply)

        if simplify:
            reply = self.simplify_text(reply)

        return reply.strip()

    def _mock_logic(self, transcript: str) -> str:
        t = transcript.lower()
        if "how" in t:
            return "there are a few steps to follow, starting with the basics"
        elif "what" in t:
            return "it is something that involves learning from data"
        elif "why" in t:
            return "the reason is usually related to efficiency and better results"
        else:
            return "that’s an interesting point and worth considering"


# Global instance
handsfree_service = HandsFreeService()


async def generate_handsfree_response_async(
    transcript: str,
    context: Optional[str] = None,
    simplify: bool = False
) -> str:
    return await asyncio.to_thread(
        handsfree_service.generate_response,
        transcript,
        context,
        simplify
    )
