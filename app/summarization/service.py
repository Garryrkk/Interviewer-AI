import logging
import asyncio
import re
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationService:

    def preprocess(self, text: str) -> str:
        """Clean up input text"""
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"[<>{}\[\]\\]", "", text)
        return text[:8000]

    def summarize(self, text: str, style: str = "concise", simplify: bool = False) -> str:
        """Generate summary (mock logic for now)"""
        logger.info(f"Summarizing text, style={style}, simplify={simplify}")

        clean = self.preprocess(text)

        # Simple heuristic mock summary
        sentences = clean.split(". ")
        if style == "bullet":
            summary = " • " + "\n • ".join(sentences[:3])
        elif style == "detailed":
            summary = ". ".join(sentences[:5]) + "."
        else:  # concise
            summary = ". ".join(sentences[:2]) + "."

        if simplify:
            summary = self._simplify_text(summary)

        return summary.strip()

    def _simplify_text(self, text: str) -> str:
        """Simplify summary with basic replacements"""
        replacements = {
            "utilize": "use",
            "commence": "start",
            "terminate": "end",
            "approximately": "about",
            "individuals": "people"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text


# Global instance
summarization_service = SummarizationService()


async def generate_summary_async(
    text: str,
    style: Optional[str] = "concise",
    simplify: bool = False
) -> str:
    return await asyncio.to_thread(
        summarization_service.summarize,
        text,
        style,
        simplify
    )
