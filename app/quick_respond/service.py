# service.py
import os
import asyncio
import logging
from typing import Tuple, List

try:
    import httpx
except Exception:
    httpx = None

logger = logging.getLogger("quickrespond.service")
logger.setLevel(logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change as needed

async def call_openai_chat(prompt: str, max_tokens: int = 80) -> str:
    """
    Call OpenAI ChatCompletions (if key available).
    This function uses the standard API shape for chat completions.
    If you use another LLM provider, adapt headers/payload here.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")

    if httpx is None:
        raise RuntimeError("httpx is required to call external APIs. `pip install httpx`")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful interview-coach assistant. Provide concise, actionable responses."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(OPENAI_API_URL, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

    # This parsing may vary depending on provider. For OpenAI: choices[0].message.content
    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"].strip()
    # fallback
    return data.get("text") or str(data)

def _simple_style_guidelines(simplify: bool) -> str:
    if simplify:
        return (
            "Simplify language to an easy-to-understand short answer (one or two sentences). "
            "Use short words, avoid jargon, and if possible give a 1-line example."
        )
    return (
        "Give a concise interview-style response (about 1-2 short paragraphs). "
        "Prefer STAR structure when applicable (Situation, Task, Action, Result). "
        "Keep it actionable and professional."
    )

def _mock_quick_reply(text: str, simplify: bool) -> Tuple[str, List[str]]:
    """
    Heuristic mock generator for local testing without external LLM calls.
    Returns (reply, suggestions)
    """
    # Basic heuristics:
    # - If the text looks like a definition e.g., contains 'is' and short => short definition
    # - If contains 'how' or 'why' => provide steps or reasons (shortened)
    # - Otherwise produce a STAR-style concise answer skeleton + one-liner simplification if requested
    lower = text.strip().lower()
    suggestions = []

    # Extract a short subject (first 6-10 words) for personalization
    words = text.strip().split()
    subject = " ".join(words[:8])

    if any(q in lower for q in ["what is", "define", "explain"]):
        reply = f"{subject} — In short: {('It appears to be ' + (subject if subject else 'the item'))}."
        suggestions = [
            "Keep it short and start with a 1-line definition.",
            "Follow up with one example."
        ]
    elif any(q in lower for q in ["how to", "how do i", "how can i", "steps", "process"]):
        reply = "Quick steps: 1) Identify goal. 2) Break task into smaller steps. 3) Execute and measure results."
        suggestions = ["Give a specific example with numbers if possible."]
    else:
        # STAR-like skeleton
        reply = ("Situation: Briefly set the scene. Action: Explain what you did. "
                 "Result: Share measurable outcomes. Keep answers to ~60-90 seconds.")
        suggestions = [
            "Start with one sentence summarizing the context.",
            "Mention concrete outcomes if you can (numbers, percentages)."
        ]

    if simplify:
        # produce an even shorter plain-language reply
        short = reply.split(".")[0]
        reply = f"{short}. (Simpler: {short.split(':')[-1].trim() if ':' in short else short})"

    # Ensure reply not too long
    if len(reply) > 500:
        reply = reply[:500].rsplit(" ", 1)[0] + "..."

    return reply, suggestions

async def generate_quick_reply(text: str, simplify: bool = False, max_tokens: int = 80) -> dict:
    """
    Main service entry. Tries to call an LLM provider if configured, otherwise
    returns a mock reply. Returns dict matching QuickResponse schema.
    """
    text = (text or "").strip()
    if not text:
        return {"reply": "Please provide some text to analyze.", "suggestions": [], "source": "validation"}

    # Build the prompt to send to the model (if available)
    prompt = (
        _simple_style_guidelines(simplify) + "\n\n"
        + "User input:\n"
        + text + "\n\n"
        + ( "If the input is a description of something on-screen, produce a concise explanation tailored for a job interview candidate." )
    )

    # Prefer OpenAI if key set
    if OPENAI_API_KEY:
        try:
            reply_text = await call_openai_chat(prompt, max_tokens=max_tokens)
            # return as single reply with optional short suggestions from model (none for now)
            return {"reply": reply_text, "suggestions": None, "source": "openai"}
        except Exception as e:
            logger.exception("OpenAI call failed, falling back to mock: %s", e)

    # Fallback: mock generator
    reply, suggestions = _mock_quick_reply(text, simplify)
    return {"reply": reply, "suggestions": suggestions, "source": "mock"}
