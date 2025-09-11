# routes.py
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

from schemas import QuickRequest, QuickResponse
from service import generate_quick_reply

logger = logging.getLogger("quickrespond.api")
logger.setLevel(logging.INFO)

app = FastAPI(title="QuickRespond API", version="0.1.0")

# Allow the frontend dev server origin (edit for production)
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"],  # during development it's okay, lock this down for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api", tags=["quickrespond"])

@router.post("/quick-respond", response_model=QuickResponse)
async def quick_respond(payload: QuickRequest):
    """
    Accepts { text, simplify (optional), max_tokens (optional) } and returns { reply, suggestions, source }.
    """
    try:
        result = await generate_quick_reply(payload.text, simplify=payload.simplify, max_tokens=payload.max_tokens or 80)
        # Ensure shape
        return QuickResponse(reply=result.get("reply", ""), suggestions=result.get("suggestions"), source=result.get("source", "mock"))
    except Exception as e:
        logger.exception("quick-respond failed: %s", e)
        raise HTTPException(status_code=500, detail="Server error generating quick reply")

app.include_router(router)

if __name__ == "__main__":
    # For local dev: uvicorn routes:app --reload
    uvicorn.run("routes:app", host="0.0.0.0", port=3000, reload=True)
