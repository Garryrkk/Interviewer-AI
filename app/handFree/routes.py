from fastapi import APIRouter, HTTPException
from .schemas import HandsFreeRequest, HandsFreeResponse
from .service import process_handsfree

router = APIRouter()

@router.post("/handsfree", response_model=HandsFreeResponse)
async def handsfree_mode(request: HandsFreeRequest):
    try:
        result = process_handsfree(request.audio_base64, request.expression)
        return HandsFreeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
