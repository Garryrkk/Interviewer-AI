from fastapi import APIRouter, HTTPException, status
import logging
import requests
import json
from typing import Dict, Any
from .schemas import QuickResponseRequest, QuickResponse

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"  
OLLAMA_MODEL = "nous-hermes2" 

router = APIRouter(
    prefix="/quick-respond",
    tags=["Quick Respond"],
    responses={404: {"description": "Not found"}}
)

async def generate_quick_response_ollama(prompt: str) -> str:
    """Generate response using Ollama API"""
    try:
        payload: Dict[str, Any] = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        }
        
        # Make request to Ollama
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract the generated text
        if "response" in result:
            return result["response"]
        else:
            raise Exception("Invalid response format from Ollama")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request failed: {str(e)}")
        raise Exception(f"Failed to connect to Ollama: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama response: {str(e)}")
        raise Exception("Invalid response from Ollama")
    except Exception as e:
        logger.error(f"Error in Ollama generation: {str(e)}")
        raise

@router.post("/", response_model=QuickResponse)
async def quick_response(request: QuickResponseRequest) -> QuickResponse:
    """Generate a quick response using Ollama"""
    try:
        # Validate prompt is not empty
        if not request.prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty"
            )
        
        # Log the incoming request
        logger.info(f"Received quick response request for prompt: {request.prompt[:50]}...")
        
        # Generate response using Ollama
        response_text = await generate_quick_response_ollama(request.prompt)
        
        # Log successful response
        logger.info("Quick response generated successfully via Ollama")
        
        return QuickResponse(response=response_text)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        logger.error(f"Error generating quick response: {str(e)}")
        
        # Return generic error to client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate quick response"
        )

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for the Quick Respond service and Ollama connection.
    
    Returns:
        dict: Service status including Ollama connectivity
    """
    try:
        # Test Ollama connection
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "disconnected"
    except Exception:
        ollama_status = "disconnected"
    
    return {
        "status": "healthy",
        "service": "quick-respond",
        "message": "Quick Respond service is running",
        "ollama_status": ollama_status,
        "model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_BASE_URL
    }

@router.post("/validate")
async def validate_prompt(request: QuickResponseRequest) -> Dict[str, Any]:
    """Validate the prompt before processing"""
    try:
        if not request.prompt.strip():
            return {
                "valid": False,
                "message": "Prompt cannot be empty"
            }
        
        if len(request.prompt) > 4000:  # Increased limit for local models
            return {
                "valid": False,
                "message": "Prompt too long (max 4000 characters)"
            }
        
        return {
            "valid": True,
            "message": "Prompt is valid"
        }
        
    except Exception as e:
        logger.error(f"Error validating prompt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate prompt"
        )

@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """List available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        
        models = response.json()
        return {
            "available_models": models.get("models", []),
            "current_model": OLLAMA_MODEL
        }
        
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch available models"
        )

@router.post("/switch-model")
async def switch_model(model_name: str) -> Dict[str, str]:
    """Switch to a different Ollama model"""
    global OLLAMA_MODEL
    
    try:
        # Verify model exists
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        
        models = response.json()
        available_models = [model["name"] for model in models.get("models", [])]
        
        if model_name not in available_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        previous_model = OLLAMA_MODEL
        OLLAMA_MODEL = "nous_hermes"
        logger.info(f"Switched to model: {model_name}")
        
        return {
            "message": f"Successfully switched to model: {model_name}",
            "previous_model": previous_model,
            "current_model": model_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to switch models"
        )