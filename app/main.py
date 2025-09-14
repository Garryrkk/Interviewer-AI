# app/main.py
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError, ResponseValidationError
import uvicorn
from typing import Dict, Any
import time
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback

# Import all schemas - keeping them as requested
from app.summarization.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    SummarizeError
)

from app.summarization.service import (
    SummarizationService,
    generate_summary_async
)

from app.handFree.service import (
    HandsFreeService,
    generate_handsfree_response_async
)

from app.handFree.schemas import (
    HandsFreeRequest,
    HandsFreeResponse
)

from app.quick_respond.schemas import (
    QuickResponseRequest,
    QuickResponse,
    QuickResponseError
)

from app.quick_respond.service import (
    QuickResponseService,
    generate_quick_response,
    generate_quick_response_async
)

from app.MainFeature.schemas import (
    InvisibilityStartRequest,
    InvisibilitySendRequest,
    InvisibilityEndRequest,
    InvisibilityResponse
)
from app.MainFeature.service import (
    start_invisibility_session,
    send_invisible_response,
    end_invisibility_session,
    get_invisible_responses
)

from app.image_recognition.schemas import (
    ScreenCaptureRequest,
    ScreenCaptureResponse,
    CameraCaptureRequest,
    CameraCaptureResponse
)

from app.image_recognition.service import (
    analyze_screen,
    analyze_camera
)

from app.key_insights.schemas import (
    KeyInsight,
    KeyInsightRequest,
    KeyInsightResponse
)

from app.key_insights.services import (
    KeyInsight,
    KeyInsightRequest,
    KeyInsightsService,
    extract_key_insights
)

from app.voice_recognition.schemas import (
    PreopConfig,
    PreopResponse,
    AudioChunk,
    ProcessingResponse,
    RecognitionSession,
    RecognitionResponse,
    SessionControl,
    SessionStatus,
    ErrorResponse
)
from app.voice_recognition.services import voice_service

# Import routes - keeping them as requested
from app.summarization import routes as summarization_routes
from app.key_insights import routes as insights_routes
from app.quick_respond import routes as quickrespond_routes
from app.MainFeature import routes as mainfeature_routes
from app.handFree import routes as handsfree_routes
from app.image_recognition import routes as image_recognition_routes
from app.voice_recognition import routes as voice_recognition_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state for interview sessions
interview_sessions: Dict[str, Dict[str, Any]] = {}

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Interview AI Assistant Backend Starting...")
    
    # Initialize services
    try:
        # Initialize Ollama connection
        await initialize_ollama_connection()
        
        # Initialize voice service
        await voice_service.initialize()
        
        # Initialize other services
        await initialize_core_services()
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Interview AI Assistant Backend Shutting Down...")
    
    # Cleanup services
    try:
        await cleanup_services()
        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

async def initialize_ollama_connection():
    """Initialize connection to Ollama with Nous Hermes model"""
    try:
        # Test Ollama connection
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            hermes_models = [m for m in models if "nous-hermes" in m.get("name", "").lower()]
            if hermes_models:
                logger.info(f"✅ Ollama connected. Found Nous Hermes models: {[m['name'] for m in hermes_models]}")
            else:
                logger.warning("⚠️  Ollama connected but Nous Hermes model not found")
        else:
            raise Exception(f"Ollama not responding: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Ollama: {e}")
        raise

async def initialize_core_services():
    """Initialize core AI services"""
    try:
        # Initialize summarization service
        SummarizationService.initialize()
        
        # Initialize quick response service
        QuickResponseService.initialize()
        
        # Initialize hands-free service
        HandsFreeService.initialize()
        
        # Initialize key insights service
        KeyInsightsService.initialize()
        
        logger.info("✅ Core AI services initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize core services: {e}")
        raise

async def cleanup_services():
    """Cleanup services and resources"""
    try:
        # Cleanup voice service
        await voice_service.cleanup()
        
        # Clear active sessions
        interview_sessions.clear()
        
        logger.info("✅ Services cleaned up successfully")
    except Exception as e:
        logger.error(f"❌ Error during service cleanup: {e}")

# Create FastAPI App with lifespan
app = FastAPI(
    title="Interview AI Assistant Backend",
    description="Intelligent interview support system with invisibility mode, real-time analysis, and adaptive responses using Ollama & Nous Hermes",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security Middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

# CORS Middleware - Enhanced for interview scenarios
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # React dev server
        "http://localhost:5173",     # Vite dev server  
        "http://localhost:8080",     # Alternative dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization", 
        "X-Requested-With",
        "Accept",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "X-Session-ID",
        "X-Interview-Mode"
    ],
    expose_headers=["X-Session-ID", "X-Response-Time"]
)

# Custom middleware for request timing and session tracking
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Enhanced Exception handlers
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed logging"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail} - Path: {request.url.path} - Method: {request.method}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": time.time()
        }
    )

@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle Starlette HTTP exceptions"""
    logger.warning(f"Starlette HTTP Exception: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed field information"""
    logger.error(f"Validation Error: {exc.errors()} - Path: {request.url.path} - Method: {request.method}")
    
    # Extract detailed error information
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": error_details,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": time.time()
        }
    )

@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    """Handle response validation errors"""
    logger.error(f"Response Validation Error: {exc.errors()} - Path: {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Response Validation Error",
            "message": "Server response validation failed",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )

@app.exception_handler(asyncio.TimeoutError)
async def timeout_exception_handler(request: Request, exc: asyncio.TimeoutError):
    """Handle timeout errors"""
    logger.error(f"Timeout Error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=408,
        content={
            "error": "Timeout Error",
            "message": "Request timeout - operation took too long to complete",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )

@app.exception_handler(ConnectionError)
async def connection_exception_handler(request: Request, exc: ConnectionError):
    """Handle connection errors (e.g., Ollama service down)"""
    logger.error(f"Connection Error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Unavailable",
            "message": "Unable to connect to required services",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    """Handle value errors with context"""
    logger.error(f"Value Error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid Value",
            "message": f"Invalid input value: {str(exc)}",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )

@app.exception_handler(KeyError)
async def key_error_exception_handler(request: Request, exc: KeyError):
    """Handle missing key errors"""
    logger.error(f"Key Error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Missing Required Field",
            "message": f"Required field missing: {str(exc)}",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_exception_handler(request: Request, exc: FileNotFoundError):
    """Handle file not found errors"""
    logger.error(f"File Not Found Error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "File Not Found",
            "message": f"Required file not found: {exc.filename}",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )

@app.exception_handler(PermissionError)
async def permission_exception_handler(request: Request, exc: PermissionError):
    """Handle permission errors"""
    logger.error(f"Permission Error: {str(exc)} - Path: {request.url.path}")
    return JSONResponse(
        status_code=403,
        content={
            "error": "Permission Denied",
            "message": "Insufficient permissions to perform this operation",
            "path": str(request.url.path),
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other unhandled exceptions with full traceback logging"""
    # Get full traceback for logging
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb)
    
    logger.error(f"Unhandled Exception: {str(exc)} - Path: {request.url.path} - Method: {request.method}")
    logger.error(f"Full Traceback:\n{tb_str}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "exception_type": type(exc).__name__,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": time.time()
        }
    )

# Include routers with proper prefixes and tags
app.include_router(
    summarization_routes.router, 
    prefix="/api/v1/summarization",
    tags=["Summarization"]
)

app.include_router(
    insights_routes.router, 
    prefix="/api/v1/insights", 
    tags=["Key Insights"]
)

app.include_router(
    quickrespond_routes.router,
    prefix="/api/v1/quick-response", 
    tags=["Quick Response"]
)

app.include_router(
    mainfeature_routes.router,
    prefix="/api/v1/invisibility",
    tags=["Main Feature - Invisibility Mode"]
)

app.include_router(
    handsfree_routes.router,
    prefix="/api/v1/hands-free",
    tags=["Hands-Free Operation"]
)

app.include_router(
    image_recognition_routes.router,
    prefix="/api/v1/vision",
    tags=["Image Recognition & Screen Analysis"]
)

app.include_router(
    voice_recognition_routes.router,
    prefix="/api/v1/voice",
    tags=["Voice Recognition & Processing"]
)

# Enhanced root endpoint with system status
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with system status"""
    try:
        # Check Ollama connection
        import requests
        ollama_status = "connected"
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                ollama_status = "disconnected"
        except:
            ollama_status = "disconnected"
        
        return {
            "message": "🎯 Interview AI Assistant Backend is running",
            "version": "2.0.0",
            "status": "operational",
            "features": [
                "Invisibility Mode",
                "Real-time Voice Recognition", 
                "Screen & Camera Analysis",
                "Quick Response Generation",
                "Adaptive Interview Support",
                "Hands-free Operation"
            ],
            "ai_model": "Nous Hermes (via Ollama)",
            "ollama_status": ollama_status,
            "active_sessions": len(interview_sessions),
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health"
            }
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        return {"message": "Interview AI Assistant Backend", "status": "error"}

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "ollama": "unknown",
            "voice_service": "unknown",
            "core_services": "unknown"
        },
        "memory": {
            "active_sessions": len(interview_sessions)
        }
    }
    
    try:
        # Check Ollama
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        health_status["services"]["ollama"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        health_status["services"]["ollama"] = "unhealthy"
    
    try:
        # Check voice service
        health_status["services"]["voice_service"] = "healthy" if hasattr(voice_service, 'is_initialized') else "unknown"
    except:
        health_status["services"]["voice_service"] = "unhealthy"
    
    # Check if any critical service is down
    if "unhealthy" in health_status["services"].values():
        health_status["status"] = "degraded"
    
    return health_status

# Interview session management endpoints
@app.post("/api/v1/interview/session/start", tags=["Interview Session"])
async def start_interview_session(session_data: Dict[str, Any] = None):
    """Start a new interview session"""
    try:
        import uuid
        session_id = str(uuid.uuid4())
        
        interview_sessions[session_id] = {
            "id": session_id,
            "start_time": time.time(),
            "status": "active",
            "data": session_data or {},
            "interactions": []
        }
        
        logger.info(f"Started interview session: {session_id}")
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Interview session initialized successfully"
        }
    except Exception as e:
        logger.error(f"Error starting interview session: {e}")
        raise HTTPException(status_code=500, detail="Failed to start interview session")

@app.get("/api/v1/interview/session/{session_id}/status", tags=["Interview Session"])
async def get_session_status(session_id: str):
    """Get interview session status"""
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = interview_sessions[session_id]
    return {
        "session_id": session_id,
        "status": session["status"],
        "start_time": session["start_time"],
        "duration": time.time() - session["start_time"],
        "interactions_count": len(session["interactions"])
    }

@app.delete("/api/v1/interview/session/{session_id}", tags=["Interview Session"])
async def end_interview_session(session_id: str):
    """End an interview session"""
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = interview_sessions[session_id]
    session["status"] = "ended"
    session["end_time"] = time.time()
    
    logger.info(f"Ended interview session: {session_id}")
    return {
        "session_id": session_id,
        "status": "ended",
        "message": "Interview session ended successfully"
    }

# Configuration endpoint for Ollama model
@app.post("/api/v1/config/model", tags=["Configuration"])
async def configure_ollama_model(model_config: Dict[str, Any]):
    """Configure Ollama model settings"""
    try:
        # Validate model exists
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            requested_model = model_config.get("model_name", "")
            if requested_model and requested_model not in model_names:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model {requested_model} not found. Available models: {model_names}"
                )
        
        return {
            "status": "configured",
            "model_config": model_config,
            "message": "Model configuration updated successfully"
        }
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    except Exception as e:
        logger.error(f"Error configuring model: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure model")

# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )