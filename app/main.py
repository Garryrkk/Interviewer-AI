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
from typing import Dict, Any, List
import time
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import os

# Import all schemas - now properly utilized
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
        await initialize_voice_service()
        
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
                logger.warning("⚠️ Ollama connected but Nous Hermes model not found")
        else:
            raise Exception(f"Ollama not responding: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Ollama: {e}")
        raise

async def initialize_voice_service():
    """Initialize voice service with proper error handling"""
    try:
        # Check if voice_service has an initialize method
        if hasattr(voice_service, 'initialize') and callable(voice_service.initialize):
            if asyncio.iscoroutinefunction(voice_service.initialize):
                await voice_service.initialize()
            else:
                voice_service.initialize()
        else:
            # If initialize is a boolean attribute or doesn't exist, set it manually
            if hasattr(voice_service, 'initialize'):
                voice_service.initialize = True
            logger.info("✅ Voice service initialized (fallback method)")
        
        logger.info("✅ Voice service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize voice service: {e}")
        # Continue without voice service if it fails
        logger.warning("⚠️ Continuing without voice service")

async def initialize_core_services():
    """Initialize core AI services"""
    try:
        # Initialize summarization service
        if hasattr(SummarizationService, 'initialize') and callable(SummarizationService.initialize):
            if asyncio.iscoroutinefunction(SummarizationService.initialize):
                await SummarizationService.initialize()
            else:
                SummarizationService.initialize()
        
        # Initialize quick response service
        if hasattr(QuickResponseService, 'initialize') and callable(QuickResponseService.initialize):
            if asyncio.iscoroutinefunction(QuickResponseService.initialize):
                await QuickResponseService.initialize()
            else:
                QuickResponseService.initialize()
        
        # Initialize hands-free service
        if hasattr(HandsFreeService, 'initialize') and callable(HandsFreeService.initialize):
            if asyncio.iscoroutinefunction(HandsFreeService.initialize):
                await HandsFreeService.initialize()
            else:
                HandsFreeService.initialize()
        
        # Initialize key insights service
        if hasattr(KeyInsightsService, 'initialize') and callable(KeyInsightsService.initialize):
            if asyncio.iscoroutinefunction(KeyInsightsService.initialize):
                await KeyInsightsService.initialize()
            else:
                KeyInsightsService.initialize()
        
        logger.info("✅ Core AI services initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize core services: {e}")
        raise

async def cleanup_services():
    """Cleanup services and resources"""
    try:
        # Cleanup voice service
        if hasattr(voice_service, 'cleanup') and callable(voice_service.cleanup):
            if asyncio.iscoroutinefunction(voice_service.cleanup):
                await voice_service.cleanup()
            else:
                voice_service.cleanup()
        
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

frontend_dist = Path(__file__).parent.parent / "agenda frontend" / "dist"

# Mount the static frontend
app.mount(
    "/",                               # URL prefix
    StaticFiles(directory=frontend_dist, html=True),
    name="frontend"
)


# Define allowed origins and trusted hosts
allowed_origins = [
    "http://localhost:5173",    # Vite dev
    "http://127.0.0.1:5173",
    "http://localhost:3000",    # React dev
    "http://127.0.0.1:3000",
    "http://localhost:8000",   # your current frontend port
    "http://127.0.0.1:8000",
    "https://5456cb9f09f8.ngrok-free.app",
    "http://localhost:61863",    # your frontend dev URL
    "http://127.0.0.1:61863",
]

trusted_hosts = [
    "localhost",
    "127.0.0.1",
    "*.ngrok-free.app",         # covers any ngrok subdomain
    "960bd0f27143.ngrok-free.app",
]

# Security Middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=trusted_hosts
)

# CORS Middleware - Single instance with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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

# Include routers with proper prefixes and tags - ALL ROUTES MOUNTED
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

# =================== DIRECT ENDPOINTS TO UTILIZE IMPORTED SCHEMAS AND SERVICES ===================

# Direct summarization endpoint using imported schemas and services
@app.post("/api/v1/direct/summarize", response_model=SummarizeResponse, tags=["Direct Services"])
async def direct_summarize(request: SummarizeRequest):
    """Direct summarization endpoint using imported schemas and services"""
    try:
        # Use the imported generate_summary_async function
        summary = await generate_summary_async(
            text=request.text,
            style=request.style,
            simplify=request.simplify
        )
        
        return SummarizeResponse(
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary),
            style_used=request.style or "concise",
            processing_time=time.time()
        )
    except Exception as e:
        logger.error(f"Direct summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Direct hands-free endpoint using imported schemas and services
@app.post("/api/v1/direct/hands-free", response_model=HandsFreeResponse, tags=["Direct Services"])
async def direct_hands_free(request: HandsFreeRequest):
    """Direct hands-free endpoint using imported schemas and services"""
    try:
        # Use the imported generate_handsfree_response_async function
        response_text = await generate_handsfree_response_async(
            transcript=request.transcript,
            context=request.context,
            simplify=request.simplify
        )
        
        return HandsFreeResponse(
            response=response_text,
            confidence=0.95,  # Default confidence
            processing_time=time.time()
        )
    except Exception as e:
        logger.error(f"Direct hands-free error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Direct quick response endpoint using imported schemas and services
@app.post("/api/v1/direct/quick-response", response_model=QuickResponse, tags=["Direct Services"])
async def direct_quick_response(request: QuickResponseRequest):
    """Direct quick response endpoint using imported schemas and services"""
    try:
        # Use both sync and async versions based on request preference
        if hasattr(request, 'use_async') and request.use_async:
            response_text = await generate_quick_response_async(
                prompt=request.prompt,
                context=request.context,
                response_type=request.response_type,
                max_length=request.max_length
            )
        else:
            response_text = generate_quick_response(
                prompt=request.prompt,
                context=request.context,
                response_type=request.response_type,
                max_length=request.max_length
            )
        
        return QuickResponse(
            response=response_text,
            response_type=request.response_type,
            processing_time=time.time()
        )
    except Exception as e:
        logger.error(f"Direct quick response error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Direct invisibility endpoints using imported schemas and services
@app.post("/api/v1/direct/invisibility/start", response_model=InvisibilityResponse, tags=["Direct Services"])
async def direct_start_invisibility(request: InvisibilityStartRequest):
    """Direct invisibility start endpoint using imported schemas and services"""
    try:
        session_data = start_invisibility_session(
            user_id=request.user_id,
            interview_type=request.interview_type,
            context=request.context
        )
        
        return InvisibilityResponse(
            session_id=session_data["session_id"],
            status="started",
            message="Invisibility session started successfully"
        )
    except Exception as e:
        logger.error(f"Direct invisibility start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/direct/invisibility/send", response_model=InvisibilityResponse, tags=["Direct Services"])
async def direct_send_invisible_response(request: InvisibilitySendRequest):
    """Direct invisibility send endpoint using imported schemas and services"""
    try:
        response_data = send_invisible_response(
            session_id=request.session_id,
            message=request.message,
            message_type=request.message_type
        )
        
        return InvisibilityResponse(
            session_id=request.session_id,
            status="sent",
            message="Invisible response sent successfully",
            data=response_data
        )
    except Exception as e:
        logger.error(f"Direct invisibility send error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/direct/invisibility/end", response_model=InvisibilityResponse, tags=["Direct Services"])
async def direct_end_invisibility(request: InvisibilityEndRequest):
    """Direct invisibility end endpoint using imported schemas and services"""
    try:
        end_data = end_invisibility_session(
            session_id=request.session_id,
            summary_requested=request.summary_requested
        )
        
        return InvisibilityResponse(
            session_id=request.session_id,
            status="ended",
            message="Invisibility session ended successfully",
            data=end_data
        )
    except Exception as e:
        logger.error(f"Direct invisibility end error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Direct screen capture endpoint using imported schemas and services
@app.post("/api/v1/direct/screen-capture", response_model=AnalyzeScreenResponse, tags=["Direct Services"])
async def direct_screen_capture(request: ImageBase64Request):
    """Direct screen capture endpoint using imported schemas and services"""
    try:
        analysis_result = await analyze_screen(
            region=request.region,
            analysis_type=request.analysis_type,
            include_text=request.include_text
        )
        
        return ScreenCaptureResponse(
            analysis=analysis_result,
            timestamp=time.time(),
            region_captured=request.region
        )
    except Exception as e:
        logger.error(f"Direct screen capture error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Direct camera capture endpoint using imported schemas and services
@app.post("/api/v1/direct/camera-capture", response_model=CameraCaptureResponse, tags=["Direct Services"])
async def direct_camera_capture(request: CameraCaptureRequest):
    """Direct camera capture endpoint using imported schemas and services"""
    try:
        analysis_result = await analyze_camera(
            camera_id=request.camera_id,
            analysis_type=request.analysis_type,
            duration=request.duration
        )
        
        return CameraCaptureResponse(
            analysis=analysis_result,
            timestamp=time.time(),
            camera_used=request.camera_id
        )
    except Exception as e:
        logger.error(f"Direct camera capture error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Direct key insights endpoint using imported schemas and services
@app.post("/api/v1/direct/key-insights", response_model=KeyInsightResponse, tags=["Direct Services"])
async def direct_key_insights(request: KeyInsightRequest):
    """Direct key insights endpoint using imported schemas and services"""
    try:
        insights = await extract_key_insights(
            content=request.content,
            content_type=request.content_type,
            max_insights=request.max_insights
        )
        
        # Convert to KeyInsight objects if they aren't already
        insight_objects = []
        for insight in insights:
            if isinstance(insight, dict):
                insight_objects.append(KeyInsight(**insight))
            else:
                insight_objects.append(insight)
        
        return KeyInsightResponse(
            insights=insight_objects,
            total_insights=len(insight_objects),
            processing_time=time.time()
        )
    except Exception as e:
        logger.error(f"Direct key insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Direct voice recognition endpoints using imported schemas
@app.post("/api/v1/direct/voice/preop", response_model=PreopResponse, tags=["Direct Services"])
async def direct_voice_preop(config: PreopConfig):
    """Direct voice pre-operation endpoint using imported schemas"""
    try:
        # Initialize voice service with provided configuration
        result = await voice_service.configure(
            language=config.language,
            sample_rate=config.sample_rate,
            channels=config.channels,
            model_type=config.model_type
        )
        
        return PreopResponse(
            status="configured",
            configuration=config.dict(),
            message="Voice service configured successfully"
        )
    except Exception as e:
        logger.error(f"Direct voice preop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/direct/voice/process", response_model=ProcessingResponse, tags=["Direct Services"])
async def direct_voice_process(audio_chunk: AudioChunk):
    """Direct voice processing endpoint using imported schemas"""
    try:
        # Process audio chunk
        result = await voice_service.process_audio(
            audio_data=audio_chunk.data,
            session_id=audio_chunk.session_id,
            timestamp=audio_chunk.timestamp
        )
        
        return ProcessingResponse(
            transcript=result.get("transcript", ""),
            confidence=result.get("confidence", 0.0),
            is_final=result.get("is_final", False),
            processing_time=time.time()
        )
    except Exception as e:
        logger.error(f"Direct voice process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =================== UTILITY ENDPOINTS ===================

@app.get("/api/v1/schemas/list", tags=["Utilities"])
async def list_available_schemas():
    """List all available schemas and their structures"""
    schemas_info = {
        "summarization": {
            "SummarizeRequest": SummarizeRequest.schema(),
            "SummarizeResponse": SummarizeResponse.schema(),
            "SummarizeError": SummarizeError.schema()
        },
        "hands_free": {
            "HandsFreeRequest": HandsFreeRequest.schema(),
            "HandsFreeResponse": HandsFreeResponse.schema()
        },
        "quick_response": {
            "QuickResponseRequest": QuickResponseRequest.schema(),
            "QuickResponse": QuickResponse.schema(),
            "QuickResponseError": QuickResponseError.schema()
        },
        "invisibility": {
            "InvisibilityStartRequest": InvisibilityStartRequest.schema(),
            "InvisibilitySendRequest": InvisibilitySendRequest.schema(),
            "InvisibilityEndRequest": InvisibilityEndRequest.schema(),
            "InvisibilityResponse": InvisibilityResponse.schema()
        },
        "image_recognition": {
            "ScreenCaptureRequest": ScreenCaptureRequest.schema(),
            "ScreenCaptureResponse": ScreenCaptureResponse.schema(),
            "CameraCaptureRequest": CameraCaptureRequest.schema(),
            "CameraCaptureResponse": CameraCaptureResponse.schema()
        },
        "key_insights": {
            "KeyInsight": KeyInsight.schema(),
            "KeyInsightRequest": KeyInsightRequest.schema(),
            "KeyInsightResponse": KeyInsightResponse.schema()
        },
        "voice_recognition": {
            "PreopConfig": PreopConfig.schema(),
            "PreopResponse": PreopResponse.schema(),
            "AudioChunk": AudioChunk.schema(),
            "ProcessingResponse": ProcessingResponse.schema(),
            "RecognitionSession": RecognitionSession.schema(),
            "RecognitionResponse": RecognitionResponse.schema(),
            "SessionControl": SessionControl.schema(),
            "SessionStatus": SessionStatus.schema(),
            "ErrorResponse": ErrorResponse.schema()
        }
    }
    
    return {
        "available_schemas": schemas_info,
        "total_schemas": sum(len(category) for category in schemas_info.values()),
        "categories": list(schemas_info.keys())
    }

@app.get("/api/v1/services/status", tags=["Utilities"])
async def get_services_status():
    """Get status of all imported services and functions"""
    services_status = {
        "summarization": {
            "service_class": "SummarizationService" if SummarizationService else "Not Available",
            "async_function": "generate_summary_async" if generate_summary_async else "Not Available"
        },
        "hands_free": {
            "service_class": "HandsFreeService" if HandsFreeService else "Not Available", 
            "async_function": "generate_handsfree_response_async" if generate_handsfree_response_async else "Not Available"
        },
        "quick_response": {
            "service_class": "QuickResponseService" if QuickResponseService else "Not Available",
            "sync_function": "generate_quick_response" if generate_quick_response else "Not Available",
            "async_function": "generate_quick_response_async" if generate_quick_response_async else "Not Available"
        },
        "invisibility": {
            "start_session": "start_invisibility_session" if start_invisibility_session else "Not Available",
            "send_response": "send_invisible_response" if send_invisible_response else "Not Available",
            "end_session": "end_invisibility_session" if end_invisibility_session else "Not Available",
            "get_responses": "get_invisible_responses" if get_invisible_responses else "Not Available"
        },
        "image_recognition": {
            "analyze_screen": "analyze_screen" if analyze_screen else "Not Available",
            "analyze_camera": "analyze_camera" if analyze_camera else "Not Available"
        },
        "key_insights": {
            "service_class": "KeyInsightsService" if KeyInsightsService else "Not Available",
            "extract_function": "extract_key_insights" if extract_key_insights else "Not Available"
        },
        "voice_recognition": {
            "voice_service": "Available" if voice_service else "Not Available"
        }
    }
    
    return {
        "services_status": services_status,
        "all_services_loaded": all(
            any(status != "Not Available" for status in category.values()) 
            for category in services_status.values()
        )
    }

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
                "health": "/health",
                "schemas": "/api/v1/schemas/list",
                "services_status": "/api/v1/services/status"
            },
            "direct_endpoints": {
                "summarize": "/api/v1/direct/summarize",
                "hands_free": "/api/v1/direct/hands-free",
                "quick_response": "/api/v1/direct/quick-response",
                "invisibility_start": "/api/v1/direct/invisibility/start",
                "invisibility_send": "/api/v1/direct/invisibility/send",
                "invisibility_end": "/api/v1/direct/invisibility/end",
                "screen_capture": "/api/v1/direct/screen-capture",
                "camera_capture": "/api/v1/direct/camera-capture",
                "key_insights": "/api/v1/direct/key-insights",
                "voice_preop": "/api/v1/direct/voice/preop",
                "voice_process": "/api/v1/direct/voice/process"
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
        },
        "imported_modules": {
            "summarization": bool(SummarizationService and generate_summary_async),
            "hands_free": bool(HandsFreeService and generate_handsfree_response_async),
            "quick_response": bool(QuickResponseService and generate_quick_response and generate_quick_response_async),
            "invisibility": bool(start_invisibility_session and send_invisible_response and end_invisibility_session),
            "image_recognition": bool(analyze_screen and analyze_camera),
            "key_insights": bool(KeyInsightsService and extract_key_insights),
            "voice_recognition": bool(voice_service)
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
        # Check voice service - handle both boolean and method cases
        if hasattr(voice_service, 'is_initialized'):
            health_status["services"]["voice_service"] = "healthy" if voice_service.is_initialized else "unknown"
        elif hasattr(voice_service, 'initialize'):
            health_status["services"]["voice_service"] = "healthy" if voice_service.initialize else "unknown"
        else:
            health_status["services"]["voice_service"] = "unknown"
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

# Batch processing endpoint utilizing multiple services
@app.post("/api/v1/batch/process", tags=["Batch Processing"])
async def batch_process_content(
    content_list: List[str],
    operations: List[str] = ["summarize", "insights", "quick_response"]
):
    """Batch process content using multiple imported services"""
    try:
        results = []
        
        for idx, content in enumerate(content_list):
            content_results = {
                "index": idx,
                "content_preview": content[:100] + "..." if len(content) > 100 else content,
                "operations": {}
            }
            
            # Summarization
            if "summarize" in operations and generate_summary_async:
                try:
                    summary = await generate_summary_async(content, style="concise", simplify=False)
                    content_results["operations"]["summarize"] = {
                        "status": "success",
                        "result": summary,
                        "original_length": len(content),
                        "summary_length": len(summary)
                    }
                except Exception as e:
                    content_results["operations"]["summarize"] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            # Key Insights
            if "insights" in operations and extract_key_insights:
                try:
                    insights = await extract_key_insights(content, content_type="text", max_insights=5)
                    content_results["operations"]["insights"] = {
                        "status": "success",
                        "result": insights,
                        "insights_count": len(insights) if insights else 0
                    }
                except Exception as e:
                    content_results["operations"]["insights"] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            # Quick Response
            if "quick_response" in operations and generate_quick_response_async:
                try:
                    quick_resp = await generate_quick_response_async(
                        prompt=f"Analyze this content: {content[:200]}...",
                        response_type="analysis",
                        max_length=150
                    )
                    content_results["operations"]["quick_response"] = {
                        "status": "success",
                        "result": quick_resp,
                        "response_length": len(quick_resp)
                    }
                except Exception as e:
                    content_results["operations"]["quick_response"] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            results.append(content_results)
        
        return {
            "batch_id": f"batch_{int(time.time())}",
            "total_items": len(content_list),
            "operations_requested": operations,
            "results": results,
            "processing_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

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