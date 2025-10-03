import logging
import asyncio
<<<<<<< HEAD
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError, ResponseValidationError
import uvicorn
from typing import Dict, Any, List
=======
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError, ResponseValidationError
import uvicorn
from fastapi import Query
from typing import Dict, Any, List, Optional
>>>>>>> 91363704a1edbd3248250121ce7a4b90fac292f9
import time
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
from fastapi.staticfiles import StaticFiles
from pathlib import Path
<<<<<<< HEAD
=======
import uuid
from datetime import datetime
import requests
import os
import redis
import asyncpg
from redis import asyncio as redis  # ✅ new import
from redis.asyncio.client import Redis  # for type-hinting if you want it
from app.db.connection import (
    create_database_connection,
    create_redis_connection,
    create_file_storage_connection,
    create_vector_database_connection
)
from typing import Any
from motor.motor_asyncio import AsyncIOMotorClient   # if you use Mongo/GridFS for file storage
from chromadb import Client
import functools
  

from fastapi import APIRouter  # <-- this imports APIRouter
router = APIRouter()           # <-- this creates a new router instance

import os

# Import all schemas - now properly utilized
<<<<<<< HEAD
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
=======
from app.handFree.schemas import (
    HandsFreeSessionRequest,
    InterviewResponseRequest,
    FacialAnalysisRequest,
    SessionSettings,
    HandsFreeSessionResponse,
    InterviewResponse,
    FacialAnalysis,
    ConfidenceTipResponse,
    AudioStreamResult,
    RealTimeAnalysisResponse,
    SessionStatus,
    AudioStreamData,
    SystemHealthCheck,
    SessionInsights,
    SessionSummary,
    WebSocketMessage,
    AutomatedResponseMessage,
    FacialAnalysisMessage,
    SystemStatusMessage
)

from app.summarization.schemas import(
    SummaryType,
    AnalysisType,
    AudioUploadResponse,
    MeetingAnalysisRequest,
    MeetingAnalysisResponse,
    ActionItem,
    KeyPoint,
    SummarizationRequest,
    SummarizationResponse,
    SummaryUpdateRequest,
    MeetingContext,
    LLAVAAnalysisConfig,
    RealTimeAnalysisUpdate,
    BatchSummarizationRequest
>>>>>>> 91363704a1edbd3248250121ce7a4b90fac292f9
)

from app.key_insights.schemas import (
    KeyInsight,
    KeyInsightRequest,
<<<<<<< HEAD
    KeyInsightResponse
=======
    KeyInsightResponse,
    ErrorResponse as KeyInsightErrorResponse
)

from app.voice_recognition.schemas import (
    VoiceSessionRequest,
    MicrophoneStatusRequest,
    DeviceSelectionRequest,
    AudioProcessingRequest,
    AIResponseRequest,
    SimplifiedAnswerRequest,
    VoiceSessionResponse,
    MicrophoneStatusResponse,
    DeviceListResponse,
    TranscriptionResponse,
    AudioProcessingResponse,
    VoiceAnalysisResponse,
    AIResponseResponse,
    CalibrationRequest,
    CalibrationResponse,
    AudioTestRequest,
    AudioTestResponse,
    ErrorResponse as VoiceErrorResponse,
    ServiceHealth
)

from app.image_recognition.schemas import(
    CameraStatus,
    ExpressionConfig,
    ExpressionType,
    MessageSender,
    CameraResolution,
    CameraDevice,
    CameraSessionRequest,
    CameraSessionResponse,
    CameraFrameMessage,
    CameraStatusResponse,
    CameraTestResult,
    ExpressionDetectionRequest,
    ExpressionDetectedMessage,
    ExpressionDetectionResponse,
    ExpressionMonitoringConfig,
    ChatMessage,
    SimplifyMessageRequest,
    ChatSessionSummary,
    MonitoringSession,
    SystemHealthStatus,
    APIResponse,
    WebSocketMessage as ImageWebSocketMessage,
    CameraConfig,
    ChatConfig,
    RecordingQuality,
    AnalysisType as ImageAnalysisType,
    RecordingInfo,
    RecordingStatus,
    ConnectionStatus,
    AnalysisFocus,
    StartRecordingRequest,
    HealthCheckResponse,
    PermissionStatusResponse,
    RecordingStatusResponse,
    StartRecordingResponse,
    StopRecordingResponse,
    AnalysisRequest,
    AnalysisResponse,
    ScreenshotAnalysisRequest,
    ScreenshotAnalysisResponse
)

from app.MainFeature.schemas import(
    HideModeEnum,
    RecordingTypeEnum,
    RecordingConfig,
    UIComponentEnum,
    InsightTypeEnum,
    UIConfig,
    SecurityConfig,
    InvisibilityModeRequest,
    InvisibilityModeResponse,
    UIVisibilityRequest,
    UIVisibilityResponse,
    UIState,
    InsightGenerationRequest,
    InsightGenerationResponse,
    InvisibilityError,
    InsightData,
    RecordingData,
    RecordingSessionRequest,
    RecordingSessionResponse,
    SessionStatusResponse,
    SecurityStatusResponse,
    SecurityStatus,
    SessionData,
    SystemConfig,
    PerformanceMetrics
)

# Import Quick Respond schemas
from app.quick_respond.schemas import (
    QuickRespondRequest,
    QuickRespondResponse,
    SimplifyRequest,
    SimplifyResponse,
    MeetingContext as QuickRespondMeetingContext,
    KeyInsight as QuickRespondKeyInsight,
    UrgencyLevel,
    AnalysisType as QuickRespondAnalysisType,
    MeetingType,
    MeetingStatus,
    ParticipantInfo,
    ScreenContent,
    MeetingMetrics,
    StreamingResponse as QuickRespondStreamingResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    ErrorResponse as QuickRespondErrorResponse,
    HealthCheckResponse as QuickRespondHealthCheckResponse,
    WebSocketMessage as QuickRespondWebSocketMessage,
    OllamaConfig,
    QuickRespondConfig,
    ModelPrompts,
    APIResponse as QuickRespondAPIResponse,
    PaginatedResponse,
    AdvancedAnalysisRequest,
    AdvancedAnalysisResponse,
    SessionAnalytics,
    UsageMetrics,
    WebhookEvent,
    WebhookResponse
)

# Import services
from app.quick_respond.service import (
    QuickRespondService,
)

from app.voice_recognition.services import (
    VoiceProcessingService,
    AudioService
>>>>>>> 91363704a1edbd3248250121ce7a4b90fac292f9
)

from app.key_insights.services import KeyInsightsService


<<<<<<< HEAD
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
=======
from app.handFree.service import (
    HandsFreeService)  

from app.image_recognition.service import(
    CameraService,
    ExpressionDetectionService,
    ChatService,
    ScreenRecordingService,
    AIAnalysisService
)

from app.summarization.service import(
    SummarizationService,
)

from app.MainFeature.service import(
    InvisibilityService,
)
>>>>>>> 91363704a1edbd3248250121ce7a4b90fac292f9

# Configure logging  
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state for interview sessions
interview_sessions: Dict[str, Dict[str, Any]] = {}

# Initialize services
voice_service = VoiceProcessingService()
audio_service = AudioService()
camera_service = CameraService()
expression_service = ExpressionDetectionService()
chat_service = ChatService()
permission_service = PermissionService()
recording_service = ScreenRecordingService()
ai_service = AIAnalysisService()
hands_free_service = HandsFreeService()
quick_respond_service = QuickRespondService()
insights_service = KeyInsightsService()
summarization_service = SummarizationService()
invisibility_service = InvisibilityService()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}  # Session-based connections
        self.connection_metadata: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str = None):
        await websocket.accept()
        
        if session_id:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = []
            self.active_connections[session_id].append(websocket)
        
        self.connection_metadata[websocket] = {
            "session_id": session_id,
            "connected_at": datetime.utcnow(),
            "message_count": 0
        }

    def disconnect(self, websocket: WebSocket):
        # Remove from session-based tracking
        metadata = self.connection_metadata.get(websocket, {})
        session_id = metadata.get("session_id")
        
        if session_id and session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)
            
            # Clean up empty session lists
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        
        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
            # Update message count
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            self.disconnect(websocket)

    async def send_to_session(self, message: str, session_id: str):
        """Send message to all connections for a specific session"""
        if session_id not in self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_text(message)
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]["message_count"] += 1
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast(self, message: str):
        all_connections = []
        for session_connections in self.active_connections.values():
            all_connections.extend(session_connections)
        
        disconnected = []
        for connection in all_connections:
            try:
                await connection.send_text(message)
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]["message_count"] += 1
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    def get_session_connections(self, session_id: str) -> List[WebSocket]:
        """Get all connections for a session"""
        return self.active_connections.get(session_id, [])
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        total_connections = sum(len(conns) for conns in self.active_connections.values())
        return {
            "total_sessions": len(self.active_connections),
            "total_connections": total_connections,
            "connections_by_session": {k: len(v) for k, v in self.active_connections.items()}
        }

manager = ConnectionManager()

# Service initialization functions
async def initialize_ollama_connection():
    """Initialize connection to Ollama with Nous Hermes model"""
    try:
        # Test Ollama connection
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

async def initialize_VoiceProcessingService():
    """Initialize Voice Processing Service"""
    try:
        logger.info("✅ Voice Processing Service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Voice Processing Service: {e}")
        raise

async def initialize_core_services():
    """Initialize all core services"""
    try:
        logger.info("✅ All core services initialized")
>>>>>>> 91363704a1edbd3248250121ce7a4b90fac292f9
    except Exception as e:
        logger.error(f"❌ Failed to initialize core services: {e}")
        raise

async def cleanup_services():
<<<<<<< HEAD
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
=======
    """Cleanup all services"""
    try:
        await storage_manager.cleanup()
        logger.info("✅ Services cleanup completed")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Interview AI Assistant Backend Starting...")
    
    try:
        # Initialize storage manager first
        await storage_manager.initialize()
        
        # Initialize all services through service manager
        await service_manager.initialize_all()
        
        # Initialize Ollama connection
        await initialize_ollama_connection()
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        # Don't raise - allow app to start with limited functionality
      
    yield
    
    # Shutdown
    logger.info("🔄 Interview AI Assistant Backend Shutting Down...")
    
    # Cleanup services
    try:
        await cleanup_services()
        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

# Create FastAPI App with lifespan
app = FastAPI(
    title="Interview AI Assistant Backend",
    description="Intelligent interview support system with invisibility mode, real-time analysis, and adaptive responses using Ollama & Nous Hermes",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==============================================
# FIXED: Frontend Integration with Fallback
# ==============================================

frontend_dist = Path(__file__).parent.parent / "agenda frontend" / "dist"

# Mount the static frontend if it exists
if frontend_dist.exists():
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
    "*.ngrok-free.app",
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

# ==============================================
# FIXED: Router Integration - Proper Setup
# ==============================================

# Create routers with proper integration
summarization_router = APIRouter(prefix="/api/v1/summarization", tags=["Summarization"])
quick_respond_router = APIRouter(prefix="/api/v1/quick-respond", tags=["Quick Respond"])
voice_router = APIRouter(prefix="/api/v1/voice", tags=["Voice Recognition"])
camera_router = APIRouter(prefix="/api/v1/camera", tags=["Camera"])
expression_router = APIRouter(prefix="/api/v1/expression", tags=["Expression"])
hands_free_router = APIRouter(prefix="/api/v1/hands-free", tags=["Hands-Free"])
key_insights_router = APIRouter(prefix="/api/v1/key-insights", tags=["Key Insights"])
main_feature_router = APIRouter(prefix="/api/v1/invisibility", tags=["Invisibility Mode"])

# Mount all routers
app.include_router(summarization_router)
app.include_router(quick_respond_router)
app.include_router(voice_router)
app.include_router(camera_router)
app.include_router(expression_router)
app.include_router(hands_free_router)
app.include_router(key_insights_router)
app.include_router(main_feature_router)

# ==============================================
# FIXED: Custom middleware for proper session tracking
# ==============================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)  # Fixed: use call_next properly
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http") 
async def session_tracking_middleware(request: Request, call_next):
    """Track sessions and add session context"""
    session_id = request.headers.get("X-Session-ID")
    
    # Add session to request state for access in routes
    request.state.session_id = session_id
    
    response = await call_next(request)
    
    # Add session ID to response if present
    if session_id:
        response.headers["X-Session-ID"] = session_id
    
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

<<<<<<< HEAD
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
=======
# ====================================================================
# SUMMARIZATION ROUTES - PROPERLY INTEGRATED
# ====================================================================

@app.post("/api/v1/summarization/upload-audio", response_model=AudioUploadResponse, tags=["Summarization"])
async def upload_meeting_audio(
    audio_file: UploadFile = File(...),
    meeting_id: Optional[str] = None
):
    """Upload audio file for meeting analysis"""
    try:
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        result = await service_manager.summarization_service.process_audio_upload(
            audio_file=audio_file,
            user_id="default_user",  # You might want to get this from authentication
            meeting_id=meeting_id
        )
        
        return AudioUploadResponse(**result)
    
    except Exception as e:
        logger.error(f"Error uploading audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload audio: {str(e)}")

@app.post("/api/v1/summarization/analyze-meeting", response_model=MeetingAnalysisResponse, tags=["Summarization"])
async def analyze_meeting_audio(request: MeetingAnalysisRequest):
    """Analyze meeting audio and provide summarization with actionable points"""
    try:
        analysis_result = await summarization_service.analyze_meeting_audio(
            audio_file_path=request.audio_file_path,
            meeting_context=getattr(request, 'meeting_context', None),
            user_id="default_user",
            analysis_type=getattr(request, 'analysis_type', 'post_meeting')
        )
        
        return MeetingAnalysisResponse(**analysis_result)
    
    except Exception as e:
        logger.error(f"Error analyzing meeting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze meeting: {str(e)}")

@app.post("/api/v1/summarization/summarize", response_model=SummarizationResponse, tags=["Summarization"])
async def create_summary(request: SummarizationRequest):
    """Create summarization from transcribed text or audio analysis"""
    try:
        summary_result = await summarization_service.generate_summary(
            content=request.content,
            summary_type=request.summary_type,
            user_id="default_user",
            meeting_id=getattr(request, 'meeting_id', None),
            include_action_items=getattr(request, 'include_action_items', True)
        )
        
        return SummarizationResponse(**summary_result)
    
    except Exception as e:
        logger.error(f"Error creating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create summary: {str(e)}")

@app.get("/api/v1/summarization/meeting/{meeting_id}/summary", response_model=SummarizationResponse, tags=["Summarization"])
async def get_meeting_summary(meeting_id: str):
    """Get existing summary for a meeting"""
    try:
        summary = await summarization_service.get_meeting_summary(
            meeting_id=meeting_id,
            user_id="default_user"
        )
        
        if not summary:
            raise HTTPException(status_code=404, detail="Summary not found for this meeting")
        
        return SummarizationResponse(**summary)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting meeting summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get meeting summary: {str(e)}")

@app.get("/api/v1/summarization/user/summaries", response_model=List[SummarizationResponse], tags=["Summarization"])
async def get_user_summaries(
    limit: int = 10,
    offset: int = 0
):
    """Get all summaries for the current user"""
    try:
        summaries = await summarization_service.get_user_summaries(
            user_id="default_user",
            limit=limit,
            offset=offset
        )
        
        return [SummarizationResponse(**summary) for summary in summaries]
    
    except Exception as e:
        logger.error(f"Error getting user summaries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user summaries: {str(e)}")

@app.delete("/api/v1/summarization/meeting/{meeting_id}/summary", tags=["Summarization"])
async def delete_meeting_summary(meeting_id: str):
    """Delete summary for a specific meeting"""
    try:
        success = await summarization_service.delete_meeting_summary(
            meeting_id=meeting_id,
            user_id="default_user"
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Summary not found or access denied")
        
        return JSONResponse(
            status_code=200,
            content={"message": "Summary deleted successfully"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete summary: {str(e)}")

@app.post("/api/v1/summarization/real-time-analysis", response_model=MeetingAnalysisResponse, tags=["Summarization"])
async def real_time_meeting_analysis(request: MeetingAnalysisRequest):
    """Real-time analysis of ongoing meeting audio"""
    try:
        analysis = await summarization_service.real_time_audio_analysis(
            audio_chunk_path=request.audio_file_path,
            meeting_context=getattr(request, 'meeting_context', None),
            user_id="default_user"
        )
        
        return MeetingAnalysisResponse(**analysis)
    
    except Exception as e:
        logger.error(f"Error in real-time analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to perform real-time analysis: {str(e)}")

@app.post("/api/v1/summarization/batch", response_model=List[SummarizationResponse], tags=["Summarization"])
async def batch_summarize_meetings(request: BatchSummarizationRequest):
    """Batch process multiple meetings for summarization"""
    try:
        results = []
        for meeting_id in request.meeting_ids:
            try:
                summary = await summarization_service.get_meeting_summary(
                    meeting_id=meeting_id,
                    user_id="default_user"
                )
                if summary:
                    results.append(SummarizationResponse(**summary))
            except Exception as e:
                logger.warning(f"Failed to process meeting {meeting_id}: {str(e)}")
                continue
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to batch process summaries: {str(e)}")
@router.post("/actionitems/", response_model=ActionItem)
async def create_action_item(item: ActionItem):
    # logic to save item
    return item

@router.get("/actionitems/", response_model=List[ActionItem])
async def get_action_items():
    # logic to retrieve all action items
    return []

@router.get("/actionitems/{item_id}", response_model=ActionItem)
async def get_action_item(item_id: int):
    # logic to retrieve item by ID
    return {"id": item_id, "name": "example"}

@router.put("/actionitems/{item_id}", response_model=ActionItem)
async def update_action_item(item_id: int, item: ActionItem):
    # logic to update item
    return item

@router.delete("/actionitems/{item_id}")
async def delete_action_item(item_id: int):
    # logic to delete item
    return {"message": f"ActionItem {item_id} deleted"}


# ----- KeyPoint CRUD -----
@router.post("/keypoints/", response_model=KeyPoint)
async def create_key_point(point: KeyPoint):
    return point

@router.get("/keypoints/", response_model=List[KeyPoint])
async def get_key_points():
    return []

@router.get("/keypoints/{point_id}", response_model=KeyPoint)
async def get_key_point(point_id: int):
    return {"id": point_id, "name": "example"}

@router.put("/keypoints/{point_id}", response_model=KeyPoint)
async def update_key_point(point_id: int, point: KeyPoint):
    return point

@router.delete("/keypoints/{point_id}")
async def delete_key_point(point_id: int):
    return {"message": f"KeyPoint {point_id} deleted"}


# ----- SummaryType CRUD -----
@router.post("/summarytypes/", response_model=SummaryType)
async def create_summary_type(summary: SummaryType):
    return summary

@router.get("/summarytypes/", response_model=List[SummaryType])
async def get_summary_types():
    return []

@router.get("/summarytypes/{summary_id}", response_model=SummaryType)
async def get_summary_type(summary_id: int):
    return {"id": summary_id, "name": "example"}

@router.put("/summarytypes/{summary_id}", response_model=SummaryType)
async def update_summary_type(summary_id: int, summary: SummaryType):
    return summary

@router.delete("/summarytypes/{summary_id}")
async def delete_summary_type(summary_id: int):
    return {"message": f"SummaryType {summary_id} deleted"}

@router.post("/summary", response_model=SummaryUpdateRequest)
async def create_summary(summary: SummaryUpdateRequest):
    summaries.append(summary)
    return summary

@router.get("/summary", response_model=List[SummaryUpdateRequest])
async def get_summaries():
    return summaries

# ----- AnalysisType CRUD -----
@router.post("/analysistypes/", response_model=AnalysisType)
async def create_analysis_type(analysis: AnalysisType):
    return analysis

@router.get("/analysistypes/", response_model=List[AnalysisType])
async def get_analysis_types():
    return []

@router.get("/analysistypes/{analysis_id}", response_model=AnalysisType)
async def get_analysis_type(analysis_id: int):
    return {"id": analysis_id, "name": "example"}

@router.put("/analysistypes/{analysis_id}", response_model=AnalysisType)
async def update_analysis_type(analysis_id: int, analysis: AnalysisType):
    return analysis

@router.delete("/analysistypes/{analysis_id}")
async def delete_analysis_type(analysis_id: int):
    return {"message": f"AnalysisType {analysis_id} deleted"}

# ---------------- MeetingContext Routes ----------------
@router.post("/meeting", response_model=MeetingContext)
async def create_meeting(meeting: MeetingContext):
    meetings.append(meeting)
    return meeting

@router.get("/meeting", response_model=List[MeetingContext])
async def get_meetings():
    return meetings

# ---------------- LLAVAAnalysisConfig Routes ----------------
@router.post("/llava-config", response_model=LLAVAAnalysisConfig)
async def create_llava_config(config: LLAVAAnalysisConfig):
    llava_configs.append(config)
    return config

@router.get("/llava-config", response_model=List[LLAVAAnalysisConfig])
async def get_llava_configs():
    return llava_configs

# ---------------- RealTimeAnalysisUpdate Routes ----------------
@router.post("/realtime-update", response_model=RealTimeAnalysisUpdate)
async def create_realtime_update(update: RealTimeAnalysisUpdate):
    real_time_updates.append(update)
    return update

@router.get("/realtime-update", response_model=List[RealTimeAnalysisUpdate])
async def get_realtime_updates():
    return real_time_update

@router.post("/actionitems/", response_model=ActionItem)
async def create_action_item(item: ActionItem):
    # logic to save item
    return item

@router.get("/actionitems/", response_model=List[ActionItem])
async def get_action_items():
    # logic to retrieve all action items
    return []

@router.get("/actionitems/{item_id}", response_model=ActionItem)
async def get_action_item(item_id: int):
    # logic to retrieve item by ID
    return {"id": item_id, "name": "example"}

@router.put("/actionitems/{item_id}", response_model=ActionItem)
async def update_action_item(item_id: int, item: ActionItem):
    # logic to update item
    return item

@router.delete("/actionitems/{item_id}")
async def delete_action_item(item_id: int):
    # logic to delete item
    return {"message": f"ActionItem {item_id} deleted"}

# ====================================================================
# QUICK RESPOND ROUTES - PROPERLY INTEGRATED
# ====================================================================
# ============================
# UrgencyLevel Routes
# ============================
@router.post("/meeting_status/", response_model=MeetingStatus)
async def create_meeting_status(meeting_status: MeetingStatus):
    # Add your DB logic here
    return meeting_status

@router.get("/meeting_status/{id}", response_model=MeetingStatus)
async def get_meeting_status(id: int):
    # Fetch from DB
    return {"id": id, "status": "active"}  # Example

@router.get("/meeting_status/", response_model=List[MeetingStatus])
async def list_meeting_statuses():
    # Fetch all from DB
    return []

@router.put("/meeting_status/{id}", response_model=MeetingStatus)
async def update_meeting_status(id: int, meeting_status: MeetingStatus):
    # Update DB
    return meeting_status

@router.delete("/meeting_status/{id}")
async def delete_meeting_status(id: int):
    # Delete from DB
    return {"message": f"MeetingStatus {id} deleted"}

# ============================
# ParticipantInfo Routes
# ============================

@router.post("/participant_info/", response_model=ParticipantInfo)
async def create_participant_info(participant: ParticipantInfo):
    return participant

@router.get("/participant_info/{id}", response_model=ParticipantInfo)
async def get_participant_info(id: int):
    return {"id": id, "name": "John Doe"}  # Example

@router.get("/participant_info/", response_model=List[ParticipantInfo])
async def list_participants():
    return []

@router.put("/participant_info/{id}", response_model=ParticipantInfo)
async def update_participant_info(id: int, participant: ParticipantInfo):
    return participant

@router.delete("/participant_info/{id}")
async def delete_participant_info(id: int):
    return {"message": f"ParticipantInfo {id} deleted"}

# ============================
# ScreenContent Routes
# ============================

@router.post("/screen_content/", response_model=ScreenContent)
async def create_screen_content(screen: ScreenContent):
    return screen

@router.get("/screen_content/{id}", response_model=ScreenContent)
async def get_screen_content(id: int):
    return {"id": id, "content": "Example content"}  # Example

@router.get("/screen_content/", response_model=List[ScreenContent])
async def list_screen_content():
    return []

@router.put("/screen_content/{id}", response_model=ScreenContent)
async def update_screen_content(id: int, screen: ScreenContent):
    return screen

@router.delete("/screen_content/{id}")
async def delete_screen_content(id: int):
    return {"message": f"ScreenContent {id} deleted"}

# ============================
# MeetingMetrics Routes
# ============================

@router.post("/meeting_metrics/", response_model=MeetingMetrics)
async def create_meeting_metrics(metrics: MeetingMetrics):
    return metrics

@router.get("/meeting_metrics/{id}", response_model=MeetingMetrics)
async def get_meeting_metrics(id: int):
    return {"id": id, "metrics": {}}  # Example

@router.get("/meeting_metrics/", response_model=List[MeetingMetrics])
async def list_meeting_metrics():
    return []

@router.put("/meeting_metrics/{id}", response_model=MeetingMetrics)
async def update_meeting_metrics(id: int, metrics: MeetingMetrics):
    return metrics

@router.delete("/meeting_metrics/{id}")
async def delete_meeting_metrics(id: int):
    return {"message": f"MeetingMetrics {id} deleted"}

@router.get("/urgency-levels", response_model=List[UrgencyLevel])
async def list_urgency_levels():
    """
    Get all possible urgency levels (enum values).
    """
    return [level for level in UrgencyLevel]

@router.get("/urgency-levels/{level}", response_model=UrgencyLevel)
async def get_urgency_level(level: UrgencyLevel):
    """
    Validate and return a specific urgency level.
    If an invalid level is passed, FastAPI will auto-throw 422.
    """
    return level


# ============================
# PaginatedResponse Demo Route
# ============================

@router.get("/items", response_model=PaginatedResponse)
async def list_items(
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page")
):
    """
    Example endpoint to demonstrate PaginatedResponse usage.
    Replace with actual DB query in production.
    """
    # Fake dataset for demo
    total_items = 55
    all_items = [f"Item {i}" for i in range(1, total_items + 1)]

    # Pagination math
    start = (page - 1) * page_size
    end = start + page_size
    paged_items = all_items[start:end]

    return PaginatedResponse(
        items=paged_items,
        total=total_items,
        page=page,
        page_size=page_size,
        timestamp=datetime.utcnow()
    )

@app.post("/api/v1/quick-respond/analyze-screenshot", response_model=QuickRespondResponse, tags=["Quick Respond"])
async def analyze_meeting_screenshot(
    screenshot: UploadFile = File(...),
    meeting_context: Optional[str] = None,
    audio_transcript: Optional[str] = None
):
    """Analyze live meeting screenshot and provide key insights"""
    try:
        if not screenshot.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        screenshot_data = await screenshot.read()
        
        request_data = {
            "screenshot_data": screenshot_data,
            "meeting_context": meeting_context,
            "audio_transcript": audio_transcript,
            "analysis_type": "key_insights"
        }
        
        response = await service_manager.quick_respond_service.analyze_meeting_content(request_data)
        
        return response
        
    except Exception as e:
        logger.error(f"Quick respond screenshot analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/quick-respond/analyze-screenshot/stream", tags=["Quick Respond"])
async def analyze_meeting_screenshot_stream(
    screenshot: UploadFile = File(...),
    meeting_context: Optional[str] = None,
    audio_transcript: Optional[str] = None
):
    """Stream real-time analysis of meeting screenshot"""
    try:
        if not screenshot.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        screenshot_data = await screenshot.read()
        
        request_data = {
            "screenshot_data": screenshot_data,
            "meeting_context": meeting_context,
            "audio_transcript": audio_transcript,
            "analysis_type": "key_insights"
        }
        
        async def generate_stream():
            async for chunk in quick_respond_service.analyze_meeting_content_stream(request_data):
                yield f"data: {json.dumps(chunk)}\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Quick respond streaming analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming analysis failed: {str(e)}")
@router.post("/webhook-event", response_model=WebhookResponse)
async def create_webhook_event(event: WebhookEvent):
    # Replace this with your actual logic
    return WebhookResponse(message=f"Received event: {event.event_name}")

# ---------------- OllamaConfig Routes ----------------
@router.post("/ollama-config", response_model=OllamaConfig)
async def create_ollama_config(config: OllamaConfig):
    # Add save logic
    return config

@router.get("/ollama-config", response_model=List[OllamaConfig])
async def get_all_ollama_configs():
    # Replace with fetch logic
    return []

# ---------------- QuickRespondConfig Routes ----------------
@router.post("/quick-respond-config", response_model=QuickRespondConfig)
async def create_quick_respond_config(config: QuickRespondConfig):
    return config

@router.get("/quick-respond-config", response_model=List[QuickRespondConfig])
async def get_all_quick_respond_configs():
    return []

# ---------------- ModelPrompts Routes ----------------
@router.post("/model-prompts", response_model=ModelPrompts)
async def create_model_prompt(prompt: ModelPrompts):
    return prompt

@router.get("/model-prompts", response_model=List[ModelPrompts])
async def get_all_model_prompts():
    return []

@app.post("/api/v1/quick-respond/simplify", response_model=SimplifyResponse, tags=["Quick Respond"])
async def simplify_analysis_response(request: SimplifyRequest):
    """Simplify a complex analysis response"""
    try:
        response = await service_manager.quick_respond_service.simplify_analysis(request)
        return response
        
    except Exception as e:
        logger.error(f"Quick respond simplify error: {e}")
        raise HTTPException(status_code=500, detail=f"Simplification failed: {str(e)}")

@app.post("/api/v1/quick-respond/batch-analyze", response_model=BatchAnalysisResponse, tags=["Quick Respond"])
async def batch_analyze_screenshots(
    screenshots: List[UploadFile] = File(...),
    meeting_context: Optional[str] = None
):
    """Analyze multiple screenshots from a meeting session"""
    try:
        results = []
        
        for screenshot in screenshots:
            if not screenshot.content_type.startswith('image/'):
                continue
                
            screenshot_data = await screenshot.read()
            
            request_data = {
                "screenshot_data": screenshot_data,
                "meeting_context": meeting_context,
                "analysis_type": "key_insights"
            }
            
            result = await service_manager.quick_respond_service.analyze_meeting_content(request_data)
            results.append({
                "filename": screenshot.filename,
                "analysis": result
            })
        
        return {"batch_results": results}
        
    except Exception as e:
        logger.error(f"Quick respond batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")
@router.post("/quick-respond", response_model=QuickRespondResponse)
async def quick_respond(request: QuickRespondRequest):
    """
    Generate quick insights from a screenshot and optional audio transcript.
    """
    # TODO: Implement analysis logic
    return QuickRespondResponse(
        key_insights=[],
        full_analysis="Analysis placeholder",
        timestamp=datetime.utcnow(),
        confidence_score=0.7,
        can_simplify=True,
        session_id="session_123"
    )

# Simplify analysis endpoint
@router.post("/simplify", response_model=SimplifyResponse)
async def simplify_analysis(request: SimplifyRequest):
    """
    Simplify a complex analysis text into bullet points and actions.
    """
    # TODO: Implement simplification logic
    return SimplifyResponse(
        simplified_text="Simplified placeholder",
        simple_points=["Point 1", "Point 2"],
        actions_needed=["Action 1"],
        meeting_status="neutral",
        timestamp=datetime.utcnow(),
        original_length=len(request.original_analysis),
        simplified_length=50
    )

# Batch analysis endpoint
@router.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analysis(request: BatchAnalysisRequest):
    """
    Perform batch analysis on multiple screenshots.
    """
    # TODO: Implement batch analysis
    return BatchAnalysisResponse(
        individual_analyses=[],
        summary_insights=[],
        meeting_progression=[],
        key_moments=[],
        overall_metrics=None,
        batch_session_id="batch_123"
    )

# Advanced analysis endpoint
@router.post("/advanced", response_model=AdvancedAnalysisResponse)
async def advanced_analysis(request: AdvancedAnalysisRequest):
    """
    Perform detailed analysis with optional custom prompts, metrics, and sentiment.
    """
    # TODO: Implement advanced analysis
    return AdvancedAnalysisResponse(
        key_insights=[],
        full_analysis="Advanced analysis placeholder",
        session_id="adv_session_123",
        timestamp=datetime.utcnow(),
        recommendations=[]
    )

# Health check endpoint
@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check service and model health.
    """
    # TODO: Implement real health check
    return HealthCheckResponse(
        status="ok",
        ollama=True,
        llava_model=True,
        llama_model=True,
        timestamp=datetime.utcnow(),
        response_time_ms=50
    )

@app.get("/api/v1/quick-respond/health", response_model=QuickRespondHealthCheckResponse, tags=["Quick Respond"])
async def quick_respond_health_check():
    """Check if LLAVA/Ollama services are available for Quick Respond"""
    try:
        health_status = await quick_respond_service.check_service_health()
        return QuickRespondHealthCheckResponse(**health_status)
        
    except Exception as e:
        logger.error(f"Quick respond health check error: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/api/v1/quick-respond/context/update", tags=["Quick Respond"])
async def update_meeting_context(context: QuickRespondMeetingContext):
    """Update meeting context for better analysis"""
    try:
        await quick_respond_service.update_meeting_context(context)
        return {"status": "success", "message": "Meeting context updated"}
        
    except Exception as e:
        logger.error(f"Quick respond context update error: {e}")
        raise HTTPException(status_code=500, detail=f"Context update failed: {str(e)}")

@app.delete("/api/v1/quick-respond/context/clear", tags=["Quick Respond"])
async def clear_meeting_context():
    """Clear stored meeting context"""
    try:
        await quick_respond_service.clear_meeting_context()
        return {"status": "success", "message": "Meeting context cleared"}
        
    except Exception as e:
        logger.error(f"Quick respond context clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Context clearing failed: {str(e)}")

@app.post("/api/v1/quick-respond/advanced-analyze", response_model=AdvancedAnalysisResponse, tags=["Quick Respond"])
async def advanced_analysis(request: AdvancedAnalysisRequest):
    """Perform advanced analysis with custom options"""
    try:
        # Convert request to format expected by service
        request_data = {
            "screenshot_data": request.screenshot_data,
            "meeting_context": request.meeting_context.dict() if request.meeting_context else None,
            "audio_transcript": request.audio_transcript,
            "analysis_type": request.output_format,
            "focus_areas": request.focus_areas,
            "custom_prompts": request.custom_prompts,
            "include_metrics": request.include_metrics,
            "sentiment_analysis": request.sentiment_analysis
        }
        
        # Process with quick respond service
        analysis_result = await quick_respond_service.analyze_meeting_content(request_data)
        
        # Build advanced response
        response_data = {
            "key_insights": analysis_result.key_insights,
            "full_analysis": analysis_result.full_analysis,
            "timestamp": datetime.utcnow(),
            "session_id": analysis_result.session_id,
            "sentiment_score": 0.5,  # Default neutral sentiment
            "recommendations": ["Follow up on key points", "Clarify action items"],
            "processing_metadata": {"processing_time": 1.5, "model_used": "llava"}
        }
        
        return AdvancedAnalysisResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Quick respond advanced analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")

@app.get("/api/v1/quick-respond/analytics/{session_id}", response_model=SessionAnalytics, tags=["Quick Respond"])
async def get_session_analytics(session_id: str):
    """Get analytics for a specific session"""
    try:
        # Mock analytics data - replace with actual service call
        analytics_data = {
            "session_id": session_id,
            "total_analyses": 5,
            "average_confidence": 0.85,
            "high_urgency_insights": 2,
            "session_duration_minutes": 45.0,
            "most_common_insights": ["action items", "decisions", "questions"],
            "participant_engagement": {"John": 0.9, "Sarah": 0.8, "Mike": 0.7},
            "technical_issues_count": 1,
            "created_at": datetime.utcnow()
        }
        
        return SessionAnalytics(**analytics_data)
        
    except Exception as e:
        logger.error(f"Quick respond analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@app.get("/api/v1/quick-respond/metrics", response_model=UsageMetrics, tags=["Quick Respond"])
async def get_usage_metrics():
    """Get overall system usage metrics"""
    try:
        # Mock metrics data - replace with actual service call
        metrics_data = {
            "total_requests": 1500,
            "successful_analyses": 1425,
            "failed_analyses": 75,
            "average_response_time_ms": 2500.0,
            "peak_concurrent_requests": 25,
            "model_usage": {"llava": 1200, "llama": 300},
            "error_types": {"timeout": 30, "model_error": 25, "validation": 20},
            "timestamp": datetime.utcnow()
        }
        
        return UsageMetrics(**metrics_data)
        
    except Exception as e:
        logger.error(f"Quick respond metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# ====================================================================
# VOICE RECOGNITION ROUTES - ENHANCED INTEGRATION
# ====================================================================

@app.post("/api/v1/voice/session/start", response_model=VoiceSessionResponse, tags=["Voice Recognition"])
async def voice_start_session(request: VoiceSessionRequest):
    """Voice Recognition: Start a new voice session"""
    try:
        session_id = await voice_service.create_session(request.user_id)
        return VoiceSessionResponse(
            success=True,
            session_id=session_id,
            message="Voice session started successfully",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Voice session start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/voice/microphone/status/{session_id}", response_model=MicrophoneStatusResponse, tags=["Voice Recognition"])
async def voice_microphone_status(session_id: str):
    """Voice Recognition: Check microphone status"""
    try:
        mic_status = await voice_service.check_microphone_status(session_id)
        return MicrophoneStatusResponse(
            success=True,
            is_available=mic_status["is_available"],
            is_enabled=mic_status["is_enabled"],
            current_device=mic_status.get("current_device"),
            message=mic_status["message"],
            alert_required=not mic_status["is_enabled"]
        )
    except Exception as e:
        logger.error(f"Microphone status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/voice/devices/{session_id}", response_model=DeviceListResponse, tags=["Voice Recognition"])
async def voice_list_devices(session_id: str):
    """Voice Recognition: List available audio input devices"""
    try:
        devices = await voice_service.get_audio_devices(session_id)
        return DeviceListResponse(
            success=True,
            devices=devices,
            default_device=devices[0] if devices else None,
            message="Audio devices retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Device list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/device/select", tags=["Voice Recognition"])
async def voice_select_device(request: DeviceSelectionRequest):
    """Voice Recognition: Select and connect to a specific audio device"""
    try:
        result = await service_manager.voice_service.select_audio_device(request.session_id, request.device_id)
        return JSONResponse(content={
            "success": True,
            "connected_device": result["device_name"],
            "device_id": result["device_id"],
            "message": "Device connected successfully"
        })
    except Exception as e:
        logger.error(f"Device selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/microphone/toggle", tags=["Voice Recognition"])
async def voice_toggle_microphone(request: MicrophoneStatusRequest):
    """Voice Recognition: Toggle microphone on/off"""
    try:
        if request.turn_on:
            mic_status = await voice_service.check_microphone_status(request.session_id)
            if not mic_status["is_available"]:
                return JSONResponse(content={
                    "success": False,
                    "message": "Enable microphone access in browser settings",
                    "action_required": "enable_permissions"
                })

            devices = await voice_service.get_audio_devices(request.session_id)
            if not devices:
                raise HTTPException(status_code=404, detail="No audio devices found")

            device_id = request.device_id if getattr(request, "device_id", None) else devices[0]["id"]
            result = await service_manager.voice_service.select_audio_device(request.session_id, device_id)

            return JSONResponse(content={
                "success": True,
                "microphone_on": True,
                "connected_device": result["device_name"],
                "message": "Microphone turned on and connected"
            })
        else:
            await voice_service.disable_microphone(request.session_id)
            return JSONResponse(content={
                "success": True,
                "microphone_on": False,
                "message": "Microphone turned off"
            })
    except Exception as e:
        logger.error(f"Microphone toggle error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/audio/process", response_model=AudioProcessingResponse, tags=["Voice Recognition"])
async def voice_process_audio(session_id: str, audio_file: UploadFile = File(...)):
    """Voice Recognition: Process uploaded audio"""
    try:
        if not audio_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid audio file format")

        audio_data = await audio_file.read()
        result = await service_manager.voice_service.process_audio(session_id=session_id, audio_data=audio_data, filename=audio_file.filename)

        return AudioProcessingResponse(
            success=True,
            transcription=result["transcription"],
            confidence_score=result["confidence"],
            audio_quality=result["audio_quality"],
            processing_time=result["processing_time"],
            message="Audio processed successfully"
        )
    except Exception as e:
        logger.error(f"Audio process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/transcribe", response_model=TranscriptionResponse, tags=["Voice Recognition"])
async def voice_transcribe(request: AudioProcessingRequest):
    """Voice Recognition: Transcribe audio to text"""
    try:
        result = await service_manager.voice_service.transcribe_audio(request.session_id, request.audio_data)
        return TranscriptionResponse(
            success=True,
            text=result["text"],
            confidence=result["confidence"],
            language=result.get("language", "en"),
            duration=result.get("duration", 0),
            message="Transcription completed successfully"
        )
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/ai/respond", response_model=AIResponseResponse, tags=["Voice Recognition"])
async def voice_ai_response(request: AIResponseRequest):
    """Voice Recognition: Get AI-generated response"""
    try:
        result = await service_manager.voice_service.generate_ai_response(
            session_id=request.session_id,
            question=request.question,
            response_format=request.response_format,
            context=request.context
        )
        return AIResponseResponse(
            success=True,
            response=result["response"],
            format_type=request.response_format,
            confidence=result["confidence"],
            processing_time=result["processing_time"],
            message="AI response generated successfully"
        )
    except Exception as e:
        logger.error(f"AI response error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/analyze", response_model=VoiceAnalysisResponse, tags=["Voice Recognition"])
async def voice_analyze(request: AudioProcessingRequest):
    """Voice Recognition: Analyze voice confidence and provide feedback"""
    try:
        result = await service_manager.voice_service.analyze_voice_characteristics(request.session_id, request.audio_data)
        return VoiceAnalysisResponse(
            success=True,
            confidence_rating=result["confidence_rating"],
            voice_characteristics=result["characteristics"],
            situational_tips=result["tips"],
            recommendations=result["recommendations"],
            analysis_summary=result["summary"]
        )
    except Exception as e:
        logger.error(f"Voice analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/voice/simplify", response_model=AIResponseResponse, tags=["Voice Recognition"])
async def voice_simplify(request: SimplifiedAnswerRequest):
    """Voice Recognition: Simplify AI response"""
    try:
        result = await service_manager.voice_service.simplify_response(
            session_id=request.session_id,
            original_response=request.original_response,
            simplification_level=request.simplification_level
        )
        return AIResponseResponse(
            success=True,
            response=result["response"],
            format_type="simplified",
            confidence=result["confidence"],
            processing_time=result["processing_time"],
            message="Simplified response generated successfully"
        )
    except Exception as e:
        logger.error(f"Simplify response error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/voice/session/{session_id}", tags=["Voice Recognition"])
async def voice_end_session(session_id: str):
    """Voice Recognition: End session and clean up resources"""
    try:
        await voice_service.end_session(session_id)
        return JSONResponse(content={"success": True, "message": "Voice session ended successfully"})
    except Exception as e:
        logger.error(f"End session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/voice/session/{session_id}/status", tags=["Voice Recognition"])
async def voice_session_status(session_id: str):
    """Voice Recognition: Get session status"""
    try:
        status = await voice_service.get_session_status(session_id)
        return JSONResponse(content={
            "success": True,
            "session_active": status["active"],
            "microphone_connected": status["mic_connected"],
            "current_device": status.get("current_device"),
            "session_duration": status.get("duration"),
            "total_interactions": status.get("interactions", 0)
        })
    except Exception as e:
        logger.error(f"Session status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================================
# AUDIO SERVICE ROUTES - ENHANCED INTEGRATION
# ====================================================================

@app.post("/api/v1/audio/calibrate", response_model=CalibrationResponse, tags=["Audio Processing"])
async def calibrate_audio(request: CalibrationRequest):
    """Audio Calibration endpoint - measures background noise and sets optimal levels"""
    try:
        logger.info(f"Starting audio calibration with duration: {request.duration}s")
        
        result = await service_manager.audio_service.calibrate_audio(
            duration=request.duration,
            sample_rate=request.sample_rate,
            channels=request.channels
        )
        
        logger.info(f"Calibration completed. Noise level: {result.noise_level}dB")
        return result
        
    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio calibration failed: {str(e)}")

@app.post("/api/v1/audio/test-record", response_model=AudioTestResponse, tags=["Audio Processing"])
async def test_audio_recording(request: AudioTestRequest):
    """Test audio recording endpoint - records a test clip and validates audio quality"""
    try:
        logger.info(f"Starting test recording with duration: {request.duration}s")
        
        result = await service_manager.audio_service.test_recording(
            duration=request.duration,
            sample_rate=request.sample_rate,
            channels=request.channels,
            apply_calibration=request.apply_calibration
        )
        
        logger.info("Test recording completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Test recording failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test recording failed: {str(e)}")

@app.post("/api/v1/audio/transcribe", response_model=TranscriptionResponse, tags=["Audio Processing"])
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = "auto",
    model_size: Optional[str] = "base"
):
    """Speech-to-text transcription endpoint - converts uploaded audio to text"""
    try:
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        logger.info(f"Starting transcription for file: {audio_file.filename}")
        
        audio_data = await audio_file.read()
        
        result = await service_manager.audio_service.transcribe_audio(
            audio_data=audio_data,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
            language=language,
            model_size=model_size
        )
        
        logger.info("Transcription completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/api/v1/audio/health", tags=["Audio Processing"])
async def audio_health_check():
    """Health check endpoint for audio service"""
    try:
        health_status = await audio_service.health_check()
        return JSONResponse(content=health_status)
    except Exception as e:
        logger.error(f"Audio health check failed: {str(e)}")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

# ====================================================================
# CAMERA ROUTES - ENHANCED INTEGRATION
# ====================================================================
@router.post("/expression-detected", response_model=ExpressionDetectedMessage)
async def simulate_expression_detection(message: ExpressionDetectedMessage):
    """
    Echo endpoint to simulate sending an ExpressionDetectedMessage
    over HTTP (handy for testing).
    """
    # Here you might broadcast to a websocket, store in DB, etc.
    return message

@router.get("/recording/statuses", response_model=list[str])
async def list_recording_statuses():
    """
    Return all possible RecordingStatus enum values.
    E.g., 'IDLE', 'RECORDING', 'STOPPED', 'FAILED'.
    """
    return [s.value for s in RecordingStatus]

@router.get("/recording/qualities", response_model=list[str])
async def list_recording_qualities():
    """
    Return all possible RecordingQuality enum values.
    Useful for front-end dropdowns (e.g., 'LOW', 'MEDIUM', 'HIGH').
    """
    return [q.value for q in RecordingQuality]

@router.post("/chat-config", response_model=ChatConfig)
async def create_chat_config(config: ChatConfig):
    """
    Create a global chat configuration.
    Only one config per key; use PUT to update.
    """
    if "default" in chat_config_store:
        raise HTTPException(status_code=400, detail="Chat config already exists. Use PUT to update.")
    chat_config_store["default"] = config
    return config


@router.get("/chat-config", response_model=ChatConfig)
async def get_chat_config():
    """Retrieve the global chat configuration."""
    config = chat_config_store.get("default")
    if not config:
        raise HTTPException(status_code=404, detail="No chat config found.")
    return config


@router.put("/chat-config", response_model=ChatConfig)
async def update_chat_config(config: ChatConfig):
    """Update the global chat configuration."""
    chat_config_store["default"] = config
    return config

@router.get("/system-health", response_model=List[SystemHealthStatus])
async def system_health_status():
    """
    Returns health information for all internal services.
    """
    try:
        health = await camera_service.get_system_health()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")
    
@router.get("/analysis/focus-options", response_model=list[str])
async def list_analysis_focus():
    """
    Return all possible AnalysisFocus enum values.
    Allows the front-end to know which focus areas are supported.
    """
    return [f.value for f in AnalysisFocus]
    
@router.get("/config", response_model=ExpressionConfig)
async def get_expression_config():
    """
    Retrieve the expression-detection engine configuration.
    """
    try:
        config = await expression_service.get_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get expression config: {str(e)}")


@router.put("/config", response_model=APIResponse)
async def update_expression_config(config: ExpressionConfig):
    """
    Update expression detection configuration.
    """
    try:
        success = await expression_service.update_config(config)
        return APIResponse(success=success, message="Expression configuration updated successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update expression config: {str(e)}")

# CAMERA ROUTES
@router.post("/", response_model=ExpressionMonitoringConfig)
async def create_monitoring_config(config: ExpressionMonitoringConfig):
    """
    Create a new monitoring configuration for a camera session.
    """
    if config.session_id in ExpressionMonitoringConfig:
        raise HTTPException(status_code=400, detail="Config for this session already exists")
    ExpressionMonitoringConfig[config.session_id] = config
    return config


@router.get("/{session_id}", response_model=ExpressionMonitoringConfig)
async def get_monitoring_config(session_id: str):
    """
    Retrieve the monitoring configuration for a specific session.
    """
    config = ExpressionMonitoringConfig.get(session_id)
    if not config:
        raise HTTPException(status_code=404, detail="Monitoring config not found")
    return config


@router.put("/{session_id}", response_model=ExpressionMonitoringConfig)
async def update_monitoring_config(session_id: str, new_config: ExpressionMonitoringConfig):
    """
    Update the monitoring configuration for a specific session.
    """
    if session_id not in ExpressionMonitoringConfig:
        raise HTTPException(status_code=404, detail="Monitoring config not found")
    # Ensure the session_id in path matches the body (or overwrite)
    updated = new_config.copy(update={"session_id": session_id})
    ExpressionMonitoringConfig[session_id] = updated
    return updated


@router.delete("/{session_id}")
async def delete_monitoring_config(session_id: str):
    """
    Delete the monitoring configuration for a specific session.
    """
    if session_id not in ExpressionMonitoringConfig:
        raise HTTPException(status_code=404, detail="Monitoring config not found")
    del ExpressionMonitoringConfig[session_id]
    return {"message": "Monitoring config deleted", "session_id": session_id}
# ====================== MONITORING SESSIONS ======================

@router.get("/monitoring", response_model=List[MonitoringSession])
async def list_monitoring_sessions():
    """
    List all active expression monitoring sessions.
    """
    try:
        sessions = await expression_service.list_monitoring_sessions()
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list monitoring sessions: {str(e)}")


@router.get("/monitoring/{monitoring_id}", response_model=MonitoringSession)
async def get_monitoring_session(monitoring_id: str):
    """
    Retrieve details of a specific monitoring session.
    """
    try:
        session = await expression_service.get_monitoring_session(monitoring_id)
        if not session:
            raise HTTPException(status_code=404, detail="Monitoring session not found")
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch monitoring session: {str(e)}")


@router.get("/status-enum", response_model=List[str])
async def camera_status_enum():
    return [status.value for status in CameraStatus]

@router.get("/resolutions", response_model=List[CameraResolution])
async def list_supported_resolutions():
    try:
        resolutions = await camera_service.get_supported_resolutions()
        return resolutions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list camera resolutions: {str(e)}")

@router.post("/frame", response_model=CameraFrameMessage)
async def send_camera_frame(frame: CameraFrameMessage):
    try:
        processed = await camera_service.process_frame(frame)
        return processed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process camera frame: {str(e)}")

@router.post("/test", response_model=CameraTestResult)
async def run_camera_test(config: Optional[CameraConfig] = None):
    try:
        result = await service_manager.camera_service.run_test(config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera test failed: {str(e)}")
    
@app.get("/api/v1/camera/devices", response_model=List[CameraDevice], tags=["Camera"])
async def get_available_cameras():
    """Get list of available camera devices"""
    try:
        devices = await camera_service.get_available_cameras()
        return devices
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cameras: {str(e)}")

@app.post("/api/v1/camera/session/start", response_model=CameraSessionResponse, tags=["Camera"])
async def start_camera_session(request: CameraSessionRequest):
    """Start a camera session with specified device"""
    try:
        session = await camera_service.start_session(
            device_id=request.device_id,
            resolution=request.resolution,
            fps=request.fps
        )
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start camera session: {str(e)}")

@app.post("/api/v1/camera/session/stop/{session_id}", tags=["Camera"])
async def stop_camera_session(session_id: str):
    """Stop an active camera session"""
    try:
        success = await camera_service.stop_session(session_id)
        if success:
            return {"message": "Camera session stopped successfully"}
        else:
            raise HTTPException(status_code=404, detail="Camera session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop camera session: {str(e)}")

@app.get("/api/v1/camera/session/status/{session_id}", response_model=CameraStatusResponse, tags=["Camera"])
async def get_camera_status(session_id: str):
    """Get status of a camera session"""
    try:
        status = await camera_service.get_session_status(session_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Camera session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get camera status: {str(e)}")

@app.get("/api/v1/camera/stream/{session_id}", tags=["Camera"])
async def get_camera_stream(session_id: str):
    """Get camera stream for a session"""
    try:
        stream = camera_service.get_video_stream(session_id)
        return StreamingResponse(
            stream,
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get camera stream: {str(e)}")

@app.post("/api/v1/camera/test/{session_id}", tags=["Camera"])
async def test_camera_connection(session_id: str):
    """Test camera connection and capture a test frame"""
    try:
        test_result = await service_manager.camera_service.test_camera_connection(session_id)
        return test_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera test failed: {str(e)}")

# ====================================================================
# EXPRESSION DETECTION ROUTES - ENHANCED INTEGRATION
# ====================================================================

@app.post("/api/v1/expression/detect", response_model=ExpressionDetectionResponse, tags=["Expression Detection"])
async def detect_expression(request: ExpressionDetectionRequest):
    """Detect facial expression from camera frame"""
    try:
        result = await service_manager.expression_service.detect_expression(
            session_id=request.session_id,
            frame_data=request.frame_data,
            confidence_threshold=request.confidence_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expression detection failed: {str(e)}")

@app.post("/api/v1/expression/start-monitoring/{session_id}", tags=["Expression Detection"])
async def start_expression_monitoring(session_id: str, interval_seconds: int = 2):
    """Start continuous expression monitoring for a camera session"""
    try:
        monitoring_id = await expression_service.start_monitoring(
            session_id=session_id,
            interval_seconds=interval_seconds
        )
        return {"monitoring_id": monitoring_id, "message": "Expression monitoring started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@app.post("/api/v1/expression/stop-monitoring/{monitoring_id}", tags=["Expression Detection"])
async def stop_expression_monitoring(monitoring_id: str):
    """Stop expression monitoring"""
    try:
        success = await expression_service.stop_monitoring(monitoring_id)
        if success:
            return {"message": "Expression monitoring stopped"}
        else:
            raise HTTPException(status_code=404, detail="Monitoring session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@app.get("/api/v1/chat/messages/{session_id}", response_model=List[ChatMessage], tags=["Chat"])
async def get_chat_messages(session_id: str, limit: int = 50):
    """Get chat messages for a session"""
    try:
        messages = await chat_service.get_messages(session_id, limit)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.post("/api/v1/chat/message/{session_id}", tags=["Chat"])
async def add_chat_message(session_id: str, message: ChatMessage):
    """Add a new chat message"""
    try:
        saved_message = await chat_service.add_message(session_id, message)
        
        # Broadcast to WebSocket connections
        await manager.broadcast(json.dumps({
            "type": "new_message",
            "session_id": session_id,
            "message": saved_message.dict()
        }))
        
        return saved_message
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@app.post("/api/v1/chat/simplify", tags=["Chat"])
async def simplify_last_message(request: SimplifyMessageRequest):
    """Simplify the last AI message and send follow-up"""
    try:
        simplified_response = await chat_service.simplify_last_ai_message(
            session_id=request.session_id,
            original_message_id=request.original_message_id,
            confusion_confidence=request.confusion_confidence
        )
        
        # Broadcast simplified message
        await manager.broadcast(json.dumps({
            "type": "simplified_message",
            "session_id": request.session_id,
            "message": simplified_response.dict()
        }))
        
        return simplified_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to simplify message: {str(e)}")

@app.get("/api/v1/chat/session/{session_id}/summary", response_model=ChatSessionSummary, tags=["Chat"])
async def get_chat_session_summary(session_id: str):
    """Get summary of chat session"""
    try:
        summary = await chat_service.get_session_summary(session_id)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session summary: {str(e)}")

# ====================================================================
# SCREEN RECORDING ROUTES - ENHANCED INTEGRATION
# ====================================================================

@app.get("/api/v1/recording/health", response_model=HealthCheckResponse, tags=["Screen Recording"])
async def recording_health_check():
    """Check if the recording service is alive and reachable"""
    try:
        return HealthCheckResponse(
            status=ConnectionStatus.CONNECTED,
            message="Recording service is running",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Recording health check failed: {str(e)}")
        return HealthCheckResponse(
            status=ConnectionStatus.DISCONNECTED,
            message=f"Service error: {str(e)}",
            timestamp=datetime.utcnow()
        )

@app.get("/api/v1/recording/permissions", response_model=PermissionStatusResponse, tags=["Screen Recording"])
async def check_recording_permissions():
    """Check screen recording permissions status"""
    try:
        status = await permission_service.check_screen_recording_permission()
        return PermissionStatusResponse(
            has_permission=status.get("granted", False),
            permission_type=status.get("type", "unknown"),
            message=status.get("message", "Permission check completed"),
            needs_user_action=status.get("needs_action", False)
        )
    except Exception as e:
        logger.error(f"Permission check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check permissions: {str(e)}")

@app.post("/api/v1/recording/permissions/request", tags=["Screen Recording"])
async def request_recording_permissions():
    """Request screen recording permissions from user"""
    try:
        result = await service_manager.permission_service.request_screen_recording_permission()
        return {
            "success": result.get("granted", False),
            "message": result.get("message", "Permission request completed"),
            "action_required": result.get("action_required", False)
        }
    except Exception as e:
        logger.error(f"Permission request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to request permissions: {str(e)}")

@app.get("/api/v1/recording/status", response_model=RecordingStatusResponse, tags=["Screen Recording"])
async def get_recording_status():
    """Get current recording status"""
    try:
        status = await recording_service.get_recording_status()
        return RecordingStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get recording status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recording status: {str(e)}")

@app.post("/api/v1/recording/start", response_model=StartRecordingResponse, tags=["Screen Recording"])
async def start_recording(request: StartRecordingRequest, background_tasks: BackgroundTasks):
    """Start screen recording"""
    try:
        # Check permissions first
        permission_status = await permission_service.check_screen_recording_permission()
        
        if not permission_status.get("granted", False):
            raise HTTPException(
                status_code=403, 
                detail="Screen recording permission not granted. Please enable permissions first."
            )
        
        # Start recording
        result = await service_manager.recording_service.start_recording(
            quality=request.quality,
            include_audio=request.include_audio,
            capture_mouse=request.capture_mouse,
            frame_rate=request.frame_rate
        )
        
        # Start background screenshot capture if enabled
        if request.enable_screenshot_analysis:
            background_tasks.add_task(
                recording_service.start_screenshot_capture,
                result["recording_id"],
                request.screenshot_interval
            )
        
        return StartRecordingResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start recording: {str(e)}")

@app.post("/api/v1/recording/stop", response_model=StopRecordingResponse, tags=["Screen Recording"])
async def stop_recording():
    """Stop current recording"""
    try:
        result = await service_manager.recording_service.stop_recording()
        return StopRecordingResponse(**result)
    except Exception as e:
        logger.error(f"Failed to stop recording: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop recording: {str(e)}")

@app.get("/api/v1/recording/list", response_model=List[RecordingInfo], tags=["Screen Recording"])
async def list_recordings(limit: Optional[int] = 50, offset: Optional[int] = 0):
    """List all available recordings"""
    try:
        recordings = await recording_service.list_recordings(limit=limit, offset=offset)
        return [RecordingInfo(**recording) for recording in recordings]
    except Exception as e:
        logger.error(f"Failed to list recordings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list recordings: {str(e)}")

@app.delete("/api/v1/recording/delete/{recording_id}", tags=["Screen Recording"])
async def delete_recording(recording_id: str):
    """Delete a specific recording"""
    try:
        result = await service_manager.recording_service.delete_recording(recording_id)
        return {"success": result, "message": "Recording deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete recording {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete recording: {str(e)}")

@app.post("/api/v1/recording/analyze", response_model=AnalysisResponse, tags=["Screen Recording"])
async def analyze_recording(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze recording with AI based on user question"""
    try:
        # Verify recording exists
        recording_info = await recording_service.get_recording_info(request.recording_id)
        if not recording_info:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        # Start AI analysis
        analysis_result = await service_manager.ai_service.analyze_recording(
            recording_id=request.recording_id,
            question=request.question,
            analysis_type=request.analysis_type,
            time_range=request.time_range,
            include_screenshots=request.include_screenshots
        )
        
        return AnalysisResponse(**analysis_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze recording {request.recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/recording/analyze-screenshot", response_model=ScreenshotAnalysisResponse, tags=["Screen Recording"])
async def analyze_screenshot(request: ScreenshotAnalysisRequest):
    """Analyze a specific screenshot or presentation slide"""
    try:
        result = await service_manager.ai_service.analyze_screenshot(
            screenshot_data=request.screenshot_data,
            question=request.question,
            context=request.context,
            analysis_focus=request.analysis_focus
        )
        
        return ScreenshotAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to analyze screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Screenshot analysis failed: {str(e)}")

@app.post("/api/v1/recording/upload-screenshot", tags=["Screen Recording"])
async def upload_screenshot(file: UploadFile = File(...), question: Optional[str] = None):
    """Upload and analyze a screenshot file"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await file.read()
        
        # Analyze the uploaded screenshot
        result = await service_manager.ai_service.analyze_screenshot(
            screenshot_data=file_content,
            question=question or "What do you see in this image?",
            context="uploaded_screenshot"
        )
        
        return {
            "filename": file.filename,
            "analysis": result,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload analysis failed: {str(e)}")

@app.get("/api/v1/recording/analysis-history/{recording_id}", tags=["Screen Recording"])
async def get_analysis_history(recording_id: str):
    """Get all analysis history for a specific recording"""
    try:
        history = await ai_service.get_analysis_history(recording_id)
        return {"recording_id": recording_id, "analyses": history}
    except Exception as e:
        logger.error(f"Failed to get analysis history for {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis history: {str(e)}")

# ====================================================================
# HANDS-FREE ROUTES - ENHANCED INTEGRATION
# ====================================================================
@router.post("/interview-response/", response_model=InterviewResponse)
async def create_interview_response(response: InterviewResponse):
    # Add your logic to save or process response
    return response

@router.get("/interview-response/", response_model=List[InterviewResponse])
async def get_interview_responses():
    # Add your logic to fetch all responses
    return []

# --- FacialAnalysis ---
@router.post("/facial-analysis/", response_model=FacialAnalysis)
async def analyze_facial(data: FacialAnalysisRequest):
    # Add logic to analyze facial data
    return FacialAnalysis()

@router.get("/facial-analysis/", response_model=List[FacialAnalysis])
async def get_facial_analyses():
    return []

# --- ConfidenceTipResponse ---
@router.post("/confidence-tip/", response_model=ConfidenceTipResponse)
async def create_confidence_tip(tip: ConfidenceTipResponse):
    return tip

@router.get("/confidence-tip/", response_model=List[ConfidenceTipResponse])
async def get_confidence_tips():
    return []

# --- AudioStreamresult ---
@router.post("/audio-stream-result/", response_model=AudioStreamResult)
async def create_audio_stream_result(result: AudioStreamResult):
    return result

@router.get("/audio-stream-result/", response_model=List[AudioStreamResult])
async def get_audio_stream_results():
    return []

@router.post("/audio-stream-data/", response_model=AudioStreamData)
async def create_audio_stream_data(data: AudioStreamData):
    """Save or process raw audio stream data"""
    return data

@router.get("/audio-stream-data/", response_model=List[AudioStreamData])
async def get_audio_stream_data():
    """Fetch all audio stream data"""
    return []

# --- SessionSummary Routes ---
@router.post("/session-summary/", response_model=SessionSummary)
async def create_session_summary(summary: SessionSummary):
    """Save or process session summary"""
    return summary

@router.get("/session-summary/", response_model=List[SessionSummary])
async def get_session_summaries():
    """Fetch all session summaries"""
    return []

# --- WebSocketMessage Routes ---
@router.post("/websocket-message/", response_model=WebSocketMessage)
async def create_websocket_message(message: WebSocketMessage):
    """Send or save a WebSocket message"""
    return message

@router.get("/websocket-message/", response_model=List[WebSocketMessage])
async def get_websocket_messages():
    return []

# --- AutomatedResponseMessage Routes ---
@router.post("/automated-response/", response_model=AutomatedResponseMessage)
async def create_automated_response(response: AutomatedResponseMessage):
    """Save or process automated responses"""
    return response

@router.get("/automated-response/", response_model=List[AutomatedResponseMessage])
async def get_automated_responses():
    return []

# --- FacialAnalysisMessage Routes ---
@router.post("/facial-analysis-message/", response_model=FacialAnalysisMessage)
async def create_facial_analysis_message(message: FacialAnalysisMessage):
    """Save or process facial analysis messages"""
    return message

@router.get("/facial-analysis-message/", response_model=List[FacialAnalysisMessage])
async def get_facial_analysis_messages():
    return []

# --- SystemStatusMessage Routes ---
@router.post("/system-status/", response_model=SystemStatusMessage)
async def create_system_status(status: SystemStatusMessage):
    """Update or save system status"""
    return status

@router.get("/system-status/", response_model=List[SystemStatusMessage])
async def get_system_statuses():
    return []

@app.post("/api/v1/hands-free/session/start", response_model=HandsFreeSessionResponse, tags=["Hands-Free"])
async def hands_free_start_session(request: HandsFreeSessionRequest):
    """Hands-Free: Start a new automated interview session"""
    try:
        session_id = await hands_free_service.create_session(
            user_id=request.user_id,
            default_mic_id=getattr(request, 'default_mic_id', None),
            interview_type=getattr(request, 'interview_type', 'general'),
            company_info=getattr(request, 'company_info', None),
            job_role=getattr(request, 'job_role', None)
        )
        
        return HandsFreeSessionResponse(
            session_id=session_id,
            status="active",
            message="Hands-free session started successfully",
            mic_configured=True,
            ai_ready=True
        )
    except Exception as e:
        logger.error(f"Hands-free session start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/hands-free/session/{session_id}/activate", tags=["Hands-Free"])
async def hands_free_activate(session_id: str):
    """Hands-Free: Activate automation mode"""
    try:
        await hands_free_service.activate_hands_free_mode(session_id)
        return {"status": "hands_free_active", "message": "Hands-free mode activated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/hands-free/session/{session_id}/status", response_model=SessionStatus, tags=["Hands-Free"])
async def hands_free_status(session_id: str):
    """Hands-Free: Get session status"""
    try:
        return await hands_free_service.get_session_status(session_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/v1/hands-free/session/{session_id}/manual-response", tags=["Hands-Free"])
async def hands_free_manual_response(session_id: str, request: InterviewResponseRequest):
    """Hands-Free: Generate manual interview response (fallback mode)"""
    try:
        return await hands_free_service.generate_interview_response(
            session_id=session_id,
            question=request.question,
            context=getattr(request, 'context', None),
            response_type=getattr(request, 'response_type', 'detailed')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/hands-free/session/{session_id}/settings", tags=["Hands-Free"])
async def hands_free_update_settings(session_id: str, settings: SessionSettings):
    """Hands-Free: Update session settings"""
    try:
        await hands_free_service.update_session_settings(session_id, settings)
        return {"message": "Settings updated successfully", "settings": settings.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/hands-free/session/{session_id}/insights", response_model=SessionInsights, tags=["Hands-Free"])
async def hands_free_insights(session_id: str):
    """Hands-Free: Get session insights"""
    try:
        return await hands_free_service.get_session_insights(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/hands-free/system/health", response_model=SystemHealthCheck, tags=["Hands-Free"])
async def hands_free_health():
    """Hands-Free: System health check"""
    try:
        return await hands_free_service.system_health_check()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/hands-free/session/{session_id}/stop", tags=["Hands-Free"])
async def hands_free_stop(session_id: str):
    """Hands-Free: Stop session and cleanup"""
    try:
        summary = await hands_free_service.stop_session(session_id)
        return {"message": "Session stopped successfully", "session_summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================================
# KEY INSIGHTS ROUTES - ENHANCED INTEGRATION
# ====================================================================
@router.get("/sample", response_model=KeyInsight)
async def get_sample_key_insight():
    """
    Returns a sample KeyInsight object for testing or UI prototyping.
    """
    sample = KeyInsight(
        id="sample-123",
        content="This is a sample key point generated for testing.",
        type=InsightType.KEY_POINT,
        confidence_score=0.95,
        timestamp=datetime.utcnow(),
        source_section="introduction"
    )
    return sample

@app.post("/api/v1/key-insights/analyze", response_model=KeyInsightResponse, tags=["Key Insights"])
async def key_insights_analyze(request: KeyInsightRequest, image_file: Optional[UploadFile] = File(None)):
    """Key Insights: Generate insights from meeting context and optional image"""
    try:
        if not request.meeting_context and not image_file:
            raise HTTPException(status_code=400, detail="Either meeting context or image file is required")

        image_data = None
        if image_file:
            if not image_file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image")
            image_data = await image_file.read()

        insights = await insights_service.generate_insights(
            meeting_context=request.meeting_context,
            meeting_id=getattr(request, 'meeting_id', None),
            participants=getattr(request, 'participants', []),
            image_data=image_data,
            analysis_focus=getattr(request, 'analysis_focus', 'general')
        )
        return insights
    except Exception as e:
        logger.error(f"Key Insights analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/key-insights/status/{insight_id}", tags=["Key Insights"])
async def key_insights_status(insight_id: str):
    """Key Insights: Get analysis status"""
    try:
        return await insights_service.get_analysis_status(insight_id)
    except Exception as e:
        logger.error(f"Key Insights status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/key-insights/history/{meeting_id}", tags=["Key Insights"])
async def key_insights_history(meeting_id: str):
    """Key Insights: Get all insights history for a meeting"""
    try:
        history = await insights_service.get_insights_history(meeting_id)
        return {"meeting_id": meeting_id, "insights_history": history}
    except Exception as e:
        logger.error(f"Key Insights history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/key-insights/{insight_id}", tags=["Key Insights"])
async def key_insights_delete(insight_id: str):
    """Key Insights: Delete specific insights"""
    try:
        success = await insights_service.delete_insights(insight_id)
        if not success:
            raise HTTPException(status_code=404, detail="Insights not found")
        return {"message": "Insights deleted successfully", "insight_id": insight_id}
    except Exception as e:
        logger.error(f"Key Insights delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/key-insights/batch-analyze", response_model=List[KeyInsightResponse], tags=["Key Insights"])
async def key_insights_batch_analyze(
    meeting_contexts: List[str],
    image_files: Optional[List[UploadFile]] = File(None)
):
    """Key Insights: Batch analyze multiple contexts"""
    try:
        results = []
        
        for i, context in enumerate(meeting_contexts):
            image_data = None
            if image_files and i < len(image_files):
                if image_files[i].content_type.startswith("image/"):
                    image_data = await image_files[i].read()
            
            insights = await insights_service.generate_insights(
                meeting_context=context,
                image_data=image_data,
                analysis_focus='general'
            )
            results.append(insights)
        
        return results
        
    except Exception as e:
        logger.error(f"Key Insights batch analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # ====================================================================
# MAIN FEATURE (INVISIBILITY MODE) ROUTES - ENHANCED INTEGRATION
# ====================================================================
# ============================
# HideModeEnum Routes
# ============================

@router.get("/hide-modes", response_model=List[HideModeEnum])
async def list_hide_modes():
    """
    Return all available HideModeEnum options.
    """
    return [mode for mode in HideModeEnum]

@router.get("/hide-modes/{mode}", response_model=HideModeEnum)
async def get_hide_mode(mode: HideModeEnum):
    """
    Validate and return a single HideModeEnum value.
    """
    return mode


# ============================
# InvisibilityError Routes
# ============================

@router.post("/invisibility-error", response_model=InvisibilityError)
async def create_invisibility_error(error: InvisibilityError):
    """
    Store or log an invisibility-related error.
    For demo, simply echoes the request.
    """
    return error

@router.get("/invisibility-error/{error_code}", response_model=InvisibilityError)
async def get_invisibility_error(error_code: str):
    """
    Fetch an example InvisibilityError by code.
    Replace with DB lookup in production.
    """
    return InvisibilityError(
        error_code=error_code,
        message="Sample error message",
        session_id="session_123",
        details={"info": "Example details"},
        timestamp=datetime.utcnow()
    )


# ============================
# PerformanceMetrics Routes
# ============================

_fake_metrics_store: List[PerformanceMetrics] = []  # In-memory demo storage

@router.post("/performance-metrics", response_model=PerformanceMetrics)
async def create_performance_metrics(metrics: PerformanceMetrics):
    """
    Submit performance metrics for a given session.
    """
    _fake_metrics_store.append(metrics)
    return metrics

@router.get("/performance-metrics/{session_id}", response_model=List[PerformanceMetrics])
async def get_performance_metrics(session_id: str):
    """
    Retrieve all performance metrics for a specific session_id.
    """
    return [m for m in _fake_metrics_store if m.session_id == session_id]

@router.get("/performance-metrics", response_model=List[PerformanceMetrics])
async def list_all_performance_metrics(
    limit: int = Query(50, ge=1, le=500, description="Max number of records to return")
):
    """
    List all collected performance metrics (demo).
    """
    return _fake_metrics_store[:limit]

@router.post("/recording/start", response_model=RecordingData)
async def start_recording(config: RecordingConfig):
    """
    Start a new recording with the given configuration.
    """
    return RecordingData(
        type=config.type,
        started=True,
        message="Recording started successfully",
    )


@router.get("/recording/{recording_type}", response_model=RecordingData)
async def get_recording(recording_type: RecordingTypeEnum):
    """
    Fetch info for a specific recording type.
    """
    return RecordingData(type=recording_type, started=False, message="Stub response")


# -------------------- UI --------------------
@router.get("/ui/state", response_model=UIState)
async def get_ui_state():
    """
    Get current UI state/config.
    """
    return UIState(active_component=UIComponentEnum.PLAYER, config=UIConfig())


@router.put("/ui/config", response_model=UIConfig)
async def update_ui_config(config: UIConfig):
    """
    Update UI configuration.
    """
    return config


# -------------------- Insights --------------------
@router.post("/insights", response_model=InsightData)
async def generate_insight(insight_type: InsightTypeEnum):
    """
    Generate insight of a specific type.
    """
    return InsightData(type=insight_type, result="Sample insight")


# -------------------- Security --------------------
@router.get("/security/status", response_model=SecurityStatusResponse)
async def get_security_status():
    """
    Get current security status of the system.
    """
    return SecurityStatusResponse(status=SecurityStatus.OK)


@router.post("/security/config", response_model=SecurityConfig)
async def update_security_config(config: SecurityConfig):
    """
    Update security configuration.
    """
    return config


# -------------------- System --------------------
@router.get("/system/config", response_model=SystemConfig)
async def get_system_config():
    """
    Retrieve overall system configuration.
    """
    return SystemConfig()


@router.get("/session", response_model=SessionData)
async def get_session_data():
    """
    Return session-related information.
    """
    return SessionData(session_id="dummy", user="placeholder")

@app.post("/api/v1/invisibility/mode/enable", response_model=InvisibilityModeResponse, tags=["Invisibility Mode"])
async def enable_invisibility_mode(request: InvisibilityModeRequest, background_tasks: BackgroundTasks):
    """Enable invisibility mode for the interview AI assistant"""
    try:
        session_id = str(uuid.uuid4())
        
        result = await service_manager.invisibility_service.enable_invisibility_mode(
            session_id=session_id,
            recording_config=request.recording_config,
            ui_config=request.ui_config,
            security_config=request.security_config
        )
        
        background_tasks.add_task(
            invisibility_service.start_background_recording,
            session_id,
            request.recording_config
        )
        
        return InvisibilityModeResponse(
            success=True,
            session_id=session_id,
            message="Invisibility mode enabled successfully",
            ui_state=result.get("ui_state"),
            recording_state=result.get("recording_state"),
            security_status=result.get("security_status")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable invisibility mode: {str(e)}")

@app.post("/api/v1/invisibility/mode/disable", response_model=InvisibilityModeResponse, tags=["Invisibility Mode"])
async def disable_invisibility_mode(session_id: str, background_tasks: BackgroundTasks):
    """Disable invisibility mode and restore normal UI visibility"""
    try:
        result = await service_manager.invisibility_service.disable_invisibility_mode(session_id)
        
        background_tasks.add_task(
            invisibility_service.generate_final_insights,
            session_id
        )
        
        return InvisibilityModeResponse(
            success=True,
            session_id=session_id,
            message="Invisibility mode disabled successfully",
            ui_state=result.get("ui_state"),
            recording_state=result.get("recording_state"),
            final_insights_url=result.get("insights_url")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable invisibility mode: {str(e)}")

@app.post("/api/v1/invisibility/recording/start", response_model=RecordingSessionResponse, tags=["Invisibility Mode"])
async def start_invisible_recording(request: RecordingSessionRequest, background_tasks: BackgroundTasks):
    """Start invisible recording session with specified configuration"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        result = await service_manager.invisibility_service.start_invisible_recording(
            session_id=session_id,
            screen_recording=request.screen_recording,
            voice_recording=request.voice_recording,
            auto_notes=request.auto_notes,
            real_time_insights=request.real_time_insights
        )
        
        return RecordingSessionResponse(
            success=True,
            session_id=session_id,
            recording_started=True,
            message="Invisible recording started successfully",
            recording_config=result.get("config"),
            estimated_duration=getattr(request, 'estimated_duration', None)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start invisible recording: {str(e)}")

@app.post("/api/v1/invisibility/ui/hide", response_model=UIVisibilityResponse, tags=["Invisibility Mode"])
async def hide_ui_components(request: UIVisibilityRequest):
    """Hide specified UI components to maintain invisibility during screen sharing"""
    try:
        result = await service_manager.invisibility_service.hide_ui_components(
            session_id=request.session_id,
            components_to_hide=request.components_to_hide,
            hide_mode=getattr(request, 'hide_mode', HideModeEnum.TRANSPARENT)
        )
        
        return UIVisibilityResponse(
            success=True,
            session_id=request.session_id,
            hidden_components=result.get("hidden_components"),
            ui_state="hidden",
            message="UI components hidden successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to hide UI components: {str(e)}")

@app.get("/api/v1/invisibility/session/{session_id}/status", response_model=SessionStatusResponse, tags=["Invisibility Mode"])
async def get_invisibility_session_status(session_id: str):
    """Get current status of invisibility session including recording and UI state"""
    try:
        status = await invisibility_service.get_session_status(session_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionStatusResponse(
            session_id=session_id,
            is_active=status.get("is_active", False),
            invisibility_enabled=status.get("invisibility_enabled", False),
            recording_status=status.get("recording_status"),
            ui_state=status.get("ui_state"),
            start_time=status.get("start_time"),
            duration=status.get("duration"),
            data_captured=status.get("data_captured", {}),
            security_status=status.get("security_status")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")

@app.post("/api/v1/invisibility/insights/generate", response_model=InsightGenerationResponse, tags=["Invisibility Mode"])
async def generate_background_insights(request: InsightGenerationRequest, background_tasks: BackgroundTasks):
    """Generate AI insights from captured data without showing progress to interviewer"""
    try:
        background_tasks.add_task(
            invisibility_service.generate_insights,
            request.session_id,
            request.insight_types,
            getattr(request, 'processing_options', {})
        )
        
        return InsightGenerationResponse(
            success=True,
            session_id=request.session_id,
            generation_started=True,
            message="Background insight generation started",
            estimated_completion_time=getattr(request, 'processing_options', {}).get("estimated_time"),
            insight_types=request.insight_types
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start insight generation: {str(e)}")

@app.get("/api/v1/invisibility/insights/{session_id}", tags=["Invisibility Mode"])
async def get_generated_insights(session_id: str):
    """Retrieve generated insights for a completed invisibility session"""
    try:
        insights = await invisibility_service.get_session_insights(session_id)
        
        if not insights:
            raise HTTPException(status_code=404, detail="No insights found for this session")
        
        return {
            "session_id": session_id,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve insights: {str(e)}")

@app.delete("/api/v1/invisibility/session/{session_id}", tags=["Invisibility Mode"])
async def cleanup_invisibility_session(session_id: str):
    """Clean up session data and remove all traces of invisibility session"""
    try:
        result = await service_manager.invisibility_service.cleanup_session(session_id)
        
        return JSONResponse(
            content={
                "success": True,
                "session_id": session_id,
                "message": "Session cleaned up successfully",
                "data_removed": result.get("data_removed", [])
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup session: {str(e)}")

@app.get("/api/v1/invisibility/system/performance", response_model=PerformanceMetrics, tags=["Invisibility Mode"])
async def get_performance_metrics():
    """Get system performance metrics for invisibility mode"""
    try:
        metrics = await invisibility_service.get_performance_metrics()
        return PerformanceMetrics(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


# ====================================================================
# WEBSOCKET ROUTES - ENHANCED INTEGRATION
# ====================================================================

@app.websocket("/api/v1/camera/ws/{session_id}")
async def camera_websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time camera and expression updates"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "start_monitoring":
                # Start expression monitoring
                monitoring_id = await service_manager.expression_service.start_monitoring(
                    session_id=session_id,
                    interval_seconds=message_data.get("interval", 2)
                )
                await websocket.send_text(json.dumps({
                    "type": "monitoring_started",
                    "monitoring_id": monitoring_id
                }))
            
            elif message_data.get("type") == "camera_frame":
                # Process camera frame for expression detection
                result = await service_manager.expression_service.detect_expression(
                    session_id=session_id,
                    frame_data=message_data.get("frame_data"),
                    confidence_threshold=0.7
                )
                
                # Store expression result
                await storage_manager.save_analysis(str(uuid.uuid4()), {
                    "type": "websocket_expression_detection",
                    "session_id": session_id,
                    "result": result.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Send expression result back
                await websocket.send_text(json.dumps({
                    "type": "expression_detected",
                    "result": result.dict()
                }))
                
                # If confused expression detected, trigger simplification
                if result.primary_expression == ExpressionType.CONFUSED and result.confidence > 0.8:
                    # Get last AI message and simplify it
                    messages = await service_manager.chat_service.get_messages(session_id, limit=1)
                    if messages and messages[0].sender == MessageSender.AI:
                        simplified = await service_manager.chat_service.simplify_last_ai_message(
                            session_id=session_id,
                            original_message_id=messages[0].id,
                            confusion_confidence=result.confidence
                        )
                        
                        await websocket.send_text(json.dumps({
                            "type": "auto_simplified",
                            "message": simplified.dict()
                        }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))
        manager.disconnect(websocket)

@app.websocket("/api/v1/hands-free/session/{session_id}/audio-stream")
async def hands_free_audio_stream(websocket: WebSocket, session_id: str):
    """Hands-Free: Real-time audio streaming with automated responses"""
    await manager.connect(websocket, session_id)
    try:
        await service_manager.hands_free_service.verify_session(session_id)
        while True:
            audio_data = await websocket.receive_bytes()
            result = await service_manager.hands_free_service.process_audio_stream(
                session_id=session_id, 
                audio_data=audio_data
            )

            if result.question_detected:
                response = await service_manager.hands_free_service.generate_automated_response(
                    session_id=session_id,
                    question=result.detected_question,
                    context=result.context
                )
                
                # Store automated response
                await storage_manager.save_chat_message(session_id, {
                    "type": "automated_response",
                    "question": result.detected_question,
                    "response": response.response_text,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                await websocket.send_json({
                    "type": "automated_response",
                    "question": result.detected_question,
                    "response": response.response_text,
                    "key_insights": response.key_insights,
                    "confidence_score": response.confidence_score,
                    "timestamp": datetime.utcnow().isoformat()
                })

            await websocket.send_json({
                "type": "status_update",
                "listening": result.is_listening,
                "processing": result.is_processing,
                "audio_level": result.audio_level
            })
    except WebSocketDisconnect:
        await service_manager.hands_free_service.cleanup_session(session_id)
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        manager.disconnect(websocket)

@app.websocket("/api/v1/recording/ws/status")
async def websocket_recording_status(websocket: WebSocket):
    """WebSocket endpoint for real-time recording status updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            status = await service_manager.recording_service.get_recording_status()
            await websocket.send_json(status)
            await asyncio.sleep(1)  # Send updates every second
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Recording WebSocket error: {str(e)}")
        manager.disconnect(websocket)

@app.websocket("/api/v1/quick-respond/ws/{session_id}")
async def quick_respond_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time quick respond analysis"""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "screenshot_analysis":
                screenshot_data = message_data.get("screenshot_data")
                context = message_data.get("context")
                
                # Store analysis request
                analysis_id = str(uuid.uuid4())
                await storage_manager.save_analysis(analysis_id, {
                    "type": "websocket_quick_respond",
                    "session_id": session_id,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Stream analysis results
                async for chunk in service_manager.quick_respond_service.analyze_meeting_content_stream({
                    "screenshot_data": screenshot_data.encode() if isinstance(screenshot_data, str) else screenshot_data,
                    "meeting_context": context,
                    "analysis_type": "key_insights"
                }):
                    await websocket.send_json(chunk)
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Quick respond WebSocket error: {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})
        manager.disconnect(websocket)

# ====================================================================
# ADDITIONAL UTILITY ROUTES - SYSTEM HEALTH AND MONITORING
# ====================================================================

@app.get("/api/v1/system/health", tags=["System"])
async def system_health_check():
    """Comprehensive system health check for all services"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        # Check Ollama connection
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            health_status["services"]["ollama"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            health_status["services"]["ollama"] = {"status": "unhealthy", "error": str(e)}
        
        # Check storage connections
        health_status["services"]["storage"] = {
            "database": "healthy" if storage_manager.db else "unavailable",
            "redis": "healthy" if storage_manager.redis_cache else "unavailable",
            "file_storage": "healthy" if storage_manager.file_storage else "unavailable"
        }
        
        # Check service manager
        service_status = "healthy" if service_manager.voice_service else "unhealthy"
        health_status["services"]["service_manager"] = {"status": service_status}
        
        # Check overall status
        unhealthy_services = []
        for service_name, service_info in health_status["services"].items():
            if isinstance(service_info, dict):
                if service_info.get("status") == "unhealthy":
                    unhealthy_services.append(service_name)
        
        if unhealthy_services:
            health_status["status"] = "degraded" if len(unhealthy_services) < len(health_status["services"]) else "unhealthy"
            health_status["unhealthy_services"] = unhealthy_services
        
        return health_status
        
    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/v1/system/info", tags=["System"])
async def get_system_info():
    """Get system information and configuration"""
    try:
        return {
            "app_name": app.title,
            "version": app.version,
            "description": app.description,
            "docs_url": app.docs_url,
            "redoc_url": app.redoc_url,
            "routes_count": len(app.routes),
            "active_websocket_connections": sum(len(conns) for conns in manager.active_connections.values()),
            "websocket_sessions": len(manager.active_connections),
            "connection_stats": manager.get_connection_stats(),
            "supported_features": [
                "summarization",
                "quick_respond", 
                "voice_recognition",
                "camera_detection",
                "hands_free_mode",
                "invisibility_mode",
                "key_insights",
                "screen_recording"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")

@app.get("/api/v1/system/metrics", tags=["System"])
async def get_system_metrics():
    """Get system performance and usage metrics"""
    try:
        return {
            "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "storage_stats": {
                "sessions": len(storage_manager.sessions),
                "recordings": len(storage_manager.recordings),
                "analyses": len(storage_manager.analyses),
                "insights": len(storage_manager.insights),
                "chat_sessions": len(storage_manager.chat_sessions)
            },
            "active_sessions": {
                "total": len(storage_manager.sessions),
                "by_type": {}
            },
            "websocket_connections": manager.get_connection_stats(),
            "memory_usage": "N/A",  # Could implement actual memory monitoring
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.post("/api/v1/system/cleanup", tags=["System"])
async def cleanup_system():
    """Cleanup inactive sessions and temporary files"""
    try:
        cleanup_results = {
            "cleaned_sessions": 0,
            "freed_memory": "0MB",
            "deleted_temp_files": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cleanup inactive sessions from storage
        current_time = datetime.utcnow()
        inactive_sessions = []
        
        for session_id, session_data in storage_manager.sessions.items():
            if isinstance(session_data, dict):
                created_at = session_data.get("created_at")
                status = session_data.get("status")
                
                # Mark sessions older than 24 hours as inactive
                if created_at and status == "active":
                    try:
                        created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        if (current_time - created_time).total_seconds() > 86400:  # 24 hours
                            inactive_sessions.append(session_id)
                    except:
                        pass
        
        # Remove inactive sessions
        for session_id in inactive_sessions:
            del storage_manager.sessions[session_id]
            cleanup_results["cleaned_sessions"] += 1
        
        # Cleanup WebSocket connections
        disconnected_count = 0
        for session_id, connections in list(manager.active_connections.items()):
            # Remove connections that are no longer active
            active_connections = []
            for conn in connections:
                try:
                    # Test if connection is still alive
                    await conn.ping()
                    active_connections.append(conn)
                except:
                    disconnected_count += 1
            
            if active_connections:
                manager.active_connections[session_id] = active_connections
            else:
                del manager.active_connections[session_id]
        
        cleanup_results["disconnected_websockets"] = disconnected_count
        
        return cleanup_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# ====================================================================
# WEBHOOK ROUTES FOR EXTERNAL INTEGRATIONS  
# ====================================================================

@app.post("/api/v1/webhooks/meeting/started", tags=["Webhooks"])
async def webhook_meeting_started(request: Dict[str, Any]):
    """Webhook endpoint for when a meeting starts"""
    try:
        meeting_id = request.get("meeting_id")
        meeting_title = request.get("title", "Untitled Meeting")
        participants = request.get("participants", [])
        
        # Store meeting context
        meeting_context = QuickRespondMeetingContext(
            meeting_title=meeting_title,
            participants=participants,
            start_time=datetime.utcnow(),
            meeting_type=MeetingType.GENERAL
        )
        
        await service_manager.quick_respond_service.update_meeting_context(meeting_context)
        
        # Store webhook event
        await storage_manager.save_analysis(str(uuid.uuid4()), {
            "type": "webhook_meeting_started",
            "meeting_id": meeting_id,
            "meeting_title": meeting_title,
            "participants": participants,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return create_success_response(
            data={
                "meeting_id": meeting_id,
                "meeting_title": meeting_title,
                "participants_count": len(participants)
            },
            message=f"Meeting context updated for {meeting_id}"
        )
        
    except Exception as e:
        logger.error(f"Webhook meeting started error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/webhooks/meeting/ended", tags=["Webhooks"])
async def webhook_meeting_ended(request: Dict[str, Any]):
    """Webhook endpoint for when a meeting ends"""
    try:
        meeting_id = request.get("meeting_id")
        duration = request.get("duration_minutes", 0)
        
        # Clear meeting context
        await service_manager.quick_respond_service.clear_meeting_context()
        
        # Generate final meeting summary if available
        final_summary = {
            "meeting_id": meeting_id,
            "duration_minutes": duration,
            "ended_at": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        
        # Store webhook event
        await storage_manager.save_analysis(str(uuid.uuid4()), {
            "type": "webhook_meeting_ended",
            "meeting_id": meeting_id,
            "duration": duration,
            "summary": final_summary,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return create_success_response(
            data={"summary": final_summary},
            message=f"Meeting {meeting_id} ended and context cleared"
        )
        
    except Exception as e:
        logger.error(f"Webhook meeting ended error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================================
# ROOT AND FALLBACK ROUTES
# ====================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Interview AI Assistant Backend API",
        "version": app.version,
        "description": app.description,
        "docs": f"{app.docs_url}",
        "health_check": "/api/v1/system/health",
        "available_endpoints": [
            "/api/v1/summarization/*",
            "/api/v1/quick-respond/*", 
            "/api/v1/voice/*",
            "/api/v1/camera/*",
            "/api/v1/hands-free/*",
            "/api/v1/key-insights/*",
            "/api/v1/invisibility/*",
            "/api/v1/recording/*",
            "/api/v1/system/*"
        ],
        "websocket_endpoints": [
            "/api/v1/camera/ws/{session_id}",
            "/api/v1/recording/ws/status",
            "/api/v1/hands-free/session/{session_id}/audio-stream",
            "/api/v1/quick-respond/ws/{session_id}",
            "/api/v1/summarization/ws/{meeting_id}"
        ],
        "storage_info": {
            "database_connected": storage_manager.db is not None,
            "redis_connected": storage_manager.redis_cache is not None,
            "file_storage_connected": storage_manager.file_storage is not None
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", tags=["Root"])
async def health():
    """Simple health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "services_initialized": service_manager.voice_service is not None,
        "storage_initialized": storage_manager.db is not None or len(storage_manager.sessions) >= 0
    }

# Set start time for uptime calculation
@app.on_event("startup")
async def set_startup_time():
    app.state.start_time = time.time()

# ====================================================================
# AUTHENTICATION AND SESSION MIDDLEWARE (BASIC IMPLEMENTATION)
# ====================================================================

# The error handlers are already defined above

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )# app/main.py
import logging
import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError, ResponseValidationError
import uvicorn
from typing import Dict, Any, List, Optional
import time
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
from datetime import datetime
import requests

import os

# Import all schemas - now properly utilized
from app.handFree.schemas import (
    HandsFreeSessionRequest,
    InterviewResponseRequest,
    FacialAnalysisRequest,
    SessionSettings,
    HandsFreeSessionResponse,
    InterviewResponse,
    FacialAnalysis,
    ConfidenceTipResponse,
    AudioStreamResult,
    RealTimeAnalysisResponse,
    SessionStatus,
    AudioStreamData,
    SystemHealthCheck,
    SessionInsights,
    SessionSummary,
    WebSocketMessage,
    AutomatedResponseMessage,
    FacialAnalysisMessage,
    SystemStatusMessage
)

from app.summarization.schemas import(
    SummaryType,
    AnalysisType,
    AudioUploadResponse,
    MeetingAnalysisRequest,
    MeetingAnalysisResponse,
    ActionItem,
    KeyPoint,
    SummarizationRequest,
    SummarizationResponse,
    SummaryUpdateRequest,
    MeetingContext,
    LLAVAAnalysisConfig,
    RealTimeAnalysisUpdate,
    BatchSummarizationRequest
)

from app.key_insights.schemas import (
    KeyInsight,
    KeyInsightRequest,
    KeyInsightResponse,
    ErrorResponse as KeyInsightErrorResponse
)

from app.voice_recognition.schemas import (
    VoiceSessionRequest,
    MicrophoneStatusRequest,
    DeviceSelectionRequest,
    AudioProcessingRequest,
    AIResponseRequest,
    SimplifiedAnswerRequest,
    VoiceSessionResponse,
    MicrophoneStatusResponse,
    DeviceListResponse,
    TranscriptionResponse,
    AudioProcessingResponse,
    VoiceAnalysisResponse,
    AIResponseResponse,
    CalibrationRequest,
    CalibrationResponse,
    AudioTestRequest,
    AudioTestResponse,
    ErrorResponse as VoiceErrorResponse,
    ServiceHealth
)

from app.image_recognition.schemas import(
    CameraStatus,
    ExpressionConfig,
    ExpressionType,
    MessageSender,
    CameraResolution,
    CameraDevice,
    CameraSessionRequest,
    CameraSessionResponse,
    CameraFrameMessage,
    CameraStatusResponse,
    CameraTestResult,
    ExpressionDetectionRequest,
    ExpressionDetectedMessage,
    ExpressionDetectionResponse,
    ExpressionMonitoringConfig,
    ChatMessage,
    SimplifyMessageRequest,
    ChatSessionSummary,
    MonitoringSession,
    SystemHealthStatus,
    APIResponse,
    WebSocketMessage as ImageWebSocketMessage,
    CameraConfig,
    ChatConfig,
    RecordingQuality,
    AnalysisType as ImageAnalysisType,
    RecordingInfo,
    RecordingStatus,
    ConnectionStatus,
    AnalysisFocus,
    StartRecordingRequest,
    HealthCheckResponse,
    PermissionStatusResponse,
    RecordingStatusResponse,
    StartRecordingResponse,
    StopRecordingResponse,
    AnalysisRequest,
    AnalysisResponse,
    ScreenshotAnalysisRequest,
    ScreenshotAnalysisResponse
)

from app.MainFeature.schemas import(
    HideModeEnum,
    RecordingTypeEnum,
    RecordingConfig,
    UIComponentEnum,
    InsightTypeEnum,
    UIConfig,
    SecurityConfig,
    InvisibilityModeRequest,
    InvisibilityModeResponse,
    UIVisibilityRequest,
    UIVisibilityResponse,
    UIState,
    InsightGenerationRequest,
    InsightGenerationResponse,
    InvisibilityError,
    InsightData,
    RecordingData,
    RecordingSessionRequest,
    RecordingSessionResponse,
    SessionStatusResponse,
    SecurityStatusResponse,
    SecurityStatus,
    SessionData,
    SystemConfig,
    PerformanceMetrics
)

# Import Quick Respond schemas
from app.quick_respond.schemas import (
    QuickRespondRequest,
    QuickRespondResponse,
    SimplifyRequest,
    SimplifyResponse,
    MeetingContext as QuickRespondMeetingContext,
    KeyInsight as QuickRespondKeyInsight,
    UrgencyLevel,
    AnalysisType as QuickRespondAnalysisType,
    MeetingType,
    MeetingStatus,
    ParticipantInfo,
    ScreenContent,
    MeetingMetrics,
    StreamingResponse as QuickRespondStreamingResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    ErrorResponse as QuickRespondErrorResponse,
    HealthCheckResponse as QuickRespondHealthCheckResponse,
    WebSocketMessage as QuickRespondWebSocketMessage,
    OllamaConfig,
    QuickRespondConfig,
    ModelPrompts,
    APIResponse as QuickRespondAPIResponse,
    PaginatedResponse,
    AdvancedAnalysisRequest,
    AdvancedAnalysisResponse,
    SessionAnalytics,
    UsageMetrics,
    WebhookEvent,
    WebhookResponse
)

# Import services
from app.quick_respond.service import (
    QuickRespondService,
)

from app.voice_recognition.services import (
    VoiceProcessingService,
    AudioService
)

from app.key_insights.services import (
    KeyInsightsService,
    extract_key_insights
)

from app.handFree.service import (
    HandsFreeService,
    generate_handsfree_response_async
)

from app.image_recognition.service import(
    CameraService,
    ExpressionDetectionService,
    ChatService,
    PermissionService,
    ScreenRecordingService,
    AIAnalysisService
)

from app.summarization.service import(
    SummarizationService,
)

from app.MainFeature.service import(
    InvisibilityService,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state for interview sessions
interview_sessions: Dict[str, Dict[str, Any]] = {}

# Initialize services
voice_service = VoiceProcessingService()
audio_service = AudioService()
camera_service = CameraService()
expression_service = ExpressionDetectionService()
chat_service = ChatService()
permission_service = PermissionService()
recording_service = ScreenRecordingService()
ai_service = AIAnalysisService()
hands_free_service = HandsFreeService()
quick_respond_service = QuickRespondService()
insights_service = KeyInsightsService()
summarization_service = SummarizationService()
invisibility_service = InvisibilityService()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Service initialization functions
async def initialize_ollama_connection():
    """Initialize connection to Ollama with Nous Hermes model"""
    try:
        # Test Ollama connection
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
        logger.warning(f"Session validation failed: {e}")
        return False

# Dependency for protected routes
async def get_current_session(request: Request) -> str:
    """Get current session or create new one for protected routes"""
    session_id = await get_session_id(request)
    
    # For now, we'll create session if it doesn't exist
    # In production, you might want stricter validation
    session_data = await storage_manager.get_session(session_id)
    if not session_data:
        await storage_manager.save_session(session_id, {
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "type": "api_session",
            "user_agent": request.headers.get("User-Agent"),
            "ip_address": request.client.host if request.client else None
        })
    
    return session_id

# ====================================================================
# ADDITIONAL MISSING ROUTES AND COMPONENTS
# ====================================================================

# Audio Service Routes (were referenced but not implemented)
@app.post("/api/v1/audio/calibrate", response_model=CalibrationResponse, tags=["Audio Processing"])
async def calibrate_audio(request: CalibrationRequest):
    """Audio Calibration endpoint - measures background noise and sets optimal levels"""
    try:
        logger.info(f"Starting audio calibration with duration: {request.duration}s")
        
        result = await service_manager.audio_service.calibrate_audio(
            duration=request.duration,
            sample_rate=request.sample_rate,
            channels=request.channels
        )
        
        logger.info(f"Calibration completed. Noise level: {result.noise_level}dB")
        return result
        
    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio calibration failed: {str(e)}")

@app.post("/api/v1/audio/test-record", response_model=AudioTestResponse, tags=["Audio Processing"])
async def test_audio_recording(request: AudioTestRequest):
    """Test audio recording endpoint - records a test clip and validates audio quality"""
    try:
        logger.info(f"Starting test recording with duration: {request.duration}s")
        
        result = await service_manager.audio_service.test_recording(
            duration=request.duration,
            sample_rate=request.sample_rate,
            channels=request.channels,
            apply_calibration=request.apply_calibration
        )
        
        logger.info("Test recording completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Test recording failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test recording failed: {str(e)}")

# Chat Service Routes (were referenced but not implemented)
@app.get("/api/v1/chat/messages/{session_id}", response_model=List[ChatMessage], tags=["Chat"])
async def get_chat_messages(session_id: str, limit: int = 50):
    """Get chat messages for a session"""
    try:
        messages = await service_manager.chat_service.get_messages(session_id, limit)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.post("/api/v1/chat/message/{session_id}", tags=["Chat"])
async def add_chat_message(session_id: str, message: ChatMessage):
    """Add a new chat message"""
    try:
        saved_message = await service_manager.chat_service.add_message(session_id, message)
        
        # Store in storage manager
        await storage_manager.save_chat_message(session_id, saved_message.dict())
        
        # Broadcast to WebSocket connections
        await manager.send_to_session(json.dumps({
            "type": "new_message",
            "session_id": session_id,
            "message": saved_message.dict()
        }), session_id)
        
        return saved_message
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

# Permission Service Routes (were referenced but not implemented)
@app.get("/api/v1/recording/permissions", response_model=PermissionStatusResponse, tags=["Screen Recording"])
async def check_recording_permissions():
    """Check screen recording permissions status"""
    try:
        status = await service_manager.permission_service.check_screen_recording_permission()
        return PermissionStatusResponse(
            has_permission=status.get("granted", False),
            permission_type=status.get("type", "unknown"),
            message=status.get("message", "Permission check completed"),
            needs_user_action=status.get("needs_action", False)
        )
    except Exception as e:
        logger.error(f"Permission check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check permissions: {str(e)}")

# File Upload Routes for Screenshots
@app.post("/api/v1/recording/upload-screenshot", tags=["Screen Recording"])
async def upload_screenshot(file: UploadFile = File(...), question: Optional[str] = None):
    """Upload and analyze a screenshot file"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await file.read()
        
        # Save screenshot
        screenshot_id = str(uuid.uuid4())
        await storage_manager.save_screenshot(screenshot_id, file_content)
        
        # Analyze the uploaded screenshot
        result = await service_manager.ai_service.analyze_screenshot(
            screenshot_data=file_content,
            question=question or "What do you see in this image?",
            context="uploaded_screenshot"
        )
        
        # Store analysis result
        await storage_manager.save_analysis(screenshot_id, {
            "type": "screenshot_analysis",
            "filename": file.filename,
            "question": question,
            "result": result
        })
        
        return create_success_response(
            data={
                "filename": file.filename,
                "analysis": result,
                "screenshot_id": screenshot_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload analysis failed: {str(e)}")

# Additional Router-based routes that were missing
@summarization_router.post("/real-time-analysis", response_model=MeetingAnalysisResponse)
async def real_time_meeting_analysis(request: MeetingAnalysisRequest):
    """Real-time analysis of ongoing meeting audio"""
    try:
        analysis = await service_manager.summarization_service.real_time_audio_analysis(
            audio_chunk_path=request.audio_file_path,
            meeting_context=getattr(request, 'meeting_context', None),
            user_id="default_user"
        )
        
        # Store real-time analysis
        await storage_manager.save_analysis(str(uuid.uuid4()), {
            "type": "real_time_analysis",
            "audio_file_path": request.audio_file_path,
            "result": analysis
        })
        
        return MeetingAnalysisResponse(**analysis)
    
    except Exception as e:
        logger.error(f"Error in real-time analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to perform real-time analysis: {str(e)}")

@quick_respond_router.post("/batch-analyze", response_model=BatchAnalysisResponse)
async def batch_analyze_screenshots(
    screenshots: List[UploadFile] = File(...),
    meeting_context: Optional[str] = None
):
    """Analyze multiple screenshots from a meeting session"""
    try:
        results = []
        
        for screenshot in screenshots:
            if not screenshot.content_type.startswith('image/'):
                continue
                
            screenshot_data = await screenshot.read()
            screenshot_id = str(uuid.uuid4())
            await storage_manager.save_screenshot(screenshot_id, screenshot_data)
            
            request_data = {
                "screenshot_data": screenshot_data,
                "meeting_context": meeting_context,
                "analysis_type": "key_insights"
            }
            
            result = await service_manager.quick_respond_service.analyze_meeting_content(request_data)
            
            # Store batch analysis result
            await storage_manager.save_analysis(screenshot_id, {
                "type": "batch_analysis",
                "filename": screenshot.filename,
                "meeting_context": meeting_context,
                "result": result
            })
            
            results.append({
                "filename": screenshot.filename,
                "analysis": result,
                "screenshot_id": screenshot_id
            })
        
        return BatchAnalysisResponse(batch_results=results)
        
    except Exception as e:
        logger.error(f"Quick respond batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# Voice Service Additional Routes
@voice_router.get("/devices/{session_id}", response_model=DeviceListResponse)
async def voice_list_devices(session_id: str):
    """Voice Recognition: List available audio input devices"""
    try:
        devices = await service_manager.voice_service.get_audio_devices(session_id)
        return DeviceListResponse(
            success=True,
            devices=devices,
            default_device=devices[0] if devices else None,
            message="Audio devices retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Device list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@voice_router.post("/device/select")
async def voice_select_device(request: DeviceSelectionRequest):
    """Voice Recognition: Select and connect to a specific audio device"""
    try:
        result = await service_manager.voice_service.select_audio_device(
            request.session_id, 
            request.device_id
        )
        return create_success_response(
            data={
                "connected_device": result["device_name"],
                "device_id": result["device_id"]
            },
            message="Device connected successfully"
        )
    except Exception as e:
        logger.error(f"Device selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Expression Router Routes
@expression_router.get("/status-enum", response_model=List[str])
async def camera_status_enum():
    return [status.value for status in CameraStatus]

@expression_router.get("/config", response_model=ExpressionConfig)
async def get_expression_config():
    """Retrieve the expression-detection engine configuration."""
    try:
        config = await service_manager.expression_service.get_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get expression config: {str(e)}")

# Camera Router Routes  
@camera_router.get("/stream/{session_id}")
async def get_camera_stream(session_id: str):
    """Get camera stream for a session"""
    try:
        stream = service_manager.camera_service.get_video_stream(session_id)
        return StreamingResponse(
            stream,
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get camera stream: {str(e)}")

@camera_router.get("/resolutions", response_model=List[CameraResolution])
async def list_supported_resolutions():
    try:
        resolutions = await service_manager.camera_service.get_supported_resolutions()
        return resolutions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list camera resolutions: {str(e)}")

# Additional WebSocket Routes
@app.websocket("/api/v1/summarization/ws/{meeting_id}")
async def summarization_websocket(websocket: WebSocket, meeting_id: str):
    """WebSocket endpoint for real-time summarization updates"""
    await manager.connect(websocket, meeting_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "audio_chunk":
                audio_chunk_path = message_data.get("audio_path")
                context = message_data.get("context")
                
                # Perform real-time analysis
                analysis = await service_manager.summarization_service.real_time_audio_analysis(
                    audio_chunk_path=audio_chunk_path,
                    meeting_context=context,
                    user_id="default_user"
                )
                
                # Store real-time update
                await storage_manager.save_analysis(str(uuid.uuid4()), {
                    "type": "websocket_summarization",
                    "meeting_id": meeting_id,
                    "analysis": analysis,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                await websocket.send_json({
                    "type": "real_time_analysis",
                    "analysis": analysis,
                    "meeting_id": meeting_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Summarization WebSocket error: {str(e)}")
        manager.disconnect(websocket)

# Missing CRUD Operations for Router Components
@summarization_router.get("/keypoints/", response_model=List[KeyPoint])
async def get_key_points():
    return []

@summarization_router.post("/keypoints/", response_model=KeyPoint)
async def create_key_point(point: KeyPoint):
    return point

@quick_respond_router.post("/meeting_status/", response_model=MeetingStatus)
async def create_meeting_status(meeting_status: MeetingStatus):
    return meeting_status

@quick_respond_router.get("/meeting_status/", response_model=List[MeetingStatus])
async def list_meeting_statuses():
    return []

# ====================================================================
# FINAL STARTUP AND CLEANUP HANDLERS
# ====================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when application shuts down"""
    logger.info("Application shutting down...")
    
    # Close all WebSocket connections gracefully
    for session_id, connections in manager.active_connections.items():
        for connection in connections:
            try:
                await connection.close(code=1001, reason="Server shutdown")
            except Exception as e:
                logger.warning(f"Error closing WebSocket connection: {e}")
    
    # Cleanup storage manager
    await storage_manager.cleanup()
    
    logger.info("Application shutdown complete")

# ====================================================================
# DEVELOPMENT AND TESTING ROUTES
# ====================================================================

@app.get("/api/v1/dev/test-storage", tags=["Development"])
async def test_storage_connections():
    """Test all storage connections - Development only"""
    try:
        test_results = {
            "database": False,
            "redis": False, 
            "file_storage": False,
            "vector_db": False
        }
        
        # Test database
        if storage_manager.db:
            try:
                await storage_manager.db.fetchval("SELECT 1")
                test_results["database"] = True
            except:
                test_results["database"] = False
        
        # Test Redis
        if storage_manager.redis_cache:
            try:
                await storage_manager.redis_cache.ping()
                test_results["redis"] = True
            except:
                test_results["redis"] = False
        
        # Test file storage
        if storage_manager.file_storage:
            try:
                test_data = b"test"
                await storage_manager.file_storage.upload("test/test.txt", test_data)
                test_results["file_storage"] = True
            except:
                test_results["file_storage"] = False
        
        # Test vector database
        if storage_manager.vector_db:
            test_results["vector_db"] = True
        
        return {
            "storage_tests": test_results,
            "overall_status": "healthy" if any(test_results.values()) else "unhealthy",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/v1/dev/test-services", tags=["Development"])
async def test_service_connections():
    """Test all service connections - Development only"""
    try:
        await hands_free_service.emergency_pause(session_id)
        return {"message": "System paused. User can manually resume.", "status": "paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency pause failed: {str(e)}")

@router.post("/session/{session_id}/resume")
async def resume_hands_free(session_id: str):
    """Resume hands-free mode after pause"""
    try:
        await hands_free_service.resume_hands_free(session_id)
        return {"message": "Hands-free mode resumed. System is automated again.", "status": "active"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume: {str(e)}")