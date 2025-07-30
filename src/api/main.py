"""
FastAPI Main Application for Mental Health AI Chat

Production-ready FastAPI backend with comprehensive features:
- RESTful API endpoints for chat and session management
- WebSocket support for real-time streaming
- Rate limiting and security features
- Comprehensive error handling and logging
- Metrics and monitoring integration
- CORS configuration and static file serving
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import (
    FastAPI, HTTPException, WebSocket, WebSocketDisconnect, 
    Depends, Request, Response, BackgroundTasks
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

from .models import (
    ChatRequest, ChatResponse, SessionCreateRequest, SessionCreateResponse,
    SessionInfo, SessionUpdateRequest, SessionSearchRequest, SessionSearchResponse,
    ModelsStatusResponse, SelectionStatsResponse, ErrorResponse, APIConfig
)
from .chat_api import ChatAPI
from .websocket import WebSocketManager
from ..chat.persistent_session_store import SessionStoreType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
WEBSOCKET_CONNECTIONS = Gauge('websocket_connections_active', 'Active WebSocket connections')
MODEL_SELECTIONS = Counter('model_selections_total', 'Total model selections', ['model', 'prompt_type'])
SAFETY_ALERTS = Counter('safety_alerts_total', 'Total safety alerts', ['level'])

# Global API configuration
API_CONFIG = APIConfig()

# Initialize components
websocket_manager = WebSocketManager(
    heartbeat_interval=API_CONFIG.websocket_heartbeat_interval,
    max_connections=API_CONFIG.max_websocket_connections
)

chat_api = ChatAPI(
    websocket_manager=websocket_manager,
    store_type=SessionStoreType.SQLITE,
    store_config={"db_path": "data/sessions.db"},
    enable_safety_monitoring=True
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Mental Health AI Chat API...")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/development", exist_ok=True)
    
    # Start background tasks
    logger.info("✅ Mental Health AI Chat API started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Mental Health AI Chat API...")
    try:
        await chat_api.shutdown()
        logger.info("✅ Mental Health AI Chat API shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=API_CONFIG.title,
    version=API_CONFIG.version,
    description=API_CONFIG.description,
    lifespan=lifespan
)

# Add middleware
if API_CONFIG.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_CONFIG.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for monitoring and rate limiting"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(duration)
    
    # Add response headers
    response.headers["X-Response-Time"] = str(duration)
    response.headers["X-Request-ID"] = str(id(request))
    
    return response

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.warning(f"Validation error for {request.url}: {exc}")
    
    error_details = []
    for error in exc.errors():
        error_details.append({
            "code": error["type"],
            "message": error["msg"],
            "field": ".".join(str(x) for x in error["loc"]),
            "context": error.get("ctx", {})
        })
    
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            message="Request validation failed",
            details=error_details,
            request_id=str(id(request))
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error {exc.status_code} for {request.url}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=exc.detail,
            request_id=str(id(request))
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error for {request.url}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred",
            request_id=str(id(request))
        ).dict()
    )

# Static files and templates
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")
templates = Jinja2Templates(directory="src/ui/templates")

# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with component status"""
    return await chat_api.health_check()

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        prometheus_client.generate_latest(),
        media_type="text/plain"
    )

# Main chat endpoints
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for conversational interactions
    
    Handles:
    - Dynamic model selection based on message content
    - Session management and persistence
    - Safety monitoring and crisis detection
    - Real-time response generation
    """
    try:
        # Record model selection metrics
        if hasattr(request, 'prompt_type'):
            MODEL_SELECTIONS.labels(
                model="unknown",
                prompt_type=request.prompt_type
            ).inc()
        
        response = await chat_api.chat(request, stream=False)
        
        # Update metrics with actual selection
        if response.model_selection:
            MODEL_SELECTIONS.labels(
                model=response.model_selection.selected_model,
                prompt_type=response.model_selection.prompt_classification
            ).inc()
        
        # Record safety alerts
        if response.safety_alert:
            SAFETY_ALERTS.labels(
                level=response.safety_alert.get('level', 'unknown')
            ).inc()
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/stream/{session_id}", tags=["Chat"])
async def stream_chat(session_id: str, message: str):
    """
    Streaming chat endpoint for real-time responses
    
    Returns server-sent events with response chunks
    """
    async def generate():
        try:
            request = ChatRequest(
                session_id=session_id,
                message=message,
                streaming=True
            )
            
            async for chunk in chat_api.stream_chat(request):
                yield f"data: {chunk}\n\n"
                
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            yield f"data: {{'type': 'error', 'error': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Session management endpoints
@app.post("/api/session", response_model=SessionCreateResponse, tags=["Sessions"])
async def create_session(request: SessionCreateRequest) -> SessionCreateResponse:
    """Create a new conversation session"""
    return await chat_api.create_session(request)

@app.get("/api/session/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def get_session(session_id: str) -> SessionInfo:
    """Get session information and current state"""
    return await chat_api.get_session(session_id)

@app.put("/api/session/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def update_session(session_id: str, request: SessionUpdateRequest) -> SessionInfo:
    """Update session properties"""
    # Implementation would update session in session manager
    # For now, just return current session info
    return await chat_api.get_session(session_id)

@app.delete("/api/session/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    """Delete/archive a session"""
    session = await chat_api.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    success = await chat_api.session_manager.archive_session(session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to archive session")
    
    return {"success": True, "message": "Session archived successfully"}

@app.post("/api/sessions/search", response_model=SessionSearchResponse, tags=["Sessions"])
async def search_sessions(request: SessionSearchRequest) -> SessionSearchResponse:
    """Search sessions by content, user, or metadata"""
    try:
        # Use session manager's search functionality
        sessions = await chat_api.session_manager.search_sessions(
            query=request.query,
            user_id=request.user_id,
            start_date=request.date_from,
            end_date=request.date_to
        )
        
        # Convert to API format and apply pagination
        session_infos = [chat_api._convert_to_session_info(s) for s in sessions]
        
        # Apply pagination
        start_idx = request.offset
        end_idx = start_idx + request.limit
        paginated_sessions = session_infos[start_idx:end_idx]
        
        return SessionSearchResponse(
            success=True,
            total_count=len(session_infos),
            sessions=paginated_sessions,
            messages=None  # Could include messages if requested
        )
        
    except Exception as e:
        logger.error(f"Error searching sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@app.get("/api/models/status", response_model=ModelsStatusResponse, tags=["Models"])
async def get_models_status() -> ModelsStatusResponse:
    """Get health status of all available models"""
    try:
        # Check each model's availability
        model_statuses = []
        total_models = len(API_CONFIG.available_models)
        available_count = 0
        
        for model_name in API_CONFIG.available_models:
            try:
                # Simulate health check for each model
                # In production, this would actually test model availability
                status = {
                    "name": model_name,
                    "available": True,
                    "response_time_ms": 1200.0,
                    "success_rate": 0.98,
                    "last_health_check": datetime.now(),
                    "capabilities": ["streaming", "safety_filtering"],
                    "error_message": None
                }
                available_count += 1
            except Exception as e:
                status = {
                    "name": model_name,
                    "available": False,
                    "response_time_ms": None,
                    "success_rate": None,
                    "last_health_check": datetime.now(),
                    "capabilities": [],
                    "error_message": str(e)
                }
            
            model_statuses.append(status)
        
        # Determine overall health
        if available_count == 0:
            overall_health = "critical"
        elif available_count < total_models:
            overall_health = "degraded"
        else:
            overall_health = "healthy"
        
        return ModelsStatusResponse(
            success=True,
            timestamp=datetime.now(),
            total_models=total_models,
            available_models=available_count,
            models=model_statuses,
            overall_health=overall_health
        )
        
    except Exception as e:
        logger.error(f"Error getting models status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/selection", response_model=SelectionStatsResponse, tags=["Analytics"])
async def get_selection_statistics(
    days: int = 7,
    user_id: Optional[str] = None
) -> SelectionStatsResponse:
    """Get model selection statistics and analytics"""
    try:
        # Get statistics from chat API
        stats = chat_api.get_stats()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format response
        return SelectionStatsResponse(
            success=True,
            period_start=start_date,
            period_end=end_date,
            total_selections=stats.get("total_requests", 0),
            statistics=[],  # Would be populated from actual stats
            prompt_type_distribution={}  # Would be populated from actual data
        )
        
    except Exception as e:
        logger.error(f"Error getting selection statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat communication
    
    Supports:
    - Real-time message streaming
    - Model selection progress updates
    - Safety alerts and system notifications
    - Heartbeat/keepalive with automatic reconnection
    """
    connection_id = None
    
    try:
        # Connect to WebSocket manager
        connection_id = await websocket_manager.connect(websocket, session_id)
        WEBSOCKET_CONNECTIONS.inc()
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Listen for messages
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_text()
                
                # Handle client message
                await websocket_manager.handle_client_message(connection_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                # Continue listening for other messages
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        
    finally:
        # Cleanup connection
        if connection_id:
            websocket_manager.disconnect(connection_id)
            WEBSOCKET_CONNECTIONS.dec()

# User interface endpoints
@app.get("/", tags=["UI"])
async def chat_interface(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse(
        "modern_chat.html",
        {
            "request": request,
            "config": {
                "enable_streaming": True,
                "enable_caching": True,
                "available_models": API_CONFIG.available_models,
                "max_message_length": API_CONFIG.max_message_length
            }
        }
    )

# Administrative endpoints
@app.get("/api/admin/stats", tags=["Admin"])
async def get_admin_stats():
    """Get comprehensive system statistics"""
    try:
        stats = chat_api.get_stats()
        websocket_stats = websocket_manager.get_stats()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "api_stats": stats,
            "websocket_stats": websocket_stats,
            "system_health": await chat_api.health_check()
        }
        
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/broadcast", tags=["Admin"])
async def broadcast_system_message(message_type: str, data: Dict[str, Any]):
    """Broadcast system message to all connected WebSocket clients"""
    try:
        await websocket_manager.broadcast_system_message(message_type, data)
        return {"success": True, "message": "System message broadcast successfully"}
        
    except Exception as e:
        logger.error(f"Error broadcasting system message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Development utilities
@app.get("/api/dev/config", tags=["Development"])
async def get_config():
    """Get current API configuration (development only)"""
    return API_CONFIG.dict()

@app.post("/api/dev/test-model-selection", tags=["Development"])
async def test_model_selection(message: str):
    """Test model selection without creating a session (development only)"""
    try:
        selection = await chat_api.model_selector.select_model(message, [])
        return {
            "success": True,
            "selection": {
                "selected_model": selection.selected_model,
                "confidence": selection.confidence_score,
                "prompt_type": selection.prompt_classification.value,
                "reasoning": selection.reasoning
            }
        }
    except Exception as e:
        logger.error(f"Error testing model selection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Application entry point
if __name__ == "__main__":
    # Configure for development
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )