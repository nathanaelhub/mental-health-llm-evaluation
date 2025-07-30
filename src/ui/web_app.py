"""
Web Application for Dynamic Model Selection Chat

FastAPI-based web interface for the mental health chatbot with
real-time streaming and intelligent model selection.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, WebSocket, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from ..chat.chat_interface import ChatInterface, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


# Pydantic models for API
class ChatMessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    use_cache: bool = True


class ChatMessageResponse(BaseModel):
    message: str
    session_id: str
    model_used: str
    response_time_ms: float
    is_new_session: bool
    selection_info: Optional[Dict[str, Any]] = None
    cached: bool = False
    error: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    last_activity: str
    selected_model: str
    turn_count: int
    initial_prompt: str


class SystemStatus(BaseModel):
    available_models: List[str]
    model_health: Dict[str, bool]
    session_analytics: Dict[str, Any]
    cache_stats: Dict[str, Any]
    features: Dict[str, bool]


def create_app(
    available_models: List[str] = None,
    enable_caching: bool = True,
    enable_streaming: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000
) -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Args:
        available_models: List of models to use
        enable_caching: Whether to enable response caching
        enable_streaming: Whether to support streaming responses
        host: Host to bind to
        port: Port to bind to
        
    Returns:
        Configured FastAPI application
    """
    
    # Initialize FastAPI app
    app = FastAPI(
        title="Mental Health LLM Chat",
        description="Dynamic model selection chatbot for mental health support",
        version="1.0.0"
    )
    
    # Initialize chat interface
    chat_interface = ChatInterface(
        available_models=available_models,
        enable_caching=enable_caching,
        enable_streaming=enable_streaming
    )
    
    # Configure templates and static files
    templates = Jinja2Templates(directory="src/ui/templates")
    app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")
    
    # Store app configuration
    app.state.chat_interface = chat_interface
    app.state.config = {
        'available_models': available_models or ['openai', 'deepseek', 'claude', 'gemma'],
        'enable_caching': enable_caching,
        'enable_streaming': enable_streaming,
        'host': host,
        'port': port
    }
    
    # Routes
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Serve the main chat interface"""
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "config": app.state.config
            }
        )
    
    @app.post("/api/chat", response_model=ChatMessageResponse)
    async def chat_message(request: ChatMessageRequest) -> ChatMessageResponse:
        """
        Send a chat message and get response
        
        This endpoint handles both new conversations (with model selection)
        and continuing existing conversations.
        """
        try:
            chat_request = ChatRequest(
                message=request.message,
                user_id=request.user_id,
                session_id=request.session_id,
                system_prompt=request.system_prompt,
                use_cache=request.use_cache,
                enable_streaming=False
            )
            
            response = await chat_interface.chat(chat_request)
            
            return ChatMessageResponse(
                message=response.message,
                session_id=response.session_id,
                model_used=response.model_used,
                response_time_ms=response.response_time_ms,
                is_new_session=response.is_new_session,
                selection_info=response.selection_info,
                cached=response.cached,
                error=response.error
            )
            
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.websocket("/api/chat/stream")
    async def chat_stream(websocket: WebSocket):
        """
        WebSocket endpoint for streaming chat responses
        
        Message format:
        {
            "message": "user message",
            "user_id": "optional_user_id",
            "session_id": "optional_session_id",
            "system_prompt": "optional_system_prompt"
        }
        """
        await websocket.accept()
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                
                chat_request = ChatRequest(
                    message=data.get("message", ""),
                    user_id=data.get("user_id"),
                    session_id=data.get("session_id"),
                    system_prompt=data.get("system_prompt"),
                    use_cache=data.get("use_cache", True),
                    enable_streaming=True
                )
                
                # Send initial response with session info
                await websocket.send_json({
                    "type": "start",
                    "session_id": chat_request.session_id,
                    "is_new_session": not bool(chat_request.session_id)
                })
                
                # Stream the response
                full_response = ""
                async for chunk in chat_interface.stream_chat(chat_request):
                    full_response += chunk
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk
                    })
                
                # Send completion message
                await websocket.send_json({
                    "type": "complete",
                    "full_response": full_response
                })
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        finally:
            try:
                await websocket.close()
            except:
                pass
    
    @app.get("/api/sessions/{user_id}", response_model=List[SessionInfo])
    async def get_user_sessions(user_id: str) -> List[SessionInfo]:
        """Get all sessions for a user"""
        try:
            sessions_data = chat_interface.get_user_sessions(user_id)
            
            return [
                SessionInfo(
                    session_id=session['session_id'],
                    created_at=session['created_at'],
                    last_activity=session['last_activity'],
                    selected_model=session['selected_model'],
                    turn_count=session['turn_count'],
                    initial_prompt=session['initial_prompt']
                )
                for session in sessions_data
            ]
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/sessions/{session_id}/info")
    async def get_session_info(session_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific session"""
        try:
            session_info = await chat_interface.get_session_info(session_id)
            
            if not session_info:
                raise HTTPException(status_code=404, detail="Session not found")
            
            return session_info
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> Dict[str, str]:
        """Delete a conversation session"""
        try:
            success = chat_interface.delete_session(session_id)
            
            if not success:
                raise HTTPException(status_code=404, detail="Session not found")
            
            return {"message": "Session deleted successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/sessions/{session_id}/switch-model")
    async def switch_model(session_id: str, model_name: str) -> Dict[str, str]:
        """Switch to a different model mid-conversation"""
        try:
            success = await chat_interface.switch_model(session_id, model_name)
            
            if not success:
                raise HTTPException(status_code=400, detail="Failed to switch model")
            
            return {"message": f"Switched to {model_name} successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/status", response_model=SystemStatus)
    async def get_system_status() -> SystemStatus:
        """Get system status and health information"""
        try:
            status = await chat_interface.get_system_status()
            
            return SystemStatus(
                available_models=status['available_models'],
                model_health=status['model_health'],
                session_analytics=status['session_analytics'],
                cache_stats=status['cache_stats'],
                features=status['features']
            )
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/cleanup")
    async def cleanup_system(background_tasks: BackgroundTasks) -> Dict[str, str]:
        """Clean up expired sessions and cache entries"""
        try:
            background_tasks.add_task(chat_interface.cleanup)
            return {"message": "Cleanup initiated"}
            
        except Exception as e:
            logger.error(f"Error initiating cleanup: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/health")
    async def health_check() -> Dict[str, str]:
        """Simple health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "mental-health-chat"
        }
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Mental Health Chat API starting up")
        logger.info(f"Available models: {app.state.config['available_models']}")
        logger.info(f"Caching enabled: {app.state.config['enable_caching']}")
        logger.info(f"Streaming enabled: {app.state.config['enable_streaming']}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Mental Health Chat API shutting down")
        # Perform cleanup
        try:
            chat_interface.cleanup()
        except Exception as e:
            logger.error(f"Error during shutdown cleanup: {e}")
    
    return app


def run_app(
    available_models: List[str] = None,
    enable_caching: bool = True,
    enable_streaming: bool = True,
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False
):
    """
    Run the web application
    
    Args:
        available_models: List of models to use
        enable_caching: Whether to enable response caching
        enable_streaming: Whether to support streaming responses
        host: Host to bind to
        port: Port to bind to
        debug: Whether to run in debug mode
    """
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not debug else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create app
    app = create_app(
        available_models=available_models,
        enable_caching=enable_caching,
        enable_streaming=enable_streaming,
        host=host,
        port=port
    )
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info" if not debug else "debug",
        reload=debug
    )


if __name__ == "__main__":
    run_app(debug=True)