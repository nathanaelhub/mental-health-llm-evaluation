"""
ChatAPI - Main orchestration layer for the mental health chat system

Coordinates model selection, session management, safety monitoring,
and response generation with comprehensive error handling and monitoring.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import HTTPException
from pydantic import ValidationError

from .models import (
    ChatRequest, ChatResponse, SessionCreateRequest, SessionCreateResponse,
    SessionInfo, MessageResponse, ModelSelectionInfo, SafetyAlert,
    MessageRole, SessionStatus, SafetyLevel, PromptType, validate_message_content
)
from .websocket import WebSocketManager, StreamingChunk, ModelSelectionUpdate
from ..chat.dynamic_model_selector import DynamicModelSelector, ModelSelection
from ..chat.conversation_session_manager import ConversationSessionManager, Message
from ..chat.persistent_session_store import SessionStoreType
from ..evaluation.evaluation_metrics import TherapeuticEvaluator

logger = logging.getLogger(__name__)


class ChatAPI:
    """
    Main ChatAPI orchestration class
    
    Coordinates all aspects of the mental health chat system including:
    - Dynamic model selection with real-time updates
    - Session lifecycle management with persistence
    - Safety monitoring and crisis detection
    - Real-time WebSocket streaming
    - Performance monitoring and analytics
    """
    
    def __init__(self, 
                 websocket_manager: WebSocketManager,
                 store_type: SessionStoreType = SessionStoreType.SQLITE,
                 store_config: Optional[Dict[str, Any]] = None,
                 enable_safety_monitoring: bool = True):
        """
        Initialize ChatAPI with all required components
        
        Args:
            websocket_manager: WebSocket manager for real-time communication
            store_type: Type of session store to use
            store_config: Configuration for the session store
            enable_safety_monitoring: Whether to enable safety monitoring
        """
        self.websocket_manager = websocket_manager
        
        # Initialize core components with all 4 models
        models_config = {
            'models': {
                'openai': {'enabled': True, 'cost_per_token': 0.0001, 'model_name': 'gpt-4'},
                'claude': {'enabled': True, 'cost_per_token': 0.00015, 'model_name': 'claude-3'},
                'deepseek': {'enabled': True, 'cost_per_token': 0.00005, 'model_name': 'deepseek/deepseek-r1-0528-qwen3-8b'},
                'gemma': {'enabled': True, 'cost_per_token': 0.00003, 'model_name': 'google/gemma-3-12b'}
            },
            'default_model': 'openai',
            'selection_timeout': 10.0,  # Increased timeout
            'similarity_threshold': 0.9
        }
        self.model_selector = DynamicModelSelector(models_config)
        self.session_manager = ConversationSessionManager(
            store_type=store_type,
            store_config=store_config,
            enable_safety_monitoring=enable_safety_monitoring,
            enable_audit_trail=True
        )
        
        # Safety and evaluation
        self.evaluator = TherapeuticEvaluator() if enable_safety_monitoring else None
        self.enable_safety_monitoring = enable_safety_monitoring
        
        # Performance tracking
        self.request_counts: Dict[str, int] = {}
        self.response_times: List[float] = []
        self.model_usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_minute = 60
        
        logger.info("ChatAPI initialized successfully")
    
    async def chat(self, request: ChatRequest, stream: bool = False) -> ChatResponse:
        """
        Main chat endpoint - orchestrates the entire conversation flow
        
        Args:
            request: Chat request containing message and session info
            stream: Whether to stream the response via WebSocket
            
        Returns:
            ChatResponse with the assistant's reply and metadata
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Input validation and sanitization
            message_content = validate_message_content(request.message)
            
            # Rate limiting check
            if self._is_rate_limited(request.session_id or "anonymous"):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please wait before sending another message."
                )
            
            # Get or create session
            session = await self._get_or_create_session(request)
            
            # Add user message to session
            user_message = await self.session_manager.add_message(
                session.session_id,
                MessageRole.USER,
                message_content
            )
            
            if not user_message:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to add user message to session"
                )
            
            # Perform model selection with progress updates
            model_selection = await self._select_model_with_updates(
                session, message_content, stream
            )
            
            # Generate response using selected model
            assistant_message = await self._generate_response(
                session, model_selection, stream
            )
            
            # Safety monitoring
            safety_alert = None
            if self.enable_safety_monitoring and assistant_message:
                safety_alert = await self._check_safety(session, assistant_message)
            
            # Update performance metrics
            response_time = (time.time() - start_time) * 1000
            self._update_metrics(model_selection.selected_model, response_time)
            
            # Create response
            response = ChatResponse(
                success=True,
                session_id=session.session_id,
                message=self._convert_to_message_response(assistant_message),
                model_selection=self._convert_to_model_selection_info(
                    model_selection, response_time
                ),
                session_updated=True,
                safety_alert=safety_alert.dict() if safety_alert else None
            )
            
            logger.info(
                f"Chat request completed in {response_time:.2f}ms using {model_selection.selected_model} "
                f"for session {session.session_id}"
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in chat endpoint (request_id: {request_id}): {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error processing chat request: {str(e)}"
            )
    
    async def create_session(self, request: SessionCreateRequest) -> SessionCreateResponse:
        """
        Create a new conversation session
        
        Args:
            request: Session creation request
            
        Returns:
            SessionCreateResponse with new session info
        """
        try:
            # Create session using session manager
            session = await self.session_manager.create_session(
                user_id=request.user_id,
                selected_model="intelligent-selection",
                initial_message=request.initial_message or "Hello, I'm looking for support.",
                metadata=request.metadata
            )
            
            # Convert to API response format
            session_info = self._convert_to_session_info(session)
            
            response = SessionCreateResponse(
                success=True,
                session_id=session.session_id,
                session_info=session_info
            )
            
            logger.info(f"Created new session {session.session_id} for user {request.user_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating session: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create session: {str(e)}"
            )
    
    async def get_session(self, session_id: str) -> SessionInfo:
        """
        Get session information and history
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionInfo with current session state
        """
        try:
            session = await self.session_manager.get_session(session_id)
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found"
                )
            
            return self._convert_to_session_info(session)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve session: {str(e)}"
            )
    
    async def stream_chat(self, request: ChatRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response in real-time chunks
        
        Args:
            request: Chat request
            
        Yields:
            Streaming response chunks
        """
        try:
            # Start chat processing
            session = await self._get_or_create_session(request)
            
            # Add user message
            user_message = await self.session_manager.add_message(
                session.session_id,
                MessageRole.USER,
                validate_message_content(request.message)
            )
            
            # Select model with streaming updates
            model_selection = await self._select_model_with_updates(
                session, request.message, stream=True
            )
            
            # Stream response generation
            async for chunk in self._stream_response_generation(session, model_selection):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_or_create_session(self, request: ChatRequest):
        """Get existing session or create new one"""
        if request.session_id:
            session = await self.session_manager.get_session(request.session_id)
            if session:
                return session
        
        # Create new session
        session = await self.session_manager.create_session(
            user_id=request.user_id,
            selected_model="intelligent-selection",
            initial_message=request.message,
            metadata=request.metadata
        )
        
        return session
    
    async def _select_model_with_updates(self, session, message: str, stream: bool = False):
        """Select model with real-time progress updates"""
        if stream:
            # Send selection progress updates via WebSocket
            await self.websocket_manager.send_model_selection_update(
                session.session_id,
                ModelSelectionUpdate(
                    stage="analyzing",
                    progress=0.1,
                    current_model=None
                )
            )
        
        # Perform model selection
        start_time = time.time()
        model_selection = await self.model_selector.select_model(
            message, 
            session.conversation_history[-5:]  # Recent context
        )
        
        if stream:
            # Send final selection update
            await self.websocket_manager.send_model_selection_update(
                session.session_id,
                ModelSelectionUpdate(
                    stage="complete",
                    progress=1.0,
                    current_model=model_selection.selected_model,
                    preliminary_results={
                        "selected_model": model_selection.selected_model,
                        "confidence": model_selection.confidence_score,
                        "prompt_type": model_selection.prompt_classification.value
                    }
                )
            )
        
        return model_selection
    
    async def _generate_response(self, session, model_selection: ModelSelection, stream: bool = False):
        """Generate assistant response using selected model"""
        # This would integrate with your actual model inference
        # For now, we'll simulate the response generation
        
        response_content = await self._simulate_model_response(
            model_selection.selected_model,
            session.conversation_history[-1].content,
            model_selection.prompt_classification
        )
        
        # Add assistant message to session
        assistant_message = await self.session_manager.add_message(
            session.session_id,
            MessageRole.ASSISTANT,
            response_content,
            model_used=model_selection.selected_model
        )
        
        return assistant_message
    
    async def _simulate_model_response(self, model: str, user_message: str, prompt_type: PromptType) -> str:
        """Simulate model response generation (replace with actual model calls)"""
        # This is a placeholder - in production, this would call your actual models
        
        responses = {
            PromptType.ANXIETY: [
                "I understand you're feeling anxious. That's a completely valid response to stress. Let's explore some techniques that might help you manage these feelings.",
                "Anxiety can feel overwhelming, but there are proven strategies we can work through together. Would you like to start with some breathing exercises?",
                "It sounds like you're experiencing anxiety. This is very common and treatable. Let's focus on some immediate coping strategies."
            ],
            PromptType.DEPRESSION: [
                "I hear that you're going through a difficult time. Depression can make everything feel harder, but you've taken an important step by reaching out.",
                "Thank you for sharing this with me. Depression affects many people, and it's important to know that support and effective treatments are available.",
                "It takes courage to talk about depression. I'm here to listen and help you explore ways to feel better."
            ],
            PromptType.CRISIS: [
                "I'm very concerned about what you've shared. Your safety is the most important thing right now. Are there trusted people in your life you can reach out to?",
                "Thank you for trusting me with this. I want you to know that help is available. Would you like information about crisis support resources?",
                "I'm glad you're talking about this rather than keeping it to yourself. Let's focus on keeping you safe and connecting you with professional support."
            ]
        }
        
        # Simulate processing delay
        await asyncio.sleep(0.5)
        
        # Select appropriate response based on prompt type
        default_responses = [
            "I understand what you're sharing, and I want you to know that seeking support is a positive step.",
            "Thank you for opening up about this. How are you feeling right now in this moment?",
            "I hear you, and I want to help. Can you tell me more about what's been on your mind lately?"
        ]
        
        response_list = responses.get(prompt_type, default_responses)
        return response_list[0]  # In production, you'd select based on context
    
    async def _stream_response_generation(self, session, model_selection: ModelSelection):
        """Stream response generation with real-time chunks"""
        response_content = await self._simulate_model_response(
            model_selection.selected_model,
            session.conversation_history[-1].content,
            model_selection.prompt_classification
        )
        
        # Simulate streaming by breaking response into chunks
        words = response_content.split()
        chunk_size = 3  # words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            
            chunk = StreamingChunk(
                chunk_id=i // chunk_size,
                content=chunk_content + " ",
                is_final=(i + chunk_size >= len(words)),
                model_used=model_selection.selected_model
            )
            
            # Send via WebSocket
            await self.websocket_manager.send_streaming_chunk(session.session_id, chunk)
            
            # Yield for HTTP streaming
            yield {
                "type": "chunk",
                "data": chunk.dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Simulate delay between chunks
            await asyncio.sleep(0.1)
        
        # Add complete message to session
        await self.session_manager.add_message(
            session.session_id,
            MessageRole.ASSISTANT,
            response_content,
            model_used=model_selection.selected_model
        )
        
        # Send completion notification
        yield {
            "type": "complete",
            "session_id": session.session_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _check_safety(self, session, message: Message) -> Optional[SafetyAlert]:
        """Check message safety and create alerts if needed"""
        if not self.evaluator or not message:
            return None
        
        try:
            # Use existing safety evaluation from message
            safety_score = message.safety_score
            if safety_score is None:
                return None
            
            # Determine safety level and create alert if needed
            if safety_score < 0.3:  # Critical safety threshold
                alert = SafetyAlert(
                    level=SafetyLevel.CRITICAL,
                    message="Critical safety concern detected. Immediate professional support recommended.",
                    recommended_actions=[
                        "Contact emergency services if in immediate danger",
                        "Reach out to a crisis helpline",
                        "Contact a trusted friend or family member",
                        "Seek professional mental health support"
                    ],
                    resources=[
                        {"name": "National Suicide Prevention Lifeline", "contact": "988"},
                        {"name": "Crisis Text Line", "contact": "Text HOME to 741741"},
                        {"name": "Emergency Services", "contact": "911"}
                    ]
                )
                
                # Send alert via WebSocket
                await self.websocket_manager.send_safety_alert(session.session_id, alert)
                
                return alert
            
            elif safety_score < 0.5:  # Warning level
                alert = SafetyAlert(
                    level=SafetyLevel.WARNING,
                    message="Concerning content detected. Consider professional support resources.",
                    recommended_actions=[
                        "Consider talking to a mental health professional",
                        "Reach out to trusted friends or family",
                        "Practice self-care and safety planning"
                    ]
                )
                
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            return None
    
    def _is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited"""
        now = time.time()
        window_start = now - self.rate_limit_window
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Remove old requests
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier]
            if req_time > window_start
        ]
        
        # Check if over limit
        if len(self.rate_limits[identifier]) >= self.max_requests_per_minute:
            return True
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return False
    
    def _update_metrics(self, model: str, response_time: float):
        """Update performance metrics"""
        self.response_times.append(response_time)
        
        # Keep only recent response times (last 1000)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        # Update model usage stats
        if model not in self.model_usage_stats:
            self.model_usage_stats[model] = {
                "usage_count": 0,
                "total_response_time": 0,
                "avg_response_time": 0
            }
        
        stats = self.model_usage_stats[model]
        stats["usage_count"] += 1
        stats["total_response_time"] += response_time
        stats["avg_response_time"] = stats["total_response_time"] / stats["usage_count"]
    
    def _convert_to_message_response(self, message: Message) -> MessageResponse:
        """Convert internal Message to API MessageResponse"""
        return MessageResponse(
            message_id=message.message_id,
            role=MessageRole(message.role.value),
            content=message.content,
            timestamp=message.timestamp,
            model_used=message.model_used,
            token_count=message.token_count,
            safety_score=message.safety_score,
            response_time_ms=None,  # Could be calculated separately
            metadata=message.metadata
        )
    
    def _convert_to_session_info(self, session) -> SessionInfo:
        """Convert internal session to API SessionInfo"""
        return SessionInfo(
            session_id=session.session_id,
            user_id=session.user_id,
            status=SessionStatus(session.status.value),
            safety_level=SafetyLevel(session.safety_level.value),
            selected_model=session.selected_model,
            created_at=session.created_at,
            last_activity=session.last_activity,
            message_count=len(session.conversation_history),
            total_tokens=session.total_tokens,
            model_switches=len(session.model_switches),
            crisis_flags=len(session.crisis_flags),
            metadata=session.metadata
        )
    
    def _convert_to_model_selection_info(self, selection: ModelSelection, response_time: float) -> ModelSelectionInfo:
        """Convert internal ModelSelection to API ModelSelectionInfo"""
        return ModelSelectionInfo(
            selected_model=selection.selected_model,
            selection_reason=selection.reasoning,
            prompt_classification=PromptType(selection.prompt_classification.value),
            confidence_score=selection.confidence_score,
            alternatives=[
                {
                    "model": alt.model,
                    "score": alt.score,
                    "reason": alt.reasoning
                }
                for alt in selection.alternatives[:3]  # Top 3 alternatives
            ],
            selection_time_ms=response_time
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API performance statistics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "total_requests": len(self.response_times),
            "avg_response_time_ms": round(avg_response_time, 2),
            "model_usage": self.model_usage_stats,
            "websocket_stats": self.websocket_manager.get_stats(),
            "active_rate_limits": len(self.rate_limits)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            # Check model selector
            model_health = await self._check_model_health()
            
            # Check session manager
            session_health = await self._check_session_manager_health()
            
            # Check WebSocket manager
            websocket_health = self.websocket_manager.get_stats()
            
            overall_health = "healthy"
            if not model_health["available"] or not session_health["available"]:
                overall_health = "critical"
            elif model_health["degraded"] or session_health["degraded"]:
                overall_health = "degraded"
            
            return {
                "status": overall_health,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "models": model_health,
                    "sessions": session_health,
                    "websockets": websocket_health
                },
                "stats": self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "critical",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _check_model_health(self) -> Dict[str, Any]:
        """Check health of model selector and models"""
        try:
            # Test model selection with a simple message
            test_message = "How are you today?"
            start_time = time.time()
            
            selection = await self.model_selector.select_model(test_message, [])
            response_time = (time.time() - start_time) * 1000
            
            return {
                "available": True,
                "degraded": response_time > 5000,  # Over 5 seconds is degraded
                "response_time_ms": response_time,
                "selected_model": selection.selected_model,
                "confidence": selection.confidence_score
            }
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return {
                "available": False,
                "degraded": True,
                "error": str(e)
            }
    
    async def _check_session_manager_health(self) -> Dict[str, Any]:
        """Check health of session manager"""
        try:
            # Test session creation and retrieval
            test_session = await self.session_manager.create_session(
                user_id="health_check",
                selected_model="test",
                initial_message="Health check test"
            )
            
            retrieved_session = await self.session_manager.get_session(test_session.session_id)
            
            # Cleanup test session
            await self.session_manager.archive_session(test_session.session_id)
            
            return {
                "available": True,
                "degraded": False,
                "test_session_created": retrieved_session is not None
            }
        except Exception as e:
            logger.error(f"Session manager health check failed: {e}")
            return {
                "available": False,
                "degraded": True,
                "error": str(e)
            }
    
    async def shutdown(self):
        """Gracefully shutdown ChatAPI"""
        logger.info("Shutting down ChatAPI...")
        
        try:
            # Close session manager
            await self.session_manager.close()
            
            # Shutdown WebSocket manager
            await self.websocket_manager.shutdown()
            
            logger.info("ChatAPI shutdown complete")
        except Exception as e:
            logger.error(f"Error during ChatAPI shutdown: {e}")
            raise