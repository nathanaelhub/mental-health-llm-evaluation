"""
Chat Interface - Main Orchestration Layer

Coordinates all components of the dynamic model selection chat system
to provide a unified interface for mental health conversations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass

from .model_selector import ModelSelector, ModelSelectionResult
from .dynamic_model_selector import DynamicModelSelector, ModelSelection
from .session_manager import SessionManager, ChatSession
from .conversation_handler import ConversationHandler, ConversationResponse
from .response_cache import ResponseCache

logger = logging.getLogger(__name__)


@dataclass
class ChatRequest:
    """Request for chat interaction"""
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    use_cache: bool = True
    enable_streaming: bool = False


@dataclass
class ChatResponse:
    """Complete chat response with metadata"""
    message: str
    session_id: str
    model_used: str
    response_time_ms: float
    is_new_session: bool
    selection_info: Optional[Dict[str, Any]] = None
    cached: bool = False
    error: Optional[str] = None


class ChatInterface:
    """
    Main interface for the dynamic model selection chat system
    
    This class orchestrates all components to provide:
    - Intelligent model selection for new conversations
    - Efficient conversation continuation
    - Response caching for performance
    - Session management and persistence
    - Real-time streaming support
    """
    
    def __init__(self,
                 available_models: List[str] = None,
                 enable_caching: bool = True,
                 enable_streaming: bool = True,
                 system_prompt: Optional[str] = None):
        """
        Initialize the chat interface
        
        Args:
            available_models: List of models to use
            enable_caching: Whether to enable response caching
            enable_streaming: Whether to support streaming responses
            system_prompt: Default system prompt for mental health context
        """
        self.available_models = available_models or ['openai', 'deepseek', 'claude', 'gemma']
        self.enable_caching = enable_caching
        self.enable_streaming = enable_streaming
        
        # Default mental health system prompt
        self.default_system_prompt = system_prompt or """You are a compassionate and professional mental health support assistant. 

Your role is to:
- Provide empathetic, supportive responses
- Offer evidence-based coping strategies and techniques
- Maintain appropriate professional boundaries
- Recognize when to suggest professional help
- Never provide medical diagnoses or replace professional therapy

Guidelines:
- Listen actively and validate emotions
- Ask clarifying questions when appropriate
- Provide practical, actionable advice
- Be culturally sensitive and inclusive
- Maintain confidentiality and privacy
- Use clear, accessible language

Remember: You are here to support, not replace professional mental health services."""
        
        # Initialize components - use new dynamic selector
        models_config = {
            'models': {model: {} for model in self.available_models},
            'default_model': 'openai',
            'selection_timeout': 5.0,
            'similarity_threshold': 0.9
        }
        
        self.model_selector = DynamicModelSelector(
            models_config=models_config,
            evaluation_framework=None  # Will use default TherapeuticEvaluator
        )
        
        # Keep legacy selector for backwards compatibility if needed
        self.legacy_selector = ModelSelector(
            available_models=self.available_models,
            fallback_model='openai'
        )
        
        self.session_manager = SessionManager(
            session_storage_dir="temp/chat_sessions",
            session_timeout_hours=24
        )
        
        self.conversation_handler = ConversationHandler(
            session_manager=self.session_manager,
            model_selector=self.legacy_selector,  # Use legacy for conversation handler
            enable_streaming=self.enable_streaming
        )
        
        if self.enable_caching:
            self.response_cache = ResponseCache(
                cache_dir="temp/response_cache",
                max_entries=1000,
                ttl_hours=24
            )
        else:
            self.response_cache = None
        
        logger.info("ChatInterface initialized with dynamic model selection")
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Main chat method - handles both new and continuing conversations
        
        Args:
            request: Chat request with message and metadata
            
        Returns:
            ChatResponse with the assistant's reply and metadata
        """
        logger.info(f"Processing chat request for user {request.user_id}")
        
        # Check for cached response first
        if self.enable_caching and request.use_cache and self.response_cache:
            cached_response = self.response_cache.get_cached_response(
                prompt=request.message,
                system_prompt=request.system_prompt or self.default_system_prompt
            )
            
            if cached_response:
                logger.info("Returning cached response")
                return ChatResponse(
                    message=cached_response,
                    session_id=request.session_id or "cached",
                    model_used="cached",
                    response_time_ms=0,
                    is_new_session=False,
                    cached=True
                )
        
        # Handle existing session
        if request.session_id:
            return await self._continue_conversation(request)
        
        # Start new conversation
        return await self._start_new_conversation(request)
    
    async def stream_chat(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """
        Stream chat response in real-time
        
        Args:
            request: Chat request with streaming enabled
            
        Yields:
            Response chunks as they're generated
        """
        if not self.enable_streaming:
            response = await self.chat(request)
            yield response.message
            return
        
        # For new conversations, we need to do model selection first
        if not request.session_id:
            # Do model selection
            selection_result = await self._perform_model_selection(
                request.message,
                request.system_prompt or self.default_system_prompt,
                None
            )
            
            # Create session
            session = self.session_manager.create_session(
                user_id=request.user_id,
                initial_prompt=request.message,
                selection_result=selection_result
            )
            
            request.session_id = session.session_id
            
            # Yield selection info
            yield f"[Model selected: {selection_result.selected_model}]\n\n"
        
        # Stream the conversation
        async for chunk in self.conversation_handler.stream_conversation(
            session_id=request.session_id,
            user_message=request.message,
            system_prompt=request.system_prompt or self.default_system_prompt
        ):
            yield chunk
    
    async def _start_new_conversation(self, request: ChatRequest) -> ChatResponse:
        """Start a new conversation with model selection"""
        logger.info("Starting new conversation with dynamic model selection")
        
        # Check for cached selection
        cached_selection = None
        if self.enable_caching and request.use_cache and self.response_cache:
            cached_selection = self.response_cache.get_cached_selection(
                prompt=request.message,
                system_prompt=request.system_prompt or self.default_system_prompt
            )
        
        # Perform model selection if not cached
        if cached_selection:
            logger.info("Using cached model selection")
            # Convert ModelSelection to ModelSelectionResult for compatibility
            selection_result = ModelSelectionResult(
                selected_model=cached_selection.selected_model_id,
                selection_score=cached_selection.confidence_score * 10,  # Scale to 0-10
                selection_time_ms=cached_selection.latency_metrics.get('total_time_ms', 0),
                all_scores=cached_selection.model_scores,
                response_preview=cached_selection.response_content[:100],
                timestamp=cached_selection.timestamp
            )
        else:
            # Use new dynamic selector
            dynamic_selection = await self.model_selector.select_best_model(
                prompt=request.message,
                context=request.system_prompt or self.default_system_prompt
            )
            
            # Convert to legacy format for compatibility
            selection_result = ModelSelectionResult(
                selected_model=dynamic_selection.selected_model_id,
                selection_score=dynamic_selection.confidence_score * 10,
                selection_time_ms=dynamic_selection.latency_metrics.get('total_time_ms', 0),
                all_scores=dynamic_selection.model_scores,
                response_preview=dynamic_selection.response_content[:100] if dynamic_selection.response_content else "Model selection completed",
                timestamp=dynamic_selection.timestamp
            )
            
            # Cache the selection if enabled
            if self.enable_caching and self.response_cache:
                self.response_cache.cache_selection(
                    prompt=request.message,
                    system_prompt=request.system_prompt or self.default_system_prompt,
                    selection_result=selection_result
                )
        
        # Create new session
        session = self.session_manager.create_session(
            user_id=request.user_id,
            initial_prompt=request.message,
            selection_result=selection_result
        )
        
        # Generate first response
        response = await self.conversation_handler.continue_conversation(
            session_id=session.session_id,
            user_message=request.message,
            system_prompt=request.system_prompt or self.default_system_prompt
        )
        
        # Cache the response if enabled
        if self.enable_caching and self.response_cache and not response.error:
            self.response_cache.cache_response(
                prompt=request.message,
                system_prompt=request.system_prompt or self.default_system_prompt,
                model_name=response.model_used,
                response_text=response.message,
                selection_result=selection_result
            )
        
        return ChatResponse(
            message=response.message,
            session_id=session.session_id,
            model_used=response.model_used,
            response_time_ms=response.response_time_ms,
            is_new_session=True,
            selection_info={
                'selected_model': selection_result.selected_model,
                'selection_score': selection_result.selection_score,
                'all_scores': selection_result.all_scores,
                'selection_time_ms': selection_result.selection_time_ms
            },
            error=response.error
        )
    
    async def _continue_conversation(self, request: ChatRequest) -> ChatResponse:
        """Continue an existing conversation"""
        logger.info(f"Continuing conversation in session {request.session_id}")
        
        response = await self.conversation_handler.continue_conversation(
            session_id=request.session_id,
            user_message=request.message,
            system_prompt=request.system_prompt or self.default_system_prompt
        )
        
        # Cache the response if enabled
        if self.enable_caching and self.response_cache and not response.error:
            self.response_cache.cache_response(
                prompt=request.message,
                system_prompt=request.system_prompt or self.default_system_prompt,
                model_name=response.model_used,
                response_text=response.message
            )
        
        return ChatResponse(
            message=response.message,
            session_id=response.session_id,
            model_used=response.model_used,
            response_time_ms=response.response_time_ms,
            is_new_session=False,
            error=response.error
        )
    
    async def _perform_model_selection(self,
                                     message: str,
                                     system_prompt: str,
                                     conversation_history: Optional[List[Dict[str, str]]]) -> ModelSelectionResult:
        """Perform model selection for a new conversation"""
        
        return await self.model_selector.select_best_model(
            user_prompt=message,
            system_prompt=system_prompt,
            conversation_history=conversation_history
        )
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a session"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None
        
        # Get conversation summary
        conversation_summary = await self.conversation_handler.get_conversation_summary(session_id)
        
        return {
            'session': session.to_dict(),
            'conversation_summary': conversation_summary
        }
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        sessions = self.session_manager.get_user_sessions(user_id)
        
        return [
            {
                'session_id': session.session_id,
                'created_at': session.created_at.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'selected_model': session.selected_model,
                'turn_count': len(session.conversation_history),
                'initial_prompt': session.session_metadata.get('initial_prompt', '')[:100]
            }
            for session in sessions
        ]
    
    async def switch_model(self, session_id: str, new_model: str) -> bool:
        """Switch to a different model mid-conversation"""
        return await self.conversation_handler.switch_model(session_id, new_model)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session"""
        return self.session_manager.delete_session(session_id)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health checks"""
        # Check model health using legacy selector
        model_health = await self.legacy_selector.get_model_health_status()
        
        # Get session analytics
        session_analytics = self.session_manager.get_session_analytics()
        
        # Get cache statistics
        cache_stats = {}
        if self.response_cache:
            cache_stats = self.response_cache.get_cache_stats()
        
        # Get dynamic selector analytics
        selector_analytics = self.model_selector.get_analytics()
        
        return {
            'available_models': self.model_selector.get_available_models(),
            'model_health': model_health,
            'session_analytics': session_analytics,
            'cache_stats': cache_stats,
            'selector_analytics': selector_analytics,
            'features': {
                'caching_enabled': self.enable_caching,
                'streaming_enabled': self.enable_streaming,
                'dynamic_selection_enabled': True
            }
        }
    
    def cleanup(self):
        """Clean up expired sessions and cache entries"""
        expired_sessions = self.session_manager.cleanup_expired_sessions()
        
        expired_cache = 0
        if self.response_cache:
            expired_cache = self.response_cache.cleanup_expired()
        
        logger.info(f"Cleanup complete: {expired_sessions} sessions, {expired_cache} cache entries removed")
        
        return {
            'expired_sessions': expired_sessions,
            'expired_cache_entries': expired_cache
        }