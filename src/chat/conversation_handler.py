"""
Conversation Handler for Dynamic Model Selection Chat

Manages ongoing conversations with the selected model, handles
streaming responses, and maintains conversation flow.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass

from .session_manager import SessionManager, ChatSession
from .model_selector import ModelSelector
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ConversationResponse:
    """Response from conversation handler"""
    message: str
    model_used: str
    response_time_ms: float
    session_id: str
    turn_count: int
    is_streaming: bool = False
    error: Optional[str] = None


class ConversationHandler:
    """
    Handles ongoing conversations after initial model selection
    
    Features:
    - Continues conversations with the selected model
    - Supports streaming responses
    - Handles model failover if primary model fails
    - Maintains conversation context
    - Tracks conversation quality over time
    """
    
    def __init__(self,
                 session_manager: SessionManager,
                 model_selector: ModelSelector,
                 enable_streaming: bool = True,
                 failover_enabled: bool = True):
        """
        Initialize conversation handler
        
        Args:
            session_manager: Session management instance
            model_selector: Model selection instance  
            enable_streaming: Whether to support streaming responses
            failover_enabled: Whether to failover on model errors
        """
        self.session_manager = session_manager
        self.model_selector = model_selector
        self.enable_streaming = enable_streaming
        self.failover_enabled = failover_enabled
        
        logger.info("ConversationHandler initialized")
    
    async def continue_conversation(self,
                                  session_id: str,
                                  user_message: str,
                                  system_prompt: Optional[str] = None) -> ConversationResponse:
        """
        Continue an existing conversation
        
        Args:
            session_id: Session identifier
            user_message: New user message
            system_prompt: Optional system prompt override
            
        Returns:
            ConversationResponse with the assistant's reply
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return ConversationResponse(
                message="Session not found or expired",
                model_used="none",
                response_time_ms=0,
                session_id=session_id,
                turn_count=0,
                error="Session not found"
            )
        
        logger.info(f"Continuing conversation in session {session_id} with {session.selected_model}")
        
        # Get conversation context
        conversation_history = self.session_manager.get_conversation_context(session_id)
        
        # Generate response with selected model
        start_time = time.time()
        
        try:
            response = await self._generate_with_model(
                model_name=session.selected_model,
                user_message=user_message,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Add turn to session
            self.session_manager.add_conversation_turn(
                session_id=session_id,
                user_message=user_message,
                assistant_message=response,
                response_time_ms=response_time_ms
            )
            
            return ConversationResponse(
                message=response,
                model_used=session.selected_model,
                response_time_ms=response_time_ms,
                session_id=session_id,
                turn_count=len(session.conversation_history) + 1
            )
            
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
            
            # Try failover if enabled
            if self.failover_enabled:
                return await self._handle_failover(
                    session, user_message, system_prompt, conversation_history, start_time
                )
            
            return ConversationResponse(
                message="I apologize, but I'm having trouble responding right now. Please try again.",
                model_used=session.selected_model,
                response_time_ms=(time.time() - start_time) * 1000,
                session_id=session_id,
                turn_count=len(session.conversation_history),
                error=str(e)
            )
    
    async def stream_conversation(self,
                                session_id: str,
                                user_message: str,
                                system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream conversation response in real-time
        
        Args:
            session_id: Session identifier
            user_message: New user message
            system_prompt: Optional system prompt override
            
        Yields:
            Response chunks as they're generated
        """
        if not self.enable_streaming:
            # Fall back to regular response
            response = await self.continue_conversation(session_id, user_message, system_prompt)
            yield response.message
            return
        
        session = self.session_manager.get_session(session_id)
        if not session:
            yield "Session not found or expired"
            return
        
        # Get the model client
        model_client = self.model_selector.model_clients.get(session.selected_model)
        if not model_client or not model_client.supports_streaming():
            # Fall back to regular response
            response = await self.continue_conversation(session_id, user_message, system_prompt)
            yield response.message
            return
        
        logger.info(f"Streaming conversation in session {session_id}")
        
        conversation_history = self.session_manager.get_conversation_context(session_id)
        full_response = ""
        start_time = time.time()
        
        try:
            async for chunk in model_client.stream_response(
                prompt=user_message,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            ):
                full_response += chunk
                yield chunk
            
            # Add completed turn to session
            response_time_ms = (time.time() - start_time) * 1000
            self.session_manager.add_conversation_turn(
                session_id=session_id,
                user_message=user_message,
                assistant_message=full_response,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            logger.error(f"Error in streaming conversation: {e}")
            yield f"\\n\\nI apologize, but I encountered an error: {str(e)}"
    
    async def _generate_with_model(self,
                                 model_name: str,
                                 user_message: str,
                                 system_prompt: Optional[str] = None,
                                 conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response with specific model"""
        
        client = self.model_selector.model_clients.get(model_name)
        if not client:
            raise ValueError(f"Model {model_name} not available")
        
        if hasattr(client, 'generate_response'):
            # Use standardized async interface
            response_obj = await client.generate_response(
                prompt=user_message,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
            
            if hasattr(response_obj, 'content'):
                return response_obj.content
            else:
                return str(response_obj)
        
        else:
            # Fallback for older clients
            response_obj = client.chat(user_message)
            
            if isinstance(response_obj, dict):
                return response_obj.get('content', str(response_obj))
            else:
                return str(response_obj)
    
    async def _handle_failover(self,
                             session: ChatSession,
                             user_message: str,
                             system_prompt: Optional[str],
                             conversation_history: List[Dict[str, str]],
                             start_time: float) -> ConversationResponse:
        """Handle failover to alternate model"""
        
        logger.warning(f"Primary model {session.selected_model} failed, attempting failover")
        
        # Get list of available models excluding the failed one
        available_models = [
            model for model in self.model_selector.get_available_models() 
            if model != session.selected_model
        ]
        
        if not available_models:
            return ConversationResponse(
                message="I apologize, but all models are currently unavailable. Please try again later.",
                model_used=session.selected_model,
                response_time_ms=(time.time() - start_time) * 1000,
                session_id=session.session_id,
                turn_count=len(session.conversation_history),
                error="No fallback models available"
            )
        
        # Try fallback model (use first available)
        fallback_model = available_models[0]
        
        try:
            response = await self._generate_with_model(
                model_name=fallback_model,
                user_message=user_message,
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Add turn to session with fallback model noted
            self.session_manager.add_conversation_turn(
                session_id=session.session_id,
                user_message=user_message,
                assistant_message=response,
                response_time_ms=response_time_ms
            )
            
            # Update session metadata to note failover
            self.session_manager.update_session_metadata(
                session.session_id,
                {'last_failover': {
                    'from_model': session.selected_model,
                    'to_model': fallback_model,
                    'timestamp': time.time()
                }}
            )
            
            logger.info(f"Successful failover from {session.selected_model} to {fallback_model}")
            
            return ConversationResponse(
                message=response,
                model_used=fallback_model,
                response_time_ms=response_time_ms,
                session_id=session.session_id,
                turn_count=len(session.conversation_history) + 1
            )
            
        except Exception as e:
            logger.error(f"Failover to {fallback_model} also failed: {e}")
            
            return ConversationResponse(
                message="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                model_used=session.selected_model,
                response_time_ms=(time.time() - start_time) * 1000,
                session_id=session.session_id,
                turn_count=len(session.conversation_history),
                error=f"Failover failed: {str(e)}"
            )
    
    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation performance"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if not session.conversation_history:
            return {
                'session_id': session_id,
                'turn_count': 0,
                'avg_response_time': 0,
                'model_used': session.selected_model,
                'selection_info': session.selection_result.to_dict()
            }
        
        # Calculate metrics
        response_times = [turn.response_time_ms for turn in session.conversation_history]
        evaluation_scores = [turn.evaluation_score for turn in session.conversation_history if turn.evaluation_score is not None]
        
        return {
            'session_id': session_id,
            'turn_count': len(session.conversation_history),
            'avg_response_time': sum(response_times) / len(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'avg_evaluation_score': sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else None,
            'model_used': session.selected_model,
            'selection_info': session.selection_result.to_dict(),
            'total_duration_minutes': (session.last_activity - session.created_at).total_seconds() / 60,
            'failovers': session.session_metadata.get('last_failover', {})
        }
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if specific model is available"""
        return model_name in self.model_selector.model_clients
    
    async def switch_model(self, session_id: str, new_model: str) -> bool:
        """
        Switch to a different model mid-conversation
        
        Args:
            session_id: Session identifier
            new_model: New model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return False
        
        if not self.is_model_available(new_model):
            logger.warning(f"Cannot switch to unavailable model: {new_model}")
            return False
        
        # Update session
        old_model = session.selected_model
        session.selected_model = new_model
        
        # Update metadata
        self.session_manager.update_session_metadata(
            session_id,
            {'model_switch': {
                'from_model': old_model,
                'to_model': new_model,
                'timestamp': time.time()
            }}
        )
        
        logger.info(f"Switched session {session_id} from {old_model} to {new_model}")
        return True