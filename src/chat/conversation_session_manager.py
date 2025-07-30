"""
Conversation Session Manager for Mental Health Chat System

Manages conversation state, model assignments, and context for ongoing chats
with comprehensive safety features and persistent storage.
"""

import json
import uuid
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set, AsyncGenerator
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from collections import defaultdict

from .persistent_session_store import PersistentSessionStore, SessionStoreType
from .dynamic_model_selector import PromptType
from ..evaluation.evaluation_metrics import TherapeuticEvaluator

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Role of message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SessionStatus(Enum):
    """Status of conversation session"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"  # Safety triggered
    ARCHIVED = "archived"
    MIGRATING = "migrating"  # Model switch in progress


class SafetyLevel(Enum):
    """Safety assessment levels"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Message:
    """Single message in conversation"""
    message_id: str
    role: MessageRole
    content: str
    timestamp: datetime
    model_used: Optional[str] = None
    token_count: Optional[int] = None
    safety_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['role'] = self.role.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        data['role'] = MessageRole(data['role'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ConversationSession:
    """Complete conversation session with all context"""
    session_id: str
    user_id: Optional[str]
    selected_model: str
    conversation_history: List[Message]
    metadata: Dict[str, Any]
    created_at: datetime
    last_activity: datetime
    evaluation_scores: Dict[str, float]
    status: SessionStatus = SessionStatus.ACTIVE
    safety_level: SafetyLevel = SafetyLevel.SAFE
    total_tokens: int = 0
    model_switches: List[Dict[str, Any]] = field(default_factory=list)
    crisis_flags: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['safety_level'] = self.safety_level.value
        result['created_at'] = self.created_at.isoformat()
        result['last_activity'] = self.last_activity.isoformat()
        result['conversation_history'] = [msg.to_dict() for msg in self.conversation_history]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        data['status'] = SessionStatus(data['status'])
        data['safety_level'] = SafetyLevel(data['safety_level'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        data['conversation_history'] = [Message.from_dict(msg) for msg in data['conversation_history']]
        return cls(**data)
    
    def get_context_messages(self, max_tokens: int = 4000) -> List[Dict[str, str]]:
        """Get conversation context within token limit"""
        context = []
        current_tokens = 0
        
        # Traverse history in reverse to prioritize recent messages
        for message in reversed(self.conversation_history):
            if message.token_count:
                if current_tokens + message.token_count > max_tokens:
                    break
                current_tokens += message.token_count
            
            context.insert(0, {
                'role': message.role.value,
                'content': message.content
            })
        
        return context
    
    def add_message(self, message: Message):
        """Add message to conversation history"""
        self.conversation_history.append(message)
        self.last_activity = datetime.now()
        if message.token_count:
            self.total_tokens += message.token_count


class ConversationSessionManager:
    """
    Manages conversation state, model assignments, and context for ongoing chats.
    
    Features:
    - Session lifecycle management with automatic cleanup
    - Persistent storage with multiple backend options
    - Token count management and context optimization
    - Model-specific adaptations and switching
    - Comprehensive safety monitoring and crisis detection
    - Audit trail for compliance
    - WebSocket support for real-time updates
    """
    
    def __init__(self,
                 store_type: SessionStoreType = SessionStoreType.SQLITE,
                 store_config: Optional[Dict[str, Any]] = None,
                 session_timeout_minutes: int = 30,
                 max_context_tokens: int = 4000,
                 enable_safety_monitoring: bool = True,
                 enable_audit_trail: bool = True):
        """
        Initialize session manager
        
        Args:
            store_type: Type of persistent store to use
            store_config: Configuration for the store
            session_timeout_minutes: Minutes before session becomes inactive
            max_context_tokens: Maximum tokens for context window
            enable_safety_monitoring: Whether to monitor for safety issues
            enable_audit_trail: Whether to maintain audit trail
        """
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.session_store = PersistentSessionStore(store_type, store_config)
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_context_tokens = max_context_tokens
        self.enable_safety_monitoring = enable_safety_monitoring
        self.enable_audit_trail = enable_audit_trail
        
        # Safety monitoring
        self.evaluator = TherapeuticEvaluator() if enable_safety_monitoring else None
        self.safety_thresholds = {
            SafetyLevel.SAFE: 0.7,
            SafetyLevel.CAUTION: 0.5,
            SafetyLevel.WARNING: 0.3,
            SafetyLevel.CRITICAL: 0.0
        }
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Dict[str, Set[Any]] = defaultdict(set)
        
        # Model-specific configurations
        self.model_configs = {
            'openai': {
                'max_tokens': 4096,
                'context_format': 'openai',
                'token_multiplier': 1.3  # Rough estimate
            },
            'claude': {
                'max_tokens': 8192,
                'context_format': 'claude',
                'token_multiplier': 1.2
            },
            'deepseek': {
                'max_tokens': 4096,
                'context_format': 'openai',  # Compatible format
                'token_multiplier': 1.3
            },
            'gemma': {
                'max_tokens': 2048,
                'context_format': 'openai',
                'token_multiplier': 1.4
            }
        }
        
        # Start background cleanup task
        asyncio.create_task(self._cleanup_inactive_sessions())
        
        logger.info(f"ConversationSessionManager initialized with {store_type.value} store")
    
    async def create_session(self,
                           user_id: Optional[str],
                           selected_model: str,
                           initial_message: str,
                           metadata: Optional[Dict[str, Any]] = None) -> ConversationSession:
        """
        Create a new conversation session
        
        Args:
            user_id: Optional user identifier
            selected_model: Model selected for this conversation
            initial_message: First user message
            metadata: Optional session metadata
            
        Returns:
            New ConversationSession instance
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create initial user message
        user_message = Message(
            message_id=str(uuid.uuid4()),
            role=MessageRole.USER,
            content=initial_message,
            timestamp=now,
            token_count=self._estimate_tokens(initial_message, selected_model)
        )
        
        # Create session
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            selected_model=selected_model,
            conversation_history=[user_message],
            metadata=metadata or {},
            created_at=now,
            last_activity=now,
            evaluation_scores={},
            total_tokens=user_message.token_count or 0
        )
        
        # Store in memory and persistent storage
        self.active_sessions[session_id] = session
        await self.session_store.save_session(session)
        
        # Audit trail
        if self.enable_audit_trail:
            await self._audit_log("session_created", session_id, {
                'user_id': user_id,
                'model': selected_model
            })
        
        logger.info(f"Created session {session_id} for user {user_id} with model {selected_model}")
        
        # Notify WebSocket connections
        await self._broadcast_session_update(session_id, "session_created", session)
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get session by ID, loading from store if needed
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationSession or None if not found
        """
        # Check memory cache first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Check if session is still active
            if self._is_session_active(session):
                return session
            else:
                # Mark as inactive
                session.status = SessionStatus.INACTIVE
                await self.session_store.save_session(session)
        
        # Try loading from persistent store
        session = await self.session_store.load_session(session_id)
        if session and self._is_session_active(session):
            self.active_sessions[session_id] = session
            return session
        
        return None
    
    async def add_message(self,
                         session_id: str,
                         role: MessageRole,
                         content: str,
                         model_used: Optional[str] = None) -> Optional[Message]:
        """
        Add a message to the conversation
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            model_used: Model that generated the message (for assistant messages)
            
        Returns:
            Created Message or None if session not found
        """
        session = await self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return None
        
        # Create message
        message = Message(
            message_id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            model_used=model_used or session.selected_model,
            token_count=self._estimate_tokens(content, session.selected_model)
        )
        
        # Perform safety check if enabled and it's an assistant message
        if self.enable_safety_monitoring and role == MessageRole.ASSISTANT:
            safety_score = await self._evaluate_safety(content, session)
            message.safety_score = safety_score
            
            # Update session safety level
            await self._update_safety_level(session, safety_score)
        
        # Add to session
        session.add_message(message)
        
        # Check token limits
        if await self._check_token_limit(session):
            # Trigger summarization if needed
            await self._handle_long_conversation(session)
        
        # Save to persistent store
        await self.session_store.save_session(session)
        
        # Audit trail
        if self.enable_audit_trail:
            await self._audit_log("message_added", session_id, {
                'role': role.value,
                'model': model_used,
                'safety_score': message.safety_score
            })
        
        # Notify WebSocket connections
        await self._broadcast_session_update(session_id, "message_added", message)
        
        return message
    
    async def migrate_session_model(self,
                                  session_id: str,
                                  new_model: str,
                                  reason: str) -> bool:
        """
        Migrate session to a different model
        
        Args:
            session_id: Session identifier
            new_model: New model to use
            reason: Reason for migration
            
        Returns:
            True if successful, False otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        old_model = session.selected_model
        
        # Mark session as migrating
        session.status = SessionStatus.MIGRATING
        
        try:
            # Record model switch
            session.model_switches.append({
                'from_model': old_model,
                'to_model': new_model,
                'timestamp': datetime.now().isoformat(),
                'reason': reason
            })
            
            # Update model
            session.selected_model = new_model
            
            # Add system message about switch
            system_message = Message(
                message_id=str(uuid.uuid4()),
                role=MessageRole.SYSTEM,
                content=f"Model switched from {old_model} to {new_model}: {reason}",
                timestamp=datetime.now(),
                model_used=new_model
            )
            session.add_message(system_message)
            
            # Reactivate session
            session.status = SessionStatus.ACTIVE
            
            # Save changes
            await self.session_store.save_session(session)
            
            # Audit trail
            if self.enable_audit_trail:
                await self._audit_log("model_migrated", session_id, {
                    'from_model': old_model,
                    'to_model': new_model,
                    'reason': reason
                })
            
            # Notify WebSocket connections
            await self._broadcast_session_update(session_id, "model_migrated", {
                'old_model': old_model,
                'new_model': new_model
            })
            
            logger.info(f"Migrated session {session_id} from {old_model} to {new_model}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating session {session_id}: {e}")
            session.status = SessionStatus.ACTIVE  # Revert status
            return False
    
    def get_model_context(self,
                         session: ConversationSession,
                         max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation context formatted for specific model
        
        Args:
            session: Conversation session
            max_tokens: Maximum tokens (uses model default if not specified)
            
        Returns:
            Context messages formatted for the model
        """
        model_config = self.model_configs.get(session.selected_model, {})
        max_tokens = max_tokens or model_config.get('max_tokens', self.max_context_tokens)
        
        # Get messages within token limit
        context = session.get_context_messages(max_tokens)
        
        # Apply model-specific formatting if needed
        context_format = model_config.get('context_format', 'openai')
        
        if context_format == 'claude':
            # Claude-specific formatting (if different)
            return self._format_context_claude(context)
        else:
            # Default OpenAI-compatible format
            return context
    
    async def get_user_sessions(self,
                              user_id: str,
                              include_inactive: bool = False) -> List[ConversationSession]:
        """
        Get all sessions for a user
        
        Args:
            user_id: User identifier
            include_inactive: Whether to include inactive sessions
            
        Returns:
            List of user's sessions
        """
        sessions = await self.session_store.get_user_sessions(user_id)
        
        if not include_inactive:
            sessions = [s for s in sessions if self._is_session_active(s)]
        
        # Update memory cache
        for session in sessions:
            if session.session_id not in self.active_sessions:
                self.active_sessions[session.session_id] = session
        
        return sessions
    
    async def search_sessions(self,
                            query: str,
                            user_id: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[ConversationSession]:
        """
        Search sessions by content or metadata
        
        Args:
            query: Search query
            user_id: Optional user filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Matching sessions
        """
        return await self.session_store.search_sessions(query, user_id, start_date, end_date)
    
    async def archive_session(self, session_id: str) -> bool:
        """Archive a session"""
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.status = SessionStatus.ARCHIVED
        await self.session_store.save_session(session)
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Audit trail
        if self.enable_audit_trail:
            await self._audit_log("session_archived", session_id, {})
        
        return True
    
    # Safety monitoring methods
    
    async def _evaluate_safety(self, content: str, session: ConversationSession) -> float:
        """Evaluate safety score of content"""
        if not self.evaluator:
            return 1.0
        
        # Get recent context for evaluation
        recent_messages = session.conversation_history[-5:]
        context = "\n".join([f"{msg.role.value}: {msg.content}" for msg in recent_messages])
        
        # Evaluate using the therapeutic evaluator
        evaluation = self.evaluator.evaluate_response(
            prompt=context,
            response=content,
            response_time_ms=0,
            input_tokens=0,
            output_tokens=0
        )
        
        # Extract safety score
        if hasattr(evaluation, 'safety_score'):
            return evaluation.safety_score / 10.0  # Normalize to 0-1
        elif isinstance(evaluation, dict):
            return evaluation.get('safety_score', 10.0) / 10.0
        
        return 1.0
    
    async def _update_safety_level(self, session: ConversationSession, safety_score: float):
        """Update session safety level based on score"""
        old_level = session.safety_level
        
        # Determine new safety level
        for level in [SafetyLevel.CRITICAL, SafetyLevel.WARNING, SafetyLevel.CAUTION, SafetyLevel.SAFE]:
            if safety_score <= self.safety_thresholds[level]:
                session.safety_level = level
                break
        
        # Handle safety level changes
        if old_level != session.safety_level:
            logger.warning(f"Session {session.session_id} safety level changed from {old_level.value} to {session.safety_level.value}")
            
            # Crisis detection
            if session.safety_level in [SafetyLevel.WARNING, SafetyLevel.CRITICAL]:
                await self._handle_crisis_detection(session, safety_score)
    
    async def _handle_crisis_detection(self, session: ConversationSession, safety_score: float):
        """Handle crisis detection and escalation"""
        crisis_info = {
            'timestamp': datetime.now().isoformat(),
            'safety_score': safety_score,
            'safety_level': session.safety_level.value,
            'last_messages': [msg.content for msg in session.conversation_history[-3:]]
        }
        
        session.crisis_flags.append(crisis_info)
        
        # Suspend session if critical
        if session.safety_level == SafetyLevel.CRITICAL:
            session.status = SessionStatus.SUSPENDED
            logger.critical(f"Session {session.session_id} suspended due to critical safety level")
        
        # Audit trail
        if self.enable_audit_trail:
            await self._audit_log("crisis_detected", session.session_id, crisis_info)
        
        # Notify WebSocket connections for immediate intervention
        await self._broadcast_session_update(session.session_id, "crisis_alert", crisis_info)
    
    # Token management methods
    
    def _estimate_tokens(self, text: str, model: str) -> int:
        """Estimate token count for text"""
        model_config = self.model_configs.get(model, {})
        multiplier = model_config.get('token_multiplier', 1.3)
        
        # Simple estimation: words * multiplier
        word_count = len(text.split())
        return int(word_count * multiplier)
    
    async def _check_token_limit(self, session: ConversationSession) -> bool:
        """Check if session is approaching token limit"""
        model_config = self.model_configs.get(session.selected_model, {})
        max_tokens = model_config.get('max_tokens', self.max_context_tokens)
        
        # Check if we're at 80% of limit
        return session.total_tokens >= (max_tokens * 0.8)
    
    async def _handle_long_conversation(self, session: ConversationSession):
        """Handle conversations approaching token limits"""
        logger.info(f"Session {session.session_id} approaching token limit, considering summarization")
        
        # For now, just log - full summarization implementation would go here
        # In production, this would:
        # 1. Summarize older messages
        # 2. Create a new system message with summary
        # 3. Archive original messages
        # 4. Reset token count
        
        if self.enable_audit_trail:
            await self._audit_log("token_limit_approached", session.session_id, {
                'total_tokens': session.total_tokens,
                'message_count': len(session.conversation_history)
            })
    
    # Utility methods
    
    def _is_session_active(self, session: ConversationSession) -> bool:
        """Check if session is still active"""
        if session.status != SessionStatus.ACTIVE:
            return False
        
        time_since_activity = datetime.now() - session.last_activity
        return time_since_activity < self.session_timeout
    
    def _format_context_claude(self, context: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format context for Claude models"""
        # Claude uses a similar format, but this is where any
        # Claude-specific transformations would go
        return context
    
    async def _cleanup_inactive_sessions(self):
        """Background task to cleanup inactive sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                inactive_sessions = []
                for session_id, session in list(self.active_sessions.items()):
                    if not self._is_session_active(session):
                        inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    session = self.active_sessions[session_id]
                    session.status = SessionStatus.INACTIVE
                    await self.session_store.save_session(session)
                    del self.active_sessions[session_id]
                    
                    logger.info(f"Marked session {session_id} as inactive")
                
                if inactive_sessions and self.enable_audit_trail:
                    await self._audit_log("sessions_cleanup", "system", {
                        'inactive_count': len(inactive_sessions),
                        'session_ids': inactive_sessions
                    })
                    
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    async def _audit_log(self, action: str, session_id: str, details: Dict[str, Any]):
        """Log audit trail entry"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'session_id': session_id,
            'details': details
        }
        
        await self.session_store.save_audit_log(audit_entry)
    
    # WebSocket support methods
    
    def register_websocket(self, session_id: str, websocket: Any):
        """Register a WebSocket connection for a session"""
        self.websocket_connections[session_id].add(websocket)
        logger.debug(f"WebSocket registered for session {session_id}")
    
    def unregister_websocket(self, session_id: str, websocket: Any):
        """Unregister a WebSocket connection"""
        self.websocket_connections[session_id].discard(websocket)
        if not self.websocket_connections[session_id]:
            del self.websocket_connections[session_id]
        logger.debug(f"WebSocket unregistered for session {session_id}")
    
    async def _broadcast_session_update(self, session_id: str, event_type: str, data: Any):
        """Broadcast update to all WebSocket connections for a session"""
        if session_id not in self.websocket_connections:
            return
        
        message = {
            'type': event_type,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'data': data.to_dict() if hasattr(data, 'to_dict') else data
        }
        
        # Send to all connected WebSockets
        disconnected = []
        for websocket in self.websocket_connections[session_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected WebSockets
        for websocket in disconnected:
            self.unregister_websocket(session_id, websocket)
    
    async def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a specific session"""
        session = await self.get_session(session_id)
        if not session:
            return {}
        
        # Calculate analytics
        message_count = len(session.conversation_history)
        user_messages = [m for m in session.conversation_history if m.role == MessageRole.USER]
        assistant_messages = [m for m in session.conversation_history if m.role == MessageRole.ASSISTANT]
        
        avg_user_length = sum(len(m.content) for m in user_messages) / len(user_messages) if user_messages else 0
        avg_assistant_length = sum(len(m.content) for m in assistant_messages) / len(assistant_messages) if assistant_messages else 0
        
        safety_scores = [m.safety_score for m in assistant_messages if m.safety_score is not None]
        avg_safety_score = sum(safety_scores) / len(safety_scores) if safety_scores else None
        
        duration = session.last_activity - session.created_at
        
        return {
            'session_id': session_id,
            'status': session.status.value,
            'safety_level': session.safety_level.value,
            'message_count': message_count,
            'user_message_count': len(user_messages),
            'assistant_message_count': len(assistant_messages),
            'avg_user_message_length': avg_user_length,
            'avg_assistant_message_length': avg_assistant_length,
            'avg_safety_score': avg_safety_score,
            'total_tokens': session.total_tokens,
            'duration_minutes': duration.total_seconds() / 60,
            'model_switches': len(session.model_switches),
            'crisis_flags': len(session.crisis_flags)
        }
    
    async def close(self):
        """Cleanup resources"""
        # Save all active sessions
        for session in self.active_sessions.values():
            await self.session_store.save_session(session)
        
        # Close persistent store
        await self.session_store.close()
        
        logger.info("ConversationSessionManager closed")