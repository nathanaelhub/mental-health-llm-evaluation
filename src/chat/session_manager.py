"""
Session Management for Dynamic Model Selection Chat

Manages conversation sessions, tracks selected models per conversation,
and maintains conversation context and history.
"""

import json
import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from .model_selector import ModelSelectionResult

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    turn_id: str
    user_message: str
    assistant_message: str
    model_used: str
    timestamp: datetime
    response_time_ms: float
    evaluation_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'turn_id': self.turn_id,
            'user_message': self.user_message,
            'assistant_message': self.assistant_message,
            'model_used': self.model_used,
            'timestamp': self.timestamp.isoformat(),
            'response_time_ms': self.response_time_ms,
            'evaluation_score': self.evaluation_score
        }


@dataclass
class ChatSession:
    """Complete chat session with model selection and conversation history"""
    session_id: str
    user_id: Optional[str]
    selected_model: str
    selection_result: ModelSelectionResult
    conversation_history: List[ConversationTurn]
    created_at: datetime
    last_activity: datetime
    session_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'selected_model': self.selected_model,
            'selection_result': self.selection_result.to_dict(),
            'conversation_history': [turn.to_dict() for turn in self.conversation_history],
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'session_metadata': self.session_metadata
        }


class SessionManager:
    """
    Manages chat sessions with dynamic model selection
    
    Features:
    - Session creation and tracking
    - Conversation history management
    - Model selection persistence
    - Session cleanup and expiration
    - Analytics and session insights
    """
    
    def __init__(self, 
                 session_storage_dir: str = "temp/chat_sessions",
                 session_timeout_hours: int = 24,
                 max_sessions_per_user: int = 10):
        """
        Initialize session manager
        
        Args:
            session_storage_dir: Directory to store session data
            session_timeout_hours: Hours after which sessions expire
            max_sessions_per_user: Maximum sessions per user
        """
        self.storage_dir = Path(session_storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.max_sessions_per_user = max_sessions_per_user
        
        # In-memory session cache
        self.active_sessions: Dict[str, ChatSession] = {}
        
        # Load existing sessions
        self._load_sessions()
        
        logger.info(f"SessionManager initialized with storage: {self.storage_dir}")
    
    def create_session(self,
                      user_id: Optional[str] = None,
                      initial_prompt: str = "",
                      selection_result: ModelSelectionResult = None) -> ChatSession:
        """
        Create a new chat session
        
        Args:
            user_id: Optional user identifier
            initial_prompt: The initial user message
            selection_result: Result from model selection process
            
        Returns:
            New ChatSession object
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Clean up old sessions for this user
        if user_id:
            self._cleanup_user_sessions(user_id)
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            selected_model=selection_result.selected_model if selection_result else "openai",
            selection_result=selection_result,
            conversation_history=[],
            created_at=now,
            last_activity=now,
            session_metadata={
                'initial_prompt': initial_prompt,
                'selection_scores': selection_result.all_scores if selection_result else {},
                'selection_time_ms': selection_result.selection_time_ms if selection_result else 0
            }
        )
        
        self.active_sessions[session_id] = session
        self._save_session(session)
        
        logger.info(f"Created session {session_id} for user {user_id} with model {session.selected_model}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID"""
        if session_id not in self.active_sessions:
            self._load_session(session_id)
        
        session = self.active_sessions.get(session_id)
        
        # Check if session has expired
        if session and self._is_session_expired(session):
            self.delete_session(session_id)
            return None
        
        return session
    
    def add_conversation_turn(self,
                            session_id: str,
                            user_message: str,
                            assistant_message: str,
                            response_time_ms: float,
                            evaluation_score: Optional[float] = None) -> bool:
        """
        Add a conversation turn to the session
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_message: Assistant's response
            response_time_ms: Response generation time
            evaluation_score: Optional quality score
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
        
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            user_message=user_message,
            assistant_message=assistant_message,
            model_used=session.selected_model,
            timestamp=datetime.now(),
            response_time_ms=response_time_ms,
            evaluation_score=evaluation_score
        )
        
        session.conversation_history.append(turn)
        session.last_activity = datetime.now()
        
        self._save_session(session)
        
        logger.debug(f"Added turn to session {session_id}")
        return True
    
    def get_conversation_context(self, session_id: str, max_turns: int = 10) -> List[Dict[str, str]]:
        """
        Get conversation context for the model
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of turns to include
            
        Returns:
            List of conversation turns in model format
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        # Get recent turns
        recent_turns = session.conversation_history[-max_turns:] if max_turns > 0 else session.conversation_history
        
        # Convert to model format
        context = []
        for turn in recent_turns:
            context.extend([
                {"role": "user", "content": turn.user_message},
                {"role": "assistant", "content": turn.assistant_message}
            ])
        
        return context
    
    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """Update session metadata"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.session_metadata.update(metadata)
        session.last_activity = datetime.now()
        self._save_session(session)
        
        return True
    
    def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        """Get all active sessions for a user"""
        user_sessions = [
            session for session in self.active_sessions.values()
            if session.user_id == user_id and not self._is_session_expired(session)
        ]
        
        # Sort by last activity (most recent first)
        user_sessions.sort(key=lambda s: s.last_activity, reverse=True)
        
        return user_sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Delete from storage
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            logger.info(f"Deleted session {session_id}")
            return True
        
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count removed"""
        expired_sessions = []
        
        for session_id, session in list(self.active_sessions.items()):
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics about current sessions"""
        self.cleanup_expired_sessions()
        
        if not self.active_sessions:
            return {
                'total_sessions': 0,
                'model_distribution': {},
                'avg_conversation_length': 0,
                'total_turns': 0
            }
        
        # Model distribution
        model_counts = {}
        total_turns = 0
        conversation_lengths = []
        
        for session in self.active_sessions.values():
            model = session.selected_model
            model_counts[model] = model_counts.get(model, 0) + 1
            
            turn_count = len(session.conversation_history)
            total_turns += turn_count
            conversation_lengths.append(turn_count)
        
        avg_conversation_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
        
        return {
            'total_sessions': len(self.active_sessions),
            'model_distribution': model_counts,
            'avg_conversation_length': avg_conversation_length,
            'total_turns': total_turns,
            'active_users': len(set(s.user_id for s in self.active_sessions.values() if s.user_id))
        }
    
    def _load_sessions(self):
        """Load existing sessions from storage"""
        if not self.storage_dir.exists():
            return
        
        for session_file in self.storage_dir.glob("*.json"):
            try:
                self._load_session(session_file.stem)
            except Exception as e:
                logger.error(f"Error loading session {session_file}: {e}")
    
    def _load_session(self, session_id: str):
        """Load a specific session from storage"""
        session_file = self.storage_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return
        
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            # Convert timestamps back to datetime objects
            selection_result_data = data['selection_result']
            selection_result = ModelSelectionResult(
                selected_model=selection_result_data['selected_model'],
                selection_score=selection_result_data['selection_score'],
                selection_time_ms=selection_result_data['selection_time_ms'],
                all_scores=selection_result_data['all_scores'],
                response_preview=selection_result_data['response_preview'],
                timestamp=datetime.fromisoformat(selection_result_data['timestamp'])
            )
            
            conversation_history = []
            for turn_data in data['conversation_history']:
                turn = ConversationTurn(
                    turn_id=turn_data['turn_id'],
                    user_message=turn_data['user_message'],
                    assistant_message=turn_data['assistant_message'],
                    model_used=turn_data['model_used'],
                    timestamp=datetime.fromisoformat(turn_data['timestamp']),
                    response_time_ms=turn_data['response_time_ms'],
                    evaluation_score=turn_data.get('evaluation_score')
                )
                conversation_history.append(turn)
            
            session = ChatSession(
                session_id=data['session_id'],
                user_id=data['user_id'],
                selected_model=data['selected_model'],
                selection_result=selection_result,
                conversation_history=conversation_history,
                created_at=datetime.fromisoformat(data['created_at']),
                last_activity=datetime.fromisoformat(data['last_activity']),
                session_metadata=data['session_metadata']
            )
            
            self.active_sessions[session_id] = session
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
    
    def _save_session(self, session: ChatSession):
        """Save session to storage"""
        session_file = self.storage_dir / f"{session.session_id}.json"
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
    
    def _is_session_expired(self, session: ChatSession) -> bool:
        """Check if session has expired"""
        return datetime.now() - session.last_activity > self.session_timeout
    
    def _cleanup_user_sessions(self, user_id: str):
        """Clean up old sessions for a user if over limit"""
        user_sessions = self.get_user_sessions(user_id)
        
        if len(user_sessions) >= self.max_sessions_per_user:
            # Remove oldest sessions
            sessions_to_remove = user_sessions[self.max_sessions_per_user - 1:]
            for session in sessions_to_remove:
                self.delete_session(session.session_id)