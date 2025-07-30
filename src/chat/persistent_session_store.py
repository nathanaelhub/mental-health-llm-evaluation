"""
Persistent Session Store with Multiple Backend Support

Provides flexible persistence options for conversation sessions including
Redis for production, SQLite for development, and JSON file backup.
"""

import json
import sqlite3
import asyncio
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    aiofiles = None
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SessionStoreType(Enum):
    """Available session store backends"""
    REDIS = "redis"
    SQLITE = "sqlite"
    JSON_FILE = "json_file"
    MEMORY = "memory"  # For testing


class BaseSessionStore(ABC):
    """Abstract base class for session stores"""
    
    @abstractmethod
    async def save_session(self, session: Any) -> bool:
        """Save a session to the store"""
        pass
    
    @abstractmethod
    async def load_session(self, session_id: str) -> Optional[Any]:
        """Load a session from the store"""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from the store"""
        pass
    
    @abstractmethod
    async def get_user_sessions(self, user_id: str) -> List[Any]:
        """Get all sessions for a user"""
        pass
    
    @abstractmethod
    async def search_sessions(self, query: str, user_id: Optional[str] = None, 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[Any]:
        """Search sessions by content or metadata"""
        pass
    
    @abstractmethod
    async def save_audit_log(self, audit_entry: Dict[str, Any]) -> bool:
        """Save an audit log entry"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the store and cleanup resources"""
        pass


class RedisSessionStore(BaseSessionStore):
    """Redis-based session store for production use"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 key_prefix: str = "mh_session:",
                 ttl_seconds: int = 86400):  # 24 hours default
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self.redis = None
        self._connected = False
    
    async def _ensure_connected(self):
        """Ensure Redis connection is established"""
        if not REDIS_AVAILABLE:
            raise ImportError("aioredis not available. Install with: pip install aioredis")
        
        if not self._connected:
            try:
                self.redis = await aioredis.create_redis_pool(self.redis_url)
                self._connected = True
                logger.info("Connected to Redis session store")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
    
    async def save_session(self, session: Any) -> bool:
        """Save session to Redis with TTL"""
        await self._ensure_connected()
        
        try:
            key = f"{self.key_prefix}{session.session_id}"
            value = json.dumps(session.to_dict())
            
            # Save with TTL
            await self.redis.setex(key, self.ttl_seconds, value)
            
            # Add to user index
            if session.user_id:
                user_key = f"{self.key_prefix}user:{session.user_id}"
                await self.redis.sadd(user_key, session.session_id)
                await self.redis.expire(user_key, self.ttl_seconds)
            
            # Add to search index (simplified - in production use Redis Search)
            search_key = f"{self.key_prefix}search:all"
            await self.redis.zadd(search_key, session.last_activity.timestamp(), session.session_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving session to Redis: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[Any]:
        """Load session from Redis"""
        await self._ensure_connected()
        
        try:
            key = f"{self.key_prefix}{session_id}"
            value = await self.redis.get(key)
            
            if value:
                # Import here to avoid circular dependency
                from .conversation_session_manager import ConversationSession
                
                data = json.loads(value)
                return ConversationSession.from_dict(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading session from Redis: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from Redis"""
        await self._ensure_connected()
        
        try:
            key = f"{self.key_prefix}{session_id}"
            
            # Get session to find user_id
            session = await self.load_session(session_id)
            
            # Delete main key
            await self.redis.delete(key)
            
            # Remove from user index
            if session and session.user_id:
                user_key = f"{self.key_prefix}user:{session.user_id}"
                await self.redis.srem(user_key, session_id)
            
            # Remove from search index
            search_key = f"{self.key_prefix}search:all"
            await self.redis.zrem(search_key, session_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session from Redis: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[Any]:
        """Get all sessions for a user"""
        await self._ensure_connected()
        
        try:
            user_key = f"{self.key_prefix}user:{user_id}"
            session_ids = await self.redis.smembers(user_key)
            
            sessions = []
            for session_id in session_ids:
                session = await self.load_session(session_id.decode())
                if session:
                    sessions.append(session)
            
            # Sort by last activity
            sessions.sort(key=lambda s: s.last_activity, reverse=True)
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting user sessions from Redis: {e}")
            return []
    
    async def search_sessions(self, query: str, user_id: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[Any]:
        """Search sessions - simplified implementation"""
        await self._ensure_connected()
        
        try:
            # Get session IDs from time range
            search_key = f"{self.key_prefix}search:all"
            
            min_score = start_date.timestamp() if start_date else "-inf"
            max_score = end_date.timestamp() if end_date else "+inf"
            
            session_ids = await self.redis.zrangebyscore(search_key, min_score, max_score)
            
            # Load and filter sessions
            sessions = []
            for session_id in session_ids:
                session = await self.load_session(session_id.decode())
                if not session:
                    continue
                
                # Filter by user if specified
                if user_id and session.user_id != user_id:
                    continue
                
                # Simple text search in messages
                if query:
                    query_lower = query.lower()
                    found = any(
                        query_lower in msg.content.lower()
                        for msg in session.conversation_history
                    )
                    if not found:
                        continue
                
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error searching sessions in Redis: {e}")
            return []
    
    async def save_audit_log(self, audit_entry: Dict[str, Any]) -> bool:
        """Save audit log entry to Redis"""
        await self._ensure_connected()
        
        try:
            key = f"{self.key_prefix}audit:{audit_entry['timestamp']}"
            value = json.dumps(audit_entry)
            
            # Save with longer TTL for audit logs
            await self.redis.setex(key, self.ttl_seconds * 30, value)  # 30 days
            
            # Add to audit index
            audit_key = f"{self.key_prefix}audit:index"
            await self.redis.zadd(audit_key, datetime.fromisoformat(audit_entry['timestamp']).timestamp(), key)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving audit log to Redis: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            self._connected = False
            logger.info("Closed Redis session store")


class SQLiteSessionStore(BaseSessionStore):
    """SQLite-based session store for development/testing"""
    
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure database tables are created"""
        if self._initialized:
            return
        
        # Run synchronous SQLite operations in thread pool
        await asyncio.get_event_loop().run_in_executor(None, self._create_tables)
        self._initialized = True
    
    def _create_tables(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                selected_model TEXT,
                status TEXT,
                safety_level TEXT,
                created_at TIMESTAMP,
                last_activity TIMESTAMP,
                data TEXT
            )
        """)
        
        # Messages table for efficient searching
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                action TEXT,
                session_id TEXT,
                details TEXT
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions ON sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_activity ON sessions(last_activity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_message_session ON messages(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        
        conn.commit()
        conn.close()
    
    async def save_session(self, session: Any) -> bool:
        """Save session to SQLite"""
        await self._ensure_initialized()
        
        def _save():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # Save session
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (session_id, user_id, selected_model, status, safety_level, 
                     created_at, last_activity, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.user_id,
                    session.selected_model,
                    session.status.value,
                    session.safety_level.value,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    json.dumps(session.to_dict())
                ))
                
                # Delete existing messages for this session
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session.session_id,))
                
                # Save messages for searching
                for msg in session.conversation_history:
                    cursor.execute("""
                        INSERT INTO messages (message_id, session_id, role, content, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        msg.message_id,
                        session.session_id,
                        msg.role.value,
                        msg.content,
                        msg.timestamp.isoformat()
                    ))
                
                conn.commit()
                return True
                
            except Exception as e:
                logger.error(f"Error saving session to SQLite: {e}")
                conn.rollback()
                return False
                
            finally:
                conn.close()
        
        return await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def load_session(self, session_id: str) -> Optional[Any]:
        """Load session from SQLite"""
        await self._ensure_initialized()
        
        def _load():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("SELECT data FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                
                if row:
                    from .conversation_session_manager import ConversationSession
                    data = json.loads(row[0])
                    return ConversationSession.from_dict(data)
                
                return None
                
            finally:
                conn.close()
        
        return await asyncio.get_event_loop().run_in_executor(None, _load)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from SQLite"""
        await self._ensure_initialized()
        
        def _delete():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                return cursor.rowcount > 0
                
            except Exception as e:
                logger.error(f"Error deleting session from SQLite: {e}")
                conn.rollback()
                return False
                
            finally:
                conn.close()
        
        return await asyncio.get_event_loop().run_in_executor(None, _delete)
    
    async def get_user_sessions(self, user_id: str) -> List[Any]:
        """Get all sessions for a user"""
        await self._ensure_initialized()
        
        def _get():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT data FROM sessions 
                    WHERE user_id = ? 
                    ORDER BY last_activity DESC
                """, (user_id,))
                
                sessions = []
                from .conversation_session_manager import ConversationSession
                
                for row in cursor.fetchall():
                    data = json.loads(row[0])
                    sessions.append(ConversationSession.from_dict(data))
                
                return sessions
                
            finally:
                conn.close()
        
        return await asyncio.get_event_loop().run_in_executor(None, _get)
    
    async def search_sessions(self, query: str, user_id: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[Any]:
        """Search sessions by content"""
        await self._ensure_initialized()
        
        def _search():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # Build query
                sql = """
                    SELECT DISTINCT s.data 
                    FROM sessions s
                    LEFT JOIN messages m ON s.session_id = m.session_id
                    WHERE 1=1
                """
                params = []
                
                if user_id:
                    sql += " AND s.user_id = ?"
                    params.append(user_id)
                
                if start_date:
                    sql += " AND s.last_activity >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    sql += " AND s.last_activity <= ?"
                    params.append(end_date.isoformat())
                
                if query:
                    sql += " AND (m.content LIKE ? OR s.session_id = ?)"
                    params.extend([f"%{query}%", query])
                
                sql += " ORDER BY s.last_activity DESC"
                
                cursor.execute(sql, params)
                
                sessions = []
                from .conversation_session_manager import ConversationSession
                
                for row in cursor.fetchall():
                    data = json.loads(row[0])
                    sessions.append(ConversationSession.from_dict(data))
                
                return sessions
                
            finally:
                conn.close()
        
        return await asyncio.get_event_loop().run_in_executor(None, _search)
    
    async def save_audit_log(self, audit_entry: Dict[str, Any]) -> bool:
        """Save audit log entry"""
        await self._ensure_initialized()
        
        def _save():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO audit_log (timestamp, action, session_id, details)
                    VALUES (?, ?, ?, ?)
                """, (
                    audit_entry['timestamp'],
                    audit_entry['action'],
                    audit_entry['session_id'],
                    json.dumps(audit_entry['details'])
                ))
                
                conn.commit()
                return True
                
            except Exception as e:
                logger.error(f"Error saving audit log: {e}")
                conn.rollback()
                return False
                
            finally:
                conn.close()
        
        return await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def close(self):
        """No persistent connection to close for SQLite"""
        logger.info("Closed SQLite session store")


class JSONFileSessionStore(BaseSessionStore):
    """JSON file-based session store for backup/recovery"""
    
    def __init__(self, storage_dir: str = "sessions_backup"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Indices for efficient lookups
        self.user_index_file = self.storage_dir / "user_index.json"
        self.search_index_file = self.storage_dir / "search_index.json"
        self.audit_dir = self.storage_dir / "audit"
        self.audit_dir.mkdir(exist_ok=True)
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get file path for a session"""
        # Use subdirectories to avoid too many files in one directory
        subdir = self.storage_dir / session_id[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{session_id}.json"
    
    async def save_session(self, session: Any) -> bool:
        """Save session to JSON file"""
        try:
            session_file = self._get_session_file(session.session_id)
            
            # Save session data
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(session_file, 'w') as f:
                    await f.write(json.dumps(session.to_dict(), indent=2))
            else:
                # Fallback to synchronous file operations
                with open(session_file, 'w') as f:
                    f.write(json.dumps(session.to_dict(), indent=2))
            
            # Update user index
            await self._update_user_index(session.session_id, session.user_id)
            
            # Update search index
            await self._update_search_index(session.session_id, session.last_activity)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving session to JSON: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[Any]:
        """Load session from JSON file"""
        try:
            session_file = self._get_session_file(session_id)
            
            if not session_file.exists():
                return None
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(session_file, 'r') as f:
                    data = json.loads(await f.read())
            else:
                with open(session_file, 'r') as f:
                    data = json.loads(f.read())
            
            from .conversation_session_manager import ConversationSession
            return ConversationSession.from_dict(data)
            
        except Exception as e:
            logger.error(f"Error loading session from JSON: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session JSON file"""
        try:
            session_file = self._get_session_file(session_id)
            
            if session_file.exists():
                # Load session to get user_id
                session = await self.load_session(session_id)
                
                # Delete file
                session_file.unlink()
                
                # Update indices
                if session:
                    await self._remove_from_user_index(session_id, session.user_id)
                await self._remove_from_search_index(session_id)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting session JSON: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str) -> List[Any]:
        """Get all sessions for a user"""
        try:
            # Load user index
            user_index = await self._load_user_index()
            session_ids = user_index.get(user_id, [])
            
            # Load sessions
            sessions = []
            for session_id in session_ids:
                session = await self.load_session(session_id)
                if session:
                    sessions.append(session)
            
            # Sort by last activity
            sessions.sort(key=lambda s: s.last_activity, reverse=True)
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    async def search_sessions(self, query: str, user_id: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[Any]:
        """Search sessions - basic implementation"""
        try:
            # Get all session IDs from search index
            search_index = await self._load_search_index()
            
            # Filter by date range
            session_ids = []
            for session_id, timestamp_str in search_index.items():
                timestamp = datetime.fromisoformat(timestamp_str)
                
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue
                
                session_ids.append(session_id)
            
            # Load and filter sessions
            sessions = []
            for session_id in session_ids:
                session = await self.load_session(session_id)
                if not session:
                    continue
                
                # Filter by user
                if user_id and session.user_id != user_id:
                    continue
                
                # Text search
                if query:
                    query_lower = query.lower()
                    found = any(
                        query_lower in msg.content.lower()
                        for msg in session.conversation_history
                    )
                    if not found:
                        continue
                
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error searching sessions: {e}")
            return []
    
    async def save_audit_log(self, audit_entry: Dict[str, Any]) -> bool:
        """Save audit log entry"""
        try:
            # Generate filename based on timestamp
            timestamp = datetime.fromisoformat(audit_entry['timestamp'])
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{audit_entry['action']}.json"
            
            # Create daily subdirectory
            daily_dir = self.audit_dir / timestamp.strftime('%Y%m%d')
            daily_dir.mkdir(exist_ok=True)
            
            audit_file = daily_dir / filename
            
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(audit_file, 'w') as f:
                    await f.write(json.dumps(audit_entry, indent=2))
            else:
                with open(audit_file, 'w') as f:
                    f.write(json.dumps(audit_entry, indent=2))
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving audit log: {e}")
            return False
    
    async def _load_user_index(self) -> Dict[str, List[str]]:
        """Load user index"""
        if not self.user_index_file.exists():
            return {}
        
        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.user_index_file, 'r') as f:
                    return json.loads(await f.read())
            else:
                with open(self.user_index_file, 'r') as f:
                    return json.loads(f.read())
        except:
            return {}
    
    async def _update_user_index(self, session_id: str, user_id: Optional[str]):
        """Update user index"""
        if not user_id:
            return
        
        index = await self._load_user_index()
        
        if user_id not in index:
            index[user_id] = []
        
        if session_id not in index[user_id]:
            index[user_id].append(session_id)
        
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(self.user_index_file, 'w') as f:
                await f.write(json.dumps(index))
        else:
            with open(self.user_index_file, 'w') as f:
                f.write(json.dumps(index))
    
    async def _remove_from_user_index(self, session_id: str, user_id: Optional[str]):
        """Remove from user index"""
        if not user_id:
            return
        
        index = await self._load_user_index()
        
        if user_id in index and session_id in index[user_id]:
            index[user_id].remove(session_id)
            
            if not index[user_id]:
                del index[user_id]
        
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(self.user_index_file, 'w') as f:
                await f.write(json.dumps(index))
        else:
            with open(self.user_index_file, 'w') as f:
                f.write(json.dumps(index))
    
    async def _load_search_index(self) -> Dict[str, str]:
        """Load search index"""
        if not self.search_index_file.exists():
            return {}
        
        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.search_index_file, 'r') as f:
                    return json.loads(await f.read())
            else:
                with open(self.search_index_file, 'r') as f:
                    return json.loads(f.read())
        except:
            return {}
    
    async def _update_search_index(self, session_id: str, last_activity: datetime):
        """Update search index"""
        index = await self._load_search_index()
        index[session_id] = last_activity.isoformat()
        
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(self.search_index_file, 'w') as f:
                await f.write(json.dumps(index))
        else:
            with open(self.search_index_file, 'w') as f:
                f.write(json.dumps(index))
    
    async def _remove_from_search_index(self, session_id: str):
        """Remove from search index"""
        index = await self._load_search_index()
        
        if session_id in index:
            del index[session_id]
        
        if AIOFILES_AVAILABLE:
            async with aiofiles.open(self.search_index_file, 'w') as f:
                await f.write(json.dumps(index))
        else:
            with open(self.search_index_file, 'w') as f:
                f.write(json.dumps(index))
    
    async def close(self):
        """No resources to close for JSON store"""
        logger.info("Closed JSON file session store")


class MemorySessionStore(BaseSessionStore):
    """In-memory session store for testing"""
    
    def __init__(self):
        self.sessions: Dict[str, Any] = {}
        self.audit_logs: List[Dict[str, Any]] = []
    
    async def save_session(self, session: Any) -> bool:
        self.sessions[session.session_id] = session
        return True
    
    async def load_session(self, session_id: str) -> Optional[Any]:
        return self.sessions.get(session_id)
    
    async def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    async def get_user_sessions(self, user_id: str) -> List[Any]:
        return [s for s in self.sessions.values() if s.user_id == user_id]
    
    async def search_sessions(self, query: str, user_id: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[Any]:
        results = []
        for session in self.sessions.values():
            # Filter by user
            if user_id and session.user_id != user_id:
                continue
            
            # Filter by date
            if start_date and session.last_activity < start_date:
                continue
            if end_date and session.last_activity > end_date:
                continue
            
            # Text search
            if query:
                query_lower = query.lower()
                found = any(
                    query_lower in msg.content.lower()
                    for msg in session.conversation_history
                )
                if not found:
                    continue
            
            results.append(session)
        
        return results
    
    async def save_audit_log(self, audit_entry: Dict[str, Any]) -> bool:
        self.audit_logs.append(audit_entry)
        return True
    
    async def close(self):
        logger.info("Closed memory session store")


class PersistentSessionStore:
    """Factory class for creating appropriate session store"""
    
    def __new__(cls, store_type: SessionStoreType, config: Optional[Dict[str, Any]] = None) -> BaseSessionStore:
        """Create appropriate session store based on type"""
        
        config = config or {}
        
        if store_type == SessionStoreType.REDIS:
            return RedisSessionStore(
                redis_url=config.get('redis_url', 'redis://localhost:6379'),
                key_prefix=config.get('key_prefix', 'mh_session:'),
                ttl_seconds=config.get('ttl_seconds', 86400)
            )
        
        elif store_type == SessionStoreType.SQLITE:
            return SQLiteSessionStore(
                db_path=config.get('db_path', 'sessions.db')
            )
        
        elif store_type == SessionStoreType.JSON_FILE:
            return JSONFileSessionStore(
                storage_dir=config.get('storage_dir', 'sessions_backup')
            )
        
        elif store_type == SessionStoreType.MEMORY:
            return MemorySessionStore()
        
        else:
            raise ValueError(f"Unknown store type: {store_type}")