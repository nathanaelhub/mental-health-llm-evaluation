"""
WebSocket Manager for Real-time Chat Communication

Handles WebSocket connections, streaming responses, model selection updates,
and safety alerts with automatic reconnection and heartbeat support.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Set, Optional, Any, List
from datetime import datetime, timedelta

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .models import (
    WebSocketMessage, StreamingChunk, ModelSelectionUpdate, 
    SafetyAlert, SafetyLevel
)

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Individual WebSocket connection wrapper"""
    
    def __init__(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None):
        self.websocket = websocket
        self.session_id = session_id
        self.user_id = user_id
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()
        self.is_active = True
        self.message_count = 0
        
    async def send_message(self, message: WebSocketMessage):
        """Send a message through the WebSocket"""
        try:
            await self.websocket.send_text(message.json())
            self.message_count += 1
            logger.debug(f"Sent WebSocket message to session {self.session_id}: {message.type}")
        except Exception as e:
            logger.error(f"Error sending WebSocket message to session {self.session_id}: {e}")
            self.is_active = False
            raise
    
    async def send_json(self, data: Dict[str, Any]):
        """Send JSON data directly"""
        try:
            await self.websocket.send_json(data)
            self.message_count += 1
        except Exception as e:
            logger.error(f"Error sending JSON to session {self.session_id}: {e}")
            self.is_active = False
            raise
    
    async def ping(self):
        """Send heartbeat ping"""
        try:
            await self.websocket.ping()
            self.last_ping = datetime.now()
            logger.debug(f"Sent ping to session {self.session_id}")
        except Exception as e:
            logger.warning(f"Ping failed for session {self.session_id}: {e}")
            self.is_active = False
    
    def is_stale(self, timeout_minutes: int = 30) -> bool:
        """Check if connection is stale"""
        return (datetime.now() - self.last_ping).total_seconds() > (timeout_minutes * 60)


class WebSocketManager:
    """
    Manages WebSocket connections and handles real-time communication
    
    Features:
    - Connection lifecycle management
    - Message broadcasting
    - Heartbeat monitoring
    - Automatic cleanup of stale connections
    - Rate limiting per connection
    - Message queuing for offline connections
    """
    
    def __init__(self, heartbeat_interval: int = 30, max_connections: int = 1000):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.session_connections: Dict[str, Set[str]] = {}  # session_id -> connection_ids
        self.heartbeat_interval = heartbeat_interval
        self.max_connections = max_connections
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}  # connection_id -> timestamps
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max = 100  # messages per window
        
        # Message queuing for disconnected sessions
        self.message_queue: Dict[str, List[Dict[str, Any]]] = {}
        self.max_queue_size = 50
        
        # Start background tasks
        self._heartbeat_task = None
        self._cleanup_task = None
        
        logger.info(f"WebSocketManager initialized with {max_connections} max connections")
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None) -> str:
        """
        Accept and register a new WebSocket connection
        
        Returns:
            Connection ID for the new connection
        """
        if len(self.connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Server overloaded")
            raise Exception("Maximum WebSocket connections reached")
        
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = f"{session_id}_{int(time.time() * 1000)}"
        
        # Create connection wrapper
        connection = WebSocketConnection(websocket, session_id, user_id)
        self.connections[connection_id] = connection
        
        # Track session connections
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()
        self.session_connections[session_id].add(connection_id)
        
        # Initialize rate limiting
        self.rate_limits[connection_id] = []
        
        # Send queued messages if any
        await self._send_queued_messages(session_id, connection)
        
        # Start background tasks if this is the first connection
        if len(self.connections) == 1:
            await self._start_background_tasks()
        
        logger.info(f"WebSocket connected: {connection_id} for session {session_id}")
        
        # Send welcome message
        welcome_msg = WebSocketMessage(
            type="connection_established",
            session_id=session_id,
            data={
                "connection_id": connection_id,
                "server_time": datetime.now().isoformat(),
                "heartbeat_interval": self.heartbeat_interval
            }
        )
        await connection.send_message(welcome_msg)
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Disconnect and cleanup WebSocket connection"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        session_id = connection.session_id
        
        # Remove from tracking
        del self.connections[connection_id]
        
        if session_id in self.session_connections:
            self.session_connections[session_id].discard(connection_id)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
        
        # Cleanup rate limiting
        if connection_id in self.rate_limits:
            del self.rate_limits[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id} from session {session_id}")
        
        # Stop background tasks if no connections remain
        if not self.connections:
            asyncio.create_task(self._stop_background_tasks())
    
    async def send_to_session(self, session_id: str, message: WebSocketMessage):
        """Send message to all connections for a session"""
        if session_id not in self.session_connections:
            # Queue message for when connection is established
            await self._queue_message(session_id, message.dict())
            return
        
        connection_ids = list(self.session_connections[session_id])
        disconnected = []
        
        for connection_id in connection_ids:
            if connection_id not in self.connections:
                disconnected.append(connection_id)
                continue
            
            connection = self.connections[connection_id]
            if not connection.is_active:
                disconnected.append(connection_id)
                continue
            
            try:
                await connection.send_message(message)
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Cleanup disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def send_streaming_chunk(self, session_id: str, chunk: StreamingChunk):
        """Send streaming response chunk to session"""
        message = WebSocketMessage(
            type="streaming_chunk",
            session_id=session_id,
            data=chunk.dict()
        )
        await self.send_to_session(session_id, message)
    
    async def send_model_selection_update(self, session_id: str, update: ModelSelectionUpdate):
        """Send model selection progress update"""
        message = WebSocketMessage(
            type="model_selection_update",
            session_id=session_id,
            data=update.dict()
        )
        await self.send_to_session(session_id, message)
    
    async def send_safety_alert(self, session_id: str, alert: SafetyAlert):
        """Send safety alert to session"""
        message = WebSocketMessage(
            type="safety_alert",
            session_id=session_id,
            data=alert.dict()
        )
        await self.send_to_session(session_id, message)
        
        # Log critical safety alerts
        if alert.level == SafetyLevel.CRITICAL:
            logger.critical(f"Critical safety alert sent to session {session_id}: {alert.message}")
    
    async def broadcast_system_message(self, message_type: str, data: Dict[str, Any]):
        """Broadcast system message to all connected sessions"""
        message = WebSocketMessage(
            type=message_type,
            session_id="system",
            data=data
        )
        
        # Send to all active connections
        disconnected = []
        for connection_id, connection in self.connections.items():
            if not connection.is_active:
                disconnected.append(connection_id)
                continue
            
            try:
                await connection.send_message(message)
            except Exception:
                disconnected.append(connection_id)
        
        # Cleanup disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    def is_rate_limited(self, connection_id: str) -> bool:
        """Check if connection is rate limited"""
        if connection_id not in self.rate_limits:
            return False
        
        now = time.time()
        window_start = now - self.rate_limit_window
        
        # Remove old timestamps
        self.rate_limits[connection_id] = [
            ts for ts in self.rate_limits[connection_id] 
            if ts > window_start
        ]
        
        # Check if over limit
        if len(self.rate_limits[connection_id]) >= self.rate_limit_max:
            return True
        
        # Add current timestamp
        self.rate_limits[connection_id].append(now)
        return False
    
    async def handle_client_message(self, connection_id: str, message: str):
        """Handle incoming message from client"""
        if connection_id not in self.connections:
            return
        
        # Check rate limiting
        if self.is_rate_limited(connection_id):
            logger.warning(f"Rate limit exceeded for connection {connection_id}")
            connection = self.connections[connection_id]
            error_msg = WebSocketMessage(
                type="error",
                session_id=connection.session_id,
                data={"error": "rate_limit_exceeded", "message": "Too many messages"}
            )
            await connection.send_message(error_msg)
            return
        
        try:
            # Parse client message
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                # Handle client ping
                connection = self.connections[connection_id]
                connection.last_ping = datetime.now()
                
                pong_msg = WebSocketMessage(
                    type="pong",
                    session_id=connection.session_id,
                    data={"timestamp": datetime.now().isoformat()}
                )
                await connection.send_message(pong_msg)
            
            elif message_type == "subscribe":
                # Handle subscription to specific event types
                await self._handle_subscription(connection_id, data)
            
            else:
                logger.warning(f"Unknown client message type: {message_type}")
        
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Invalid client message from {connection_id}: {e}")
            connection = self.connections[connection_id]
            error_msg = WebSocketMessage(
                type="error",
                session_id=connection.session_id,
                data={"error": "invalid_message", "message": "Invalid message format"}
            )
            await connection.send_message(error_msg)
    
    async def _queue_message(self, session_id: str, message_data: Dict[str, Any]):
        """Queue message for offline session"""
        if session_id not in self.message_queue:
            self.message_queue[session_id] = []
        
        # Add to queue
        self.message_queue[session_id].append(message_data)
        
        # Trim queue if too large
        if len(self.message_queue[session_id]) > self.max_queue_size:
            self.message_queue[session_id] = self.message_queue[session_id][-self.max_queue_size:]
        
        logger.debug(f"Queued message for offline session {session_id}")
    
    async def _send_queued_messages(self, session_id: str, connection: WebSocketConnection):
        """Send queued messages to newly connected session"""
        if session_id not in self.message_queue:
            return
        
        messages = self.message_queue[session_id]
        if not messages:
            return
        
        logger.info(f"Sending {len(messages)} queued messages to session {session_id}")
        
        for message_data in messages:
            try:
                message = WebSocketMessage(**message_data)
                await connection.send_message(message)
            except Exception as e:
                logger.error(f"Failed to send queued message: {e}")
        
        # Clear queue after sending
        del self.message_queue[session_id]
    
    async def _handle_subscription(self, connection_id: str, data: Dict[str, Any]):
        """Handle client subscription requests"""
        # Implementation for event subscriptions
        # (e.g., subscribe to specific model updates, safety alerts, etc.)
        pass
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Started WebSocket background tasks")
    
    async def _stop_background_tasks(self):
        """Stop background maintenance tasks"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        
        logger.info("Stopped WebSocket background tasks")
    
    async def _heartbeat_loop(self):
        """Background task for sending heartbeat pings"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send ping to all connections
                disconnected = []
                for connection_id, connection in self.connections.items():
                    if not connection.is_active or connection.is_stale():
                        disconnected.append(connection_id)
                        continue
                    
                    try:
                        await connection.ping()
                    except Exception:
                        disconnected.append(connection_id)
                
                # Cleanup disconnected connections
                for connection_id in disconnected:
                    self.disconnect(connection_id)
                
                if self.connections:
                    logger.debug(f"Heartbeat sent to {len(self.connections)} connections")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task for periodic cleanup"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup stale connections
                stale_connections = [
                    conn_id for conn_id, conn in self.connections.items()
                    if conn.is_stale()
                ]
                
                for connection_id in stale_connections:
                    self.disconnect(connection_id)
                
                # Cleanup old rate limit data
                cutoff = time.time() - self.rate_limit_window
                for connection_id in list(self.rate_limits.keys()):
                    if connection_id not in self.connections:
                        del self.rate_limits[connection_id]
                    else:
                        self.rate_limits[connection_id] = [
                            ts for ts in self.rate_limits[connection_id]
                            if ts > cutoff
                        ]
                
                # Cleanup old message queues
                old_queues = [
                    session_id for session_id, messages in self.message_queue.items()
                    if not messages or (datetime.now() - datetime.fromisoformat(messages[-1].get('timestamp', '1970-01-01T00:00:00'))).days > 1
                ]
                
                for session_id in old_queues:
                    del self.message_queue[session_id]
                
                if stale_connections or old_queues:
                    logger.info(f"Cleanup: removed {len(stale_connections)} stale connections, {len(old_queues)} old message queues")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            "total_connections": len(self.connections),
            "active_sessions": len(self.session_connections),
            "queued_messages": sum(len(msgs) for msgs in self.message_queue.values()),
            "connections_by_session": {
                session_id: len(conn_ids) 
                for session_id, conn_ids in self.session_connections.items()
            },
            "oldest_connection": min(
                (conn.connected_at for conn in self.connections.values()),
                default=None
            ),
            "total_messages_sent": sum(conn.message_count for conn in self.connections.values())
        }
    
    async def shutdown(self):
        """Gracefully shutdown WebSocket manager"""
        logger.info("Shutting down WebSocket manager...")
        
        # Stop background tasks
        await self._stop_background_tasks()
        
        # Close all connections
        for connection_id, connection in list(self.connections.items()):
            try:
                await connection.websocket.close(code=1001, reason="Server shutdown")
            except Exception:
                pass
            self.disconnect(connection_id)
        
        # Clear all data
        self.connections.clear()
        self.session_connections.clear()
        self.rate_limits.clear()
        self.message_queue.clear()
        
        logger.info("WebSocket manager shutdown complete")