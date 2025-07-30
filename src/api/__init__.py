"""
Mental Health AI Chat API Package

FastAPI-based backend for the mental health chat system with dynamic model selection,
real-time WebSocket streaming, and comprehensive session management.
"""

from .main import app, ChatAPI
from .models import *
from .websocket import WebSocketManager

__version__ = "1.0.0"

__all__ = [
    "app",
    "ChatAPI", 
    "WebSocketManager"
]