"""
Pydantic Models for API Request/Response Validation

Comprehensive data models for the mental health chat API with validation,
serialization, and documentation support.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid


class MessageRole(str, Enum):
    """Message role enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SessionStatus(str, Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"
    MIGRATING = "migrating"


class SafetyLevel(str, Enum):
    """Safety assessment levels"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


class PromptType(str, Enum):
    """Prompt classification types"""
    CRISIS = "crisis"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    RELATIONSHIPS = "relationships"
    WORK_STRESS = "work_stress"
    INFORMATION_SEEKING = "information_seeking"
    GENERAL_WELLNESS = "general_wellness"
    COPING_STRATEGIES = "coping_strategies"


# Request Models

class ChatRequest(BaseModel):
    """Main chat request model"""
    session_id: Optional[str] = Field(None, description="Session ID for existing conversation")
    user_id: Optional[str] = Field(None, description="User identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message content")
    streaming: bool = Field(True, description="Enable streaming response")
    model_preference: Optional[str] = Field(None, description="Preferred model override")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user_123",
                "message": "I've been feeling anxious about my upcoming presentation at work.",
                "streaming": True,
                "metadata": {"client_version": "1.0.0"}
            }
        }


class SessionCreateRequest(BaseModel):
    """Request to create a new session"""
    user_id: Optional[str] = Field(None, description="User identifier")
    initial_message: Optional[str] = Field(None, min_length=1, max_length=2000, description="Optional initial message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "initial_message": "Hello, I'm looking for some mental health support.",
                "metadata": {"referral_source": "website", "client_info": "web_app_v1.0"}
            }
        }


class SessionUpdateRequest(BaseModel):
    """Request to update session properties"""
    status: Optional[SessionStatus] = Field(None, description="New session status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class MessageFeedbackRequest(BaseModel):
    """Request to provide feedback on a message"""
    message_id: str = Field(..., description="ID of the message being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, max_length=500, description="Optional feedback text")
    categories: List[str] = Field(default_factory=list, description="Feedback categories")


# Response Models

class ModelSelectionInfo(BaseModel):
    """Information about model selection process"""
    selected_model: str = Field(..., description="Name of selected model")
    selection_reason: str = Field(..., description="Explanation for model selection")
    prompt_classification: PromptType = Field(..., description="Classified prompt type")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Selection confidence")
    alternatives: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative model options")
    selection_time_ms: float = Field(..., description="Time taken for model selection")
    
    class Config:
        schema_extra = {
            "example": {
                "selected_model": "claude",
                "selection_reason": "High empathy and safety scores for anxiety-related content",
                "prompt_classification": "anxiety",
                "confidence_score": 0.92,
                "alternatives": [
                    {"model": "openai", "score": 0.85, "reason": "Good therapeutic understanding"},
                    {"model": "deepseek", "score": 0.78, "reason": "Solid general response quality"}
                ],
                "selection_time_ms": 45.2
            }
        }


class MessageResponse(BaseModel):
    """Response for a single message"""
    message_id: str = Field(..., description="Unique message identifier")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    model_used: Optional[str] = Field(None, description="Model that generated the message")
    token_count: Optional[int] = Field(None, description="Token count for the message")
    safety_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Safety assessment score")
    response_time_ms: Optional[float] = Field(None, description="Response generation time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")


class ChatResponse(BaseModel):
    """Main chat response model"""
    success: bool = Field(..., description="Whether the request was successful")
    session_id: str = Field(..., description="Session identifier")
    message: MessageResponse = Field(..., description="Generated message response")
    model_selection: Optional[ModelSelectionInfo] = Field(None, description="Model selection details")
    session_updated: bool = Field(False, description="Whether session state was updated")
    safety_alert: Optional[Dict[str, Any]] = Field(None, description="Safety alert information if triggered")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": {
                    "message_id": "msg_123",
                    "role": "assistant",
                    "content": "I understand you're feeling anxious about your presentation...",
                    "timestamp": "2025-01-29T10:30:00Z",
                    "model_used": "claude",
                    "token_count": 85,
                    "safety_score": 0.95,
                    "response_time_ms": 1250.5
                },
                "model_selection": {
                    "selected_model": "claude",
                    "selection_reason": "High empathy score for anxiety content",
                    "prompt_classification": "anxiety",
                    "confidence_score": 0.92
                }
            }
        }


class SessionInfo(BaseModel):
    """Session information response"""
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    status: SessionStatus = Field(..., description="Current session status")
    safety_level: SafetyLevel = Field(..., description="Current safety level")
    selected_model: str = Field(..., description="Currently selected model")
    created_at: datetime = Field(..., description="Session creation timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    message_count: int = Field(..., description="Total number of messages")
    total_tokens: int = Field(..., description="Total token count")
    model_switches: int = Field(0, description="Number of model switches")
    crisis_flags: int = Field(0, description="Number of crisis flags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")


class SessionCreateResponse(BaseModel):
    """Response for session creation"""
    success: bool = Field(..., description="Whether session creation was successful")
    session_id: str = Field(..., description="New session identifier")
    session_info: SessionInfo = Field(..., description="Created session information")


class SessionHistoryResponse(BaseModel):
    """Response containing session history"""
    success: bool = Field(..., description="Whether the request was successful")
    session_info: SessionInfo = Field(..., description="Session information")
    messages: List[MessageResponse] = Field(..., description="Message history")
    summary: Optional[str] = Field(None, description="Conversation summary if available")


class ModelStatus(BaseModel):
    """Status of a single model"""
    name: str = Field(..., description="Model name")
    available: bool = Field(..., description="Whether model is available")
    response_time_ms: Optional[float] = Field(None, description="Average response time")
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Success rate percentage")
    last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
    error_message: Optional[str] = Field(None, description="Error message if unavailable")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "claude",
                "available": True,
                "response_time_ms": 1200.5,
                "success_rate": 0.98,
                "last_health_check": "2025-01-29T10:29:00Z",
                "capabilities": ["streaming", "safety_filtering", "multilingual"]
            }
        }


class ModelsStatusResponse(BaseModel):
    """Response for models health status"""
    success: bool = Field(..., description="Whether the health check was successful")
    timestamp: datetime = Field(..., description="Health check timestamp")
    total_models: int = Field(..., description="Total number of models")
    available_models: int = Field(..., description="Number of available models")
    models: List[ModelStatus] = Field(..., description="Individual model statuses")
    overall_health: Literal["healthy", "degraded", "critical"] = Field(..., description="Overall system health")


class SelectionStatistics(BaseModel):
    """Model selection statistics"""
    model_name: str = Field(..., description="Model name")
    selection_count: int = Field(..., description="Number of times selected")
    avg_confidence: float = Field(..., ge=0.0, le=1.0, description="Average selection confidence")
    prompt_types: Dict[PromptType, int] = Field(..., description="Selection count by prompt type")
    avg_response_time_ms: float = Field(..., description="Average response time")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=5.0, description="Average user rating")


class SelectionStatsResponse(BaseModel):
    """Response for model selection statistics"""
    success: bool = Field(..., description="Whether the request was successful")
    period_start: datetime = Field(..., description="Statistics period start")
    period_end: datetime = Field(..., description="Statistics period end")
    total_selections: int = Field(..., description="Total number of model selections")
    statistics: List[SelectionStatistics] = Field(..., description="Per-model statistics")
    prompt_type_distribution: Dict[PromptType, int] = Field(..., description="Distribution by prompt type")


# WebSocket Models

class WebSocketMessage(BaseModel):
    """Base WebSocket message model"""
    type: str = Field(..., description="Message type")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")


class StreamingChunk(BaseModel):
    """Streaming response chunk"""
    chunk_id: int = Field(..., description="Chunk sequence number")
    content: str = Field(..., description="Chunk content")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    model_used: Optional[str] = Field(None, description="Model generating the chunk")


class ModelSelectionUpdate(BaseModel):
    """Model selection progress update"""
    stage: Literal["analyzing", "evaluating", "selecting", "complete"] = Field(..., description="Selection stage")
    progress: float = Field(..., ge=0.0, le=1.0, description="Progress percentage")
    current_model: Optional[str] = Field(None, description="Currently evaluating model")
    preliminary_results: Optional[Dict[str, Any]] = Field(None, description="Preliminary selection results")


class SafetyAlert(BaseModel):
    """Safety alert message"""
    level: SafetyLevel = Field(..., description="Safety alert level")
    message: str = Field(..., description="Alert message")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    resources: List[Dict[str, str]] = Field(default_factory=list, description="Crisis resources")


# Error Models

class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: List[ErrorDetail] = Field(default_factory=list, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "validation_error",
                "message": "Request validation failed",
                "details": [
                    {
                        "code": "field_required",
                        "message": "Message content is required",
                        "field": "message"
                    }
                ],
                "request_id": "req_abc123",
                "timestamp": "2025-01-29T10:30:00Z"
            }
        }


# Search and Analytics Models

class SessionSearchRequest(BaseModel):
    """Request for searching sessions"""
    query: Optional[str] = Field(None, description="Search query")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    status: Optional[SessionStatus] = Field(None, description="Filter by session status")
    safety_level: Optional[SafetyLevel] = Field(None, description="Filter by safety level")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    limit: int = Field(50, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Results offset for pagination")
    include_messages: bool = Field(False, description="Include message content in results")


class SessionSearchResponse(BaseModel):
    """Response for session search"""
    success: bool = Field(..., description="Whether the search was successful")
    total_count: int = Field(..., description="Total number of matching sessions")
    sessions: List[SessionInfo] = Field(..., description="Matching sessions")
    messages: Optional[Dict[str, List[MessageResponse]]] = Field(None, description="Messages by session ID")


# Configuration Models

class APIConfig(BaseModel):
    """API configuration model"""
    title: str = Field("Mental Health AI Chat API", description="API title")
    version: str = Field("1.0.0", description="API version")
    description: str = Field("FastAPI backend for mental health chat with dynamic model selection")
    
    # Model configuration
    available_models: List[str] = Field(default_factory=lambda: ["openai", "claude", "deepseek", "gemma"])
    default_model: str = Field("intelligent-selection", description="Default model selection strategy")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(60, description="Rate limit per minute per session")
    max_message_length: int = Field(2000, description="Maximum message length")
    max_session_duration_hours: int = Field(24, description="Maximum session duration")
    
    # WebSocket configuration
    websocket_heartbeat_interval: int = Field(30, description="WebSocket heartbeat interval in seconds")
    max_websocket_connections: int = Field(1000, description="Maximum concurrent WebSocket connections")
    
    # Security
    enable_cors: bool = Field(True, description="Enable CORS")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins")
    require_auth: bool = Field(False, description="Require authentication")
    
    # Monitoring
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(False, description="Enable distributed tracing")
    log_level: str = Field("INFO", description="Logging level")


# Validation Functions

def validate_session_id(session_id: str) -> str:
    """Validate session ID format"""
    try:
        uuid.UUID(session_id)
        return session_id
    except ValueError:
        raise ValueError("Invalid session ID format")


def validate_message_content(content: str) -> str:
    """Validate and sanitize message content"""
    if not content or not content.strip():
        raise ValueError("Message content cannot be empty")
    
    # Basic sanitization - remove potential script tags
    import re
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
    
    return content.strip()