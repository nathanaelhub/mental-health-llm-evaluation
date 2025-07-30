#!/usr/bin/env python3
"""
Simple Chat Server for Testing
Bypasses model selection to test basic chat functionality
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Simple Chat Test Server")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_mode: Optional[str] = "selection"
    selected_model: Optional[str] = None
    force_reselection: Optional[bool] = False

@app.get("/")
async def root():
    return {"message": "Simple Chat Test Server Running"}

@app.get("/api/status")
async def status():
    return {
        "status": "healthy",
        "version": "test-1.0.0",
        "available_models": ["openai", "claude", "deepseek", "gemma"],
        "note": "Simple test server - bypasses model selection"
    }

@app.post("/api/chat")
async def chat_test(request: ChatRequest):
    """Simple chat endpoint that returns mock responses"""
    
    # Mock model selection result
    mock_models = ["openai", "claude", "deepseek", "gemma"]
    selected_model = request.selected_model or "openai"
    
    # Mock response based on conversation mode
    if request.conversation_mode == "selection" or not request.session_id:
        # First message - simulate model selection
        response = {
            "response": f"Thank you for sharing that with me. I understand you're feeling {request.message.lower()}. I'm here to help.",
            "conversation_mode": "selection",
            "selected_model": selected_model,
            "session_id": request.session_id or f"test-session-{hash(request.message) % 10000}",
            "turn_number": 1,
            "can_reset": True,
            "model_selection_results": {
                "selected_model": selected_model,
                "confidence_score": 0.85,
                "selection_reasoning": f"Selected {selected_model} for general support conversation",
                "all_models_evaluated": mock_models,
                "evaluation_time_ms": 150
            },
            "metrics": {
                "empathy_score": 0.9,
                "therapeutic_score": 0.8,
                "safety_score": 0.95,
                "clarity_score": 0.85
            }
        }
    else:
        # Continuation message
        response = {
            "response": f"I understand. Can you tell me more about how that makes you feel?",
            "conversation_mode": "continuation", 
            "selected_model": selected_model,
            "session_id": request.session_id,
            "turn_number": 2,
            "can_reset": True,
            "continuation_info": {
                "continuing_with_model": selected_model,
                "conversation_flow": "working properly"
            }
        }
    
    print(f"💬 Chat request processed:")
    print(f"   Message: {request.message[:50]}...")
    print(f"   Mode: {request.conversation_mode}")
    print(f"   Model: {selected_model}")
    print(f"   Session: {response['session_id']}")
    
    return JSONResponse(content=response)

if __name__ == "__main__":
    print("🧪 Starting Simple Chat Test Server")
    print("This server bypasses model selection for testing")
    print("Access at: http://localhost:8001")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")