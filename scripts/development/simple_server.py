#!/usr/bin/env python3
"""
Simple Mental Health Chat Server
A minimal FastAPI server for demonstrating the chat system functionality.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import time
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import our components
from src.chat.dynamic_model_selector import DynamicModelSelector, PromptType
from src.chat.conversation_session_manager import ConversationSessionManager, MessageRole
from src.chat.persistent_session_store import SessionStoreType

# Global components - will be initialized at startup
model_selector = None
session_manager = None
server_start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_selector, session_manager, server_start_time
    
    print("🚀 Mental Health Chat Server starting up...")
    server_start_time = time.time()
    
    # Initialize model selector
    try:
        model_selector = DynamicModelSelector(models_config)
        print("✅ Model selector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize model selector: {e}")
        model_selector = None

    # Initialize session manager with proper async context
    try:
        session_manager = ConversationSessionManager(
            store_type=SessionStoreType.MEMORY,  # Use in-memory for demo
            enable_safety_monitoring=True,
            enable_audit_trail=True
        )
        print("✅ Session manager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize session manager: {e}")
        session_manager = None
    
    yield
    
    # Shutdown
    print("🛑 Server shutting down...")

app = FastAPI(
    title="Mental Health LLM Chat System",
    description="Dynamic model selection chat system for mental health support",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")

# Templates
templates = Jinja2Templates(directory="src/ui/templates")

# Basic request/response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: str = "anonymous"
    force_reselection: bool = False

class ChatResponse(BaseModel):
    response: str
    selected_model: str
    session_id: str
    confidence_score: float
    reasoning: str
    is_new_session: bool = False
    turn_count: int = 1
    conversation_mode: str = "continuation"  # "selection" or "continuation"
    turn_number: int = 1
    can_reset: bool = True

class StatusResponse(BaseModel):
    status: str
    version: str
    available_models: list
    uptime_seconds: float

class ModelStatusResponse(BaseModel):
    models: Dict[str, Dict[str, Any]]
    total_available: int

# Configuration - placed before lifespan function
models_config = {
    'models': {
        'openai': {'enabled': True, 'cost_per_token': 0.0001, 'model_name': 'gpt-4'},
        'claude': {'enabled': True, 'cost_per_token': 0.00015, 'model_name': 'claude-3'},
        'deepseek': {'enabled': True, 'cost_per_token': 0.00005, 'model_name': 'deepseek/deepseek-r1-0528-qwen3-8b'},
        'gemma': {'enabled': True, 'cost_per_token': 0.00003, 'model_name': 'google/gemma-3-12b'}
    },
    'default_model': 'openai',
    'selection_timeout': 40.0,  # Increased timeout for slow local models
    'similarity_threshold': 0.9
}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mental Health LLM Chat</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .chat-form { margin: 20px 0; }
            .chat-input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0; }
            .chat-button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .chat-button:hover { background: #2980b9; }
            .links { margin: 20px 0; }
            .links a { display: inline-block; margin: 10px; padding: 10px; background: #ecf0f1; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Mental Health LLM Chat System</h1>
            
            <div class="status">
                <strong>✅ Server Status:</strong> Running<br>
                <strong>🤖 AI Models:</strong> Dynamic selection enabled<br>
                <strong>🔒 Safety:</strong> Crisis detection active
            </div>
            
            <div class="chat-form">
                <h3>Quick Test Chat</h3>
                <input type="text" id="messageInput" class="chat-input" placeholder="Type your message here..." value="I'm feeling anxious about work">
                <br>
                <button class="chat-button" onclick="sendMessage()">Send Message</button>
                <div id="response" style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; display: none;"></div>
            </div>
            
            <div class="links">
                <h3>API Endpoints</h3>
                <a href="/api/status">📊 System Status</a>
                <a href="/api/models/status">🤖 Model Status</a>
                <a href="/docs">📖 API Documentation</a>
            </div>
        </div>
        
        <script>
            async function sendMessage() {
                const message = document.getElementById('messageInput').value;
                const responseDiv = document.getElementById('response');
                
                responseDiv.style.display = 'block';
                responseDiv.innerHTML = '⏳ Processing...';
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: 'web-demo',
                            user_id: 'demo-user'
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        responseDiv.innerHTML = `
                            <strong>🤖 Response:</strong> ${data.response}<br><br>
                            <strong>📋 Selected Model:</strong> ${data.selected_model}<br>
                            <strong>🎯 Confidence:</strong> ${(data.confidence_score * 100).toFixed(1)}%<br>
                            <strong>💭 Reasoning:</strong> ${data.reasoning}
                        `;
                    } else {
                        responseDiv.innerHTML = `<strong>❌ Error:</strong> ${data.detail || 'Unknown error'}`;
                    }
                } catch (error) {
                    responseDiv.innerHTML = `<strong>❌ Network Error:</strong> ${error.message}`;
                }
            }
            
            // Allow Enter key to send message
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/api/status")
async def get_status():
    """Get system status"""
    import time
    
    uptime = time.time() - server_start_time if server_start_time else 0
    
    return StatusResponse(
        status="healthy",
        version="1.0.0",
        available_models=["openai", "claude", "deepseek", "gemma"] if model_selector else [],
        uptime_seconds=uptime
    )

@app.get("/api/models/status")
async def get_models_status():
    """Get model availability status"""
    if not model_selector:
        return ModelStatusResponse(
            models={},
            total_available=0
        )
    
    models = {
        "openai": {
            "enabled": True,
            "status": "available",
            "cost_per_token": 0.0001,
            "model_name": "gpt-4",
            "specialties": ["general_support", "crisis", "anxiety", "depression"]
        },
        "claude": {
            "enabled": True,
            "status": "available",
            "cost_per_token": 0.00015,
            "model_name": "claude-3",
            "specialties": ["empathy", "therapeutic", "crisis", "trauma"]
        },
        "deepseek": {
            "enabled": True,
            "status": "available",
            "cost_per_token": 0.00005,
            "model_name": "deepseek/deepseek-r1-0528-qwen3-8b",
            "specialties": ["information_seeking", "general_support", "analysis"]
        },
        "gemma": {
            "enabled": True,
            "status": "available",
            "cost_per_token": 0.00003,
            "model_name": "google/gemma-3-12b",
            "specialties": ["general_support", "relationship", "wellness"]
        }
    }
    
    return ModelStatusResponse(
        models=models,
        total_available=len(models)
    )

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with proper conversation continuation support
    
    Logic:
    1. On first message (no session or no selected_model): Run full model selection
    2. On subsequent messages (selected_model exists): Skip selection, use stored model
    """
    if not model_selector or not session_manager:
        raise HTTPException(status_code=503, detail="Chat services not available")
    
    try:
        # Get existing session if provided
        session = None
        if request.session_id:
            session = await session_manager.get_session(request.session_id)
        
        # Determine if we need to run model selection
        needs_selection = (
            not session or  # No session exists
            not hasattr(session, 'selected_model') or  # No selected model
            not session.selected_model or  # Empty selected model
            request.force_reselection  # Forced reselection
        )
        
        if needs_selection:
            # FIRST MESSAGE: Run full model selection
            print(f"🔍 First message - running model selection for: '{request.message[:50]}...'")
            
            # Run model selection
            selection = await model_selector.select_best_model(prompt=request.message)
            
            if not selection:
                raise HTTPException(status_code=500, detail="Model selection failed")
            
            selected_model = selection.selected_model_id
            confidence = selection.confidence_score
            reasoning = selection.selection_reasoning
            
            # Create new session with selected model
            session = await session_manager.create_session(
                user_id=request.user_id,
                selected_model=selected_model,
                initial_message=request.message,
                metadata={
                    'selection_result': selection.to_dict(),
                    'selection_confidence': confidence,
                    'selection_reasoning': reasoning
                }
            )
            
            print(f"✅ Selected {selected_model.upper()} with {confidence:.1%} confidence")
            
            # Add user message to session
            await session_manager.add_message(
                session.session_id,
                MessageRole.USER,
                request.message
            )
            
            # Generate response with selected model
            response_text = await generate_mock_response(request.message, selected_model)
            
            # Add assistant response to session
            await session_manager.add_message(
                session.session_id,
                MessageRole.ASSISTANT,
                response_text,
                model_used=selected_model
            )
            
            return ChatResponse(
                response=response_text,
                selected_model=selected_model,
                session_id=session.session_id,
                confidence_score=confidence,
                reasoning=f"Selected {selected_model.upper()} for this conversation. {reasoning[:100]}...",
                is_new_session=True,
                turn_count=1,
                conversation_mode="selection",
                turn_number=1,
                can_reset=True
            )
        
        else:
            # SUBSEQUENT MESSAGE: Use stored model directly
            selected_model = session.selected_model
            print(f"💬 Continuing conversation with {selected_model.upper()} in session {session.session_id[:8]}")
            
            # Add user message to session
            await session_manager.add_message(
                session.session_id,
                MessageRole.USER,
                request.message
            )
            
            # Generate response with the stored model (maintaining context)
            response_text = await generate_mock_response(request.message, selected_model)
            
            # Add assistant response to session
            await session_manager.add_message(
                session.session_id,
                MessageRole.ASSISTANT,
                response_text,
                model_used=selected_model
            )
            
            # Calculate turn number (user messages count)
            turn_number = len([msg for msg in session.conversation_history if msg.role == MessageRole.USER])
            
            return ChatResponse(
                response=response_text,
                selected_model=selected_model,
                session_id=session.session_id,
                confidence_score=session.metadata.get('selection_confidence', 0.6),
                reasoning=f"Continuing conversation with {selected_model.upper()} (turn {turn_number})",
                is_new_session=False,
                turn_count=turn_number,
                conversation_mode="continuation",
                turn_number=turn_number,
                can_reset=True
            )
        
    except Exception as e:
        print(f"❌ Chat processing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


async def generate_mock_response(message: str, model_name: str) -> str:
    """
    Generate contextual mock responses based on message and model
    Maintains conversation context for selected model
    """
    message_lower = message.lower()
    
    # Crisis responses (highest priority)
    if any(word in message_lower for word in ['suicide', 'kill myself', 'end it all', 'want to die', 'hurt myself']):
        responses = {
            'openai': "I'm very concerned about what you've shared. Your safety is the most important thing right now. Please consider reaching out to the 988 Suicide & Crisis Lifeline (available 24/7). You don't have to face this alone - there are people who want to help you through this difficult time.",
            'claude': "I hear you're in tremendous pain right now, and I'm deeply concerned for your wellbeing. Please know that there are people trained to help you through this crisis. The 988 Lifeline is available 24/7 at 988. Your life has value, and this pain can be addressed with proper support.",
            'deepseek': "CRISIS ALERT: Immediate professional intervention required. Contact 988 Suicide & Crisis Lifeline (24/7) or emergency services (911). Risk factors identified. Professional assessment needed immediately. You matter and help is available.",
            'gemma': "I'm very worried about you and what you're going through. This sounds like you need immediate support. Please contact the 988 Lifeline right now or go to your nearest emergency room. You deserve care and support, and there are people who want to help you through this."
        }
    
    # Anxiety responses
    elif any(word in message_lower for word in ['anxious', 'anxiety', 'worry', 'panic', 'stress', 'overwhelmed']):
        if any(followup in message_lower for followup in ['help', 'do', 'techniques', 'strategies', 'cope']):
            # Follow-up about coping strategies
            responses = {
                'openai': "Here are some evidence-based techniques that can help with anxiety: 1) Box breathing (4-4-4-4 pattern), 2) Progressive muscle relaxation, 3) Grounding with the 5-4-3-2-1 technique, and 4) Mindfulness meditation. Which of these sounds most manageable for you to try right now?",
                'claude': "I'm glad you're asking for practical help. Let's start with something immediate: try the 5-4-3-2-1 grounding technique - name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste. This can help anchor you in the present moment. How does that feel?",
                'deepseek': "Evidence-based anxiety management techniques: Controlled breathing (4-7-8 method), progressive muscle relaxation, cognitive restructuring, exposure therapy principles. Immediate relief: focus on slow exhales, longer than inhales. Which technique would you like detailed instructions for?",
                'gemma': "There are several gentle techniques we can try together. My favorite is the 'soft belly' breathing - place one hand on your chest, one on your belly, and breathe so only the bottom hand moves. It's amazing how this can calm your nervous system. Would you like to try this with me?"
            }
        else:
            # Initial anxiety response
            responses = {
                'openai': "I understand you're feeling anxious - that's a very common experience, and you're not alone. Anxiety can feel overwhelming, but there are effective ways to manage it. What's contributing most to your anxiety right now? Understanding the source can help us work through it together.",
                'claude': "Anxiety can feel like it's taking over everything, can't it? I want you to know that what you're experiencing is valid, and it's actually quite brave of you to reach out. Let's work through this together. Can you tell me what's making you feel most anxious today?",
                'deepseek': "Anxiety symptoms detected. This is a common stress response that can be effectively managed. Current physiological state likely includes elevated heart rate, muscle tension, racing thoughts. What specific situations or thoughts are triggering your anxiety today?",
                'gemma': "Oh, I hear you. Anxiety can be so overwhelming and exhausting. You're not alone in feeling this way, and I'm really glad you reached out. It takes courage to acknowledge when we're struggling. What's been weighing on your mind the most lately?"
            }
    
    # Depression responses
    elif any(word in message_lower for word in ['depressed', 'depression', 'sad', 'hopeless', 'empty', 'numb']):
        responses = {
            'openai': "I hear that you're struggling with depression, and I want you to know that your feelings are completely valid. Depression can make everything feel overwhelming and hopeless, but it's important to remember that it's treatable. You've taken a brave step by reaching out. Have you been able to connect with any mental health professionals?",
            'claude': "Thank you for trusting me with something so deeply personal. Depression can make the world feel colorless and heavy, and I want you to know that what you're experiencing is real. Even in this darkness, there's hope. What's one small thing that used to bring you even a tiny bit of joy?",
            'deepseek': "Depression indicators identified. This is a treatable medical condition affecting neurotransmitter systems. Treatment modalities include: cognitive behavioral therapy (60-70% efficacy), medication options, lifestyle interventions, support systems. Have you considered professional evaluation for treatment planning?",
            'gemma': "I'm so sorry you're going through such a difficult time. Depression can make it feel like you're carrying the weight of the world, and it's exhausting. Please know that what you're feeling right now isn't permanent, even though it might feel that way. You're incredibly brave for reaching out. What does support look like for you?"
        }
    
    # Thank you / gratitude responses
    elif any(word in message_lower for word in ['thank', 'thanks', 'grateful', 'appreciate', 'helpful']):
        responses = {
            'openai': "You're very welcome. I'm genuinely glad I could be helpful to you. Remember, seeking support and taking care of your mental health is an ongoing process, and you're doing great by being proactive about it. How are you feeling about the strategies we've discussed?",
            'claude': "It means so much to hear that this has been helpful for you. Your willingness to engage and try new approaches shows real strength and commitment to your wellbeing. I'm honored to be part of your support system. How are you planning to use what we've talked about?",
            'deepseek': "Positive feedback received. Treatment engagement and therapeutic alliance indicators are favorable. Continued application of discussed strategies recommended for optimal outcomes. Regular self-monitoring and follow-up beneficial for sustained progress.",
            'gemma': "Aw, you're so welcome! It really warms my heart to know that our conversation has been helpful. You've been doing the hard work - I'm just here to support you along the way. You should feel proud of yourself for taking these positive steps."
        }
    
    # Follow-up or continuation responses
    elif any(word in message_lower for word in ['what else', 'what about', 'also', 'more', 'another']):
        responses = {
            'openai': "I'm glad you want to explore this further. Building on what we've discussed, let's look at additional strategies that might be helpful for your situation. What specific area would you like to focus on next?",
            'claude': "I love that you're thinking deeper about this - that shows real commitment to your healing journey. Let's explore some other approaches that might resonate with you. What feels most important to address right now?",
            'deepseek': "Additional intervention strategies available. Expanding treatment modalities: mindfulness-based approaches, behavioral activation, social support enhancement, sleep hygiene optimization. Which domain requires immediate attention?",
            'gemma': "I'm so glad you're curious to learn more! That enthusiasm for taking care of yourself is wonderful to see. There are definitely more tools we can add to your self-care toolkit. What area of your life feels like it needs the most support right now?"
        }
    
    # General/continuation responses
    else:
        responses = {
            'openai': "Thank you for continuing to share with me. I can hear that this is important to you, and I want to make sure I'm providing the most helpful support possible. Can you tell me more about what you're experiencing or what would be most beneficial to explore together?",
            'claude': "I appreciate you opening up further. Your willingness to continue this conversation shows real courage and self-awareness. I'm here to listen and support you through whatever you're facing. What feels most pressing for you right now?",
            'deepseek': "Continued engagement noted. Maintaining therapeutic relationship and active listening protocols. Please elaborate on current concerns, symptoms, or specific support needs for optimal assistance delivery.",
            'gemma': "I'm really glad you're continuing to share with me. It shows how much you care about your wellbeing, and that's something to be proud of. I'm here to support you however I can. What's been on your heart or mind that you'd like to talk through?"
        }
    
    return responses.get(model_name, responses['openai'])

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session.session_id,
            "selected_model": session.selected_model,
            "turn_count": len([msg for msg in session.conversation_history if msg.role == MessageRole.USER]),
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "status": session.status.value,
            "conversation_history": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "model_used": msg.model_used
                }
                for msg in session.conversation_history[-10:]  # Last 10 messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.post("/api/sessions/{session_id}/switch-model")
async def switch_model(session_id: str, new_model: str = None):
    """Switch to a different model for the session"""
    if not session_manager or not model_selector:
        raise HTTPException(status_code=503, detail="Services not available")
    
    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if model is available
        available_models = model_selector.get_available_models()
        if new_model not in available_models:
            raise HTTPException(status_code=400, detail=f"Model {new_model} not available")
        
        # Update session model
        old_model = session.selected_model
        session.selected_model = new_model
        
        # Add metadata about the switch
        session.model_switches.append({
            'from_model': old_model,
            'to_model': new_model,
            'timestamp': time.time(),
            'reason': 'user_request'
        })
        
        return {
            "success": True,
            "old_model": old_model,
            "new_model": new_model,
            "session_id": session_id,
            "message": f"Successfully switched from {old_model} to {new_model}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2025-07-29T15:00:00Z"}

if __name__ == "__main__":
    print("🧠 Starting Mental Health LLM Chat Server...")
    print("📋 Available at: http://localhost:8000")
    print("🔧 API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "simple_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )