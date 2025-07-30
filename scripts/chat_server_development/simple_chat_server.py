#!/usr/bin/env python3
"""
Simple Chat Server for Testing
==============================

Minimal server to test the simple chat interface.
Serves the simple_chat.html and simple_chat.js files.
"""

import sys
from pathlib import Path
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Import the main server app for API routes
from simple_server import app as main_app

# Create new app
app = FastAPI(
    title="Simple Mental Health Chat",
    description="Simplified chat interface for testing",
    version="1.0.0"
)

# Mount all API routes from main server
app.mount("/api", main_app, name="api")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the simple chat interface"""
    html_file = Path(__file__).parent / "simple_chat.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    else:
        return HTMLResponse(content="<h1>simple_chat.html not found</h1>", status_code=404)

@app.get("/simple_chat.js")
async def chat_js():
    """Serve the JavaScript file"""
    js_file = Path(__file__).parent / "simple_chat.js"
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    else:
        return FileResponse("", status_code=404)

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "message": "Simple chat server running"}

if __name__ == "__main__":
    print("🧠 SIMPLE MENTAL HEALTH CHAT SERVER")
    print("=" * 50)
    print("✅ Features:")
    print("   • Dark theme hardcoded (no toggle)")
    print("   • Message bubbles (user right, AI left)")
    print("   • First message triggers model selection")
    print("   • All subsequent messages go to selected model")
    print("   • Clear visual indication of selected model")
    print("   • Working 'New Chat' button")
    print("   • No page-breaking buttons - pure JavaScript")
    print()
    print("🚀 Server URLs:")
    print("   📱 Simple Chat: http://localhost:8000")
    print("   📊 Health Check: http://localhost:8000/health")
    print("   🔧 API Status: http://localhost:8000/api/status")
    print("=" * 50)
    print("🎯 Test Flow:")
    print("   1. Open http://localhost:8000")
    print("   2. Send: 'I'm feeling anxious' → Model selection")
    print("   3. Send: 'What can help?' → Same model continues")
    print("   4. Click 'New Chat' → Reset and start fresh")
    print("=" * 50)
    
    uvicorn.run(
        "simple_chat_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )