#!/usr/bin/env python3
"""
Mental Health Chat Server Launcher
==================================

Simplified server launcher that handles startup issues and provides clear status.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🧠 MENTAL HEALTH CHAT SERVER STARTUP")
    print("=" * 50)
    
    try:
        # Import and run the server
        import uvicorn
        
        # Set environment variables for better startup
        os.environ['PYTHONPATH'] = str(Path(__file__).parent)
        
        print("✅ Dependencies loaded successfully")
        print("🚀 Starting server...")
        print("📍 URL: http://localhost:8000")
        print("📋 API docs: http://localhost:8000/docs")
        print("💻 Web interface: http://localhost:8000")
        print("-" * 50)
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start uvicorn server
        uvicorn.run(
            "simple_server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to avoid issues
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install dependencies: pip install fastapi uvicorn jinja2 --break-system-packages")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        print("🔧 Try running: python simple_server.py")

if __name__ == "__main__":
    main()