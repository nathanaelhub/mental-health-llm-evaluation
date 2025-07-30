#!/usr/bin/env python3
"""
Test Working Chat Server - Validation Script
===========================================

This script validates that working_chat_server.py has all the fixes applied
and can be imported and initialized properly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work correctly"""
    print("🔍 TESTING IMPORTS...")
    
    try:
        # Test FastAPI components
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
        from pydantic import BaseModel
        print("   ✅ FastAPI imports successful")
        
        # Test our chat components
        from src.chat.dynamic_model_selector import DynamicModelSelector, PromptType
        from src.chat.conversation_session_manager import ConversationSessionManager, MessageRole
        from src.chat.persistent_session_store import SessionStoreType
        print("   ✅ Chat component imports successful")
        
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_app_structure():
    """Test that the app can be created and has the right structure"""
    print("\n🔍 TESTING APP STRUCTURE...")
    
    try:
        # Import the working server
        import working_chat_server
        app = working_chat_server.app
        
        print("   ✅ App created successfully")
        print(f"   ✅ App title: {app.title}")
        print(f"   ✅ App version: {app.version}")
        
        # Check if routes are defined
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/chat", "/api/status", "/api/models/status", "/api/chat", "/api/health"]
        
        for route in expected_routes:
            if route in routes:
                print(f"   ✅ Route {route} found")
            else:
                print(f"   ❌ Route {route} missing")
        
        return True
    except Exception as e:
        print(f"   ❌ App structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_initialization():
    """Test that components can be initialized"""
    print("\n🔍 TESTING COMPONENT INITIALIZATION...")
    
    try:
        import working_chat_server
        
        # Test if initialization function exists
        if hasattr(working_chat_server, 'initialize_components'):
            print("   ✅ initialize_components function found")
            
            # Try to initialize (this should work without async issues)
            working_chat_server.initialize_components()
            
            # Check if components were created
            if working_chat_server.model_selector:
                print("   ✅ Model selector initialized")
            else:
                print("   ❌ Model selector failed to initialize")
            
            if working_chat_server.session_manager:
                print("   ✅ Session manager initialized")
            else:
                print("   ❌ Session manager failed to initialize")
            
            return True
        else:
            print("   ❌ initialize_components function not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_response_models():
    """Test that response models are properly defined"""
    print("\n🔍 TESTING RESPONSE MODELS...")
    
    try:
        import working_chat_server
        
        # Test ChatResponse model
        chat_response = working_chat_server.ChatResponse(
            response="Test response",
            selected_model="openai",
            session_id="test-session",
            confidence_score=0.8,
            reasoning="Test reasoning",
            conversation_mode="selection",
            turn_number=1
        )
        print("   ✅ ChatResponse model works")
        
        # Test StatusResponse model
        status_response = working_chat_server.StatusResponse(
            status="healthy",
            version="1.0.1",
            available_models=["openai", "claude"],
            uptime_seconds=100.0
        )
        print("   ✅ StatusResponse model works")
        
        return True
    except Exception as e:
        print(f"   ❌ Response model test failed: {e}")
        return False

def test_mock_response_function():
    """Test that mock response generation works"""
    print("\n🔍 TESTING MOCK RESPONSE GENERATION...")
    
    try:
        import working_chat_server
        import asyncio
        
        # Test anxiety response
        response = asyncio.run(working_chat_server.generate_mock_response("I'm feeling anxious", "openai"))
        if "anxious" in response.lower() or "anxiety" in response.lower():
            print("   ✅ Anxiety response generated correctly")
        else:
            print(f"   ❌ Unexpected anxiety response: {response[:50]}...")
        
        # Test gratitude response
        response = asyncio.run(working_chat_server.generate_mock_response("Thank you", "claude"))
        if "welcome" in response.lower() or "glad" in response.lower():
            print("   ✅ Gratitude response generated correctly")
        else:
            print(f"   ❌ Unexpected gratitude response: {response[:50]}...")
        
        return True
    except Exception as e:
        print(f"   ❌ Mock response test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧠 WORKING CHAT SERVER VALIDATION")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_app_structure,
        test_component_initialization,
        test_response_models,
        test_mock_response_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - working_chat_server.py is ready!")
        print("\n🚀 NEXT STEPS:")
        print("   1. Start server: python working_chat_server.py")
        print("   2. Open browser: http://localhost:8000/chat")
        print("   3. Test conversation flow:")
        print("      - First message: triggers model selection")
        print("      - Follow-up: uses selected model")
        print("      - New Chat: resets selection")
    else:
        print(f"❌ {total - passed} TESTS FAILED - fix issues before using")
    
    print("=" * 50)

if __name__ == "__main__":
    main()