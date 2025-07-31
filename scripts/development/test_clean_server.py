#!/usr/bin/env python3
"""
Quick test script to verify the cleaned-up chat server works correctly
"""
import requests
import json
import time

def test_chat_server():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Clean Chat Server")
    print("=" * 40)
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check: {health['status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure to run: python chat_server.py")
        return False
    
    # Test chat endpoint with simple message
    test_message = "I'm feeling anxious about work"
    
    try:
        print(f"\n📤 Sending: '{test_message}'")
        print("⏳ Waiting for REAL model evaluation...")
        
        chat_data = {
            "message": test_message,
            "user_id": "test-user"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/chat", 
            json=chat_data,
            timeout=60  # Allow time for real evaluation
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received in {end_time - start_time:.1f}s")
            print(f"🤖 Selected model: {result['selected_model']}")
            print(f"🎯 Confidence: {result['confidence_score']:.1%}")
            print(f"💬 Response: {result['response'][:100]}...")
            
            if result.get('model_scores'):
                print(f"📊 Model scores: {result['model_scores']}")
            
            return True
        else:
            print(f"❌ Chat request failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>60s)")
        print("💡 This might indicate an issue with model evaluation")
        return False
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_chat_server()
    if success:
        print("\n🎉 Chat server is working correctly!")
        print("🌐 Visit http://localhost:8000/chat to try the UI")
    else:
        print("\n❌ Chat server test failed")
        print("💡 Check server logs for detailed error information")