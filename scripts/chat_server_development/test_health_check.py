#!/usr/bin/env python3
"""
Test the new health check and fallback system
"""
import requests
import json
import time

def test_chat_with_health_checks():
    payload = {
        "message": "Hello, I feel anxious",
        "user_id": "test-user",
        "session_id": None
    }
    
    print("Testing chat with health check fallback system...")
    print("This should respond quickly with fallback if no models are available")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    
    try:
        # Use longer timeout to see the full process
        response = requests.post("http://localhost:8000/api/chat", json=payload, timeout=20)
        
        elapsed = time.time() - start_time
        print(f"⏱️ Response time: {elapsed:.2f} seconds")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS - Chat API responded!")
            print(f"Selected model: {data.get('selected_model', 'unknown')}")
            print(f"Response: {data.get('response', 'no response')[:100]}...")
            
            if 'fallback_mode' in data:
                print("✅ Fallback mode activated")
            elif 'model_selection_results' in data:
                results = data['model_selection_results']
                print(f"Available models: {results.get('all_models_evaluated', [])}")
                
            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time  
        print(f"❌ Still timing out after {elapsed:.2f} seconds")
        print("The health check system may not be working properly")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f} seconds: {e}")
        return False

if __name__ == "__main__":
    success = test_chat_with_health_checks()
    if success:
        print("\n🎉 Health check and fallback system is working!")
    else:
        print("\n❌ Health check system needs debugging")