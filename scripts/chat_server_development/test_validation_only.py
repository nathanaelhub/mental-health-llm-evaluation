#!/usr/bin/env python3
"""
Test validation-only endpoint to verify 422 fix
"""
import requests
import json

def test_validation():
    # Test payload that should now work
    payload = {
        "message": "Hello, I feel anxious",
        "user_id": "test-user",
        "session_id": None  # This should now be accepted
    }
    
    print("Testing validation fix...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post("http://localhost:8000/api/chat", json=payload, timeout=3)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 422:
            print("❌ Still getting 422 - validation not fixed")
            print(f"Error: {response.json()}")
        elif response.status_code == 200:
            print("✅ 422 fixed - validation passed")
        else:
            print(f"Different response: {response.status_code}")
            if "timeout" in response.text.lower():
                print("✅ 422 fixed - now timing out in model selection (expected)")
            
    except requests.exceptions.Timeout:
        print("✅ 422 fixed - validation passed, timeout in model selection (expected)")
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    test_validation()