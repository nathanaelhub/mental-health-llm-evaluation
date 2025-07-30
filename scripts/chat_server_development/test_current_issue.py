#!/usr/bin/env python3
"""
Test current issue with chat API
"""
import requests
import json

# Test the exact payload that the UI would send
def test_ui_payload():
    payload = {
        "message": "I'm feeling anxious",
        "session_id": None,  # This is what UI sends on first message
        "user_id": "demo-user",
        "force_reselection": False
    }
    
    print("Testing UI-style payload...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/chat",
            json=payload,
            timeout=20
        )
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS - Chat is working!")
            data = response.json()
            print(f"Selected model: {data.get('selected_model')}")
            print(f"Response: {data.get('response', '')[:100]}...")
        elif response.status_code == 422:
            print("❌ 422 Error - Validation failed")
            print(f"Error: {response.json()}")
        else:
            print(f"❌ Error: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - model selection is running")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_ui_payload()