#!/usr/bin/env python3
"""Quick API test to verify chat server is working."""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_chat():
    """Test basic chat functionality."""
    
    # Test 1: Initial message
    print("\n🧪 Test 1: Initial message (model selection)")
    start = time.time()
    
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": "I'm feeling anxious about a presentation tomorrow."},
        headers={"Content-Type": "application/json"}
    )
    
    elapsed = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success in {elapsed:.2f}s")
        print(f"   Model: {data.get('selected_model', 'unknown')}")
        print(f"   Confidence: {data.get('confidence', 0):.1f}%")
        print(f"   Session ID: {data.get('session_id', 'none')}")
        session_id = data.get('session_id')
    else:
        print(f"❌ Failed with status {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    # Test 2: Continuation
    print("\n🧪 Test 2: Continuation (should be faster)")
    time.sleep(2)
    
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "message": "What breathing techniques can help?",
            "session_id": session_id
        },
        headers={"Content-Type": "application/json"}
    )
    
    elapsed = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success in {elapsed:.2f}s")
        print(f"   Model: {data.get('selected_model', 'unknown')}")
        if elapsed < 10:
            print(f"   ⚡ Fast continuation confirmed!")
    else:
        print(f"❌ Failed with status {response.status_code}")
    
    print("\n✨ API test complete!")

if __name__ == "__main__":
    print("🔍 Testing Chat Server API...")
    print(f"Server: {BASE_URL}")
    
    try:
        test_chat()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the chat server is running: python chat_server.py")