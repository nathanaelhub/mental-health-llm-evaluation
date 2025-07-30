#!/usr/bin/env python3
"""
Test Chat API Endpoints
Simple test script to verify chat API functionality and determine correct request format.
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8001"  # Test simple server first

def test_endpoint(endpoint: str, payload: Dict[str, Any], description: str):
    """Test a single API endpoint with given payload."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Endpoint: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 60)
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
            try:
                json_response = response.json()
                print(f"Response: {json.dumps(json_response, indent=2)}")
                return True, json_response
            except json.JSONDecodeError:
                print(f"Response Text: {response.text}")
                return True, response.text
        else:
            print("❌ FAILED")
            print(f"Error: {response.text}")
            return False, response.text
            
    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR: {e}")
        return False, str(e)

def test_server_status():
    """Test if server is running."""
    print("Testing server connectivity...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"✅ Server is running (Status: {response.status_code})")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        return False

def main():
    print("🔍 Chat API Testing Script")
    print("=" * 60)
    
    # Test server connectivity first
    if not test_server_status():
        print("\n❌ Cannot connect to server. Make sure it's running on localhost:8000")
        return
    
    # Test payloads to try
    test_cases = [
        {
            "endpoint": "/api/chat",
            "payload": {
                "message": "Hello, I feel anxious",
                "session_id": "test-123",
                "user_id": "test-user"
            },
            "description": "Basic chat request with minimal fields"
        },
        {
            "endpoint": "/api/chat",
            "payload": {
                "message": "Hello, I feel anxious",
                "session_id": "test-123",
                "user_id": "test-user",
                "conversation_mode": "selection"
            },
            "description": "Chat request with conversation_mode"
        },
        {
            "endpoint": "/api/chat",
            "payload": {
                "message": "Hello, I feel anxious",
                "session_id": "test-123",
                "user_id": "test-user",
                "selected_model": "openai"
            },
            "description": "Chat request with selected_model"
        },
        {
            "endpoint": "/api/chat",
            "payload": {
                "message": "Hello, I feel anxious",
                "session_id": "test-123",
                "user_id": "test-user",
                "conversation_mode": "selection",
                "force_reselection": False
            },
            "description": "Chat request with all conversation fields"
        },
        {
            "endpoint": "/api/chat",
            "payload": {
                "message": "This is a follow-up message",
                "session_id": "test-123",
                "user_id": "test-user",
                "conversation_mode": "continuation"
            },
            "description": "Follow-up message (continuation mode)"
        }
    ]
    
    # Test other endpoints too
    other_endpoints = [
        {
            "endpoint": "/api/status",
            "payload": {},
            "description": "Server status endpoint"
        },
        {
            "endpoint": "/api/models/status",
            "payload": {},
            "description": "Models status endpoint"
        }
    ]
    
    successful_requests = []
    failed_requests = []
    
    # Test chat endpoints
    for test_case in test_cases:
        success, response = test_endpoint(
            test_case["endpoint"],
            test_case["payload"],
            test_case["description"]
        )
        
        if success:
            successful_requests.append({
                "test": test_case["description"],
                "payload": test_case["payload"],
                "response": response
            })
        else:
            failed_requests.append({
                "test": test_case["description"],
                "payload": test_case["payload"],
                "error": response
            })
        
        time.sleep(0.5)  # Brief pause between requests
    
    # Test other endpoints with GET
    for test_case in other_endpoints:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['description']}")
        print(f"Endpoint: GET {test_case['endpoint']}")
        print("-" * 60)
        
        try:
            response = requests.get(f"{BASE_URL}{test_case['endpoint']}", timeout=10)
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ SUCCESS")
                try:
                    json_response = response.json()
                    print(f"Response: {json.dumps(json_response, indent=2)}")
                    successful_requests.append({
                        "test": test_case["description"],
                        "payload": "GET request",
                        "response": json_response
                    })
                except json.JSONDecodeError:
                    print(f"Response Text: {response.text}")
            else:
                print("❌ FAILED")
                print(f"Error: {response.text}")
                failed_requests.append({
                    "test": test_case["description"],
                    "payload": "GET request",
                    "error": response.text
                })
                
        except requests.exceptions.RequestException as e:
            print(f"❌ CONNECTION ERROR: {e}")
            failed_requests.append({
                "test": test_case["description"],
                "payload": "GET request",
                "error": str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Successful requests: {len(successful_requests)}")
    print(f"❌ Failed requests: {len(failed_requests)}")
    
    if successful_requests:
        print("\n✅ WORKING REQUEST FORMATS:")
        for req in successful_requests:
            print(f"  • {req['test']}")
            if isinstance(req['payload'], dict):
                print(f"    Payload: {json.dumps(req['payload'], indent=6)}")
    
    if failed_requests:
        print("\n❌ FAILED REQUESTS:")
        for req in failed_requests:
            print(f"  • {req['test']}")
            print(f"    Error: {req['error'][:100]}...")
    
    print("\n" + "=" * 60)
    print("💡 Next steps:")
    if successful_requests:
        print("1. Use the working request format(s) above")
        print("2. Test with different messages and session IDs")
        print("3. Verify conversation flow works")
    else:
        print("1. Check if server is running: python chat_server.py")
        print("2. Check server logs for errors")
        print("3. Verify API routes are mounted correctly")

if __name__ == "__main__":
    main()