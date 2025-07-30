#!/usr/bin/env python3
"""
Debug 422 Unprocessable Entity Error
Identifies exact field mismatches between frontend and backend
"""

import requests
import json
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000"  # Main chat server

def test_payload(payload: Dict[str, Any], description: str):
    """Test a payload and show detailed 422 error info"""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 70)
    
    try:
        response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=5)
        
        if response.status_code == 200:
            print("✅ SUCCESS - Payload accepted")
            return True
        elif response.status_code == 422:
            print("❌ 422 UNPROCESSABLE ENTITY")
            try:
                error_detail = response.json()
                print("Error details:")
                print(json.dumps(error_detail, indent=2))
                
                # Extract field validation errors
                if 'detail' in error_detail:
                    for error in error_detail['detail']:
                        if 'loc' in error and 'msg' in error:
                            field = " -> ".join(str(x) for x in error['loc'])
                            print(f"  ❌ Field '{field}': {error['msg']}")
                            if 'type' in error:
                                print(f"     Error type: {error['type']}")
                
            except json.JSONDecodeError:
                print(f"Raw error response: {response.text}")
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
        
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ CONNECTION ERROR: {e}")
        return False

def test_server_running():
    """Check if main server is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=3)
        if response.status_code == 200:
            print("✅ Main server is running")
            return True
    except:
        pass
    
    print(f"❌ Main server not running on {BASE_URL}")
    print("Please start: python chat_server.py")
    return False

def main():
    print("🔍 Debugging 422 Unprocessable Entity Error")
    print("=" * 70)
    
    if not test_server_running():
        return
    
    # Test various payload combinations to find the exact issue
    test_cases = [
        # 1. Minimal payload
        {
            "payload": {
                "message": "Hello, I feel anxious"
            },
            "description": "Minimal - message only"
        },
        
        # 2. Add user_id
        {
            "payload": {
                "message": "Hello, I feel anxious",
                "user_id": "test-user"
            },
            "description": "With user_id"
        },
        
        # 3. Add session_id
        {
            "payload": {
                "message": "Hello, I feel anxious",
                "user_id": "test-user",
                "session_id": "test-123"
            },
            "description": "With user_id and session_id"
        },
        
        # 4. Complete payload (what our test API used)
        {
            "payload": {
                "message": "Hello, I feel anxious",
                "user_id": "test-user", 
                "session_id": "test-123",
                "conversation_mode": "selection",
                "selected_model": None,
                "force_reselection": False
            },
            "description": "Complete payload with all fields"
        },
        
        # 5. What frontend might be sending (common field names)
        {
            "payload": {
                "message": "Hello, I feel anxious",
                "userId": "test-user",  # camelCase
                "sessionId": "test-123"  # camelCase
            },
            "description": "Frontend style (camelCase fields)"
        },
        
        # 6. Alternative field names
        {
            "payload": {
                "message": "Hello, I feel anxious",
                "user": "test-user",
                "session": "test-123"
            },
            "description": "Alternative field names"
        },
        
        # 7. Empty strings
        {
            "payload": {
                "message": "Hello, I feel anxious",
                "user_id": "",
                "session_id": ""
            },
            "description": "With empty string fields"
        },
        
        # 8. Null values (should work now)
        {
            "payload": {
                "message": "Hello, I feel anxious",
                "user_id": "test-user",
                "session_id": None
            },
            "description": "With null session_id (fixed)"
        },
        
        # 9. Wrong data types
        {
            "payload": {
                "message": "Hello, I feel anxious",
                "user_id": 123,  # number instead of string
                "session_id": True  # boolean instead of string
            },
            "description": "Wrong data types"
        },
        
        # 10. Missing message
        {
            "payload": {
                "user_id": "test-user",
                "session_id": "test-123"
            },
            "description": "Missing message field"
        }
    ]
    
    successful_payloads = []
    failed_payloads = []
    
    for test_case in test_cases:
        success = test_payload(test_case["payload"], test_case["description"])
        if success:
            successful_payloads.append(test_case)
        else:
            failed_payloads.append(test_case)
    
    # Summary
    print(f"\n{'='*70}")
    print("🏁 DEBUGGING SUMMARY")
    print("=" * 70)
    print(f"✅ Successful payloads: {len(successful_payloads)}")
    print(f"❌ Failed payloads: {len(failed_payloads)}")
    
    if successful_payloads:
        print("\n✅ WORKING PAYLOAD FORMATS:")
        for case in successful_payloads:
            print(f"  • {case['description']}")
            print(f"    {json.dumps(case['payload'])}")
    
    if failed_payloads:
        print("\n❌ FAILED PAYLOAD FORMATS:")
        for case in failed_payloads[:3]:  # Show first 3 failures
            print(f"  • {case['description']}")
    
    print(f"\n💡 NEXT STEPS:")
    if successful_payloads:
        print("1. Use the working payload format above")
        print("2. Update frontend to send exactly this format")
        print("3. Test chat interface with working format")
    else:
        print("1. Check ChatRequest model in chat_server.py")
        print("2. Compare with what frontend is sending")
        print("3. Fix field name/type mismatches")

if __name__ == "__main__":
    main()