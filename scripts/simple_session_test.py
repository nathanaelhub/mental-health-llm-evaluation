#!/usr/bin/env python3
"""
Simple Session Continuation Test
"""
import requests
import json
import time

def test_session_continuation():
    base_url = "http://localhost:8000"
    session_id = f"test-{int(time.time())}"
    
    print("🧪 Simple Session Test")
    print("=" * 30)
    print(f"Session ID: {session_id}")
    
    # First message
    print("\n1️⃣ First message...")
    try:
        response1 = requests.post(
            f"{base_url}/api/chat",
            json={
                "message": "I feel anxious",
                "session_id": session_id,
                "user_id": "test"
            },
            timeout=60
        )
        
        if response1.status_code == 200:
            data1 = response1.json()
            model1 = data1.get('selected_model', 'NONE')
            mode1 = data1.get('conversation_mode', 'unknown')
            print(f"✅ Model: {model1}, Mode: {mode1}")
        else:
            print(f"❌ HTTP {response1.status_code}: {response1.text}")
            return
            
    except requests.exceptions.Timeout:
        print("❌ First message timed out")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Small delay
    time.sleep(2)
    
    # Second message
    print("\n2️⃣ Second message...")
    try:
        response2 = requests.post(
            f"{base_url}/api/chat",
            json={
                "message": "Can you help me?",
                "session_id": session_id,  # Same session
                "user_id": "test"
            },
            timeout=60
        )
        
        if response2.status_code == 200:
            data2 = response2.json()
            model2 = data2.get('selected_model', 'NONE')
            mode2 = data2.get('conversation_mode', 'unknown')
            print(f"✅ Model: {model2}, Mode: {mode2}")
        else:
            print(f"❌ HTTP {response2.status_code}: {response2.text}")
            return
            
    except requests.exceptions.Timeout:
        print("❌ Second message timed out")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Analysis
    print("\n🔍 Analysis:")
    print(f"First:  {model1} ({mode1})")
    print(f"Second: {model2} ({mode2})")
    
    if model1 == model2 and model1 != 'NONE':
        print("✅ SUCCESS: Session continuation working!")
    else:
        print("❌ FAILURE: Session continuation broken!")
        print("\n💡 Possible issues:")
        print("- Session not being stored properly")
        print("- Session retrieval failing")
        print("- Model selection running every time")

if __name__ == "__main__":
    test_session_continuation()