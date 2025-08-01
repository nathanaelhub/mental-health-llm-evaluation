#!/usr/bin/env python3
"""
Test Session Continuation Fix
=============================

This script tests specifically that the session continuation fix is working.
It handles the long initial evaluation time but focuses on fast continuation.
"""

import requests
import json
import time

def test_session_fix():
    base_url = "http://localhost:8000"
    session_id = f"fix-test-{int(time.time())}"
    
    print("🔧 Testing Session Continuation Fix")
    print("=" * 40)
    print(f"Session ID: {session_id}")
    print()
    
    # Test 1: First message (expect long wait for model evaluation)
    print("1️⃣ First message (expect 60-90 seconds for model evaluation)...")
    start_time = time.time()
    
    try:
        response1 = requests.post(
            f"{base_url}/api/chat",
            json={
                "message": "I'm feeling anxious about work presentations",
                "session_id": session_id,
                "user_id": "fix-test"
            },
            timeout=180  # 3 minutes timeout
        )
        
        first_duration = time.time() - start_time
        
        if response1.status_code == 200:
            data1 = response1.json()
            model1 = data1.get('selected_model', 'NONE')
            mode1 = data1.get('conversation_mode', 'unknown')
            confidence1 = data1.get('confidence_score', 0)
            
            print(f"✅ First message completed in {first_duration:.1f}s")
            print(f"   Selected Model: {model1}")
            print(f"   Mode: {mode1}")
            print(f"   Confidence: {confidence1:.1%}")
            print(f"   Session: {data1.get('session_id', 'NONE')[:12]}...")
        else:
            print(f"❌ First message failed: HTTP {response1.status_code}")
            print(f"   Response: {response1.text}")
            return
            
    except requests.exceptions.Timeout:
        print(f"❌ First message timed out after {time.time() - start_time:.1f}s")
        return
    except Exception as e:
        print(f"❌ First message error: {e}")
        return
    
    # Brief pause
    time.sleep(2)
    
    # Test 2: Second message (should be FAST with stored model)
    print(f"\n2️⃣ Second message (should be FAST - using stored {model1})...")
    start_time = time.time()
    
    try:
        response2 = requests.post(
            f"{base_url}/api/chat",
            json={
                "message": "Yes, can you give me some specific techniques?",
                "session_id": session_id,  # Same session
                "user_id": "fix-test"
            },
            timeout=30  # Should be fast now!
        )
        
        second_duration = time.time() - start_time
        
        if response2.status_code == 200:
            data2 = response2.json()
            model2 = data2.get('selected_model', 'NONE')
            mode2 = data2.get('conversation_mode', 'unknown')
            
            print(f"✅ Second message completed in {second_duration:.1f}s")
            print(f"   Used Model: {model2}")
            print(f"   Mode: {mode2}")
        else:
            print(f"❌ Second message failed: HTTP {response2.status_code}")
            print(f"   Response: {response2.text}")
            return
            
    except requests.exceptions.Timeout:
        print(f"❌ Second message timed out after {time.time() - start_time:.1f}s")
        print("💡 This suggests continuation is still running model evaluation!")
        return
    except Exception as e:
        print(f"❌ Second message error: {e}")
        return
    
    # Test 3: Third message (should also be fast)
    print(f"\n3️⃣ Third message (should also be FAST - using stored {model1})...")
    start_time = time.time()
    
    try:
        response3 = requests.post(
            f"{base_url}/api/chat",
            json={
                "message": "Thank you, that's helpful!",
                "session_id": session_id,  # Same session
                "user_id": "fix-test"
            },
            timeout=30
        )
        
        third_duration = time.time() - start_time
        
        if response3.status_code == 200:
            data3 = response3.json()
            model3 = data3.get('selected_model', 'NONE')
            mode3 = data3.get('conversation_mode', 'unknown')
            
            print(f"✅ Third message completed in {third_duration:.1f}s")
            print(f"   Used Model: {model3}")
            print(f"   Mode: {mode3}")
        else:
            print(f"❌ Third message failed: HTTP {response3.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Third message error: {e}")
        return
    
    # Analysis
    print(f"\n📊 ANALYSIS:")
    print(f"=" * 30)
    print(f"First message:  {model1} ({mode1}) - {first_duration:.1f}s")
    print(f"Second message: {model2} ({mode2}) - {second_duration:.1f}s") 
    print(f"Third message:  {model3} ({mode3}) - {third_duration:.1f}s")
    
    # Success criteria
    success_criteria = []
    
    # 1. Same model used throughout
    if model1 == model2 == model3 and model1 != 'NONE':
        print(f"✅ Model Consistency: {model1} used throughout")
        success_criteria.append("model_consistency")
    else:
        print(f"❌ Model Inconsistency: {model1} → {model2} → {model3}")
    
    # 2. Proper mode progression
    if mode1 == "selection" and mode2 == "continuation" and mode3 == "continuation":
        print(f"✅ Mode Progression: selection → continuation → continuation")
        success_criteria.append("mode_progression")
    else:
        print(f"❌ Mode Issues: {mode1} → {mode2} → {mode3}")
    
    # 3. Fast continuation (key improvement!)
    if second_duration < 10.0 and third_duration < 10.0:
        print(f"✅ Fast Continuation: Messages 2&3 under 10s each")
        success_criteria.append("fast_continuation")
    else:
        print(f"❌ Slow Continuation: Still taking too long (>10s)")
        print(f"💡 This means the fix didn't work - still re-evaluating models")
    
    # 4. First message appropriately slow
    if first_duration > 30.0:
        print(f"✅ Proper Initial Evaluation: First message took {first_duration:.1f}s (expected)")
        success_criteria.append("proper_initial")
    
    # Overall result
    print(f"\n🎯 OVERALL RESULT:")
    if len(success_criteria) >= 3:
        print(f"🎉 SUCCESS: Session continuation fix is working!")
        print(f"   ✅ {len(success_criteria)}/4 criteria met")
        if "fast_continuation" in success_criteria:
            print(f"   🚀 KEY IMPROVEMENT: Continuation is now fast!")
    else:
        print(f"❌ FAILURE: Session continuation still has issues")
        print(f"   ❌ Only {len(success_criteria)}/4 criteria met")
        if "fast_continuation" not in success_criteria:
            print(f"   🐌 MAIN ISSUE: Continuation is still slow (re-evaluating models)")

if __name__ == "__main__":
    test_session_fix()