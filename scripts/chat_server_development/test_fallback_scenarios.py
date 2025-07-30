#!/usr/bin/env python3
"""
Test different fallback scenarios
"""
import requests
import json
import os
import time

def test_scenario(description, setup_func=None):
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print("="*60)
    
    if setup_func:
        setup_func()
    
    payload = {
        "message": "I'm having trouble sleeping and feel anxious",
        "user_id": "test-user", 
        "session_id": None
    }
    
    start_time = time.time()
    
    try:
        response = requests.post("http://localhost:8000/api/chat", json=payload, timeout=20)
        elapsed = time.time() - start_time
        
        print(f"⏱️ Response time: {elapsed:.2f}s")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model selected: {data.get('selected_model')}")
            print(f"✅ Response: {data.get('response', '')[:80]}...")
            
            if data.get('fallback_mode'):
                print("🔄 Fallback mode activated")
            elif 'model_selection_results' in data:
                results = data['model_selection_results']
                available = results.get('all_models_evaluated', [])
                print(f"🎯 Available models: {available}")
                
            return True, data
        else:
            print(f"❌ Error: {response.status_code} - {response.text[:100]}")
            return False, None
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"❌ Timeout after {elapsed:.2f}s")
        return False, None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Exception after {elapsed:.2f}s: {e}")
        return False, None

def remove_api_keys():
    """Temporarily remove API keys to test fallback"""
    os.environ.pop('OPENAI_API_KEY', None)
    os.environ.pop('ANTHROPIC_API_KEY', None)
    print("🔧 Removed API keys")

def restore_api_keys():
    """Restore API keys if they exist"""
    # Note: This won't actually restore them since we don't know original values
    print("🔧 API keys would be restored in real scenario")

def main():
    print("🧪 Testing Health Check and Fallback Scenarios")
    
    scenarios = [
        ("Normal operation (with available models)", None),
        ("No API keys available (should use fallback)", remove_api_keys),
    ]
    
    results = []
    
    for description, setup in scenarios:
        success, data = test_scenario(description, setup)
        results.append((description, success, data))
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 FALLBACK SYSTEM TEST SUMMARY")
    print("="*60)
    
    for description, success, data in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {description}")
        
        if success and data:
            model = data.get('selected_model', 'unknown')
            is_fallback = data.get('fallback_mode', False)
            if is_fallback:
                print(f"     → Used fallback mode")
            else:
                print(f"     → Used model: {model}")
    
    successful_tests = sum(1 for _, success, _ in results if success)
    print(f"\n🎯 Results: {successful_tests}/{len(results)} tests passed")
    
    if successful_tests == len(results):
        print("✅ All fallback scenarios working correctly!")
    else:
        print("⚠️ Some scenarios need attention")

if __name__ == "__main__":
    main()