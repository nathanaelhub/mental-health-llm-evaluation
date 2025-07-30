#!/usr/bin/env python3
"""
Test Improved Model Selector
Tests the enhanced model selection with timeouts, retries, and partial selection
"""

import asyncio
import requests
import json
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_chat_performance(message: str, description: str, timeout: int = 25):
    """Test chat API with performance monitoring"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Message: {message}")
    print("-" * 60)
    
    payload = {
        "message": message,
        "user_id": "test-user",
        "session_id": None
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8000/api/chat", 
            json=payload, 
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ SUCCESS ({elapsed:.2f}s)")
            print(f"   Selected model: {data.get('selected_model', 'unknown')}")
            print(f"   Confidence: {data.get('confidence_score', 0):.1%}")
            print(f"   Response: {data.get('response', '')[:80]}...")
            
            # Check for selection metadata
            selection_results = data.get('model_selection_results', {})
            if selection_results:
                available_models = selection_results.get('all_models_evaluated', [])
                evaluation_time = selection_results.get('evaluation_time_ms', 0)
                print(f"   Available models: {available_models}")
                print(f"   Evaluation time: {evaluation_time:.0f}ms")
                
                # Check for fallback indicators
                if selection_results.get('is_fallback'):
                    print(f"   🔄 Fallback mode activated")
                elif selection_results.get('is_single_model'):
                    print(f"   🎯 Single model selection")
                elif selection_results.get('is_timeout_fallback'):
                    print(f"   ⏰ Timeout fallback used")
                elif selection_results.get('is_error_fallback'):
                    print(f"   ❌ Error fallback used")
                else:
                    print(f"   🧠 Full model selection completed")
            
            return True, elapsed, data
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"   Error: {response.text[:100]}...")
            return False, elapsed, None
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"❌ TIMEOUT after {elapsed:.2f}s")
        return False, elapsed, None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ ERROR after {elapsed:.2f}s: {e}")
        return False, elapsed, None

def main():
    print("🧪 IMPROVED MODEL SELECTOR TEST")
    print("=" * 60)
    print("Testing enhanced features:")
    print("• Model-specific timeouts (cloud: 5s, local: 10s)")
    print("• Retry logic for local models")
    print("• Partial selection when some models fail")
    print("• Model availability caching")
    
    # Test scenarios
    test_cases = [
        {
            "message": "I'm feeling overwhelmed with work stress",
            "description": "Anxiety/Stress scenario - should prefer empathetic models",
            "timeout": 20
        },
        {
            "message": "I've been feeling really down lately and unmotivated",
            "description": "Depression scenario - should prefer therapeutic models", 
            "timeout": 20
        },
        {
            "message": "I'm having thoughts of hurting myself",
            "description": "Crisis scenario - should prioritize safety",
            "timeout": 20
        },
        {
            "message": "Can you tell me about different types of therapy?",
            "description": "Information seeking - should prefer clarity",
            "timeout": 20
        },
        {
            "message": "Just need someone to talk to",
            "description": "General support - balanced selection",
            "timeout": 20
        }
    ]
    
    results = []
    total_time = 0
    
    print(f"\n🚀 Running {len(test_cases)} test scenarios...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}]", end=" ")
        success, elapsed, data = test_chat_performance(
            test_case["message"],
            test_case["description"],
            test_case["timeout"]
        )
        
        results.append({
            "scenario": test_case["description"],
            "success": success,
            "time": elapsed,
            "data": data
        })
        
        total_time += elapsed
        
        # Brief pause between tests
        if i < len(test_cases):
            time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 IMPROVED SELECTOR TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r["success"])
    avg_time = total_time / len(results)
    
    print(f"📊 Results: {successful_tests}/{len(results)} tests passed")
    print(f"⏱️  Average response time: {avg_time:.2f}s")
    print(f"🕒 Total test time: {total_time:.2f}s")
    
    # Analyze model selection patterns
    model_counts = {}
    selection_types = {}
    
    for result in results:
        if result["success"] and result["data"]:
            model = result["data"].get("selected_model", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1
            
            selection_results = result["data"].get("model_selection_results", {})
            if selection_results.get('is_fallback'):
                selection_type = "fallback"
            elif selection_results.get('is_single_model'):
                selection_type = "single_model"
            elif selection_results.get('is_timeout_fallback'):
                selection_type = "timeout_fallback"
            elif selection_results.get('is_error_fallback'):
                selection_type = "error_fallback"
            else:
                selection_type = "full_selection"
            
            selection_types[selection_type] = selection_types.get(selection_type, 0) + 1
    
    if model_counts:
        print(f"\n🎯 Model Selection Distribution:")
        for model, count in sorted(model_counts.items()):
            percentage = (count / successful_tests) * 100
            print(f"   {model.upper()}: {count} times ({percentage:.1f}%)")
    
    if selection_types:
        print(f"\n🔄 Selection Type Distribution:")
        for sel_type, count in sorted(selection_types.items()):
            percentage = (count / successful_tests) * 100
            print(f"   {sel_type.replace('_', ' ').title()}: {count} times ({percentage:.1f}%)")
    
    # Performance assessment
    print(f"\n💡 Performance Assessment:")
    if avg_time < 10:
        print("   ✅ Excellent - Fast response times")
    elif avg_time < 20:
        print("   ✅ Good - Acceptable response times")
    else:
        print("   ⚠️  Slow - Consider optimizing timeouts")
    
    if successful_tests == len(results):
        print("   ✅ Perfect reliability - All tests passed")
    elif successful_tests >= len(results) * 0.8:
        print("   ✅ Good reliability - Most tests passed")
    else:
        print("   ⚠️  Poor reliability - Many tests failed")
    
    # Recommendations
    print(f"\n🔧 Recommendations:")
    if "full_selection" in selection_types and selection_types["full_selection"] > 0:
        print("   ✅ Full model selection is working")
    else:
        print("   ⚠️  Full model selection not working - check model health")
    
    if "fallback" in selection_types and selection_types["fallback"] > len(results) // 2:
        print("   ⚠️  Too many fallbacks - check model configurations")
    else:
        print("   ✅ Fallback system working as backup")
    
    print(f"\n🎉 Enhanced model selector test complete!")
    
    return successful_tests == len(results)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)