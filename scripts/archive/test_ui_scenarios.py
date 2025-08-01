#!/usr/bin/env python3
"""
Test various mental health scenarios through the UI and document patterns.
For capstone documentation of real-world usage patterns.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import os

# Test scenarios from documentation
SCENARIOS = [
    # Anxiety scenarios
    {
        "category": "workplace_anxiety",
        "messages": [
            "I'm feeling really anxious about a big presentation I have to give tomorrow. My heart is racing and I can't stop worrying about all the things that could go wrong.",
            "What specific techniques can I use to calm my nerves?",
            "How can I prepare mentally for the presentation?"
        ],
        "expected_model": "openai or claude",
        "expected_confidence": "60-70%"
    },
    # Depression scenarios
    {
        "category": "depression",
        "messages": [
            "I've been feeling really down lately. I don't have motivation to do anything and I feel like I'm just going through the motions of life. Nothing brings me joy anymore.",
            "How can I start feeling more engaged with life again?",
            "What small steps can I take when everything feels overwhelming?"
        ],
        "expected_model": "claude",
        "expected_confidence": "high"
    },
    # Relationship issues
    {
        "category": "relationship_communication",
        "messages": [
            "I'm having trouble communicating with my partner. We keep having the same arguments over and over, and I don't know how to break this cycle.",
            "Can you give me some practical communication strategies?",
            "How do I bring up difficult topics without starting a fight?"
        ],
        "expected_model": "gemma or claude",
        "expected_confidence": "moderate"
    },
    # Work-life balance
    {
        "category": "work_life_balance",
        "messages": [
            "I feel like I'm drowning in work responsibilities. I'm working 60+ hours a week and I barely have time for my family or myself. I'm exhausted all the time.",
            "How can I set better boundaries with work?",
            "What strategies help with feeling overwhelmed by deadlines?"
        ],
        "expected_model": "openai",
        "expected_confidence": "moderate-high"
    },
    # Social anxiety
    {
        "category": "social_anxiety",
        "messages": [
            "I avoid social gatherings because I always feel like people are judging me. Even thinking about going to parties makes me anxious.",
            "What techniques can help me feel more comfortable in social situations?",
            "How do I stop worrying so much about what others think?"
        ],
        "expected_model": "openai or claude",
        "expected_confidence": "moderate"
    },
    # Sleep issues
    {
        "category": "sleep_issues",
        "messages": [
            "I can't fall asleep because my mind races with worries about everything I need to do tomorrow. This happens almost every night.",
            "What can I do to quiet my mind before bed?",
            "Are there any techniques to help with racing thoughts at night?"
        ],
        "expected_model": "openai",
        "expected_confidence": "moderate"
    },
    # Academic pressure
    {
        "category": "academic_pressure",
        "messages": [
            "I'm a college student and the pressure to get perfect grades is crushing me. I study constantly but never feel like it's enough.",
            "How can I manage academic stress more effectively?",
            "What's a healthy approach to academic achievement?"
        ],
        "expected_model": "openai",
        "expected_confidence": "moderate-high"
    },
    # Mild crisis (safe for demo)
    {
        "category": "mild_crisis",
        "messages": [
            "I'm having thoughts of hurting myself and I don't know what to do. Everything feels hopeless right now.",
            "Where can I find immediate support?",
            "What coping strategies can help right now?"
        ],
        "expected_model": "claude",
        "expected_confidence": "very high (safety)"
    }
]

BASE_URL = "http://localhost:8000"

def test_scenario(scenario: Dict) -> Dict:
    """Test a single scenario through the UI."""
    results = {
        "category": scenario["category"],
        "expected_model": scenario["expected_model"],
        "expected_confidence": scenario["expected_confidence"],
        "messages": [],
        "timings": [],
        "session_id": None
    }
    
    session_id = None
    
    for i, message in enumerate(scenario["messages"]):
        start_time = time.time()
        
        try:
            # Prepare request
            data = {"message": message}
            if session_id:
                data["session_id"] = session_id
            
            # Send message
            response = requests.post(
                f"{BASE_URL}/api/chat",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract session ID from first response
                if not session_id and "session_id" in result:
                    session_id = result["session_id"]
                    results["session_id"] = session_id
                
                # Store results
                results["messages"].append({
                    "index": i + 1,
                    "user_message": message,
                    "ai_response": result.get("response", ""),
                    "selected_model": result.get("selected_model", ""),
                    "confidence": result.get("confidence", 0),
                    "is_continuation": i > 0
                })
                
                results["timings"].append({
                    "message_index": i + 1,
                    "elapsed_seconds": round(elapsed, 2),
                    "type": "continuation" if i > 0 else "initial"
                })
                
                print(f"  Message {i+1}: {elapsed:.2f}s - Model: {result.get('selected_model', 'unknown')}")
                
            else:
                print(f"  Message {i+1}: ERROR - {response.status_code}")
                results["messages"].append({
                    "index": i + 1,
                    "error": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"  Message {i+1}: ERROR - {str(e)}")
            results["messages"].append({
                "index": i + 1,
                "error": str(e)
            })
        
        # Small delay between messages
        if i < len(scenario["messages"]) - 1:
            time.sleep(2)
    
    return results

def main():
    """Run all scenario tests and generate report."""
    print("\n🧪 Testing UI Scenarios for Capstone Documentation")
    print("=" * 60)
    
    # Check server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code not in [200, 302]:  # 302 for redirect
            print(f"❌ Server not responding at {BASE_URL}")
            print("Please start the chat server first: python chat_server.py")
            return
    except:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print("Please start the chat server first: python chat_server.py")
        return
    
    print("✅ Server is running\n")
    
    # Run all scenarios
    all_results = []
    
    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n📋 Scenario {i}/{len(SCENARIOS)}: {scenario['category']}")
        print("-" * 40)
        
        results = test_scenario(scenario)
        all_results.append(results)
        
        # Summary
        if results["messages"]:
            first_msg = results["messages"][0]
            if "selected_model" in first_msg:
                print(f"\n  ✓ Selected: {first_msg['selected_model']} ({first_msg['confidence']:.1f}% confidence)")
                print(f"  ✓ Expected: {scenario['expected_model']}")
                
                # Check if expectation met
                selected = first_msg['selected_model'].lower()
                expected = scenario['expected_model'].lower()
                if selected in expected or expected in selected:
                    print("  ✅ Expectation MET")
                else:
                    print("  ⚠️  Different than expected")
        
        # Timing summary
        if results["timings"]:
            initial = [t for t in results["timings"] if t["type"] == "initial"]
            continuation = [t for t in results["timings"] if t["type"] == "continuation"]
            
            if initial:
                print(f"\n  ⏱️  Initial response: {initial[0]['elapsed_seconds']}s")
            if continuation:
                avg_cont = sum(t['elapsed_seconds'] for t in continuation) / len(continuation)
                print(f"  ⏱️  Avg continuation: {avg_cont:.2f}s")
        
        time.sleep(3)  # Pause between scenarios
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"results/development/ui_scenario_test_{timestamp}.json"
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "server_url": BASE_URL,
            "scenarios_tested": len(SCENARIOS),
            "results": all_results
        }, f, indent=2)
    
    print(f"\n\n📊 Test Results Summary")
    print("=" * 60)
    
    # Analyze patterns
    model_selections = {}
    timing_stats = {"initial": [], "continuation": []}
    expectation_matches = 0
    
    for result in all_results:
        if result["messages"] and "selected_model" in result["messages"][0]:
            model = result["messages"][0]["selected_model"]
            model_selections[model] = model_selections.get(model, 0) + 1
            
            # Check expectation
            if model.lower() in result["expected_model"].lower():
                expectation_matches += 1
        
        # Collect timing stats
        for timing in result["timings"]:
            timing_stats[timing["type"]].append(timing["elapsed_seconds"])
    
    # Print analysis
    print("\n📈 Model Selection Distribution:")
    for model, count in sorted(model_selections.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(SCENARIOS)) * 100
        print(f"  - {model}: {count} selections ({percentage:.1f}%)")
    
    print(f"\n🎯 Expectation Accuracy: {expectation_matches}/{len(SCENARIOS)} ({(expectation_matches/len(SCENARIOS)*100):.1f}%)")
    
    print("\n⏱️  Performance Metrics:")
    if timing_stats["initial"]:
        avg_initial = sum(timing_stats["initial"]) / len(timing_stats["initial"])
        print(f"  - Average initial response: {avg_initial:.2f}s")
    if timing_stats["continuation"]:
        avg_cont = sum(timing_stats["continuation"]) / len(timing_stats["continuation"])
        print(f"  - Average continuation: {avg_cont:.2f}s")
        print(f"  - Performance improvement: {((avg_initial - avg_cont) / avg_initial * 100):.1f}%")
    
    print(f"\n💾 Full results saved to: {report_path}")
    print("\nUse these results for your capstone documentation!")

if __name__ == "__main__":
    main()