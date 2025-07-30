#!/usr/bin/env python3
"""
Simple Research Data Collection Script
=====================================

Collects research data by directly using the chat server API
to generate model comparisons across multiple scenarios.
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Any

# Mental health scenarios for evaluation
RESEARCH_SCENARIOS = [
    # Anxiety scenarios
    "I feel extremely anxious about my upcoming job interview tomorrow",
    "I've been having panic attacks and I don't understand why",
    "I can't sleep at night because I keep worrying about everything",
    "I feel overwhelmed by all my responsibilities at work and home",
    "How can I manage my social anxiety at work?",
    
    # Depression scenarios  
    "I've been feeling really depressed and hopeless for the past few weeks",
    "I feel like I'm not good enough and everyone judges me",
    "I'm dealing with grief after losing a loved one",
    "How can I improve my mental health during winter months?",
    "I have no motivation to do anything anymore",
    
    # Crisis scenarios
    "I'm having thoughts of hurting myself and don't know what to do",
    "I feel like giving up on everything",
    "I can't take this pain anymore",
    "Nobody would miss me if I was gone",
    
    # Information seeking
    "What are the symptoms of PTSD and how is it treated?",
    "How do I know if I have social anxiety disorder?",
    "I think I might have ADHD, what are the signs?",
    "What are some healthy coping strategies for stress?",
    "How can I build better self-esteem and confidence?",
    
    # Relationship issues
    "My partner and I are having serious relationship problems",
    "I'm going through a difficult divorce and feeling lost",
    "What should I do if I'm concerned about a friend's mental health?",
    "My child has been showing signs of depression, what should I do?",
    
    # General support
    "I'm struggling with substance abuse and want to get help",
    "I'm experiencing burnout at work and feel exhausted",
    "How can I take better care of my mental health?",
    "I need someone to talk to about my problems"
]

def collect_model_response(prompt: str, session_id: int) -> Dict[str, Any]:
    """Collect a response from the chat server for a given prompt"""
    
    url = "http://localhost:8000/api/chat"
    payload = {
        "message": prompt,
        "user_id": f"research_session_{session_id}",
        "session_id": None,  # Force new session for model selection
        "force_reselection": False
    }
    
    try:
        print(f"  🔍 Evaluating: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        start_time = time.time()
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        end_time = time.time()
        data = response.json()
        
        result = {
            'prompt': prompt,
            'response': data.get('response', ''),
            'selected_model': data.get('selected_model', ''),
            'confidence_score': data.get('confidence_score', 0),
            'model_scores': data.get('model_scores', {}),
            'prompt_type': data.get('prompt_type', ''),
            'reasoning': data.get('reasoning', ''),
            'response_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        }
        
        print(f"    ✅ Selected: {result['selected_model']} ({result['confidence_score']:.1%} confidence)")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"    ❌ Request failed: {e}")
        return {
            'prompt': prompt,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        }
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        return {
            'prompt': prompt,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        }

def main():
    """Main data collection function"""
    print("🧠 Mental Health LLM Research Data Collection")
    print("=" * 60)
    
    # Check if chat server is running
    try:
        response = requests.get("http://localhost:8000/api/status", timeout=5)
        response.raise_for_status()
        print("✅ Chat server is running")
    except:
        print("❌ Chat server is not running. Please start it with:")
        print("   python chat_server.py")
        sys.exit(1)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/development/research_data_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Collecting data for {len(RESEARCH_SCENARIOS)} scenarios")
    print()
    
    # Collect data for each scenario
    all_results = []
    
    for i, scenario in enumerate(RESEARCH_SCENARIOS, 1):
        print(f"📋 Scenario {i}/{len(RESEARCH_SCENARIOS)}")
        result = collect_model_response(scenario, i)
        all_results.append(result)
        
        # Small delay between requests
        time.sleep(1)
        print()
    
    # Save results
    output_file = os.path.join(output_dir, "research_data.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary
    successful_results = [r for r in all_results if 'error' not in r]
    failed_results = [r for r in all_results if 'error' in r]
    
    print("📊 COLLECTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {len(successful_results)}")
    print(f"❌ Failed: {len(failed_results)}")
    
    if successful_results:
        # Model selection statistics
        model_counts = {}
        prompt_type_counts = {}
        
        for result in successful_results:
            model = result.get('selected_model', 'unknown')
            prompt_type = result.get('prompt_type', 'unknown')
            
            model_counts[model] = model_counts.get(model, 0) + 1
            prompt_type_counts[prompt_type] = prompt_type_counts.get(prompt_type, 0) + 1
        
        print(f"\n🤖 Model Selection Distribution:")
        for model, count in sorted(model_counts.items()):
            percentage = (count / len(successful_results)) * 100
            print(f"   {model}: {count} ({percentage:.1f}%)")
        
        print(f"\n📝 Prompt Type Distribution:")
        for prompt_type, count in sorted(prompt_type_counts.items()):
            percentage = (count / len(successful_results)) * 100
            print(f"   {prompt_type}: {count} ({percentage:.1f}%)")
        
        # Average confidence and response time
        avg_confidence = sum(r.get('confidence_score', 0) for r in successful_results) / len(successful_results)
        avg_response_time = sum(r.get('response_time', 0) for r in successful_results) / len(successful_results)
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   Average Response Time: {avg_response_time:.2f}s")
    
    print(f"\n💾 Results saved to: {output_file}")
    print("🎯 Research data collection complete!")

if __name__ == "__main__":
    main()