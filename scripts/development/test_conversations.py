#!/usr/bin/env python3
"""
Test multiple conversations to see different model selections
"""
import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def send_message(message: str, session_id: str = None, user_id: str = "test-user") -> Dict[str, Any]:
    """Send a message and return the response"""
    chat_data = {
        "message": message,
        "user_id": user_id
    }
    if session_id:
        chat_data["session_id"] = session_id
    
    print(f"\n💬 USER: {message}")
    print("⏳ Waiting for real model evaluation...")
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/chat", json=chat_data, timeout=120)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        print(f"⚡ Response time: {end_time - start_time:.1f}s")
        print(f"🤖 Selected: {result['selected_model'].upper()}")
        print(f"🎯 Confidence: {result['confidence_score']:.1%}")
        if result.get('model_scores'):
            print(f"📊 Scores: {result['model_scores']}")
        print(f"🤖 {result['selected_model'].upper()}: {result['response']}")
        return result
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        return None

def run_conversation_test(title, messages):
    """Run a full conversation test"""
    print("\n" + "="*60)
    print(f"🗣️  {title}")
    print("="*60)
    
    session_id = None
    
    for i, message in enumerate(messages):
        result = send_message(message, session_id)
        if result:
            session_id = result['session_id']
            if i == 0:  # First message
                print(f"🆕 New session: {session_id[:8]}...")
        else:
            print("❌ Conversation failed")
            break
        
        # Small delay between messages
        time.sleep(1)
    
    print(f"\n✅ Conversation '{title}' completed")

def main():
    print("🧪 TESTING REAL MODEL SELECTION WITH FULL CONVERSATIONS")
    print("🎯 Goal: See different models get selected for different conversation types")
    
    # Test 1: Anxiety-focused conversation
    anxiety_convo = [
        "I'm feeling really anxious about my job interview tomorrow",
        "What specific techniques can you suggest to help me calm down?",
        "Thank you, that's helpful. How do I handle the physical symptoms?"
    ]
    run_conversation_test("ANXIETY MANAGEMENT", anxiety_convo)
    
    # Test 2: Depression-focused conversation  
    depression_convo = [
        "I've been feeling really depressed and hopeless lately",
        "I don't know how to get out of this dark place I'm in",
        "What small steps can I take to start feeling better?"
    ]
    run_conversation_test("DEPRESSION SUPPORT", depression_convo)
    
    # Test 3: Crisis conversation
    crisis_convo = [
        "I'm having thoughts of hurting myself and don't know what to do",
        "I feel like nobody would care if I was gone",
        "Thank you for caring. What resources are available to help me?"
    ]
    run_conversation_test("CRISIS INTERVENTION", crisis_convo)
    
    # Test 4: Relationship issues
    relationship_convo = [
        "My partner and I have been fighting constantly and I don't know how to fix it",
        "We can't seem to communicate without it turning into an argument",
        "What are some healthy ways to resolve conflicts in relationships?"
    ]
    run_conversation_test("RELATIONSHIP COUNSELING", relationship_convo)
    
    # Test 5: General wellness
    wellness_convo = [
        "I want to work on my overall mental health and wellbeing",
        "What daily practices would you recommend for maintaining good mental health?",
        "How do I build resilience for handling future challenges?"
    ]
    run_conversation_test("WELLNESS & PREVENTION", wellness_convo)

if __name__ == "__main__":
    main()