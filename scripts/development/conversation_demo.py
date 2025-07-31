#!/usr/bin/env python3
"""
Demo script showing full conversations with model selection and continuation
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def chat_message(message, session_id=None, user_id="demo-user"):
    """Send a chat message and return response"""
    data = {"message": message, "user_id": user_id}
    if session_id:
        data["session_id"] = session_id
    
    print(f"\n💬 USER: {message}")
    print("⏳ Processing..." if session_id else "⏳ Selecting model...")
    
    start = time.time()
    response = requests.post(f"{BASE_URL}/api/chat", json=data, timeout=120)
    duration = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        model = result['selected_model'].upper()
        confidence = result['confidence_score']
        
        if not session_id:  # First message - show selection details
            scores = result.get('model_scores', {})
            print(f"🎯 MODEL SELECTION: {model} selected ({confidence:.1%} confidence)")
            print(f"📊 Scores: {scores}")
            print(f"⚡ Time: {duration:.1f}s")
        else:  # Continuation
            print(f"💭 CONTINUATION with {model} ({duration:.1f}s)")
        
        print(f"\n🤖 {model}: {result['response']}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def run_conversation(title, messages):
    """Run a complete conversation"""
    print("\n" + "="*80)
    print(f"🗣️  {title}")
    print("="*80)
    
    session_id = None
    
    for i, message in enumerate(messages):
        result = chat_message(message, session_id)
        if result:
            session_id = result['session_id']
            if i == 0:
                print(f"📋 Session: {session_id[:12]}...")
        else:
            print("❌ Conversation failed!")
            return
        
        time.sleep(1)  # Brief pause between messages
    
    print(f"\n✅ '{title}' conversation completed successfully!")

def main():
    print("🎭 REAL MODEL SELECTION & CONVERSATION DEMO")
    print("🎯 Testing different conversation types to see model selection patterns")
    
    # Conversation 1: Anxiety → Should favor Claude
    anxiety_msgs = [
        "I'm having panic attacks at work and can't focus",
        "What are some immediate techniques I can use when I feel a panic attack starting?",
        "Thank you, that's really helpful. How can I prevent them from happening in the first place?"
    ]
    run_conversation("ANXIETY & PANIC ATTACKS", anxiety_msgs)
    
    # Conversation 2: Career Analysis → Should favor OpenAI
    career_msgs = [
        "I need to make a strategic decision about changing careers from engineering to product management",
        "What are the key skills I should develop for this transition?",
        "How should I structure my job search timeline for this career change?"
    ]
    run_conversation("STRATEGIC CAREER PLANNING", career_msgs)
    
    # Conversation 3: Depression → Should favor Claude
    depression_msgs = [
        "I've lost interest in everything I used to enjoy and feel numb all the time",
        "How do I find motivation to do even basic things like getting out of bed?",
        "Is it normal to feel like this will never get better?"
    ]
    run_conversation("DEPRESSION & MOTIVATION", depression_msgs)

if __name__ == "__main__":
    main()