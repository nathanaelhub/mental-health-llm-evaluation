#!/usr/bin/env python3
"""
Test Full Conversations
=======================

Tests complete conversation flows through the chat server to verify
different models are selected based on scenario type and research scores.
"""

import requests
import json
import time
from typing import Dict, Any

def send_message(message: str, session_id: str = None, user_id: str = "conversation-test") -> Dict[str, Any]:
    """Send a message to the chat server"""
    url = "http://localhost:8000/api/chat"
    payload = {
        "message": message,
        "user_id": user_id,
        "session_id": session_id,
        "force_reselection": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def print_conversation_header(title: str, scenario_type: str):
    """Print a formatted conversation header"""
    print("\n" + "="*80)
    print(f"🗣️  CONVERSATION: {title}")
    print(f"📋 Scenario Type: {scenario_type}")
    print("="*80)

def print_message(speaker: str, message: str, model_info: str = None):
    """Print a formatted message"""
    if speaker == "👤 User":
        print(f"\n{speaker}: {message}")
    else:
        model_part = f" ({model_info})" if model_info else ""
        print(f"\n🤖 {speaker}{model_part}: {message}")

def run_conversation_1():
    """Information seeking - should select DeepSeek"""
    print_conversation_header("Information Seeking About Machine Learning", "Information/Technical")
    
    # Message 1 - Initial selection
    response1 = send_message("How do machine learning algorithms work and what are the different types?", user_id="conv1-user")
    if "error" in response1:
        print(f"❌ Error: {response1['error']}")
        return
    
    session_id = response1.get("session_id")
    selected_model = response1.get("selected_model", "unknown")
    model_scores = response1.get("model_scores", {})
    confidence = response1.get("confidence_score", 0)
    
    print(f"🎯 Model Selection: {selected_model.upper()} (confidence: {confidence:.1%})")
    print(f"📊 Scores: {model_scores}")
    
    print_message("👤 User", "How do machine learning algorithms work and what are the different types?")
    print_message("Assistant", response1.get("response", ""), f"{selected_model.upper()}, Turn 1")
    
    time.sleep(1)
    
    # Message 2 - Continuation
    response2 = send_message("Can you explain neural networks in more detail?", session_id=session_id, user_id="conv1-user")
    if "error" not in response2:
        print_message("👤 User", "Can you explain neural networks in more detail?")
        print_message("Assistant", response2.get("response", ""), f"{selected_model.upper()}, Turn 2")
        
        time.sleep(1)
        
        # Message 3 - Final
        response3 = send_message("What programming languages are best for machine learning?", session_id=session_id, user_id="conv1-user")
        if "error" not in response3:
            print_message("👤 User", "What programming languages are best for machine learning?")
            print_message("Assistant", response3.get("response", ""), f"{selected_model.upper()}, Turn 3")
    
    print(f"\n✅ Conversation completed with {selected_model.upper()}")

def run_conversation_2():
    """Anxiety scenario - might select different model"""
    print_conversation_header("Anxiety About Job Interview", "Anxiety/Mental Health")
    
    # Message 1 - Initial selection
    response1 = send_message("I'm feeling extremely anxious about my job interview tomorrow. My hands are shaking and I can't sleep.", user_id="conv2-user")
    if "error" in response1:
        print(f"❌ Error: {response1['error']}")
        return
    
    session_id = response1.get("session_id")
    selected_model = response1.get("selected_model", "unknown")
    model_scores = response1.get("model_scores", {})
    confidence = response1.get("confidence_score", 0)
    
    print(f"🎯 Model Selection: {selected_model.upper()} (confidence: {confidence:.1%})")
    print(f"📊 Scores: {model_scores}")
    
    print_message("👤 User", "I'm feeling extremely anxious about my job interview tomorrow. My hands are shaking and I can't sleep.")
    print_message("Assistant", response1.get("response", ""), f"{selected_model.upper()}, Turn 1")
    
    time.sleep(1)
    
    # Message 2 - Continuation
    response2 = send_message("What can I do right now to calm down?", session_id=session_id, user_id="conv2-user")
    if "error" not in response2:
        print_message("👤 User", "What can I do right now to calm down?")
        print_message("Assistant", response2.get("response", ""), f"{selected_model.upper()}, Turn 2")
        
        time.sleep(1)
        
        # Message 3 - Final
        response3 = send_message("Thank you, that actually helps a lot. I feel a bit better now.", session_id=session_id, user_id="conv2-user")
        if "error" not in response3:
            print_message("👤 User", "Thank you, that actually helps a lot. I feel a bit better now.")
            print_message("Assistant", response3.get("response", ""), f"{selected_model.upper()}, Turn 3")
    
    print(f"\n✅ Conversation completed with {selected_model.upper()}")

def run_conversation_3():
    """General support - could select various models"""
    print_conversation_header("General Life Advice", "General Support")
    
    # Message 1 - Initial selection  
    response1 = send_message("I need some advice on how to better manage my time and be more productive.", user_id="conv3-user")
    if "error" in response1:
        print(f"❌ Error: {response1['error']}")
        return
    
    session_id = response1.get("session_id")
    selected_model = response1.get("selected_model", "unknown")
    model_scores = response1.get("model_scores", {})
    confidence = response1.get("confidence_score", 0)
    
    print(f"🎯 Model Selection: {selected_model.upper()} (confidence: {confidence:.1%})")
    print(f"📊 Scores: {model_scores}")
    
    print_message("👤 User", "I need some advice on how to better manage my time and be more productive.")
    print_message("Assistant", response1.get("response", ""), f"{selected_model.upper()}, Turn 1")
    
    time.sleep(1)
    
    # Message 2 - Continuation
    response2 = send_message("I struggle especially with procrastination. Any specific techniques?", session_id=session_id, user_id="conv3-user")
    if "error" not in response2:
        print_message("👤 User", "I struggle especially with procrastination. Any specific techniques?")
        print_message("Assistant", response2.get("response", ""), f"{selected_model.upper()}, Turn 2")
        
        time.sleep(1)
        
        # Message 3 - Final
        response3 = send_message("These are great suggestions. How do I stay motivated long-term?", session_id=session_id, user_id="conv3-user")
        if "error" not in response3:
            print_message("👤 User", "These are great suggestions. How do I stay motivated long-term?")
            print_message("Assistant", response3.get("response", ""), f"{selected_model.upper()}, Turn 3")
    
    print(f"\n✅ Conversation completed with {selected_model.upper()}")

def run_conversation_4():
    """Relationship advice - might favor different model"""
    print_conversation_header("Relationship Problems", "Relationship Support")
    
    # Message 1 - Initial selection
    response1 = send_message("My partner and I have been arguing a lot lately and I don't know how to fix our relationship.", user_id="conv4-user")
    if "error" in response1:
        print(f"❌ Error: {response1['error']}")
        return
    
    session_id = response1.get("session_id")
    selected_model = response1.get("selected_model", "unknown")
    model_scores = response1.get("model_scores", {})
    confidence = response1.get("confidence_score", 0)
    
    print(f"🎯 Model Selection: {selected_model.upper()} (confidence: {confidence:.1%})")
    print(f"📊 Scores: {model_scores}")
    
    print_message("👤 User", "My partner and I have been arguing a lot lately and I don't know how to fix our relationship.")
    print_message("Assistant", response1.get("response", ""), f"{selected_model.upper()}, Turn 1")
    
    time.sleep(1)
    
    # Message 2 - Continuation
    response2 = send_message("We seem to communicate differently and misunderstand each other often.", session_id=session_id, user_id="conv4-user")
    if "error" not in response2:
        print_message("👤 User", "We seem to communicate differently and misunderstand each other often.")
        print_message("Assistant", response2.get("response", ""), f"{selected_model.upper()}, Turn 2")
        
        time.sleep(1)
        
        # Message 3 - Final
        response3 = send_message("Should we consider couples therapy?", session_id=session_id, user_id="conv4-user")
        if "error" not in response3:
            print_message("👤 User", "Should we consider couples therapy?")
            print_message("Assistant", response3.get("response", ""), f"{selected_model.upper()}, Turn 3")
    
    print(f"\n✅ Conversation completed with {selected_model.upper()}")

def main():
    """Run all test conversations"""
    print("🧪 TESTING FULL CONVERSATIONS WITH UNBIASED MODEL SELECTION")
    print("=" * 80)
    print("Purpose: Verify different models are selected based on scenario type")
    print("Expected: DeepSeek for technical, various models for other scenarios")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/api/status", timeout=5)
        response.raise_for_status()
        print("✅ Chat server is running")
    except:
        print("❌ Chat server is not running. Please start it first.")
        return
    
    # Run all conversations
    try:
        run_conversation_1()  # Information seeking - expect DeepSeek
        run_conversation_2()  # Anxiety - could be any model
        run_conversation_3()  # General support - could be any model  
        run_conversation_4()  # Relationship - could be any model
        
        print("\n" + "="*80)
        print("🎯 TESTING COMPLETE")
        print("="*80)
        print("✅ All conversations completed successfully")
        print("📊 Check the model selections above to verify diversity")
        print("🔍 Look for different models being chosen for different scenario types")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed: {e}")

if __name__ == "__main__":
    main()