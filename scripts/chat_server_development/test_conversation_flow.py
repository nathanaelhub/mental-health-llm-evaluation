#!/usr/bin/env python3
"""
Test Conversation Flow - Backend Verification
=============================================

Test script to verify the updated backend conversation continuation logic:
1. First message: Model selection
2. Subsequent messages: Model continuation  
3. New session: Fresh model selection
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any

class ConversationFlowTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.session = None
        
    async def test_conversation_flow(self):
        """Test complete conversation flow"""
        print("🧠 TESTING CONVERSATION FLOW - BACKEND VERIFICATION")
        print("=" * 60)
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Test 1: First message (should trigger model selection)
            await self.test_first_message()
            
            # Test 2: Subsequent messages (should continue with selected model)
            await self.test_continuation_messages()
            
            # Test 3: New session (should trigger fresh model selection)
            await self.test_new_session()
    
    async def test_first_message(self):
        """Test first message triggers model selection"""
        print("\n🔍 TEST 1: FIRST MESSAGE - MODEL SELECTION")
        print("-" * 40)
        
        payload = {
            "message": "I'm feeling very anxious about my job interview tomorrow",
            "user_id": "test-user",
            "session_id": None,  # No session ID = first message
            "force_reselection": False
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    print("✅ First message successful")
                    print(f"   Response: '{data.get('response', '')[:60]}...'")
                    print(f"   Selected model: {data.get('selected_model', 'None')}")
                    print(f"   Session ID: {data.get('session_id', 'None')[:8]}...")
                    print(f"   Conversation mode: {data.get('conversation_mode', 'None')}")
                    print(f"   Turn number: {data.get('turn_number', 'None')}")
                    print(f"   Confidence: {data.get('confidence_score', 0) * 100:.1f}%")
                    print(f"   Is new session: {data.get('is_new_session', False)}")
                    print(f"   Can reset: {data.get('can_reset', False)}")
                    
                    # Verify expected fields for first message
                    assert data.get('conversation_mode') == 'selection', f"Expected 'selection', got '{data.get('conversation_mode')}'"
                    assert data.get('is_new_session') == True, "Expected is_new_session=True"
                    assert data.get('turn_number') == 1, f"Expected turn_number=1, got {data.get('turn_number')}"
                    assert data.get('selected_model') is not None, "Expected selected_model to be set"
                    assert data.get('session_id') is not None, "Expected session_id to be set"
                    
                    # Store for next test
                    self.test_session_id = data.get('session_id')
                    self.test_selected_model = data.get('selected_model')
                    
                    print("   ✅ All assertions passed for first message")
                    
                else:
                    print(f"❌ First message failed with status {response.status}")
                    text = await response.text()
                    print(f"   Error: {text}")
                    
        except Exception as e:
            print(f"❌ First message test failed: {e}")
    
    async def test_continuation_messages(self):
        """Test subsequent messages continue with selected model"""
        print("\n💬 TEST 2: CONTINUATION MESSAGES - MODEL PERSISTENCE")
        print("-" * 40)
        
        if not hasattr(self, 'test_session_id'):
            print("❌ Skipping continuation test - no session from first message")
            return
        
        continuation_messages = [
            "What techniques can help me calm down?",
            "How long should I practice these exercises?",
            "Thank you, that's very helpful advice"
        ]
        
        for i, message in enumerate(continuation_messages, 2):  # Start from turn 2
            print(f"\n   Turn {i}: '{message[:40]}...'")
            
            payload = {
                "message": message,
                "user_id": "test-user",
                "session_id": self.test_session_id,  # Use existing session
                "force_reselection": False
            }
            
            try:
                async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        print(f"   ✅ Turn {i} successful")
                        print(f"      Response: '{data.get('response', '')[:50]}...'")
                        print(f"      Model: {data.get('selected_model', 'None')}")
                        print(f"      Mode: {data.get('conversation_mode', 'None')}")
                        print(f"      Turn: {data.get('turn_number', 'None')}")
                        print(f"      Same session: {data.get('session_id') == self.test_session_id}")
                        
                        # Verify continuation behavior
                        assert data.get('conversation_mode') == 'continuation', f"Expected 'continuation', got '{data.get('conversation_mode')}'"
                        assert data.get('selected_model') == self.test_selected_model, f"Model changed! Expected {self.test_selected_model}, got {data.get('selected_model')}"
                        assert data.get('session_id') == self.test_session_id, "Session ID changed!"
                        assert data.get('turn_number') == i, f"Expected turn_number={i}, got {data.get('turn_number')}"
                        assert data.get('is_new_session') == False, "Expected is_new_session=False"
                        
                        print(f"      ✅ All assertions passed for turn {i}")
                        
                    else:
                        print(f"   ❌ Turn {i} failed with status {response.status}")
                        text = await response.text()
                        print(f"      Error: {text}")
                        break
                        
            except Exception as e:
                print(f"   ❌ Turn {i} test failed: {e}")
                break
    
    async def test_new_session(self):
        """Test new session triggers fresh model selection"""
        print("\n🆕 TEST 3: NEW SESSION - FRESH MODEL SELECTION")
        print("-" * 40)
        
        payload = {
            "message": "I'm dealing with depression and need support",
            "user_id": "test-user-2",  # Different user
            "session_id": None,  # No session ID = new session
            "force_reselection": False
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    print("✅ New session successful")
                    print(f"   Response: '{data.get('response', '')[:60]}...'")
                    print(f"   Selected model: {data.get('selected_model', 'None')}")
                    print(f"   Session ID: {data.get('session_id', 'None')[:8]}...")
                    print(f"   Conversation mode: {data.get('conversation_mode', 'None')}")
                    print(f"   Turn number: {data.get('turn_number', 'None')}")
                    print(f"   Different session: {data.get('session_id') != getattr(self, 'test_session_id', None)}")
                    
                    # Verify new session behavior
                    assert data.get('conversation_mode') == 'selection', f"Expected 'selection', got '{data.get('conversation_mode')}'"
                    assert data.get('is_new_session') == True, "Expected is_new_session=True"
                    assert data.get('turn_number') == 1, f"Expected turn_number=1, got {data.get('turn_number')}"
                    assert data.get('session_id') != getattr(self, 'test_session_id', None), "Session ID should be different"
                    
                    print("   ✅ All assertions passed for new session")
                    
                else:
                    print(f"❌ New session failed with status {response.status}")
                    text = await response.text()
                    print(f"   Error: {text}")
                    
        except Exception as e:
            print(f"❌ New session test failed: {e}")
    
    async def test_server_status(self):
        """Test server status endpoint"""
        print("\n🔍 TESTING SERVER STATUS")
        print("-" * 40)
        
        try:
            async with self.session.get(f"{self.base_url}/api/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Server status healthy")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Available models: {len(data.get('available_models', []))}")
                    print(f"   Models: {', '.join(data.get('available_models', []))}")
                else:
                    print(f"❌ Server status failed: {response.status}")
        except Exception as e:
            print(f"❌ Server status test failed: {e}")

async def main():
    """Run all tests"""
    tester = ConversationFlowTester()
    
    # Test server status first
    async with aiohttp.ClientSession() as session:
        tester.session = session
        await tester.test_server_status()
    
    # Test conversation flow
    await tester.test_conversation_flow()
    
    print("\n" + "=" * 60)
    print("🎯 CONVERSATION FLOW TESTING COMPLETE")
    print("=" * 60)
    print("\n✅ Expected Behavior:")
    print("   1. First message: conversation_mode='selection', turn_number=1")
    print("   2. Continuation: conversation_mode='continuation', same model")
    print("   3. New session: conversation_mode='selection', fresh selection")
    print("\n🚀 Ready for frontend testing!")
    print("   python simple_chat_server.py")
    print("   Open: http://localhost:8000")

if __name__ == "__main__":
    asyncio.run(main())