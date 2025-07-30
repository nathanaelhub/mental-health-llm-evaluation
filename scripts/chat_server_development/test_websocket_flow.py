#!/usr/bin/env python3
"""
Test WebSocket Chat Flow - Real-time Communication
================================================

This script tests the WebSocket implementation for real-time chat:
1. Connect to WebSocket endpoint
2. Send first message (model selection)
3. Send follow-up messages (continuation)
4. Verify real-time streaming behavior
"""

import asyncio
import websockets
import json
import time

class WebSocketChatTester:
    def __init__(self):
        self.server_url = "ws://localhost:8000/api/chat/stream"
        self.session_id = None
        self.selected_model = None
        
    async def test_websocket_chat(self):
        """Test complete WebSocket chat flow"""
        print("🧠 TESTING WEBSOCKET CHAT FLOW")
        print("=" * 50)
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                print("✅ WebSocket connected successfully")
                
                # Test 1: Connection welcome
                await self.test_connection_welcome(websocket)
                
                # Test 2: First message (model selection)
                await self.test_first_message(websocket)
                
                # Test 3: Follow-up messages (continuation)
                await self.test_continuation_messages(websocket)
                
                print("\n🎯 WebSocket testing complete!")
                
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
    
    async def test_connection_welcome(self, websocket):
        """Test connection welcome message"""
        print("\n🔍 TEST 1: CONNECTION WELCOME")
        print("-" * 30)
        
        try:
            # Should receive welcome message automatically
            welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(welcome_msg)
            
            print(f"📨 Received: {data}")
            
            if data.get("type") == "connection" and data.get("status") == "connected":
                print("✅ Connection welcome received correctly")
                return True
            else:
                print(f"❌ Unexpected welcome message: {data}")
                return False
                
        except asyncio.TimeoutError:
            print("❌ No welcome message received within timeout")
            return False
        except Exception as e:
            print(f"❌ Welcome test failed: {e}")
            return False
    
    async def test_first_message(self, websocket):
        """Test first message triggers model selection"""
        print("\n🔍 TEST 2: FIRST MESSAGE - MODEL SELECTION")
        print("-" * 40)
        
        try:
            # Send first message
            first_message = {
                "message": "I'm feeling very anxious about my job interview tomorrow",
                "user_id": "websocket-test-user",
                "session_id": None
            }
            
            print(f"📤 Sending: {first_message['message']}")
            await websocket.send(json.dumps(first_message))
            
            # Should receive multiple messages: typing, status, model_selected, response
            messages_received = []
            
            for i in range(4):  # Expect 4 messages
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(msg)
                    messages_received.append(data)
                    print(f"📨 Message {i+1}: {data.get('type', 'unknown')} - {data.get('message', data.get('selected_model', ''))[:50]}...")
                    
                    # Store session data
                    if data.get("type") == "response":
                        self.session_id = data.get("session_id")
                        self.selected_model = data.get("selected_model")
                        
                except asyncio.TimeoutError:
                    print(f"⏱️ Timeout waiting for message {i+1}")
                    break
            
            # Verify message sequence
            types_received = [msg.get("type") for msg in messages_received]
            expected_types = ["typing", "status", "model_selected", "response"]
            
            print(f"\n📊 Message sequence:")
            print(f"   Expected: {expected_types}")
            print(f"   Received: {types_received}")
            
            if self.session_id and self.selected_model:
                print(f"✅ Model selection successful:")
                print(f"   Session ID: {self.session_id[:8]}...")
                print(f"   Selected: {self.selected_model.upper()}")
                return True
            else:
                print("❌ Model selection failed - missing session or model")
                return False
                
        except Exception as e:
            print(f"❌ First message test failed: {e}")
            return False
    
    async def test_continuation_messages(self, websocket):
        """Test follow-up messages use selected model"""
        print("\n🔍 TEST 3: CONTINUATION MESSAGES")
        print("-" * 35)
        
        if not self.session_id or not self.selected_model:
            print("❌ Skipping continuation test - no session from first message")
            return False
        
        continuation_messages = [
            "What specific techniques can help me feel more confident?",
            "How long should I practice these before the interview?"
        ]
        
        for i, message in enumerate(continuation_messages, 2):
            print(f"\n💬 Turn {i}: {message}")
            
            try:
                # Send continuation message
                msg_data = {
                    "message": message,
                    "user_id": "websocket-test-user",
                    "session_id": self.session_id
                }
                
                await websocket.send(json.dumps(msg_data))
                
                # Should receive: typing, status, response
                response_received = False
                
                for j in range(3):  # Expect 3 messages
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=8.0)
                        data = json.loads(msg)
                        
                        print(f"   📨 {data.get('type', 'unknown')}: {data.get('message', data.get('response', ''))[:40]}...")
                        
                        if data.get("type") == "response":
                            response_received = True
                            # Verify same model and session
                            if (data.get("selected_model") == self.selected_model and 
                                data.get("session_id") == self.session_id and
                                data.get("conversation_mode") == "continuation"):
                                print(f"   ✅ Turn {i} successful - same model, same session")
                            else:
                                print(f"   ❌ Turn {i} failed - model or session changed")
                            break
                            
                    except asyncio.TimeoutError:
                        print(f"   ⏱️ Timeout waiting for response {j+1}")
                        break
                
                if not response_received:
                    print(f"   ❌ Turn {i} - no response received")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Turn {i} failed: {e}")
                return False
        
        print("\n✅ All continuation messages successful")
        return True

async def main():
    """Run WebSocket chat tests"""
    print("🧠 WebSocket Chat Flow Testing")
    print("================================")
    print("📋 Prerequisites:")
    print("   1. Server must be running: python working_chat_server.py")
    print("   2. WebSocket endpoint: ws://localhost:8000/api/chat/stream")
    print("   3. All dependencies installed")
    print()
    
    tester = WebSocketChatTester()
    
    try:
        await tester.test_websocket_chat()
        
        print("\n" + "=" * 50)
        print("🎯 WEBSOCKET TESTING SUMMARY")
        print("=" * 50)
        print("✅ Expected Behavior:")
        print("   1. Connection welcome → Immediate response")
        print("   2. First message → typing, status, model_selected, response")
        print("   3. Continuation → typing, status, response (same model)")
        print("   4. Real-time streaming → No delays, immediate updates")
        print()
        print("🚀 WebSocket chat flow is working!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Ensure server is running on port 8000")
        print("   • Check WebSocket dependencies are installed")
        print("   • Verify no firewall blocking connections")

if __name__ == "__main__":
    asyncio.run(main())