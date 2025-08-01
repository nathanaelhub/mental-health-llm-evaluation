#!/usr/bin/env python3
"""
Debug Session Continuation Issues
=================================

This script tests the conversation flow to identify where session management
is breaking and why the selected model isn't being maintained.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, Optional

class SessionDebugger:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"debug-session-{int(time.time())}"
        
    async def test_conversation_flow(self):
        """Test complete conversation flow and identify issues"""
        print("🔍 DEBUGGING SESSION CONTINUATION")
        print("=" * 50)
        print(f"Session ID: {self.session_id}")
        print(f"Base URL: {self.base_url}")
        print()
        
        # Test server availability first
        if not await self._test_server_health():
            return
            
        # Step 1: First message (model selection)
        first_response = await self._send_first_message()
        if not first_response:
            return
            
        # Step 2: Check session state after first message
        await self._check_session_state("after first message")
        
        # Step 3: Second message (should continue with same model)
        second_response = await self._send_follow_up_message()
        if not second_response:
            return
            
        # Step 4: Analyze the results
        await self._analyze_conversation_flow(first_response, second_response)
        
        # Step 5: Test edge cases
        await self._test_edge_cases()
        
    async def _test_server_health(self) -> bool:
        """Check if server is running and healthy"""
        print("🏥 Testing server health...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/health", timeout=5) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        print(f"✅ Server healthy: {health_data.get('status', 'unknown')}")
                        return True
                    else:
                        print(f"❌ Server unhealthy: HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("💡 Make sure to run: python chat_server.py")
            return False
    
    async def _send_first_message(self) -> Optional[Dict[str, Any]]:
        """Send first message to trigger model selection"""
        print("1️⃣ Sending first message (should trigger model selection)...")
        
        request_data = {
            "message": "I'm feeling really anxious about my upcoming presentation at work",
            "session_id": self.session_id,
            "user_id": "debug-user"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=request_data,
                    timeout=120  # Long timeout for model evaluation
                ) as response:
                    
                    response_data = await response.json()
                    
                    print(f"   Status: {response.status}")
                    print(f"   Selected Model: {response_data.get('selected_model', 'NONE')}")
                    print(f"   Confidence: {response_data.get('confidence_score', 0):.1%}")
                    print(f"   Mode: {response_data.get('conversation_mode', 'unknown')}")
                    print(f"   Session ID: {response_data.get('session_id', 'NONE')}")
                    print(f"   Is New Session: {response_data.get('is_new_session', 'unknown')}")
                    print(f"   Turn Number: {response_data.get('turn_number', 'unknown')}")
                    
                    if response.status == 200:
                        print("   ✅ First message successful")
                        return response_data
                    else:
                        print(f"   ❌ First message failed: {response_data}")
                        return None
                        
        except asyncio.TimeoutError:
            print("   ❌ First message timed out (>120s)")
            return None
        except Exception as e:
            print(f"   ❌ First message error: {e}")
            return None
    
    async def _check_session_state(self, context: str):
        """Check the current session state"""
        print(f"\\n🔍 Checking session state ({context})...")
        
        # Try to get session info if endpoint exists
        endpoints_to_try = [
            f"/api/session/{self.session_id}",
            f"/api/sessions/{self.session_id}",
            f"/api/sessions/debug/{self.session_id}"
        ]
        
        session_found = False
        for endpoint in endpoints_to_try:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        if response.status == 200:
                            session_data = await response.json()
                            print(f"   ✅ Session found at {endpoint}")
                            print(f"   Stored Model: {session_data.get('selected_model', 'NONE')}")
                            print(f"   Turn Count: {session_data.get('turn_count', 'unknown')}")
                            print(f"   Created: {session_data.get('created_at', 'unknown')}")
                            session_found = True
                            break
                        elif response.status == 404:
                            continue  # Try next endpoint
                        else:
                            print(f"   ⚠️  {endpoint}: HTTP {response.status}")
            except Exception as e:
                print(f"   ⚠️  {endpoint}: {e}")
        
        if not session_found:
            print("   ❌ No session endpoint found or session not stored")
            print("   💡 This might indicate session storage issues")
    
    async def _send_follow_up_message(self) -> Optional[Dict[str, Any]]:
        """Send follow-up message to test continuation"""
        print("\\n2️⃣ Sending follow-up message (should continue with same model)...")
        
        request_data = {
            "message": "Yes, can you give me some specific techniques to manage this anxiety?",
            "session_id": self.session_id,  # Same session ID
            "user_id": "debug-user"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=request_data,
                    timeout=120
                ) as response:
                    
                    response_data = await response.json()
                    
                    print(f"   Status: {response.status}")
                    print(f"   Used Model: {response_data.get('selected_model', 'NONE')}")
                    print(f"   Mode: {response_data.get('conversation_mode', 'unknown')}")
                    print(f"   Is New Session: {response_data.get('is_new_session', 'unknown')}")
                    print(f"   Turn Number: {response_data.get('turn_number', 'unknown')}")
                    
                    if response.status == 200:
                        print("   ✅ Follow-up message successful")
                        return response_data
                    else:
                        print(f"   ❌ Follow-up message failed: {response_data}")
                        return None
                        
        except asyncio.TimeoutError:
            print("   ❌ Follow-up message timed out (>120s)")
            return None
        except Exception as e:
            print(f"   ❌ Follow-up message error: {e}")
            return None
    
    async def _analyze_conversation_flow(self, first_response: Dict[str, Any], 
                                       second_response: Dict[str, Any]):
        """Analyze the conversation flow for issues"""
        print("\\n📊 CONVERSATION FLOW ANALYSIS")
        print("=" * 30)
        
        # Extract key information
        first_model = first_response.get('selected_model', 'NONE')
        second_model = second_response.get('selected_model', 'NONE')
        first_mode = first_response.get('conversation_mode', 'unknown')
        second_mode = second_response.get('conversation_mode', 'unknown')
        first_session = first_response.get('session_id', 'NONE')
        second_session = second_response.get('session_id', 'NONE')
        
        print(f"First Message:")
        print(f"   Model: {first_model}")
        print(f"   Mode: {first_mode}")
        print(f"   Session: {first_session}")
        print(f"   New Session: {first_response.get('is_new_session', 'unknown')}")
        
        print(f"\\nSecond Message:")
        print(f"   Model: {second_model}")
        print(f"   Mode: {second_mode}")
        print(f"   Session: {second_session}")
        print(f"   New Session: {second_response.get('is_new_session', 'unknown')}")
        
        # Analysis
        print(f"\\n🔍 DIAGNOSIS:")
        
        issues_found = []
        
        # Check 1: Same model used?
        if first_model == second_model:
            print("✅ Model Consistency: Same model used for both messages")
        else:
            print(f"❌ Model Inconsistency: Changed from {first_model} to {second_model}")
            issues_found.append("model_inconsistency")
        
        # Check 2: Conversation mode progression
        if first_mode == "selection" and second_mode == "continuation":
            print("✅ Mode Progression: Proper selection → continuation flow")
        else:
            print(f"❌ Mode Issue: {first_mode} → {second_mode} (expected: selection → continuation)")
            issues_found.append("mode_progression")
        
        # Check 3: Session consistency
        if first_session == second_session:
            print("✅ Session Consistency: Same session ID maintained")
        else:
            print(f"❌ Session Issue: Changed from {first_session} to {second_session}")
            issues_found.append("session_inconsistency")
        
        # Check 4: New session flags
        if first_response.get('is_new_session') and not second_response.get('is_new_session'):
            print("✅ Session Flags: Proper new → existing session progression")
        else:
            print(f"❌ Session Flag Issue: First={first_response.get('is_new_session')}, Second={second_response.get('is_new_session')}")
            issues_found.append("session_flags")
        
        # Summary
        if not issues_found:
            print("\\n🎉 SUCCESS: Conversation flow is working correctly!")
        else:
            print(f"\\n❌ ISSUES FOUND: {len(issues_found)} problems detected")
            print("💡 POTENTIAL CAUSES:")
            
            if "model_inconsistency" in issues_found:
                print("   • Session not storing selected model properly")
                print("   • Model selection running on every message instead of just first")
                print("   • Session retrieval failing, triggering new selection")
            
            if "mode_progression" in issues_found:
                print("   • Conversation mode logic not working correctly")
                print("   • Session state not being checked properly")
            
            if "session_inconsistency" in issues_found:
                print("   • Session ID generation issues")
                print("   • Session storage/retrieval problems")
            
            if "session_flags" in issues_found:
                print("   • is_new_session flag logic incorrect")
                print("   • Session existence check failing")
    
    async def _test_edge_cases(self):
        """Test edge cases that might break session continuation"""
        print("\\n🧪 TESTING EDGE CASES")
        print("=" * 25)
        
        # Test 1: Invalid session ID
        print("1. Testing with non-existent session ID...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "message": "Hello",
                        "session_id": "non-existent-session-12345",
                        "user_id": "debug-user"
                    },
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Handled gracefully: created new session {data.get('session_id', 'NONE')}")
                    else:
                        print(f"   ❌ Failed to handle: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 2: Missing session ID
        print("\\n2. Testing without session ID...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "message": "Hello",
                        "user_id": "debug-user"
                        # No session_id provided
                    },
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Handled gracefully: created session {data.get('session_id', 'NONE')}")
                    else:
                        print(f"   ❌ Failed to handle: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 3: Force reselection flag
        print("\\n3. Testing force reselection...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "message": "I need different help now",
                        "session_id": self.session_id,
                        "user_id": "debug-user",
                        "force_reselection": True
                    },
                    timeout=120
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Force reselection worked: {data.get('selected_model', 'NONE')}")
                        print(f"   Mode: {data.get('conversation_mode', 'unknown')}")
                    else:
                        print(f"   ❌ Force reselection failed: HTTP {response.status}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

async def main():
    """Main debugging function"""
    debugger = SessionDebugger()
    await debugger.test_conversation_flow()
    
    print("\\n" + "=" * 50)
    print("🔧 DEBUGGING COMPLETE")
    print("💡 If issues were found, check the chat_server.py logs")
    print("💡 Look for session storage/retrieval error messages")

if __name__ == "__main__":
    asyncio.run(main())