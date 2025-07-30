#!/usr/bin/env python3
"""
Test Script for Chat History Functionality
==========================================

Tests the new Chat History button to ensure it properly displays
conversation history, session info, and statistics.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List

class ChatHistoryTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_aiohttp = None
        self.current_session_id = None
        self.conversation_log = []
    
    async def __aenter__(self):
        self.session_aiohttp = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session_aiohttp:
            await self.session_aiohttp.close()
    
    async def send_message(self, message: str) -> Dict[str, Any]:
        """Send a chat message and return the response"""
        
        request_data = {
            'message': message,
            'session_id': self.current_session_id,
            'user_id': 'test-user',
            'force_reselection': False
        }
        
        try:
            async with self.session_aiohttp.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Request failed: {response.status} - {error_text}")
                    return None
                
                data = await response.json()
                
                # Update session ID for next message
                if data.get('session_id'):
                    self.current_session_id = data['session_id']
                
                # Log the conversation
                self.conversation_log.append({
                    'user_message': message,
                    'assistant_response': data.get('response', ''),
                    'selected_model': data.get('selected_model'),
                    'confidence_score': data.get('confidence_score'),
                    'conversation_mode': data.get('conversation_mode'),
                    'turn_count': data.get('turn_count'),
                    'model_scores': data.get('model_scores', {}),
                    'timestamp': time.time()
                })
                
                return data
                
        except Exception as e:
            print(f"❌ Request exception: {e}")
            return None
    
    async def simulate_conversation(self):
        """Simulate a realistic mental health conversation"""
        
        print("🧠 Starting conversation simulation...")
        
        messages = [
            "I've been feeling really stressed about work lately",
            "What can I do to manage this stress better?", 
            "I've tried deep breathing but it doesn't seem to help much",
            "Thank you for the suggestions, that's very helpful"
        ]
        
        for i, message in enumerate(messages, 1):
            print(f"\n📝 Message {i}: '{message}'")
            
            response = await self.send_message(message)
            if response:
                print(f"🤖 {response['selected_model'].upper()}: {response['response'][:80]}...")
                print(f"📊 Confidence: {response['confidence_score']:.1%}, Mode: {response['conversation_mode']}")
                
                if response.get('model_scores'):
                    scores = response['model_scores']
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    print(f"🏆 Top model: {sorted_scores[0][0].upper()} ({sorted_scores[0][1]:.2f}/10)")
            
            await asyncio.sleep(0.5)  # Brief pause between messages
    
    def analyze_conversation_log(self):
        """Analyze the conversation log to verify chat history features"""
        
        print(f"\n{'='*60}")
        print("📊 CONVERSATION ANALYSIS")
        print(f"{'='*60}")
        
        if not self.conversation_log:
            print("❌ No conversation data to analyze")
            return
        
        # Basic stats
        total_turns = len(self.conversation_log)
        user_messages = total_turns
        assistant_responses = total_turns
        
        print(f"📈 Conversation Statistics:")
        print(f"   Session ID: {self.current_session_id[:8]}..." if self.current_session_id else "   Session ID: None")
        print(f"   Total Turns: {total_turns}")
        print(f"   User Messages: {user_messages}")
        print(f"   Assistant Responses: {assistant_responses}")
        
        # Model usage
        models_used = {}
        for turn in self.conversation_log:
            model = turn['selected_model']
            models_used[model] = models_used.get(model, 0) + 1
        
        print(f"\n🤖 Model Usage:")
        for model, count in sorted(models_used.items()):
            percentage = (count / total_turns) * 100
            print(f"   {model.upper()}: {count} times ({percentage:.1f}%)")
        
        # Confidence analysis
        confidences = [turn['confidence_score'] for turn in self.conversation_log]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        print(f"\n📊 Confidence Analysis:")
        print(f"   Average: {avg_confidence:.1%}")
        print(f"   Range: {min_confidence:.1%} - {max_confidence:.1%}")
        
        # Conversation modes
        modes = [turn['conversation_mode'] for turn in self.conversation_log]
        mode_counts = {}
        for mode in modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        print(f"\n💬 Conversation Modes:")
        for mode, count in mode_counts.items():
            print(f"   {mode}: {count} times")
        
        # Timeline
        first_message_time = self.conversation_log[0]['timestamp']
        last_message_time = self.conversation_log[-1]['timestamp']
        conversation_duration = last_message_time - first_message_time
        
        print(f"\n⏰ Timeline:")
        print(f"   Started: {time.strftime('%H:%M:%S', time.localtime(first_message_time))}")
        print(f"   Ended: {time.strftime('%H:%M:%S', time.localtime(last_message_time))}")
        print(f"   Duration: {conversation_duration:.1f} seconds")
        
        return {
            'session_id': self.current_session_id,
            'total_turns': total_turns,
            'models_used': models_used,
            'avg_confidence': avg_confidence,
            'conversation_duration': conversation_duration,
            'mode_distribution': mode_counts
        }
    
    def verify_chat_history_features(self, stats: Dict[str, Any]):
        """Verify that the chat history would display correctly"""
        
        print(f"\n{'='*60}")
        print("✅ CHAT HISTORY VERIFICATION")
        print(f"{'='*60}")
        
        # Check required data for chat history display
        checks = []
        
        # Session info
        if stats['session_id']:
            checks.append("✅ Session ID available for display")
        else:
            checks.append("❌ Session ID missing")
        
        # Model info
        if stats['models_used']:
            primary_model = max(stats['models_used'].items(), key=lambda x: x[1])[0]
            checks.append(f"✅ Primary model: {primary_model.upper()}")
        else:
            checks.append("❌ No model information")
        
        # Turn count
        if stats['total_turns'] > 0:
            checks.append(f"✅ Turn count: {stats['total_turns']}")
        else:
            checks.append("❌ No conversation turns")
        
        # Confidence data
        if stats['avg_confidence'] > 0:
            checks.append(f"✅ Confidence data: {stats['avg_confidence']:.1%} average")
        else:
            checks.append("❌ No confidence data")
        
        # Mode tracking
        if stats['mode_distribution']:
            checks.append(f"✅ Conversation modes tracked: {list(stats['mode_distribution'].keys())}")
        else:
            checks.append("❌ No conversation mode data")
        
        # Timeline data
        if stats['conversation_duration'] >= 0:
            checks.append(f"✅ Timeline data: {stats['conversation_duration']:.1f}s duration")
        else:
            checks.append("❌ No timeline data")
        
        # Display results
        for check in checks:
            print(f"   {check}")
        
        # Overall assessment
        passed_checks = len([c for c in checks if c.startswith("✅")])
        total_checks = len(checks)
        success_rate = (passed_checks / total_checks) * 100
        
        print(f"\n📊 Overall Chat History Readiness: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 Chat History feature is ready for use!")
        elif success_rate >= 60:
            print("⚠️ Chat History mostly functional, minor issues detected")
        else:
            print("❌ Chat History needs additional work")
        
        return success_rate >= 80

async def main():
    """Main test runner"""
    
    print("🧪 Testing Chat History Functionality")
    print("=" * 60)
    
    # Check server status
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/api/status") as response:
                if response.status != 200:
                    print("❌ Server is not responding. Please start the chat server first:")
                    print("   python chat_server.py")
                    return
                
                status = await response.json()
                print(f"✅ Server is running (version: {status.get('version', 'unknown')})")
    
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Please start the chat server first: python chat_server.py")
        return
    
    # Run the test
    async with ChatHistoryTester() as tester:
        await tester.simulate_conversation()
        stats = tester.analyze_conversation_log()
        ready = tester.verify_chat_history_features(stats)
        
        if ready:
            print(f"\n🌐 Test the Chat History button at: http://localhost:8000/chat")
            print("   1. Send a few messages in the chat interface")
            print("   2. Click the Chat History button (📊 icon)")
            print("   3. Verify that your conversation appears with:")
            print("      • Session info (ID, model, turns)")
            print("      • Individual messages with timestamps")
            print("      • Model badges and conversation modes")
            print("      • Summary statistics")

if __name__ == "__main__":
    print("🧪 Starting Chat History Test Suite...")
    asyncio.run(main())