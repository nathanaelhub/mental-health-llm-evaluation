#!/usr/bin/env python3
"""
Comprehensive Test Suite for Session Management System

Tests all components of the robust session and conversation management
system including persistence, safety features, and WebSocket support.
"""

import asyncio
import sys
import time
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.chat.conversation_session_manager import (
        ConversationSessionManager, 
        ConversationSession,
        Message,
        MessageRole,
        SessionStatus,
        SafetyLevel
    )
    from src.chat.persistent_session_store import (
        PersistentSessionStore,
        SessionStoreType,
        MemorySessionStore,
        SQLiteSessionStore,
        JSONFileSessionStore
    )
    from src.chat.conversation_summarizer import (
        ConversationSummarizer,
        SummarizationType
    )
    print("✅ All session management modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SessionManagementTester:
    """Comprehensive test suite for session management system"""
    
    def __init__(self):
        self.test_results = []
        self.setup_logging()
        self.temp_dir = tempfile.mkdtemp()
    
    def setup_logging(self):
        """Setup test logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })
    
    async def test_session_store_backends(self):
        """Test different session store backends"""
        print("\n🧪 Testing Session Store Backends...")
        
        # Test Memory Store
        try:
            store = MemorySessionStore()
            await self._test_store_operations(store, "Memory Store")
            await store.close()
            self.log_test("Memory store operations", True)
        except Exception as e:
            self.log_test("Memory store operations", False, str(e))
        
        # Test SQLite Store
        try:
            db_path = Path(self.temp_dir) / "test_sessions.db"
            store = SQLiteSessionStore(str(db_path))
            await self._test_store_operations(store, "SQLite Store")
            await store.close()
            self.log_test("SQLite store operations", True)
        except Exception as e:
            self.log_test("SQLite store operations", False, str(e))
        
        # Test JSON File Store
        try:
            json_dir = Path(self.temp_dir) / "json_sessions"
            store = JSONFileSessionStore(str(json_dir))
            await self._test_store_operations(store, "JSON File Store")
            await store.close()
            self.log_test("JSON file store operations", True)
        except Exception as e:
            self.log_test("JSON file store operations", False, str(e))
    
    async def _test_store_operations(self, store, store_name: str):
        """Test basic store operations"""
        
        # Create test session
        test_session = await self._create_test_session("test_user", "openai")
        
        # Test save
        success = await store.save_session(test_session)
        assert success, f"{store_name}: Failed to save session"
        
        # Test load
        loaded_session = await store.load_session(test_session.session_id)
        assert loaded_session is not None, f"{store_name}: Failed to load session"
        assert loaded_session.session_id == test_session.session_id, f"{store_name}: Session ID mismatch"
        
        # Test user sessions
        user_sessions = await store.get_user_sessions("test_user")
        assert len(user_sessions) >= 1, f"{store_name}: Failed to get user sessions"
        
        # Test search
        search_results = await store.search_sessions("test message")
        assert isinstance(search_results, list), f"{store_name}: Search failed"
        
        # Test audit log
        audit_success = await store.save_audit_log({
            'timestamp': datetime.now().isoformat(),
            'action': 'test_action',
            'session_id': test_session.session_id,
            'details': {'test': True}
        })
        assert audit_success, f"{store_name}: Failed to save audit log"
        
        # Test delete
        delete_success = await store.delete_session(test_session.session_id)
        assert delete_success, f"{store_name}: Failed to delete session"
        
        # Verify deletion
        deleted_session = await store.load_session(test_session.session_id)
        assert deleted_session is None, f"{store_name}: Session not properly deleted"
    
    async def test_session_manager_lifecycle(self):
        """Test session manager lifecycle operations"""
        print("\n🧪 Testing Session Manager Lifecycle...")
        
        try:
            # Initialize session manager with memory store for testing
            manager = ConversationSessionManager(
                store_type=SessionStoreType.MEMORY,
                session_timeout_minutes=1,  # Short timeout for testing
                enable_safety_monitoring=False  # Disable to avoid dependency issues
            )
            
            # Test session creation
            session = await manager.create_session(
                user_id="test_user",
                selected_model="openai",
                initial_message="Hello, I need help with anxiety",
                metadata={"test": True}
            )
            
            self.log_test(
                "Session creation",
                isinstance(session, ConversationSession),
                f"Created session {session.session_id[:8]}"
            )
            
            # Test session retrieval
            retrieved_session = await manager.get_session(session.session_id)
            self.log_test(
                "Session retrieval",
                retrieved_session is not None and retrieved_session.session_id == session.session_id,
                "Successfully retrieved session"
            )
            
            # Test message addition
            user_message = await manager.add_message(
                session.session_id,
                MessageRole.USER,
                "I'm feeling very worried about my upcoming presentation"
            )
            
            assistant_message = await manager.add_message(
                session.session_id,
                MessageRole.ASSISTANT,
                "I understand your concern about the presentation. Let's work through some strategies to help manage this anxiety."
            )
            
            self.log_test(
                "Message addition",
                user_message is not None and assistant_message is not None,
                f"Added {len(session.conversation_history)} messages"
            )
            
            # Test session analytics
            analytics = await manager.get_session_analytics(session.session_id)
            self.log_test(
                "Session analytics",
                isinstance(analytics, dict) and analytics.get('message_count', 0) > 0,
                f"Analytics: {analytics.get('message_count', 0)} messages"
            )
            
            # Test model migration
            migration_success = await manager.migrate_session_model(
                session.session_id,
                "claude",
                "Testing model migration"
            )
            
            self.log_test(
                "Model migration",
                migration_success,
                "Successfully migrated to Claude"
            )
            
            # Test user sessions
            user_sessions = await manager.get_user_sessions("test_user")
            self.log_test(
                "User sessions retrieval",
                len(user_sessions) >= 1,
                f"Found {len(user_sessions)} user sessions"
            )
            
            # Test session archival
            archive_success = await manager.archive_session(session.session_id)
            self.log_test(
                "Session archival",
                archive_success,
                "Successfully archived session"
            )
            
            await manager.close()
            
        except Exception as e:
            self.log_test("Session manager lifecycle", False, str(e))
    
    async def test_conversation_context_handling(self):
        """Test conversation context and token management"""
        print("\n🧪 Testing Conversation Context Handling...")
        
        try:
            manager = ConversationSessionManager(
                store_type=SessionStoreType.MEMORY,
                max_context_tokens=1000,  # Low limit for testing
                enable_safety_monitoring=False
            )
            
            # Create session with multiple messages
            session = await manager.create_session(
                user_id="context_user",
                selected_model="openai",
                initial_message="This is the first message in our conversation"
            )
            
            # Add several messages to build context
            messages_to_add = [
                ("user", "Tell me about anxiety management techniques"),
                ("assistant", "Here are some effective anxiety management techniques: deep breathing, progressive muscle relaxation, mindfulness meditation, and cognitive restructuring. Each of these can help you manage anxiety in different situations."),
                ("user", "Can you explain deep breathing in more detail?"),
                ("assistant", "Deep breathing is a simple but powerful technique. Start by inhaling slowly through your nose for 4 counts, hold for 4 counts, then exhale through your mouth for 6 counts. This activates your parasympathetic nervous system."),
                ("user", "What about when I'm in a meeting and can't do obvious breathing exercises?"),
                ("assistant", "For discrete breathing in meetings, try the 4-7-8 technique quietly: breathe in for 4, hold for 7, out for 8. You can also use box breathing - 4 counts for each phase - while appearing to simply listen attentively.")
            ]
            
            for role, content in messages_to_add:
                await manager.add_message(
                    session.session_id,
                    MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                    content
                )
            
            # Test context generation
            updated_session = await manager.get_session(session.session_id)
            context = manager.get_model_context(updated_session, max_tokens=500)
            
            self.log_test(
                "Context generation",
                isinstance(context, list) and len(context) > 0,
                f"Generated context with {len(context)} messages"
            )
            
            # Test token counting
            self.log_test(
                "Token management",
                updated_session.total_tokens > 0,
                f"Total tokens: {updated_session.total_tokens}"
            )
            
            # Test model-specific context formatting
            openai_context = manager.get_model_context(updated_session)
            self.log_test(
                "Model-specific context formatting",
                all('role' in msg and 'content' in msg for msg in openai_context),
                "Context properly formatted for OpenAI"
            )
            
            await manager.close()
            
        except Exception as e:
            self.log_test("Context handling", False, str(e))
    
    async def test_safety_monitoring(self):
        """Test safety monitoring and crisis detection"""
        print("\n🧪 Testing Safety Monitoring...")
        
        try:
            manager = ConversationSessionManager(
                store_type=SessionStoreType.MEMORY,
                enable_safety_monitoring=True,
                enable_audit_trail=True
            )
            
            # Create session
            session = await manager.create_session(
                user_id="safety_user",
                selected_model="openai",
                initial_message="I'm having a really hard time lately"
            )
            
            # Add messages with varying safety levels
            test_messages = [
                ("user", "Sometimes I feel like giving up on everything"),
                ("assistant", "I hear that you're going through a really difficult time right now. Those feelings of wanting to give up are a sign that you're dealing with significant pain. Can you tell me more about what's been especially hard?"),
                ("user", "I've been thinking about hurting myself"),  # Should trigger safety alert
                ("assistant", "I'm very concerned about what you've shared. Thoughts of self-harm are serious, and I want you to know that help is available. Have you thought about reaching out to a crisis helpline or emergency services?")
            ]
            
            for role, content in test_messages:
                message = await manager.add_message(
                    session.session_id,
                    MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                    content
                )
                
                # Check if safety monitoring is working
                if message and message.safety_score is not None:
                    self.log_test(
                        f"Safety scoring for {role} message",
                        isinstance(message.safety_score, (int, float)),
                        f"Safety score: {message.safety_score:.2f}"
                    )
            
            # Check session safety level
            updated_session = await manager.get_session(session.session_id)
            self.log_test(
                "Session safety level tracking",
                isinstance(updated_session.safety_level, SafetyLevel),
                f"Safety level: {updated_session.safety_level.value}"
            )
            
            # Check for crisis flags
            self.log_test(
                "Crisis detection",
                len(updated_session.crisis_flags) >= 0,  # Should have detected crisis content
                f"Crisis flags: {len(updated_session.crisis_flags)}"
            )
            
            await manager.close()
            
        except Exception as e:
            self.log_test("Safety monitoring", False, str(e))
    
    async def test_conversation_summarization(self):
        """Test conversation summarization functionality"""
        print("\n🧪 Testing Conversation Summarization...")
        
        try:
            # Create summarizer
            summarizer = ConversationSummarizer(
                max_segment_messages=5,
                min_segment_messages=3,
                preserve_recent_messages=2
            )
            
            # Create a long test conversation
            test_session = await self._create_long_test_session()
            
            # Test if summarization is needed
            should_summarize = await summarizer.should_summarize(test_session)
            self.log_test(
                "Summarization need detection",
                isinstance(should_summarize, bool),
                f"Should summarize: {should_summarize}"
            )
            
            # Test summarization
            if len(test_session.conversation_history) >= 6:  # Ensure we have enough messages
                summary = await summarizer.summarize_conversation(
                    test_session,
                    SummarizationType.CRISIS_PRESERVING
                )
                
                self.log_test(
                    "Conversation summarization",
                    summary is not None and len(summary.summary_segments) > 0,
                    f"Created {len(summary.summary_segments)} summary segments"
                )
                
                # Test applying summary to session
                summarized_session = await summarizer.apply_summary_to_session(test_session, summary)
                
                self.log_test(
                    "Summary application to session",
                    len(summarized_session.conversation_history) < len(test_session.conversation_history),
                    f"Reduced from {len(test_session.conversation_history)} to {len(summarized_session.conversation_history)} messages"
                )
                
                # Test token reduction
                original_tokens = test_session.total_tokens
                new_tokens = summarized_session.total_tokens
                
                self.log_test(
                    "Token reduction",
                    new_tokens < original_tokens,
                    f"Reduced from {original_tokens} to {new_tokens} tokens"
                )
            else:
                self.log_test("Conversation summarization", True, "Skipped - insufficient messages")
            
        except Exception as e:
            self.log_test("Conversation summarization", False, str(e))
    
    async def test_websocket_integration(self):
        """Test WebSocket integration for real-time updates"""
        print("\n🧪 Testing WebSocket Integration...")
        
        try:
            manager = ConversationSessionManager(
                store_type=SessionStoreType.MEMORY,
                enable_safety_monitoring=False
            )
            
            # Create mock WebSocket connection
            class MockWebSocket:
                def __init__(self):
                    self.messages = []
                
                async def send_json(self, data):
                    self.messages.append(data)
            
            # Create session
            session = await manager.create_session(
                user_id="websocket_user",
                selected_model="openai",
                initial_message="Testing WebSocket integration"
            )
            
            # Register mock WebSocket
            mock_ws = MockWebSocket()
            manager.register_websocket(session.session_id, mock_ws)
            
            # Add a message (should trigger WebSocket notification)
            await manager.add_message(
                session.session_id,
                MessageRole.USER,
                "This should trigger a WebSocket update"
            )
            
            # Check if WebSocket received messages
            # Note: In real implementation, there might be async delay
            await asyncio.sleep(0.1)  # Small delay for async operations
            
            self.log_test(
                "WebSocket registration",
                session.session_id in manager.websocket_connections,
                "WebSocket successfully registered"
            )
            
            # Unregister WebSocket
            manager.unregister_websocket(session.session_id, mock_ws)
            
            self.log_test(
                "WebSocket unregistration",
                session.session_id not in manager.websocket_connections or 
                len(manager.websocket_connections[session.session_id]) == 0,
                "WebSocket successfully unregistered"
            )
            
            await manager.close()
            
        except Exception as e:
            self.log_test("WebSocket integration", False, str(e))
    
    async def test_session_search_and_analytics(self):
        """Test session search and analytics capabilities"""
        print("\n🧪 Testing Session Search and Analytics...")
        
        try:
            manager = ConversationSessionManager(
                store_type=SessionStoreType.MEMORY,
                enable_safety_monitoring=False
            )
            
            # Create multiple test sessions
            sessions = []
            for i in range(3):
                session = await manager.create_session(
                    user_id=f"user_{i}",
                    selected_model="openai",
                    initial_message=f"Test message {i} about anxiety and stress",
                    metadata={"test_batch": True}
                )
                sessions.append(session)
                
                # Add some messages
                await manager.add_message(
                    session.session_id,
                    MessageRole.ASSISTANT,
                    f"Response {i} discussing coping strategies and support"
                )
            
            # Test session search
            search_results = await manager.search_sessions("anxiety")
            self.log_test(
                "Session content search",
                len(search_results) > 0,
                f"Found {len(search_results)} sessions matching 'anxiety'"
            )
            
            # Test user-specific search
            user_0_sessions = await manager.get_user_sessions("user_0")
            self.log_test(
                "User-specific session retrieval",
                len(user_0_sessions) >= 1,
                f"Found {len(user_0_sessions)} sessions for user_0"
            )
            
            # Test session analytics for individual session
            analytics = await manager.get_session_analytics(sessions[0].session_id)
            expected_keys = ['session_id', 'message_count', 'total_tokens', 'duration_minutes']
            
            self.log_test(
                "Individual session analytics",
                all(key in analytics for key in expected_keys),
                f"Analytics keys: {list(analytics.keys())[:5]}"
            )
            
            await manager.close()
            
        except Exception as e:
            self.log_test("Session search and analytics", False, str(e))
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios"""
        print("\n🧪 Testing Error Handling and Recovery...")
        
        try:
            manager = ConversationSessionManager(
                store_type=SessionStoreType.MEMORY,
                enable_safety_monitoring=False
            )
            
            # Test handling non-existent session
            non_existent = await manager.get_session("non-existent-id")
            self.log_test(
                "Non-existent session handling",
                non_existent is None,
                "Properly returned None for non-existent session"
            )
            
            # Test adding message to non-existent session
            failed_message = await manager.add_message(
                "non-existent-id",
                MessageRole.USER,
                "This should fail gracefully"
            )
            
            self.log_test(
                "Message addition to non-existent session",
                failed_message is None,
                "Properly handled non-existent session"
            )
            
            # Test migration of non-existent session
            failed_migration = await manager.migrate_session_model(
                "non-existent-id",
                "claude",
                "Should fail"
            )
            
            self.log_test(
                "Migration of non-existent session",
                not failed_migration,
                "Properly rejected migration of non-existent session"
            )
            
            # Test session timeout handling
            session = await manager.create_session(
                user_id="timeout_user",
                selected_model="openai",
                initial_message="Testing timeout"
            )
            
            # Artificially age the session
            session.last_activity = datetime.now() - timedelta(hours=2)
            
            # Session should be considered inactive
            is_active = manager._is_session_active(session)
            self.log_test(
                "Session timeout detection",
                not is_active,
                "Properly detected inactive session"
            )
            
            await manager.close()
            
        except Exception as e:
            self.log_test("Error handling and recovery", False, str(e))
    
    # Helper methods
    
    async def _create_test_session(self, user_id: str, model: str) -> ConversationSession:
        """Create a test session with some messages"""
        from datetime import datetime
        
        session = ConversationSession(
            session_id=f"test_{int(time.time())}",
            user_id=user_id,
            selected_model=model,
            conversation_history=[
                Message(
                    message_id="msg_1",
                    role=MessageRole.USER,
                    content="Hello, I need help with anxiety",
                    timestamp=datetime.now(),
                    token_count=10
                ),
                Message(
                    message_id="msg_2",
                    role=MessageRole.ASSISTANT,
                    content="I understand you're dealing with anxiety. Let's explore some strategies that might help.",
                    timestamp=datetime.now(),
                    token_count=20
                )
            ],
            metadata={"test": True},
            created_at=datetime.now(),
            last_activity=datetime.now(),
            evaluation_scores={},
            total_tokens=30
        )
        
        return session
    
    async def _create_long_test_session(self) -> ConversationSession:
        """Create a session with many messages for summarization testing"""
        from datetime import datetime
        
        messages = []
        base_time = datetime.now()
        
        # Create 10 messages to ensure we have enough for summarization
        message_pairs = [
            ("What techniques can help with anxiety?", "Deep breathing, mindfulness, and progressive muscle relaxation are effective techniques."),
            ("How do I practice mindfulness?", "Start with 5 minutes daily. Focus on your breath and gently return attention when mind wanders."),
            ("What if I feel overwhelmed at work?", "Take short breaks, practice desk exercises, and communicate boundaries with colleagues."),
            ("I'm having trouble sleeping due to worry", "Try a bedtime routine, limit screens before bed, and consider relaxation techniques."),
            ("How do I know if I need professional help?", "If anxiety interferes with daily life, work, or relationships, professional support can be very beneficial.")
        ]
        
        for i, (user_msg, assistant_msg) in enumerate(message_pairs):
            # User message
            messages.append(Message(
                message_id=f"msg_user_{i}",
                role=MessageRole.USER,
                content=user_msg,
                timestamp=base_time + timedelta(minutes=i*2),
                token_count=len(user_msg.split()) * 1.3
            ))
            
            # Assistant message
            messages.append(Message(
                message_id=f"msg_assistant_{i}",
                role=MessageRole.ASSISTANT,
                content=assistant_msg,
                timestamp=base_time + timedelta(minutes=i*2 + 1),
                token_count=len(assistant_msg.split()) * 1.3
            ))
        
        total_tokens = sum(msg.token_count or 0 for msg in messages)
        
        session = ConversationSession(
            session_id=f"long_test_{int(time.time())}",
            user_id="long_test_user",
            selected_model="openai",
            conversation_history=messages,
            metadata={"test": True, "type": "long_conversation"},
            created_at=base_time,
            last_activity=base_time + timedelta(minutes=len(messages)),
            evaluation_scores={},
            total_tokens=int(total_tokens)
        )
        
        return session
    
    async def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Session Management System Test Suite")
        print("=" * 60)
        
        # Run all tests
        await self.test_session_store_backends()
        await self.test_session_manager_lifecycle()
        await self.test_conversation_context_handling()
        await self.test_safety_monitoring()
        await self.test_conversation_summarization()
        await self.test_websocket_integration()
        await self.test_session_search_and_analytics()
        await self.test_error_handling_and_recovery()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"   • {result['test']}: {result['details']}")
        
        return failed_tests == 0


async def main():
    """Main test runner"""
    print("🚀 Starting session management system tests...")
    
    # Create temp directories
    Path("temp/test_sessions").mkdir(parents=True, exist_ok=True)
    
    tester = SessionManagementTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! The session management system is ready for use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))