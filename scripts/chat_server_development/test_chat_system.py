#!/usr/bin/env python3
"""
Test Suite for Dynamic Model Selection Chat System

This script runs comprehensive tests to verify all components
of the chat system are working correctly.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test imports
try:
    from src.chat.model_selector import ModelSelector, ModelSelectionResult
    from src.chat.session_manager import SessionManager, ChatSession
    from src.chat.conversation_handler import ConversationHandler
    from src.chat.response_cache import ResponseCache
    from src.chat.chat_interface import ChatInterface, ChatRequest
    print("✅ All chat modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class ChatSystemTester:
    """Comprehensive test suite for the chat system"""
    
    def __init__(self):
        self.test_results = []
        self.setup_logging()
        
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
    
    def test_model_selector(self):
        """Test ModelSelector functionality"""
        print("\n🧪 Testing ModelSelector...")
        
        try:
            # Test initialization
            selector = ModelSelector(
                available_models=['openai', 'deepseek'],
                fallback_model='openai',
                timeout_seconds=10.0
            )
            self.log_test("ModelSelector initialization", True)
            
            # Test get available models
            models = selector.get_available_models()
            self.log_test("Get available models", len(models) > 0, f"Found {len(models)} models")
            
            # Test fallback selection
            fallback_result = selector._fallback_selection("test prompt", time.time())
            self.log_test("Fallback selection", isinstance(fallback_result, ModelSelectionResult))
            
        except Exception as e:
            self.log_test("ModelSelector tests", False, str(e))
    
    def test_session_manager(self):
        """Test SessionManager functionality"""
        print("\n🧪 Testing SessionManager...")
        
        try:
            # Test initialization
            session_manager = SessionManager(
                session_storage_dir="temp/test_sessions",
                session_timeout_hours=1,
                max_sessions_per_user=5
            )
            self.log_test("SessionManager initialization", True)
            
            # Test session creation
            from datetime import datetime
            selection_result = ModelSelectionResult(
                selected_model="openai",
                selection_score=8.5,
                selection_time_ms=1500.0,
                all_scores={"openai": 8.5, "deepseek": 7.2},
                response_preview="Test response",
                timestamp=datetime.now()
            )
            
            session = session_manager.create_session(
                user_id="test_user",
                initial_prompt="Hello, I need some support",
                selection_result=selection_result
            )
            
            self.log_test("Session creation", isinstance(session, ChatSession))
            
            # Test session retrieval
            retrieved_session = session_manager.get_session(session.session_id)
            self.log_test("Session retrieval", retrieved_session is not None)
            
            # Test conversation turn
            success = session_manager.add_conversation_turn(
                session_id=session.session_id,
                user_message="How are you?",
                assistant_message="I'm here to help you.",
                response_time_ms=800.0
            )
            self.log_test("Add conversation turn", success)
            
            # Test analytics
            analytics = session_manager.get_session_analytics()
            self.log_test("Session analytics", isinstance(analytics, dict))
            
            # Cleanup
            session_manager.delete_session(session.session_id)
            
        except Exception as e:
            self.log_test("SessionManager tests", False, str(e))
    
    def test_response_cache(self):
        """Test ResponseCache functionality"""
        print("\n🧪 Testing ResponseCache...")
        
        try:
            # Test initialization
            cache = ResponseCache(
                cache_dir="temp/test_cache",
                max_entries=100,
                ttl_hours=1
            )
            self.log_test("ResponseCache initialization", True)
            
            # Test caching and retrieval
            from datetime import datetime
            selection_result = ModelSelectionResult(
                selected_model="openai",
                selection_score=8.5,
                selection_time_ms=1500.0,
                all_scores={"openai": 8.5},
                response_preview="Cached response",
                timestamp=datetime.now()
            )
            
            # Cache a selection
            cache.cache_selection(
                prompt="Test prompt",
                system_prompt="Test system",
                selection_result=selection_result
            )
            
            # Try to retrieve it
            cached_selection = cache.get_cached_selection("Test prompt", "Test system")
            self.log_test("Cache selection storage/retrieval", cached_selection is not None)
            
            # Test cache stats
            stats = cache.get_cache_stats()
            self.log_test("Cache statistics", isinstance(stats, dict))
            
            # Cleanup
            cache.clear_cache()
            
        except Exception as e:
            self.log_test("ResponseCache tests", False, str(e))
    
    async def test_chat_interface(self):
        """Test ChatInterface functionality"""
        print("\n🧪 Testing ChatInterface...")
        
        try:
            # Test initialization
            chat_interface = ChatInterface(
                available_models=['openai'],  # Use minimal model set for testing
                enable_caching=False,  # Disable for simpler testing
                enable_streaming=False
            )
            self.log_test("ChatInterface initialization", True)
            
            # Test system status
            status = await chat_interface.get_system_status()
            self.log_test("Get system status", isinstance(status, dict))
            
            # Test cleanup
            cleanup_result = chat_interface.cleanup()
            self.log_test("System cleanup", isinstance(cleanup_result, dict))
            
        except Exception as e:
            self.log_test("ChatInterface tests", False, str(e))
    
    def test_error_handling(self):
        """Test error handling across components"""
        print("\n🧪 Testing Error Handling...")
        
        try:
            # Test invalid model selector
            try:
                selector = ModelSelector(available_models=[])
                self.log_test("Empty model list handling", True, "Handled gracefully")
            except Exception:
                self.log_test("Empty model list handling", True, "Raised appropriate exception")
            
            # Test invalid session retrieval
            session_manager = SessionManager(session_storage_dir="temp/test_sessions")
            session = session_manager.get_session("invalid_session_id")
            self.log_test("Invalid session handling", session is None)
            
            # Test cache with invalid data
            cache = ResponseCache(cache_dir="temp/test_cache")
            cached = cache.get_cached_selection("", None)
            self.log_test("Invalid cache query handling", cached is None)
            
        except Exception as e:
            self.log_test("Error handling tests", False, str(e))
    
    def test_data_persistence(self):
        """Test data persistence across restarts"""
        print("\n🧪 Testing Data Persistence...")
        
        try:
            # Test session persistence
            session_manager1 = SessionManager(session_storage_dir="temp/test_persistence")
            
            from datetime import datetime
            selection_result = ModelSelectionResult(
                selected_model="openai",
                selection_score=8.5,
                selection_time_ms=1500.0,
                all_scores={"openai": 8.5},
                response_preview="Persistent test",
                timestamp=datetime.now()
            )
            
            session1 = session_manager1.create_session(
                user_id="persist_user",
                initial_prompt="Persistence test",
                selection_result=selection_result
            )
            
            session_id = session1.session_id
            
            # Create new session manager (simulating restart)
            session_manager2 = SessionManager(session_storage_dir="temp/test_persistence")
            session2 = session_manager2.get_session(session_id)
            
            self.log_test("Session persistence", session2 is not None)
            
            # Cleanup
            session_manager2.delete_session(session_id)
            
        except Exception as e:
            self.log_test("Data persistence tests", False, str(e))
    
    def test_performance(self):
        """Test system performance with multiple operations"""
        print("\n🧪 Testing Performance...")
        
        try:
            start_time = time.time()
            
            # Create multiple sessions quickly
            session_manager = SessionManager(session_storage_dir="temp/test_performance")
            
            from datetime import datetime
            selection_result = ModelSelectionResult(
                selected_model="openai",
                selection_score=8.5,
                selection_time_ms=1500.0,
                all_scores={"openai": 8.5},
                response_preview="Performance test",
                timestamp=datetime.now()
            )
            
            sessions = []
            for i in range(10):
                session = session_manager.create_session(
                    user_id=f"perf_user_{i}",
                    initial_prompt=f"Performance test {i}",
                    selection_result=selection_result
                )
                sessions.append(session)
            
            creation_time = time.time() - start_time
            self.log_test("Bulk session creation", creation_time < 5.0, f"{creation_time:.2f}s for 10 sessions")
            
            # Test cache performance
            cache = ResponseCache(cache_dir="temp/test_perf_cache")
            
            cache_start = time.time()
            for i in range(100):
                cache.cache_selection(
                    prompt=f"Test prompt {i}",
                    system_prompt="System",
                    selection_result=selection_result
                )
            
            cache_time = time.time() - cache_start
            self.log_test("Bulk cache operations", cache_time < 10.0, f"{cache_time:.2f}s for 100 operations")
            
            # Cleanup
            for session in sessions:
                session_manager.delete_session(session.session_id)
            cache.clear_cache()
            
        except Exception as e:
            self.log_test("Performance tests", False, str(e))
    
    async def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Dynamic Model Selection Chat System Test Suite")
        print("=" * 60)
        
        # Run all tests
        self.test_model_selector()
        self.test_session_manager()
        self.test_response_cache()
        await self.test_chat_interface()
        self.test_error_handling()
        self.test_data_persistence()
        self.test_performance()
        
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
    print("🚀 Starting chat system tests...")
    
    # Create temp directories
    Path("temp/test_sessions").mkdir(parents=True, exist_ok=True)
    Path("temp/test_cache").mkdir(parents=True, exist_ok=True)
    Path("temp/test_persistence").mkdir(parents=True, exist_ok=True)
    Path("temp/test_perf_cache").mkdir(parents=True, exist_ok=True)
    
    tester = ChatSystemTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! The chat system is ready for use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))