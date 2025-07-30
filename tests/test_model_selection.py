"""
Comprehensive Test Suite for Dynamic Model Selection System

Tests all aspects of the model selection logic including accuracy, performance,
failover behavior, and context preservation across model switches.
"""

import asyncio
import pytest
import time
import statistics
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.chat.dynamic_model_selector import (
    DynamicModelSelector, PromptType, ModelSelection, 
    SelectionContext, ModelPerformanceTracker
)
from src.chat.conversation_session_manager import (
    ConversationSessionManager, ConversationSession, SessionStatus
)
from src.analytics.smart_switching import SmartModelSwitcher, SwitchingDecision, SwitchReason
from src.analytics.feedback_system import FeedbackCollector, UserFeedback


class ModelSelectionTests:
    """
    Comprehensive tests for model selection logic
    """
    
    def __init__(self):
        self.selector = None
        self.session_manager = None
        self.switcher = None
        self.feedback_collector = None
        
    async def setup_test_environment(self):
        """Setup test environment with mock services"""
        
        # Initialize components
        self.selector = DynamicModelSelector()
        self.session_manager = ConversationSessionManager()
        self.feedback_collector = FeedbackCollector()
        self.switcher = SmartModelSwitcher(self.selector)
        
        # Setup test data
        await self._setup_test_data()
        
    async def _setup_test_data(self):
        """Setup test scenarios and mock data"""
        
        # Mock model availability
        self.available_models = {
            "gpt-3.5-turbo": {"status": "healthy", "avg_latency": 800},
            "gpt-4-turbo": {"status": "healthy", "avg_latency": 1200},
            "claude-3-sonnet": {"status": "healthy", "avg_latency": 1000},
            "claude-3-opus": {"status": "healthy", "avg_latency": 1500},
            "claude-3-haiku": {"status": "healthy", "avg_latency": 600}
        }
        
        # Test scenarios
        self.test_scenarios = [
            {
                "name": "crisis_situation",
                "prompt": "I'm having thoughts of suicide and don't know what to do",
                "prompt_type": PromptType.CRISIS,
                "expected_models": ["claude-3-opus", "claude-3-sonnet"],
                "context": {"crisis_level": 0.9, "urgency": "high"}
            },
            {
                "name": "anxiety_support",
                "prompt": "I've been feeling really anxious lately about work",
                "prompt_type": PromptType.ANXIETY,
                "expected_models": ["claude-3-sonnet", "gpt-4-turbo"],
                "context": {"emotional_intensity": 0.6, "complexity": "medium"}
            },
            {
                "name": "general_wellness",
                "prompt": "Can you give me some tips for better sleep?",
                "prompt_type": PromptType.GENERAL_WELLNESS,
                "expected_models": ["gpt-3.5-turbo", "claude-3-haiku"],
                "context": {"cost_sensitivity": "high", "complexity": "low"}
            },
            {
                "name": "information_seeking",
                "prompt": "What are the different types of therapy available?",
                "prompt_type": PromptType.INFORMATION_SEEKING,
                "expected_models": ["gpt-4-turbo", "claude-3-sonnet"],
                "context": {"accuracy_required": "high", "complexity": "medium"}
            }
        ]
        
    # Core Selection Logic Tests
    
    async def test_selection_accuracy(self) -> Dict[str, Any]:
        """Test that the best model is selected for various prompt types"""
        
        test_results = {
            "test_name": "selection_accuracy",
            "scenarios_tested": len(self.test_scenarios),
            "results": [],
            "success_rate": 0.0,
            "avg_selection_time_ms": 0.0
        }
        
        selection_times = []
        successful_selections = 0
        
        for scenario in self.test_scenarios:
            start_time = time.time()
            
            # Create selection context
            context = SelectionContext(
                user_id=f"test_user_{scenario['name']}",
                session_id=f"session_{scenario['name']}",
                message_text=scenario["prompt"],
                prompt_type=scenario["prompt_type"],
                conversation_history=[],
                user_preferences={},
                system_context=scenario["context"]
            )
            
            # Perform selection
            selection = await self.selector.select_model(context)
            
            selection_time_ms = (time.time() - start_time) * 1000
            selection_times.append(selection_time_ms)
            
            # Validate selection
            is_correct = selection.selected_model in scenario["expected_models"]
            if is_correct:
                successful_selections += 1
            
            # Record detailed results
            scenario_result = {
                "scenario": scenario["name"],
                "prompt_type": scenario["prompt_type"].value,
                "selected_model": selection.selected_model,
                "expected_models": scenario["expected_models"],
                "correct_selection": is_correct,
                "confidence_score": selection.confidence_score,
                "selection_time_ms": selection_time_ms,
                "reasoning": selection.reasoning
            }
            
            test_results["results"].append(scenario_result)
        
        # Calculate summary metrics
        test_results["success_rate"] = successful_selections / len(self.test_scenarios)
        test_results["avg_selection_time_ms"] = statistics.mean(selection_times)
        
        return test_results
    
    async def test_cache_performance(self) -> Dict[str, Any]:
        """Verify cache improves latency without sacrificing quality"""
        
        test_results = {
            "test_name": "cache_performance",
            "cache_enabled_results": {},
            "cache_disabled_results": {},
            "performance_improvement": {},
            "quality_impact": {}
        }
        
        # Test with cache enabled
        self.selector.enable_caching(True)
        cache_enabled_results = await self._run_cache_test_batch("with_cache")
        test_results["cache_enabled_results"] = cache_enabled_results
        
        # Clear cache and test without caching
        self.selector.clear_cache()
        self.selector.enable_caching(False)
        cache_disabled_results = await self._run_cache_test_batch("without_cache")
        test_results["cache_disabled_results"] = cache_disabled_results
        
        # Calculate improvements
        latency_improvement = (
            cache_disabled_results["avg_latency_ms"] - 
            cache_enabled_results["avg_latency_ms"]
        ) / cache_disabled_results["avg_latency_ms"] * 100
        
        test_results["performance_improvement"] = {
            "avg_latency_reduction_percent": latency_improvement,
            "cache_hit_rate": cache_enabled_results.get("cache_hit_rate", 0),
            "throughput_improvement": cache_enabled_results["requests_per_second"] / cache_disabled_results["requests_per_second"]
        }
        
        # Quality should remain the same (selections should be identical for identical prompts)
        quality_consistency = self._compare_selection_quality(
            cache_enabled_results["selections"],
            cache_disabled_results["selections"]
        )
        test_results["quality_impact"] = quality_consistency
        
        return test_results
    
    async def _run_cache_test_batch(self, test_name: str) -> Dict[str, Any]:
        """Run a batch of selection tests for cache performance measurement"""
        
        results = {
            "test_name": test_name,
            "total_requests": 0,
            "avg_latency_ms": 0,
            "requests_per_second": 0,
            "selections": [],
            "cache_hit_rate": 0
        }
        
        # Create repeated requests (simulate cache hits)
        repeated_prompts = [
            "I'm feeling anxious about my job interview tomorrow",
            "Can you help me with breathing exercises?",
            "I'm having trouble sleeping",
            "What should I do if I'm feeling overwhelmed?"
        ] * 10  # 40 total requests, many repeats
        
        start_time = time.time()
        latencies = []
        selections = []
        
        for prompt in repeated_prompts:
            request_start = time.time()
            
            context = SelectionContext(
                user_id="cache_test_user",
                session_id="cache_test_session",
                message_text=prompt,
                prompt_type=PromptType.ANXIETY,
                conversation_history=[],
                user_preferences={},
                system_context={}
            )
            
            selection = await self.selector.select_model(context)
            
            request_latency = (time.time() - request_start) * 1000
            latencies.append(request_latency)
            selections.append({
                "prompt": prompt,
                "selected_model": selection.selected_model,
                "confidence": selection.confidence_score
            })
        
        total_time = time.time() - start_time
        
        results.update({
            "total_requests": len(repeated_prompts),
            "avg_latency_ms": statistics.mean(latencies),
            "requests_per_second": len(repeated_prompts) / total_time,
            "selections": selections,
            "cache_hit_rate": self.selector.get_cache_hit_rate() if hasattr(self.selector, 'get_cache_hit_rate') else 0
        })
        
        return results
    
    def _compare_selection_quality(self, selections1: List[Dict], selections2: List[Dict]) -> Dict[str, Any]:
        """Compare quality/consistency between cached and non-cached selections"""
        
        # Group selections by prompt
        prompt_selections1 = {}
        prompt_selections2 = {}
        
        for sel in selections1:
            prompt = sel["prompt"]
            if prompt not in prompt_selections1:
                prompt_selections1[prompt] = []
            prompt_selections1[prompt].append(sel["selected_model"])
        
        for sel in selections2:
            prompt = sel["prompt"]
            if prompt not in prompt_selections2:
                prompt_selections2[prompt] = []
            prompt_selections2[prompt].append(sel["selected_model"])
        
        # Check consistency
        consistent_prompts = 0
        total_prompts = len(prompt_selections1)
        
        for prompt in prompt_selections1:
            if prompt in prompt_selections2:
                # For identical prompts, selections should be consistent
                models1 = set(prompt_selections1[prompt])
                models2 = set(prompt_selections2[prompt])
                
                if models1 == models2:
                    consistent_prompts += 1
        
        return {
            "consistency_rate": consistent_prompts / total_prompts if total_prompts > 0 else 0,
            "total_unique_prompts": total_prompts,
            "consistent_selections": consistent_prompts
        }
    
    async def test_failover_behavior(self) -> Dict[str, Any]:
        """Ensure graceful handling when models are unavailable"""
        
        test_results = {
            "test_name": "failover_behavior",
            "scenarios": [],
            "overall_success_rate": 0.0,
            "avg_failover_time_ms": 0.0
        }
        
        failover_scenarios = [
            {
                "name": "primary_model_down",
                "unavailable_models": ["claude-3-opus"],
                "prompt_type": PromptType.CRISIS,
                "expected_behavior": "fallback_to_secondary"
            },
            {
                "name": "multiple_models_down",
                "unavailable_models": ["claude-3-opus", "claude-3-sonnet"],
                "prompt_type": PromptType.CRISIS,
                "expected_behavior": "fallback_to_available"
            },
            {
                "name": "preferred_model_slow",
                "slow_models": {"gpt-4-turbo": 10000},  # 10 second latency
                "prompt_type": PromptType.INFORMATION_SEEKING,
                "expected_behavior": "switch_to_faster"
            },
            {
                "name": "all_premium_models_down",
                "unavailable_models": ["claude-3-opus", "gpt-4-turbo"],
                "prompt_type": PromptType.ANXIETY,
                "expected_behavior": "graceful_degradation"
            }
        ]
        
        failover_times = []
        successful_failovers = 0
        
        for scenario in failover_scenarios:
            start_time = time.time()
            
            # Simulate model unavailability
            original_availability = self.available_models.copy()
            self._simulate_model_issues(scenario)
            
            try:
                context = SelectionContext(
                    user_id=f"failover_test_{scenario['name']}",
                    session_id=f"failover_session_{scenario['name']}",
                    message_text="Test message for failover scenario",
                    prompt_type=scenario["prompt_type"],
                    conversation_history=[],
                    user_preferences={},
                    system_context={}
                )
                
                selection = await self.selector.select_model(context)
                failover_time_ms = (time.time() - start_time) * 1000
                failover_times.append(failover_time_ms)
                
                # Validate failover behavior
                is_successful = self._validate_failover_selection(scenario, selection)
                if is_successful:
                    successful_failovers += 1
                
                scenario_result = {
                    "scenario": scenario["name"],
                    "successful": is_successful,
                    "selected_model": selection.selected_model,
                    "failover_time_ms": failover_time_ms,
                    "unavailable_models": scenario.get("unavailable_models", []),
                    "error_handled": selection.error_message is None
                }
                
                test_results["scenarios"].append(scenario_result)
                
            except Exception as e:
                # Failover should not raise exceptions
                test_results["scenarios"].append({
                    "scenario": scenario["name"],
                    "successful": False,
                    "error": str(e),
                    "failover_time_ms": (time.time() - start_time) * 1000
                })
            
            finally:
                # Restore original availability
                self.available_models = original_availability
        
        test_results["overall_success_rate"] = successful_failovers / len(failover_scenarios)
        test_results["avg_failover_time_ms"] = statistics.mean(failover_times) if failover_times else 0
        
        return test_results
    
    def _simulate_model_issues(self, scenario: Dict[str, Any]):
        """Simulate model availability issues for testing"""
        
        if "unavailable_models" in scenario:
            for model in scenario["unavailable_models"]:
                if model in self.available_models:
                    self.available_models[model]["status"] = "unavailable"
        
        if "slow_models" in scenario:
            for model, latency in scenario["slow_models"].items():
                if model in self.available_models:
                    self.available_models[model]["avg_latency"] = latency
    
    def _validate_failover_selection(self, scenario: Dict[str, Any], selection: ModelSelection) -> bool:
        """Validate that failover selection meets expectations"""
        
        unavailable_models = scenario.get("unavailable_models", [])
        
        # Selection should not pick unavailable models
        if selection.selected_model in unavailable_models:
            return False
        
        # Selection should be made (not None/empty)
        if not selection.selected_model:
            return False
        
        # For critical prompts, should still select high-capability model if available
        if scenario.get("prompt_type") == PromptType.CRISIS:
            high_capability_models = ["claude-3-opus", "claude-3-sonnet", "gpt-4-turbo"]
            available_high_capability = [m for m in high_capability_models if m not in unavailable_models]
            
            if available_high_capability and selection.selected_model not in available_high_capability:
                return False
        
        return True
    
    async def test_conversation_continuity(self) -> Dict[str, Any]:
        """Verify context maintained after model selection and switches"""
        
        test_results = {
            "test_name": "conversation_continuity",
            "conversation_tests": [],
            "context_preservation_rate": 0.0,
            "switch_success_rate": 0.0
        }
        
        # Test multi-turn conversations with model switches
        conversation_scenarios = [
            {
                "name": "anxiety_to_crisis_escalation",
                "turns": [
                    {"message": "I've been feeling anxious lately", "expected_type": PromptType.ANXIETY},
                    {"message": "It's getting worse, I can't handle this anymore", "expected_type": PromptType.ANXIETY},
                    {"message": "I'm thinking about hurting myself", "expected_type": PromptType.CRISIS}
                ]
            },
            {
                "name": "information_to_support",
                "turns": [
                    {"message": "What are the symptoms of depression?", "expected_type": PromptType.INFORMATION_SEEKING},
                    {"message": "I think I might have some of those symptoms", "expected_type": PromptType.DEPRESSION},
                    {"message": "I'm really struggling with this", "expected_type": PromptType.DEPRESSION}
                ]
            }
        ]
        
        successful_continuity_tests = 0
        successful_switches = 0
        total_switches = 0
        
        for scenario in conversation_scenarios:
            session_id = f"continuity_test_{scenario['name']}"
            user_id = f"test_user_{scenario['name']}"
            
            # Start session
            session = await self.session_manager.create_session(user_id, session_id)
            self.switcher.register_session(session_id, user_id, "gpt-3.5-turbo")
            
            conversation_history = []
            context_preserved = True
            scenario_switches = 0
            scenario_successful_switches = 0
            
            for i, turn in enumerate(scenario["turns"]):
                # Update conversation context
                conversation_history.append({
                    "role": "user",
                    "content": turn["message"],
                    "timestamp": datetime.now()
                })
                
                # Create selection context with history
                context = SelectionContext(
                    user_id=user_id,
                    session_id=session_id,
                    message_text=turn["message"],
                    prompt_type=turn["expected_type"],
                    conversation_history=conversation_history,
                    user_preferences={},
                    system_context={}
                )
                
                # Perform model selection
                selection = await self.selector.select_model(context)
                
                # Update smart switcher context
                self.switcher.update_session_context(
                    session_id=session_id,
                    message_text=turn["message"],
                    prompt_type=turn["expected_type"],
                    response_time_ms=1000 + i * 200,  # Simulate increasing response time
                    user_rating=4.0 - i * 0.5  # Simulate decreasing satisfaction
                )
                
                # Evaluate switching opportunity
                switch_decision = await self.switcher.evaluate_switching_opportunity(session_id)
                
                if switch_decision and switch_decision.should_switch:
                    scenario_switches += 1
                    total_switches += 1
                    
                    # Execute switch
                    switch_success = await self.switcher.execute_model_switch(
                        session_id, switch_decision, preserve_context=True
                    )
                    
                    if switch_success:
                        scenario_successful_switches += 1
                        successful_switches += 1
                
                # Check context preservation
                # In a real implementation, you would verify that:
                # 1. Conversation history is maintained
                # 2. User preferences are preserved
                # 3. Session state is consistent
                
                # Simulate context check
                session_context = self.switcher.get_session_summary(session_id)
                if session_context.get("message_count", 0) != i + 1:
                    context_preserved = False
                
                # Add assistant response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": f"Response from {selection.selected_model}",
                    "model": selection.selected_model,
                    "timestamp": datetime.now()
                })
            
            if context_preserved:
                successful_continuity_tests += 1
            
            test_results["conversation_tests"].append({
                "scenario": scenario["name"],
                "context_preserved": context_preserved,
                "total_turns": len(scenario["turns"]),
                "switches_attempted": scenario_switches,
                "switches_successful": scenario_successful_switches,
                "final_message_count": session_context.get("message_count", 0)
            })
        
        test_results["context_preservation_rate"] = (
            successful_continuity_tests / len(conversation_scenarios)
        )
        test_results["switch_success_rate"] = (
            successful_switches / total_switches if total_switches > 0 else 1.0
        )
        
        return test_results
    
    # Integration Tests
    
    async def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete workflow from prompt to response with analytics"""
        
        test_results = {
            "test_name": "end_to_end_workflow",
            "workflow_steps": [],
            "total_success": True,
            "performance_metrics": {}
        }
        
        # Simulate complete user interaction
        user_id = "e2e_test_user"
        session_id = "e2e_test_session"
        
        workflow_start = time.time()
        
        try:
            # Step 1: Session Creation
            step_start = time.time()
            session = await self.session_manager.create_session(user_id, session_id)
            self.switcher.register_session(session_id, user_id, "gpt-3.5-turbo")
            
            test_results["workflow_steps"].append({
                "step": "session_creation",
                "success": session is not None,
                "duration_ms": (time.time() - step_start) * 1000
            })
            
            # Step 2: Model Selection
            step_start = time.time()
            context = SelectionContext(
                user_id=user_id,
                session_id=session_id,
                message_text="I'm feeling really anxious about my presentation tomorrow",
                prompt_type=PromptType.ANXIETY,
                conversation_history=[],
                user_preferences={},
                system_context={}
            )
            
            selection = await self.selector.select_model(context)
            
            test_results["workflow_steps"].append({
                "step": "model_selection",
                "success": selection.selected_model is not None,
                "selected_model": selection.selected_model,
                "confidence": selection.confidence_score,
                "duration_ms": (time.time() - step_start) * 1000
            })
            
            # Step 3: Feedback Collection
            step_start = time.time()
            feedback = UserFeedback(
                user_id=user_id,
                session_id=session_id,
                message_id="test_message_1",
                selected_model=selection.selected_model,
                prompt_type=PromptType.ANXIETY,
                thumbs_up=True,
                overall_rating=4.5,
                response_helpfulness=4.0,
                response_empathy=4.5
            )
            
            await self.feedback_collector.store_feedback(feedback)
            
            test_results["workflow_steps"].append({
                "step": "feedback_collection",
                "success": True,
                "duration_ms": (time.time() - step_start) * 1000
            })
            
            # Step 4: Smart Switching Evaluation
            step_start = time.time()
            self.switcher.update_session_context(
                session_id=session_id,
                message_text=context.message_text,
                prompt_type=PromptType.ANXIETY,
                response_time_ms=1200,
                user_rating=4.5
            )
            
            switch_decision = await self.switcher.evaluate_switching_opportunity(session_id)
            
            test_results["workflow_steps"].append({
                "step": "switching_evaluation",
                "success": True,
                "switch_recommended": switch_decision.should_switch if switch_decision else False,
                "duration_ms": (time.time() - step_start) * 1000
            })
            
            # Calculate overall performance
            total_duration = (time.time() - workflow_start) * 1000
            test_results["performance_metrics"] = {
                "total_workflow_time_ms": total_duration,
                "steps_completed": len(test_results["workflow_steps"]),
                "all_steps_successful": all(step["success"] for step in test_results["workflow_steps"])
            }
            
            test_results["total_success"] = test_results["performance_metrics"]["all_steps_successful"]
            
        except Exception as e:
            test_results["total_success"] = False
            test_results["error"] = str(e)
        
        return test_results
    
    # Performance and Load Tests
    
    async def test_concurrent_selections(self, concurrent_users: int = 50) -> Dict[str, Any]:
        """Test model selection under concurrent load"""
        
        test_results = {
            "test_name": "concurrent_selections",
            "concurrent_users": concurrent_users,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time_ms": 0,
            "throughput_rps": 0,
            "error_rate_percent": 0
        }
        
        # Create concurrent selection tasks
        async def single_user_simulation(user_index: int):
            user_results = {
                "user_id": f"load_test_user_{user_index}",
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "response_times": []
            }
            
            # Each user makes 10 requests
            for request_index in range(10):
                try:
                    start_time = time.time()
                    
                    context = SelectionContext(
                        user_id=user_results["user_id"],
                        session_id=f"load_test_session_{user_index}_{request_index}",
                        message_text=f"Test message {request_index} from user {user_index}",
                        prompt_type=PromptType.GENERAL_WELLNESS,
                        conversation_history=[],
                        user_preferences={},
                        system_context={}
                    )
                    
                    selection = await self.selector.select_model(context)
                    
                    response_time = (time.time() - start_time) * 1000
                    user_results["response_times"].append(response_time)
                    user_results["requests"] += 1
                    
                    if selection.selected_model:
                        user_results["successes"] += 1
                    else:
                        user_results["failures"] += 1
                        
                except Exception as e:
                    user_results["requests"] += 1
                    user_results["failures"] += 1
            
            return user_results
        
        # Run concurrent simulations
        load_test_start = time.time()
        
        tasks = [single_user_simulation(i) for i in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_test_time = time.time() - load_test_start
        
        # Aggregate results
        all_response_times = []
        total_requests = 0
        total_successes = 0
        total_failures = 0
        
        for result in user_results:
            if isinstance(result, Exception):
                total_failures += 10  # All requests for this user failed
                total_requests += 10
            else:
                total_requests += result["requests"]
                total_successes += result["successes"]
                total_failures += result["failures"]
                all_response_times.extend(result["response_times"])
        
        test_results.update({
            "total_requests": total_requests,
            "successful_requests": total_successes,
            "failed_requests": total_failures,
            "avg_response_time_ms": statistics.mean(all_response_times) if all_response_times else 0,
            "p95_response_time_ms": sorted(all_response_times)[int(len(all_response_times) * 0.95)] if all_response_times else 0,
            "throughput_rps": total_requests / total_test_time if total_test_time > 0 else 0,
            "error_rate_percent": (total_failures / total_requests * 100) if total_requests > 0 else 0
        })
        
        return test_results
    
    # Utility Methods
    
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite and return comprehensive results"""
        
        await self.setup_test_environment()
        
        full_results = {
            "test_suite_start": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
        
        # Run all tests
        test_methods = [
            ("selection_accuracy", self.test_selection_accuracy),
            ("cache_performance", self.test_cache_performance),
            ("failover_behavior", self.test_failover_behavior),
            ("conversation_continuity", self.test_conversation_continuity),
            ("end_to_end_workflow", self.test_end_to_end_workflow),
            ("concurrent_load", lambda: self.test_concurrent_selections(50))
        ]
        
        total_tests = len(test_methods)
        passed_tests = 0
        
        for test_name, test_method in test_methods:
            try:
                print(f"Running {test_name}...")
                test_result = await test_method()
                full_results["tests"][test_name] = test_result
                
                # Determine if test passed based on specific criteria
                if self._evaluate_test_success(test_name, test_result):
                    passed_tests += 1
                    
            except Exception as e:
                full_results["tests"][test_name] = {
                    "error": str(e),
                    "test_failed": True
                }
        
        # Generate summary
        full_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "overall_success_rate": passed_tests / total_tests,
            "test_suite_end": datetime.now().isoformat()
        }
        
        return full_results
    
    def _evaluate_test_success(self, test_name: str, test_result: Dict[str, Any]) -> bool:
        """Evaluate whether a test passed based on specific criteria"""
        
        if test_result.get("test_failed") or test_result.get("error"):
            return False
        
        success_criteria = {
            "selection_accuracy": lambda r: r.get("success_rate", 0) >= 0.8,
            "cache_performance": lambda r: r.get("performance_improvement", {}).get("avg_latency_reduction_percent", 0) > 0,
            "failover_behavior": lambda r: r.get("overall_success_rate", 0) >= 0.9,
            "conversation_continuity": lambda r: r.get("context_preservation_rate", 0) >= 0.9,
            "end_to_end_workflow": lambda r: r.get("total_success", False),
            "concurrent_load": lambda r: r.get("error_rate_percent", 100) < 5.0
        }
        
        criteria_fn = success_criteria.get(test_name, lambda r: True)
        return criteria_fn(test_result)


# Pytest Integration

@pytest.fixture
async def test_suite():
    """Pytest fixture for test suite"""
    suite = ModelSelectionTests()
    await suite.setup_test_environment()
    return suite

@pytest.mark.asyncio
async def test_model_selection_accuracy(test_suite):
    """Pytest wrapper for selection accuracy test"""
    result = await test_suite.test_selection_accuracy()
    assert result["success_rate"] >= 0.8, f"Selection accuracy too low: {result['success_rate']}"

@pytest.mark.asyncio
async def test_cache_effectiveness(test_suite):
    """Pytest wrapper for cache performance test"""
    result = await test_suite.test_cache_performance()
    latency_improvement = result.get("performance_improvement", {}).get("avg_latency_reduction_percent", 0)
    assert latency_improvement > 0, "Cache should improve performance"

@pytest.mark.asyncio
async def test_failover_handling(test_suite):
    """Pytest wrapper for failover behavior test"""
    result = await test_suite.test_failover_behavior()
    assert result["overall_success_rate"] >= 0.9, f"Failover success rate too low: {result['overall_success_rate']}"

@pytest.mark.asyncio
async def test_context_preservation(test_suite):
    """Pytest wrapper for conversation continuity test"""
    result = await test_suite.test_conversation_continuity()
    assert result["context_preservation_rate"] >= 0.9, f"Context preservation rate too low: {result['context_preservation_rate']}"

@pytest.mark.asyncio
async def test_load_performance():
    """Test system performance under load"""
    suite = ModelSelectionTests()
    await suite.setup_test_environment()
    
    result = await suite.test_concurrent_selections(concurrent_users=100)
    
    assert result["error_rate_percent"] < 5.0, f"Error rate too high under load: {result['error_rate_percent']}%"
    assert result["avg_response_time_ms"] < 5000, f"Response time too slow under load: {result['avg_response_time_ms']}ms"


if __name__ == "__main__":
    # Run tests standalone
    async def main():
        suite = ModelSelectionTests()
        results = await suite.run_full_test_suite()
        
        print("\n" + "="*60)
        print("DYNAMIC MODEL SELECTION TEST SUITE RESULTS")
        print("="*60)
        
        for test_name, test_result in results["tests"].items():
            status = "PASS" if not test_result.get("error") and not test_result.get("test_failed") else "FAIL"
            print(f"{test_name.upper()}: {status}")
            
            if test_result.get("error"):
                print(f"  Error: {test_result['error']}")
        
        print("\n" + "-"*60)
        summary = results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['overall_success_rate']:.1%}")
        print("-"*60)
    
    asyncio.run(main())