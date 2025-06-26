"""
Integration Tests for End-to-End Conversation Generation

Tests the complete conversation generation workflow from scenario loading
through model interaction to evaluation and storage.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import json

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from conversation.conversation_manager import ConversationManager
from conversation.batch_processor import BatchProcessor, BatchConfig
from scenarios.scenario import ScenarioLoader
from models.openai_client import OpenAIClient
from evaluation.composite_scorer import CompositeScorer


@pytest.mark.integration
class TestEndToEndConversationGeneration:
    """Test complete conversation generation workflow."""
    
    @pytest.fixture
    async def conversation_manager(self, temp_test_dir):
        """Create conversation manager with mocked dependencies."""
        config = {
            "output_directory": str(temp_test_dir),
            "enable_safety_monitoring": True,
            "enable_metrics_collection": True
        }
        
        manager = ConversationManager(config)
        return manager
    
    @pytest.fixture
    def mock_scenario_loader(self, sample_scenario):
        """Mock scenario loader."""
        loader = Mock(spec=ScenarioLoader)
        loader.load_scenario.return_value = sample_scenario
        loader.load_all_scenarios.return_value = [sample_scenario]
        return loader
    
    @pytest.fixture
    def mock_model_client(self):
        """Mock model client with realistic responses."""
        client = AsyncMock()
        client.model_name = "test_model"
        
        # Mock responses for different conversation turns
        responses = [
            "I understand you're feeling anxious. Can you tell me more about what's been worrying you?",
            "That sounds really challenging. It's completely normal to feel overwhelmed in that situation.",
            "Have you noticed any physical symptoms when you feel this anxiety?",
            "Those are common anxiety symptoms. Let's talk about some techniques that might help.",
            "One helpful technique is deep breathing. Would you like to try that together?",
            "You're doing great. Remember, managing anxiety is a process and you're taking positive steps."
        ]
        
        response_iter = iter(responses)
        
        async def mock_generate_response(prompt, context=None):
            try:
                content = next(response_iter)
            except StopIteration:
                content = "Thank you for sharing. Is there anything else you'd like to discuss?"
            
            return {
                "content": content,
                "usage": {
                    "total_tokens": 150,
                    "prompt_tokens": 50,
                    "completion_tokens": 100
                },
                "response_time_ms": 2000 + (len(content) * 10)  # Realistic timing
            }
        
        client.generate_response = mock_generate_response
        return client
    
    @pytest.mark.asyncio
    async def test_single_conversation_generation(
        self, 
        conversation_manager, 
        sample_scenario, 
        mock_model_client
    ):
        """Test generating a single complete conversation."""
        
        conversation = await conversation_manager.generate_conversation(
            model_client=mock_model_client,
            scenario=sample_scenario,
            conversation_id="test_conv_001"
        )
        
        # Verify conversation structure
        assert conversation is not None
        assert "conversation_metadata" in conversation
        assert "conversation_turns" in conversation
        assert "analytics_data" in conversation
        
        # Verify metadata
        metadata = conversation["conversation_metadata"]
        assert metadata["conversation_id"] == "test_conv_001"
        assert metadata["scenario_id"] == sample_scenario.scenario_id
        assert metadata["model_name"] == "test_model"
        assert metadata["total_turns"] >= sample_scenario.conversation_flow["min_turns"]
        assert metadata["total_turns"] <= sample_scenario.conversation_flow["max_turns"]
        
        # Verify conversation turns
        turns = conversation["conversation_turns"]
        assert len(turns) >= 4  # At least a few exchanges
        
        # Check turn structure
        for turn in turns:
            assert "turn_number" in turn
            assert "speaker" in turn
            assert "message" in turn
            assert "timestamp" in turn
            assert turn["speaker"] in ["patient", "assistant"]
        
        # Verify analytics
        analytics = conversation["analytics_data"]
        assert "empathy_scores" in analytics
        assert "safety_flags" in analytics
        assert "conversation_flow_rating" in analytics
    
    @pytest.mark.asyncio
    async def test_conversation_with_branching(
        self,
        conversation_manager,
        sample_scenario,
        mock_model_client
    ):
        """Test conversation with branching logic."""
        
        # Add branching points to scenario
        sample_scenario.conversation_flow["branching_points"] = [
            {
                "turn": 3,
                "condition": "mentions_physical_symptoms",
                "options": ["explore_symptoms", "focus_cognitive", "discuss_coping"]
            }
        ]
        
        conversation = await conversation_manager.generate_conversation(
            model_client=mock_model_client,
            scenario=sample_scenario,
            conversation_id="test_branch_conv"
        )
        
        # Verify branching was considered
        assert conversation is not None
        assert len(conversation["conversation_turns"]) >= 6  # Should continue past branching point
        
        # Check that branching logic was applied
        analytics = conversation["analytics_data"]
        assert "branching_decisions" in analytics or len(conversation["conversation_turns"]) > 3
    
    @pytest.mark.asyncio
    async def test_conversation_error_handling(
        self,
        conversation_manager,
        sample_scenario
    ):
        """Test conversation error handling and recovery."""
        
        # Create model client that fails occasionally
        failing_client = AsyncMock()
        failing_client.model_name = "failing_model"
        
        call_count = 0
        async def mock_generate_with_failures(prompt, context=None):
            nonlocal call_count
            call_count += 1
            
            if call_count == 3:  # Fail on third call
                raise Exception("API timeout error")
            
            return {
                "content": f"Response {call_count}",
                "usage": {"total_tokens": 100},
                "response_time_ms": 2000
            }
        
        failing_client.generate_response = mock_generate_with_failures
        
        conversation = await conversation_manager.generate_conversation(
            model_client=failing_client,
            scenario=sample_scenario,
            conversation_id="test_error_conv"
        )
        
        # Should handle error gracefully
        assert conversation is not None
        
        # Check error was logged
        if "errors" in conversation["conversation_metadata"]:
            errors = conversation["conversation_metadata"]["errors"]
            assert len(errors) > 0
            assert any("timeout" in str(error).lower() for error in errors)
    
    @pytest.mark.asyncio
    async def test_conversation_safety_monitoring(
        self,
        conversation_manager,
        crisis_scenario,
        mock_model_client
    ):
        """Test safety monitoring during conversation."""
        
        # Mock model to give concerning response
        crisis_client = AsyncMock()
        crisis_client.model_name = "crisis_test_model"
        
        responses = [
            "I understand you're in pain.",
            "It sounds like you're having thoughts of hurting yourself. That's very concerning.",
            "I'm worried about your safety right now. Are you thinking about suicide?",
            "Thank you for telling me. This is serious and I want to help keep you safe."
        ]
        
        response_iter = iter(responses)
        
        async def crisis_response_generator(prompt, context=None):
            try:
                content = next(response_iter)
            except StopIteration:
                content = "Let's talk about getting you immediate help."
            
            return {
                "content": content,
                "usage": {"total_tokens": 120},
                "response_time_ms": 1800
            }
        
        crisis_client.generate_response = crisis_response_generator
        
        conversation = await conversation_manager.generate_conversation(
            model_client=crisis_client,
            scenario=crisis_scenario,
            conversation_id="test_crisis_conv"
        )
        
        # Verify safety monitoring was active
        assert conversation is not None
        
        # Check for safety flags
        analytics = conversation["analytics_data"]
        assert "safety_analysis" in analytics
        
        safety_analysis = analytics["safety_analysis"]
        assert "crisis_intervention_triggered" in safety_analysis
        
        # Should have detected crisis situation
        if "safety_flags" in analytics:
            safety_flags = analytics["safety_flags"]
            assert len(safety_flags) > 0
    
    @pytest.mark.asyncio
    async def test_conversation_evaluation_integration(
        self,
        conversation_manager,
        sample_scenario,
        mock_model_client
    ):
        """Test integration with evaluation system."""
        
        conversation = await conversation_manager.generate_conversation(
            model_client=mock_model_client,
            scenario=sample_scenario,
            conversation_id="test_eval_conv",
            enable_evaluation=True
        )
        
        # Verify evaluation was performed
        assert conversation is not None
        assert "evaluation_scores" in conversation
        
        eval_scores = conversation["evaluation_scores"]
        required_scores = [
            "overall_score", "empathy_score", "safety_score", 
            "coherence_score", "therapeutic_score"
        ]
        
        for score_type in required_scores:
            assert score_type in eval_scores
            assert 0.0 <= eval_scores[score_type] <= 10.0
    
    @pytest.mark.asyncio
    async def test_conversation_data_persistence(
        self,
        conversation_manager,
        sample_scenario,
        mock_model_client,
        temp_test_dir
    ):
        """Test conversation data persistence."""
        
        conversation = await conversation_manager.generate_conversation(
            model_client=mock_model_client,
            scenario=sample_scenario,
            conversation_id="test_persist_conv",
            save_to_database=True,
            save_to_file=True
        )
        
        # Verify conversation was saved to file
        conversation_files = list(temp_test_dir.glob("**/*.json"))
        assert len(conversation_files) > 0
        
        # Verify file content
        saved_conversation = None
        for file_path in conversation_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data["conversation_metadata"]["conversation_id"] == "test_persist_conv":
                    saved_conversation = data
                    break
        
        assert saved_conversation is not None
        assert saved_conversation["conversation_metadata"]["conversation_id"] == "test_persist_conv"


@pytest.mark.integration
class TestBatchConversationGeneration:
    """Test batch conversation generation."""
    
    @pytest.fixture
    def batch_config(self, temp_test_dir):
        """Create batch configuration for testing."""
        return BatchConfig(
            conversations_per_scenario_per_model=2,  # Small number for testing
            max_concurrent_conversations=2,
            output_directory=str(temp_test_dir),
            enable_safety_monitoring=True,
            enable_metrics_collection=True
        )
    
    @pytest.fixture
    def batch_processor(self, batch_config):
        """Create batch processor instance."""
        return BatchProcessor(batch_config)
    
    @pytest.mark.asyncio
    async def test_batch_single_model_multiple_scenarios(
        self,
        batch_processor,
        mock_model_client,
        sample_scenario
    ):
        """Test batch processing with single model and multiple scenarios."""
        
        # Create multiple scenarios
        scenarios = []
        for i in range(3):
            scenario = sample_scenario.__class__(
                scenario_id=f"MH-TEST-{i:03d}",
                title=f"Test Scenario {i}",
                category="Test Category",
                severity="Mild",
                patient_profile=sample_scenario.patient_profile,
                opening_statement=f"Test opening statement {i}",
                conversation_goals=sample_scenario.conversation_goals,
                expected_therapeutic_elements=sample_scenario.expected_therapeutic_elements,
                red_flags=sample_scenario.red_flags,
                conversation_flow=sample_scenario.conversation_flow,
                evaluation_criteria=sample_scenario.evaluation_criteria
            )
            scenarios.append(scenario)
        
        batch_results = await batch_processor.process_batch(
            models=[mock_model_client],
            scenarios=scenarios
        )
        
        # Verify batch results
        assert batch_results is not None
        assert "batch_summary" in batch_results
        assert "performance_metrics" in batch_results
        
        # Check completion counts
        performance = batch_results["performance_metrics"]
        expected_total = len(scenarios) * 2  # 2 conversations per scenario
        assert performance["total_conversations_completed"] == expected_total
    
    @pytest.mark.asyncio
    async def test_batch_multiple_models_single_scenario(
        self,
        batch_processor,
        sample_scenario
    ):
        """Test batch processing with multiple models and single scenario."""
        
        # Create multiple mock model clients
        models = []
        for i in range(2):
            client = AsyncMock()
            client.model_name = f"test_model_{i}"
            
            async def make_response_generator(model_id):
                async def generate_response(prompt, context=None):
                    return {
                        "content": f"Response from model {model_id}",
                        "usage": {"total_tokens": 100},
                        "response_time_ms": 2000
                    }
                return generate_response
            
            client.generate_response = await make_response_generator(i)
            models.append(client)
        
        batch_results = await batch_processor.process_batch(
            models=models,
            scenarios=[sample_scenario]
        )
        
        # Verify results for multiple models
        assert batch_results is not None
        
        # Check model-specific results
        if "model_comparison" in batch_results:
            model_comparison = batch_results["model_comparison"]
            assert len(model_comparison) == 2
            
            for model_name in ["test_model_0", "test_model_1"]:
                assert model_name in model_comparison
                model_data = model_comparison[model_name]
                assert "conversations_completed" in model_data
                assert model_data["conversations_completed"] == 2  # conversations_per_scenario_per_model
    
    @pytest.mark.asyncio
    async def test_batch_error_handling_and_recovery(
        self,
        batch_processor,
        sample_scenario
    ):
        """Test batch processing error handling and recovery."""
        
        # Create model that fails for some conversations
        unreliable_client = AsyncMock()
        unreliable_client.model_name = "unreliable_model"
        
        call_count = 0
        async def unreliable_generate_response(prompt, context=None):
            nonlocal call_count
            call_count += 1
            
            if call_count % 3 == 0:  # Fail every third call
                raise Exception("Simulated API failure")
            
            return {
                "content": f"Successful response {call_count}",
                "usage": {"total_tokens": 100},
                "response_time_ms": 2000
            }
        
        unreliable_client.generate_response = unreliable_generate_response
        
        batch_results = await batch_processor.process_batch(
            models=[unreliable_client],
            scenarios=[sample_scenario]
        )
        
        # Should complete some conversations despite failures
        assert batch_results is not None
        
        performance = batch_results["performance_metrics"]
        assert performance["total_conversations_completed"] > 0
        assert performance["total_conversations_failed"] > 0
        assert performance["overall_success_rate"] < 100.0
    
    @pytest.mark.asyncio
    async def test_batch_progress_monitoring(
        self,
        batch_processor,
        mock_model_client,
        sample_scenario
    ):
        """Test batch processing progress monitoring."""
        
        progress_updates = []
        
        def progress_callback(progress_data):
            progress_updates.append(progress_data)
        
        batch_results = await batch_processor.process_batch(
            models=[mock_model_client],
            scenarios=[sample_scenario],
            progress_callback=progress_callback
        )
        
        # Verify progress updates were sent
        assert len(progress_updates) > 0
        
        # Check progress data structure
        for update in progress_updates:
            assert "conversations_completed" in update
            assert "conversations_total" in update
            assert "progress_percentage" in update
            assert 0.0 <= update["progress_percentage"] <= 100.0
    
    @pytest.mark.asyncio
    async def test_batch_concurrent_processing(
        self,
        batch_processor,
        sample_scenario
    ):
        """Test concurrent conversation processing."""
        
        # Create multiple model clients with timing
        models = []
        for i in range(3):
            client = AsyncMock()
            client.model_name = f"concurrent_model_{i}"
            
            async def generate_with_delay(model_id):
                async def generate_response(prompt, context=None):
                    # Simulate processing time
                    await asyncio.sleep(0.1)
                    return {
                        "content": f"Response from model {model_id}",
                        "usage": {"total_tokens": 100},
                        "response_time_ms": 2000
                    }
                return generate_response
            
            client.generate_response = await generate_with_delay(i)
            models.append(client)
        
        import time
        start_time = time.time()
        
        batch_results = await batch_processor.process_batch(
            models=models,
            scenarios=[sample_scenario]
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # With concurrency, should complete faster than sequential processing
        # (This is a rough check - exact timing depends on system)
        expected_sequential_time = len(models) * 2 * 0.1  # 3 models * 2 conversations * 0.1s delay
        assert total_time < expected_sequential_time * 1.5  # Allow some overhead