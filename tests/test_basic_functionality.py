"""
Basic Functionality Tests for Mental Health LLM System

These tests verify that the core components can be imported and initialized
without requiring external API keys or network connections.
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))


class TestBasicImports:
    """Test that all core modules can be imported successfully"""

    def test_import_dynamic_model_selector(self):
        """Test importing dynamic model selector"""
        from src.chat.dynamic_model_selector import DynamicModelSelector, PromptType
        assert DynamicModelSelector is not None
        assert PromptType is not None

    def test_import_analytics_modules(self):
        """Test importing analytics modules"""
        from src.analytics.smart_switching import SmartModelSwitcher, SwitchReason
        from src.analytics.feedback_system import FeedbackCollector, UserFeedback
        from src.analytics.ab_testing import ExperimentManager
        
        assert SmartModelSwitcher is not None
        assert SwitchReason is not None
        assert FeedbackCollector is not None
        assert UserFeedback is not None
        assert ExperimentManager is not None

    def test_import_chat_modules(self):
        """Test importing chat modules"""
        from src.chat.conversation_session_manager import ConversationSessionManager
        from src.chat.response_cache import ResponseCache
        
        assert ConversationSessionManager is not None
        assert ResponseCache is not None

    def test_import_monitoring_modules(self):
        """Test importing monitoring modules"""
        from src.monitoring.health_checker import HealthMonitor
        from src.monitoring.alerts import AlertManager
        from src.monitoring.failover import CircuitBreaker
        
        assert HealthMonitor is not None
        assert AlertManager is not None
        assert CircuitBreaker is not None


class TestBasicInitialization:
    """Test that core components can be initialized"""

    def test_prompt_type_enum(self):
        """Test PromptType enum functionality"""
        from src.chat.dynamic_model_selector import PromptType
        
        # Test that enum values exist
        assert hasattr(PromptType, 'GENERAL_WELLNESS')
        assert hasattr(PromptType, 'ANXIETY')
        assert hasattr(PromptType, 'DEPRESSION')
        assert hasattr(PromptType, 'CRISIS')

    def test_dynamic_model_selector_initialization(self):
        """Test DynamicModelSelector can be initialized"""
        from src.chat.dynamic_model_selector import DynamicModelSelector
        
        # Mock the config loading to avoid file dependencies
        with patch('src.chat.dynamic_model_selector.yaml.safe_load'), \
             patch('builtins.open'), \
             patch('os.path.exists', return_value=True):
            
            selector = DynamicModelSelector()
            assert selector is not None
            assert hasattr(selector, 'select_model')

    def test_response_cache_initialization(self):
        """Test ResponseCache can be initialized"""
        from src.chat.response_cache import ResponseCache
        
        cache = ResponseCache(max_size=100, ttl_seconds=300)
        assert cache is not None
        assert hasattr(cache, 'get')
        assert hasattr(cache, 'set')

    def test_feedback_collector_initialization(self):
        """Test FeedbackCollector can be initialized"""
        from src.analytics.feedback_system import FeedbackCollector
        
        collector = FeedbackCollector()
        assert collector is not None
        assert hasattr(collector, 'collect_feedback')

    def test_experiment_manager_initialization(self):
        """Test ExperimentManager can be initialized"""
        from src.analytics.ab_testing import ExperimentManager
        
        manager = ExperimentManager()
        assert manager is not None
        assert hasattr(manager, 'create_experiment')


class TestDataStructures:
    """Test that data structures work correctly"""

    def test_model_selection_dataclass(self):
        """Test ModelSelection dataclass"""
        from src.chat.dynamic_model_selector import ModelSelection
        
        selection = ModelSelection(
            selected_model="test_model",
            confidence_score=0.85,
            reasoning="Test reasoning",
            fallback_models=["fallback1", "fallback2"],
            estimated_cost=0.001,
            expected_quality=0.9
        )
        
        assert selection.selected_model == "test_model"
        assert selection.confidence_score == 0.85
        assert selection.reasoning == "Test reasoning"
        assert len(selection.fallback_models) == 2
        assert selection.estimated_cost == 0.001
        assert selection.expected_quality == 0.9

    def test_user_feedback_dataclass(self):
        """Test UserFeedback dataclass"""
        from src.analytics.feedback_system import UserFeedback, FeedbackType
        
        feedback = UserFeedback(
            user_id="test_user",
            conversation_id="test_conv",
            model_used="test_model",
            feedback_type=FeedbackType.RATING,
            rating=4,
            text_feedback="Good response"
        )
        
        assert feedback.user_id == "test_user"
        assert feedback.conversation_id == "test_conv"
        assert feedback.model_used == "test_model"
        assert feedback.feedback_type == FeedbackType.RATING
        assert feedback.rating == 4
        assert feedback.text_feedback == "Good response"


class TestConfigurationLoading:
    """Test configuration loading functionality"""

    def test_config_loader_import(self):
        """Test that config loader can be imported"""
        from src.config.config_loader import ConfigLoader
        assert ConfigLoader is not None

    @patch('builtins.open')
    @patch('os.path.exists', return_value=True)
    @patch('src.config.config_loader.yaml.safe_load')
    def test_config_loader_initialization(self, mock_yaml, mock_exists, mock_open):
        """Test ConfigLoader initialization with mocked file system"""
        from src.config.config_loader import ConfigLoader
        
        # Mock a basic config structure
        mock_yaml.return_value = {
            'models': {'openai': {'enabled': True}},
            'chat': {'default_model': 'openai'},
            'cache': {'enabled': True, 'ttl': 300}
        }
        
        config = ConfigLoader()
        assert config is not None


class TestUtilities:
    """Test utility functions and classes"""

    def test_paths_import(self):
        """Test paths module import"""
        from src.utils.paths import get_results_dir, get_temp_dir
        
        results_dir = get_results_dir()
        temp_dir = get_temp_dir()
        
        assert results_dir is not None
        assert temp_dir is not None
        assert str(results_dir).endswith('results')
        assert str(temp_dir).endswith('temp')

    def test_logging_config_import(self):
        """Test logging config import"""
        from src.utils.logging_config import setup_logging
        assert setup_logging is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])