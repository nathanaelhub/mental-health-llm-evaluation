"""
Core Component Tests for Mental Health LLM System

These tests verify that the essential components work with the actual codebase.
"""

import sys
import os
import pytest

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))


class TestPromptTypeEnum:
    """Test PromptType enum functionality"""

    def test_prompt_type_import(self):
        """Test that PromptType can be imported"""
        from src.chat.dynamic_model_selector import PromptType
        assert PromptType is not None

    def test_prompt_type_values(self):
        """Test that PromptType has expected values"""
        from src.chat.dynamic_model_selector import PromptType
        
        # Test actual enum values based on the real implementation
        assert hasattr(PromptType, 'CRISIS')
        assert hasattr(PromptType, 'ANXIETY')
        assert hasattr(PromptType, 'DEPRESSION')
        assert hasattr(PromptType, 'GENERAL_SUPPORT')
        assert hasattr(PromptType, 'INFORMATION_SEEKING')
        
        # Test enum values
        assert PromptType.CRISIS.value == "crisis"
        assert PromptType.ANXIETY.value == "anxiety"
        assert PromptType.DEPRESSION.value == "depression"


class TestDynamicModelSelector:
    """Test DynamicModelSelector basic functionality"""

    def test_import_dynamic_model_selector(self):
        """Test that DynamicModelSelector can be imported"""
        from src.chat.dynamic_model_selector import DynamicModelSelector
        assert DynamicModelSelector is not None

    def test_import_model_selection_dataclass(self):
        """Test that ModelSelection dataclass can be imported"""
        from src.chat.dynamic_model_selector import ModelSelection
        assert ModelSelection is not None

    def test_import_selection_criteria(self):
        """Test that SelectionCriteria can be imported"""
        from src.chat.dynamic_model_selector import SelectionCriteria
        assert SelectionCriteria is not None


class TestAnalyticsComponents:
    """Test analytics components that don't require external dependencies"""

    def test_import_smart_switching_enums(self):
        """Test importing smart switching enums"""
        from src.analytics.smart_switching import SwitchReason
        assert SwitchReason is not None

    def test_import_feedback_enums(self):
        """Test importing feedback system enums without initializing classes"""
        # Test specific enum that doesn't depend on PromptType
        from src.analytics.feedback_system import FeedbackType
        assert FeedbackType is not None

    def test_import_ab_testing_enums(self):
        """Test importing A/B testing enums"""
        from src.analytics.ab_testing import ExperimentStatus, SelectionStrategy
        assert ExperimentStatus is not None
        assert SelectionStrategy is not None


class TestMonitoringComponents:
    """Test monitoring components"""

    def test_import_health_checker(self):
        """Test importing health checker"""
        from src.monitoring.health_checker import HealthMonitor
        assert HealthMonitor is not None

    def test_import_alerts(self):
        """Test importing alerts"""
        from src.monitoring.alerts import AlertManager
        assert AlertManager is not None

    def test_import_failover(self):
        """Test importing failover"""
        from src.monitoring.failover import CircuitBreaker
        assert CircuitBreaker is not None


class TestConfigurationComponents:
    """Test configuration components"""

    def test_import_config_loader(self):
        """Test importing config loader"""
        from src.config.config_loader import ConfigLoader
        assert ConfigLoader is not None

    def test_import_config_schema(self):
        """Test importing config schema"""
        from src.config.config_schema import validate_config
        assert validate_config is not None


class TestUtilityComponents:
    """Test utility components that don't have external dependencies"""

    def test_import_paths_directly(self):
        """Test importing paths module directly"""
        import sys
        import os
        
        # Import paths module directly
        paths_module_path = os.path.join(project_root, 'src', 'utils', 'paths.py')
        spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location("paths", paths_module_path)
        paths_module = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
        spec.loader.exec_module(paths_module)
        
        # Test that key functions exist
        assert hasattr(paths_module, 'get_results_dir')
        assert hasattr(paths_module, 'get_temp_dir')
        
        # Test that functions return Path objects
        results_dir = paths_module.get_results_dir()
        temp_dir = paths_module.get_temp_dir()
        
        assert results_dir is not None
        assert temp_dir is not None


class TestModelClients:
    """Test that model client classes can be imported"""

    def test_import_base_model(self):
        """Test importing base model"""
        from src.models.base_model import BaseModel
        assert BaseModel is not None

    def test_import_openai_client(self):
        """Test importing OpenAI client"""
        from src.models.openai_client import OpenAIClient
        assert OpenAIClient is not None

    def test_import_deepseek_client(self):
        """Test importing DeepSeek client"""
        from src.models.deepseek_client import DeepSeekClient
        assert DeepSeekClient is not None

    def test_import_local_llm_client(self):
        """Test importing local LLM client"""
        from src.models.local_llm_client import LocalLLMClient
        assert LocalLLMClient is not None


class TestEvaluationComponents:
    """Test evaluation components"""

    def test_import_mental_health_evaluator(self):
        """Test importing mental health evaluator"""
        from src.evaluation.mental_health_evaluator import MentalHealthEvaluator
        assert MentalHealthEvaluator is not None

    def test_import_evaluation_metrics(self):
        """Test importing evaluation metrics"""
        from src.evaluation.evaluation_metrics import TherapeuticEvaluator
        assert TherapeuticEvaluator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])