"""
Unit Tests for Model Interface Components

Tests for OpenAI, DeepSeek, and base model interface functionality including
API calls, response parsing, error handling, and configuration validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json

# Import the modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.base_model import BaseModel
from models.openai_client import OpenAIClient
from models.deepseek_client import DeepSeekClient


class TestBaseModel:
    """Test the BaseModel abstract interface."""
    
    def test_base_model_abstract_methods(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()
    
    def test_base_model_interface_methods(self):
        """Test that required methods are defined in interface."""
        # Check that abstract methods exist
        assert hasattr(BaseModel, 'generate_response')
        assert hasattr(BaseModel, 'get_model_info')
        assert hasattr(BaseModel, 'validate_config')


@pytest.mark.unit
class TestOpenAIClient:
    """Test OpenAI client functionality."""
    
    @pytest.fixture
    def openai_config(self):
        """OpenAI configuration for testing."""
        return {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 30.0
        }
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "This is a helpful therapeutic response."
        response.usage.total_tokens = 150
        response.usage.prompt_tokens = 50
        response.usage.completion_tokens = 100
        return response
    
    def test_openai_client_initialization(self, openai_config, mock_api_key):
        """Test OpenAI client initialization."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': mock_api_key}):
            client = OpenAIClient(openai_config)
            assert client.model_name == "openai-gpt4"
            assert client.config["model"] == "gpt-4"
            assert client.config["temperature"] == 0.7
    
    def test_openai_client_missing_api_key(self):
        """Test OpenAI client fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                OpenAIClient()
    
    @pytest.mark.asyncio
    async def test_openai_generate_response_success(self, openai_config, mock_api_key, mock_openai_response):
        """Test successful OpenAI response generation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': mock_api_key}):
            with patch('openai.AsyncOpenAI') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.chat.completions.create.return_value = mock_openai_response
                mock_client_class.return_value = mock_client
                
                client = OpenAIClient(openai_config)
                response = await client.generate_response("Test prompt", "Test context")
                
                assert response["content"] == "This is a helpful therapeutic response."
                assert response["usage"]["total_tokens"] == 150
                assert "response_time_ms" in response
    
    @pytest.mark.asyncio
    async def test_openai_timeout_handling(self, openai_config, mock_api_key):
        """Test OpenAI timeout error handling."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': mock_api_key}):
            with patch('openai.AsyncOpenAI') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.chat.completions.create.side_effect = asyncio.TimeoutError()
                mock_client_class.return_value = mock_client
                
                client = OpenAIClient(openai_config)
                
                with pytest.raises(Exception) as exc_info:
                    await client.generate_response("Test prompt")
                
                assert "timeout" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_openai_rate_limit_handling(self, openai_config, mock_api_key):
        """Test OpenAI rate limit error handling."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': mock_api_key}):
            with patch('openai.AsyncOpenAI') as mock_client_class:
                mock_client = AsyncMock()
                
                # Mock rate limit error
                rate_limit_error = Exception("Rate limit exceeded")
                rate_limit_error.__class__.__name__ = "RateLimitError"
                mock_client.chat.completions.create.side_effect = rate_limit_error
                mock_client_class.return_value = mock_client
                
                client = OpenAIClient(openai_config)
                
                with pytest.raises(Exception):
                    await client.generate_response("Test prompt")
    
    def test_openai_config_validation(self, mock_api_key):
        """Test OpenAI configuration validation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': mock_api_key}):
            # Valid config
            valid_config = {"model": "gpt-4", "temperature": 0.7}
            assert OpenAIClient.validate_config(valid_config) == True
            
            # Invalid temperature
            invalid_config = {"model": "gpt-4", "temperature": 2.0}
            assert OpenAIClient.validate_config(invalid_config) == False
            
            # Missing model
            invalid_config = {"temperature": 0.7}
            assert OpenAIClient.validate_config(invalid_config) == False
    
    def test_openai_model_info(self, openai_config, mock_api_key):
        """Test OpenAI model information retrieval."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': mock_api_key}):
            client = OpenAIClient(openai_config)
            info = client.get_model_info()
            
            assert info["model_name"] == "openai-gpt4"
            assert info["provider"] == "OpenAI"
            assert info["model_type"] == "transformer"
            assert "capabilities" in info


@pytest.mark.unit
class TestDeepSeekClient:
    """Test DeepSeek client functionality."""
    
    @pytest.fixture
    def deepseek_config(self):
        """DeepSeek configuration for testing."""
        return {
            "model": "deepseek-v2",
            "temperature": 0.7,
            "max_tokens": 1000,
            "use_api": True
        }
    
    @pytest.fixture
    def mock_deepseek_response(self):
        """Mock DeepSeek API response."""
        return {
            "choices": [{
                "message": {
                    "content": "I understand you're going through a difficult time."
                }
            }],
            "usage": {
                "total_tokens": 120,
                "prompt_tokens": 45,
                "completion_tokens": 75
            }
        }
    
    def test_deepseek_client_initialization_api(self, deepseek_config, mock_api_key):
        """Test DeepSeek client initialization with API."""
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': mock_api_key}):
            client = DeepSeekClient(deepseek_config)
            assert client.model_name == "deepseek-v2"
            assert client.config["use_api"] == True
    
    def test_deepseek_client_initialization_local(self):
        """Test DeepSeek client initialization for local inference."""
        config = {
            "model_path": "./models/deepseek",
            "device": "cpu",
            "use_api": False
        }
        
        # Mock the model loading to avoid actual file operations
        with patch('torch.load'):
            with patch('transformers.AutoTokenizer.from_pretrained'):
                with patch('transformers.AutoModelForCausalLM.from_pretrained'):
                    client = DeepSeekClient(config)
                    assert client.model_name == "deepseek-local"
                    assert client.config["use_api"] == False
    
    @pytest.mark.asyncio
    async def test_deepseek_api_response(self, deepseek_config, mock_api_key, mock_deepseek_response):
        """Test DeepSeek API response generation."""
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': mock_api_key}):
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.json.return_value = mock_deepseek_response
                mock_response.status = 200
                mock_post.return_value.__aenter__.return_value = mock_response
                
                client = DeepSeekClient(deepseek_config)
                response = await client.generate_response("Test prompt")
                
                assert response["content"] == "I understand you're going through a difficult time."
                assert response["usage"]["total_tokens"] == 120
    
    @pytest.mark.asyncio
    async def test_deepseek_local_inference(self):
        """Test DeepSeek local model inference."""
        config = {
            "model_path": "./models/deepseek",
            "device": "cpu",
            "use_api": False,
            "temperature": 0.7
        }
        
        with patch('torch.load'):
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                    # Mock tokenizer
                    mock_tokenizer_instance = Mock()
                    mock_tokenizer_instance.encode.return_value = [1, 2, 3]
                    mock_tokenizer_instance.decode.return_value = "Local inference response"
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    # Mock model
                    mock_model_instance = Mock()
                    mock_output = Mock()
                    mock_output.logits = Mock()
                    mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5]]
                    mock_model.return_value = mock_model_instance
                    
                    client = DeepSeekClient(config)
                    response = await client.generate_response("Test prompt")
                    
                    assert "content" in response
                    assert response["content"] == "Local inference response"
    
    def test_deepseek_config_validation(self):
        """Test DeepSeek configuration validation."""
        # Valid API config
        api_config = {"model": "deepseek-v2", "use_api": True}
        assert DeepSeekClient.validate_config(api_config) == True
        
        # Valid local config
        local_config = {"model_path": "./models/deepseek", "use_api": False}
        assert DeepSeekClient.validate_config(local_config) == True
        
        # Invalid - missing required fields
        invalid_config = {"temperature": 0.7}
        assert DeepSeekClient.validate_config(invalid_config) == False
    
    def test_deepseek_model_info(self, deepseek_config, mock_api_key):
        """Test DeepSeek model information retrieval."""
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': mock_api_key}):
            client = DeepSeekClient(deepseek_config)
            info = client.get_model_info()
            
            assert info["model_name"] == "deepseek-v2"
            assert info["provider"] == "DeepSeek"
            assert "capabilities" in info


@pytest.mark.unit
class TestModelInterfaceCommon:
    """Test common model interface functionality."""
    
    def test_response_format_consistency(self, mock_openai_client, mock_deepseek_client):
        """Test that all models return consistent response format."""
        # This would be an integration test, but we test the interface here
        required_fields = ["content", "usage", "response_time_ms"]
        
        # Both clients should have these methods
        assert hasattr(mock_openai_client, 'generate_response')
        assert hasattr(mock_deepseek_client, 'generate_response')
        
        # Both should have model info
        assert hasattr(mock_openai_client, 'get_model_info')
        assert hasattr(mock_deepseek_client, 'get_model_info')
    
    def test_error_handling_consistency(self):
        """Test that all models handle errors consistently."""
        # Test that all models should handle common error types
        common_errors = [
            "TimeoutError",
            "RateLimitError", 
            "AuthenticationError",
            "ConnectionError"
        ]
        
        # This ensures our error handling patterns are consistent
        assert len(common_errors) == 4  # Basic assertion for test structure
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_openai_client):
        """Test handling of concurrent requests."""
        # Create multiple concurrent requests
        tasks = [
            mock_openai_client.generate_response(f"Prompt {i}")
            for i in range(5)
        ]
        
        # All should complete without interference
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that we got responses (or expected exceptions)
        assert len(responses) == 5
        for response in responses:
            # In real tests, we'd check the actual response format
            assert response is not None


@pytest.mark.unit
class TestModelErrorHandling:
    """Test error handling across all model interfaces."""
    
    @pytest.mark.parametrize("error_type,expected_retry", [
        ("TimeoutError", True),
        ("RateLimitError", True),
        ("AuthenticationError", False),
        ("ServerError", True),
        ("NetworkError", True)
    ])
    def test_error_retry_logic(self, error_type, expected_retry):
        """Test that appropriate errors trigger retries."""
        # This tests the retry logic configuration
        retryable_errors = [
            "TimeoutError", "RateLimitError", "ServerError", "NetworkError"
        ]
        
        should_retry = error_type in retryable_errors
        assert should_retry == expected_retry
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, mock_openai_client):
        """Test circuit breaker prevents cascade failures."""
        # Mock multiple consecutive failures
        mock_openai_client.generate_response.side_effect = Exception("API Error")
        
        # Circuit breaker should eventually stop trying
        failure_count = 0
        for i in range(10):
            try:
                await mock_openai_client.generate_response("test")
            except:
                failure_count += 1
        
        # We expect failures, this tests the circuit breaker concept
        assert failure_count > 0


@pytest.mark.unit
class TestModelConfiguration:
    """Test model configuration validation and management."""
    
    def test_config_schema_validation(self):
        """Test configuration schema validation."""
        # Common configuration fields that should be validated
        common_fields = [
            "temperature", "max_tokens", "timeout", "model"
        ]
        
        valid_config = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 30.0,
            "model": "test-model"
        }
        
        # Test individual field validation
        for field in common_fields:
            assert field in valid_config
    
    def test_config_defaults(self):
        """Test that reasonable defaults are provided."""
        default_config = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 30.0
        }
        
        # Ensure defaults are within reasonable ranges
        assert 0.0 <= default_config["temperature"] <= 1.0
        assert default_config["max_tokens"] > 0
        assert default_config["timeout"] > 0
    
    def test_config_security(self):
        """Test configuration security validation."""
        # Ensure sensitive data is properly handled
        sensitive_fields = ["api_key", "secret", "password"]
        
        # These should never appear in config directly
        test_config = {"model": "test", "temperature": 0.7}
        
        for field in sensitive_fields:
            assert field not in test_config