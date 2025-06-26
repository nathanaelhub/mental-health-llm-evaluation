"""
Claude Integration Template

Ready-to-use template for integrating Anthropic Claude models.
This template provides a complete, production-ready implementation for Claude integration.

SETUP INSTRUCTIONS:
1. Install required package: pip install anthropic
2. Set environment variable: export ANTHROPIC_API_KEY=your-api-key
3. Copy this file to src/models/claude_client.py
4. Enable in config/experiment_template.yaml:
   models.cloud[1].enabled: true
5. Test: python scripts/model_management.py test claude-3

FEATURES:
- Complete Anthropic API integration
- Multiple Claude variants (Opus, Sonnet, Haiku)
- Cost calculation and token counting
- Error handling and retry logic
- Streaming support (optional)
- Mental health conversation optimization
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None

from src.models.base_model import BaseModel, ModelResponse, ModelProvider, ModelType
from src.models.model_registry import register_model_decorator


@register_model_decorator(
    name="claude-3-opus",
    provider=ModelProvider.ANTHROPIC,
    model_type=ModelType.CLOUD,
    description="Anthropic Claude-3 Opus - Most capable model for complex mental health conversations",
    requirements=["anthropic"],
    default_config={
        "model": "claude-3-opus-20240229",
        "max_tokens": 1000,
        "temperature": 0.7,
        "timeout": 30.0,
        "max_retries": 3,
        "retry_delay": 1.0
    }
)
class Claude3OpusClient(BaseModel):
    """Anthropic Claude-3 Opus client for mental health conversations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Claude-3 Opus client."""
        default_config = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 30.0,
            "max_retries": 3,
            "retry_delay": 1.0
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("claude-3-opus", ModelProvider.ANTHROPIC, ModelType.CLOUD, default_config)
        
        # Check availability
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic library not installed. Install with: pip install anthropic"
            )
        
        # Initialize client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Get your API key from: https://console.anthropic.com/"
            )
        
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=self.config.get("timeout", 30.0)
        )
        
        # Mental health optimized system prompt
        self.system_prompt = """You are Claude, an AI assistant specialized in providing empathetic, supportive mental health conversations.

Your approach:
- Listen actively and validate emotions without judgment
- Use person-first language and show genuine empathy
- Reflect feelings and summarize to show understanding
- Ask open-ended questions to encourage expression
- Offer practical coping strategies when appropriate
- Recognize crisis situations and encourage professional help
- Maintain clear boundaries about your role as AI support
- Be culturally sensitive and inclusive in all responses
- Focus on strengths and resilience building
- Provide hope while acknowledging current difficulties

Important guidelines:
- Never provide medical diagnoses or prescribe treatments
- Encourage professional help for serious mental health concerns
- Respect confidentiality and privacy
- Avoid giving advice on medication or medical decisions
- If someone expresses suicidal thoughts, strongly encourage immediate professional help

Remember: You are a supportive companion to supplement, not replace, professional mental health care."""
    
    async def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using Claude-3 Opus."""
        try:
            # Prepare messages
            messages = []
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Merge config with kwargs
            request_config = {**self.config, **kwargs}
            
            # Implement retry logic
            max_retries = request_config.get("max_retries", 3)
            retry_delay = request_config.get("retry_delay", 1.0)
            
            for attempt in range(max_retries + 1):
                try:
                    start_time = datetime.now()
                    
                    response = await self.client.messages.create(
                        model=request_config["model"],
                        max_tokens=request_config["max_tokens"],
                        temperature=request_config["temperature"],
                        system=self.system_prompt,
                        messages=messages
                    )
                    
                    end_time = datetime.now()
                    response_time_ms = (end_time - start_time).total_seconds() * 1000
                    
                    # Extract content
                    content = response.content[0].text if response.content else ""
                    
                    # Calculate cost
                    cost_usd = self._calculate_cost(response.usage)
                    
                    # Create response object
                    model_response = ModelResponse(
                        content=content,
                        model_name=self.model_name,
                        timestamp=end_time,
                        response_time_ms=response_time_ms,
                        token_count=response.usage.input_tokens + response.usage.output_tokens,
                        cost_usd=cost_usd,
                        metadata={
                            "model": request_config["model"],
                            "temperature": request_config["temperature"],
                            "max_tokens": request_config["max_tokens"],
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "stop_reason": response.stop_reason,
                            "attempt": attempt + 1
                        }
                    )
                    
                    # Update metrics
                    self.metrics.update(model_response)
                    
                    self.logger.info(
                        f"Generated response: {len(content)} chars, "
                        f"{response.usage.input_tokens + response.usage.output_tokens} tokens, "
                        f"{response_time_ms:.2f}ms (attempt {attempt + 1})"
                    )
                    
                    return model_response
                    
                except anthropic.RateLimitError as e:
                    if attempt < max_retries:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        self.logger.warning(f"Rate limited, retrying in {wait_time}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise e
                        
                except anthropic.APITimeoutError as e:
                    if attempt < max_retries:
                        self.logger.warning(f"API timeout, retrying (attempt {attempt + 1})")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            self.logger.error(f"Error generating Claude response: {e}")
            
            error_response = ModelResponse(
                content="",
                model_name=self.model_name,
                timestamp=datetime.now(),
                response_time_ms=0.0,
                error=str(e)
            )
            
            self.metrics.update(error_response)
            return error_response
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on Claude-3 Opus pricing."""
        # Claude-3 Opus pricing (as of 2024)
        input_cost_per_1k = 0.015   # $15 per million input tokens
        output_cost_per_1k = 0.075  # $75 per million output tokens
        
        input_cost = (usage.input_tokens / 1000) * input_cost_per_1k
        output_cost = (usage.output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def validate_configuration(self) -> bool:
        """Validate Claude configuration."""
        try:
            if not ANTHROPIC_AVAILABLE:
                self.logger.error("anthropic library not available")
                return False
            
            if not os.getenv("ANTHROPIC_API_KEY"):
                self.logger.error("ANTHROPIC_API_KEY environment variable required")
                return False
            
            required_params = ["model", "temperature", "max_tokens"]
            for param in required_params:
                if param not in self.config:
                    self.logger.error(f"Missing required parameter: {param}")
                    return False
            
            if not 0 <= self.config["temperature"] <= 1:
                self.logger.error("Temperature must be between 0 and 1")
                return False
            
            if not 1 <= self.config["max_tokens"] <= 4096:
                self.logger.error("Max tokens must be between 1 and 4096")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Perform health check with Claude API."""
        try:
            if not ANTHROPIC_AVAILABLE:
                return False
                
            response = await self.generate_response(
                "Hello, this is a health check.",
                **{"max_tokens": 10, "temperature": 0.1}
            )
            
            is_healthy = response.is_successful and len(response.content) > 0
            
            if is_healthy:
                self.logger.info("Claude health check passed")
            else:
                self.logger.warning("Claude health check failed")
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Claude model information."""
        return {
            "provider": "Anthropic",
            "model_name": self.model_name,
            "model_id": self.config["model"],
            "type": "cloud",
            "max_context_length": 200000,  # Claude-3 context length
            "supports_streaming": True,
            "supports_function_calling": True,
            "training_data_cutoff": "2024-02",
            "config": self.config,
            "available": ANTHROPIC_AVAILABLE,
            "pricing": {
                "input_per_1k_tokens": 0.015,
                "output_per_1k_tokens": 0.075,
                "currency": "USD"
            }
        }


@register_model_decorator(
    name="claude-3-sonnet",
    provider=ModelProvider.ANTHROPIC,
    model_type=ModelType.CLOUD,
    description="Anthropic Claude-3 Sonnet - Balanced performance and cost for mental health conversations",
    requirements=["anthropic"],
    default_config={
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1000,
        "temperature": 0.7,
        "timeout": 30.0,
        "max_retries": 3
    }
)
class Claude3SonnetClient(Claude3OpusClient):
    """Claude-3 Sonnet variant - balanced performance and cost."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 30.0,
            "max_retries": 3
        }
        
        if config:
            default_config.update(config)
        
        BaseModel.__init__(self, "claude-3-sonnet", ModelProvider.ANTHROPIC, ModelType.CLOUD, default_config)
        
        # Initialize same as parent
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library not installed. Install with: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.client = AsyncAnthropic(api_key=api_key, timeout=self.config.get("timeout", 30.0))
        self.system_prompt = """You are Claude, an AI assistant specialized in providing empathetic, supportive mental health conversations.

Your approach:
- Listen actively and validate emotions without judgment
- Use person-first language and show genuine empathy
- Reflect feelings and summarize to show understanding
- Ask open-ended questions to encourage expression
- Offer practical coping strategies when appropriate
- Recognize crisis situations and encourage professional help
- Maintain clear boundaries about your role as AI support
- Be culturally sensitive and inclusive in all responses
- Focus on strengths and resilience building
- Provide hope while acknowledging current difficulties

Important guidelines:
- Never provide medical diagnoses or prescribe treatments
- Encourage professional help for serious mental health concerns
- Respect confidentiality and privacy
- Avoid giving advice on medication or medical decisions
- If someone expresses suicidal thoughts, strongly encourage immediate professional help

Remember: You are a supportive companion to supplement, not replace, professional mental health care."""
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on Claude-3 Sonnet pricing."""
        # Claude-3 Sonnet pricing (as of 2024)
        input_cost_per_1k = 0.003   # $3 per million input tokens
        output_cost_per_1k = 0.015  # $15 per million output tokens
        
        input_cost = (usage.input_tokens / 1000) * input_cost_per_1k
        output_cost = (usage.output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Claude-3 Sonnet model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "pricing": {
                "input_per_1k_tokens": 0.003,
                "output_per_1k_tokens": 0.015,
                "currency": "USD"
            },
            "description": "Balanced performance and cost variant of Claude-3"
        })
        return info


@register_model_decorator(
    name="claude-3-haiku",
    provider=ModelProvider.ANTHROPIC,
    model_type=ModelType.CLOUD,
    description="Anthropic Claude-3 Haiku - Fast and cost-effective for mental health conversations",
    requirements=["anthropic"],
    default_config={
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1000,
        "temperature": 0.7,
        "timeout": 30.0,
        "max_retries": 3
    }
)
class Claude3HaikuClient(Claude3OpusClient):
    """Claude-3 Haiku variant - fast and cost-effective."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 30.0,
            "max_retries": 3
        }
        
        if config:
            default_config.update(config)
        
        BaseModel.__init__(self, "claude-3-haiku", ModelProvider.ANTHROPIC, ModelType.CLOUD, default_config)
        
        # Initialize same as parent
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library not installed. Install with: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.client = AsyncAnthropic(api_key=api_key, timeout=self.config.get("timeout", 30.0))
        self.system_prompt = """You are Claude, an AI assistant specialized in providing empathetic, supportive mental health conversations.

Your approach:
- Listen actively and validate emotions without judgment
- Use person-first language and show genuine empathy
- Reflect feelings and summarize to show understanding
- Ask open-ended questions to encourage expression
- Offer practical coping strategies when appropriate
- Recognize crisis situations and encourage professional help
- Maintain clear boundaries about your role as AI support
- Be culturally sensitive and inclusive in all responses
- Focus on strengths and resilience building
- Provide hope while acknowledging current difficulties

Important guidelines:
- Never provide medical diagnoses or prescribe treatments
- Encourage professional help for serious mental health concerns
- Respect confidentiality and privacy
- Avoid giving advice on medication or medical decisions
- If someone expresses suicidal thoughts, strongly encourage immediate professional help

Remember: You are a supportive companion to supplement, not replace, professional mental health care."""
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on Claude-3 Haiku pricing."""
        # Claude-3 Haiku pricing (as of 2024)
        input_cost_per_1k = 0.00025   # $0.25 per million input tokens
        output_cost_per_1k = 0.00125  # $1.25 per million output tokens
        
        input_cost = (usage.input_tokens / 1000) * input_cost_per_1k
        output_cost = (usage.output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Claude-3 Haiku model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "pricing": {
                "input_per_1k_tokens": 0.00025,
                "output_per_1k_tokens": 0.00125,
                "currency": "USD"
            },
            "description": "Fast and cost-effective variant of Claude-3"
        })
        return info


# Optional: Claude Instant for even faster/cheaper responses
@register_model_decorator(
    name="claude-instant",
    provider=ModelProvider.ANTHROPIC,
    model_type=ModelType.CLOUD,
    description="Claude Instant - Legacy fast model for basic mental health conversations",
    requirements=["anthropic"],
    default_config={
        "model": "claude-instant-1.2",
        "max_tokens": 1000,
        "temperature": 0.7,
        "timeout": 30.0
    }
)
class ClaudeInstantClient(Claude3OpusClient):
    """Claude Instant legacy model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "model": "claude-instant-1.2",
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 30.0
        }
        
        if config:
            default_config.update(config)
        
        BaseModel.__init__(self, "claude-instant", ModelProvider.ANTHROPIC, ModelType.CLOUD, default_config)
        
        # Initialize same as parent
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic library not installed. Install with: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.client = AsyncAnthropic(api_key=api_key, timeout=self.config.get("timeout", 30.0))
        self.system_prompt = """You are Claude, an AI assistant specialized in providing empathetic, supportive mental health conversations.

Your approach:
- Listen actively and validate emotions without judgment
- Use person-first language and show genuine empathy
- Reflect feelings and summarize to show understanding
- Ask open-ended questions to encourage expression
- Offer practical coping strategies when appropriate
- Recognize crisis situations and encourage professional help
- Maintain clear boundaries about your role as AI support
- Be culturally sensitive and inclusive in all responses
- Focus on strengths and resilience building
- Provide hope while acknowledging current difficulties

Important guidelines:
- Never provide medical diagnoses or prescribe treatments
- Encourage professional help for serious mental health concerns
- Respect confidentiality and privacy
- Avoid giving advice on medication or medical decisions
- If someone expresses suicidal thoughts, strongly encourage immediate professional help

Remember: You are a supportive companion to supplement, not replace, professional mental health care."""
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on Claude Instant pricing."""
        # Claude Instant pricing (as of 2024)
        input_cost_per_1k = 0.0008   # $0.80 per million input tokens
        output_cost_per_1k = 0.0024  # $2.40 per million output tokens
        
        input_cost = (usage.input_tokens / 1000) * input_cost_per_1k
        output_cost = (usage.output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Claude Instant model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "max_context_length": 100000,  # Claude Instant context length
            "pricing": {
                "input_per_1k_tokens": 0.0008,
                "output_per_1k_tokens": 0.0024,
                "currency": "USD"
            },
            "description": "Legacy fast and cost-effective Claude model"
        })
        return info