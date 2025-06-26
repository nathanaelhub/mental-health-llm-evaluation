"""
Llama Integration Template

Ready-to-use template for integrating Meta Llama models locally.
This template provides a complete, production-ready implementation for local Llama inference.

SETUP INSTRUCTIONS:
1. Install required packages: pip install transformers torch accelerate bitsandbytes
2. Download Llama models to local directory:
   - Llama-2-7B: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
   - Llama-2-13B: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
3. Copy this file to src/models/llama_client.py
4. Update model paths in config/experiment_template.yaml
5. Enable models: models.local[0].enabled: true
6. Test: python scripts/model_management.py test llama-2-7b

FEATURES:
- Complete local Llama inference
- Multiple model sizes (7B, 13B, 70B)
- GPU/CPU device management
- Quantization support (8-bit, 4-bit)
- Memory optimization
- Mental health conversation optimization
"""

import os
import asyncio
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Try to import required libraries
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        GenerationConfig,
        BitsAndBytesConfig
    )
    import accelerate
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from src.models.base_model import BaseModel, ModelResponse, ModelProvider, ModelType
from src.models.model_registry import register_model_decorator


@register_model_decorator(
    name="llama-2-7b-chat",
    provider=ModelProvider.META,
    model_type=ModelType.LOCAL,
    description="Meta Llama-2 7B Chat model for local mental health conversations",
    requirements=["transformers", "torch", "accelerate"],
    default_config={
        "model_path": "./models/llama-2-7b-chat-hf",
        "device": "auto",
        "precision": "fp16",
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "trust_remote_code": False
    }
)
class Llama27BChatClient(BaseModel):
    """Meta Llama-2 7B Chat client for local mental health conversations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Llama-2 7B Chat client."""
        default_config = {
            "model_path": "./models/llama-2-7b-chat-hf",
            "device": "auto",
            "precision": "fp16",
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "load_in_8bit": False,
            "load_in_4bit": False,
            "trust_remote_code": False,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "use_cache": True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("llama-2-7b-chat", ModelProvider.META, ModelType.LOCAL, default_config)
        
        # Check dependencies
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library not installed. Install with: "
                "pip install transformers torch accelerate bitsandbytes"
            )
        
        if not TORCH_AVAILABLE:
            raise ImportError("torch library not installed. Install with: pip install torch")
        
        # Initialize state
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._model_loaded = False
        
        # Mental health optimized system prompt for Llama-2
        self.system_prompt = """You are a helpful, caring, and empathetic AI assistant specialized in mental health support conversations. You provide compassionate responses while maintaining appropriate boundaries.

Your guidelines:
- Show deep empathy and genuine understanding
- Use active listening and validation techniques
- Ask thoughtful, open-ended questions
- Reflect feelings and summarize to show understanding
- Offer practical coping strategies when appropriate
- Encourage professional help for serious concerns
- Maintain clear boundaries about your role as AI support
- Use person-first, non-judgmental language
- Be culturally sensitive and inclusive
- Focus on strengths and resilience building
- Provide hope while acknowledging current struggles

Important limitations:
- Never provide medical diagnoses or prescribe treatments
- Don't give specific advice about medications
- Encourage professional help for serious mental health issues
- If someone expresses suicidal thoughts, strongly encourage immediate professional help
- Respect privacy and confidentiality

Remember: You are a supportive companion to supplement, not replace, professional mental health care."""
    
    def _load_model(self) -> None:
        """Load the Llama model and tokenizer."""
        if self._model_loaded:
            return
        
        try:
            model_path = self.config["model_path"]
            
            # Validate model path
            if not Path(model_path).exists() and not model_path.startswith("meta-llama/"):
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            self.logger.info(f"Loading Llama model from: {model_path}")
            
            # Configure device and precision
            device = self.config["device"]
            precision = self.config["precision"]
            
            # Set torch dtype
            if precision == "fp16":
                torch_dtype = torch.float16
            elif precision == "bf16":
                torch_dtype = torch.bfloat16
            elif precision == "fp32":
                torch_dtype = torch.float32
            else:
                torch_dtype = "auto"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config["trust_remote_code"],
                use_fast=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configure quantization
            quantization_config = None
            if self.config["load_in_4bit"]:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype
                )
            elif self.config["load_in_8bit"]:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Configure model loading arguments
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": self.config["trust_remote_code"],
                "low_cpu_mem_usage": self.config["low_cpu_mem_usage"],
                "use_cache": self.config["use_cache"]
            }
            
            # Add device mapping
            if device == "auto":
                model_kwargs["device_map"] = "auto"
            elif device != "cpu":
                model_kwargs["device_map"] = {"": device}
            
            # Add quantization config
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if device != "auto" and not quantization_config:
                self.model = self.model.to(device)
            
            # Set generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                repetition_penalty=self.config["repetition_penalty"],
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            self._model_loaded = True
            self.logger.info("Llama model loaded successfully")
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_info = self._get_memory_usage()
                self.logger.info(f"GPU Memory: {memory_info.get('gpu_memory_allocated_gb', 0):.2f}GB allocated")
            
        except Exception as e:
            self.logger.error(f"Failed to load Llama model: {e}")
            raise
    
    def _format_prompt(self, prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Format prompt for Llama-2 chat format."""
        # Llama-2 chat format: <s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]
        
        formatted_parts = []
        
        # Start with system message
        formatted_parts.append(f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n")
        
        # Add conversation history
        if conversation_history:
            for i, msg in enumerate(conversation_history):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    if i == 0:
                        # First user message continues the initial [INST]
                        formatted_parts.append(f"{content} [/INST]")
                    else:
                        # Subsequent user messages start new [INST]
                        formatted_parts.append(f"<s>[INST] {content} [/INST]")
                elif role == "assistant":
                    formatted_parts.append(f" {content} </s>")
        
        # Add current prompt
        if conversation_history:
            # If we have history, start new instruction
            formatted_parts.append(f"<s>[INST] {prompt} [/INST]")
        else:
            # First message continues the initial instruction
            formatted_parts.append(f"{prompt} [/INST]")
        
        return "".join(formatted_parts)
    
    async def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using local Llama model."""
        try:
            # Load model if not already loaded
            if not self._model_loaded:
                self._load_model()
            
            # Format the prompt
            formatted_prompt = self._format_prompt(prompt, conversation_history)
            
            # Merge config with kwargs
            request_config = {**self.config, **kwargs}
            
            # Update generation config with any overrides
            generation_config = GenerationConfig(
                max_new_tokens=request_config.get("max_tokens", self.config["max_tokens"]),
                temperature=request_config.get("temperature", self.config["temperature"]),
                top_p=request_config.get("top_p", self.config["top_p"]),
                repetition_penalty=request_config.get("repetition_penalty", self.config["repetition_penalty"]),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=4096 - generation_config.max_new_tokens  # Leave space for generation
            )
            
            # Move to appropriate device
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                device = next(self.model.parameters()).device
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            start_time = datetime.now()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,  # Don't need scores for basic generation
                    use_cache=True
                )
            
            end_time = datetime.now()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Decode response
            response_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            content = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Clean up response
            content = content.strip()
            
            # Remove any remaining special tokens or artifacts
            if content.endswith("</s>"):
                content = content[:-4].strip()
            
            # Calculate tokens
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = len(response_tokens)
            total_tokens = input_tokens + output_tokens
            
            # Estimate cost (local inference - mainly electricity/compute cost)
            cost_usd = self._calculate_cost(total_tokens, response_time_ms)
            
            # Create response object
            model_response = ModelResponse(
                content=content,
                model_name=self.model_name,
                timestamp=end_time,
                response_time_ms=response_time_ms,
                token_count=total_tokens,
                cost_usd=cost_usd,
                metadata={
                    "model_path": self.config["model_path"],
                    "temperature": generation_config.temperature,
                    "max_tokens": generation_config.max_new_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "device": str(device),
                    "precision": self.config["precision"],
                    "quantization": "4bit" if self.config["load_in_4bit"] else "8bit" if self.config["load_in_8bit"] else "none"
                }
            )
            
            # Update metrics
            self.metrics.update(model_response)
            
            self.logger.info(
                f"Generated response: {len(content)} chars, "
                f"{total_tokens} tokens, "
                f"{response_time_ms:.2f}ms"
            )
            
            return model_response
            
        except Exception as e:
            self.logger.error(f"Error generating Llama response: {e}")
            
            error_response = ModelResponse(
                content="",
                model_name=self.model_name,
                timestamp=datetime.now(),
                response_time_ms=0.0,
                error=str(e)
            )
            
            self.metrics.update(error_response)
            return error_response
    
    def _calculate_cost(self, total_tokens: int, response_time_ms: float) -> float:
        """Calculate approximate cost for local inference."""
        # Rough estimate based on compute cost
        # Factors: GPU power consumption, electricity cost, amortized hardware cost
        cost_per_1k_tokens = 0.0001  # $0.0001 per 1K tokens (very rough estimate)
        return (total_tokens / 1000) * cost_per_1k_tokens
    
    def validate_configuration(self) -> bool:
        """Validate Llama configuration."""
        try:
            # Check dependencies
            if not TRANSFORMERS_AVAILABLE:
                self.logger.error("transformers library not available")
                return False
            
            if not TORCH_AVAILABLE:
                self.logger.error("torch library not available")
                return False
            
            # Check model path
            model_path = self.config.get("model_path")
            if not model_path:
                self.logger.error("model_path not specified")
                return False
            
            # Check if path exists or is a valid HF model ID
            if not Path(model_path).exists() and not model_path.startswith("meta-llama/"):
                self.logger.warning(f"Model path may not exist: {model_path}")
            
            # Check required config parameters
            required_params = ["model_path", "temperature", "max_tokens"]
            for param in required_params:
                if param not in self.config:
                    self.logger.error(f"Missing required config parameter: {param}")
                    return False
            
            # Validate parameter ranges
            if not 0 <= self.config["temperature"] <= 2:
                self.logger.error("Temperature must be between 0 and 2")
                return False
            
            if not 1 <= self.config["max_tokens"] <= 4096:
                self.logger.error("Max tokens must be between 1 and 4096")
                return False
            
            # Check device availability
            device = self.config.get("device", "auto")
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, will use CPU")
            
            self.logger.info("Llama configuration validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Perform health check by loading model and making a simple generation."""
        try:
            if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
                self.logger.error("Required libraries not available for health check")
                return False
            
            # Try to load model
            if not self._model_loaded:
                self._load_model()
            
            # Test generation
            response = await self.generate_response(
                "Hello, this is a health check.",
                **{"max_tokens": 10, "temperature": 0.1}
            )
            
            is_healthy = response.is_successful and len(response.content) > 0
            
            if is_healthy:
                self.logger.info("Llama health check passed")
            else:
                self.logger.warning("Llama health check failed")
            
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        info = {
            "provider": "Meta",
            "model_name": self.model_name,
            "model_path": self.config["model_path"],
            "type": "local",
            "max_context_length": 4096,  # Llama-2 context length
            "supports_streaming": False,
            "supports_function_calling": False,
            "training_data_cutoff": "2023-07",
            "config": self.config,
            "available": TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE,
            "model_loaded": self._model_loaded,
            "model_size": "7B parameters"
        }
        
        # Add device info if model is loaded
        if self._model_loaded and self.model:
            if hasattr(self.model, 'device'):
                info["device"] = str(self.model.device)
            if hasattr(self.model, 'dtype'):
                info["dtype"] = str(self.model.dtype)
            info["memory_usage"] = self._get_memory_usage()
        
        return info
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved(),
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        return memory_info
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model_loaded:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model_loaded = False
            self.logger.info("Llama model unloaded")


# Llama-2 13B Chat variant
@register_model_decorator(
    name="llama-2-13b-chat",
    provider=ModelProvider.META,
    model_type=ModelType.LOCAL,
    description="Meta Llama-2 13B Chat model for high-quality local mental health conversations",
    requirements=["transformers", "torch", "accelerate", "bitsandbytes"],
    default_config={
        "model_path": "./models/llama-2-13b-chat-hf",
        "device": "auto",
        "precision": "fp16",
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "load_in_8bit": True,  # Enable 8-bit for larger model
        "load_in_4bit": False
    }
)
class Llama213BChatClient(Llama27BChatClient):
    """Llama-2 13B Chat variant for higher quality responses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "model_path": "./models/llama-2-13b-chat-hf",
            "device": "auto",
            "precision": "fp16",
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "load_in_8bit": True,
            "load_in_4bit": False,
            "trust_remote_code": False,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "use_cache": True
        }
        
        if config:
            default_config.update(config)
        
        # Call BaseModel.__init__ directly to override model_name
        BaseModel.__init__(self, "llama-2-13b-chat", ModelProvider.META, ModelType.LOCAL, default_config)
        
        # Initialize same as parent but don't load model yet
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._model_loaded = False
        
        self.system_prompt = """You are a helpful, caring, and empathetic AI assistant specialized in mental health support conversations. You provide compassionate responses while maintaining appropriate boundaries.

Your guidelines:
- Show deep empathy and genuine understanding
- Use active listening and validation techniques
- Ask thoughtful, open-ended questions
- Reflect feelings and summarize to show understanding
- Offer practical coping strategies when appropriate
- Encourage professional help for serious concerns
- Maintain clear boundaries about your role as AI support
- Use person-first, non-judgmental language
- Be culturally sensitive and inclusive
- Focus on strengths and resilience building
- Provide hope while acknowledging current struggles

Important limitations:
- Never provide medical diagnoses or prescribe treatments
- Don't give specific advice about medications
- Encourage professional help for serious mental health issues
- If someone expresses suicidal thoughts, strongly encourage immediate professional help
- Respect privacy and confidentiality

Remember: You are a supportive companion to supplement, not replace, professional mental health care."""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Llama-2 13B model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "model_size": "13B parameters",
            "description": "Larger, higher-quality variant of Llama-2"
        })
        return info


# Llama-2 70B Chat variant (requires significant resources)
@register_model_decorator(
    name="llama-2-70b-chat",
    provider=ModelProvider.META,
    model_type=ModelType.LOCAL,
    description="Meta Llama-2 70B Chat model for highest quality local mental health conversations (requires 40GB+ GPU memory)",
    requirements=["transformers", "torch", "accelerate", "bitsandbytes"],
    default_config={
        "model_path": "./models/llama-2-70b-chat-hf",
        "device": "auto",
        "precision": "fp16",
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "load_in_8bit": False,
        "load_in_4bit": True,  # 4-bit quantization essential for 70B
        "trust_remote_code": False
    }
)
class Llama270BChatClient(Llama27BChatClient):
    """Llama-2 70B Chat variant for highest quality responses."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "model_path": "./models/llama-2-70b-chat-hf",
            "device": "auto",
            "precision": "fp16",
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "load_in_8bit": False,
            "load_in_4bit": True,
            "trust_remote_code": False,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "use_cache": True
        }
        
        if config:
            default_config.update(config)
        
        BaseModel.__init__(self, "llama-2-70b-chat", ModelProvider.META, ModelType.LOCAL, default_config)
        
        # Initialize same as parent
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._model_loaded = False
        
        self.system_prompt = """You are a helpful, caring, and empathetic AI assistant specialized in mental health support conversations. You provide compassionate responses while maintaining appropriate boundaries.

Your guidelines:
- Show deep empathy and genuine understanding
- Use active listening and validation techniques
- Ask thoughtful, open-ended questions
- Reflect feelings and summarize to show understanding
- Offer practical coping strategies when appropriate
- Encourage professional help for serious concerns
- Maintain clear boundaries about your role as AI support
- Use person-first, non-judgmental language
- Be culturally sensitive and inclusive
- Focus on strengths and resilience building
- Provide hope while acknowledging current struggles

Important limitations:
- Never provide medical diagnoses or prescribe treatments
- Don't give specific advice about medications
- Encourage professional help for serious mental health issues
- If someone expresses suicidal thoughts, strongly encourage immediate professional help
- Respect privacy and confidentiality

Remember: You are a supportive companion to supplement, not replace, professional mental health care."""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Llama-2 70B model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "model_size": "70B parameters",
            "description": "Largest, highest-quality variant of Llama-2 (requires significant GPU memory)",
            "minimum_gpu_memory": "40GB (with 4-bit quantization)"
        })
        return info