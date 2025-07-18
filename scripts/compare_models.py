#!/usr/bin/env python3
"""
Multi-Model Comparison CLI Tool
===============================

Multi-model comparison tool for OpenAI, Claude, DeepSeek, and Gemma models.
Provides side-by-side comparison with performance metrics and automatic response saving.

Usage:
    python compare_models.py "Your prompt here"
    python compare_models.py --models openai,claude,deepseek
    python compare_models.py --interactive
    python compare_models.py "Hello, how are you?" --save results.json
    python compare_models.py --batch prompts.txt --verbose
    python compare_models.py --scenarios 3 "Your prompt"
    
Features:
- Multi-model response comparison (OpenAI, Claude, DeepSeek, Gemma)
- Automatic response saving to results/ directory
- Clean terminal output with shortened responses
- Response time measurement
- Token usage tracking
- Cost estimation
- Mental health quality assessment
- Interactive mode
- Batch processing

Prerequisites:
- Configure .env file with API keys for desired models
- For local models: Ensure your local LLM server is running
"""

import asyncio
import argparse
import json
import time
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import logging

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.local_llm_client import LocalLLMClient
import openai

# Try to import additional model clients
try:
    from src.models.claude_client import ClaudeClient
    HAS_CLAUDE = True
except ImportError:
    ClaudeClient = None
    HAS_CLAUDE = False

try:
    from src.models.gemma_client import GemmaClient
    HAS_GEMMA = True
except ImportError:
    GemmaClient = None
    HAS_GEMMA = False


@dataclass
class QualityMetrics:
    """Mental health quality assessment metrics."""
    empathy_score: float = 0.0
    therapeutic_value: float = 0.0
    clarity_score: float = 0.0
    crisis_detection: bool = False
    safety_score: float = 0.0
    conciseness_score: float = 0.0
    professional_help_encouragement: float = 0.0
    

@dataclass
class ModelResponse:
    """Response data from a model."""
    model_name: str
    content: str
    original_content: str  # Store original before filtering
    response_time_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    quality_metrics: Optional[QualityMetrics] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.quality_metrics is None:
            self.quality_metrics = QualityMetrics()


@dataclass
class ComparisonResult:
    """Comparison result between models."""
    prompt: str
    timestamp: str
    model_responses: Dict[str, ModelResponse]
    speed_winner: Optional[str] = None
    quality_winner: Optional[str] = None
    overall_winner: Optional[str] = None
    winner_mode: str = "speed"  # "speed", "quality", "balanced", "research"
    confidence_score: float = 0.0
    notes: str = ""
    
    # Backward compatibility properties
    @property
    def openai_response(self) -> Optional[ModelResponse]:
        return self.model_responses.get('openai')
    
    @property
    def local_response(self) -> Optional[ModelResponse]:
        return self.model_responses.get('deepseek', self.model_responses.get('local'))


class ModelComparator:
    """Handles comparison between multiple LLM models."""
    
    # Model pricing per 1K tokens (as of 2024)
    MODEL_PRICING = {
        # OpenAI models
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        # Claude models
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        # Gemini models
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
        "gemini-1.5-pro": {"input": 0.001, "output": 0.002},
        # Local models (free)
        "deepseek": {"input": 0.0, "output": 0.0},
        "local": {"input": 0.0, "output": 0.0},
    }
    
    # Model name mappings for cleaner display
    MODEL_NAME_MAP = {
        "deepseek": "DeepSeek R1",
        "llama": "Llama",
        "mistral": "Mistral",
        "codellama": "Code Llama",
        "qwen": "Qwen",
        "vicuna": "Vicuna",
        "claude": "Claude",
        "gpt": "GPT",
        "phi": "Phi",
        "gemma": "Gemma"
    }
    
    def _safe_asdict(self, obj) -> dict:
        """Safely convert dataclass to dict, handling serialization issues."""
        try:
            result = asdict(obj)
            # Test JSON serialization
            json.dumps(result, default=str)
            return result
        except (TypeError, ValueError) as e:
            print(f"⚠️  Warning: Serialization issue with result data: {e}")
            print(f"   Attempting to create safe version...")
            
            # Create a safe version manually
            if hasattr(obj, '__dict__'):
                safe_dict = {}
                for key, value in obj.__dict__.items():
                    try:
                        # Test if this specific field can be serialized
                        json.dumps(value, default=str)
                        safe_dict[key] = value
                    except (TypeError, ValueError):
                        safe_dict[key] = str(value)
                return safe_dict
            else:
                return {"error": "Could not serialize object", "type": type(obj).__name__}
    
    def __init__(self, verbose: bool = False, quiet: bool = False, hide_reasoning: bool = False,
                 max_response_length: Optional[int] = None, winner_mode: str = "speed", 
                 selected_models: Optional[List[str]] = None):
        """Initialize the comparator with selected model clients."""
        load_dotenv()
        
        self.verbose = verbose
        self.quiet = quiet
        self.hide_reasoning = hide_reasoning
        self.max_response_length = max_response_length
        self.winner_mode = winner_mode
        self.selected_models = selected_models or ['openai', 'claude', 'deepseek', 'gemma']
        
        # Initialize model clients
        self.model_clients = {}
        self.model_names = {}
        
        # Initialize OpenAI client
        if 'openai' in self.selected_models:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.model_clients['openai'] = openai.OpenAI(api_key=openai_key)
                self.model_names['openai'] = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
            else:
                print("⚠️  OPENAI_API_KEY not found, skipping OpenAI")
                self.selected_models.remove('openai')
        
        # Initialize Claude client
        if 'claude' in self.selected_models and HAS_CLAUDE:
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if claude_key:
                self.model_clients['claude'] = ClaudeClient()
                self.model_names['claude'] = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
            else:
                print("⚠️  ANTHROPIC_API_KEY not found, skipping Claude")
                self.selected_models.remove('claude')
        elif 'claude' in self.selected_models:
            print("⚠️  Claude client not available, skipping Claude")
            self.selected_models.remove('claude')
        
        # Initialize DeepSeek/Local client
        if 'deepseek' in self.selected_models:
            local_url = os.getenv("LOCAL_LLM_BASE_URL", "http://192.168.86.30:1234/v1")
            local_model = os.getenv("LOCAL_LLM_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b")
            
            self.model_clients['deepseek'] = LocalLLMClient(
                base_url=local_url,
                model_name=local_model,
                timeout=float(os.getenv("LOCAL_LLM_TIMEOUT", "60"))
            )
            self.model_names['deepseek'] = local_model
        
        # Initialize Gemma client
        if 'gemma' in self.selected_models and HAS_GEMMA:
            self.model_clients['gemma'] = GemmaClient()
            self.model_names['gemma'] = os.getenv("GEMMA_MODEL", "gemini-pro")
        elif 'gemma' in self.selected_models:
            print("⚠️  Gemma client not available, skipping Gemma")
            self.selected_models.remove('gemma')
        
        # Ensure we have at least one model
        if len(self.selected_models) == 0:
            raise ValueError("No valid models available. Please check your API keys and configuration.")
        
        # Set up logging based on verbosity
        log_level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        if not quiet:
            print(f"🤖 Multi-Model Comparison Tool")
            for model in self.selected_models:
                model_name = self.model_names.get(model, model)
                print(f"   {model.title()}: {self._get_friendly_model_name(model_name)}")
            if verbose:
                print(f"🔍 Verbose logging enabled")
    
    async def query_openai(self, prompt: str, system_prompt: Optional[str] = None) -> ModelResponse:
        """Query OpenAI model."""
        start_time = time.time()
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Extract response data
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost
            cost = self._calculate_model_cost(
                self.model_names['openai'],
                usage.prompt_tokens,
                usage.completion_tokens
            )
            
            # Filter and assess content
            original_content = content
            filtered_content = self._filter_reasoning_and_length(content)
            quality_metrics = self._assess_mental_health_quality(filtered_content, original_content)
            
            return ModelResponse(
                model_name=self.model_names['openai'],
                content=filtered_content,
                original_content=original_content,
                response_time_ms=response_time_ms,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=cost,
                quality_metrics=quality_metrics,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            )
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return ModelResponse(
                model_name=self.model_names['openai'],
                content="",
                original_content="",
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    async def _query_claude(self, prompt: str, system_prompt: Optional[str] = None) -> ModelResponse:
        """Query Claude model."""
        start_time = time.time()
        
        try:
            response = await self.model_clients['claude'].generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2048
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Filter and assess content
            original_content = response.content
            filtered_content = self._filter_reasoning_and_length(response.content)
            quality_metrics = self._assess_mental_health_quality(filtered_content, original_content)
            
            # Calculate cost
            cost = self._calculate_model_cost(
                self.model_names['claude'],
                response.input_token_count or 0,
                response.output_token_count or 0
            )
            
            return ModelResponse(
                model_name=self.model_names['claude'],
                content=filtered_content,
                original_content=original_content,
                response_time_ms=response_time_ms,
                input_tokens=response.input_token_count,
                output_tokens=response.output_token_count,
                total_tokens=response.token_count,
                cost_usd=cost,
                quality_metrics=quality_metrics,
                metadata={
                    "finish_reason": response.finish_reason
                }
            )
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return ModelResponse(
                model_name=self.model_names['claude'],
                content="",
                original_content="",
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    async def _query_deepseek(self, prompt: str, system_prompt: Optional[str] = None) -> ModelResponse:
        """Query DeepSeek/Local model."""
        start_time = time.time()
        
        try:
            response = await self.model_clients['deepseek'].generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2048
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Filter and assess content
            original_content = response.content
            filtered_content = self._filter_reasoning_and_length(response.content)
            quality_metrics = self._assess_mental_health_quality(filtered_content, original_content)
            
            return ModelResponse(
                model_name=self.model_names['deepseek'],
                content=filtered_content,
                original_content=original_content,
                response_time_ms=response_time_ms,
                input_tokens=response.input_token_count,
                output_tokens=response.output_token_count,
                total_tokens=response.token_count,
                cost_usd=0.0,  # Local model has no cost
                quality_metrics=quality_metrics,
                metadata={
                    "finish_reason": response.finish_reason,
                    "endpoint": self.model_clients['deepseek'].base_url
                }
            )
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            self.logger.error(f"DeepSeek query failed: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            return ModelResponse(
                model_name=self.model_names['deepseek'],
                content="",
                original_content="",
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    async def _query_gemma(self, prompt: str, system_prompt: Optional[str] = None) -> ModelResponse:
        """Query Gemma model."""
        start_time = time.time()
        
        try:
            response = await self.model_clients['gemma'].generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2048
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Filter and assess content
            original_content = response.content
            filtered_content = self._filter_reasoning_and_length(response.content)
            quality_metrics = self._assess_mental_health_quality(filtered_content, original_content)
            
            # Calculate cost
            cost = self._calculate_model_cost(
                self.model_names['gemma'],
                response.input_token_count or 0,
                response.output_token_count or 0
            )
            
            return ModelResponse(
                model_name=self.model_names['gemma'],
                content=filtered_content,
                original_content=original_content,
                response_time_ms=response_time_ms,
                input_tokens=response.input_token_count,
                output_tokens=response.output_token_count,
                total_tokens=response.token_count,
                cost_usd=cost,
                quality_metrics=quality_metrics,
                metadata={
                    "finish_reason": response.finish_reason
                }
            )
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return ModelResponse(
                model_name=self.model_names['gemma'],
                content="",
                original_content="",
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    def _calculate_model_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for model API usage."""
        if model not in self.MODEL_PRICING:
            return 0.0
        
        pricing = self.MODEL_PRICING[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def query_model(self, model_key: str, prompt: str, system_prompt: Optional[str] = None) -> ModelResponse:
        """Query a specific model."""
        start_time = time.time()
        
        try:
            if model_key == 'openai':
                return await self._query_openai(prompt, system_prompt)
            elif model_key == 'claude':
                return await self._query_claude(prompt, system_prompt)
            elif model_key == 'deepseek':
                return await self._query_deepseek(prompt, system_prompt)
            elif model_key == 'gemma':
                return await self._query_gemma(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown model: {model_key}")
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return ModelResponse(
                model_name=model_key,
                content="",
                original_content="",
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    async def compare(self, prompt: str, system_prompt: Optional[str] = None) -> ComparisonResult:
        """Compare responses from all selected models."""
        if not self.quiet:
            print(f"\n🤖 Multi-Model Comparison Tool (OpenAI, Claude, DeepSeek, Gemma)")
            print(f"\nComparing {len(self.selected_models)} models for: \"{prompt}\"")
        else:
            print(f"\n🔄 Comparing responses for: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
        
        # Query all selected models concurrently
        tasks = []
        for model_key in self.selected_models:
            tasks.append(self.query_model(model_key, prompt, system_prompt))
        
        responses = await asyncio.gather(*tasks)
        
        # Create model responses dictionary
        model_responses = {}
        for model_key, response in zip(self.selected_models, responses):
            model_responses[model_key] = response
        
        # Determine winners based on different criteria
        speed_winner = self._determine_speed_winner(model_responses)
        quality_winner = self._determine_quality_winner(model_responses)
        overall_winner, confidence = self._determine_overall_winner(
            model_responses, speed_winner, quality_winner
        )
        
        result = ComparisonResult(
            prompt=prompt,
            timestamp=datetime.now().isoformat(),
            model_responses=model_responses,
            speed_winner=speed_winner,
            quality_winner=quality_winner,
            overall_winner=overall_winner,
            winner_mode=self.winner_mode,
            confidence_score=confidence
        )
        
        # Auto-save result to file
        saved_file = self._auto_save_result(result)
        
        # Add saved file info to result for display
        result.notes = f"Full results saved to: {saved_file}" if saved_file else ""
        
        return result
    
    def _get_friendly_model_name(self, technical_name: str) -> str:
        """Convert technical model name to friendly display name."""
        technical_name = technical_name.lower()
        
        # Check for known model patterns
        for key, friendly_name in self.MODEL_NAME_MAP.items():
            if key in technical_name:
                # Extract version info if available
                version_match = re.search(r'(\d+(?:\.\d+)?(?:b|m)?)', technical_name)
                if version_match:
                    version = version_match.group(1)
                    return f"{friendly_name} {version}"
                return friendly_name
        
        # Fallback: clean up the technical name
        clean_name = technical_name.replace('/', ' ').replace('-', ' ').replace('_', ' ')
        return ' '.join(word.capitalize() for word in clean_name.split())
    
    def _determine_speed_winner(self, model_responses: Dict[str, ModelResponse]) -> Optional[str]:
        """Determine winner based on speed."""
        # Filter out models with errors
        valid_responses = {k: v for k, v in model_responses.items() if not v.error}
        
        if len(valid_responses) == 0:
            return None
        elif len(valid_responses) == 1:
            return list(valid_responses.keys())[0]
        
        # Find the fastest response
        fastest_model = min(valid_responses.keys(), key=lambda k: valid_responses[k].response_time_ms)
        fastest_time = valid_responses[fastest_model].response_time_ms
        
        # Check if there's a significant difference (>100ms)
        close_competitors = [k for k, v in valid_responses.items() 
                           if abs(v.response_time_ms - fastest_time) < 100]
        
        if len(close_competitors) > 1:
            return None  # Too close to call
        
        return fastest_model
    
    def _determine_quality_winner(self, model_responses: Dict[str, ModelResponse]) -> Optional[str]:
        """Determine winner based on mental health quality metrics."""
        # Filter out models with errors
        valid_responses = {k: v for k, v in model_responses.items() if not v.error}
        
        if len(valid_responses) == 0:
            return None
        elif len(valid_responses) == 1:
            return list(valid_responses.keys())[0]
        
        # Calculate quality scores (weighted)
        def calculate_quality_score(metrics: QualityMetrics) -> float:
            return (
                metrics.empathy_score * 0.25 +          # 25% empathy
                metrics.therapeutic_value * 0.25 +      # 25% therapeutic value
                metrics.safety_score * 0.20 +           # 20% safety
                metrics.clarity_score * 0.15 +          # 15% clarity
                metrics.professional_help_encouragement * 0.10 +  # 10% professional help
                metrics.conciseness_score * 0.05        # 5% conciseness
            )
        
        model_scores = {}
        for model, response in valid_responses.items():
            model_scores[model] = calculate_quality_score(response.quality_metrics)
        
        # Find the best score
        best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
        best_score = model_scores[best_model]
        
        # Check if there's a significant difference (>0.5 points)
        close_competitors = [k for k, v in model_scores.items() 
                           if abs(v - best_score) < 0.5]
        
        if len(close_competitors) > 1:
            return None  # Too close to call
        
        return best_model
    
    def _determine_overall_winner(self, model_responses: Dict[str, ModelResponse],
                                speed_winner: Optional[str], quality_winner: Optional[str]) -> Tuple[Optional[str], float]:
        """Determine overall winner based on configured mode."""
        
        # Filter out models with errors
        valid_responses = {k: v for k, v in model_responses.items() if not v.error}
        
        if len(valid_responses) == 0:
            return None, 0.0
        elif len(valid_responses) == 1:
            return list(valid_responses.keys())[0], 0.9
        
        if self.winner_mode == "speed":
            return speed_winner, 0.8 if speed_winner else 0.5
        
        elif self.winner_mode == "quality" or self.winner_mode == "research":
            return quality_winner, 0.8 if quality_winner else 0.5
        
        elif self.winner_mode == "balanced":
            # Balanced mode: combine speed and quality
            points = {model: 0 for model in valid_responses.keys()}
            
            if speed_winner:
                points[speed_winner] += 1
            
            if quality_winner:
                points[quality_winner] += 2  # Quality weighted more heavily
            
            # Cost consideration (prefer free models)
            for model, response in valid_responses.items():
                if response.cost_usd == 0.0:
                    points[model] += 1
            
            # Find the winner
            if points:
                best_model = max(points.keys(), key=lambda k: points[k])
                max_points = points[best_model]
                
                # Check for ties
                tied_models = [k for k, v in points.items() if v == max_points]
                if len(tied_models) > 1:
                    # Break tie with cost (prefer free models)
                    free_models = [k for k in tied_models if valid_responses[k].cost_usd == 0.0]
                    if free_models:
                        return free_models[0], 0.5
                    else:
                        return tied_models[0], 0.5
                
                return best_model, 0.7
        
        return None, 0.5
    
    def _filter_reasoning_and_length(self, content: str) -> str:
        """Filter reasoning blocks and limit response length if configured."""
        filtered = content
        
        # Remove <think>...</think> blocks if hide_reasoning is enabled
        if self.hide_reasoning:
            pattern = r'<think>.*?</think>'
            filtered = re.sub(pattern, '', filtered, flags=re.DOTALL | re.IGNORECASE)
            filtered = filtered.strip()
        
        # Limit response length if configured
        if self.max_response_length and len(filtered) > self.max_response_length:
            # Try to cut at sentence boundary
            sentences = re.split(r'[.!?]+', filtered[:self.max_response_length])
            if len(sentences) > 1:
                filtered = '.'.join(sentences[:-1]) + '.'
            else:
                filtered = filtered[:self.max_response_length] + '...'
        
        return filtered
    
    def _assess_mental_health_quality(self, content: str, original_content: str) -> QualityMetrics:
        """Assess mental health specific quality metrics."""
        metrics = QualityMetrics()
        
        if not content:
            return metrics
        
        content_lower = content.lower()
        
        # Empathy Score (0-10)
        empathy_indicators = [
            r'\bi understand\b', r'\bi hear you\b', r'\bthat sounds?\b',
            r'\bi can imagine\b', r'\bfeeling\b', r'\bemotion(s|al)?\b',
            r'\bvalidat\w+\b', r'\bempathiz\w+\b', r'\bsupport\w*\b',
            r'\bcare about\b', r'\bconcern\w+\b', r'\bhere for you\b',
            r'\byour experience\b', r'\bmakes sense\b', r'\bnormal to feel\b'
        ]
        
        empathy_score = 0.0
        for pattern in empathy_indicators:
            matches = len(re.findall(pattern, content_lower))
            empathy_score += matches * 0.8
        
        metrics.empathy_score = min(empathy_score, 10.0)
        
        # Therapeutic Value (0-10)
        therapeutic_indicators = [
            r'\bcoping\b', r'\bstrateg(y|ies)\b', r'\btechnique\b',
            r'\bskill(s)?\b', r'\bexercise\b', r'\bpractice\b',
            r'\bmindful\b', r'\bbreath\b', r'\brelax\b',
            r'\bjournal\b', r'\bself-care\b', r'\bbound(ar|ies)\b'
        ]
        
        therapeutic_score = 0.0
        for pattern in therapeutic_indicators:
            matches = len(re.findall(pattern, content_lower))
            therapeutic_score += matches * 1.0
        
        metrics.therapeutic_value = min(therapeutic_score, 10.0)
        
        # Crisis Detection
        crisis_indicators = [
            r'\bharm\w*\s+(myself|yourself)\b', r'\bsuicid\w+\b', r'\bend\w*\s+life\b',
            r'\bkill\w*\s+(myself|yourself)\b', r'\bdie\b', r'\bwant to die\b',
            r'\bno point\b', r'\bcan\'t go on\b', r'\bgive up\b'
        ]
        
        for pattern in crisis_indicators:
            if re.search(pattern, content_lower):
                metrics.crisis_detection = True
                break
        
        # Safety Score (0-10) - higher is safer
        safety_score = 10.0
        harmful_indicators = [
            r'\bjust get over it\b', r'\bsnap out of it\b', r'\bit\'s all in your head\b',
            r'\bstop being dramatic\b', r'\byou\'re overreacting\b', r'\bman up\b',
            r'\bself-medicate\b', r'\bdrink\s+away\b', r'\bsubstance\s+abuse\b'
        ]
        
        for pattern in harmful_indicators:
            matches = len(re.findall(pattern, content_lower))
            safety_score -= matches * 2.0
        
        metrics.safety_score = max(safety_score, 0.0)
        
        # Clarity Score (0-10)
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        clarity_score = 10.0
        if avg_sentence_length > 30:  # Too long sentences
            clarity_score -= 2.0
        elif avg_sentence_length < 5:  # Too short sentences
            clarity_score -= 1.0
        
        if word_count > 300:  # Too verbose
            clarity_score -= 1.0
        
        metrics.clarity_score = max(clarity_score, 0.0)
        
        # Conciseness Score (0-10)
        if word_count < 50:
            metrics.conciseness_score = 10.0
        elif word_count < 100:
            metrics.conciseness_score = 8.0
        elif word_count < 200:
            metrics.conciseness_score = 6.0
        elif word_count < 300:
            metrics.conciseness_score = 4.0
        else:
            metrics.conciseness_score = 2.0
        
        # Professional Help Encouragement (0-10)
        professional_indicators = [
            r'\btherapist\b', r'\bcounselor\b', r'\bpsychologist\b',
            r'\bprofessional help\b', r'\bmental health professional\b',
            r'\bseek help\b', r'\btalk to someone\b', r'\bget support\b'
        ]
        
        professional_score = 0.0
        for pattern in professional_indicators:
            matches = len(re.findall(pattern, content_lower))
            professional_score += matches * 2.0
        
        metrics.professional_help_encouragement = min(professional_score, 10.0)
        
        return metrics
    
    def _filter_and_limit_response(self, text, word_limit=100):
        """Limit response to specified word count for terminal display"""
        if not text:
            return "No response"
        
        words = text.split()
        if len(words) <= word_limit:
            return text
        
        return " ".join(words[:word_limit]) + "..."
    
    def _auto_save_result(self, result: ComparisonResult) -> Optional[str]:
        """Auto-save comparison result to results/ directory"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/comparison_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Multi-Model Comparison Result\n")
                f.write(f"={'='*50}\n")
                f.write(f"Timestamp: {result.timestamp}\n")
                f.write(f"Prompt: {result.prompt}\n")
                f.write(f"Models Compared: {', '.join(result.model_responses.keys())}\n")
                f.write(f"\n")
                
                # Write full responses
                for model, response in result.model_responses.items():
                    friendly_name = self._get_friendly_model_name(self.model_names.get(model, model))
                    f.write(f"\n{friendly_name.upper()} RESPONSE:\n")
                    f.write(f"{'-'*30}\n")
                    
                    if response.error:
                        f.write(f"ERROR: {response.error}\n")
                    else:
                        f.write(f"Response Time: {response.response_time_ms:.0f}ms\n")
                        if response.cost_usd:
                            f.write(f"Cost: ${response.cost_usd:.4f}\n")
                        else:
                            f.write(f"Cost: FREE\n")
                        f.write(f"\nFull Response:\n{response.original_content}\n")
                
                # Write comparison summary
                f.write(f"\nCOMPARISON SUMMARY:\n")
                f.write(f"{'-'*20}\n")
                if result.speed_winner:
                    f.write(f"Speed Winner: {result.speed_winner}\n")
                if result.quality_winner:
                    f.write(f"Quality Winner: {result.quality_winner}\n")
                if result.overall_winner:
                    f.write(f"Overall Winner: {result.overall_winner} ({result.confidence_score:.1%} confidence)\n")
            
            return filename
        except Exception as e:
            if not self.quiet:
                print(f"⚠️  Failed to save result: {e}")
            return None
    
    def display_comparison(self, result: ComparisonResult):
        """Display comparison results in a clean format."""
        if self.quiet:
            self._display_quiet_comparison(result)
        else:
            # Show model responses with terminal summaries
            model_emojis = {
                'openai': '🌐',
                'claude': '🤖', 
                'deepseek': '🏠',
                'gemma': '💎'
            }
            
            print("\n")
            for model, response in result.model_responses.items():
                emoji = model_emojis.get(model, '🤖')
                friendly_name = self._get_friendly_model_name(self.model_names.get(model, model))
                
                if response.error:
                    print(f"{emoji} {friendly_name}: ERROR - {response.error}")
                else:
                    # Show 100-word summary
                    summary = self._filter_and_limit_response(response.content, 100)
                    print(f"{emoji} {friendly_name}: {summary} ...Full response saved to file")
            
            # Show winner
            if result.overall_winner:
                winner_emoji = model_emojis.get(result.overall_winner, '🤖')
                winner_name = self._get_friendly_model_name(self.model_names.get(result.overall_winner, result.overall_winner))
                print(f"\n🏆 Winner: {winner_name} (Best therapeutic response)")
            
            # Show saved file info
            if result.notes:
                print(f"💾 {result.notes}")
    
    def _display_quiet_comparison(self, result: ComparisonResult):
        """Display minimal comparison output."""
        print(f"\n📝 {result.prompt[:60]}{'...' if len(result.prompt) > 60 else ''}")
        
        # Model emojis
        model_emojis = {
            'openai': '🌐',
            'claude': '🤖', 
            'deepseek': '🏠',
            'gemma': '💎'
        }
        
        # Display all model responses
        for model, response in result.model_responses.items():
            emoji = model_emojis.get(model, '🤖')
            friendly_name = self._get_friendly_model_name(self.model_names.get(model, model))
            
            if response.error:
                print(f"{emoji} {friendly_name}: ERROR")
            else:
                content = self._filter_and_limit_response(response.content, 100)
                print(f"{emoji} {friendly_name} ({response.response_time_ms:.0f}ms): {content}")
        
        # Winner summary
        if result.overall_winner:
            winner_emoji = model_emojis.get(result.overall_winner, '🤖')
            winner_name = self._get_friendly_model_name(self.model_names.get(result.overall_winner, result.overall_winner))
            print(f"🏆 Winner: {winner_name} ({result.confidence_score:.1%} confidence)")
    
    
    def save_results(self, results: List[ComparisonResult], filename: str):
        """Save comparison results to JSON file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        data = {
            "comparison_session": {
                "timestamp": datetime.now().isoformat(),
                "selected_models": self.selected_models,
                "model_names": self.model_names,
                "local_endpoint": getattr(self.model_clients.get('deepseek'), 'base_url', None) if 'deepseek' in self.model_clients else None
            },
            "results": []
        }
        
        # Safely convert each result to dict
        for i, result in enumerate(results):
            try:
                result_dict = self._safe_asdict(result)
                data["results"].append(result_dict)
                print(f"✅ Successfully serialized result {i+1}/{len(results)}")
            except Exception as e:
                print(f"❌ Failed to serialize result {i+1}: {e}")
                data["results"].append({
                    "error": f"Serialization failed: {e}",
                    "prompt": getattr(result, 'prompt', 'Unknown'),
                    "timestamp": getattr(result, 'timestamp', datetime.now().isoformat())
                })
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            # Try saving a minimal version
            minimal_data = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "result_count": len(results)
            }
            with open(f"error_{filename}", 'w') as f:
                json.dump(minimal_data, f, indent=2)
            print(f"💾 Error report saved to error_{filename}")
    
    async def close(self):
        """Clean up resources."""
        if 'deepseek' in self.model_clients:
            await self.model_clients['deepseek'].close()
        # Add cleanup for other async clients if needed


async def interactive_mode(comparator: ModelComparator):
    """Run interactive comparison mode."""
    print("\n🎯 Interactive Comparison Mode")
    print("Type your prompts, or 'quit' to exit, 'save <filename>' to save results")
    
    results = []
    
    while True:
        try:
            prompt = input("\n💬 Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if prompt.lower().startswith('save '):
                filename = prompt[5:].strip() or f"comparison_results_{int(time.time())}.json"
                comparator.save_results(results, filename)
                continue
            
            if not prompt:
                continue
            
            # Run comparison
            result = await comparator.compare(prompt)
            comparator.display_comparison(result)
            results.append(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Auto-save results
    if results:
        filename = f"comparison_results_{int(time.time())}.json"
        comparator.save_results(results, filename)


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Multi-model comparison tool for OpenAI, Claude, DeepSeek, and Gemma models",
        epilog="""
Examples:
  %(prog)s "Your prompt here"
  %(prog)s --models openai,claude,deepseek "Compare these models"
  %(prog)s --interactive
  %(prog)s "Hello, how are you?" --save results.json
  %(prog)s --batch prompts.txt --verbose
  %(prog)s --scenarios 3 "Your prompt"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to send to all selected models")
    parser.add_argument("--models", default="openai,claude,deepseek,gemma", 
                       help="Comma-separated list of models to compare (openai,claude,deepseek,gemma) - default: all models")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--save", help="Save results to file")
    parser.add_argument("--batch", help="File with prompts to test (one per line)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output and metrics")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output - just essentials")
    parser.add_argument('--scenarios', type=int, default=3, 
                       help='Number of scenarios to evaluate (default: 3)')
    
    args = parser.parse_args()
    
    # Parse selected models
    selected_models = [model.strip() for model in args.models.split(',')]
    
    try:
        comparator = ModelComparator(
            verbose=args.verbose,
            quiet=args.quiet,
            selected_models=selected_models,
            max_response_length=100 if not args.verbose else None
        )
        
        if args.interactive:
            await interactive_mode(comparator)
        elif args.batch:
            # Batch mode
            with open(args.batch, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            # Limit to specified number of scenarios
            prompts = prompts[:args.scenarios]
            
            results = []
            for i, prompt in enumerate(prompts, 1):
                print(f"\n[{i}/{len(prompts)}] Processing: {prompt[:50]}...")
                result = await comparator.compare(prompt)
                comparator.display_comparison(result)
                results.append(result)
            
            # Save batch results
            filename = args.save or f"batch_comparison_{int(time.time())}.json"
            comparator.save_results(results, filename)
            
        elif args.prompt:
            # Single prompt mode
            result = await comparator.compare(args.prompt)
            comparator.display_comparison(result)
            
            if args.save:
                comparator.save_results([result], args.save)
        else:
            print("❌ Please provide a prompt or use --interactive mode")
            parser.print_help()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        if 'comparator' in locals():
            await comparator.close()
    
    return 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)