#!/usr/bin/env python3
"""
Model Comparison CLI Tool
========================

Simple command-line tool to compare responses between OpenAI (cloud) and DeepSeek (local).
Provides side-by-side comparison with performance metrics.

Usage:
    python compare_models.py "Your prompt here"
    python compare_models.py --interactive
    python compare_models.py --prompt "Hello, how are you?" --save results.json
    
Features:
- Side-by-side response comparison
- Response time measurement
- Token usage tracking
- Cost estimation
- Results saving
- Interactive mode

Prerequisites:
- Configure .env file with both OPENAI_API_KEY and LOCAL_LLM settings
- Ensure your local LLM server is running at 192.168.86.30:1234
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
    openai_response: ModelResponse
    local_response: ModelResponse
    speed_winner: Optional[str] = None
    quality_winner: Optional[str] = None
    overall_winner: Optional[str] = None
    winner_mode: str = "speed"  # "speed", "quality", "balanced", "research"
    confidence_score: float = 0.0
    notes: str = ""


class ModelComparator:
    """Handles comparison between OpenAI and local LLM models."""
    
    # OpenAI pricing per 1K tokens (as of 2024)
    OPENAI_PRICING = {
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
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
                 max_response_length: Optional[int] = None, winner_mode: str = "speed"):
        """Initialize the comparator with API clients."""
        load_dotenv()
        
        self.verbose = verbose
        self.quiet = quiet
        self.hide_reasoning = hide_reasoning
        self.max_response_length = max_response_length
        self.winner_mode = winner_mode
        
        # Initialize OpenAI client
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        # Initialize local LLM client
        local_url = os.getenv("LOCAL_LLM_BASE_URL", "http://192.168.86.30:1234/v1")
        local_model = os.getenv("LOCAL_LLM_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b")
        
        self.local_client = LocalLLMClient(
            base_url=local_url,
            model_name=local_model,
            timeout=float(os.getenv("LOCAL_LLM_TIMEOUT", "60"))
        )
        
        # Set up logging based on verbosity
        log_level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        if not quiet:
            print(f"🤖 Model Comparison Tool")
            print(f"   OpenAI: {self._get_friendly_model_name(self.openai_model)}")
            print(f"   Local:  {self._get_friendly_model_name(local_model)}")
            if verbose:
                print(f"   Local URL: {local_url}")
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
            cost = self._calculate_openai_cost(
                self.openai_model,
                usage.prompt_tokens,
                usage.completion_tokens
            )
            
            # Filter and assess content
            original_content = content
            filtered_content = self._filter_reasoning_and_length(content)
            quality_metrics = self._assess_mental_health_quality(filtered_content, original_content)
            
            return ModelResponse(
                model_name=self.openai_model,
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
                model_name=self.openai_model,
                content="",
                original_content="",
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    async def query_local(self, prompt: str, system_prompt: Optional[str] = None) -> ModelResponse:
        """Query local LLM model."""
        start_time = time.time()
        
        try:
            response = await self.local_client.generate_response(
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
                model_name=self.local_client.model_name,
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
                    "endpoint": self.local_client.base_url
                }
            )
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            self.logger.error(f"Local LLM query failed: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            return ModelResponse(
                model_name=self.local_client.model_name,
                content="",
                original_content="",
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    def _calculate_openai_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for OpenAI API usage."""
        if model not in self.OPENAI_PRICING:
            return 0.0
        
        pricing = self.OPENAI_PRICING[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def compare(self, prompt: str, system_prompt: Optional[str] = None) -> ComparisonResult:
        """Compare responses from both models."""
        print(f"\n🔄 Comparing responses for: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"")
        
        # Query both models concurrently
        tasks = [
            self.query_openai(prompt, system_prompt),
            self.query_local(prompt, system_prompt)
        ]
        
        openai_response, local_response = await asyncio.gather(*tasks)
        
        # Determine winners based on different criteria
        speed_winner = self._determine_speed_winner(openai_response, local_response)
        quality_winner = self._determine_quality_winner(openai_response, local_response)
        overall_winner, confidence = self._determine_overall_winner(
            openai_response, local_response, speed_winner, quality_winner
        )
        
        return ComparisonResult(
            prompt=prompt,
            timestamp=datetime.now().isoformat(),
            openai_response=openai_response,
            local_response=local_response,
            speed_winner=speed_winner,
            quality_winner=quality_winner,
            overall_winner=overall_winner,
            winner_mode=self.winner_mode,
            confidence_score=confidence
        )
    
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
    
    def _determine_speed_winner(self, openai_response: ModelResponse, local_response: ModelResponse) -> Optional[str]:
        """Determine winner based on speed."""
        if openai_response.error and not local_response.error:
            return "local"
        elif local_response.error and not openai_response.error:
            return "openai"
        elif openai_response.error and local_response.error:
            return None
        
        # Both succeeded, compare response times
        time_diff = abs(openai_response.response_time_ms - local_response.response_time_ms)
        if time_diff < 100:  # Less than 100ms difference is a tie
            return None
        
        return "openai" if openai_response.response_time_ms < local_response.response_time_ms else "local"
    
    def _determine_quality_winner(self, openai_response: ModelResponse, local_response: ModelResponse) -> Optional[str]:
        """Determine winner based on mental health quality metrics."""
        if openai_response.error and not local_response.error:
            return "local"
        elif local_response.error and not openai_response.error:
            return "openai"
        elif openai_response.error and local_response.error:
            return None
        
        openai_quality = openai_response.quality_metrics
        local_quality = local_response.quality_metrics
        
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
        
        openai_score = calculate_quality_score(openai_quality)
        local_score = calculate_quality_score(local_quality)
        
        # Crisis detection overrides (safety first)
        if openai_quality.crisis_detection and not local_quality.crisis_detection:
            return "local"  # Local didn't detect crisis, safer
        elif local_quality.crisis_detection and not openai_quality.crisis_detection:
            return "openai"  # OpenAI didn't detect crisis, safer
        
        # Score difference must be significant (>0.5 points)
        score_diff = abs(openai_score - local_score)
        if score_diff < 0.5:
            return None  # Too close to call
        
        return "openai" if openai_score > local_score else "local"
    
    def _determine_overall_winner(self, openai_response: ModelResponse, local_response: ModelResponse,
                                speed_winner: Optional[str], quality_winner: Optional[str]) -> Tuple[Optional[str], float]:
        """Determine overall winner based on configured mode."""
        
        if openai_response.error and local_response.error:
            return None, 0.0
        elif openai_response.error:
            return "local", 0.9
        elif local_response.error:
            return "openai", 0.9
        
        if self.winner_mode == "speed":
            return speed_winner, 0.8 if speed_winner else 0.5
        
        elif self.winner_mode == "quality" or self.winner_mode == "research":
            return quality_winner, 0.8 if quality_winner else 0.5
        
        elif self.winner_mode == "balanced":
            # Balanced mode: combine speed and quality
            points = {"openai": 0, "local": 0}
            
            if speed_winner == "openai":
                points["openai"] += 1
            elif speed_winner == "local":
                points["local"] += 1
            
            if quality_winner == "openai":
                points["openai"] += 2  # Quality weighted more heavily
            elif quality_winner == "local":
                points["local"] += 2
            
            # Cost consideration (local always wins on cost)
            if openai_response.cost_usd and openai_response.cost_usd > 0.001:
                points["local"] += 1
            
            if points["openai"] > points["local"]:
                return "openai", 0.7
            elif points["local"] > points["openai"]:
                return "local", 0.7
            else:
                return "local", 0.5  # Default to local on tie due to cost
        
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
    
    def display_comparison(self, result: ComparisonResult):
        """Display comparison results in a clean format."""
        if self.quiet:
            self._display_quiet_comparison(result)
        else:
            self._display_full_comparison(result)
    
    def _display_quiet_comparison(self, result: ComparisonResult):
        """Display minimal comparison output."""
        print(f"\n📝 {result.prompt[:60]}{'...' if len(result.prompt) > 60 else ''}")
        
        # OpenAI Response
        if result.openai_response.error:
            print(f"🌐 {self._get_friendly_model_name(self.openai_model)}: ERROR")
        else:
            content = result.openai_response.content[:100] + ('...' if len(result.openai_response.content) > 100 else '')
            print(f"🌐 {self._get_friendly_model_name(self.openai_model)} ({result.openai_response.response_time_ms:.0f}ms): {content}")
        
        # Local Response
        if result.local_response.error:
            print(f"🏠 {self._get_friendly_model_name(self.local_client.model_name)}: ERROR")
        else:
            content = result.local_response.content[:100] + ('...' if len(result.local_response.content) > 100 else '')
            print(f"🏠 {self._get_friendly_model_name(self.local_client.model_name)} ({result.local_response.response_time_ms:.0f}ms): {content}")
        
        # Winner summary
        if result.speed_winner:
            speed_name = self._get_friendly_model_name(self.openai_model if result.speed_winner == 'openai' else self.local_client.model_name)
            print(f"⚡ Faster: {speed_name}")
        
        if result.quality_winner:
            quality_name = self._get_friendly_model_name(self.openai_model if result.quality_winner == 'openai' else self.local_client.model_name)
            print(f"🎯 Better Quality: {quality_name}")
        
        if result.overall_winner:
            overall_name = self._get_friendly_model_name(self.openai_model if result.overall_winner == 'openai' else self.local_client.model_name)
            print(f"🏆 Winner ({result.winner_mode}): {overall_name} ({result.confidence_score:.1%} confidence)")
    
    def _display_full_comparison(self, result: ComparisonResult):
        """Display detailed comparison output."""
        print("\n" + "="*80)
        print(f"📝 PROMPT: {result.prompt}")
        if self.verbose:
            print(f"⏰ TIME: {result.timestamp}")
        print("="*80)
        
        # OpenAI Response
        print(f"\n🌐 {self._get_friendly_model_name(self.openai_model).upper()} RESPONSE:")
        print("-" * 50)
        if result.openai_response.error:
            print(f"❌ ERROR: {result.openai_response.error}")
        else:
            print(f"\n{result.openai_response.content}\n")
            print(f"⏱️  Response Time: {result.openai_response.response_time_ms:.0f}ms")
            if result.openai_response.total_tokens and self.verbose:
                print(f"🔢 Tokens: {result.openai_response.total_tokens} (in: {result.openai_response.input_tokens}, out: {result.openai_response.output_tokens})")
            if result.openai_response.cost_usd:
                print(f"💰 Cost: ${result.openai_response.cost_usd:.4f}")
            
            # Quality metrics for OpenAI if verbose
            if self.verbose and result.openai_response.quality_metrics:
                metrics = result.openai_response.quality_metrics
                print(f"🏥 Quality Metrics:")
                print(f"   Empathy: {metrics.empathy_score:.1f}/10")
                print(f"   Therapeutic: {metrics.therapeutic_value:.1f}/10")
                print(f"   Safety: {metrics.safety_score:.1f}/10")
                if metrics.crisis_detection:
                    print(f"   ⚠️  Crisis indicators detected")
        
        # Local Response
        local_name = self._get_friendly_model_name(self.local_client.model_name)
        print(f"\n🏠 {local_name.upper()} RESPONSE:")
        print("-" * 50)
        if result.local_response.error:
            print(f"❌ ERROR: {result.local_response.error}")
        else:
            print(f"\n{result.local_response.content}\n")
            print(f"⏱️  Response Time: {result.local_response.response_time_ms:.0f}ms")
            if result.local_response.total_tokens and self.verbose:
                print(f"🔢 Tokens: {result.local_response.total_tokens} (in: {result.local_response.input_tokens}, out: {result.local_response.output_tokens})")
            print(f"💰 Cost: FREE")
            
            # Show quality metrics if verbose
            if self.verbose and result.local_response.quality_metrics:
                metrics = result.local_response.quality_metrics
                print(f"🏥 Quality Metrics:")
                print(f"   Empathy: {metrics.empathy_score:.1f}/10")
                print(f"   Therapeutic: {metrics.therapeutic_value:.1f}/10")
                print(f"   Safety: {metrics.safety_score:.1f}/10")
                if metrics.crisis_detection:
                    print(f"   ⚠️  Crisis indicators detected")
            
            # Show reasoning if verbose and not filtered
            if self.verbose and not self.hide_reasoning and '<think>' in result.local_response.original_content.lower():
                print(f"\n💭 Internal Reasoning Available (use --hide-reasoning to hide)")
        
        # Comparison Summary
        if not self.quiet:
            print(f"\n{'='*80}")
            print("📊 COMPARISON SUMMARY")
            print(f"{'='*80}")
            
            # Winner summary
            if result.speed_winner:
                speed_name = self._get_friendly_model_name(self.openai_model if result.speed_winner == 'openai' else self.local_client.model_name)
                print(f"⚡ Faster Model: {speed_name}")
            
            if result.quality_winner:
                quality_name = self._get_friendly_model_name(self.openai_model if result.quality_winner == 'openai' else self.local_client.model_name)
                print(f"🎯 Better Quality: {quality_name}")
            
            if result.overall_winner:
                overall_name = self._get_friendly_model_name(self.openai_model if result.overall_winner == 'openai' else self.local_client.model_name)
                print(f"🏆 Overall Winner ({result.winner_mode}): {overall_name} ({result.confidence_score:.1%} confidence)")
            
            if not result.openai_response.error and not result.local_response.error:
                time_diff = abs(result.openai_response.response_time_ms - result.local_response.response_time_ms)
                faster_model = self._get_friendly_model_name(self.openai_model) if result.openai_response.response_time_ms < result.local_response.response_time_ms else local_name
                print(f"⚡ Speed Difference: {time_diff:.0f}ms")
                
                if result.openai_response.cost_usd:
                    print(f"💵 Cost Savings: ${result.openai_response.cost_usd:.4f} (using local model)")
            
            print("="*80)
    
    def save_results(self, results: List[ComparisonResult], filename: str):
        """Save comparison results to JSON file."""
        data = {
            "comparison_session": {
                "timestamp": datetime.now().isoformat(),
                "openai_model": self.openai_model,
                "local_model": self.local_client.model_name,
                "local_endpoint": self.local_client.base_url
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
        await self.local_client.close()


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
            
            # Optional system prompt
            system_prompt = input("🔧 System prompt (optional): ").strip() or None
            
            # Run comparison
            result = await comparator.compare(prompt, system_prompt)
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
    parser = argparse.ArgumentParser(description="Compare OpenAI and Local LLM responses")
    parser.add_argument("prompt", nargs="?", help="Prompt to send to both models")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--system-prompt", "-s", help="System prompt to use")
    parser.add_argument("--save", help="Save results to file")
    parser.add_argument("--batch", help="File with prompts to test (one per line)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output with debug info")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output - just essentials")
    parser.add_argument("--hide-reasoning", action="store_true", help="Hide DeepSeek <think> reasoning blocks")
    parser.add_argument("--max-length", type=int, help="Maximum response length (characters)")
    parser.add_argument("--judge-quality", action="store_const", const="quality", dest="winner_mode", 
                       help="Use quality metrics instead of speed for winner determination")
    parser.add_argument("--balanced", action="store_const", const="balanced", dest="winner_mode",
                       help="Combine speed and quality for winner determination")
    parser.add_argument("--research-mode", action="store_const", const="research", dest="winner_mode",
                       help="Focus on therapeutic effectiveness for mental health research")
    parser.set_defaults(winner_mode="speed")
    
    args = parser.parse_args()
    
    try:
        comparator = ModelComparator(
            verbose=args.verbose,
            quiet=args.quiet,
            hide_reasoning=args.hide_reasoning,
            max_response_length=args.max_length,
            winner_mode=args.winner_mode
        )
        
        if args.interactive:
            await interactive_mode(comparator)
        elif args.batch:
            # Batch mode
            with open(args.batch, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            results = []
            for i, prompt in enumerate(prompts, 1):
                print(f"\n[{i}/{len(prompts)}] Processing: {prompt[:50]}...")
                result = await comparator.compare(prompt, args.system_prompt)
                comparator.display_comparison(result)
                results.append(result)
            
            # Save batch results
            filename = args.save or f"batch_comparison_{int(time.time())}.json"
            comparator.save_results(results, filename)
            
        elif args.prompt:
            # Single prompt mode
            result = await comparator.compare(args.prompt, args.system_prompt)
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