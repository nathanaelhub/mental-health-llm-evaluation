#!/usr/bin/env python3
"""
Simple Multi-Model Prompt Comparison Tool
========================================

Lightweight tool for comparing AI model responses to single prompts.
Provides quick comparison with simple scoring and auto-save functionality.

Usage:
    python compare_models.py "How can I manage anxiety?"
    python compare_models.py --interactive
    python compare_models.py "Hello" --models openai,claude
    python compare_models.py --batch prompts.txt

Features:
- Direct prompt → model → response → score comparison
- Multi-model support (OpenAI, Claude, DeepSeek, Gemma)
- Simple scoring (empathy, helpfulness, safety, clarity)
- Clean terminal output with auto-save
- No complex research frameworks
"""

import asyncio
import argparse
import json
import time
import os
import sys
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

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

def get_display_width(text: str) -> int:
    """
    Calculate the actual display width of text including Unicode characters.
    Emojis and wide characters count as 2, normal characters as 1.
    """
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            width += 2  # Full-width or Wide characters
        elif unicodedata.category(char).startswith('So'):
            width += 2  # Symbols (including emojis)
        else:
            width += 1  # Normal characters
    return width

def pad_to_display_width(text: str, target_width: int, align: str = 'left') -> str:
    """
    Pad text to exact display width, accounting for Unicode character widths.
    """
    current_width = get_display_width(text)
    padding_needed = target_width - current_width
    
    if padding_needed <= 0:
        return text[:target_width] if len(text) > target_width else text
    
    if align == 'left':
        return text + ' ' * padding_needed
    elif align == 'right':
        return ' ' * padding_needed + text
    else:  # center
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        return ' ' * left_pad + text + ' ' * right_pad

@dataclass
class SimpleScore:
    """Simple scoring metrics for mental health responses."""
    empathy: float = 0.0      # 0-10: Shows understanding and compassion
    helpfulness: float = 0.0  # 0-10: Provides practical advice
    safety: float = 0.0       # 0-10: Appropriate boundaries and safety
    clarity: float = 0.0      # 0-10: Easy to understand
    overall: float = 0.0      # Average of above


@dataclass
class ModelResponse:
    """Response data from a model."""
    model_name: str
    content: str
    full_content: str  # Complete original response
    response_time_ms: float
    score: SimpleScore
    error: Optional[str] = None
    tokens: Optional[int] = None
    cost_usd: Optional[float] = None


@dataclass
class ComparisonResult:
    """Simple comparison result."""
    prompt: str
    timestamp: str
    responses: Dict[str, ModelResponse]
    winner: Optional[str] = None
    winner_reason: str = ""
    saved_file: str = ""


class SimpleModelComparator:
    """Simple prompt comparison tool."""
    
    # Model pricing per 1K tokens (simplified)
    MODEL_COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
        "deepseek": {"input": 0.0, "output": 0.0},  # Local = free
    }
    
    def __init__(self, selected_models: Optional[List[str]] = None):
        """Initialize with selected models."""
        load_dotenv()
        
        self.selected_models = selected_models or ['openai', 'claude', 'deepseek', 'gemma']
        self.model_clients = {}
        self.model_names = {}
        
        # Initialize available model clients
        self._init_clients()
        
        # Ensure we have at least one model
        if not self.model_clients:
            raise ValueError("No valid models available. Check your API keys and configuration.")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
    
    def _init_clients(self):
        """Initialize model clients."""
        # OpenAI
        if 'openai' in self.selected_models:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.model_clients['openai'] = openai.OpenAI(api_key=openai_key)
                self.model_names['openai'] = os.getenv("OPENAI_MODEL", "gpt-4")
            else:
                print("⚠️  OPENAI_API_KEY not found, skipping OpenAI")
        
        # Claude
        if 'claude' in self.selected_models and HAS_CLAUDE:
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if claude_key:
                self.model_clients['claude'] = ClaudeClient()
                self.model_names['claude'] = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
            else:
                print("⚠️  ANTHROPIC_API_KEY not found, skipping Claude")
        
        # DeepSeek (Local)
        if 'deepseek' in self.selected_models:
            local_url = os.getenv("LOCAL_LLM_BASE_URL", "http://192.168.86.30:1234/v1")
            local_model = os.getenv("LOCAL_LLM_MODEL", "deepseek-r1")
            
            self.model_clients['deepseek'] = LocalLLMClient(
                base_url=local_url,
                model_name=local_model,
                timeout=float(os.getenv("LOCAL_LLM_TIMEOUT", "60"))
            )
            self.model_names['deepseek'] = local_model
        
        # Gemma
        if 'gemma' in self.selected_models and HAS_GEMMA:
            self.model_clients['gemma'] = GemmaClient()
            self.model_names['gemma'] = os.getenv("GEMMA_MODEL", "gemini-pro")
    
    async def query_model(self, model_key: str, prompt: str) -> ModelResponse:
        """Query a specific model with the prompt."""
        start_time = time.time()
        
        try:
            if model_key == 'openai':
                response = self.model_clients['openai'].chat.completions.create(
                    model=self.model_names['openai'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                end_time = time.time()
                content = response.choices[0].message.content
                tokens = response.usage.total_tokens
                cost = self._calculate_cost("gpt-4", response.usage.prompt_tokens, response.usage.completion_tokens)
                
            elif model_key == 'claude':
                response = await self.model_clients['claude'].generate_response(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                end_time = time.time()
                content = response.content
                tokens = response.token_count or 0
                cost = self._calculate_cost("claude-3-sonnet", response.input_token_count or 0, response.output_token_count or 0)
                
            elif model_key == 'deepseek':
                response = await self.model_clients['deepseek'].generate_response(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                end_time = time.time()
                content = response.content
                tokens = response.token_count or 0
                cost = 0.0  # Local is free
                
            elif model_key == 'gemma':
                response = await self.model_clients['gemma'].generate_response(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                end_time = time.time()
                content = response.content
                tokens = response.token_count or 0
                cost = self._calculate_cost("gemini-pro", response.input_token_count or 0, response.output_token_count or 0)
            
            else:
                raise ValueError(f"Unknown model: {model_key}")
            
            response_time_ms = (end_time - start_time) * 1000
            
            # Create summary (100 words) and score the response
            summary = self._create_summary(content)
            score = self._score_response(content)
            
            return ModelResponse(
                model_name=self._get_display_name(model_key),
                content=summary,
                full_content=content,
                response_time_ms=response_time_ms,
                score=score,
                tokens=tokens,
                cost_usd=cost
            )
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return ModelResponse(
                model_name=self._get_display_name(model_key),
                content="",
                full_content="",
                response_time_ms=response_time_ms,
                score=SimpleScore(),
                error=str(e)
            )
    
    def _calculate_cost(self, model_type: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost."""
        if model_type not in self.MODEL_COSTS:
            return 0.0
        
        pricing = self.MODEL_COSTS[model_type]
        return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
    
    def _get_display_name(self, model_key: str) -> str:
        """Get friendly display name for model."""
        names = {
            'openai': 'GPT-4',
            'claude': 'Claude',
            'deepseek': 'DeepSeek',
            'gemma': 'Gemma'
        }
        return names.get(model_key, model_key.title())
    
    def _create_summary(self, content: str, word_limit: int = 100) -> str:
        """Create 100-word summary of response."""
        if not content:
            return "No response"
        
        words = content.split()
        if len(words) <= word_limit:
            return content
        
        return " ".join(words[:word_limit]) + "..."
    
    def _score_response(self, content: str) -> SimpleScore:
        """Score response on simple criteria (0-10 each)."""
        if not content:
            return SimpleScore()
        
        content_lower = content.lower()
        
        # Empathy Score (0-10)
        empathy_words = [
            'understand', 'hear you', 'sounds', 'imagine', 'feeling', 'emotion',
            'validate', 'empathize', 'support', 'care', 'concern', 'here for you',
            'experience', 'makes sense', 'normal to feel'
        ]
        empathy_score = min(sum(1.5 for word in empathy_words if word in content_lower), 10.0)
        
        # Helpfulness Score (0-10)
        helpful_words = [
            'try', 'can help', 'strategy', 'technique', 'practice', 'exercise',
            'skill', 'cope', 'manage', 'steps', 'approach', 'method', 'solution'
        ]
        helpfulness_score = min(sum(1.5 for word in helpful_words if word in content_lower), 10.0)
        
        # Safety Score (0-10) - starts at 10, deduct for unsafe content
        safety_score = 10.0
        unsafe_phrases = [
            'just get over it', 'snap out of it', 'all in your head',
            'stop being dramatic', 'overreacting', 'man up'
        ]
        safety_score -= sum(3.0 for phrase in unsafe_phrases if phrase in content_lower)
        safety_score = max(safety_score, 0.0)
        
        # Clarity Score (0-10) - based on readability
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        clarity_score = 10.0
        if avg_sentence_length > 25:  # Too complex
            clarity_score -= 2.0
        elif avg_sentence_length < 5:  # Too choppy
            clarity_score -= 1.0
        
        if word_count > 200:  # Too verbose
            clarity_score -= 1.0
        
        clarity_score = max(clarity_score, 0.0)
        
        # Overall score (average)
        overall_score = (empathy_score + helpfulness_score + safety_score + clarity_score) / 4
        
        return SimpleScore(
            empathy=empathy_score,
            helpfulness=helpfulness_score,
            safety=safety_score,
            clarity=clarity_score,
            overall=overall_score
        )
    
    async def compare(self, prompt: str) -> ComparisonResult:
        """Compare responses from all available models."""
        print(f"\n🤖 Multi-Model Comparison: \"{prompt}\"")
        
        # Query all models concurrently
        tasks = []
        for model_key in self.model_clients.keys():
            tasks.append(self.query_model(model_key, prompt))
        
        responses = await asyncio.gather(*tasks)
        
        # Create responses dict
        model_responses = {}
        for model_key, response in zip(self.model_clients.keys(), responses):
            model_responses[model_key] = response
        
        # Determine winner
        winner, winner_reason = self._determine_winner(model_responses)
        
        # Auto-save full responses
        saved_file = self._save_full_responses(prompt, model_responses)
        
        return ComparisonResult(
            prompt=prompt,
            timestamp=datetime.now().isoformat(),
            responses=model_responses,
            winner=winner,
            winner_reason=winner_reason,
            saved_file=saved_file
        )
    
    def _determine_winner(self, responses: Dict[str, ModelResponse]) -> tuple[Optional[str], str]:
        """Determine winner based on overall score."""
        valid_responses = {k: v for k, v in responses.items() if not v.error}
        
        if not valid_responses:
            return None, "No valid responses"
        
        if len(valid_responses) == 1:
            model = list(valid_responses.keys())[0]
            return model, "Only working model"
        
        # Find highest scoring model
        best_model = max(valid_responses.keys(), key=lambda k: valid_responses[k].score.overall)
        best_score = valid_responses[best_model].score.overall
        
        # Check for ties (within 0.5 points)
        tied_models = [k for k, v in valid_responses.items() if abs(v.score.overall - best_score) < 0.5]
        
        if len(tied_models) > 1:
            return best_model, f"Close call (tied with {len(tied_models)-1} others)"
        else:
            return best_model, f"Best overall therapeutic response"
    
    def _save_full_responses(self, prompt: str, responses: Dict[str, ModelResponse]) -> str:
        """Save full responses to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/comparison_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Multi-Model Comparison\n")
                f.write(f"{'='*50}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                for model_key, response in responses.items():
                    f.write(f"{response.model_name.upper()} RESPONSE:\n")
                    f.write(f"{'-'*30}\n")
                    
                    if response.error:
                        f.write(f"ERROR: {response.error}\n")
                    else:
                        f.write(f"Response Time: {response.response_time_ms:.0f}ms\n")
                        f.write(f"Tokens: {response.tokens or 'N/A'}\n")
                        f.write(f"Cost: ${response.cost_usd:.4f}\n" if response.cost_usd else "Cost: FREE\n")
                        f.write(f"Score: {response.score.overall:.1f}/10\n")
                        f.write(f"\nFull Response:\n{response.full_content}\n")
                    
                    f.write(f"\n")
            
            return filename
        except Exception as e:
            print(f"⚠️  Failed to save responses: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def display_results(self, result: ComparisonResult):
        """Display comparison results with fancy box formatting."""
        model_emojis = {
            'openai': '🌐',
            'claude': '🤖',
            'deepseek': '🏠',
            'gemma': '💎'
        }
        
        # Calculate box width based on content with proper Unicode handling
        # Target: ║ model_col (18 display chars) │ content_col (50 display chars) ║ = 72 total
        model_col_width = 18  # Display width for model column
        content_col_width = 50  # Display width for content column
        box_width = model_col_width + content_col_width + 4  # +4 for ║ │ ║ and space
        inner_width = box_width - 2  # box_width - 2 for the ║ characters
        
        # Print main header with dynamic width
        header_line = "╔" + "═" * inner_width + "╗"
        separator_line = "╠" + "═" * inner_width + "╣"
        footer_line = "╚" + "═" * inner_width + "╝"
        
        print(f"\n{header_line}")
        title_padded = pad_to_display_width("Model Comparison Results", inner_width, align='center')
        print(f"║{title_padded}║")
        print(separator_line)
        
        # Sort responses by score (highest first)
        sorted_responses = sorted(
            result.responses.items(), 
            key=lambda x: x[1].score.overall if not x[1].error else -1, 
            reverse=True
        )
        
        for i, (model_key, response) in enumerate(sorted_responses):
            emoji = model_emojis.get(model_key, '🤖')
            
            if response.error:
                # Error display - use proper Unicode width formatting
                model_display = f"{emoji} {response.model_name}"
                error_msg = f"ERROR: {response.error[:35]}"  # Shorter to fit width
                model_padded = pad_to_display_width(model_display, model_col_width)
                error_padded = pad_to_display_width(error_msg, content_col_width)
                print(f"║ {model_padded} │ {error_padded} ║")
            else:
                # Model name and score line - use proper Unicode width formatting
                model_display = f"{emoji} {response.model_name}"
                score_display = f"Score: {response.score.overall:.1f}/10"
                model_padded = pad_to_display_width(model_display, model_col_width)
                score_padded = pad_to_display_width(score_display, content_col_width)
                print(f"║ {model_padded} │ {score_padded} ║")
                
                # Detailed scores line - align with pipe character
                empathy = f"E:{response.score.empathy:.1f}"
                helpful = f"H:{response.score.helpfulness:.1f}"
                safe = f"S:{response.score.safety:.1f}"
                clear = f"C:{response.score.clarity:.1f}"
                details_display = f"{empathy} | {helpful} | {safe} | {clear}"
                empty_model = pad_to_display_width("", model_col_width)
                details_padded = pad_to_display_width(details_display, content_col_width)
                print(f"║ {empty_model} │ {details_padded} ║")
            
            # Add separator between models (except for last one)
            if i < len(sorted_responses) - 1:
                print(separator_line)
        
        # Winner section
        if result.winner:
            print(separator_line)
            winner_emoji = model_emojis.get(result.winner, '🤖')
            winner_name = result.responses[result.winner].model_name
            winner_text = f"🏆 Winner: {winner_name} ({result.winner_reason})"
            winner_padded = pad_to_display_width(winner_text, inner_width, align='center')
            print(f"║ {winner_padded} ║")
        
        # Footer
        print(footer_line)
        
        # Show saved file outside the box
        if result.saved_file and os.path.exists(result.saved_file):
            print(f"\n💾 Full responses saved to: {result.saved_file}")
        elif result.saved_file:
            print(f"\n⚠️  File saving reported but file not found: {result.saved_file}")
    
    async def close(self):
        """Clean up resources."""
        if 'deepseek' in self.model_clients:
            await self.model_clients['deepseek'].close()


async def interactive_mode(comparator: SimpleModelComparator):
    """Interactive prompt comparison mode."""
    print("\n🎯 Interactive Mode - Enter prompts to compare models")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            prompt = input("💬 Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            result = await comparator.compare(prompt)
            comparator.display_results(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def batch_mode(comparator: SimpleModelComparator, filename: str):
    """Process prompts from file."""
    try:
        with open(filename, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"\n📁 Processing {len(prompts)} prompts from {filename}")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Processing: {prompt[:50]}...")
            result = await comparator.compare(prompt)
            comparator.display_results(result)
            
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
    except Exception as e:
        print(f"❌ Error processing batch: {e}")


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Simple Multi-Model Prompt Comparison Tool",
        epilog="""
Examples:
  %(prog)s "How can I manage anxiety?"              # Default: openai,deepseek
  %(prog)s --interactive                            # Interactive mode
  %(prog)s "Hello" --models openai,claude          # Specific models
  %(prog)s "Hello" --all-models                    # All 4 models
  %(prog)s --batch prompts.txt --all-models        # Batch mode, all models
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("prompt", nargs="?", help="Prompt to compare across models")
    parser.add_argument("--models", help="Comma-separated models: openai,claude,deepseek,gemma (default: openai,deepseek)")
    parser.add_argument("--all-models", action="store_true", help="Use all available models (openai,claude,deepseek,gemma)")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive mode")
    parser.add_argument("--batch", help="File with prompts (one per line)")
    
    args = parser.parse_args()
    
    # Parse selected models
    if args.all_models:
        selected_models = ['openai', 'claude', 'deepseek', 'gemma']
    elif args.models:
        if args.models.lower() == 'all':
            selected_models = ['openai', 'claude', 'deepseek', 'gemma']
        else:
            selected_models = [model.strip() for model in args.models.split(',')]
    else:
        # Default models when no selection is made
        selected_models = ['openai', 'deepseek']
    
    try:
        comparator = SimpleModelComparator(selected_models=selected_models)
        
        if args.interactive:
            await interactive_mode(comparator)
        elif args.batch:
            await batch_mode(comparator, args.batch)
        elif args.prompt:
            result = await comparator.compare(args.prompt)
            comparator.display_results(result)
        else:
            print("❌ Please provide a prompt, use --interactive, or --batch mode")
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        if 'comparator' in locals():
            await comparator.close()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)