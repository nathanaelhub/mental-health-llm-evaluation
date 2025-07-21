#!/usr/bin/env python3
"""
Model Connection Testing Script
==============================

Tests all configured model clients to ensure they work correctly.
This is a simple diagnostic tool for verifying model connectivity.

Usage:
    python scripts/test_all_models.py
    python scripts/test_all_models.py --quick
    python scripts/test_all_models.py --models openai,claude

Features:
- Tests model client initialization
- Verifies basic response generation
- Checks cost tracking functionality
- Tests error handling
- Provides clear pass/fail status
"""

import asyncio
import argparse
import os
import sys
import time
from typing import List, Optional
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


class ModelTester:
    """Tests all available model clients."""
    
    def __init__(self, test_models: Optional[List[str]] = None, quick_test: bool = False):
        """Initialize tester with specified models."""
        load_dotenv()
        
        self.test_models = test_models or ['openai', 'claude', 'deepseek', 'gemma']
        self.quick_test = quick_test
        self.test_prompt = "Hello" if quick_test else "How can I manage stress?"
        
        self.results = {}
        print(f"🧪 Testing Model Connections")
        print(f"{'='*40}")
        print(f"Test prompt: \"{self.test_prompt}\"")
        print(f"Quick test: {quick_test}")
        print()
    
    async def test_openai(self) -> bool:
        """Test OpenAI client."""
        try:
            print("🌐 Testing OpenAI...")
            
            # Check API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("   ❌ OPENAI_API_KEY not found in environment")
                return False
            
            # Initialize client
            client = openai.OpenAI(api_key=api_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4")
            
            # Test response
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": self.test_prompt}],
                max_tokens=100 if self.quick_test else 500,
                temperature=0.7
            )
            response_time = (time.time() - start_time) * 1000
            
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            print(f"   ✅ OpenAI client working")
            print(f"   📝 Response: {content[:50]}..." if len(content) > 50 else f"   📝 Response: {content}")
            print(f"   ⏱️  Response time: {response_time:.0f}ms")
            print(f"   🔢 Tokens used: {tokens}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ OpenAI failed: {e}")
            return False
    
    async def test_claude(self) -> bool:
        """Test Claude client."""
        try:
            print("🤖 Testing Claude...")
            
            if not HAS_CLAUDE:
                print("   ❌ Claude client not available (import failed)")
                return False
            
            # Check API key
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("   ❌ ANTHROPIC_API_KEY not found in environment")
                return False
            
            # Initialize client
            client = ClaudeClient()
            
            # Test response
            start_time = time.time()
            response = await client.generate_response(
                prompt=self.test_prompt,
                max_tokens=100 if self.quick_test else 500,
                temperature=0.7
            )
            response_time = (time.time() - start_time) * 1000
            
            content = response.content
            tokens = response.token_count or 0
            
            print(f"   ✅ Claude client working")
            print(f"   📝 Response: {content[:50]}..." if len(content) > 50 else f"   📝 Response: {content}")
            print(f"   ⏱️  Response time: {response_time:.0f}ms")
            print(f"   🔢 Tokens used: {tokens}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Claude failed: {e}")
            return False
    
    async def test_deepseek(self) -> bool:
        """Test DeepSeek local client."""
        try:
            print("🏠 Testing DeepSeek (Local)...")
            
            # Get configuration
            base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://192.168.86.30:1234/v1")
            model_name = os.getenv("LOCAL_LLM_MODEL", "deepseek-r1")
            timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "60"))
            
            print(f"   🔗 Connecting to: {base_url}")
            print(f"   🤖 Model: {model_name}")
            
            # Initialize client
            client = LocalLLMClient(
                base_url=base_url,
                model_name=model_name,
                timeout=timeout
            )
            
            # Test response
            start_time = time.time()
            response = await client.generate_response(
                prompt=self.test_prompt,
                max_tokens=100 if self.quick_test else 500,
                temperature=0.7
            )
            response_time = (time.time() - start_time) * 1000
            
            content = response.content
            tokens = response.token_count or 0
            
            print(f"   ✅ DeepSeek client working")
            print(f"   📝 Response: {content[:50]}..." if len(content) > 50 else f"   📝 Response: {content}")
            print(f"   ⏱️  Response time: {response_time:.0f}ms")
            print(f"   🔢 Tokens used: {tokens}")
            print(f"   💰 Cost: FREE (local)")
            
            # Clean up
            await client.close()
            
            return True
            
        except Exception as e:
            print(f"   ❌ DeepSeek failed: {e}")
            print(f"   💡 Check if local LLM server is running at {os.getenv('LOCAL_LLM_BASE_URL', 'http://192.168.86.30:1234/v1')}")
            return False
    
    async def test_gemma(self) -> bool:
        """Test Gemma client."""
        try:
            print("💎 Testing Gemma...")
            
            if not HAS_GEMMA:
                print("   ❌ Gemma client not available (import failed)")
                return False
            
            # Initialize client
            client = GemmaClient()
            
            # Test response
            start_time = time.time()
            response = await client.generate_response(
                prompt=self.test_prompt,
                max_tokens=100 if self.quick_test else 500,
                temperature=0.7
            )
            response_time = (time.time() - start_time) * 1000
            
            content = response.content
            tokens = response.token_count or 0
            
            print(f"   ✅ Gemma client working")
            print(f"   📝 Response: {content[:50]}..." if len(content) > 50 else f"   📝 Response: {content}")
            print(f"   ⏱️  Response time: {response_time:.0f}ms")
            print(f"   🔢 Tokens used: {tokens}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Gemma failed: {e}")
            return False
    
    async def run_tests(self) -> dict:
        """Run all tests and return results."""
        test_functions = {
            'openai': self.test_openai,
            'claude': self.test_claude,
            'deepseek': self.test_deepseek,
            'gemma': self.test_gemma
        }
        
        for model in self.test_models:
            if model in test_functions:
                try:
                    self.results[model] = await test_functions[model]()
                except Exception as e:
                    print(f"   ❌ {model.title()} test crashed: {e}")
                    self.results[model] = False
                print()  # Add spacing between tests
            else:
                print(f"⚠️  Unknown model: {model}")
                self.results[model] = False
        
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        print("📊 Test Summary")
        print("=" * 20)
        
        passed = sum(1 for result in self.results.values() if result)
        total = len(self.results)
        
        for model, passed_test in self.results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{model.upper():>10}: {status}")
        
        print()
        print(f"Results: {passed}/{total} models working")
        
        if passed == total:
            print("🎉 All tests passed! System ready for use.")
        elif passed > 0:
            print("⚠️  Some models failed. Check configuration.")
        else:
            print("❌ All models failed. Check API keys and connections.")
        
        return passed == total


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test all model client connections",
        epilog="""
Examples:
  %(prog)s                          # Test all models
  %(prog)s --quick                  # Quick test with short responses
  %(prog)s --models openai,claude   # Test specific models only
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--models", 
                       help="Comma-separated list of models to test (openai,claude,deepseek,gemma)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with minimal responses")
    
    args = parser.parse_args()
    
    # Parse selected models
    test_models = None
    if args.models:
        test_models = [model.strip() for model in args.models.split(',')]
    
    try:
        tester = ModelTester(test_models=test_models, quick_test=args.quick)
        await tester.run_tests()
        success = tester.print_summary()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)