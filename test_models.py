import warnings
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")
#!/usr/bin/env python3
"""Test all configured model clients with proper async handling."""

import os
import sys
import asyncio
from typing import Any, Type

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.openai_client import OpenAIClient
from src.models.claude_client import ClaudeClient
from src.models.deepseek_client import DeepSeekClient
from src.models.gemma_client import GemmaClient


async def test_model_async(model_class: Type, model_name: str) -> bool:
    """Test a model that uses async methods."""
    print(f"\n🧪 Testing {model_name}...")
    try:
        client = model_class()
        
        # Try to get a response - handle both sync and async
        test_prompt = "Hello! Please respond with a brief greeting."
        
        # Check if the client has an async generate_response method
        if hasattr(client, 'generate_response'):
            response_method = getattr(client, 'generate_response')
            
            # Check if it's a coroutine function
            if asyncio.iscoroutinefunction(response_method):
                # Async method - await it
                response = await response_method(test_prompt)
            else:
                # Sync method - call directly
                # Try with temperature if supported
                try:
                    response = response_method(test_prompt, temperature=0.7)
                except TypeError:
                    # Try without temperature
                    response = response_method(test_prompt)
        else:
            print(f"❌ {model_name} has no generate_response method")
            return False
        
        # Handle different response types
        if hasattr(response, 'content'):
            response_text = response.content
        elif hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'choices'):
            response_text = response.choices[0].message.content
        elif isinstance(response, dict):
            response_text = response.get('content', response.get('text', str(response)))
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)
        
        print(f"✅ {model_name} working! Response: {response_text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        import traceback
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def test_model_sync(model_class: Type, model_name: str) -> bool:
    """Test a model synchronously (wrapper for async models)."""
    return asyncio.run(test_model_async(model_class, model_name))


async def test_all_models():
    """Test all models asynchronously."""
    models = [
        (OpenAIClient, "OpenAI GPT-4"),
        (ClaudeClient, "Anthropic Claude"),
        (DeepSeekClient, "DeepSeek Local"),
        (GemmaClient, "Gemma Local")
    ]
    
    results = []
    for model_class, model_name in models:
        result = await test_model_async(model_class, model_name)
        results.append(result)
    
    return results


def main():
    """Test all models."""
    print("🤖 Mental Health LLM Model Test")
    print("=" * 50)
    
    # First, check environment variables
    print("\n📋 Checking environment variables...")
    env_vars = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "DEEPSEEK_API_BASE": bool(os.getenv("DEEPSEEK_API_BASE", "http://192.168.86.23:1234")),
        "GEMMA_API_BASE": bool(os.getenv("GEMMA_API_BASE", "http://192.168.86.23:1234"))
    }
    
    for var, is_set in env_vars.items():
        status = "✅ Set" if is_set else "❌ Not set"
        print(f"   {var}: {status}")
    
    # Run tests
    results = asyncio.run(test_all_models())
    
    # Summary
    print(f"\n📊 Summary: {sum(results)}/{len(results)} models working")
    
    if not all(results):
        print("\n💡 Tips for failed models:")
        print("- Local models: Ensure LM Studio/Ollama is running")
        print("- Cloud models: Check API keys in .env file")
        print("- Check API endpoints in model client files")
        print("- Run with --verbose for detailed error traces")
        
        # Specific tips based on what failed
        if not results[0]:  # OpenAI
            print("\n🔧 OpenAI: Make sure OPENAI_API_KEY is set in .env")
        if not results[1]:  # Claude
            print("\n🔧 Claude: Make sure ANTHROPIC_API_KEY is set in .env")
        if not results[2]:  # DeepSeek
            print("\n🔧 DeepSeek: Make sure LM Studio is running on http://192.168.86.23:1234")
        if not results[3]:  # Gemma
            print("\n🔧 Gemma: Make sure LM Studio is running on http://192.168.86.23:1234")


if __name__ == "__main__":
    main()