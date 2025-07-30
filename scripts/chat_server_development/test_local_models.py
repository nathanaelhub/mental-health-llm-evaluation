#!/usr/bin/env python3
"""
Test Local Model Connectivity
Verify local model servers and API credentials before starting the main chat server
"""

import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

async def test_local_model(name: str, base_url: str) -> bool:
    """Test if local model server is reachable"""
    print(f"\n{'='*50}")
    print(f"Testing {name} at {base_url}")
    print("-" * 50)
    
    # Test 1: Basic connectivity - check /models endpoint
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{base_url}/models") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ {name} server is reachable (status: {response.status})")
                    
                    # Display available models
                    if 'data' in data and isinstance(data['data'], list):
                        models = data['data']
                        print(f"   📋 Available models ({len(models)}):")
                        for model in models[:3]:  # Show first 3
                            model_id = model.get('id', 'unknown')
                            print(f"      • {model_id}")
                        if len(models) > 3:
                            print(f"      • ... and {len(models) - 3} more")
                    else:
                        print(f"   📋 Response data: {data}")
                        
                else:
                    print(f"❌ {name} server returned HTTP {response.status}")
                    return False
                    
    except asyncio.TimeoutError:
        print(f"❌ {name} server connection timed out (5s)")
        return False
    except aiohttp.ClientError as e:
        print(f"❌ {name} server connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ {name} server error: {str(e)}")
        return False
    
    # Test 2: Try a simple completion request
    print(f"   🧪 Testing completion endpoint...")
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": "default",  # Most local servers accept this
                "messages": [{"role": "user", "content": "Hello, respond with just 'Hi there!'"}],
                "temperature": 0.1,
                "max_tokens": 10
            }
            
            async with session.post(
                f"{base_url}/chat/completions", 
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
                    print(f"✅ {name} completion works: '{response_text.strip()}'")
                    return True
                else:
                    print(f"⚠️  {name} completion returned HTTP {response.status}")
                    try:
                        error_text = await response.text()
                        print(f"   Error: {error_text[:100]}...")
                    except:
                        pass
                    return True  # Server is up, just completion might have issues
                    
    except asyncio.TimeoutError:
        print(f"⚠️  {name} completion request timed out (10s)")
        return True  # Server is up, just slow
    except Exception as e:
        print(f"⚠️  {name} completion test failed: {str(e)}")
        return True  # Server is up, endpoint might be different
    
    return True

async def test_api_credentials() -> dict:
    """Test API model credentials"""
    print(f"\n{'='*50}")
    print("Testing API Model Credentials")
    print("-" * 50)
    
    results = {}
    
    # OpenAI
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key and len(openai_key.strip()) > 0:
        print("✅ OpenAI: API key is configured")
        print(f"   Key format: {openai_key[:10]}...{openai_key[-4:] if len(openai_key) > 14 else 'short'}")
        results['openai'] = True
    else:
        print("❌ OpenAI: No API key found (set OPENAI_API_KEY)")
        results['openai'] = False
    
    # Claude/Anthropic
    claude_key = os.getenv('ANTHROPIC_API_KEY')
    if claude_key and len(claude_key.strip()) > 0:
        print("✅ Claude: API key is configured")
        print(f"   Key format: {claude_key[:10]}...{claude_key[-4:] if len(claude_key) > 14 else 'short'}")
        results['claude'] = True
    else:
        print("❌ Claude: No API key found (set ANTHROPIC_API_KEY)")
        results['claude'] = False
    
    return results

async def main():
    print("🔍 LOCAL MODEL CONNECTIVITY TEST")
    print("=" * 50)
    print("This script verifies local model servers and API credentials")
    print("Run this before starting the main chat server")
    
    # Test local models
    local_results = {}
    
    # DeepSeek
    deepseek_url = os.getenv('DEEPSEEK_BASE_URL', 'http://192.168.86.23:1234/v1')
    local_results['deepseek'] = await test_local_model('DeepSeek', deepseek_url)
    
    # Gemma  
    gemma_url = os.getenv('GEMMA_BASE_URL', 'http://192.168.86.23:1234/v1')
    local_results['gemma'] = await test_local_model('Gemma', gemma_url)
    
    # Test API credentials
    api_results = await test_api_credentials()
    
    # Summary
    print(f"\n{'='*50}")
    print("🏁 CONNECTIVITY SUMMARY")
    print("=" * 50)
    
    available_models = []
    unavailable_models = []
    
    # Local models
    for model, available in local_results.items():
        if available:
            available_models.append(model.upper())
            print(f"✅ {model.upper()}: Local server reachable")
        else:
            unavailable_models.append(model.upper())
            print(f"❌ {model.upper()}: Local server unavailable")
    
    # API models
    for model, available in api_results.items():
        if available:
            available_models.append(model.upper())
            print(f"✅ {model.upper()}: API credentials configured")
        else:
            unavailable_models.append(model.upper())
            print(f"❌ {model.upper()}: No API credentials")
    
    print(f"\n📊 Results:")
    print(f"   Available models: {len(available_models)}/4")
    print(f"   ✅ Working: {', '.join(available_models) if available_models else 'None'}")
    print(f"   ❌ Missing: {', '.join(unavailable_models) if unavailable_models else 'None'}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if len(available_models) == 0:
        print("   ⚠️  No models available - chat server will use fallback responses only")
        print("   🔧 Configure at least one model for better experience")
    elif len(available_models) == 1:
        print(f"   ✅ One model available - server will skip model selection")
        print(f"   🔧 Consider adding more models for intelligent selection")
    else:
        print(f"   ✅ Multiple models available - full model selection will work")
        print(f"   🚀 Chat server should work optimally")
    
    # Configuration help
    if 'DEEPSEEK' in unavailable_models or 'GEMMA' in unavailable_models:
        print(f"\n🔧 Local Model Setup:")
        print(f"   • Start your local model server (e.g., LM Studio, Ollama)")
        print(f"   • Set DEEPSEEK_BASE_URL and GEMMA_BASE_URL environment variables")
        print(f"   • Default: http://192.168.86.23:1234/v1")
    
    if 'OPENAI' in unavailable_models or 'CLAUDE' in unavailable_models:
        print(f"\n🔧 API Model Setup:")
        print(f"   • Get API keys from OpenAI and/or Anthropic")
        print(f"   • Set OPENAI_API_KEY and ANTHROPIC_API_KEY in .env file")
    
    print(f"\n🚀 Ready to start chat server:")
    print(f"   python chat_server.py")
    
    return len(available_models)

if __name__ == "__main__":
    try:
        available_count = asyncio.run(main())
        sys.exit(0 if available_count > 0 else 1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)