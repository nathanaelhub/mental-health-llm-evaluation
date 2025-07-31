#!/usr/bin/env python3
"""
Test all 4 models with a direct therapeutic prompt to verify they work
"""
import requests
import json
import time
import asyncio
import aiohttp
from typing import Dict, Any

async def test_openai(prompt: str) -> Dict[str, Any]:
    """Test OpenAI model"""
    try:
        start = time.time()
        # This would require actual OpenAI API setup
        return {
            "model": "openai",
            "status": "requires_api_key",
            "time": time.time() - start,
            "response": "Requires OPENAI_API_KEY"
        }
    except Exception as e:
        return {"model": "openai", "status": "error", "error": str(e)}

async def test_claude(prompt: str) -> Dict[str, Any]:
    """Test Claude model"""
    try:
        start = time.time()
        # This would require actual Claude API setup
        return {
            "model": "claude",
            "status": "requires_api_key", 
            "time": time.time() - start,
            "response": "Requires ANTHROPIC_API_KEY"
        }
    except Exception as e:
        return {"model": "claude", "status": "error", "error": str(e)}

async def test_local_model(model_name: str, prompt: str, timeout: int = 45) -> Dict[str, Any]:
    """Test local model via LM Studio"""
    try:
        start = time.time()
        
        data = {
            "model": model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a compassionate mental health support assistant. Provide helpful, empathetic responses to people seeking emotional support."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(
                "http://192.168.86.23:1234/v1/chat/completions",
                json=data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    # Clean DeepSeek thinking tags
                    if '<think>' in content:
                        content = content.split('</think>')[-1].strip()
                    
                    return {
                        "model": model_name,
                        "status": "success",
                        "time": time.time() - start,
                        "response": content[:100] + "..." if len(content) > 100 else content,
                        "tokens": result.get('usage', {}).get('total_tokens', 0)
                    }
                else:
                    return {
                        "model": model_name,
                        "status": "http_error",
                        "time": time.time() - start,
                        "error": f"HTTP {response.status}"
                    }
                    
    except asyncio.TimeoutError:
        return {
            "model": model_name,
            "status": "timeout",
            "time": timeout,
            "error": f"Timed out after {timeout}s"
        }
    except Exception as e:
        return {
            "model": model_name,
            "status": "error", 
            "time": time.time() - start,
            "error": str(e)
        }

async def main():
    print("🧪 TESTING ALL 4 MODELS WITH THERAPEUTIC PROMPT")
    print("=" * 60)
    
    therapeutic_prompt = "I'm feeling really anxious about my job interview tomorrow. What techniques can help me feel more confident?"
    
    print(f"📝 Test prompt: {therapeutic_prompt}")
    print("\n🔄 Testing models...")
    
    # Test all models concurrently
    tasks = [
        test_openai(therapeutic_prompt),
        test_claude(therapeutic_prompt),
        test_local_model("deepseek/deepseek-r1-0528-qwen3-8b", therapeutic_prompt, 45),
        test_local_model("google/gemma-3-12b", therapeutic_prompt, 45)
    ]
    
    results = await asyncio.gather(*tasks)
    
    print(f"\n📊 RESULTS:")
    print("-" * 60)
    
    working_models = []
    for result in results:
        model = result['model']
        status = result['status']
        time_taken = result.get('time', 0)
        
        if status == "success":
            emoji = "✅"
            working_models.append(model)
            print(f"{emoji} {model.upper()}: {status} ({time_taken:.1f}s)")
            print(f"   💬 Response: {result['response']}")
            print(f"   🔢 Tokens: {result.get('tokens', 'N/A')}")
        elif status == "requires_api_key":
            emoji = "🔑"
            print(f"{emoji} {model.upper()}: {status} (API key needed)")
        else:
            emoji = "❌"
            print(f"{emoji} {model.upper()}: {status} ({time_taken:.1f}s)")
            print(f"   ⚠️  Error: {result.get('error', 'Unknown')}")
        print()
    
    print(f"🎯 SUMMARY:")
    print(f"   Working models: {len(working_models)}/4")
    print(f"   Ready for evaluation: {', '.join(working_models) if working_models else 'None'}")
    
    if len(working_models) >= 2:
        print(f"✅ Sufficient models for comparison!")
    else:
        print(f"❌ Need at least 2 working models for evaluation")
        
    # Test with increased timeout settings
    print(f"\n💡 RECOMMENDATION:")
    if any(r['status'] == 'timeout' for r in results):
        print("   • Increase timeout settings in chat_server.py")
        print("   • Consider using faster models or more powerful hardware")
    if any(r['status'] == 'requires_api_key' for r in results):
        print("   • Set OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables")
        print("   • Or focus on local models only")

if __name__ == "__main__":
    asyncio.run(main())