import requests
import json
import time

def check_lm_studio_config():
    """Check and suggest optimizations for LM Studio"""
    
    base_url = "http://192.168.86.23:1234/v1"
    
    print("🔧 LM Studio Optimization Suggestions\n")
    
    # Check available models
    try:
        response = requests.get(f"{base_url}/models")
        models = response.json()
        
        print("📦 Available Models:")
        for model in models['data']:
            print(f"  - {model['id']}")
        
        print("\n💡 Optimization Tips:")
        print("1. GPU Acceleration:")
        print("   - Ensure GPU is enabled in LM Studio settings")
        print("   - Check GPU memory usage (should be >80% when model loaded)")
        
        print("\n2. Model Loading:")
        print("   - Pre-load models before testing")
        print("   - Keep models in memory (don't unload between requests)")
        
        print("\n3. Context Length:")
        print("   - Reduce max_tokens in requests (use 150-200 instead of default)")
        print("   - Clear conversation history between tests")
        
        print("\n4. CPU Threads:")
        print("   - Set CPU threads to (physical cores - 2)")
        print("   - Disable CPU fallback if GPU is available")
        
        print("\n5. Network:")
        print("   - Ensure no firewall/antivirus scanning LM Studio traffic")
        print("   - Use wired connection if possible")
        
        # Test with minimal request
        print("\n🧪 Testing minimal request performance...")
        
        for model_id in ["deepseek/deepseek-r1-0528-qwen3-8b", "google/gemma-3-12b"]:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.1,
                "max_tokens": 10,  # Very short for speed test
                "stream": False
            }
            
            start = time.time()
            
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    timeout=30
                )
                elapsed = time.time() - start
                
                if resp.status_code == 200:
                    print(f"✅ {model_id}: {elapsed:.1f}s for minimal response")
                else:
                    print(f"❌ {model_id}: Error {resp.status_code}")
                    
            except Exception as e:
                print(f"❌ {model_id}: {str(e)}")
        
        # Advanced optimization suggestions
        print("\n🚀 Advanced Optimizations:")
        print("6. Model Quantization:")
        print("   - Use GGUF Q4_K_M quantization for balance of speed/quality")
        print("   - Avoid FP16 models if memory constrained")
        
        print("\n7. Batch Processing:")
        print("   - Process multiple requests in parallel when possible")
        print("   - Use connection pooling for multiple requests")
        
        print("\n8. System Optimizations:")
        print("   - Close unnecessary applications to free RAM/VRAM")
        print("   - Ensure adequate cooling (thermal throttling slows responses)")
        print("   - Use SSD storage for model files")
        
        print("\n9. Request Optimization:")
        print("   - Use lower temperature (0.1-0.3) for faster, more focused responses")
        print("   - Set appropriate top_p (0.9) and top_k (40) values")
        print("   - Minimize system prompt length")
        
        # Test with optimized parameters
        print("\n🎯 Testing with optimized parameters...")
        
        optimized_payload = {
            "model": "deepseek/deepseek-r1-0528-qwen3-8b",
            "messages": [{"role": "user", "content": "I feel anxious"}],
            "temperature": 0.3,
            "max_tokens": 100,
            "top_p": 0.9,
            "top_k": 40,
            "stream": False
        }
        
        start = time.time()
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                json=optimized_payload,
                timeout=30
            )
            elapsed = time.time() - start
            
            if resp.status_code == 200:
                result = resp.json()
                response_text = result['choices'][0]['message']['content']
                print(f"✅ Optimized DeepSeek: {elapsed:.1f}s")
                print(f"   Response length: {len(response_text)} chars")
                print(f"   First 100 chars: {response_text[:100]}...")
            else:
                print(f"❌ Optimized test failed: {resp.status_code}")
                
        except Exception as e:
            print(f"❌ Optimized test error: {str(e)}")
                
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure LM Studio is running")
        print("2. Check if server is started on port 1234")
        print("3. Verify IP address (192.168.86.23) is correct")
        print("4. Test connection: curl http://192.168.86.23:1234/v1/models")

if __name__ == "__main__":
    check_lm_studio_config()