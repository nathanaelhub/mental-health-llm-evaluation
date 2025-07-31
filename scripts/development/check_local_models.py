#!/usr/bin/env python3
"""
Diagnostic script to check local model availability and configuration
"""
import requests
import json
import os
from typing import List, Dict

def check_endpoint(url: str, name: str) -> Dict[str, any]:
    """Check if an endpoint is available"""
    try:
        response = requests.get(url, timeout=5)
        return {
            "name": name,
            "url": url,
            "status": "available" if response.status_code == 200 else f"error_{response.status_code}",
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:100]
        }
    except requests.exceptions.ConnectionError:
        return {"name": name, "url": url, "status": "connection_refused", "response": "Not running"}
    except requests.exceptions.Timeout:
        return {"name": name, "url": url, "status": "timeout", "response": "Server too slow"}
    except Exception as e:
        return {"name": name, "url": url, "status": "error", "response": str(e)}

def check_ollama_models(base_url: str) -> List[str]:
    """Check what models are available in Ollama"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model.get('name', 'unknown') for model in data.get('models', [])]
        return []
    except Exception:
        return []

def main():
    print("🔍 LOCAL MODEL AVAILABILITY CHECK")
    print("=" * 50)
    
    # Common local model endpoints to check
    endpoints = [
        ("http://localhost:11434/api/tags", "Ollama (Standard Port)"),
        ("http://localhost:1234/v1/models", "LM Studio (Standard Port)"),
        ("http://192.168.86.23:1234/v1/models", "LM Studio (Custom IP)"),
        ("http://localhost:8080/v1/models", "Alternative Local Server"),
        ("http://127.0.0.1:11434/api/tags", "Ollama (127.0.0.1)"),
    ]
    
    print("📡 Checking endpoints...")
    available_endpoints = []
    
    for url, name in endpoints:
        result = check_endpoint(url, name)
        status_emoji = "✅" if result["status"] == "available" else "❌"
        print(f"{status_emoji} {name}: {result['status']}")
        if result["status"] == "available":
            available_endpoints.append(result)
            if "ollama" in name.lower():
                models = check_ollama_models(url.replace("/api/tags", ""))
                if models:
                    print(f"   📦 Available models: {', '.join(models)}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if not available_endpoints:
        print("❌ No local model servers found!")
        print("\n💡 TO FIX THIS:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Run: ollama serve")
        print("3. Pull models: ollama pull deepseek-coder && ollama pull gemma:7b")
        print("\nOR")
        print("1. Install LM Studio: https://lmstudio.ai/")
        print("2. Start local server on port 1234")
        print("3. Load DeepSeek and Gemma models")
        
    else:
        print("✅ Found available endpoints!")
        for endpoint in available_endpoints:
            print(f"   • {endpoint['name']}: {endpoint['url']}")
        
        print(f"\n🔧 CONFIGURATION:")
        if any("ollama" in ep['name'].lower() for ep in available_endpoints):
            print("Set environment variables:")
            print("export LOCAL_LLM_SERVER='localhost:11434'")
        elif any("lm studio" in ep['name'].lower() for ep in available_endpoints):
            if any("192.168.86.23" in ep['url'] for ep in available_endpoints):
                print("export LOCAL_LLM_SERVER='192.168.86.23:1234'")
            else:
                print("export LOCAL_LLM_SERVER='localhost:1234'")
    
    print(f"\n🧪 CURRENT ENVIRONMENT:")
    print(f"LOCAL_LLM_SERVER = {os.getenv('LOCAL_LLM_SERVER', 'Not set')}")
    print(f"DEEPSEEK_MODEL = {os.getenv('DEEPSEEK_MODEL', 'Not set')}")
    print(f"GEMMA_MODEL = {os.getenv('GEMMA_MODEL', 'Not set')}")

if __name__ == "__main__":
    main()