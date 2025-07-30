#!/usr/bin/env python3
"""
Test Script for Fixed Model Selection System
==========================================

Tests the model selection functionality with various prompt types
to verify that all models are evaluated and proper scores are returned.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

# Test prompts for different categories
TEST_PROMPTS = [
    {
        'message': 'I feel very anxious about my work presentation tomorrow',
        'expected_type': 'anxiety',
        'description': 'Anxiety prompt - should favor Claude or OpenAI'
    },
    {
        'message': 'I feel so depressed and hopeless lately',
        'expected_type': 'depression', 
        'description': 'Depression prompt - should favor Claude'
    },
    {
        'message': 'What are the symptoms of PTSD?',
        'expected_type': 'information_seeking',
        'description': 'Information seeking - should favor DeepSeek or OpenAI'
    },
    {
        'message': 'I want to kill myself',
        'expected_type': 'crisis',
        'description': 'Crisis prompt - should prioritize safety, favor Claude'
    },
    {
        'message': 'My partner and I are having relationship problems',
        'expected_type': 'relationship',
        'description': 'Relationship prompt - should favor Claude or Gemma'
    },
    {
        'message': 'Hello, how are you?',
        'expected_type': 'general_support',
        'description': 'General support - balanced scoring'
    }
]

class ModelSelectionTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_chat_endpoint(self, prompt_data: Dict[str, str]) -> Dict[str, Any]:
        """Test a single chat request and return the response"""
        
        request_data = {
            'message': prompt_data['message'],
            'session_id': None,  # New session
            'user_id': 'test-user',
            'force_reselection': False
        }
        
        print(f"\n🔍 Testing: {prompt_data['description']}")
        print(f"📝 Message: '{prompt_data['message']}'")
        print(f"🎯 Expected type: {prompt_data['expected_type']}")
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Request failed: {response.status} - {error_text}")
                    return None
                
                data = await response.json()
                response_time = (time.time() - start_time) * 1000
                
                # Extract key information
                result = {
                    'selected_model': data.get('selected_model'),
                    'confidence_score': data.get('confidence_score'),
                    'model_scores': data.get('model_scores', {}),
                    'prompt_type': data.get('prompt_type'),
                    'reasoning': data.get('reasoning'),
                    'response_time_ms': response_time,
                    'conversation_mode': data.get('conversation_mode'),
                    'response_preview': data.get('response', '')[:100] + '...' if len(data.get('response', '')) > 100 else data.get('response', '')
                }
                
                return result
                
        except Exception as e:
            print(f"❌ Request exception: {e}")
            return None
    
    def analyze_results(self, prompt_data: Dict[str, str], result: Dict[str, Any]):
        """Analyze and display the test results"""
        
        if not result:
            print("❌ No result to analyze")
            return
        
        print(f"✅ Response received in {result['response_time_ms']:.0f}ms")
        print(f"🤖 Selected Model: {result['selected_model'].upper()}")
        print(f"📊 Confidence: {result['confidence_score']:.1%}")
        print(f"📋 Detected Type: {result['prompt_type']}")
        print(f"🎯 Expected Type: {prompt_data['expected_type']}")
        
        # Check if type classification was correct
        type_match = result['prompt_type'] == prompt_data['expected_type']
        print(f"✅ Type Classification: {'Correct' if type_match else 'Incorrect'}")
        
        # Display model scores
        model_scores = result.get('model_scores', {})
        if model_scores:
            print(f"📈 Model Scores:")
            sorted_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (model, score) in enumerate(sorted_scores):
                marker = "🏆" if i == 0 else "📍"
                selected = "👈 SELECTED" if model == result['selected_model'] else ""
                print(f"   {marker} {model.upper()}: {score:.2f}/10.0 {selected}")
        else:
            print("⚠️ No model scores returned")
        
        # Show reasoning
        print(f"🤔 Reasoning: {result['reasoning']}")
        print(f"💬 Response Preview: {result['response_preview']}")
    
    async def run_comprehensive_test(self):
        """Run all test prompts and analyze results"""
        
        print("🧠 Mental Health Chat - Model Selection Testing")
        print("=" * 60)
        print(f"🎯 Testing {len(TEST_PROMPTS)} different prompt types")
        print(f"🌐 Server: {self.base_url}")
        
        results = []
        
        for i, prompt_data in enumerate(TEST_PROMPTS, 1):
            print(f"\n{'='*20} Test {i}/{len(TEST_PROMPTS)} {'='*20}")
            
            result = await self.test_chat_endpoint(prompt_data)
            if result:
                self.analyze_results(prompt_data, result)
                results.append({
                    'prompt': prompt_data,
                    'result': result
                })
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Summary analysis
        self.print_summary(results)
    
    def print_summary(self, results):
        """Print a comprehensive summary of all tests"""
        
        print(f"\n{'='*60}")
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*60}")
        
        if not results:
            print("❌ No successful tests to summarize")
            return
        
        # Model selection frequency
        model_counts = {}
        total_confidence = 0
        type_accuracy = 0
        
        for test_result in results:
            selected_model = test_result['result']['selected_model']
            model_counts[selected_model] = model_counts.get(selected_model, 0) + 1
            total_confidence += test_result['result']['confidence_score']
            
            if test_result['result']['prompt_type'] == test_result['prompt']['expected_type']:
                type_accuracy += 1
        
        print(f"📈 Model Selection Distribution:")
        for model, count in sorted(model_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"   {model.upper()}: {count} times ({percentage:.1f}%)")
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Average Confidence: {(total_confidence / len(results)):.1%}")
        print(f"   Type Classification Accuracy: {(type_accuracy / len(results)):.1%}")
        
        # Check if any model was never selected
        all_models = {'openai', 'claude', 'deepseek', 'gemma'}
        selected_models = set(model_counts.keys())
        unused_models = all_models - selected_models
        
        if unused_models:
            print(f"\n⚠️ Models never selected: {', '.join(unused_models).upper()}")
        else:
            print(f"\n✅ All models were selected at least once")
        
        # Specialization analysis
        print(f"\n🎯 Model Specialization Analysis:")
        for test_result in results:
            prompt_type = test_result['prompt']['expected_type']
            selected_model = test_result['result']['selected_model']
            confidence = test_result['result']['confidence_score']
            print(f"   {prompt_type}: {selected_model.upper()} ({confidence:.1%})")

async def main():
    """Main test runner"""
    
    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/api/status") as response:
                if response.status != 200:
                    print("❌ Server is not responding. Please start the chat server first:")
                    print("   python chat_server.py")
                    return
                
                status = await response.json()
                print(f"✅ Server is running (version: {status.get('version', 'unknown')})")
    
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Please start the chat server first: python chat_server.py")
        return
    
    # Run the tests
    async with ModelSelectionTester() as tester:
        await tester.run_comprehensive_test()

if __name__ == "__main__":
    print("🧪 Starting Model Selection Test Suite...")
    asyncio.run(main())