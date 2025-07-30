#!/usr/bin/env python3
"""
Model Selection Debugging Script
================================

Comprehensive debugging tool to identify issues with model selection,
timeouts, and confidence scoring problems.
"""

import sys
import os
import asyncio
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.chat.dynamic_model_selector import DynamicModelSelector, PromptType
    from src.models.openai_client import OpenAIClient
    from src.models.claude_client import ClaudeClient  
    from src.models.deepseek_client import DeepSeekClient
    from src.models.gemma_client import GemmaClient
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class ModelSelectionDebugger:
    """Comprehensive debugging tool for model selection issues"""
    
    def __init__(self):
        self.models_config = {
            'models': {
                'openai': {'enabled': True, 'cost_per_token': 0.0001, 'model_name': 'gpt-4'},
                'claude': {'enabled': True, 'cost_per_token': 0.00015, 'model_name': 'claude-3'},
                'deepseek': {'enabled': True, 'cost_per_token': 0.00005, 'model_name': 'deepseek/deepseek-r1-0528-qwen3-8b'},
                'gemma': {'enabled': True, 'cost_per_token': 0.00003, 'model_name': 'google/gemma-3-12b'}
            },
            'default_model': 'openai',
            'selection_timeout': 10.0,
            'similarity_threshold': 0.9
        }
        
        self.test_prompt = "I am feeling anxious about work and need some support"
        self.results = {}
    
    async def run_full_debug(self):
        """Run complete debugging suite"""
        print("🔍 MENTAL HEALTH MODEL SELECTION DEBUGGER")
        print("=" * 50)
        
        # Test 1: Environment and API Key Check
        await self.check_environment()
        
        # Test 2: Individual Model Client Testing
        await self.test_individual_models()
        
        # Test 3: Model Selector Initialization
        await self.test_model_selector_init()
        
        # Test 4: Parallel Evaluation Testing
        await self.test_parallel_evaluation()
        
        # Test 5: Scoring System Testing
        await self.test_scoring_system()
        
        # Test 6: Timeout Analysis
        await self.analyze_timeouts()
        
        # Generate Summary Report
        self.generate_report()
    
    async def check_environment(self):
        """Check environment variables and API keys"""
        print("\n🔧 ENVIRONMENT CHECK")
        print("-" * 30)
        
        # Check for .env file
        env_file = project_root / ".env"
        if env_file.exists():
            print("✅ .env file found")
        else:
            print("⚠️  .env file not found")
        
        # Check API keys
        api_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
            'GEMMA_API_KEY': os.getenv('GEMMA_API_KEY'),
        }
        
        for key, value in api_keys.items():
            if value:
                print(f"✅ {key}: {'*' * 8}{value[-4:] if len(value) > 4 else '****'}")
            else:
                print(f"❌ {key}: Not set")
        
        # Check local model endpoints
        local_endpoints = {
            'LM_STUDIO_URL': os.getenv('LM_STUDIO_URL', 'http://192.168.86.23:1234'),
            'DEEPSEEK_API_URL': os.getenv('DEEPSEEK_API_URL', 'http://192.168.86.23:1234/v1'),
            'GEMMA_API_URL': os.getenv('GEMMA_API_URL', 'http://192.168.86.23:1234/v1'),
        }
        
        for endpoint_name, url in local_endpoints.items():
            print(f"🌐 {endpoint_name}: {url}")
        
        self.results['environment'] = {
            'env_file_exists': env_file.exists(),
            'api_keys': {k: bool(v) for k, v in api_keys.items()},
            'endpoints': local_endpoints
        }
    
    async def test_individual_models(self):
        """Test each model client individually"""
        print("\n🤖 INDIVIDUAL MODEL TESTING")
        print("-" * 30)
        
        model_clients = {
            'openai': OpenAIClient,
            'claude': ClaudeClient,
            'deepseek': DeepSeekClient,
            'gemma': GemmaClient
        }
        
        self.results['individual_models'] = {}
        
        for model_name, ClientClass in model_clients.items():
            print(f"\n🔹 Testing {model_name.upper()}...")
            
            try:
                # Initialize client
                start_init = time.time()
                client = ClientClass()
                init_time = (time.time() - start_init) * 1000
                print(f"  ✅ Initialization: {init_time:.1f}ms")
                
                # Test response generation
                start_response = time.time()
                
                # Check if client has async generate_response
                if hasattr(client, 'generate_response'):
                    try:
                        if asyncio.iscoroutinefunction(client.generate_response):
                            response = await client.generate_response(
                                prompt=self.test_prompt,
                                system_prompt="You are a helpful mental health support assistant."
                            )
                        else:
                            response = client.generate_response(
                                prompt=self.test_prompt,
                                system_prompt="You are a helpful mental health support assistant."
                            )
                    except Exception as e:
                        print(f"  ❌ generate_response failed: {e}")
                        response = None
                else:
                    print(f"  ⚠️  No generate_response method, trying fallback...")
                    try:
                        response = client.chat(self.test_prompt)
                    except Exception as e:
                        print(f"  ❌ Fallback chat failed: {e}")
                        response = None
                
                response_time = (time.time() - start_response) * 1000
                
                if response:
                    if hasattr(response, 'content'):
                        content = response.content[:100] + "..." if len(response.content) > 100 else response.content
                        print(f"  ✅ Response: {content}")
                        print(f"  ⏱️  Response time: {response_time:.1f}ms")
                        
                        # Check response object structure
                        if hasattr(response, 'response_time_ms'):
                            print(f"  📊 Internal response time: {response.response_time_ms:.1f}ms")
                        if hasattr(response, 'error'):
                            if response.error:
                                print(f"  ⚠️  Response has error: {response.error}")
                    else:
                        content = str(response)[:100] + "..." if len(str(response)) > 100 else str(response)
                        print(f"  ✅ Response: {content}")
                        print(f"  ⏱️  Response time: {response_time:.1f}ms")
                    
                    self.results['individual_models'][model_name] = {
                        'status': 'success',
                        'init_time_ms': init_time,
                        'response_time_ms': response_time,
                        'has_content': bool(content),
                        'content_length': len(content) if content else 0
                    }
                else:
                    print(f"  ❌ No response received")
                    self.results['individual_models'][model_name] = {
                        'status': 'no_response',
                        'init_time_ms': init_time,
                        'response_time_ms': response_time
                    }
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                print(f"  📝 Traceback: {traceback.format_exc()}")
                self.results['individual_models'][model_name] = {
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
    
    async def test_model_selector_init(self):
        """Test DynamicModelSelector initialization"""
        print("\n🎯 MODEL SELECTOR INITIALIZATION")
        print("-" * 30)
        
        try:
            start_time = time.time()
            selector = DynamicModelSelector(self.models_config)
            init_time = (time.time() - start_time) * 1000
            
            print(f"✅ DynamicModelSelector initialized in {init_time:.1f}ms")
            print(f"📊 Available models: {selector.get_available_models()}")
            print(f"🔧 Model clients: {list(selector.models.keys())}")
            
            # Check each model client in selector
            for model_id, client in selector.models.items():
                print(f"  🔹 {model_id}: {type(client).__name__}")
                
                # Check if client has required methods
                has_generate = hasattr(client, 'generate_response')
                is_async = asyncio.iscoroutinefunction(getattr(client, 'generate_response', None))
                print(f"    - generate_response: {has_generate}")
                print(f"    - is_async: {is_async}")
            
            self.results['selector_init'] = {
                'status': 'success',
                'init_time_ms': init_time,
                'available_models': selector.get_available_models(),
                'model_count': len(selector.models)
            }
            
            self.selector = selector
            
        except Exception as e:
            print(f"❌ DynamicModelSelector initialization failed: {e}")
            print(f"📝 Traceback: {traceback.format_exc()}")
            self.results['selector_init'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.selector = None
    
    async def test_parallel_evaluation(self):
        """Test parallel model evaluation"""
        print("\n⚡ PARALLEL EVALUATION TESTING")
        print("-" * 30)
        
        if not self.selector:
            print("❌ Cannot test parallel evaluation - selector not initialized")
            self.results['parallel_evaluation'] = {'status': 'skipped', 'reason': 'selector_not_initialized'}
            return
        
        try:
            print(f"🔄 Testing parallel evaluation with prompt: '{self.test_prompt}'")
            
            start_time = time.time()
            evaluations = await self.selector.parallel_evaluate(self.test_prompt)
            total_time = (time.time() - start_time) * 1000
            
            print(f"⏱️  Total parallel evaluation time: {total_time:.1f}ms")
            print(f"📊 Evaluations returned: {len(evaluations)}")
            
            for evaluation in evaluations:
                print(f"  🔹 {evaluation.model_id}:")
                print(f"    - Response time: {evaluation.response_time_ms:.1f}ms")
                print(f"    - Composite score: {evaluation.composite_score:.2f}")
                print(f"    - Has error: {bool(evaluation.error)}")
                if evaluation.error:
                    print(f"    - Error: {evaluation.error}")
                if evaluation.response_content:
                    content_preview = evaluation.response_content[:80] + "..." if len(evaluation.response_content) > 80 else evaluation.response_content
                    print(f"    - Response: {content_preview}")
            
            self.results['parallel_evaluation'] = {
                'status': 'success',
                'total_time_ms': total_time,
                'evaluations_count': len(evaluations),
                'successful_models': [e.model_id for e in evaluations if not e.error],
                'failed_models': [e.model_id for e in evaluations if e.error]
            }
            
        except Exception as e:
            print(f"❌ Parallel evaluation failed: {e}")
            print(f"📝 Traceback: {traceback.format_exc()}")
            self.results['parallel_evaluation'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def test_scoring_system(self):
        """Test the scoring and selection system"""
        print("\n📊 SCORING SYSTEM TESTING")
        print("-" * 30)
        
        if not self.selector:
            print("❌ Cannot test scoring system - selector not initialized")
            self.results['scoring_system'] = {'status': 'skipped', 'reason': 'selector_not_initialized'}
            return
        
        try:
            print(f"🎯 Testing complete model selection with scoring...")
            
            start_time = time.time()
            selection = await self.selector.select_best_model(self.test_prompt)
            total_time = (time.time() - start_time) * 1000
            
            print(f"⏱️  Total selection time: {total_time:.1f}ms")
            
            if selection:
                print(f"✅ Model selection completed successfully!")
                print(f"  🏆 Selected model: {selection.selected_model_id}")
                print(f"  🎯 Confidence score: {selection.confidence_score:.2f}")
                print(f"  📝 Reasoning: {selection.selection_reasoning}")
                print(f"  📊 Model scores: {selection.model_scores}")
                print(f"  🏷️  Prompt type: {selection.prompt_type}")
                print(f"  ⏱️  Latency metrics: {selection.latency_metrics}")
                
                # Check if this is a cached result
                print(f"  💾 Cached result: {selection.cached}")
                
                # Analyze why confidence might be 0
                if selection.confidence_score == 0.0:
                    print("⚠️  CONFIDENCE SCORE IS 0% - ANALYZING...")
                    print(f"    - Model scores: {selection.model_scores}")
                    print(f"    - Selection reasoning: {selection.selection_reasoning}")
                    
                    # Check if it's a fallback selection
                    if "fallback" in selection.selection_reasoning.lower() or "timeout" in selection.selection_reasoning.lower():
                        print("    🔍 This appears to be a fallback selection due to timeout/error")
                    else:
                        print("    🔍 This appears to be a regular selection with 0% confidence")
                
                self.results['scoring_system'] = {
                    'status': 'success',
                    'total_time_ms': total_time,
                    'selected_model': selection.selected_model_id,
                    'confidence_score': selection.confidence_score,
                    'model_scores': selection.model_scores,
                    'is_fallback': "fallback" in selection.selection_reasoning.lower(),
                    'cached': selection.cached
                }
            else:
                print("❌ No selection returned")
                self.results['scoring_system'] = {
                    'status': 'no_selection',
                    'total_time_ms': total_time
                }
                
        except Exception as e:
            print(f"❌ Scoring system test failed: {e}")
            print(f"📝 Traceback: {traceback.format_exc()}")
            self.results['scoring_system'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def analyze_timeouts(self):
        """Analyze timeout behavior and patterns"""
        print("\n⏰ TIMEOUT ANALYSIS")
        print("-" * 30)
        
        if not self.selector:
            print("❌ Cannot analyze timeouts - selector not initialized")
            self.results['timeout_analysis'] = {'status': 'skipped', 'reason': 'selector_not_initialized'}
            return
        
        # Test with different timeout values
        timeout_tests = [2.0, 5.0, 10.0, 15.0]
        timeout_results = {}
        
        for timeout in timeout_tests:
            print(f"\n🔹 Testing with {timeout}s timeout...")
            
            # Temporarily adjust selector timeout
            original_timeout = self.selector.selection_timeout
            self.selector.selection_timeout = timeout
            
            try:
                start_time = time.time()
                selection = await self.selector.select_best_model(f"Test message with {timeout}s timeout")
                actual_time = (time.time() - start_time) * 1000
                
                if selection:
                    print(f"  ✅ Completed in {actual_time:.1f}ms")
                    print(f"  🏆 Selected: {selection.selected_model_id}")
                    print(f"  🎯 Confidence: {selection.confidence_score:.2f}")
                    
                    is_timeout = actual_time >= (timeout * 1000 * 0.9)  # Within 90% of timeout
                    timeout_results[timeout] = {
                        'status': 'success',
                        'time_ms': actual_time,
                        'selected_model': selection.selected_model_id,
                        'confidence_score': selection.confidence_score,
                        'hit_timeout': is_timeout
                    }
                else:
                    print(f"  ❌ No selection returned")
                    timeout_results[timeout] = {
                        'status': 'no_selection',
                        'time_ms': actual_time
                    }
                    
            except asyncio.TimeoutError:
                actual_time = (time.time() - start_time) * 1000
                print(f"  ⏰ Timed out after {actual_time:.1f}ms")
                timeout_results[timeout] = {
                    'status': 'timeout',
                    'time_ms': actual_time
                }
            except Exception as e:
                actual_time = (time.time() - start_time) * 1000
                print(f"  ❌ Error: {e}")
                timeout_results[timeout] = {
                    'status': 'error',
                    'time_ms': actual_time,
                    'error': str(e)
                }
            finally:
                # Restore original timeout
                self.selector.selection_timeout = original_timeout
        
        self.results['timeout_analysis'] = timeout_results
        
        # Analyze patterns
        print(f"\n📈 TIMEOUT PATTERN ANALYSIS:")
        successful_timeouts = [t for t, r in timeout_results.items() if r['status'] == 'success']
        if successful_timeouts:
            min_success_timeout = min(successful_timeouts)
            print(f"  ✅ Minimum successful timeout: {min_success_timeout}s")
        else:
            print(f"  ❌ No successful completions with any timeout")
    
    def generate_report(self):
        """Generate comprehensive debugging report"""
        print("\n" + "=" * 60)
        print("📋 DEBUGGING REPORT SUMMARY")
        print("=" * 60)
        
        # Environment Status
        env_status = self.results.get('environment', {})
        api_keys_set = sum(env_status.get('api_keys', {}).values())
        print(f"\n🔧 Environment: {api_keys_set}/4 API keys configured")
        
        # Individual Model Status
        individual_results = self.results.get('individual_models', {})
        working_models = [m for m, r in individual_results.items() if r.get('status') == 'success']
        print(f"🤖 Individual Models: {len(working_models)}/4 working ({', '.join(working_models)})")
        
        # Model Selector Status
        selector_status = self.results.get('selector_init', {}).get('status', 'unknown')
        print(f"🎯 Model Selector: {selector_status}")
        
        # Parallel Evaluation Status
        parallel_status = self.results.get('parallel_evaluation', {})
        if parallel_status.get('status') == 'success':
            successful_count = len(parallel_status.get('successful_models', []))
            print(f"⚡ Parallel Evaluation: {successful_count}/4 models successful")
        else:
            print(f"⚡ Parallel Evaluation: {parallel_status.get('status', 'unknown')}")
        
        # Scoring System Status
        scoring_status = self.results.get('scoring_system', {})
        print(f"📊 Scoring System: {scoring_status.get('status', 'unknown')}")
        if scoring_status.get('confidence_score') is not None:
            confidence = scoring_status['confidence_score']
            print(f"🎯 Confidence Score: {confidence:.1%}")
            if confidence == 0.0:
                if scoring_status.get('is_fallback'):
                    print("⚠️  ISSUE: Using fallback due to timeout/error")
                else:
                    print("⚠️  ISSUE: Zero confidence in regular selection")
        
        # Root Cause Analysis
        print(f"\n🔍 ROOT CAUSE ANALYSIS:")
        
        if len(working_models) == 0:
            print("❌ CRITICAL: No models are responding individually")
            print("   → Check API keys and endpoint configurations")
            print("   → Verify network connectivity to model services")
        elif len(working_models) < 4:
            failed_models = [m for m in individual_results.keys() if m not in working_models]
            print(f"⚠️  PARTIAL: Only {len(working_models)}/4 models working")
            print(f"   → Failed models: {', '.join(failed_models)}")
            print("   → Check configurations for failed models")
        else:
            print("✅ All models responding individually")
        
        if parallel_status.get('status') != 'success' and len(working_models) > 0:
            print("❌ ISSUE: Models work individually but parallel evaluation fails")
            print("   → Check async/await implementation in model clients")
            print("   → Verify timeout handling in parallel evaluation")
        
        if scoring_status.get('is_fallback'):
            print("❌ ISSUE: Selection falling back due to timeout")
            print("   → Increase selection timeout")
            print("   → Optimize model response times")
            print("   → Check for network latency issues")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if api_keys_set < 4:
            print("1. Configure missing API keys in .env file")
        
        if len(working_models) < 4:
            print("2. Fix non-responding model clients")
            
        timeout_results = self.results.get('timeout_analysis', {})
        successful_timeouts = [t for t, r in timeout_results.items() if r.get('status') == 'success']
        if successful_timeouts:
            recommended_timeout = min(successful_timeouts) * 1.2  # Add 20% buffer
            print(f"3. Set selection timeout to at least {recommended_timeout:.1f}s")
        
        if scoring_status.get('confidence_score', 1.0) == 0.0:
            print("4. Fix confidence score calculation in fallback scenarios")
        
        print(f"\n📁 Full results saved to debug log for detailed analysis")
        
        # Save detailed results to file
        import json
        debug_file = project_root / "results" / "development" / "model_selection_debug.json"
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(debug_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"📊 Detailed results: {debug_file}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

async def main():
    """Main debugging function"""
    debugger = ModelSelectionDebugger()
    await debugger.run_full_debug()

if __name__ == "__main__":
    asyncio.run(main())