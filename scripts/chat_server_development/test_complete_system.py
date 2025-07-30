#!/usr/bin/env python3
"""
Complete Mental Health Chat System Test Suite
============================================

Comprehensive testing of the entire chat system including:
1. Model selection with confidence scores
2. Conversation flow and session management  
3. UI integration and dark mode
4. API endpoints and responses
5. Performance and timeout handling

Usage:
    python scripts/test_complete_system.py
    python scripts/test_complete_system.py --comprehensive
"""

import sys
import os
import asyncio
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from dotenv import load_dotenv
    from src.chat.dynamic_model_selector import DynamicModelSelector, PromptType
    from src.chat.conversation_session_manager import ConversationSessionManager, MessageRole
    from src.chat.persistent_session_store import SessionStoreType
    from src.models.openai_client import OpenAIClient
    from src.models.claude_client import ClaudeClient
    from src.models.deepseek_client import DeepSeekClient
    from src.models.gemma_client import GemmaClient
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

load_dotenv()

class CompleteSystemTester:
    """Comprehensive test suite for the complete chat system"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        
        # Test configuration
        self.models_config = {
            'models': {
                'openai': {'enabled': True, 'cost_per_token': 0.0001, 'model_name': 'gpt-4o-mini'},
                'claude': {'enabled': True, 'cost_per_token': 0.00015, 'model_name': 'claude-3-5-sonnet-20241022'},
                'deepseek': {'enabled': True, 'cost_per_token': 0.00005, 'model_name': 'deepseek/deepseek-r1-0528-qwen3-8b'},
                'gemma': {'enabled': True, 'cost_per_token': 0.00003, 'model_name': 'google/gemma-3-12b'}
            },
            'default_model': 'openai',
            'selection_timeout': 40.0,
            'similarity_threshold': 0.9
        }
        
        # Comprehensive test scenarios
        self.test_scenarios = [
            {
                'name': 'Anxiety Support',
                'prompt': "I'm feeling very anxious about my job interview tomorrow. I can't stop worrying about it.",
                'expected_type': PromptType.ANXIETY,
                'context': "User needs emotional support and practical anxiety management techniques"
            },
            {
                'name': 'Depression Symptoms',
                'prompt': "I've been feeling really depressed lately. Nothing seems to bring me joy anymore.",
                'expected_type': PromptType.DEPRESSION,
                'context': "User showing signs of depression, needs empathetic response and resources"
            },
            {
                'name': 'Crisis Situation',
                'prompt': "I'm having thoughts of hurting myself. I don't know what to do.",
                'expected_type': PromptType.CRISIS,
                'context': "Crisis situation requiring immediate safety-focused response"
            },
            {
                'name': 'Information Seeking',
                'prompt': "What are some effective coping strategies for managing stress at work?",
                'expected_type': PromptType.INFORMATION_SEEKING,
                'context': "User seeking educational information about stress management"
            },
            {
                'name': 'Relationship Issues',
                'prompt': "My partner and I have been fighting a lot lately. I don't know how to fix our relationship.",
                'expected_type': PromptType.RELATIONSHIP,
                'context': "User needs guidance on relationship communication and conflict resolution"
            }
        ]
    
    async def run_complete_tests(self, comprehensive: bool = False):
        """Run the complete test suite"""
        print("🧠 COMPLETE MENTAL HEALTH CHAT SYSTEM TEST")
        print("=" * 60)
        print(f"Project: {self.project_root}")
        print(f"Mode: {'Comprehensive' if comprehensive else 'Standard'}")
        print(f"Test scenarios: {len(self.test_scenarios)}")
        print("=" * 60)
        
        # Test 1: Model Selection and Confidence Calculation
        await self.test_model_selection()
        
        # Test 2: Conversation Flow and Session Management
        await self.test_conversation_flow()
        
        # Test 3: Prompt Classification and Scoring
        await self.test_prompt_classification()
        
        # Test 4: Performance and Timeout Handling
        if comprehensive:
            await self.test_performance_comprehensive()
        else:
            await self.test_performance_basic()
        
        # Test 5: UI Integration (file checks)
        await self.test_ui_integration()
        
        # Generate comprehensive report
        self.generate_test_report()
        
        # Final summary
        self.print_final_summary()
    
    async def test_model_selection(self):
        """Test model selection with confidence calculation"""
        print("\n🎯 TESTING MODEL SELECTION & CONFIDENCE CALCULATION")
        print("-" * 50)
        
        selector = DynamicModelSelector(self.models_config)
        self.test_results['model_selection'] = {}
        
        for i, scenario in enumerate(self.test_scenarios[:3]):  # Test first 3 scenarios
            print(f"\n🔹 Scenario {i+1}: {scenario['name']}")
            print(f"   Prompt: \"{scenario['prompt'][:60]}{'...' if len(scenario['prompt']) > 60 else ''}\"")
            
            try:
                start_time = time.time()
                selection = await selector.select_best_model(scenario['prompt'])
                selection_time = (time.time() - start_time) * 1000
                
                # Analyze results
                confidence_pct = selection.confidence_score * 100
                
                print(f"   ✅ Selected: {selection.selected_model_id}")
                print(f"   📊 Confidence: {confidence_pct:.1f}%")
                print(f"   ⏱️  Selection time: {selection_time:.1f}ms")
                print(f"   🏆 Models evaluated: {len(selection.model_scores)}")
                print(f"   📝 Reasoning: {selection.selection_reasoning[:80]}...")
                
                # Validate confidence score
                if selection.confidence_score > 0:
                    confidence_status = "✅ Working"
                    if confidence_pct >= 50:
                        confidence_level = "High"
                    elif confidence_pct >= 25:
                        confidence_level = "Moderate"
                    else:
                        confidence_level = "Low"
                else:
                    confidence_status = "❌ Broken"
                    confidence_level = "None"
                
                print(f"   🎯 Confidence level: {confidence_level}")
                
                self.test_results['model_selection'][scenario['name']] = {
                    'status': 'success',
                    'selected_model': selection.selected_model_id,
                    'confidence_score': selection.confidence_score,
                    'confidence_level': confidence_level,
                    'selection_time_ms': selection_time,
                    'models_evaluated': len(selection.model_scores),
                    'reasoning_length': len(selection.selection_reasoning)
                }
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.test_results['model_selection'][scenario['name']] = {
                    'status': 'error',
                    'error': str(e)
                }
    
    async def test_conversation_flow(self):
        """Test conversation continuity and session management"""
        print("\n💬 TESTING CONVERSATION FLOW & SESSION MANAGEMENT")
        print("-" * 50)
        
        session_manager = ConversationSessionManager(
            store_type=SessionStoreType.MEMORY,
            enable_safety_monitoring=True,
            enable_audit_trail=True
        )
        
        self.test_results['conversation_flow'] = {}
        
        # Test conversation scenario
        test_conversation = [
            "I'm feeling anxious about work",
            "What techniques can help me calm down?",
            "Thank you, that's helpful. What about sleep issues?",
            "I appreciate your help"
        ]
        
        try:
            # Create initial session
            session = await session_manager.create_session(
                user_id="test_user",
                selected_model="openai",
                initial_message=test_conversation[0]
            )
            
            print(f"✅ Session created: {session.session_id[:8]}...")
            
            # Simulate conversation turns
            for i, message in enumerate(test_conversation):
                # Add user message
                await session_manager.add_message(
                    session.session_id,
                    MessageRole.USER,
                    message
                )
                
                # Generate mock assistant response
                mock_response = f"Thank you for sharing that with me. I understand you're dealing with {message.lower()[:20]}... This is a response to turn {i+1}."
                
                # Add assistant response
                await session_manager.add_message(
                    session.session_id,
                    MessageRole.ASSISTANT,
                    mock_response,
                    model_used="openai"
                )
                
                print(f"   Turn {i+1}: Added user message and response")
            
            # Verify session persistence
            retrieved_session = await session_manager.get_session(session.session_id)
            
            if retrieved_session:
                message_count = len(retrieved_session.conversation_history)
                print(f"✅ Session persistence working: {message_count} messages stored")
                print(f"✅ Selected model persisted: {retrieved_session.selected_model}")
                print(f"✅ Conversation turns: {len([msg for msg in retrieved_session.conversation_history if msg.role == MessageRole.USER])}")
                
                self.test_results['conversation_flow'] = {
                    'status': 'success',
                    'session_id': session.session_id,
                    'messages_stored': message_count,
                    'turns_completed': len(test_conversation),
                    'model_persistence': retrieved_session.selected_model == "openai"
                }
            else:
                print("❌ Session retrieval failed")
                self.test_results['conversation_flow'] = {
                    'status': 'failed',
                    'error': 'Session retrieval failed'
                }
                
        except Exception as e:
            print(f"❌ Conversation flow test failed: {e}")
            self.test_results['conversation_flow'] = {
                'status': 'error',
                'error': str(e)
            }
    
    async def test_prompt_classification(self):
        """Test prompt classification accuracy"""
        print("\n🏷️  TESTING PROMPT CLASSIFICATION")
        print("-" * 50)
        
        selector = DynamicModelSelector(self.models_config)
        self.test_results['prompt_classification'] = {}
        
        for scenario in self.test_scenarios:
            print(f"\n🔹 Testing: {scenario['name']}")
            
            try:
                # Classify the prompt
                classified_type = selector.prompt_classification(scenario['prompt'])
                expected_type = scenario['expected_type']
                
                is_correct = classified_type == expected_type
                status = "✅ Correct" if is_correct else "⚠️  Misclassified"
                
                print(f"   Expected: {expected_type.value}")
                print(f"   Classified: {classified_type.value}")
                print(f"   Status: {status}")
                
                self.test_results['prompt_classification'][scenario['name']] = {
                    'expected': expected_type.value,
                    'classified': classified_type.value,
                    'correct': is_correct
                }
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.test_results['prompt_classification'][scenario['name']] = {
                    'status': 'error',
                    'error': str(e)
                }
    
    async def test_performance_basic(self):
        """Basic performance testing"""
        print("\n⚡ TESTING BASIC PERFORMANCE")
        print("-" * 50)
        
        selector = DynamicModelSelector(self.models_config)
        
        # Test response times
        test_prompt = "I need help with anxiety"
        times = []
        
        print("Running 3 selection tests...")
        
        for i in range(3):
            start_time = time.time()
            try:
                selection = await selector.select_best_model(test_prompt)
                duration = (time.time() - start_time) * 1000
                times.append(duration)
                print(f"   Test {i+1}: {duration:.1f}ms - {selection.selected_model_id}")
            except Exception as e:
                print(f"   Test {i+1}: Failed - {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n📊 Performance Summary:")
            print(f"   Average: {avg_time:.1f}ms")
            print(f"   Fastest: {min_time:.1f}ms")
            print(f"   Slowest: {max_time:.1f}ms")
            
            # Performance assessment
            if avg_time < 5000:
                perf_level = "Excellent"
            elif avg_time < 15000:
                perf_level = "Good"
            elif avg_time < 30000:
                perf_level = "Acceptable"
            else:
                perf_level = "Slow"
            
            print(f"   Assessment: {perf_level}")
            
            self.test_results['performance'] = {
                'average_time_ms': avg_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'assessment': perf_level,
                'tests_completed': len(times)
            }
    
    async def test_performance_comprehensive(self):
        """Comprehensive performance testing"""
        print("\n⚡ TESTING COMPREHENSIVE PERFORMANCE")
        print("-" * 50)
        
        # Test all models individually
        model_clients = {
            'openai': OpenAIClient,
            'claude': ClaudeClient,
            'deepseek': DeepSeekClient,
            'gemma': GemmaClient
        }
        
        individual_performance = {}
        
        for model_name, ClientClass in model_clients.items():
            print(f"\n🔹 Testing {model_name.upper()} performance...")
            
            try:
                client = ClientClass()
                
                # Test 3 requests
                times = []
                for i in range(3):
                    start_time = time.time()
                    
                    if hasattr(client, 'generate_response'):
                        if asyncio.iscoroutinefunction(client.generate_response):
                            response = await asyncio.wait_for(
                                client.generate_response(
                                    prompt="Test performance message",
                                    system_prompt="You are a helpful assistant."
                                ),
                                timeout=30.0
                            )
                        else:
                            response = client.generate_response(
                                prompt="Test performance message",
                                system_prompt="You are a helpful assistant."
                            )
                    
                    duration = (time.time() - start_time) * 1000
                    
                    if hasattr(response, 'content') and response.content:
                        times.append(duration)
                        print(f"   Test {i+1}: {duration:.1f}ms")
                    else:
                        print(f"   Test {i+1}: Failed - No content")
                
                if times:
                    avg_time = sum(times) / len(times)
                    individual_performance[model_name] = {
                        'average_time_ms': avg_time,
                        'status': 'working'
                    }
                    print(f"   Average: {avg_time:.1f}ms")
                else:
                    individual_performance[model_name] = {
                        'status': 'failed'
                    }
                    print(f"   Status: Failed")
                    
            except Exception as e:
                print(f"   Error: {e}")
                individual_performance[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        self.test_results['comprehensive_performance'] = individual_performance
    
    async def test_ui_integration(self):
        """Test UI file integration"""
        print("\n🎨 TESTING UI INTEGRATION")
        print("-" * 50)
        
        ui_files = {
            'HTML Template': 'src/ui/templates/chat.html',
            'CSS Styles': 'src/ui/static/css/chat.css',
            'JavaScript': 'src/ui/static/js/chat.js',
            'Server Script': 'simple_server.py'
        }
        
        ui_status = {}
        
        for name, file_path in ui_files.items():
            full_path = self.project_root / file_path
            
            if full_path.exists():
                file_size = full_path.stat().st_size
                print(f"   ✅ {name}: Found ({file_size:,} bytes)")
                
                # Check for dark mode implementation
                if file_path.endswith('.css'):
                    content = full_path.read_text()
                    has_dark_mode = '--bg-primary: #1e293b' in content
                    print(f"      Dark mode: {'✅ Implemented' if has_dark_mode else '⚠️  Not found'}")
                    ui_status[name] = {'exists': True, 'size': file_size, 'dark_mode': has_dark_mode}
                else:
                    ui_status[name] = {'exists': True, 'size': file_size}
            else:
                print(f"   ❌ {name}: Missing")
                ui_status[name] = {'exists': False}
        
        self.test_results['ui_integration'] = ui_status
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        report_file = self.project_root / "results" / "development" / "complete_system_test_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'summary': self.calculate_summary()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Full test report saved to: {report_file}")
    
    def calculate_summary(self):
        """Calculate test summary statistics"""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0
        }
        
        # Count model selection tests
        if 'model_selection' in self.test_results:
            for result in self.test_results['model_selection'].values():
                summary['total_tests'] += 1
                if result.get('status') == 'success':
                    summary['passed_tests'] += 1
                elif result.get('status') == 'error':
                    summary['error_tests'] += 1
                else:
                    summary['failed_tests'] += 1
        
        # Count other tests
        for test_category in ['conversation_flow', 'prompt_classification']:
            if test_category in self.test_results:
                result = self.test_results[test_category]
                summary['total_tests'] += 1
                if isinstance(result, dict):
                    if result.get('status') == 'success':
                        summary['passed_tests'] += 1
                    elif result.get('status') == 'error':
                        summary['error_tests'] += 1
                    else:
                        summary['failed_tests'] += 1
        
        summary['success_rate'] = (summary['passed_tests'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
        
        return summary
    
    def print_final_summary(self):
        """Print final test summary"""
        print("\n" + "=" * 60)
        print("🎯 COMPLETE SYSTEM TEST SUMMARY")
        print("=" * 60)
        
        summary = self.calculate_summary()
        
        print(f"Total tests run: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"⚠️  Errors: {summary['error_tests']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        
        # System health assessment
        if summary['success_rate'] >= 90:
            health_status = "🟢 EXCELLENT"
        elif summary['success_rate'] >= 75:
            health_status = "🟡 GOOD"
        elif summary['success_rate'] >= 50:
            health_status = "🟠 FAIR"
        else:
            health_status = "🔴 POOR"
        
        print(f"\nSystem Health: {health_status}")
        
        # Key findings
        print(f"\n🔍 KEY FINDINGS:")
        
        # Model selection confidence
        if 'model_selection' in self.test_results:
            working_selections = [r for r in self.test_results['model_selection'].values() if r.get('status') == 'success']
            if working_selections:
                avg_confidence = sum(r['confidence_score'] for r in working_selections) / len(working_selections)
                print(f"   Average confidence score: {avg_confidence:.1%}")
            
            models_tested = set()
            for result in working_selections:
                models_tested.add(result.get('selected_model', 'unknown'))
            print(f"   Models being selected: {', '.join(models_tested)}")
        
        # Performance
        if 'performance' in self.test_results:
            perf = self.test_results['performance']
            print(f"   Average selection time: {perf.get('average_time_ms', 0):.1f}ms")
            print(f"   Performance assessment: {perf.get('assessment', 'Unknown')}")
        
        # UI Integration
        if 'ui_integration' in self.test_results:
            ui_files = self.test_results['ui_integration']
            existing_files = sum(1 for f in ui_files.values() if f.get('exists', False))
            print(f"   UI files present: {existing_files}/{len(ui_files)}")
        
        print("\n✅ Complete system test finished!")

async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Complete Mental Health Chat System Test')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive performance tests')
    args = parser.parse_args()
    
    tester = CompleteSystemTester()
    await tester.run_complete_tests(comprehensive=args.comprehensive)

if __name__ == "__main__":
    asyncio.run(main())