#!/usr/bin/env python3
"""
Chat Interface Debugging Script
===============================

Comprehensive debugging tool to identify issues with the chat interface:
1. Static file serving (CSS/JS)
2. JavaScript event handlers
3. Session management
4. API endpoint responses
5. Dark theme application
6. Button functionality

Usage:
    python scripts/debug_chat_interface.py
    python scripts/debug_chat_interface.py --comprehensive
"""

import sys
import os
import asyncio
import aiohttp
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

class ChatInterfaceDebugger:
    """Comprehensive debugging tool for the chat interface"""
    
    def __init__(self):
        self.project_root = project_root
        self.base_url = "http://localhost:8000"
        self.test_results = {}
        self.session = None
        
    async def run_all_tests(self, comprehensive: bool = False):
        """Run complete debugging suite"""
        print("🔍 CHAT INTERFACE DEBUGGING SUITE")
        print("=" * 60)
        print(f"Project: {self.project_root}")
        print(f"Base URL: {self.base_url}")
        print(f"Mode: {'Comprehensive' if comprehensive else 'Standard'}")
        print("=" * 60)
        
        # Initialize HTTP session
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Test 1: File System Checks
            await self.test_file_system()
            
            # Test 2: Server Status
            await self.test_server_status()
            
            # Test 3: Static File Serving
            await self.test_static_files()
            
            # Test 4: Template Serving
            await self.test_template_serving()
            
            # Test 5: API Endpoints
            await self.test_api_endpoints()
            
            # Test 6: Session Management
            await self.test_session_management()
            
            # Test 7: Conversation Flow
            await self.test_conversation_flow()
            
            if comprehensive:
                # Test 8: JavaScript Event Handlers (simulated)
                await self.test_javascript_handlers()
                
                # Test 9: Dark Theme CSS
                await self.test_dark_theme_css()
        
        # Generate final report
        self.generate_debug_report()
        self.print_recommendations()
    
    async def test_file_system(self):
        """Test if required files exist in correct locations"""
        print("\\n📁 TESTING FILE SYSTEM")
        print("-" * 40)
        
        required_files = {
            'HTML Template': 'src/ui/templates/chat.html',
            'CSS Styles': 'src/ui/static/css/chat.css',
            'JavaScript': 'src/ui/static/js/chat.js',
            'Main Server': 'simple_server.py',
            'Chat Server': 'chat_server.py'
        }
        
        file_results = {}
        
        for name, file_path in required_files.items():
            full_path = self.project_root / file_path
            exists = full_path.exists()
            
            if exists:
                size = full_path.stat().st_size
                modified = datetime.fromtimestamp(full_path.stat().st_mtime)
                
                print(f"✅ {name}: Found ({size:,} bytes, modified {modified.strftime('%Y-%m-%d %H:%M')})")
                
                file_results[name] = {
                    'exists': True,
                    'path': str(full_path),
                    'size': size,
                    'modified': modified.isoformat()
                }
                
                # Check for specific content
                if file_path.endswith('.css'):
                    content = full_path.read_text()
                    has_dark_vars = '--bg-primary: #1e293b' in content
                    has_bubbles = 'message-bubble' in content
                    print(f"   Dark mode variables: {'✅' if has_dark_vars else '❌'}")
                    print(f"   Chat bubble styles: {'✅' if has_bubbles else '❌'}")
                    file_results[name]['dark_mode'] = has_dark_vars
                    file_results[name]['chat_bubbles'] = has_bubbles
                    
                elif file_path.endswith('.js'):
                    content = full_path.read_text()
                    has_handlers = 'startNewConversation' in content
                    has_message_history = 'messageHistory' in content
                    print(f"   Event handlers: {'✅' if has_handlers else '❌'}")
                    print(f"   Message history: {'✅' if has_message_history else '❌'}")
                    file_results[name]['event_handlers'] = has_handlers
                    file_results[name]['message_history'] = has_message_history
                    
                elif file_path.endswith('.html'):
                    content = full_path.read_text()
                    has_history_container = 'conversation-history' in content
                    has_new_chat_btn = 'startNewConversation' in content
                    print(f"   History container: {'✅' if has_history_container else '❌'}")
                    print(f"   New chat button: {'✅' if has_new_chat_btn else '❌'}")
                    file_results[name]['history_container'] = has_history_container
                    file_results[name]['new_chat_button'] = has_new_chat_btn
            else:
                print(f"❌ {name}: Missing")
                file_results[name] = {'exists': False, 'path': str(full_path)}
        
        self.test_results['file_system'] = file_results
    
    async def test_server_status(self):
        """Test if server is running and responding"""
        print("\\n🚀 TESTING SERVER STATUS")
        print("-" * 40)
        
        try:
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/api/status") as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Server running: {data.get('status', 'unknown')}")
                    print(f"   Response time: {response_time:.1f}ms")
                    print(f"   Available models: {len(data.get('available_models', []))}")
                    print(f"   Uptime: {data.get('uptime_seconds', 0):.1f}s")
                    
                    self.test_results['server_status'] = {
                        'running': True,
                        'response_time_ms': response_time,
                        'status': data.get('status'),
                        'models': data.get('available_models', []),
                        'uptime': data.get('uptime_seconds', 0)
                    }
                else:
                    print(f"⚠️ Server responding with status {response.status}")
                    self.test_results['server_status'] = {
                        'running': False,
                        'status_code': response.status
                    }
        except Exception as e:
            print(f"❌ Server not responding: {e}")
            self.test_results['server_status'] = {
                'running': False,
                'error': str(e)
            }
    
    async def test_static_files(self):
        """Test static file serving (CSS, JS)"""
        print("\\n🎨 TESTING STATIC FILE SERVING")
        print("-" * 40)
        
        static_files = [
            '/static/css/chat.css',
            '/static/js/chat.js'
        ]
        
        static_results = {}
        
        for file_path in static_files:
            try:
                start_time = time.time()
                async with self.session.get(f"{self.base_url}{file_path}") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        content = await response.text()
                        content_type = response.headers.get('content-type', '')
                        
                        print(f"✅ {file_path}")
                        print(f"   Status: {response.status}")
                        print(f"   Content-Type: {content_type}")
                        print(f"   Size: {len(content):,} characters")
                        print(f"   Response time: {response_time:.1f}ms")
                        
                        # Check for specific content
                        if file_path.endswith('.css'):
                            has_dark_theme = '--bg-primary' in content
                            has_animations = '@keyframes' in content
                            print(f"   Dark theme vars: {'✅' if has_dark_theme else '❌'}")
                            print(f"   Animations: {'✅' if has_animations else '❌'}")
                            
                        elif file_path.endswith('.js'):
                            has_chat_class = 'MentalHealthChat' in content
                            has_event_binding = 'addEventListener' in content
                            print(f"   Chat class: {'✅' if has_chat_class else '❌'}")
                            print(f"   Event binding: {'✅' if has_event_binding else '❌'}")
                        
                        static_results[file_path] = {
                            'status': response.status,
                            'content_type': content_type,
                            'size': len(content),
                            'response_time_ms': response_time,
                            'served': True
                        }
                    else:
                        print(f"❌ {file_path}: Status {response.status}")
                        static_results[file_path] = {
                            'status': response.status,
                            'served': False
                        }
                        
            except Exception as e:
                print(f"❌ {file_path}: Error - {e}")
                static_results[file_path] = {
                    'served': False,
                    'error': str(e)
                }
        
        self.test_results['static_files'] = static_results
    
    async def test_template_serving(self):
        """Test HTML template serving"""
        print("\\n📄 TESTING TEMPLATE SERVING")
        print("-" * 40)
        
        template_routes = [
            '/',
            '/chat'
        ]
        
        template_results = {}
        
        for route in template_routes:
            try:
                start_time = time.time()
                async with self.session.get(f"{self.base_url}{route}") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        content = await response.text()
                        content_type = response.headers.get('content-type', '')
                        
                        print(f"✅ {route}")
                        print(f"   Status: {response.status}")
                        print(f"   Content-Type: {content_type}")
                        print(f"   Size: {len(content):,} characters")
                        print(f"   Response time: {response_time:.1f}ms")
                        
                        # Check HTML content
                        if route == '/chat':
                            has_chat_container = 'chat-container' in content
                            has_js_includes = '/static/js/chat.js' in content
                            has_css_includes = '/static/css/chat.css' in content
                            has_history_div = 'conversation-history' in content
                            
                            print(f"   Chat container: {'✅' if has_chat_container else '❌'}")
                            print(f"   JS includes: {'✅' if has_js_includes else '❌'}")
                            print(f"   CSS includes: {'✅' if has_css_includes else '❌'}")
                            print(f"   History container: {'✅' if has_history_div else '❌'}")
                            
                            template_results[route] = {
                                'status': response.status,
                                'response_time_ms': response_time,
                                'has_container': has_chat_container,
                                'has_js': has_js_includes,
                                'has_css': has_css_includes,
                                'has_history': has_history_div
                            }
                        else:
                            template_results[route] = {
                                'status': response.status,
                                'response_time_ms': response_time,
                                'size': len(content)
                            }
                    else:
                        print(f"❌ {route}: Status {response.status}")
                        
            except Exception as e:
                print(f"❌ {route}: Error - {e}")
                template_results[route] = {'error': str(e)}
        
        self.test_results['templates'] = template_results
    
    async def test_api_endpoints(self):
        """Test API endpoint functionality"""
        print("\\n🔗 TESTING API ENDPOINTS")
        print("-" * 40)
        
        api_tests = [
            {'endpoint': '/api/status', 'method': 'GET'},
            {'endpoint': '/api/models/status', 'method': 'GET'},
            {'endpoint': '/api/health', 'method': 'GET'}
        ]
        
        api_results = {}
        
        for test in api_tests:
            endpoint = test['endpoint']
            method = test['method']
            
            try:
                start_time = time.time()
                
                if method == 'GET':
                    async with self.session.get(f"{self.base_url}{endpoint}") as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            print(f"✅ {endpoint}")
                            print(f"   Status: {response.status}")
                            print(f"   Response time: {response_time:.1f}ms")
                            print(f"   Keys: {list(data.keys()) if isinstance(data, dict) else 'Not dict'}")
                            
                            api_results[endpoint] = {
                                'status': response.status,
                                'response_time_ms': response_time,
                                'working': True,
                                'data_keys': list(data.keys()) if isinstance(data, dict) else None
                            }
                        else:
                            print(f"❌ {endpoint}: Status {response.status}")
                            api_results[endpoint] = {
                                'status': response.status,
                                'working': False
                            }
                            
            except Exception as e:
                print(f"❌ {endpoint}: Error - {e}")
                api_results[endpoint] = {
                    'working': False,
                    'error': str(e)
                }
        
        self.test_results['api_endpoints'] = api_results
    
    async def test_session_management(self):
        """Test session creation and management"""
        print("\\n💾 TESTING SESSION MANAGEMENT")
        print("-" * 40)
        
        # Test initial chat message (should create session)
        chat_payload = {
            "message": "I'm feeling anxious about work",
            "user_id": "debug-test",
            "session_id": None,
            "force_reselection": False
        }
        
        try:
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=chat_payload
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    print("✅ Initial chat message")
                    print(f"   Status: {response.status}")
                    print(f"   Response time: {response_time:.1f}ms")
                    print(f"   Session ID: {data.get('session_id', 'None')[:8]}...")
                    print(f"   Selected model: {data.get('selected_model', 'None')}")
                    print(f"   Confidence: {data.get('confidence_score', 0) * 100:.1f}%")
                    print(f"   Is new session: {data.get('is_new_session', False)}")
                    print(f"   Conversation mode: {data.get('conversation_mode', 'Unknown')}")
                    print(f"   Turn count: {data.get('turn_count', 0)}")
                    
                    session_id = data.get('session_id')
                    selected_model = data.get('selected_model')
                    
                    # Test follow-up message (should continue session)
                    if session_id:
                        followup_payload = {
                            "message": "What can I do about this anxiety?",
                            "user_id": "debug-test",
                            "session_id": session_id,
                            "force_reselection": False
                        }
                        
                        start_time = time.time()
                        async with self.session.post(
                            f"{self.base_url}/api/chat",
                            json=followup_payload
                        ) as followup_response:
                            followup_time = (time.time() - start_time) * 1000
                            
                            if followup_response.status == 200:
                                followup_data = await followup_response.json()
                                
                                print("\\n✅ Follow-up message")
                                print(f"   Status: {followup_response.status}")
                                print(f"   Response time: {followup_time:.1f}ms")
                                print(f"   Same session: {followup_data.get('session_id') == session_id}")
                                print(f"   Same model: {followup_data.get('selected_model') == selected_model}")
                                print(f"   Is new session: {followup_data.get('is_new_session', True)}")
                                print(f"   Conversation mode: {followup_data.get('conversation_mode', 'Unknown')}")
                                print(f"   Turn count: {followup_data.get('turn_count', 0)}")
                                
                                self.test_results['session_management'] = {
                                    'initial_message': {
                                        'working': True,
                                        'session_created': bool(session_id),
                                        'model_selected': bool(selected_model),
                                        'response_time_ms': response_time,
                                        'conversation_mode': data.get('conversation_mode')
                                    },
                                    'followup_message': {
                                        'working': True,
                                        'session_continued': followup_data.get('session_id') == session_id,
                                        'model_continued': followup_data.get('selected_model') == selected_model,
                                        'response_time_ms': followup_time,
                                        'conversation_mode': followup_data.get('conversation_mode')
                                    }
                                }
                            else:
                                print(f"❌ Follow-up message failed: Status {followup_response.status}")
                                self.test_results['session_management'] = {
                                    'initial_message': {'working': True},
                                    'followup_message': {'working': False, 'status': followup_response.status}
                                }
                    else:
                        print("⚠️ No session ID returned from initial message")
                        self.test_results['session_management'] = {
                            'initial_message': {'working': False, 'no_session_id': True}
                        }
                        
                else:
                    print(f"❌ Initial chat message failed: Status {response.status}")
                    self.test_results['session_management'] = {
                        'initial_message': {'working': False, 'status': response.status}
                    }
                    
        except Exception as e:
            print(f"❌ Session management test failed: {e}")
            self.test_results['session_management'] = {
                'working': False,
                'error': str(e)
            }
    
    async def test_conversation_flow(self):
        """Test complete conversation flow"""
        print("\\n💬 TESTING CONVERSATION FLOW")
        print("-" * 40)
        
        conversation_messages = [
            "I'm having trouble sleeping due to stress",
            "What relaxation techniques would you recommend?",
            "How long should I practice these techniques?",
            "Thank you for the helpful advice"
        ]
        
        session_id = None
        selected_model = None
        flow_results = []
        
        for i, message in enumerate(conversation_messages):
            payload = {
                "message": message,
                "user_id": "debug-flow-test",
                "session_id": session_id,
                "force_reselection": False
            }
            
            try:
                start_time = time.time()
                async with self.session.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Update session tracking
                        if not session_id:
                            session_id = data.get('session_id')
                            selected_model = data.get('selected_model')
                        
                        turn_mode = "Initial" if i == 0 else "Continued"
                        model_consistent = data.get('selected_model') == selected_model if selected_model else True
                        
                        print(f"✅ Message {i+1} ({turn_mode})")
                        print(f"   Message: '{message[:50]}{'...' if len(message) > 50 else ''}'")
                        print(f"   Response time: {response_time:.1f}ms")
                        print(f"   Model: {data.get('selected_model', 'Unknown')}")
                        print(f"   Model consistent: {'✅' if model_consistent else '❌'}")
                        print(f"   Turn count: {data.get('turn_count', 0)}")
                        print(f"   Mode: {data.get('conversation_mode', 'Unknown')}")
                        
                        flow_results.append({
                            'message_number': i + 1,
                            'message': message,
                            'working': True,
                            'response_time_ms': response_time,
                            'model': data.get('selected_model'),
                            'model_consistent': model_consistent,
                            'turn_count': data.get('turn_count'),
                            'conversation_mode': data.get('conversation_mode')
                        })
                        
                    else:
                        print(f"❌ Message {i+1} failed: Status {response.status}")
                        flow_results.append({
                            'message_number': i + 1,
                            'working': False,
                            'status': response.status
                        })
                        break
                        
            except Exception as e:
                print(f"❌ Message {i+1} error: {e}")
                flow_results.append({
                    'message_number': i + 1,
                    'working': False,
                    'error': str(e)
                })
                break
        
        self.test_results['conversation_flow'] = {
            'messages_tested': len(flow_results),
            'messages_successful': len([r for r in flow_results if r.get('working', False)]),
            'session_id': session_id,
            'selected_model': selected_model,
            'results': flow_results
        }
    
    async def test_javascript_handlers(self):
        """Test JavaScript event handlers (simulated)"""
        print("\\n⚡ TESTING JAVASCRIPT HANDLERS (Simulated)")
        print("-" * 40)
        
        # Check if the chat.js file has the required functions
        js_file = self.project_root / "src/ui/static/js/chat.js"
        
        if js_file.exists():
            content = js_file.read_text()
            
            required_functions = [
                'startNewConversation',
                'showSystemStatus',
                'showModelStatus', 
                'closeModal',
                'addMessage',
                'sendMessage'
            ]
            
            handler_results = {}
            
            for func_name in required_functions:
                if func_name in content:
                    print(f"✅ {func_name}: Found in JS")
                    handler_results[func_name] = True
                else:
                    print(f"❌ {func_name}: Missing from JS")
                    handler_results[func_name] = False
            
            # Check global window assignments
            global_assignments = [
                'window.startNewConversation',
                'window.showSystemStatus',
                'window.showModelStatus',
                'window.closeModal'
            ]
            
            for assignment in global_assignments:
                if assignment in content:
                    print(f"✅ {assignment}: Properly exposed")
                else:
                    print(f"❌ {assignment}: Not exposed globally")
            
            self.test_results['javascript_handlers'] = handler_results
        else:
            print("❌ chat.js file not found")
            self.test_results['javascript_handlers'] = {'file_missing': True}
    
    async def test_dark_theme_css(self):
        """Test dark theme CSS implementation"""
        print("\\n🌙 TESTING DARK THEME CSS")
        print("-" * 40)
        
        css_file = self.project_root / "src/ui/static/css/chat.css"
        
        if css_file.exists():
            content = css_file.read_text()
            
            # Check for essential dark theme elements
            dark_theme_checks = {
                'CSS Variables': '--bg-primary: #1e293b' in content,
                'Dark Background': 'background: linear-gradient' in content and '#0f172a' in content,
                'Text Colors': '--text-primary: #f1f5f9' in content,
                'Chat Bubbles': '.message-bubble' in content,
                'User Message Styling': '.user-message .message-bubble' in content,
                'Assistant Message Styling': '.assistant-message .message-bubble' in content,
                'Animations': '@keyframes' in content,
                'Responsive Design': '@media' in content
            }
            
            for check_name, result in dark_theme_checks.items():
                status = "✅" if result else "❌"
                print(f"{status} {check_name}: {'Present' if result else 'Missing'}")
            
            # Calculate overall theme completeness
            completeness = sum(dark_theme_checks.values()) / len(dark_theme_checks) * 100
            print(f"\\n📊 Dark theme completeness: {completeness:.1f}%")
            
            self.test_results['dark_theme'] = {
                'file_exists': True,
                'checks': dark_theme_checks,
                'completeness_percent': completeness
            }
        else:
            print("❌ chat.css file not found")
            self.test_results['dark_theme'] = {'file_exists': False}
    
    def generate_debug_report(self):
        """Generate comprehensive debug report"""
        report_file = self.project_root / "results" / "development" / "chat_interface_debug_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'base_url': self.base_url,
            'test_results': self.test_results,
            'summary': self.calculate_summary()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\\n📋 Debug report saved to: {report_file}")
    
    def calculate_summary(self):
        """Calculate summary statistics"""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'warnings': []
        }
        
        # File system checks
        if 'file_system' in self.test_results:
            for file_info in self.test_results['file_system'].values():
                summary['total_tests'] += 1
                if file_info.get('exists', False):
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                    summary['warnings'].append(f"Missing file: {file_info.get('path', 'Unknown')}")
        
        # Server status
        if 'server_status' in self.test_results:
            summary['total_tests'] += 1
            if self.test_results['server_status'].get('running', False):
                summary['passed_tests'] += 1
            else:
                summary['failed_tests'] += 1
                summary['warnings'].append("Server not responding")
        
        # Static files
        if 'static_files' in self.test_results:
            for file_path, file_info in self.test_results['static_files'].items():
                summary['total_tests'] += 1
                if file_info.get('served', False):
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                    summary['warnings'].append(f"Static file not served: {file_path}")
        
        # Session management
        if 'session_management' in self.test_results:
            session_info = self.test_results['session_management']
            if 'initial_message' in session_info:
                summary['total_tests'] += 1
                if session_info['initial_message'].get('working', False):
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                    summary['warnings'].append("Initial message/session creation failed")
            
            if 'followup_message' in session_info:
                summary['total_tests'] += 1
                if session_info['followup_message'].get('working', False):
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                    summary['warnings'].append("Follow-up message/session continuation failed")
        
        summary['success_rate'] = (summary['passed_tests'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
        
        return summary
    
    def print_recommendations(self):
        """Print actionable recommendations"""
        print("\\n" + "=" * 60)
        print("🎯 DEBUGGING RECOMMENDATIONS")
        print("=" * 60)
        
        summary = self.calculate_summary()
        
        print(f"Overall Success Rate: {summary['success_rate']:.1f}%")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        
        if summary['warnings']:
            print("\\n⚠️ Issues Found:")
            for i, warning in enumerate(summary['warnings'], 1):
                print(f"   {i}. {warning}")
        
        # Specific recommendations based on test results
        print("\\n💡 Recommendations:")
        
        # Server recommendations
        if not self.test_results.get('server_status', {}).get('running', False):
            print("   🚀 Start the server:")
            print("      python chat_server.py")
            print("      # or")
            print("      python start_server.py")
        
        # Static file recommendations
        static_issues = [k for k, v in self.test_results.get('static_files', {}).items() if not v.get('served', False)]
        if static_issues:
            print("   📁 Fix static file serving:")
            print("      - Check if src/ui/static directory exists")
            print("      - Verify FastAPI StaticFiles mount is correct")
            print("      - Check file permissions")
        
        # Template recommendations
        if '/chat' not in self.test_results.get('templates', {}):
            print("   📄 Add /chat route:")
            print("      - Implement chat_interface() route in server")
            print("      - Mount Jinja2Templates correctly")
            print("      - Verify chat.html template exists")
        
        # Session management recommendations
        session_info = self.test_results.get('session_management', {})
        if not session_info.get('initial_message', {}).get('working', False):
            print("   💾 Fix session management:")
            print("      - Check ConversationSessionManager initialization")
            print("      - Verify model selector is working")
            print("      - Check API endpoint /api/chat")
        
        # JavaScript recommendations
        js_info = self.test_results.get('javascript_handlers', {})
        if js_info and not all(js_info.values()):
            print("   ⚡ Fix JavaScript handlers:")
            print("      - Add missing functions to chat.js")
            print("      - Ensure global window assignments")
            print("      - Check event.preventDefault() in onclick handlers")
        
        # Dark theme recommendations
        theme_info = self.test_results.get('dark_theme', {})
        if theme_info and theme_info.get('completeness_percent', 0) < 80:
            print("   🌙 Improve dark theme:")
            print("      - Add missing CSS variables")
            print("      - Implement chat bubble styles")
            print("      - Add proper animations")
        
        print("\\n✅ Quick Fix Script:")
        print("   # Test the interface manually:")
        print("   1. python chat_server.py")
        print("   2. Open http://localhost:8000/chat")
        print("   3. Check browser console for JS errors")
        print("   4. Test conversation flow:")
        print("      - Send first message")
        print("      - Send follow-up message")
        print("      - Click 'New Chat' button")

async def main():
    """Main debugging function"""
    parser = argparse.ArgumentParser(description='Chat Interface Debugging Tool')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive tests including JS and CSS analysis')
    args = parser.parse_args()
    
    debugger = ChatInterfaceDebugger()
    await debugger.run_all_tests(comprehensive=args.comprehensive)

if __name__ == "__main__":
    asyncio.run(main())