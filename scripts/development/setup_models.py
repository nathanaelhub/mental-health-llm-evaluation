#!/usr/bin/env python3
"""
Simple Model Setup and Configuration Checker
===========================================

Checks model configuration and provides setup guidance.
Non-interactive, works in any environment.

Usage:
    python scripts/setup_models.py              # Check current config
    python scripts/setup_models.py --check      # Same as above
    python scripts/setup_models.py --help       # Show setup guidance

Features:
- Validates API keys and environment variables
- Tests model connectivity
- Provides clear setup instructions
- Non-interactive (safe for automation)
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class ModelConfigChecker:
    """Simple model configuration checker."""
    
    def __init__(self):
        """Initialize checker."""
        load_dotenv()
        self.env_file = Path(".env")
        
        self.models_config = {
            'openai': {
                'name': 'OpenAI GPT-4',
                'emoji': '🌐',
                'required_vars': ['OPENAI_API_KEY'],
                'optional_vars': ['OPENAI_MODEL'],
                'default_model': 'gpt-4',
                'setup_url': 'https://platform.openai.com/api-keys'
            },
            'claude': {
                'name': 'Anthropic Claude',
                'emoji': '🤖',
                'required_vars': ['ANTHROPIC_API_KEY'],
                'optional_vars': ['CLAUDE_MODEL'],
                'default_model': 'claude-3-sonnet-20240229',
                'setup_url': 'https://console.anthropic.com/'
            },
            'deepseek': {
                'name': 'DeepSeek (Local)',
                'emoji': '🏠',
                'required_vars': ['LOCAL_LLM_BASE_URL'],
                'optional_vars': ['LOCAL_LLM_MODEL', 'LOCAL_LLM_TIMEOUT'],
                'default_model': 'deepseek-r1',
                'default_url': 'http://192.168.86.30:1234/v1',
                'setup_url': 'https://lmstudio.ai/'
            },
            'gemma': {
                'name': 'Google Gemma',
                'emoji': '💎',
                'required_vars': ['GOOGLE_API_KEY'],
                'optional_vars': ['GEMMA_MODEL'],
                'default_model': 'gemini-pro',
                'setup_url': 'https://makersuite.google.com/app/apikey'
            }
        }
    
    def check_environment_file(self) -> Tuple[bool, str]:
        """Check if .env file exists and is readable."""
        if not self.env_file.exists():
            return False, f"❌ .env file not found at {self.env_file.absolute()}"
        
        try:
            with open(self.env_file, 'r') as f:
                content = f.read()
                if not content.strip():
                    return False, "❌ .env file is empty"
            return True, f"✅ .env file found at {self.env_file.absolute()}"
        except Exception as e:
            return False, f"❌ Cannot read .env file: {e}"
    
    def check_model_config(self, model_key: str) -> Dict[str, any]:
        """Check configuration for a specific model."""
        config = self.models_config[model_key]
        result = {
            'name': config['name'],
            'emoji': config['emoji'],
            'configured': False,
            'missing_vars': [],
            'present_vars': [],
            'issues': []
        }
        
        # Check required variables
        for var in config['required_vars']:
            value = os.getenv(var)
            if value and value.strip() and value not in ['your_key_here', 'not_required']:
                result['present_vars'].append(var)
                
                # Special validation for URLs
                if 'URL' in var and not value.startswith('http'):
                    result['issues'].append(f"{var} should start with http:// or https://")
            else:
                result['missing_vars'].append(var)
        
        # Check optional variables
        for var in config['optional_vars']:
            value = os.getenv(var)
            if value and value.strip():
                result['present_vars'].append(var)
        
        result['configured'] = len(result['missing_vars']) == 0
        return result
    
    async def test_model_connection(self, model_key: str) -> Tuple[bool, str]:
        """Test if model is actually working."""
        try:
            if model_key == 'openai':
                return await self._test_openai()
            elif model_key == 'claude':
                return await self._test_claude()
            elif model_key == 'deepseek':
                return await self._test_deepseek()
            elif model_key == 'gemma':
                return await self._test_gemma()
            else:
                return False, "Unknown model type"
        except Exception as e:
            return False, str(e)
    
    async def _test_openai(self) -> Tuple[bool, str]:
        """Test OpenAI connection."""
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            
            if response.choices and response.choices[0].message.content:
                return True, "Connection successful"
            else:
                return False, "No response content"
        except Exception as e:
            return False, str(e)
    
    async def _test_claude(self) -> Tuple[bool, str]:
        """Test Claude connection."""
        try:
            from src.models.claude_client import ClaudeClient
            client = ClaudeClient()
            
            response = await client.generate_response(
                prompt="Test",
                max_tokens=5
            )
            
            if response and response.content:
                return True, "Connection successful"
            else:
                return False, "No response content"
        except ImportError:
            return False, "Claude client not available"
        except Exception as e:
            return False, str(e)
    
    async def _test_deepseek(self) -> Tuple[bool, str]:
        """Test DeepSeek local connection."""
        try:
            from src.models.local_llm_client import LocalLLMClient
            
            base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://192.168.86.30:1234/v1")
            model_name = os.getenv("LOCAL_LLM_MODEL", "deepseek-r1")
            
            client = LocalLLMClient(
                base_url=base_url,
                model_name=model_name,
                timeout=10
            )
            
            response = await client.generate_response("Test", max_tokens=5)
            await client.close()
            
            if response and response.content:
                return True, "Connection successful"
            else:
                return False, "No response content"
        except Exception as e:
            return False, str(e)
    
    async def _test_gemma(self) -> Tuple[bool, str]:
        """Test Gemma connection."""
        try:
            from src.models.gemma_client import GemmaClient
            client = GemmaClient()
            
            response = await client.generate_response(
                prompt="Test",
                max_tokens=5
            )
            
            if response and response.content:
                return True, "Connection successful"
            else:
                return False, "No response content"
        except ImportError:
            return False, "Gemma client not available"
        except Exception as e:
            return False, str(e)
    
    def print_setup_instructions(self, model_key: str):
        """Print setup instructions for a model."""
        config = self.models_config[model_key]
        
        print(f"\n{config['emoji']} {config['name']} Setup:")
        print(f"   1. Get API key from: {config['setup_url']}")
        
        if model_key == 'deepseek':
            print(f"   2. Start local LLM server (LM Studio, Ollama, etc.)")
            print(f"   3. Add to .env file:")
            print(f"      LOCAL_LLM_BASE_URL={config['default_url']}")
            print(f"      LOCAL_LLM_MODEL={config['default_model']}")
        else:
            print(f"   2. Add to .env file:")
            for var in config['required_vars']:
                if 'API_KEY' in var:
                    print(f"      {var}=your_actual_api_key_here")
                else:
                    print(f"      {var}=appropriate_value")
            
            for var in config['optional_vars']:
                if 'MODEL' in var:
                    print(f"      {var}={config['default_model']}  # Optional")
    
    async def run_check(self, test_connections: bool = False):
        """Run configuration check."""
        print("🔧 Model Configuration Checker")
        print("=" * 40)
        
        # Check .env file
        env_exists, env_message = self.check_environment_file()
        print(f"\n📁 Environment File:")
        print(f"   {env_message}")
        
        if not env_exists:
            self.print_env_setup_instructions()
            return
        
        print(f"\n🤖 Model Configuration Status:")
        
        configured_models = []
        total_models = len(self.models_config)
        
        # Check each model
        for model_key, model_config in self.models_config.items():
            config_result = self.check_model_config(model_key)
            
            print(f"\n   {config_result['emoji']} {config_result['name']}:")
            
            if config_result['configured']:
                print(f"      ✅ Configured")
                configured_models.append(model_key)
                
                if config_result['present_vars']:
                    print(f"      📋 Variables: {', '.join(config_result['present_vars'])}")
                
                if config_result['issues']:
                    for issue in config_result['issues']:
                        print(f"      ⚠️  {issue}")
                
                # Test connection if requested
                if test_connections:
                    print(f"      🔄 Testing connection...")
                    success, message = await self.test_model_connection(model_key)
                    if success:
                        print(f"      ✅ Connection test: {message}")
                    else:
                        print(f"      ❌ Connection test: {message}")
            
            else:
                print(f"      ❌ Not configured")
                if config_result['missing_vars']:
                    print(f"      📋 Missing: {', '.join(config_result['missing_vars'])}")
                
                if not test_connections:  # Only show setup instructions if not testing
                    self.print_setup_instructions(model_key)
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Models configured: {len(configured_models)}/{total_models}")
        
        if configured_models:
            print(f"   Ready to use: {', '.join(configured_models)}")
            print(f"\n✅ You can now run:")
            print(f"   python scripts/compare_models.py \"test prompt\" --models {','.join(configured_models)}")
            print(f"   python scripts/test_all_models.py --models {','.join(configured_models)}")
        else:
            print(f"   No models configured yet.")
            print(f"\n💡 Set up at least one model to start comparing responses!")
    
    def print_env_setup_instructions(self):
        """Print .env file setup instructions."""
        print(f"\n💡 .env File Setup:")
        print(f"   1. Copy the example file:")
        print(f"      cp .env.example .env")
        print(f"   2. Edit .env with your API keys:")
        print(f"      nano .env")
        print(f"   3. Re-run this script to verify configuration")


def print_full_setup_guide():
    """Print comprehensive setup guide."""
    print("""
🚀 Mental Health LLM Evaluation - Setup Guide
=============================================

This tool compares AI models for mental health applications.

QUICK START:
1. Copy environment template:
   cp .env.example .env

2. Edit .env file with your API keys:
   nano .env

3. Test configuration:
   python scripts/setup_models.py --check

4. Test model connections:
   python scripts/test_all_models.py

5. Compare models:
   python scripts/compare_models.py "How can I manage stress?"

MODEL PROVIDERS:
🌐 OpenAI (GPT-4)        - Get key: https://platform.openai.com/api-keys  
🤖 Anthropic (Claude)    - Get key: https://console.anthropic.com/
🏠 DeepSeek (Local)      - Setup: https://lmstudio.ai/ (free, private)
💎 Google (Gemini)       - Get key: https://makersuite.google.com/app/apikey

EXAMPLE .env FILE:
# Cloud API Keys (get from provider websites)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-claude-key-here
GOOGLE_API_KEY=your-google-key-here

# Local Models (requires local LLM server)  
LOCAL_LLM_BASE_URL=http://192.168.86.30:1234/v1
LOCAL_LLM_MODEL=deepseek-r1
LOCAL_LLM_TIMEOUT=60

USAGE EXAMPLES:
# Check current configuration
python scripts/setup_models.py

# Test all models
python scripts/test_all_models.py  

# Compare specific models
python scripts/compare_models.py "Hello" --models openai,claude

# Interactive comparison
python scripts/compare_models.py --interactive

TROUBLESHOOTING:
- Missing API keys → Check .env file and provider websites
- Local model errors → Ensure LM Studio/Ollama is running
- Import errors → pip install -r requirements.txt
- Connection timeouts → Check network and URLs

Ready to start? Run: python scripts/setup_models.py --check
""")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Model Setup and Configuration Checker",
        epilog="""
Examples:
  %(prog)s                  # Check configuration
  %(prog)s --check          # Same as above  
  %(prog)s --test           # Check config + test connections
  %(prog)s --help-setup     # Show detailed setup guide
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--check", action="store_true", 
                       help="Check model configuration (default)")
    parser.add_argument("--test", action="store_true",
                       help="Check configuration and test connections")
    parser.add_argument("--help-setup", action="store_true",
                       help="Show detailed setup instructions")
    
    args = parser.parse_args()
    
    if args.help_setup:
        print_full_setup_guide()
        return
    
    # Default to check if no specific action
    if not args.test:
        args.check = True
    
    try:
        checker = ModelConfigChecker()
        
        if args.check or args.test:
            await checker.run_check(test_connections=args.test)
    
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        print(f"Debug info: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())