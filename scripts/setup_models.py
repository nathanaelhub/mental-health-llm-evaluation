#!/usr/bin/env python3
"""
Model Setup Wizard - Interactive Configuration for All Model Providers
=====================================================================

This script helps users configure all 4 model providers:
- OpenAI GPT-4
- Anthropic Claude 3 
- DeepSeek (Local)
- Google Gemini

Features:
- Interactive setup wizard
- API key validation
- Connectivity testing
- Cost estimation
- Configuration summary

Usage:
    python setup_models.py
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
from datetime import datetime

# Rich imports for enhanced UI
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.spinner import Spinner
from rich.live import Live
from rich import box
from rich.columns import Columns

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

console = Console()

class ModelSetupWizard:
    """Interactive wizard for setting up all model providers"""
    
    # Model provider information
    MODEL_PROVIDERS = {
        "openai": {
            "name": "OpenAI GPT-4",
            "icon": "🌐",
            "color": "blue",
            "api_key_env": "OPENAI_API_KEY",
            "model_env": "OPENAI_MODEL",
            "default_model": "gpt-4-turbo-preview",
            "signup_url": "https://platform.openai.com/api-keys",
            "pricing": {
                "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
            },
            "description": "Most popular, reliable, good for general use"
        },
        "claude": {
            "name": "Anthropic Claude 3",
            "icon": "🤖",
            "color": "magenta",
            "api_key_env": "ANTHROPIC_API_KEY",
            "model_env": "CLAUDE_MODEL",
            "default_model": "claude-3-opus-20240229",
            "signup_url": "https://console.anthropic.com/",
            "pricing": {
                "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
                "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
            },
            "description": "Excellent for reasoning, analysis, and safety"
        },
        "deepseek": {
            "name": "DeepSeek R1 (Local)",
            "icon": "🏠",
            "color": "green",
            "api_key_env": "LOCAL_LLM_BASE_URL",
            "model_env": "LOCAL_LLM_MODEL",
            "default_model": "deepseek/deepseek-r1-0528-qwen3-8b",
            "default_url": "http://192.168.86.23:1234/v1",
            "signup_url": "https://github.com/ggerganov/llama.cpp",
            "pricing": {"deepseek": {"input": 0.0, "output": 0.0}},
            "description": "Free local model, private, no API costs"
        },
        "gemini": {
            "name": "Google Gemini Pro",
            "icon": "💎",
            "color": "yellow",
            "api_key_env": "GOOGLE_API_KEY",
            "model_env": "GEMINI_MODEL",
            "default_model": "gemini-pro",
            "signup_url": "https://makersuite.google.com/app/apikey",
            "pricing": {
                "gemini-pro": {"input": 0.0005, "output": 0.0015},
                "gemini-1.5-pro": {"input": 0.001, "output": 0.002}
            },
            "description": "Cost-effective, good for basic tasks"
        }
    }
    
    def __init__(self):
        self.env_file_path = Path(".env")
        self.env_data = {}
        self.test_results = {}
        
    def load_existing_env(self) -> Dict[str, str]:
        """Load existing .env file if it exists"""
        env_data = {}
        
        if self.env_file_path.exists():
            try:
                with open(self.env_file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_data[key.strip()] = value.strip()
            except Exception as e:
                console.print(f"[yellow]Warning: Could not read .env file: {e}[/yellow]")
        
        return env_data
    
    def save_env_file(self):
        """Save configuration to .env file"""
        try:
            # Read existing content to preserve comments and structure
            existing_content = ""
            if self.env_file_path.exists():
                with open(self.env_file_path, 'r') as f:
                    existing_content = f.read()
            
            # Update or add new values
            lines = existing_content.split('\n') if existing_content else []
            updated_keys = set()
            
            for i, line in enumerate(lines):
                if '=' in line and not line.strip().startswith('#'):
                    key = line.split('=')[0].strip()
                    if key in self.env_data:
                        lines[i] = f"{key}={self.env_data[key]}"
                        updated_keys.add(key)
            
            # Add new keys that weren't in the file
            for key, value in self.env_data.items():
                if key not in updated_keys:
                    lines.append(f"{key}={value}")
            
            # Write back to file
            with open(self.env_file_path, 'w') as f:
                f.write('\n'.join(lines))
                
            console.print(f"[green]✅ Configuration saved to {self.env_file_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to save .env file: {e}[/red]")
    
    def display_welcome(self):
        """Display welcome message and overview"""
        welcome_panel = Panel.fit(
            "[bold blue]🚀 Model Setup Wizard[/bold blue]\n\n"
            "This wizard will help you configure all 4 model providers:\n"
            "🌐 [blue]OpenAI GPT-4[/blue] - Most popular, reliable\n"
            "🤖 [magenta]Anthropic Claude 3[/magenta] - Excellent reasoning\n"
            "🏠 [green]DeepSeek R1 (Local)[/green] - Free, private\n"
            "💎 [yellow]Google Gemini Pro[/yellow] - Cost-effective\n\n"
            "[dim]We'll test each API key and show you cost estimates.[/dim]",
            title="Welcome",
            border_style="blue"
        )
        console.print(welcome_panel)
        console.print()
    
    def get_provider_info_panel(self, provider_key: str) -> Panel:
        """Create an information panel for a provider"""
        provider = self.MODEL_PROVIDERS[provider_key]
        
        # Format pricing info
        pricing_info = []
        for model, price in provider["pricing"].items():
            if price["input"] == 0 and price["output"] == 0:
                pricing_info.append(f"  • {model}: [green]FREE[/green]")
            else:
                pricing_info.append(f"  • {model}: ${price['input']:.4f}/${price['output']:.4f} per 1K tokens")
        
        pricing_text = "\n".join(pricing_info)
        
        content = f"""[bold]{provider['icon']} {provider['name']}[/bold]

[dim]{provider['description']}[/dim]

[bold]Pricing:[/bold]
{pricing_text}

[bold]Get API Key:[/bold] {provider['signup_url']}"""
        
        return Panel(
            content,
            title=f"[{provider['color']}]{provider['name']}[/{provider['color']}]",
            border_style=provider['color'],
            expand=False
        )
    
    async def test_api_key(self, provider_key: str, api_key: str, endpoint: Optional[str] = None) -> Tuple[bool, str, float]:
        """Test an API key by making a simple request"""
        provider = self.MODEL_PROVIDERS[provider_key]
        
        try:
            if provider_key == "openai":
                return await self._test_openai(api_key)
            elif provider_key == "claude":
                return await self._test_claude(api_key)
            elif provider_key == "deepseek":
                return await self._test_deepseek(endpoint or provider["default_url"])
            elif provider_key == "gemini":
                return await self._test_gemini(api_key)
            else:
                return False, "Unknown provider", 0.0
                
        except Exception as e:
            return False, str(e), 0.0
    
    async def _test_openai(self, api_key: str) -> Tuple[bool, str, float]:
        """Test OpenAI API key"""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            start_time = time.time()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for testing
                messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
                max_tokens=10
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.choices and response.choices[0].message.content:
                return True, response.choices[0].message.content.strip(), response_time
            else:
                return False, "No response content", response_time
                
        except Exception as e:
            return False, str(e), 0.0
    
    async def _test_claude(self, api_key: str) -> Tuple[bool, str, float]:
        """Test Anthropic Claude API key"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            start_time = time.time()
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Use cheaper model for testing
                max_tokens=10,
                messages=[{"role": "user", "content": "Say hello in exactly 3 words."}]
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.content and len(response.content) > 0:
                return True, response.content[0].text.strip(), response_time
            else:
                return False, "No response content", response_time
                
        except Exception as e:
            return False, str(e), 0.0
    
    async def _test_deepseek(self, endpoint: str) -> Tuple[bool, str, float]:
        """Test local DeepSeek endpoint"""
        try:
            from src.models.local_llm_client import LocalLLMClient
            
            client = LocalLLMClient(base_url=endpoint, model_name="deepseek", timeout=10)
            
            start_time = time.time()
            response = await client.generate_response("Say hello in exactly 3 words.")
            response_time = (time.time() - start_time) * 1000
            
            if response and response.content:
                return True, response.content.strip(), response_time
            else:
                return False, "No response content", response_time
                
        except Exception as e:
            return False, str(e), 0.0
    
    async def _test_gemini(self, api_key: str) -> Tuple[bool, str, float]:
        """Test Google Gemini API key"""
        try:
            from google import generativeai as genai
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            start_time = time.time()
            response = model.generate_content("Say hello in exactly 3 words.")
            response_time = (time.time() - start_time) * 1000
            
            if response and response.text:
                return True, response.text.strip(), response_time
            else:
                return False, "No response content", response_time
                
        except Exception as e:
            return False, str(e), 0.0
    
    async def setup_provider(self, provider_key: str) -> bool:
        """Interactive setup for a single provider"""
        provider = self.MODEL_PROVIDERS[provider_key]
        
        console.print()
        console.rule(f"[{provider['color']}]{provider['icon']} {provider['name']}[/{provider['color']}]")
        console.print()
        
        # Show provider info
        info_panel = self.get_provider_info_panel(provider_key)
        console.print(info_panel)
        console.print()
        
        # Check if already configured
        existing_key = self.env_data.get(provider['api_key_env'])
        if existing_key and existing_key not in ['your_key_here', 'not_required']:
            if Confirm.ask(f"[yellow]Found existing configuration. Test current setup?[/yellow]"):
                success, message, response_time = await self._test_with_spinner(provider_key, existing_key)
                if success:
                    console.print(f"[green]✅ Existing configuration works! Response: {message} ({response_time:.0f}ms)[/green]")
                    return True
                else:
                    console.print(f"[red]❌ Existing configuration failed: {message}[/red]")
        
        # Special handling for DeepSeek (local endpoint)
        if provider_key == "deepseek":
            return await self._setup_deepseek()
        
        # Regular API key setup
        if not Confirm.ask(f"[{provider['color']}]Set up {provider['name']}?[/{provider['color']}]"):
            console.print(f"[yellow]⏭️  Skipping {provider['name']}[/yellow]")
            return False
        
        console.print(f"\n[bold]Get your API key:[/bold] {provider['signup_url']}")
        
        max_attempts = 3
        for attempt in range(max_attempts):
            api_key = Prompt.ask(
                f"Enter your {provider['name']} API key",
                password=True
            )
            
            if not api_key or api_key.strip() == "":
                console.print("[red]❌ API key cannot be empty[/red]")
                continue
            
            # Test the API key
            success, message, response_time = await self._test_with_spinner(provider_key, api_key)
            
            if success:
                console.print(f"[green]✅ API key works! Response: {message} ({response_time:.0f}ms)[/green]")
                self.env_data[provider['api_key_env']] = api_key
                self.env_data[provider['model_env']] = provider['default_model']
                self.test_results[provider_key] = {
                    "success": True,
                    "response_time": response_time,
                    "message": message
                }
                return True
            else:
                console.print(f"[red]❌ Test failed: {message}[/red]")
                if attempt < max_attempts - 1:
                    if not Confirm.ask("Try again?"):
                        break
        
        console.print(f"[yellow]⏭️  Skipping {provider['name']} after {max_attempts} attempts[/yellow]")
        return False
    
    async def _setup_deepseek(self) -> bool:
        """Special setup for DeepSeek local endpoint"""
        provider = self.MODEL_PROVIDERS["deepseek"]
        
        if not Confirm.ask(f"[{provider['color']}]Set up {provider['name']} (Local)?[/{provider['color']}]"):
            console.print(f"[yellow]⏭️  Skipping {provider['name']}[/yellow]")
            return False
        
        console.print("\n[bold]Local Model Setup:[/bold]")
        console.print("DeepSeek runs locally, so you need a local LLM server running.")
        console.print(f"Default endpoint: [cyan]{provider['default_url']}[/cyan]")
        
        # Get endpoint URL
        current_url = self.env_data.get('LOCAL_LLM_BASE_URL', provider['default_url'])
        endpoint = Prompt.ask(
            "Enter your local LLM endpoint URL",
            default=current_url
        )
        
        # Test connectivity
        success, message, response_time = await self._test_with_spinner("deepseek", "", endpoint)
        
        if success:
            console.print(f"[green]✅ Local endpoint works! Response: {message} ({response_time:.0f}ms)[/green]")
            self.env_data['LOCAL_LLM_BASE_URL'] = endpoint
            self.env_data['LOCAL_LLM_MODEL'] = provider['default_model']
            self.test_results["deepseek"] = {
                "success": True,
                "response_time": response_time,
                "message": message
            }
            return True
        else:
            console.print(f"[red]❌ Connection failed: {message}[/red]")
            console.print("[yellow]💡 Make sure your local LLM server is running[/yellow]")
            return False
    
    async def _test_with_spinner(self, provider_key: str, api_key: str, endpoint: Optional[str] = None) -> Tuple[bool, str, float]:
        """Test API with a spinner"""
        provider = self.MODEL_PROVIDERS[provider_key]
        
        with console.status(f"[{provider['color']}]Testing {provider['name']}...[/{provider['color']}]", spinner="dots"):
            return await self.test_api_key(provider_key, api_key, endpoint)
    
    async def test_all_configured_models(self):
        """Test all configured models with a standard prompt"""
        console.print()
        console.rule("[bold blue]🧪 Testing All Configured Models[/bold blue]")
        console.print()
        
        test_prompt = "Say hello in exactly 5 words."
        configured_models = []
        
        # Find configured models
        for provider_key, provider in self.MODEL_PROVIDERS.items():
            api_key = self.env_data.get(provider['api_key_env'])
            if api_key and api_key not in ['your_key_here', '']:
                configured_models.append(provider_key)
        
        if not configured_models:
            console.print("[yellow]No models configured for testing.[/yellow]")
            return
        
        console.print(f"Testing {len(configured_models)} configured models with prompt: [cyan]\"{test_prompt}\"[/cyan]")
        console.print()
        
        # Test each model
        results = {}
        for provider_key in configured_models:
            provider = self.MODEL_PROVIDERS[provider_key]
            
            with console.status(f"[{provider['color']}]Testing {provider['name']}...[/{provider['color']}]"):
                api_key = self.env_data.get(provider['api_key_env'])
                endpoint = self.env_data.get('LOCAL_LLM_BASE_URL') if provider_key == "deepseek" else None
                
                success, message, response_time = await self.test_api_key(provider_key, api_key, endpoint)
                results[provider_key] = {
                    "success": success,
                    "message": message,
                    "response_time": response_time
                }
                
                if success:
                    console.print(f"[green]✅ {provider['icon']} {provider['name']}: \"{message}\" ({response_time:.0f}ms)[/green]")
                else:
                    console.print(f"[red]❌ {provider['icon']} {provider['name']}: {message}[/red]")
        
        return results
    
    def display_final_summary(self):
        """Display final configuration summary"""
        console.print()
        console.rule("[bold green]🎉 Setup Complete![/bold green]")
        console.print()
        
        # Create summary table
        table = Table(title="✅ Model Configuration Summary", box=box.ROUNDED)
        table.add_column("Provider", style="bold", min_width=20)
        table.add_column("Status", min_width=15)
        table.add_column("Response Time", min_width=12)
        table.add_column("Cost (per 1K)", min_width=15)
        
        configured_count = 0
        for provider_key, provider in self.MODEL_PROVIDERS.items():
            api_key = self.env_data.get(provider['api_key_env'])
            test_result = self.test_results.get(provider_key, {})
            
            # Status
            if test_result.get("success"):
                status = f"[green]✓ Configured[/green]"
                response_time = f"{test_result.get('response_time', 0):.0f}ms"
                configured_count += 1
            elif api_key and api_key not in ['your_key_here', '']:
                status = f"[yellow]⚠ Not tested[/yellow]"
                response_time = "N/A"
            else:
                status = f"[red]✗ Not configured[/red]"
                response_time = "N/A"
            
            # Cost info
            default_model = provider['default_model']
            pricing = provider['pricing'].get(default_model, {"input": 0, "output": 0})
            if pricing['input'] == 0 and pricing['output'] == 0:
                cost = "[green]FREE[/green]"
            else:
                cost = f"${pricing['input']:.4f}/${pricing['output']:.4f}"
            
            table.add_row(
                f"{provider['icon']} {provider['name']}", 
                status, 
                response_time, 
                cost
            )
        
        console.print(table)
        console.print()
        
        # Tips and next steps
        tips_panel = Panel.fit(
            f"[bold]🎯 Next Steps:[/bold]\n\n"
            f"• Run comparisons: [cyan]python compare_models.py --all \"test prompt\"[/cyan]\n"
            f"• Interactive mode: [cyan]python compare_models.py --interactive[/cyan]\n"
            f"• Help: [cyan]python compare_models.py --help[/cyan]\n\n"
            f"[bold]💡 Tips:[/bold]\n"
            f"• Use [cyan]--models openai,claude[/cyan] to compare specific models\n"
            f"• Try [cyan]--research-mode[/cyan] for mental health evaluations\n"
            f"• Use [cyan]--verbose[/cyan] for detailed output\n\n"
            f"[dim]Configuration saved to .env file[/dim]",
            title="Ready to Go!",
            border_style="green"
        )
        console.print(tips_panel)
        
        if configured_count > 0:
            console.print(f"\n[bold green]🚀 {configured_count} model(s) configured successfully![/bold green]")
        else:
            console.print(f"\n[yellow]⚠️  No models were configured. Re-run this script to set up API keys.[/yellow]")
    
    async def run_wizard(self):
        """Run the complete setup wizard"""
        # Load existing environment
        self.env_data = self.load_existing_env()
        
        # Welcome message
        self.display_welcome()
        
        if not Confirm.ask("[bold]Start model setup?[/bold]"):
            console.print("[yellow]Setup cancelled.[/yellow]")
            return
        
        # Setup each provider
        for provider_key in ["openai", "claude", "deepseek", "gemini"]:
            await self.setup_provider(provider_key)
        
        # Save configuration
        if self.env_data:
            self.save_env_file()
        
        # Test all configured models
        if Confirm.ask("\n[bold]Test all configured models?[/bold]"):
            await self.test_all_configured_models()
        
        # Show final summary
        self.display_final_summary()


async def main():
    """Main entry point"""
    wizard = ModelSetupWizard()
    await wizard.run_wizard()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Setup failed: {e}[/red]")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")