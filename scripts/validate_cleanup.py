#!/usr/bin/env python3
"""
Mental Health LLM Evaluation - System Validation Script
======================================================

Comprehensive validation script to ensure all components work after cleanup.
Tests imports, API connectivity, and core functionality.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")


def test_imports() -> List[Tuple[str, bool, str]]:
    """Test all critical imports"""
    print_header("Testing Core Imports")
    
    imports_to_test = [
        ("Core Package", "import src"),
        ("Models - Base", "from src.models import BaseModel"),
        ("Models - OpenAI", "from src.models import OpenAIClient"),
        ("Models - Claude", "from src.models import ClaudeClient"),
        ("Models - DeepSeek", "from src.models import DeepSeekClient"),
        ("Models - Gemma", "from src.models import GemmaClient"),
        ("Evaluation", "from src.evaluation import MentalHealthEvaluator"),
        ("Analysis", "from src.analysis import StatisticalAnalyzer"),
        ("Scenarios", "from src.scenarios import ScenarioLoader"),
        ("Config", "from src.config import config_loader"),
        ("Utils", "from src.utils import setup_logging"),
    ]
    
    results = []
    for name, import_statement in imports_to_test:
        try:
            exec(import_statement)
            print_success(f"{name}: {import_statement}")
            results.append((name, True, "OK"))
        except Exception as e:
            print_error(f"{name}: {import_statement} - {str(e)}")
            results.append((name, False, str(e)))
    
    return results


def test_environment() -> Dict[str, bool]:
    """Test environment variables and configuration"""
    print_header("Testing Environment Configuration")
    
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "LOCAL_LLM_BASE_URL": os.getenv("LOCAL_LLM_BASE_URL"),
        "GEMMA_ENDPOINT": os.getenv("GEMMA_ENDPOINT"),
    }
    
    results = {}
    for var_name, var_value in env_vars.items():
        if var_value:
            # Mask API keys for security
            if "API_KEY" in var_name:
                display_value = var_value[:8] + "..." if len(var_value) > 8 else "***"
            else:
                display_value = var_value
            print_success(f"{var_name}: {display_value}")
            results[var_name] = True
        else:
            print_warning(f"{var_name}: Not set")
            results[var_name] = False
    
    return results


def test_file_structure() -> Dict[str, bool]:
    """Test that all essential files exist"""
    print_header("Testing File Structure")
    
    essential_files = [
        "scripts/run_research.py",
        "scripts/run_conversation_generation.py",
        "scripts/compare_models.py",
        "config/main.yaml",
        "src/models/openai_client.py",
        "src/models/claude_client.py",
        "src/models/deepseek_client.py",
        "src/models/gemma_client.py",
        "src/evaluation/mental_health_evaluator.py",
        "src/analysis/statistical_analysis.py",
        "requirements.txt",
        "README.md",
        "docs/FILE_DESCRIPTIONS.md",
        "docs/TESTING_GUIDE.md",
    ]
    
    results = {}
    for file_path in essential_files:
        exists = os.path.exists(file_path)
        if exists:
            print_success(f"{file_path}")
        else:
            print_error(f"{file_path} - MISSING")
        results[file_path] = exists
    
    return results


def test_model_initialization():
    """Test model initialization"""
    print_header("Testing Model Initialization")
    
    results = []
    
    # Test OpenAI
    try:
        from src.models import OpenAIClient
        client = OpenAIClient()
        print_success("OpenAI Client initialized")
        results.append(("OpenAI", True))
    except Exception as e:
        print_error(f"OpenAI Client failed: {e}")
        results.append(("OpenAI", False))
    
    # Test Claude
    try:
        from src.models import ClaudeClient
        client = ClaudeClient()
        print_success("Claude Client initialized")
        results.append(("Claude", True))
    except Exception as e:
        print_warning(f"Claude Client initialization warning: {e}")
        results.append(("Claude", False))
    
    # Test DeepSeek
    try:
        from src.models import DeepSeekClient
        client = DeepSeekClient()
        print_success("DeepSeek Client initialized")
        results.append(("DeepSeek", True))
    except Exception as e:
        print_error(f"DeepSeek Client failed: {e}")
        results.append(("DeepSeek", False))
    
    # Test Gemma
    try:
        from src.models import GemmaClient
        client = GemmaClient()
        print_success("Gemma Client initialized")
        results.append(("Gemma", True))
    except Exception as e:
        print_error(f"Gemma Client failed: {e}")
        results.append(("Gemma", False))
    
    return results


def test_quick_functionality():
    """Test quick functionality of main scripts"""
    print_header("Testing Core Functionality")
    
    # Test if we can load the evaluator
    try:
        from src.evaluation import MentalHealthEvaluator
        evaluator = MentalHealthEvaluator()
        print_success("MentalHealthEvaluator loaded successfully")
    except Exception as e:
        print_error(f"MentalHealthEvaluator failed: {e}")
    
    # Test if we can load scenarios
    try:
        from src.scenarios import ScenarioLoader
        loader = ScenarioLoader()
        print_success("ScenarioLoader initialized successfully")
    except Exception as e:
        print_error(f"ScenarioLoader failed: {e}")
    
    # Test configuration loading
    try:
        from src.config.config_loader import load_config
        config = load_config("config/main.yaml")
        print_success(f"Configuration loaded: {len(config)} top-level keys")
    except Exception as e:
        print_error(f"Configuration loading failed: {e}")


def generate_summary_report(import_results, env_results, file_results, model_results):
    """Generate a summary report"""
    print_header("Validation Summary Report")
    
    # Calculate success rates
    import_success = sum(1 for _, success, _ in import_results if success)
    import_total = len(import_results)
    
    env_success = sum(1 for success in env_results.values() if success)
    env_total = len(env_results)
    
    file_success = sum(1 for success in file_results.values() if success)
    file_total = len(file_results)
    
    model_success = sum(1 for _, success in model_results if success)
    model_total = len(model_results)
    
    # Print summary
    print(f"\n{Colors.BOLD}Test Results:{Colors.RESET}")
    print(f"  • Imports: {import_success}/{import_total} passed ({import_success/import_total*100:.1f}%)")
    print(f"  • Environment: {env_success}/{env_total} configured ({env_success/env_total*100:.1f}%)")
    print(f"  • Files: {file_success}/{file_total} found ({file_success/file_total*100:.1f}%)")
    print(f"  • Models: {model_success}/{model_total} initialized ({model_success/model_total*100:.1f}%)")
    
    # Overall assessment
    total_tests = import_total + env_total + file_total + model_total
    total_passed = import_success + env_success + file_success + model_success
    overall_percentage = total_passed / total_tests * 100
    
    print(f"\n{Colors.BOLD}Overall: {total_passed}/{total_tests} tests passed ({overall_percentage:.1f}%){Colors.RESET}")
    
    if overall_percentage >= 90:
        print_success("System is ready for research! 🚀")
    elif overall_percentage >= 70:
        print_warning("System is mostly ready but some components need attention")
    else:
        print_error("System needs configuration before running research")
    
    # Recommendations
    if env_success < env_total:
        print(f"\n{Colors.YELLOW}Recommendations:{Colors.RESET}")
        if not env_results.get("OPENAI_API_KEY"):
            print("  • Set OPENAI_API_KEY in your .env file")
        if not env_results.get("ANTHROPIC_API_KEY"):
            print("  • Set ANTHROPIC_API_KEY in your .env file (optional)")
        if not env_results.get("LOCAL_LLM_BASE_URL"):
            print("  • Configure local LLM endpoint for DeepSeek")
        if not env_results.get("GEMMA_ENDPOINT"):
            print("  • Configure Gemma endpoint if using local Gemma")


def main():
    """Main validation function"""
    print(f"\n{Colors.BOLD}🧠 Mental Health LLM Evaluation - System Validation{Colors.RESET}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    import_results = test_imports()
    env_results = test_environment()
    file_results = test_file_structure()
    model_results = test_model_initialization()
    test_quick_functionality()
    
    # Generate summary
    generate_summary_report(import_results, env_results, file_results, model_results)
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.RESET}")
    print("1. Fix any missing environment variables in .env")
    print("2. Run: python scripts/run_research.py --quick")
    print("3. Test model comparison: python scripts/compare_models.py --help")
    print("")


if __name__ == "__main__":
    main()