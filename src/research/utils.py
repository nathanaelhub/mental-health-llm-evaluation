"""
Utils Module
============

Helper functions and utilities for the Mental Health LLM Evaluation Research.
Contains debug functions, safe arithmetic operations, and common utilities.

Key Components:
- Debug printing and variable inspection
- Safe arithmetic operations (prevents NoneType errors)
- Retry mechanisms with exponential backoff
- Common utility functions
"""

import time
from typing import Any, Callable, Optional

# =============================================================================
# DEBUG UTILITIES
# =============================================================================

def debug_print(message, debug_mode=False):
    """Print debug message only if debug mode is enabled"""
    if debug_mode:
        print(f"🔍 DEBUG: {message}")

def debug_variable(name, value, debug_mode=False):
    """Debug print variable name, type, and value"""
    if debug_mode:
        print(f"🔍 DEBUG VAR: {name} = {repr(value)} (type: {type(value).__name__})")

def debug_arithmetic(operation, var1_name, var1, var2_name=None, var2=None, debug_mode=False):
    """Debug print before arithmetic operations"""
    if debug_mode:
        if var2 is not None:
            print(f"🔍 DEBUG ARITH: About to perform {operation}")
            print(f"    {var1_name} = {repr(var1)} (type: {type(var1).__name__})")
            print(f"    {var2_name} = {repr(var2)} (type: {type(var2).__name__})")
        else:
            print(f"🔍 DEBUG ARITH: About to perform {operation}")
            print(f"    {var1_name} = {repr(var1)} (type: {type(var1).__name__})")

# =============================================================================
# SAFE ARITHMETIC OPERATIONS
# =============================================================================

def safe_add(a, b, context="", debug_mode=False):
    """Safe addition with NoneType protection and debugging"""
    if debug_mode:
        print(f"🛡️ SAFE_ADD in {context}: a={a} (type: {type(a)}), b={b} (type: {type(b)})")
    
    if a is None and b is None:
        if debug_mode:
            print(f"❌ Both values None in {context}: returning 0")
        return 0.0
    elif a is None:
        if debug_mode:
            print(f"❌ First value None in {context}: a=None, using 0 + {b}")
        return 0.0 + b
    elif b is None:
        if debug_mode:
            print(f"❌ Second value None in {context}: {a} + None, using {a} + 0")
        return a + 0.0
    else:
        try:
            result = a + b
            if debug_mode:
                print(f"✅ Normal addition in {context}: {a} + {b} = {result}")
            return result
        except TypeError as e:
            if debug_mode:
                print(f"❌ TypeError in {context}: {a} + {b} failed with {e}")
            # Try to convert to floats
            try:
                result = float(a or 0) + float(b or 0)
                if debug_mode:
                    print(f"🔧 Converted to floats: {result}")
                return result
            except:
                if debug_mode:
                    print(f"🔧 Fallback to 0.0")
                return 0.0

def safe_increment(var_name, current_value, increment=1, context="", debug_mode=False):
    """Safe increment with NoneType protection"""
    if debug_mode:
        print(f"🛡️ SAFE_INCREMENT {var_name} in {context}: current={current_value} (type: {type(current_value)}), increment={increment}")
    
    if current_value is None:
        if debug_mode:
            print(f"❌ {var_name} is None in {context}, initializing to {increment}")
        return increment
    else:
        try:
            result = current_value + increment
            if debug_mode:
                print(f"✅ {var_name} in {context}: {current_value} + {increment} = {result}")
            return result
        except TypeError as e:
            if debug_mode:
                print(f"❌ TypeError incrementing {var_name} in {context}: {e}")
            return increment

# =============================================================================
# RETRY MECHANISMS
# =============================================================================

def retry_with_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0, debug_mode: bool = False):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            debug_print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...", debug_mode)
            time.sleep(delay)
    raise Exception("Should not reach here")

# =============================================================================
# DEPENDENCY CHECKING
# =============================================================================

def check_dependencies():
    """Check for required and optional dependencies"""
    required_modules = ['rich', 'numpy', 'scipy']
    optional_modules = ['matplotlib', 'seaborn', 'psutil']
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required dependencies:")
        for module in missing_modules:
            print(f"   • {module}")
        print("\nInstall with: pip install " + " ".join(missing_modules))
        return False
    
    missing_optional = []
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)
    
    if missing_optional:
        print("ℹ️ Optional dependencies not found (recommended):")
        for module in optional_modules:
            print(f"   • {module}")
        print("   Install with: pip install " + " ".join(optional_modules))
        print()
    
    return True

# =============================================================================
# LOADING UTILITIES
# =============================================================================

def show_startup_loading():
    """Show a brief startup loading animation"""
    import time
    loading_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    for i in range(20):  # Show for about 1 second
        char = loading_chars[i % len(loading_chars)]
        print(f"\r{char} Loading...", end='', flush=True)
        time.sleep(0.05)
    
    print("\r✅ Ready!     ")  # Clear the loading line

def get_rotating_tip():
    """Get a rotating research tip for display during long operations"""
    tips = [
        "🔬 Statistical significance requires p < 0.05",
        "📊 Effect size helps interpret practical importance",
        "🎯 Composite scores balance multiple criteria",
        "⚡ Larger samples increase statistical power",
        "🧠 Mental health AI requires careful validation"
    ]
    
    # Return tip based on current time to rotate
    import time
    tip_index = int(time.time() / 10) % len(tips)
    return tips[tip_index]

# =============================================================================
# STATUS TRACKING CLASS
# =============================================================================

class StatusTracker:
    """Track API calls, costs, success rates, and timing for evaluation runs"""
    
    def __init__(self):
        self.api_calls = 0
        self.start_time = time.time()
        self.current_operation = "Initializing"
        self.total_cost = 0.0
        self.model_response_times = {"openai": [], "deepseek": [], "claude": [], "gemma": []}
        self.success_count = 0
        self.failure_count = 0
        self.last_tip_time = time.time()
        
    def increment_api_calls(self, model_name=None, response_time=None, cost=0.0, success=True, debug_mode=False):
        debug_print(f"increment_api_calls called: model={model_name}, success={success}", debug_mode)
        
        # CRITICAL FIX: Use safe arithmetic operations to prevent NoneType errors
        debug_arithmetic("self.api_calls += 1", "self.api_calls", self.api_calls, debug_mode=debug_mode)
        self.api_calls = safe_increment("api_calls", self.api_calls, 1, "StatusTracker.increment_api_calls", debug_mode)
        
        cost_to_add = cost if cost is not None else 0.0
        debug_arithmetic("self.total_cost += cost", "self.total_cost", self.total_cost, "cost_to_add", cost_to_add, debug_mode)
        self.total_cost = safe_add(self.total_cost, cost_to_add, "StatusTracker.total_cost", debug_mode)
        
        if success:
            debug_arithmetic("self.success_count += 1", "self.success_count", self.success_count, debug_mode=debug_mode)
            self.success_count = safe_increment("success_count", self.success_count, 1, "StatusTracker.success", debug_mode)
        else:
            debug_arithmetic("self.failure_count += 1", "self.failure_count", self.failure_count, debug_mode=debug_mode)
            self.failure_count = safe_increment("failure_count", self.failure_count, 1, "StatusTracker.failure", debug_mode)
            
        debug_print(f"Updated counts: success={self.success_count}, failure={self.failure_count}, total_calls={self.api_calls}", debug_mode)
            
        if model_name and response_time:
            self.model_response_times[model_name].append(response_time)
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
        
    def get_elapsed_time(self):
        """Get elapsed time since start"""
        return time.time() - self.start_time
        
    def get_average_response_time(self, model_name, debug_mode=False):
        """Get average response time for a model"""
        times = self.model_response_times.get(model_name, [])
        if times:
            avg = sum(times) / len(times)
            debug_print(f"Average response time for {model_name}: {avg:.2f}s", debug_mode)
            return avg
        else:
            debug_print(f"No times recorded for {model_name}, returning 0.0", debug_mode)
            return 0.0
        
    def get_success_rate(self, debug_mode=False):
        """Get success rate percentage"""
        debug_arithmetic("total = success_count + failure_count", "self.success_count", self.success_count, "self.failure_count", self.failure_count, debug_mode)
        # CRITICAL FIX: Use safe arithmetic to prevent NoneType errors
        total = safe_add(self.success_count, self.failure_count, "StatusTracker.get_success_rate", debug_mode)
        debug_variable("total", total, debug_mode)
        
        # Always show counts for debugging
        debug_print(f"Success count: {self.success_count}, Failure count: {self.failure_count}, Total: {total}", debug_mode)
        
        if total > 0:
            # Ensure success_count is not None before division
            success_count = self.success_count if self.success_count is not None else 0
            rate = (success_count / total * 100)
            debug_arithmetic("(success_count / total * 100)", "success_count", success_count, "total", total, debug_mode)
            debug_print(f"Calculated success rate: {rate:.1f}%", debug_mode)
            return rate
        else:
            debug_print("No operations recorded, returning 0.0", debug_mode)
            return 0.0
        
    def create_status_table(self):
        """Create a status table for live display"""
        from rich.table import Table
        from rich import box
        
        table = Table(show_header=False, box=box.ROUNDED, padding=(0, 1))
        table.add_column("Metric", style="cyan", min_width=12)
        table.add_column("Value", style="white")
        
        table.add_row("Operation", self.current_operation)
        table.add_row("API Calls Made", str(self.api_calls))
        table.add_row("Total Cost", f"${self.total_cost:.4f}")
        table.add_row("Success Rate", f"{self.get_success_rate(debug_mode=False):.1f}%")
        table.add_row("Elapsed Time", f"{self.get_elapsed_time():.1f}s")
        
        return table
    
    def create_metrics_table(self):
        """Create detailed metrics table"""
        from rich.table import Table
        from rich import box
        
        table = Table(title="📊 Evaluation Metrics", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Avg Response Time", style="green")
        table.add_column("Calls Made", style="yellow")
        table.add_column("Est. Cost", style="magenta")
        
        cost_per_call = {"openai": 0.002, "claude": 0.003, "deepseek": 0.0, "gemma": 0.0}
        
        for model in ["openai", "deepseek", "claude", "gemma"]:
            if self.model_response_times[model]:
                avg_time = self.get_average_response_time(model, debug_mode=False)
                calls = len(self.model_response_times[model])
                est_cost = calls * cost_per_call.get(model, 0.0)
                
                table.add_row(
                    model.title(),
                    f"{avg_time:.2f}s",
                    str(calls),
                    f"${est_cost:.4f}"
                )
        
        return table

# =============================================================================
# COMMON UTILITIES
# =============================================================================

def ensure_directory_exists(directory_path: str):
    """Ensure a directory exists, create if it doesn't"""
    import os
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)

def get_timestamp():
    """Get current timestamp in ISO format"""
    from datetime import datetime
    return datetime.now().isoformat()

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"