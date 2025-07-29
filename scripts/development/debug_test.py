#!/usr/bin/env python3
"""
Debug Test Script
================

Test the enhanced debugging features for the NoneType error around scenario 3-4.
This runs a quick evaluation with enhanced debug output to catch the exact issue.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def main():
    print("🔍 DEBUG TEST: Enhanced NoneType Error Detection")
    print("="*50)
    print("This will run a quick evaluation with:")
    print("- All models")
    print("- Debug mode enabled")
    print("- 4 scenarios (to trigger the scenario 3-4 issue)")
    print("- Enhanced arithmetic debugging")
    print()
    
    import subprocess
    
    cmd = [
        "python", "scripts/run_research.py",
        "--all-models",
        "--debug",
        "--scenarios", "4",
        "--output", "results/development/"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("🚀 Starting debug run...")
    print("-" * 50)
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line.rstrip())
            
        process.wait()
        
        if process.returncode == 0:
            print()
            print("✅ Debug run completed successfully!")
        else:
            print()
            print(f"❌ Debug run failed with exit code {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Debug run interrupted by user")
        process.terminate()
    except Exception as e:
        print(f"\n💥 Error running debug test: {e}")

if __name__ == "__main__":
    main()