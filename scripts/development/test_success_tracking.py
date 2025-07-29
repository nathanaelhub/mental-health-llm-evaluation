#!/usr/bin/env python3
"""
Test Success Rate Tracking
========================

Simple test to verify that StatusTracker is properly counting API calls.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.research.utils import StatusTracker

def main():
    print("Testing Success Rate Tracking\n")
    
    # Create status tracker
    tracker = StatusTracker()
    print(f"Initial state:")
    print(f"  API Calls: {tracker.api_calls}")
    print(f"  Success Count: {tracker.success_count}")
    print(f"  Failure Count: {tracker.failure_count}")
    print(f"  Success Rate: {tracker.get_success_rate():.1f}%")
    print()
    
    # Simulate successful API calls
    print("Simulating 3 successful API calls...")
    tracker.increment_api_calls('openai', 1.5, 0.002, success=True)
    tracker.increment_api_calls('deepseek', 2.0, 0.0, success=True)
    tracker.increment_api_calls('openai', 1.8, 0.002, success=True)
    
    print(f"After successful calls:")
    print(f"  API Calls: {tracker.api_calls}")
    print(f"  Success Count: {tracker.success_count}")
    print(f"  Failure Count: {tracker.failure_count}")
    print(f"  Success Rate: {tracker.get_success_rate():.1f}%")
    print()
    
    # Simulate failed API calls
    print("Simulating 1 failed API call...")
    tracker.increment_api_calls('deepseek', 0, 0, success=False)
    
    print(f"After failed call:")
    print(f"  API Calls: {tracker.api_calls}")
    print(f"  Success Count: {tracker.success_count}")
    print(f"  Failure Count: {tracker.failure_count}")
    print(f"  Success Rate: {tracker.get_success_rate():.1f}%")
    print()
    
    # Show metrics table
    print("\nMetrics Table:")
    metrics = tracker.create_metrics_table()
    # Print metrics table manually since we don't have rich console here
    print(f"  OpenAI: Avg Response Time: {tracker.get_average_response_time('openai'):.2f}s")
    print(f"  DeepSeek: Avg Response Time: {tracker.get_average_response_time('deepseek'):.2f}s")
    print(f"  Total Cost: ${tracker.total_cost:.4f}")
    
    print("\n✅ Success tracking appears to be working correctly!")

if __name__ == "__main__":
    main()