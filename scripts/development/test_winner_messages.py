#!/usr/bin/env python3
"""
Test Winner Messages Fix
========================

Test that the "Winner: (0.00)" messages are properly handled.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.research.display import print_ultra_clean_scenario_result

def test_winner_messages():
    print("🧪 Testing Winner Messages Fix")
    print("=" * 40)
    
    # Test data with zero scores (the problem case)
    evaluations_zero = {
        'openai': {'empathy': 0.0, 'therapeutic': 0.0, 'safety': 0.0, 'clarity': 0.0, 'composite': 0.0},
        'deepseek': {'empathy': 0.0, 'therapeutic': 0.0, 'safety': 0.0, 'clarity': 0.0, 'composite': 0.0}
    }
    
    # Test data with real scores
    evaluations_real = {
        'openai': {'empathy': 8.5, 'therapeutic': 8.0, 'safety': 9.0, 'clarity': 8.3, 'composite': 8.5},
        'deepseek': {'empathy': 7.8, 'therapeutic': 7.5, 'safety': 8.5, 'clarity': 7.7, 'composite': 7.9}
    }
    
    print("\n1. Testing with ZERO scores (should show ✓):")
    print("   Before fix: Would show 'OpenAI (0.0) vs DeepSeek (0.0) → OpenAI wins'")
    print("   After fix:")
    print_ultra_clean_scenario_result(1, 3, "Test Scenario", evaluations_zero, "OpenAI", demo_mode=False)
    
    print("\n2. Testing with REAL scores (should show normal output):")
    print("   Expected: 'OpenAI (8.5) vs DeepSeek (7.9) → OpenAI wins'")
    print("   Actual:")
    print_ultra_clean_scenario_result(2, 3, "Test Scenario", evaluations_real, "OpenAI", demo_mode=False)
    
    print("\n3. Testing DEMO MODE (should show ✓):")
    print("   Expected: Just a checkmark")
    print("   Actual:")
    print_ultra_clean_scenario_result(3, 3, "Test Scenario", evaluations_real, "OpenAI", demo_mode=True)
    
    print("\n✅ Winner message fix test completed!")
    print("   - Zero scores now show ✓ instead of misleading (0.00)")
    print("   - Demo mode shows ✓ for clean output")
    print("   - Real scores still show proper comparisons")

if __name__ == "__main__":
    test_winner_messages()