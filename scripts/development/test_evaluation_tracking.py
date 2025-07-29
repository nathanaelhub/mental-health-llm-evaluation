#!/usr/bin/env python3
"""
Test Evaluation with Success Tracking
===================================

Test that shows where increment_api_calls should be called during evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.research.utils import StatusTracker, debug_print
from src.research.evaluation import load_model_clients, create_model_client_instances, check_model_availability

def test_evaluation_tracking():
    print("🧪 Testing Evaluation with Success Tracking")
    print("=" * 50)
    
    # Create status tracker
    status_tracker = StatusTracker()
    print(f"📊 Initial tracker state:")
    print(f"   API Calls: {status_tracker.api_calls}")
    print(f"   Success Rate: {status_tracker.get_success_rate():.1f}%")
    print()
    
    # Test model client loading
    print("📦 Loading model clients...")
    model_clients = load_model_clients(clean_output=True)
    print(f"   Found {len(model_clients)} client classes")
    
    # Test model availability
    selected_models = ['openai', 'deepseek']
    available_models = check_model_availability(selected_models, model_clients, clean_output=True)
    print(f"   Available models: {available_models}")
    
    # Test client instance creation
    print(f"📱 Creating client instances...")
    client_instances = create_model_client_instances(
        available_models, model_clients, 
        clean_output=True, debug_mode=False, minimal=True
    )
    print(f"   Created {len(client_instances)} client instances")
    
    # Now manually simulate what happens in evaluate_model_with_retry
    print(f"\n🔄 Simulating model evaluation calls...")
    
    for model_name in available_models:
        print(f"   Evaluating {model_name}...")
        
        # Simulate a successful evaluation
        # This is where increment_api_calls should be called in the real evaluation
        response_time = 1.5  # Mock response time
        cost = 0.002 if model_name == 'openai' else 0.0  # Mock cost
        success = True  # Mock success
        
        # THE CRITICAL CALL - this should happen after each model evaluation
        status_tracker.increment_api_calls(model_name, response_time, cost, success, debug_mode=True)
        
        print(f"   ✅ {model_name} evaluation tracked")
    
    print(f"\n📊 Final tracker state:")
    print(f"   API Calls: {status_tracker.api_calls}")
    print(f"   Success Count: {status_tracker.success_count}")
    print(f"   Failure Count: {status_tracker.failure_count}")
    print(f"   Success Rate: {status_tracker.get_success_rate():.1f}%")
    print(f"   Total Cost: ${status_tracker.total_cost:.4f}")
    
    if status_tracker.api_calls > 0 and status_tracker.get_success_rate() > 0:
        print(f"\n✅ SUCCESS: Tracking is working!")
        print(f"   The issue is likely that increment_api_calls() is not being called")
        print(f"   during actual model evaluations in the main pipeline.")
    else:
        print(f"\n❌ FAILURE: Tracking is not working")

if __name__ == "__main__":
    test_evaluation_tracking()