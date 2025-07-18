"""
Fixed Cost Analysis Functions with Defensive Programming
======================================================

This module contains the fixed version of cost analysis functions that properly
handle None values and ensure robust calculation of cost metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union


def _calculate_cost_analysis(openai_scores: List[Dict[str, Any]], 
                           deepseek_scores: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate cost analysis with defensive programming to handle None values.
    
    Args:
        openai_scores: List of score dictionaries for OpenAI
        deepseek_scores: List of score dictionaries for DeepSeek
        
    Returns:
        Dictionary with cost analysis metrics
    """
    # Extract costs with defensive filtering
    openai_costs = _extract_valid_costs(openai_scores)
    deepseek_costs = _extract_valid_costs(deepseek_scores)
    
    # Calculate averages only with valid costs
    openai_avg_cost = np.mean(openai_costs) if openai_costs else 0.0
    deepseek_avg_cost = np.mean(deepseek_costs) if deepseek_costs else 0.0
    
    # Calculate total costs
    openai_total_cost = sum(openai_costs) if openai_costs else 0.0
    deepseek_total_cost = sum(deepseek_costs) if deepseek_costs else 0.0
    
    # Cost difference (positive means OpenAI is more expensive)
    cost_difference = openai_avg_cost - deepseek_avg_cost
    
    # Cost ratio (only if deepseek has non-zero cost)
    cost_ratio = openai_avg_cost / deepseek_avg_cost if deepseek_avg_cost > 0 else float('inf')
    
    return {
        'openai_avg_cost': openai_avg_cost,
        'deepseek_avg_cost': deepseek_avg_cost,
        'openai_total_cost': openai_total_cost,
        'deepseek_total_cost': deepseek_total_cost,
        'cost_difference': cost_difference,
        'cost_ratio': cost_ratio if cost_ratio != float('inf') else None,
        'openai_cost_count': len(openai_costs),
        'deepseek_cost_count': len(deepseek_costs)
    }


def _extract_valid_costs(scores: List[Dict[str, Any]]) -> List[float]:
    """
    Extract valid cost values from score dictionaries, filtering out None values.
    
    Args:
        scores: List of score dictionaries
        
    Returns:
        List of valid cost values (floats)
    """
    valid_costs = []
    
    for score in scores:
        if not isinstance(score, dict):
            continue
            
        # Try to get cost value
        cost = score.get('cost')
        
        # If cost is None or not present, try 'cost_usd'
        if cost is None:
            cost = score.get('cost_usd')
        
        # Validate and convert cost
        if cost is not None:
            try:
                # Convert to float if possible
                cost_float = float(cost)
                # Only include non-negative costs
                if cost_float >= 0:
                    valid_costs.append(cost_float)
            except (ValueError, TypeError):
                # Skip invalid cost values
                continue
    
    return valid_costs


def validate_cost_data(evaluation_data: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Validate and ensure cost data is properly structured in evaluation results.
    
    Args:
        evaluation_data: Evaluation data (can be dict or object)
        
    Returns:
        Dictionary with validated cost data
    """
    # Default cost value for missing data
    default_cost = 0.0
    
    # Handle object format (hasattr)
    if hasattr(evaluation_data, 'cost_usd'):
        cost = getattr(evaluation_data, 'cost_usd', None)
    elif hasattr(evaluation_data, 'cost'):
        cost = getattr(evaluation_data, 'cost', None)
    # Handle dict format
    elif isinstance(evaluation_data, dict):
        cost = evaluation_data.get('cost_usd') or evaluation_data.get('cost')
    else:
        cost = None
    
    # Validate cost value
    if cost is None:
        validated_cost = default_cost
    else:
        try:
            validated_cost = float(cost)
            # Ensure non-negative
            if validated_cost < 0:
                validated_cost = default_cost
        except (ValueError, TypeError):
            validated_cost = default_cost
    
    return {
        'cost': validated_cost,
        'cost_usd': validated_cost,
        'has_valid_cost': cost is not None and cost >= 0
    }


def ensure_response_has_cost(response: Any, default_cost: float = 0.0) -> Any:
    """
    Ensure a model response has a valid cost value.
    
    Args:
        response: Model response object or dict
        default_cost: Default cost to use if missing (0.0 for local models)
        
    Returns:
        Response with ensured cost value
    """
    # For dictionary responses
    if isinstance(response, dict):
        if 'cost_usd' not in response or response['cost_usd'] is None:
            response['cost_usd'] = default_cost
        if 'cost' not in response or response['cost'] is None:
            response['cost'] = default_cost
    
    # For object responses with attributes
    elif hasattr(response, '__dict__'):
        if not hasattr(response, 'cost_usd') or response.cost_usd is None:
            response.cost_usd = default_cost
        if not hasattr(response, 'cost') or response.cost is None:
            response.cost = default_cost
    
    return response


# Example usage in statistical analysis
def calculate_cost_statistics(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive cost statistics from evaluation scenarios.
    
    Args:
        scenarios: List of evaluation scenarios
        
    Returns:
        Dictionary with cost statistics
    """
    openai_scores = []
    deepseek_scores = []
    
    for scenario in scenarios:
        openai_eval = getattr(scenario, 'openai_evaluation', None)
        deepseek_eval = getattr(scenario, 'deepseek_evaluation', None)
        
        if openai_eval:
            # Validate cost data
            cost_data = validate_cost_data(openai_eval)
            
            # Build score dict with validated cost
            score_dict = {
                'composite': getattr(openai_eval, 'composite_score', 
                                   openai_eval.get('composite_score', 0)) if openai_eval else 0,
                'cost': cost_data['cost'],
                'cost_usd': cost_data['cost_usd']
            }
            openai_scores.append(score_dict)
        
        if deepseek_eval:
            # Validate cost data
            cost_data = validate_cost_data(deepseek_eval)
            
            # Build score dict with validated cost
            score_dict = {
                'composite': getattr(deepseek_eval, 'composite_score',
                                   deepseek_eval.get('composite_score', 0)) if deepseek_eval else 0,
                'cost': cost_data['cost'],
                'cost_usd': cost_data['cost_usd']
            }
            deepseek_scores.append(score_dict)
    
    # Calculate cost analysis with defensive function
    cost_analysis = _calculate_cost_analysis(openai_scores, deepseek_scores)
    
    return {
        'cost_analysis': cost_analysis,
        'total_scenarios': len(scenarios),
        'scenarios_with_cost_data': {
            'openai': cost_analysis['openai_cost_count'],
            'deepseek': cost_analysis['deepseek_cost_count']
        }
    }


# Integration patch for existing code
def patch_statistical_analysis():
    """
    Patch to replace the problematic _calculate_cost_analysis function.
    
    This can be imported and applied to fix the existing code.
    """
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from analysis import statistical_analysis
        # Replace the function
        statistical_analysis._calculate_cost_analysis = _calculate_cost_analysis
        print("✅ Patched _calculate_cost_analysis function")
    except ImportError:
        print("❌ Could not import statistical_analysis module")


if __name__ == "__main__":
    # Test the functions
    test_scores = [
        {'cost': 0.05, 'composite': 8.5},
        {'cost': None, 'composite': 7.0},  # None value
        {'cost_usd': 0.03, 'composite': 9.0},
        {'composite': 8.0},  # Missing cost
        {'cost': 'invalid', 'composite': 7.5},  # Invalid cost
    ]
    
    print("Testing _extract_valid_costs:")
    valid_costs = _extract_valid_costs(test_scores)
    print(f"Valid costs extracted: {valid_costs}")
    
    print("\nTesting _calculate_cost_analysis:")
    cost_analysis = _calculate_cost_analysis(test_scores, test_scores)
    for key, value in cost_analysis.items():
        print(f"  {key}: {value}")