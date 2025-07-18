"""
Cost Data Validation Utilities
==============================

Provides validation and normalization functions for cost data in model responses
and evaluation results to prevent NoneType errors in calculations.
"""

from typing import Any, Dict, List, Optional, Union


def validate_cost_value(cost: Any, default: float = 0.0) -> float:
    """
    Validate and normalize a single cost value.
    
    Args:
        cost: The cost value to validate (can be None, numeric, or string)
        default: Default value to use if cost is invalid
        
    Returns:
        Valid float cost value
    """
    if cost is None:
        return default
    
    try:
        cost_float = float(cost)
        # Ensure non-negative
        return max(0.0, cost_float)
    except (ValueError, TypeError):
        return default


def ensure_cost_in_response(response: Union[Dict[str, Any], Any], 
                          default_cost: float = 0.0) -> None:
    """
    Ensure a response object/dict has valid cost fields.
    
    Modifies the response in place to add/fix cost fields.
    
    Args:
        response: Response object or dictionary
        default_cost: Default cost value (0.0 for local models)
    """
    if isinstance(response, dict):
        # Handle dictionary responses
        if 'cost_usd' not in response or response['cost_usd'] is None:
            response['cost_usd'] = default_cost
        else:
            response['cost_usd'] = validate_cost_value(response['cost_usd'], default_cost)
            
        # Also ensure 'cost' field for compatibility
        if 'cost' not in response or response['cost'] is None:
            response['cost'] = response['cost_usd']
    
    elif hasattr(response, '__dict__'):
        # Handle object responses
        if not hasattr(response, 'cost_usd') or response.cost_usd is None:
            response.cost_usd = default_cost
        else:
            response.cost_usd = validate_cost_value(response.cost_usd, default_cost)
            
        # Also ensure 'cost' attribute for compatibility  
        if not hasattr(response, 'cost') or response.cost is None:
            response.cost = response.cost_usd


def extract_cost_from_evaluation(evaluation: Union[Dict[str, Any], Any]) -> float:
    """
    Extract a valid cost value from an evaluation result.
    
    Args:
        evaluation: Evaluation data (can be dict or object)
        
    Returns:
        Valid cost value (float)
    """
    # Try different ways to get cost
    cost = None
    
    # Try object attributes
    if hasattr(evaluation, 'cost_usd'):
        cost = getattr(evaluation, 'cost_usd', None)
    elif hasattr(evaluation, 'cost'):
        cost = getattr(evaluation, 'cost', None)
    
    # Try dictionary keys
    elif isinstance(evaluation, dict):
        cost = evaluation.get('cost_usd')
        if cost is None:
            cost = evaluation.get('cost')
            
        # Try nested in metadata
        if cost is None and 'metadata' in evaluation:
            metadata = evaluation['metadata']
            if isinstance(metadata, dict):
                cost = metadata.get('cost') or metadata.get('cost_usd')
    
    # Validate and return
    return validate_cost_value(cost, 0.0)


def validate_cost_array(costs: List[Any]) -> List[float]:
    """
    Validate an array of cost values, removing None and invalid values.
    
    Args:
        costs: List of cost values (may contain None or invalid values)
        
    Returns:
        List of valid float cost values
    """
    valid_costs = []
    
    for cost in costs:
        if cost is not None:
            try:
                cost_float = float(cost)
                if cost_float >= 0:
                    valid_costs.append(cost_float)
            except (ValueError, TypeError):
                continue
                
    return valid_costs


def create_cost_safe_scores(evaluations: List[Union[Dict[str, Any], Any]]) -> List[Dict[str, float]]:
    """
    Create score dictionaries with guaranteed valid cost values.
    
    Args:
        evaluations: List of evaluation results
        
    Returns:
        List of score dictionaries with validated costs
    """
    scores = []
    
    for eval_data in evaluations:
        if not eval_data:
            continue
            
        # Extract scores
        if hasattr(eval_data, 'composite_score'):
            composite = getattr(eval_data, 'composite_score', 0.0)
            empathy = getattr(eval_data, 'empathy_score', 0.0)
            therapeutic = getattr(eval_data, 'therapeutic_value_score', 0.0)
            safety = getattr(eval_data, 'safety_score', 0.0)
            clarity = getattr(eval_data, 'clarity_score', 0.0)
        else:
            composite = eval_data.get('composite_score', 0.0)
            empathy = eval_data.get('empathy_score', 0.0)
            therapeutic = eval_data.get('therapeutic_value_score', 0.0)
            safety = eval_data.get('safety_score', 0.0)
            clarity = eval_data.get('clarity_score', 0.0)
        
        # Extract and validate cost
        cost = extract_cost_from_evaluation(eval_data)
        
        scores.append({
            'composite': float(composite or 0.0),
            'empathy': float(empathy or 0.0),
            'therapeutic': float(therapeutic or 0.0),
            'safety': float(safety or 0.0),
            'clarity': float(clarity or 0.0),
            'cost': cost,
            'cost_usd': cost
        })
    
    return scores


# Example test
if __name__ == "__main__":
    # Test validation functions
    print("Testing cost validation utilities...")
    
    # Test validate_cost_value
    test_values = [0.05, None, "0.03", -0.01, "invalid", float('inf')]
    for val in test_values:
        print(f"validate_cost_value({val!r}) = {validate_cost_value(val)}")
    
    # Test extract_cost_from_evaluation
    test_eval = {
        'composite_score': 8.5,
        'cost_usd': None,
        'metadata': {'cost': 0.02}
    }
    print(f"\nextract_cost_from_evaluation({test_eval}) = {extract_cost_from_evaluation(test_eval)}")
    
    # Test validate_cost_array
    test_array = [0.05, None, 0.03, None, -0.01, "0.02", "invalid"]
    print(f"\nvalidate_cost_array({test_array}) = {validate_cost_array(test_array)}")