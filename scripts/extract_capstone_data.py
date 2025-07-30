#!/usr/bin/env python3
"""
Extract Capstone Data
====================

Analyzes research data to extract detailed statistics and examples
for capstone documentation.
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple

def analyze_research_data(data_path: str) -> Dict[str, Any]:
    """Extract comprehensive statistics from research data"""
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    analysis = {
        'total_scenarios': len(data),
        'successful_scenarios': len([d for d in data if 'error' not in d]),
        'model_distribution': defaultdict(int),
        'prompt_type_distribution': defaultdict(int),
        'confidence_stats': {
            'all': [],
            'by_type': defaultdict(list),
            'by_model': defaultdict(list)
        },
        'response_time_stats': {
            'all': [],
            'by_model': defaultdict(list)
        },
        'model_scores': {
            'by_scenario': [],
            'by_type': defaultdict(lambda: defaultdict(list)),
            'averages': defaultdict(list)
        },
        'competitive_scenarios': [],
        'crisis_scenarios': [],
        'best_performance_examples': [],
        'score_margins': []
    }
    
    for i, item in enumerate(data):
        if 'error' in item:
            continue
            
        # Basic distributions
        selected_model = item.get('selected_model', 'unknown')
        prompt_type = item.get('prompt_type', 'unknown')
        confidence = item.get('confidence_score', 0)
        response_time = item.get('response_time', 0)
        model_scores = item.get('model_scores', {})
        prompt = item.get('prompt', '')
        
        analysis['model_distribution'][selected_model] += 1
        analysis['prompt_type_distribution'][prompt_type] += 1
        
        # Confidence tracking
        analysis['confidence_stats']['all'].append(confidence)
        analysis['confidence_stats']['by_type'][prompt_type].append(confidence)
        analysis['confidence_stats']['by_model'][selected_model].append(confidence)
        
        # Response time tracking
        analysis['response_time_stats']['all'].append(response_time)
        analysis['response_time_stats']['by_model'][selected_model].append(response_time)
        
        # Model scores tracking
        scenario_scores = {}
        for model, score in model_scores.items():
            scenario_scores[model] = score
            analysis['model_scores']['by_type'][prompt_type][model].append(score)
            analysis['model_scores']['averages'][model].append(score)
        
        analysis['model_scores']['by_scenario'].append({
            'scenario': i + 1,
            'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
            'prompt_type': prompt_type,
            'selected_model': selected_model,
            'confidence': confidence,
            'scores': scenario_scores
        })
        
        # Identify competitive scenarios (margin < 0.6)
        if model_scores:
            scores = list(model_scores.values())
            max_score = max(scores)
            second_max = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0
            margin = max_score - second_max
            analysis['score_margins'].append(margin)
            
            if margin < 0.6:
                analysis['competitive_scenarios'].append({
                    'prompt': prompt,
                    'selected_model': selected_model,
                    'confidence': confidence,
                    'margin': margin,
                    'scores': model_scores
                })
        
        # Identify crisis scenarios
        crisis_keywords = ['kill', 'suicide', 'hurt myself', 'end it', 'pain', 'giving up']
        if any(keyword in prompt.lower() for keyword in crisis_keywords):
            analysis['crisis_scenarios'].append({
                'prompt': prompt,
                'selected_model': selected_model,
                'confidence': confidence,
                'scores': model_scores,
                'response': item.get('response', '')[:100] + '...'
            })
        
        # Track best performance examples (high confidence + high score)
        if confidence > 0.65 and model_scores.get(selected_model, 0) > 8.5:
            analysis['best_performance_examples'].append({
                'prompt': prompt,
                'selected_model': selected_model,
                'confidence': confidence,
                'score': model_scores.get(selected_model, 0),
                'response': item.get('response', '')[:150] + '...'
            })
    
    # Calculate derived statistics
    analysis['stats'] = calculate_derived_stats(analysis)
    
    return analysis

def calculate_derived_stats(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate derived statistics from raw data"""
    
    stats = {}
    
    # Response time statistics
    if analysis['response_time_stats']['all']:
        times = analysis['response_time_stats']['all']
        stats['response_time'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'q25': np.percentile(times, 25),
            'q75': np.percentile(times, 75)
        }
    
    # Confidence statistics
    if analysis['confidence_stats']['all']:
        confidences = analysis['confidence_stats']['all']
        stats['confidence'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences),
            'high_confidence_rate': len([c for c in confidences if c > 0.65]) / len(confidences)
        }
    
    # Model score statistics
    for model, scores in analysis['model_scores']['averages'].items():
        if scores:
            stats[f'{model}_scores'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            }
    
    # Score margin statistics
    if analysis['score_margins']:
        margins = analysis['score_margins']
        stats['score_margins'] = {
            'mean': np.mean(margins),
            'std': np.std(margins),
            'competitive_rate': len([m for m in margins if m < 0.5]) / len(margins),
            'dominant_rate': len([m for m in margins if m > 1.5]) / len(margins)
        }
    
    return stats

def print_capstone_statistics(analysis: Dict[str, Any]):
    """Print formatted statistics for capstone documentation"""
    
    print("# Capstone Statistics Extract")
    print("=" * 50)
    
    # Basic metrics
    print(f"\n## Dataset Overview")
    print(f"- Total scenarios: {analysis['total_scenarios']}")
    print(f"- Successful evaluations: {analysis['successful_scenarios']}")
    print(f"- Success rate: {analysis['successful_scenarios']/analysis['total_scenarios']*100:.1f}%")
    
    # Model distribution
    print(f"\n## Model Selection Distribution")
    total_selections = sum(analysis['model_distribution'].values())
    for model, count in sorted(analysis['model_distribution'].items()):
        percentage = count / total_selections * 100 if total_selections > 0 else 0
        print(f"- {model.upper()}: {count}/{total_selections} ({percentage:.1f}%)")
    
    # Confidence statistics
    if 'confidence' in analysis['stats']:
        conf_stats = analysis['stats']['confidence']
        print(f"\n## Confidence Score Analysis")
        print(f"- Mean: {conf_stats['mean']:.3f} ({conf_stats['mean']*100:.1f}%)")
        print(f"- Standard deviation: {conf_stats['std']:.3f}")
        print(f"- Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
        print(f"- High confidence rate (>65%): {conf_stats['high_confidence_rate']*100:.1f}%")
    
    # Response time statistics
    if 'response_time' in analysis['stats']:
        rt_stats = analysis['stats']['response_time']
        print(f"\n## Response Time Performance")
        print(f"- Mean: {rt_stats['mean']:.3f} seconds")
        print(f"- Standard deviation: {rt_stats['std']:.3f} seconds")
        print(f"- Range: {rt_stats['min']:.3f} - {rt_stats['max']:.3f} seconds")
        print(f"- Median: {rt_stats['median']:.3f} seconds")
    
    # Model performance comparison
    print(f"\n## Model Score Comparison")
    model_avgs = []
    for model in ['claude', 'openai', 'deepseek', 'gemma']:
        if f'{model}_scores' in analysis['stats']:
            stats = analysis['stats'][f'{model}_scores']
            model_avgs.append((model, stats['mean']))
            print(f"- {model.upper()}: {stats['mean']:.2f} ± {stats['std']:.2f} (n={stats['count']})")
    
    # Top competitive scenarios
    print(f"\n## Most Competitive Scenarios")
    competitive = sorted(analysis['competitive_scenarios'], key=lambda x: x['margin'])[:5]
    for i, scenario in enumerate(competitive, 1):
        print(f"{i}. \"{scenario['prompt'][:50]}...\" (margin: {scenario['margin']:.2f})")
    
    # Crisis scenarios
    print(f"\n## Crisis Scenario Analysis")
    print(f"- Total crisis scenarios identified: {len(analysis['crisis_scenarios'])}")
    if analysis['crisis_scenarios']:
        for i, crisis in enumerate(analysis['crisis_scenarios'][:3], 1):
            print(f"{i}. \"{crisis['prompt'][:50]}...\"")
            print(f"   Selected: {crisis['selected_model']} ({crisis['confidence']:.1%} confidence)")
    
    # Best performance examples
    print(f"\n## Highest Quality Examples")
    best = sorted(analysis['best_performance_examples'], 
                 key=lambda x: x['confidence'] * x['score'], reverse=True)[:3]
    for i, example in enumerate(best, 1):
        print(f"{i}. Score: {example['score']:.2f}, Confidence: {example['confidence']:.1%}")
        print(f"   Prompt: \"{example['prompt'][:60]}...\"")
        print(f"   Response: \"{example['response'][:80]}...\"")

def main():
    """Main analysis function"""
    
    # Find latest research data
    import os
    results_dir = "results/development"
    research_dirs = [d for d in os.listdir(results_dir) if d.startswith("research_data_")]
    
    if not research_dirs:
        print("No research data found")
        return
    
    latest_dir = sorted(research_dirs)[-1]
    data_path = os.path.join(results_dir, latest_dir, "research_data.json")
    
    print(f"Analyzing: {data_path}")
    
    # Perform analysis
    analysis = analyze_research_data(data_path)
    
    # Print statistics
    print_capstone_statistics(analysis)
    
    # Save detailed analysis
    output_path = os.path.join(results_dir, latest_dir, "capstone_statistics.json")
    with open(output_path, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        import json
        json.dump(analysis, f, indent=2, default=convert_numpy)
    
    print(f"\n✅ Detailed analysis saved to: {output_path}")

if __name__ == "__main__":
    main()