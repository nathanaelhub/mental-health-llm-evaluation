"""
Cost vs Quality Trade-off Analysis

Advanced analysis system for understanding the relationship between model costs
and quality outcomes, providing insights for cost optimization while maintaining
therapeutic effectiveness.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

from ..chat.dynamic_model_selector import PromptType
from .feedback_system import UserFeedback, FeedbackCollector
from .research_tools import ResearchDataPoint

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of costs in the system"""
    MODEL_INFERENCE = "model_inference"
    API_CALLS = "api_calls"
    COMPUTE_RESOURCES = "compute_resources"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    OVERHEAD = "overhead"


@dataclass
class ModelCostProfile:
    """Cost profile for a specific model"""
    model_name: str
    
    # Cost per operation (in USD)
    cost_per_1k_tokens: float = 0.0
    cost_per_request: float = 0.0
    
    # Resource requirements
    compute_cost_per_hour: float = 0.0
    memory_gb_required: float = 0.0
    storage_gb_per_month: float = 0.0
    
    # Performance characteristics affecting cost
    avg_tokens_per_response: int = 200
    avg_response_time_ms: float = 1000.0
    
    # Quality metrics
    avg_user_satisfaction: float = 0.0
    safety_score: float = 0.0
    therapeutic_effectiveness: float = 0.0
    
    def calculate_cost_per_interaction(self, tokens_used: int = None) -> float:
        """Calculate total cost per interaction"""
        tokens = tokens_used or self.avg_tokens_per_response
        
        # Token-based cost
        token_cost = (tokens / 1000) * self.cost_per_1k_tokens
        
        # Request-based cost
        request_cost = self.cost_per_request
        
        # Compute cost (based on response time)
        compute_cost = (self.avg_response_time_ms / 1000 / 3600) * self.compute_cost_per_hour
        
        return token_cost + request_cost + compute_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_name': self.model_name,
            'cost_per_1k_tokens': self.cost_per_1k_tokens,
            'cost_per_request': self.cost_per_request,
            'compute_cost_per_hour': self.compute_cost_per_hour,
            'memory_gb_required': self.memory_gb_required,
            'storage_gb_per_month': self.storage_gb_per_month,
            'avg_tokens_per_response': self.avg_tokens_per_response,
            'avg_response_time_ms': self.avg_response_time_ms,
            'avg_user_satisfaction': self.avg_user_satisfaction,
            'safety_score': self.safety_score,
            'therapeutic_effectiveness': self.therapeutic_effectiveness,
            'cost_per_interaction': self.calculate_cost_per_interaction()
        }


@dataclass
class CostQualityDataPoint:
    """Individual data point for cost-quality analysis"""
    interaction_id: str
    timestamp: datetime
    
    # Model and context
    model_used: str
    prompt_type: PromptType
    
    # Cost metrics
    actual_cost_usd: float
    tokens_used: int
    compute_time_ms: float
    
    # Quality metrics
    user_satisfaction: Optional[float] = None
    automated_quality_score: Optional[float] = None
    safety_score: Optional[float] = None
    task_completion: Optional[bool] = None
    therapeutic_value: Optional[float] = None
    
    # Efficiency metrics
    cost_per_satisfaction_point: Optional[float] = None
    quality_per_dollar: Optional[float] = None
    
    def calculate_efficiency_metrics(self):
        """Calculate efficiency metrics"""
        if self.user_satisfaction and self.actual_cost_usd > 0:
            self.cost_per_satisfaction_point = self.actual_cost_usd / self.user_satisfaction
            self.quality_per_dollar = self.user_satisfaction / self.actual_cost_usd
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'interaction_id': self.interaction_id,
            'timestamp': self.timestamp.isoformat(),
            'model_used': self.model_used,
            'prompt_type': self.prompt_type.value,
            'actual_cost_usd': self.actual_cost_usd,
            'tokens_used': self.tokens_used,
            'compute_time_ms': self.compute_time_ms,
            'user_satisfaction': self.user_satisfaction,
            'automated_quality_score': self.automated_quality_score,
            'safety_score': self.safety_score,
            'task_completion': self.task_completion,
            'therapeutic_value': self.therapeutic_value,
            'cost_per_satisfaction_point': self.cost_per_satisfaction_point,
            'quality_per_dollar': self.quality_per_dollar
        }


@dataclass
class CostOptimizationRecommendation:
    """Recommendation for cost optimization"""
    recommendation_id: str
    priority: str  # high, medium, low
    category: str
    
    title: str
    description: str
    expected_savings_usd_per_month: float
    expected_quality_impact: float  # -1 to 1, negative means quality decrease
    implementation_effort: str  # low, medium, high
    
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'recommendation_id': self.recommendation_id,
            'priority': self.priority,
            'category': self.category,
            'title': self.title,
            'description': self.description,
            'expected_savings_usd_per_month': self.expected_savings_usd_per_month,
            'expected_quality_impact': self.expected_quality_impact,
            'implementation_effort': self.implementation_effort,
            'supporting_data': self.supporting_data
        }


class CostQualityAnalyzer:
    """
    Advanced cost vs quality analysis system
    
    Features:
    - Model cost profiling and tracking
    - Quality-adjusted cost analysis
    - Optimization recommendations
    - Trade-off visualization data
    - ROI analysis for different strategies
    """
    
    def __init__(self):
        # Default model cost profiles (would be configured from actual pricing)
        self.model_cost_profiles = {
            "gpt-3.5-turbo": ModelCostProfile(
                model_name="gpt-3.5-turbo",
                cost_per_1k_tokens=0.002,
                cost_per_request=0.0001,
                avg_tokens_per_response=180,
                avg_response_time_ms=800,
                avg_user_satisfaction=3.8,
                safety_score=0.85,
                therapeutic_effectiveness=0.75
            ),
            "gpt-4-turbo": ModelCostProfile(
                model_name="gpt-4-turbo",
                cost_per_1k_tokens=0.01,
                cost_per_request=0.0005,
                avg_tokens_per_response=220,
                avg_response_time_ms=1200,
                avg_user_satisfaction=4.2,
                safety_score=0.90,
                therapeutic_effectiveness=0.85
            ),
            "claude-3-sonnet": ModelCostProfile(
                model_name="claude-3-sonnet",
                cost_per_1k_tokens=0.003,
                cost_per_request=0.0002,
                avg_tokens_per_response=200,
                avg_response_time_ms=1000,
                avg_user_satisfaction=4.1,
                safety_score=0.92,
                therapeutic_effectiveness=0.88
            ),
            "claude-3-opus": ModelCostProfile(
                model_name="claude-3-opus",
                cost_per_1k_tokens=0.015,
                cost_per_request=0.001,
                avg_tokens_per_response=250,
                avg_response_time_ms=1500,
                avg_user_satisfaction=4.4,
                safety_score=0.95,
                therapeutic_effectiveness=0.92
            ),
            "claude-3-haiku": ModelCostProfile(
                model_name="claude-3-haiku",
                cost_per_1k_tokens=0.00025,
                cost_per_request=0.00005,
                avg_tokens_per_response=150,
                avg_response_time_ms=600,
                avg_user_satisfaction=3.9,
                safety_score=0.88,
                therapeutic_effectiveness=0.80
            )
        }
        
        # Historical data for analysis
        self.cost_quality_data: List[CostQualityDataPoint] = []
        
        logger.info("CostQualityAnalyzer initialized with model cost profiles")
    
    def add_interaction_data(self, 
                           interaction_id: str,
                           model_used: str,
                           prompt_type: PromptType,
                           tokens_used: int,
                           compute_time_ms: float,
                           user_satisfaction: float = None,
                           automated_quality_score: float = None,
                           safety_score: float = None,
                           task_completion: bool = None):
        """Add interaction data for cost-quality analysis"""
        
        # Calculate actual cost
        cost_profile = self.model_cost_profiles.get(model_used)
        if not cost_profile:
            logger.warning(f"No cost profile found for model: {model_used}")
            actual_cost = 0.001  # Default minimal cost
        else:
            actual_cost = cost_profile.calculate_cost_per_interaction(tokens_used)
        
        # Create data point
        data_point = CostQualityDataPoint(
            interaction_id=interaction_id,
            timestamp=datetime.now(),
            model_used=model_used,
            prompt_type=prompt_type,
            actual_cost_usd=actual_cost,
            tokens_used=tokens_used,
            compute_time_ms=compute_time_ms,
            user_satisfaction=user_satisfaction,
            automated_quality_score=automated_quality_score,
            safety_score=safety_score,
            task_completion=task_completion
        )
        
        # Calculate efficiency metrics
        data_point.calculate_efficiency_metrics()
        
        self.cost_quality_data.append(data_point)
        
        logger.debug(f"Added cost-quality data point: {model_used}, cost: ${actual_cost:.4f}")
    
    def analyze_cost_effectiveness_by_model(self) -> Dict[str, Any]:
        """Analyze cost-effectiveness for each model"""
        
        if not self.cost_quality_data:
            return {'error': 'No cost-quality data available'}
        
        # Group data by model
        model_data = defaultdict(list)
        for data_point in self.cost_quality_data:
            model_data[data_point.model_used].append(data_point)
        
        analysis = {}
        
        for model_name, data_points in model_data.items():
            # Calculate metrics
            costs = [dp.actual_cost_usd for dp in data_points]
            satisfactions = [dp.user_satisfaction for dp in data_points if dp.user_satisfaction is not None]
            quality_scores = [dp.automated_quality_score for dp in data_points if dp.automated_quality_score is not None]
            
            # Cost-effectiveness metrics
            avg_cost = statistics.mean(costs) if costs else 0
            avg_satisfaction = statistics.mean(satisfactions) if satisfactions else 0
            avg_quality = statistics.mean(quality_scores) if quality_scores else 0
            
            # Efficiency calculations
            cost_per_satisfaction = avg_cost / avg_satisfaction if avg_satisfaction > 0 else float('inf')
            quality_per_dollar = avg_satisfaction / avg_cost if avg_cost > 0 else 0
            
            # Value score (combining multiple factors)
            cost_profile = self.model_cost_profiles.get(model_name, ModelCostProfile(model_name))
            value_score = self._calculate_value_score(avg_satisfaction, avg_cost, cost_profile.safety_score)
            
            analysis[model_name] = {
                'total_interactions': len(data_points),
                'avg_cost_per_interaction': avg_cost,
                'avg_user_satisfaction': avg_satisfaction,
                'avg_quality_score': avg_quality,
                'cost_per_satisfaction_point': cost_per_satisfaction,
                'quality_per_dollar': quality_per_dollar,
                'value_score': value_score,
                'safety_score': cost_profile.safety_score,
                'total_cost_last_30_days': sum(costs),
                'cost_trend': self._calculate_cost_trend(data_points)
            }
        
        # Rank models by cost-effectiveness
        ranked_models = sorted(
            analysis.items(), 
            key=lambda x: x[1]['value_score'], 
            reverse=True
        )
        
        return {
            'model_analysis': analysis,
            'ranking_by_value': [{'model': model, 'value_score': data['value_score']} 
                               for model, data in ranked_models],
            'best_value_model': ranked_models[0][0] if ranked_models else None,
            'total_system_cost': sum(dp.actual_cost_usd for dp in self.cost_quality_data),
            'analysis_period_days': self._get_analysis_period_days()
        }
    
    def analyze_cost_by_prompt_type(self) -> Dict[str, Any]:
        """Analyze cost patterns by prompt type"""
        
        # Group by prompt type
        prompt_type_data = defaultdict(list)
        for data_point in self.cost_quality_data:
            prompt_type_data[data_point.prompt_type.value].append(data_point)
        
        analysis = {}
        
        for prompt_type, data_points in prompt_type_data.items():
            costs = [dp.actual_cost_usd for dp in data_points]
            satisfactions = [dp.user_satisfaction for dp in data_points if dp.user_satisfaction is not None]
            
            # Model usage distribution for this prompt type
            model_usage = defaultdict(int)
            model_costs = defaultdict(list)
            
            for dp in data_points:
                model_usage[dp.model_used] += 1
                model_costs[dp.model_used].append(dp.actual_cost_usd)
            
            # Calculate average cost per model for this prompt type
            avg_model_costs = {
                model: statistics.mean(costs) 
                for model, costs in model_costs.items()
            }
            
            analysis[prompt_type] = {
                'total_interactions': len(data_points),
                'avg_cost_per_interaction': statistics.mean(costs) if costs else 0,
                'total_cost': sum(costs),
                'avg_user_satisfaction': statistics.mean(satisfactions) if satisfactions else 0,
                'model_usage_distribution': dict(model_usage),
                'avg_cost_by_model': avg_model_costs,
                'most_used_model': max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else None,
                'cheapest_effective_model': self._find_cheapest_effective_model(data_points)
            }
        
        return {
            'prompt_type_analysis': analysis,
            'cost_ranking_by_prompt_type': sorted(
                analysis.items(),
                key=lambda x: x[1]['avg_cost_per_interaction'],
                reverse=True
            ),
            'optimization_opportunities': self._identify_prompt_type_optimizations(analysis)
        }
    
    def generate_cost_optimization_recommendations(self) -> List[CostOptimizationRecommendation]:
        """Generate actionable cost optimization recommendations"""
        
        recommendations = []
        
        # Analyze current system
        model_analysis = self.analyze_cost_effectiveness_by_model()
        prompt_analysis = self.analyze_cost_by_prompt_type()
        
        # Recommendation 1: Model substitution for low-stakes prompts
        general_wellness_data = prompt_analysis.get('prompt_type_analysis', {}).get('general_wellness')
        if general_wellness_data:
            current_cost = general_wellness_data['avg_cost_per_interaction']
            cheapest_model = general_wellness_data['cheapest_effective_model']
            
            if cheapest_model and current_cost > 0.001:  # If cost is above minimal threshold
                potential_savings = (current_cost - 0.0005) * general_wellness_data['total_interactions']
                
                recommendations.append(CostOptimizationRecommendation(
                    recommendation_id="model_substitution_wellness",
                    priority="medium",
                    category="model_optimization",
                    title="Use Cheaper Model for General Wellness Queries",
                    description=f"Switch to {cheapest_model} for general wellness prompts to reduce costs while maintaining quality",
                    expected_savings_usd_per_month=potential_savings * 30,  # Approximate monthly
                    expected_quality_impact=-0.1,  # Minor quality decrease
                    implementation_effort="low",
                    supporting_data={
                        'current_avg_cost': current_cost,
                        'recommended_model': cheapest_model,
                        'affected_interactions': general_wellness_data['total_interactions']
                    }
                ))
        
        # Recommendation 2: Caching optimization
        total_interactions = len(self.cost_quality_data)
        if total_interactions > 100:
            cache_potential_savings = total_interactions * 0.0002 * 0.3  # 30% cache hit rate
            
            recommendations.append(CostOptimizationRecommendation(
                recommendation_id="improve_caching",
                priority="high",
                category="system_optimization",
                title="Improve Response Caching",
                description="Implement semantic similarity caching to reduce redundant model calls",
                expected_savings_usd_per_month=cache_potential_savings * 30,
                expected_quality_impact=0.0,  # No quality impact
                implementation_effort="medium",
                supporting_data={
                    'total_interactions': total_interactions,
                    'estimated_cache_hit_rate': 0.3
                }
            ))
        
        # Recommendation 3: Crisis model optimization
        crisis_data = prompt_analysis.get('prompt_type_analysis', {}).get('crisis')
        if crisis_data and crisis_data['total_interactions'] > 10:
            # For crisis, prioritize safety over cost, but optimize non-critical aspects
            recommendations.append(CostOptimizationRecommendation(
                recommendation_id="crisis_model_optimization",
                priority="low",
                category="safety_optimization",
                title="Optimize Crisis Response Pipeline",
                description="Implement faster crisis detection to reduce unnecessary high-cost model usage",
                expected_savings_usd_per_month=crisis_data['total_cost'] * 0.1,  # 10% savings
                expected_quality_impact=0.05,  # Slight quality improvement
                implementation_effort="high",
                supporting_data={
                    'crisis_interactions': crisis_data['total_interactions'],
                    'current_cost': crisis_data['total_cost']
                }
            ))
        
        # Recommendation 4: Model switching optimization
        if len(model_analysis.get('model_analysis', {})) > 1:
            best_value_model = model_analysis.get('best_value_model')
            total_cost = model_analysis.get('total_system_cost', 0)
            
            if best_value_model and total_cost > 1.0:  # Significant cost volume
                recommendations.append(CostOptimizationRecommendation(
                    recommendation_id="increase_best_value_model_usage",
                    priority="medium",
                    category="model_optimization",
                    title=f"Increase Usage of Best Value Model ({best_value_model})",
                    description=f"Redirect more traffic to {best_value_model} which provides the best cost-quality ratio",
                    expected_savings_usd_per_month=total_cost * 0.15,  # 15% potential savings
                    expected_quality_impact=0.02,  # Slight improvement
                    implementation_effort="medium",
                    supporting_data={
                        'best_value_model': best_value_model,
                        'current_total_cost': total_cost
                    }
                ))
        
        # Sort recommendations by expected savings
        recommendations.sort(key=lambda r: r.expected_savings_usd_per_month, reverse=True)
        
        return recommendations
    
    def calculate_roi_for_quality_improvements(self) -> Dict[str, Any]:
        """Calculate ROI for potential quality improvements"""
        
        if not self.cost_quality_data:
            return {'error': 'No data available for ROI analysis'}
        
        # Current system performance
        current_satisfaction = statistics.mean([
            dp.user_satisfaction for dp in self.cost_quality_data 
            if dp.user_satisfaction is not None
        ])
        
        current_cost = statistics.mean([dp.actual_cost_usd for dp in self.cost_quality_data])
        
        # Model improvement scenarios
        scenarios = {
            'upgrade_to_premium': {
                'description': 'Upgrade all requests to highest quality model',
                'cost_multiplier': 3.0,
                'satisfaction_improvement': 0.5,
                'implementation_cost': 1000  # One-time cost
            },
            'smart_routing': {
                'description': 'Implement intelligent model routing based on prompt complexity',
                'cost_multiplier': 1.2,
                'satisfaction_improvement': 0.3,
                'implementation_cost': 5000
            },
            'enhanced_caching': {
                'description': 'Implement advanced semantic caching',
                'cost_multiplier': 0.7,
                'satisfaction_improvement': 0.0,
                'implementation_cost': 2000
            }
        }
        
        # Calculate ROI for each scenario
        roi_analysis = {}
        total_monthly_interactions = len(self.cost_quality_data) * 30  # Approximate monthly volume
        
        for scenario_name, scenario in scenarios.items():
            new_cost_per_interaction = current_cost * scenario['cost_multiplier']
            new_satisfaction = min(5.0, current_satisfaction + scenario['satisfaction_improvement'])
            
            # Calculate monthly costs
            monthly_cost_current = current_cost * total_monthly_interactions
            monthly_cost_new = new_cost_per_interaction * total_monthly_interactions
            monthly_cost_difference = monthly_cost_new - monthly_cost_current
            
            # Estimate value of satisfaction improvement
            # Assume each satisfaction point is worth $0.50 per interaction in user retention/value
            satisfaction_value_per_interaction = scenario['satisfaction_improvement'] * 0.50
            monthly_satisfaction_value = satisfaction_value_per_interaction * total_monthly_interactions
            
            # Calculate ROI
            monthly_net_benefit = monthly_satisfaction_value - monthly_cost_difference
            payback_months = scenario['implementation_cost'] / monthly_net_benefit if monthly_net_benefit > 0 else float('inf')
            
            roi_analysis[scenario_name] = {
                'description': scenario['description'],
                'implementation_cost': scenario['implementation_cost'],
                'monthly_cost_change': monthly_cost_difference,
                'monthly_satisfaction_value': monthly_satisfaction_value,
                'monthly_net_benefit': monthly_net_benefit,
                'payback_months': payback_months,
                'annual_roi_percent': (monthly_net_benefit * 12 / scenario['implementation_cost']) * 100 if scenario['implementation_cost'] > 0 else 0,
                'new_satisfaction_score': new_satisfaction,
                'new_cost_per_interaction': new_cost_per_interaction
            }
        
        return {
            'current_metrics': {
                'avg_satisfaction': current_satisfaction,
                'avg_cost_per_interaction': current_cost,
                'monthly_interactions': total_monthly_interactions,
                'monthly_total_cost': monthly_cost_current
            },
            'scenarios': roi_analysis,
            'recommended_scenario': max(roi_analysis.items(), key=lambda x: x[1]['annual_roi_percent'])[0] if roi_analysis else None
        }
    
    def generate_cost_quality_visualization_data(self) -> Dict[str, Any]:
        """Generate data for cost vs quality visualizations"""
        
        if not self.cost_quality_data:
            return {'error': 'No data available'}
        
        # Scatter plot data: cost vs satisfaction
        scatter_data = []
        for dp in self.cost_quality_data:
            if dp.user_satisfaction is not None:
                scatter_data.append({
                    'x': dp.actual_cost_usd * 1000,  # Convert to cents for better visualization
                    'y': dp.user_satisfaction,
                    'model': dp.model_used,
                    'prompt_type': dp.prompt_type.value,
                    'size': dp.tokens_used
                })
        
        # Model comparison data
        model_comparison = {}
        for model_name, profile in self.model_cost_profiles.items():
            model_data = [dp for dp in self.cost_quality_data if dp.model_used == model_name]
            
            if model_data:
                actual_satisfaction = statistics.mean([
                    dp.user_satisfaction for dp in model_data 
                    if dp.user_satisfaction is not None
                ])
                actual_cost = statistics.mean([dp.actual_cost_usd for dp in model_data])
            else:
                actual_satisfaction = profile.avg_user_satisfaction
                actual_cost = profile.calculate_cost_per_interaction()
            
            model_comparison[model_name] = {
                'cost_cents': actual_cost * 100,
                'satisfaction': actual_satisfaction,
                'safety_score': profile.safety_score,
                'usage_count': len(model_data)
            }
        
        # Efficiency frontier (Pareto optimal points)
        efficiency_frontier = self._calculate_efficiency_frontier(model_comparison)
        
        # Time series data
        time_series = self._generate_cost_time_series()
        
        return {
            'scatter_plot': {
                'data': scatter_data,
                'xlabel': 'Cost (cents per interaction)',
                'ylabel': 'User Satisfaction (1-5)',
                'title': 'Cost vs Quality Trade-off Analysis'
            },
            'model_comparison': {
                'data': model_comparison,
                'title': 'Model Performance vs Cost'
            },
            'efficiency_frontier': {
                'data': efficiency_frontier,
                'title': 'Cost-Quality Efficiency Frontier'
            },
            'time_series': time_series,
            'summary_stats': {
                'total_interactions': len(self.cost_quality_data),
                'total_cost': sum(dp.actual_cost_usd for dp in self.cost_quality_data),
                'avg_satisfaction': statistics.mean([
                    dp.user_satisfaction for dp in self.cost_quality_data 
                    if dp.user_satisfaction is not None
                ]),
                'cost_per_satisfaction_point': self._calculate_global_cost_per_satisfaction()
            }
        }
    
    def _calculate_value_score(self, satisfaction: float, cost: float, safety_score: float) -> float:
        """Calculate a composite value score combining satisfaction, cost, and safety"""
        if cost <= 0:
            return 0
        
        # Weighted combination: satisfaction (50%), cost efficiency (30%), safety (20%)
        satisfaction_component = (satisfaction / 5.0) * 0.5
        cost_efficiency_component = min(1.0, (0.01 / cost)) * 0.3  # Inverse of cost, capped at 1.0
        safety_component = safety_score * 0.2
        
        return satisfaction_component + cost_efficiency_component + safety_component
    
    def _calculate_cost_trend(self, data_points: List[CostQualityDataPoint]) -> str:
        """Calculate cost trend over time"""
        if len(data_points) < 2:
            return "insufficient_data"
        
        # Sort by timestamp
        sorted_points = sorted(data_points, key=lambda dp: dp.timestamp)
        
        # Take first and last quarter of data
        quarter_size = len(sorted_points) // 4
        if quarter_size < 1:
            return "insufficient_data"
        
        early_costs = [dp.actual_cost_usd for dp in sorted_points[:quarter_size]]
        late_costs = [dp.actual_cost_usd for dp in sorted_points[-quarter_size:]]
        
        early_avg = statistics.mean(early_costs)
        late_avg = statistics.mean(late_costs)
        
        change_percent = ((late_avg - early_avg) / early_avg) * 100 if early_avg > 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _get_analysis_period_days(self) -> int:
        """Get the number of days covered by the analysis"""
        if not self.cost_quality_data:
            return 0
        
        timestamps = [dp.timestamp for dp in self.cost_quality_data]
        period = max(timestamps) - min(timestamps)
        return period.days
    
    def _find_cheapest_effective_model(self, data_points: List[CostQualityDataPoint]) -> Optional[str]:
        """Find the cheapest model that maintains effectiveness"""
        
        # Group by model and calculate effectiveness
        model_performance = defaultdict(list)
        
        for dp in data_points:
            if dp.user_satisfaction is not None:
                model_performance[dp.model_used].append({
                    'cost': dp.actual_cost_usd,
                    'satisfaction': dp.user_satisfaction
                })
        
        # Find cheapest model with satisfaction >= 3.5
        cheapest_effective = None
        lowest_cost = float('inf')
        
        for model, performances in model_performance.items():
            if len(performances) >= 3:  # Need minimum data
                avg_satisfaction = statistics.mean([p['satisfaction'] for p in performances])
                avg_cost = statistics.mean([p['cost'] for p in performances])
                
                if avg_satisfaction >= 3.5 and avg_cost < lowest_cost:
                    cheapest_effective = model
                    lowest_cost = avg_cost
        
        return cheapest_effective
    
    def _identify_prompt_type_optimizations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities by prompt type"""
        
        opportunities = []
        
        for prompt_type, data in analysis.items():
            avg_cost = data['avg_cost_per_interaction']
            total_cost = data['total_cost']
            
            # High-cost prompt types with optimization potential
            if avg_cost > 0.005 and total_cost > 0.10:
                opportunities.append({
                    'prompt_type': prompt_type,
                    'issue': 'high_cost',
                    'current_cost': avg_cost,
                    'potential_savings': total_cost * 0.2,  # 20% potential savings
                    'recommendation': f"Review model selection strategy for {prompt_type} prompts"
                })
            
            # Low satisfaction with high cost
            if data['avg_user_satisfaction'] < 3.5 and avg_cost > 0.002:
                opportunities.append({
                    'prompt_type': prompt_type,
                    'issue': 'poor_value',
                    'satisfaction': data['avg_user_satisfaction'],
                    'cost': avg_cost,
                    'recommendation': f"Improve model selection or consider alternative approach for {prompt_type}"
                })
        
        return opportunities
    
    def _calculate_efficiency_frontier(self, model_comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate Pareto efficiency frontier for cost vs quality"""
        
        models = list(model_comparison.items())
        frontier_points = []
        
        # Sort by cost (ascending)
        models.sort(key=lambda x: x[1]['cost_cents'])
        
        max_satisfaction_so_far = 0
        
        for model_name, data in models:
            satisfaction = data['satisfaction']
            
            # Point is on frontier if it has higher satisfaction than all cheaper models
            if satisfaction > max_satisfaction_so_far:
                frontier_points.append({
                    'model': model_name,
                    'cost_cents': data['cost_cents'],
                    'satisfaction': satisfaction,
                    'is_efficient': True
                })
                max_satisfaction_so_far = satisfaction
        
        return frontier_points
    
    def _generate_cost_time_series(self) -> Dict[str, Any]:
        """Generate time series data for cost trends"""
        
        if not self.cost_quality_data:
            return {}
        
        # Group by day
        daily_costs = defaultdict(list)
        daily_satisfaction = defaultdict(list)
        
        for dp in self.cost_quality_data:
            day_key = dp.timestamp.strftime('%Y-%m-%d')
            daily_costs[day_key].append(dp.actual_cost_usd)
            if dp.user_satisfaction is not None:
                daily_satisfaction[day_key].append(dp.user_satisfaction)
        
        # Calculate daily averages
        time_series_data = []
        for day in sorted(daily_costs.keys()):
            avg_cost = statistics.mean(daily_costs[day])
            avg_satisfaction = statistics.mean(daily_satisfaction[day]) if daily_satisfaction[day] else None
            
            time_series_data.append({
                'date': day,
                'avg_cost_cents': avg_cost * 100,
                'avg_satisfaction': avg_satisfaction,
                'interaction_count': len(daily_costs[day])
            })
        
        return {
            'data': time_series_data,
            'title': 'Daily Cost and Quality Trends'
        }
    
    def _calculate_global_cost_per_satisfaction(self) -> float:
        """Calculate global cost per satisfaction point"""
        
        data_with_satisfaction = [
            dp for dp in self.cost_quality_data 
            if dp.user_satisfaction is not None
        ]
        
        if not data_with_satisfaction:
            return 0.0
        
        total_cost = sum(dp.actual_cost_usd for dp in data_with_satisfaction)
        total_satisfaction = sum(dp.user_satisfaction for dp in data_with_satisfaction)
        
        return total_cost / total_satisfaction if total_satisfaction > 0 else 0.0
    
    def export_cost_analysis_report(self, output_file: str):
        """Export comprehensive cost analysis report"""
        
        report = {
            'export_timestamp': datetime.now().isoformat(),
            'analysis_period_days': self._get_analysis_period_days(),
            'total_interactions': len(self.cost_quality_data),
            
            'model_cost_effectiveness': self.analyze_cost_effectiveness_by_model(),
            'prompt_type_analysis': self.analyze_cost_by_prompt_type(),
            'optimization_recommendations': [rec.to_dict() for rec in self.generate_cost_optimization_recommendations()],
            'roi_analysis': self.calculate_roi_for_quality_improvements(),
            'visualization_data': self.generate_cost_quality_visualization_data(),
            
            'model_cost_profiles': {
                name: profile.to_dict() 
                for name, profile in self.model_cost_profiles.items()
            }
        }
        
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Cost analysis report exported to {output_file}")
        return output_file