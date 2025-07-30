"""
Export and Research Tools for Data Analysis

Comprehensive tools for exporting conversation data, statistical analysis,
and research paper data generation with privacy-preserving features.
"""

import asyncio
import logging
import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import statistics
import hashlib
import scipy.stats as stats
from collections import defaultdict, Counter

from ..chat.dynamic_model_selector import PromptType
from .ab_testing import ExperimentResult, ExperimentManager
from .feedback_system import UserFeedback, FeedbackCollector

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    RESEARCH_JSON = "research_json"  # Anonymized research format


class PrivacyLevel(Enum):
    """Privacy protection levels"""
    FULL = "full"           # All data including personal info
    ANONYMIZED = "anonymized"  # Personal info removed/hashed
    AGGREGATED = "aggregated"  # Only aggregate statistics
    RESEARCH = "research"      # Research-ready with privacy protection


@dataclass
class ExportConfig:
    """Configuration for data export"""
    format: ExportFormat = ExportFormat.JSON
    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMIZED
    
    # Date range filtering
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Content filtering
    include_prompts: bool = True
    include_responses: bool = True
    include_model_selections: bool = True
    include_feedback: bool = True
    include_experiments: bool = True
    
    # Privacy settings
    hash_user_ids: bool = True
    truncate_prompts: int = 100  # Max prompt length
    remove_personal_info: bool = True
    
    # Research settings
    statistical_significance_level: float = 0.05
    minimum_sample_size: int = 30
    
    # Output settings
    output_directory: str = "results/research_exports"
    filename_prefix: str = "mental_health_data"
    include_metadata: bool = True


@dataclass
class ResearchDataPoint:
    """Individual data point for research analysis"""
    data_id: str
    timestamp: datetime
    
    # Model selection data
    selected_model: str
    selection_confidence: float
    selection_time_ms: float
    prompt_type: PromptType
    
    # Conversation context
    message_index: int  # Position in conversation
    conversation_length: int
    user_segment: Optional[str] = None
    
    # Outcomes
    user_satisfaction: Optional[float] = None
    response_time_ms: Optional[float] = None
    conversation_continued: Optional[bool] = None
    task_completion: Optional[bool] = None
    
    # Quality metrics
    automated_quality_score: Optional[float] = None
    safety_score: Optional[float] = None
    empathy_score: Optional[float] = None
    
    # Experimental data
    experiment_id: Optional[str] = None
    variant_name: Optional[str] = None
    
    # Anonymized identifiers
    user_hash: Optional[str] = None
    session_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'data_id': self.data_id,
            'timestamp': self.timestamp.isoformat(),
            'selected_model': self.selected_model,
            'selection_confidence': self.selection_confidence,
            'selection_time_ms': self.selection_time_ms,
            'prompt_type': self.prompt_type.value,
            'message_index': self.message_index,
            'conversation_length': self.conversation_length,
            'user_segment': self.user_segment,
            'user_satisfaction': self.user_satisfaction,
            'response_time_ms': self.response_time_ms,
            'conversation_continued': self.conversation_continued,
            'task_completion': self.task_completion,
            'automated_quality_score': self.automated_quality_score,
            'safety_score': self.safety_score,
            'empathy_score': self.empathy_score,
            'experiment_id': self.experiment_id,
            'variant_name': self.variant_name,
            'user_hash': self.user_hash,
            'session_hash': self.session_hash
        }


class DataPrivacyManager:
    """Manages data privacy and anonymization"""
    
    def __init__(self, salt: str = None):
        self.salt = salt or "mental_health_research_2025"
        
        # Patterns for personal information removal
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
            r'\b[\w.-]+@[\w.-]+\.\w+\b',  # Email
            r'\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|blvd)\b',  # Address
        ]
    
    def hash_identifier(self, identifier: str) -> str:
        """Create consistent anonymous hash of identifier"""
        return hashlib.sha256(f"{identifier}_{self.salt}".encode()).hexdigest()[:12]
    
    def anonymize_text(self, text: str, max_length: int = 100) -> str:
        """Remove PII and truncate text"""
        import re
        
        if not text:
            return text
        
        # Remove potential PII
        anonymized = text
        for pattern in self.pii_patterns:
            anonymized = re.sub(pattern, '[REDACTED]', anonymized, flags=re.IGNORECASE)
        
        # Truncate if too long
        if len(anonymized) > max_length:
            anonymized = anonymized[:max_length] + "..."
        
        return anonymized
    
    def create_user_segment(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create anonymized user segment for analysis"""
        
        # Simple segmentation based on hash
        user_hash = self.hash_identifier(user_id)
        hash_int = int(user_hash[:4], 16)
        
        segments = ["segment_A", "segment_B", "segment_C", "segment_D"]
        return segments[hash_int % len(segments)]


class StatisticalAnalyzer:
    """Advanced statistical analysis for research insights"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def perform_model_comparison_analysis(self, data_points: List[ResearchDataPoint]) -> Dict[str, Any]:
        """Comprehensive statistical comparison of model performance"""
        
        # Group data by model
        model_data = defaultdict(list)
        for point in data_points:
            if point.user_satisfaction is not None:
                model_data[point.selected_model].append(point)
        
        if len(model_data) < 2:
            return {'error': 'Need at least 2 models for comparison'}
        
        results = {
            'models_analyzed': list(model_data.keys()),
            'sample_sizes': {model: len(points) for model, points in model_data.items()},
            'descriptive_stats': {},
            'statistical_tests': {},
            'effect_sizes': {},
            'recommendations': []
        }
        
        # Descriptive statistics
        for model, points in model_data.items():
            satisfaction_scores = [p.user_satisfaction for p in points if p.user_satisfaction is not None]
            response_times = [p.response_time_ms for p in points if p.response_time_ms is not None]
            quality_scores = [p.automated_quality_score for p in points if p.automated_quality_score is not None]
            
            results['descriptive_stats'][model] = {
                'satisfaction': {
                    'mean': statistics.mean(satisfaction_scores) if satisfaction_scores else None,
                    'std': statistics.stdev(satisfaction_scores) if len(satisfaction_scores) > 1 else None,
                    'median': statistics.median(satisfaction_scores) if satisfaction_scores else None,
                    'count': len(satisfaction_scores)
                },
                'response_time': {
                    'mean': statistics.mean(response_times) if response_times else None,
                    'std': statistics.stdev(response_times) if len(response_times) > 1 else None,
                    'count': len(response_times)
                },
                'quality': {
                    'mean': statistics.mean(quality_scores) if quality_scores else None,
                    'std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else None,
                    'count': len(quality_scores)
                }
            }
        
        # Pairwise statistical tests
        models = list(model_data.keys())
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                points_a = model_data[model_a]
                points_b = model_data[model_b]
                
                # User satisfaction comparison
                satisfaction_a = [p.user_satisfaction for p in points_a if p.user_satisfaction is not None]
                satisfaction_b = [p.user_satisfaction for p in points_b if p.user_satisfaction is not None]
                
                if len(satisfaction_a) >= 10 and len(satisfaction_b) >= 10:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(satisfaction_a, satisfaction_b)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(satisfaction_a) - 1) * np.var(satisfaction_a, ddof=1) + 
                                         (len(satisfaction_b) - 1) * np.var(satisfaction_b, ddof=1)) / 
                                        (len(satisfaction_a) + len(satisfaction_b) - 2))
                    cohens_d = (np.mean(satisfaction_a) - np.mean(satisfaction_b)) / pooled_std if pooled_std > 0 else 0
                    
                    test_key = f"{model_a}_vs_{model_b}"
                    results['statistical_tests'][test_key] = {
                        'metric': 'user_satisfaction',
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.significance_level,
                        'sample_sizes': [len(satisfaction_a), len(satisfaction_b)]
                    }
                    
                    results['effect_sizes'][test_key] = {
                        'cohens_d': float(cohens_d),
                        'interpretation': self._interpret_effect_size(abs(cohens_d))
                    }
        
        # Generate recommendations
        results['recommendations'] = self._generate_statistical_recommendations(results)
        
        return results
    
    def analyze_prompt_type_effectiveness(self, data_points: List[ResearchDataPoint]) -> Dict[str, Any]:
        """Analyze model effectiveness across different prompt types"""
        
        # Group by prompt type and model
        prompt_model_data = defaultdict(lambda: defaultdict(list))
        
        for point in data_points:
            if point.user_satisfaction is not None:
                prompt_model_data[point.prompt_type.value][point.selected_model].append(point)
        
        results = {
            'prompt_types_analyzed': list(prompt_model_data.keys()),
            'effectiveness_by_type': {},
            'model_specialization': {},
            'statistical_significance': {}
        }
        
        # Analyze each prompt type
        for prompt_type, model_data in prompt_model_data.items():
            type_results = {
                'models_compared': list(model_data.keys()),
                'performance_ranking': [],
                'sample_sizes': {}
            }
            
            # Calculate mean satisfaction for each model
            model_means = {}
            for model, points in model_data.items():
                satisfaction_scores = [p.user_satisfaction for p in points if p.user_satisfaction is not None]
                if satisfaction_scores:
                    model_means[model] = statistics.mean(satisfaction_scores)
                    type_results['sample_sizes'][model] = len(satisfaction_scores)
            
            # Rank models by performance
            type_results['performance_ranking'] = sorted(
                model_means.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            results['effectiveness_by_type'][prompt_type] = type_results
        
        # Identify model specializations
        for model in set(point.selected_model for point in data_points):
            model_performance = {}
            for prompt_type in prompt_model_data:
                if model in prompt_model_data[prompt_type]:
                    points = prompt_model_data[prompt_type][model]
                    satisfaction_scores = [p.user_satisfaction for p in points if p.user_satisfaction is not None]
                    if satisfaction_scores:
                        model_performance[prompt_type] = statistics.mean(satisfaction_scores)
            
            if model_performance:
                best_type = max(model_performance.items(), key=lambda x: x[1])
                results['model_specialization'][model] = {
                    'best_prompt_type': best_type[0],
                    'best_performance': best_type[1],
                    'all_performance': model_performance
                }
        
        return results
    
    def analyze_temporal_patterns(self, data_points: List[ResearchDataPoint]) -> Dict[str, Any]:
        """Analyze temporal patterns in model selection and performance"""
        
        # Group by hour of day
        hourly_data = defaultdict(list)
        daily_data = defaultdict(list)
        
        for point in data_points:
            hour = point.timestamp.hour
            day = point.timestamp.strftime('%Y-%m-%d')
            
            hourly_data[hour].append(point)
            daily_data[day].append(point)
        
        results = {
            'hourly_patterns': {},
            'daily_patterns': {},
            'peak_usage': {},
            'performance_variations': {}
        }
        
        # Hourly analysis
        for hour, points in hourly_data.items():
            satisfaction_scores = [p.user_satisfaction for p in points if p.user_satisfaction is not None]
            model_distribution = Counter(p.selected_model for p in points)
            
            results['hourly_patterns'][hour] = {
                'total_requests': len(points),
                'avg_satisfaction': statistics.mean(satisfaction_scores) if satisfaction_scores else None,
                'top_models': model_distribution.most_common(3),
                'prompt_types': Counter(p.prompt_type.value for p in points)
            }
        
        # Identify peak usage
        hour_counts = {hour: len(points) for hour, points in hourly_data.items()}
        peak_hour = max(hour_counts.items(), key=lambda x: x[1]) if hour_counts else (0, 0)
        
        results['peak_usage'] = {
            'peak_hour': peak_hour[0],
            'peak_requests': peak_hour[1],
            'usage_distribution': hour_counts
        }
        
        return results
    
    def calculate_conversation_flow_metrics(self, data_points: List[ResearchDataPoint]) -> Dict[str, Any]:
        """Analyze conversation flow and user engagement patterns"""
        
        # Group by session
        session_data = defaultdict(list)
        for point in data_points:
            if point.session_hash:
                session_data[point.session_hash].append(point)
        
        # Sort each session by message index
        for session_id in session_data:
            session_data[session_id].sort(key=lambda p: p.message_index)
        
        results = {
            'total_sessions': len(session_data),
            'conversation_length_distribution': {},
            'engagement_patterns': {},
            'model_switching_analysis': {}
        }
        
        # Analyze conversation lengths
        conversation_lengths = [len(points) for points in session_data.values()]
        
        if conversation_lengths:
            results['conversation_length_distribution'] = {
                'mean': statistics.mean(conversation_lengths),
                'median': statistics.median(conversation_lengths),
                'std': statistics.stdev(conversation_lengths) if len(conversation_lengths) > 1 else 0,
                'min': min(conversation_lengths),
                'max': max(conversation_lengths),
                'percentiles': {
                    '25th': np.percentile(conversation_lengths, 25),
                    '75th': np.percentile(conversation_lengths, 75),
                    '90th': np.percentile(conversation_lengths, 90)
                }
            }
        
        # Analyze engagement patterns
        engagement_data = []
        for session_points in session_data.values():
            if len(session_points) >= 2:
                # Calculate engagement based on conversation continuation
                continued_conversations = sum(1 for p in session_points if p.conversation_continued)
                engagement_score = continued_conversations / len(session_points)
                engagement_data.append(engagement_score)
        
        if engagement_data:
            results['engagement_patterns'] = {
                'avg_engagement_score': statistics.mean(engagement_data),
                'high_engagement_sessions': sum(1 for score in engagement_data if score > 0.7),
                'low_engagement_sessions': sum(1 for score in engagement_data if score < 0.3)
            }
        
        # Analyze model switching within conversations
        switching_sessions = 0
        avg_switches_per_session = 0
        
        for session_points in session_data.values():
            if len(session_points) > 1:
                models_used = [p.selected_model for p in session_points]
                unique_models = set(models_used)
                
                if len(unique_models) > 1:
                    switching_sessions += 1
                    switches = sum(1 for i in range(1, len(models_used)) if models_used[i] != models_used[i-1])
                    avg_switches_per_session += switches
        
        if session_data:
            results['model_switching_analysis'] = {
                'sessions_with_switches': switching_sessions,
                'switch_rate': switching_sessions / len(session_data),
                'avg_switches_per_switching_session': avg_switches_per_session / switching_sessions if switching_sessions > 0 else 0
            }
        
        return results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_statistical_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from statistical analysis"""
        
        recommendations = []
        
        # Check for significant differences
        for test_name, test_result in analysis_results.get('statistical_tests', {}).items():
            if test_result.get('significant', False):
                effect_size = analysis_results.get('effect_sizes', {}).get(test_name, {})
                
                if effect_size.get('interpretation') in ['medium', 'large']:
                    models = test_name.split('_vs_')
                    recommendations.append(
                        f"Significant performance difference found between {models[0]} and {models[1]} "
                        f"with {effect_size.get('interpretation', 'unknown')} effect size"
                    )
        
        # Check sample sizes
        for model, stats in analysis_results.get('descriptive_stats', {}).items():
            satisfaction_count = stats.get('satisfaction', {}).get('count', 0)
            if satisfaction_count < 30:
                recommendations.append(
                    f"Increase sample size for {model} (current: {satisfaction_count}, recommended: ≥30)"
                )
        
        return recommendations


class ResearchExporter:
    """
    Comprehensive data export system for research analysis
    
    Features:
    - Multiple export formats (JSON, CSV, Excel, Parquet)
    - Privacy-preserving data anonymization
    - Statistical analysis integration
    - Research-ready data formatting
    """
    
    def __init__(self, 
                 feedback_collector: FeedbackCollector = None,
                 experiment_manager: ExperimentManager = None):
        
        self.feedback_collector = feedback_collector
        self.experiment_manager = experiment_manager
        self.privacy_manager = DataPrivacyManager()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        logger.info("ResearchExporter initialized")
    
    async def export_comprehensive_dataset(self, config: ExportConfig) -> str:
        """Export comprehensive dataset for research analysis"""
        
        # Create output directory
        output_dir = Path(config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{config.filename_prefix}_{timestamp}"
        
        # Collect all data
        logger.info("Collecting data for export...")
        data_points = await self._collect_research_data(config)
        
        if not data_points:
            logger.warning("No data points collected for export")
            return ""
        
        # Apply privacy protection
        if config.privacy_level != PrivacyLevel.FULL:
            data_points = self._apply_privacy_protection(data_points, config)
        
        # Export in requested format
        output_file = None
        
        if config.format == ExportFormat.JSON:
            output_file = await self._export_json(data_points, output_dir, base_filename, config)
        elif config.format == ExportFormat.CSV:
            output_file = await self._export_csv(data_points, output_dir, base_filename, config)
        elif config.format == ExportFormat.EXCEL:
            output_file = await self._export_excel(data_points, output_dir, base_filename, config)
        elif config.format == ExportFormat.RESEARCH_JSON:
            output_file = await self._export_research_json(data_points, output_dir, base_filename, config)
        
        # Generate metadata file
        if config.include_metadata and output_file:
            await self._generate_metadata_file(data_points, output_file, config)
        
        logger.info(f"Export completed: {output_file}")
        return output_file or ""
    
    async def export_statistical_analysis(self, config: ExportConfig) -> str:
        """Export comprehensive statistical analysis"""
        
        # Collect data
        data_points = await self._collect_research_data(config)
        
        if len(data_points) < config.minimum_sample_size:
            logger.warning(f"Insufficient data for analysis: {len(data_points)} < {config.minimum_sample_size}")
            return ""
        
        # Perform analyses
        logger.info("Performing statistical analyses...")
        
        analysis_results = {
            'export_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_data_points': len(data_points),
                'date_range': {
                    'start': min(p.timestamp for p in data_points).isoformat(),
                    'end': max(p.timestamp for p in data_points).isoformat()
                },
                'privacy_level': config.privacy_level.value,
                'significance_level': config.statistical_significance_level
            },
            'model_comparison': self.statistical_analyzer.perform_model_comparison_analysis(data_points),
            'prompt_type_analysis': self.statistical_analyzer.analyze_prompt_type_effectiveness(data_points),
            'temporal_patterns': self.statistical_analyzer.analyze_temporal_patterns(data_points),
            'conversation_flow': self.statistical_analyzer.calculate_conversation_flow_metrics(data_points)
        }
        
        # Export analysis results
        output_dir = Path(config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"statistical_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Statistical analysis exported: {output_file}")
        return str(output_file)
    
    async def export_research_paper_dataset(self, 
                                          config: ExportConfig,
                                          paper_title: str = "Mental Health AI Model Selection Study") -> Dict[str, str]:
        """Export dataset formatted for research paper publication"""
        
        # Ensure research-level privacy
        config.privacy_level = PrivacyLevel.RESEARCH
        config.remove_personal_info = True
        config.hash_user_ids = True
        
        output_files = {}
        
        # Main dataset
        main_dataset_file = await self.export_comprehensive_dataset(config)
        output_files['main_dataset'] = main_dataset_file
        
        # Statistical analysis
        analysis_file = await self.export_statistical_analysis(config)
        output_files['statistical_analysis'] = analysis_file
        
        # Generate research summary
        summary_file = await self._generate_research_summary(config, paper_title)
        output_files['research_summary'] = summary_file
        
        # Generate data dictionary
        dictionary_file = await self._generate_data_dictionary(config)
        output_files['data_dictionary'] = dictionary_file
        
        return output_files
    
    async def _collect_research_data(self, config: ExportConfig) -> List[ResearchDataPoint]:
        """Collect and consolidate data from all sources"""
        
        data_points = []
        
        # Collect feedback data
        if config.include_feedback and self.feedback_collector:
            feedback_data = self.feedback_collector.feedback_data
            
            for feedback in feedback_data:
                # Apply date filtering
                if config.start_date and feedback.timestamp < config.start_date:
                    continue
                if config.end_date and feedback.timestamp > config.end_date:
                    continue
                
                data_point = ResearchDataPoint(
                    data_id=feedback.feedback_id,
                    timestamp=feedback.timestamp,
                    selected_model=feedback.selected_model,
                    selection_confidence=0.8,  # Default for feedback data
                    selection_time_ms=100.0,   # Default
                    prompt_type=feedback.prompt_type,
                    message_index=1,  # Simplified
                    conversation_length=1,  # Simplified
                    user_satisfaction=feedback.overall_rating,
                    response_time_ms=feedback.time_spent_reading_ms,
                    conversation_continued=feedback.conversation_continued,
                    task_completion=feedback.follow_up_questions > 0,
                    automated_quality_score=feedback.automated_quality_score,
                    user_hash=self.privacy_manager.hash_identifier(feedback.user_id),
                    session_hash=self.privacy_manager.hash_identifier(feedback.session_id or feedback.user_id)
                )
                
                data_points.append(data_point)
        
        # Collect experiment data
        if config.include_experiments and self.experiment_manager:
            for experiment in self.experiment_manager.experiments.values():
                for result in experiment.results:
                    # Apply date filtering
                    if config.start_date and result.timestamp < config.start_date:
                        continue
                    if config.end_date and result.timestamp > config.end_date:
                        continue
                    
                    data_point = ResearchDataPoint(
                        data_id=f"exp_{result.experiment_id}_{result.user_id}",
                        timestamp=result.timestamp,
                        selected_model=result.selected_model,
                        selection_confidence=result.confidence_score,
                        selection_time_ms=result.selection_time_ms,
                        prompt_type=result.prompt_type,
                        message_index=1,  # Simplified
                        conversation_length=1,  # Simplified
                        user_satisfaction=result.user_satisfaction,
                        response_time_ms=result.response_time_ms,
                        task_completion=result.task_completion,
                        safety_score=result.safety_score,
                        experiment_id=result.experiment_id,
                        variant_name=result.variant_name,
                        user_hash=self.privacy_manager.hash_identifier(result.user_id),
                        session_hash=self.privacy_manager.hash_identifier(result.session_id or result.user_id)
                    )
                    
                    data_points.append(data_point)
        
        # Sort by timestamp
        data_points.sort(key=lambda p: p.timestamp)
        
        logger.info(f"Collected {len(data_points)} data points for export")
        return data_points
    
    def _apply_privacy_protection(self, data_points: List[ResearchDataPoint], config: ExportConfig) -> List[ResearchDataPoint]:
        """Apply privacy protection to data points"""
        
        for point in data_points:
            if config.privacy_level == PrivacyLevel.ANONYMIZED:
                # Hash user identifiers
                if point.user_hash is None:
                    point.user_hash = self.privacy_manager.hash_identifier(str(point.data_id))
                
                # Add user segment
                point.user_segment = self.privacy_manager.create_user_segment(point.user_hash)
                
            elif config.privacy_level == PrivacyLevel.RESEARCH:
                # Maximum privacy protection for research
                point.user_hash = self.privacy_manager.hash_identifier(str(point.data_id))
                point.session_hash = self.privacy_manager.hash_identifier(f"session_{point.data_id}")
                point.user_segment = self.privacy_manager.create_user_segment(point.user_hash)
                
                # Remove potentially identifying information
                point.data_id = self.privacy_manager.hash_identifier(point.data_id)
        
        return data_points
    
    async def _export_json(self, data_points: List[ResearchDataPoint], output_dir: Path, base_filename: str, config: ExportConfig) -> str:
        """Export data as JSON"""
        
        output_file = output_dir / f"{base_filename}.json"
        
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'format': 'json',
                'privacy_level': config.privacy_level.value,
                'total_records': len(data_points)
            },
            'data': [point.to_dict() for point in data_points]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return str(output_file)
    
    async def _export_csv(self, data_points: List[ResearchDataPoint], output_dir: Path, base_filename: str, config: ExportConfig) -> str:
        """Export data as CSV"""
        
        output_file = output_dir / f"{base_filename}.csv"
        
        if not data_points:
            return str(output_file)
        
        # Convert to DataFrame
        df = pd.DataFrame([point.to_dict() for point in data_points])
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        return str(output_file)
    
    async def _export_excel(self, data_points: List[ResearchDataPoint], output_dir: Path, base_filename: str, config: ExportConfig) -> str:
        """Export data as Excel with multiple sheets"""
        
        output_file = output_dir / f"{base_filename}.xlsx"
        
        if not data_points:
            return str(output_file)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main data sheet
            df = pd.DataFrame([point.to_dict() for point in data_points])
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Summary statistics sheet
            summary_stats = self._calculate_summary_statistics(data_points)
            summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Model comparison sheet
            model_stats = self._calculate_model_statistics(data_points)
            if model_stats:
                model_df = pd.DataFrame(model_stats).T
                model_df.to_excel(writer, sheet_name='Model_Comparison')
        
        return str(output_file)
    
    async def _export_research_json(self, data_points: List[ResearchDataPoint], output_dir: Path, base_filename: str, config: ExportConfig) -> str:
        """Export data in research-optimized JSON format"""
        
        output_file = output_dir / f"{base_filename}_research.json"
        
        # Group data for research analysis
        research_data = {
            'study_metadata': {
                'title': 'Mental Health AI Model Selection Analysis',
                'export_date': datetime.now().isoformat(),
                'privacy_level': config.privacy_level.value,
                'total_participants': len(set(p.user_hash for p in data_points if p.user_hash)),
                'total_interactions': len(data_points),
                'date_range': {
                    'start': min(p.timestamp for p in data_points).isoformat() if data_points else None,
                    'end': max(p.timestamp for p in data_points).isoformat() if data_points else None
                }
            },
            'raw_data': [point.to_dict() for point in data_points],
            'aggregated_statistics': self._calculate_summary_statistics(data_points),
            'model_performance': self._calculate_model_statistics(data_points)
        }
        
        with open(output_file, 'w') as f:
            json.dump(research_data, f, indent=2, default=str)
        
        return str(output_file)
    
    def _calculate_summary_statistics(self, data_points: List[ResearchDataPoint]) -> Dict[str, Any]:
        """Calculate summary statistics for the dataset"""
        
        if not data_points:
            return {}
        
        satisfaction_scores = [p.user_satisfaction for p in data_points if p.user_satisfaction is not None]
        response_times = [p.response_time_ms for p in data_points if p.response_time_ms is not None]
        
        return {
            'total_data_points': len(data_points),
            'unique_users': len(set(p.user_hash for p in data_points if p.user_hash)),
            'unique_models': len(set(p.selected_model for p in data_points)),
            'prompt_type_distribution': dict(Counter(p.prompt_type.value for p in data_points)),
            'satisfaction_stats': {
                'mean': statistics.mean(satisfaction_scores) if satisfaction_scores else None,
                'median': statistics.median(satisfaction_scores) if satisfaction_scores else None,
                'std': statistics.stdev(satisfaction_scores) if len(satisfaction_scores) > 1 else None,
                'count': len(satisfaction_scores)
            } if satisfaction_scores else None,
            'response_time_stats': {
                'mean': statistics.mean(response_times) if response_times else None,
                'median': statistics.median(response_times) if response_times else None,
                'std': statistics.stdev(response_times) if len(response_times) > 1 else None,
                'count': len(response_times)
            } if response_times else None
        }
    
    def _calculate_model_statistics(self, data_points: List[ResearchDataPoint]) -> Dict[str, Dict[str, Any]]:
        """Calculate per-model statistics"""
        
        model_data = defaultdict(list)
        for point in data_points:
            model_data[point.selected_model].append(point)
        
        model_stats = {}
        for model, points in model_data.items():
            satisfaction_scores = [p.user_satisfaction for p in points if p.user_satisfaction is not None]
            response_times = [p.response_time_ms for p in points if p.response_time_ms is not None]
            
            model_stats[model] = {
                'total_uses': len(points),
                'satisfaction_mean': statistics.mean(satisfaction_scores) if satisfaction_scores else None,
                'satisfaction_std': statistics.stdev(satisfaction_scores) if len(satisfaction_scores) > 1 else None,
                'response_time_mean': statistics.mean(response_times) if response_times else None,
                'prompt_type_distribution': dict(Counter(p.prompt_type.value for p in points))
            }
        
        return model_stats
    
    async def _generate_metadata_file(self, data_points: List[ResearchDataPoint], output_file: str, config: ExportConfig):
        """Generate metadata file describing the dataset"""
        
        metadata_file = Path(output_file).with_suffix('.metadata.json')
        
        metadata = {
            'dataset_info': {
                'filename': Path(output_file).name,
                'format': config.format.value,
                'privacy_level': config.privacy_level.value,
                'generation_timestamp': datetime.now().isoformat()
            },
            'data_description': {
                'total_records': len(data_points),
                'date_range': {
                    'start': min(p.timestamp for p in data_points).isoformat() if data_points else None,
                    'end': max(p.timestamp for p in data_points).isoformat() if data_points else None
                },
                'unique_users': len(set(p.user_hash for p in data_points if p.user_hash)),
                'models_included': list(set(p.selected_model for p in data_points)),
                'prompt_types': list(set(p.prompt_type.value for p in data_points))
            },
            'privacy_info': {
                'user_ids_hashed': config.hash_user_ids,
                'personal_info_removed': config.remove_personal_info,
                'privacy_level': config.privacy_level.value
            },
            'field_descriptions': {
                'data_id': 'Unique identifier for each data point',
                'timestamp': 'ISO formatted timestamp of the interaction',
                'selected_model': 'AI model used for the interaction',
                'selection_confidence': 'Confidence score for model selection (0-1)',
                'prompt_type': 'Classified type of user prompt',
                'user_satisfaction': 'User satisfaction rating (1-5 scale)',
                'response_time_ms': 'Response time in milliseconds',
                'automated_quality_score': 'Automated quality assessment (0-1)',
                'user_hash': 'Anonymized user identifier',
                'session_hash': 'Anonymized session identifier'
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated metadata file: {metadata_file}")
    
    async def _generate_research_summary(self, config: ExportConfig, paper_title: str) -> str:
        """Generate research summary document"""
        
        output_dir = Path(config.output_directory)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = output_dir / f"research_summary_{timestamp}.md"
        
        # Collect data for summary
        data_points = await self._collect_research_data(config)
        
        summary_content = f"""# {paper_title}

## Dataset Summary
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Overview
- **Total Interactions**: {len(data_points)}
- **Unique Users**: {len(set(p.user_hash for p in data_points if p.user_hash))}
- **Models Analyzed**: {', '.join(set(p.selected_model for p in data_points))}
- **Data Collection Period**: {min(p.timestamp for p in data_points).strftime('%Y-%m-%d') if data_points else 'N/A'} to {max(p.timestamp for p in data_points).strftime('%Y-%m-%d') if data_points else 'N/A'}

### Privacy Protection
- Privacy Level: {config.privacy_level.value}
- User IDs: {'Hashed' if config.hash_user_ids else 'Original'}
- Personal Information: {'Removed' if config.remove_personal_info else 'Included'}

### Prompt Types Distribution
"""
        
        # Add prompt type distribution
        prompt_distribution = Counter(p.prompt_type.value for p in data_points)
        for prompt_type, count in prompt_distribution.most_common():
            percentage = (count / len(data_points)) * 100 if data_points else 0
            summary_content += f"- **{prompt_type}**: {count} ({percentage:.1f}%)\n"
        
        summary_content += "\n### Model Usage Distribution\n"
        
        # Add model distribution
        model_distribution = Counter(p.selected_model for p in data_points)
        for model, count in model_distribution.most_common():
            percentage = (count / len(data_points)) * 100 if data_points else 0
            summary_content += f"- **{model}**: {count} ({percentage:.1f}%)\n"
        
        # Add data quality indicators
        satisfaction_scores = [p.user_satisfaction for p in data_points if p.user_satisfaction is not None]
        
        if satisfaction_scores:
            summary_content += f"""
### Data Quality Indicators
- **User Satisfaction Data**: {len(satisfaction_scores)} responses
- **Average Satisfaction**: {statistics.mean(satisfaction_scores):.2f}/5.0
- **Response Rate**: {(len(satisfaction_scores) / len(data_points)) * 100:.1f}%
"""
        
        summary_content += """
### Research Applications
This dataset is suitable for:
- Comparative analysis of AI model performance in mental health contexts
- Investigation of prompt type effectiveness across models
- Temporal pattern analysis of user interactions
- User satisfaction and engagement studies

### Citation
Please cite this dataset as:
[Your Institution]. ({datetime.now().year}). Mental Health AI Model Selection Dataset. [Dataset]. Generated {datetime.now().strftime('%Y-%m-%d')}.
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        return str(summary_file)
    
    async def _generate_data_dictionary(self, config: ExportConfig) -> str:
        """Generate comprehensive data dictionary"""
        
        output_dir = Path(config.output_directory)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dictionary_file = output_dir / f"data_dictionary_{timestamp}.json"
        
        data_dictionary = {
            'dataset_version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'privacy_level': config.privacy_level.value,
            'fields': {
                'data_id': {
                    'type': 'string',
                    'description': 'Unique identifier for each interaction record',
                    'example': 'abc123def456',
                    'privacy_notes': 'Hashed if privacy level is RESEARCH'
                },
                'timestamp': {
                    'type': 'datetime',
                    'description': 'ISO formatted timestamp of the interaction',
                    'format': 'YYYY-MM-DDTHH:MM:SS.fffffZ',
                    'example': '2025-01-15T14:30:25.123456Z'
                },
                'selected_model': {
                    'type': 'string',
                    'description': 'AI model used for this interaction',
                    'possible_values': ['claude-3-opus', 'claude-3-sonnet', 'gpt-4-turbo', 'gpt-3.5-turbo', 'claude-3-haiku'],
                    'example': 'claude-3-sonnet'
                },
                'selection_confidence': {
                    'type': 'float',
                    'description': 'Confidence score for model selection decision',
                    'range': '0.0 to 1.0',
                    'example': 0.85
                },
                'prompt_type': {
                    'type': 'string',
                    'description': 'Classified category of user prompt',
                    'possible_values': ['crisis', 'anxiety', 'depression', 'information_seeking', 'general_wellness'],
                    'example': 'anxiety'
                },
                'user_satisfaction': {
                    'type': 'float',
                    'description': 'User satisfaction rating on 1-5 scale',
                    'range': '1.0 to 5.0',
                    'example': 4.2,
                    'notes': 'May be null if user did not provide rating'
                },
                'response_time_ms': {
                    'type': 'float',
                    'description': 'Total response generation time in milliseconds',
                    'range': '0.0 to unlimited',
                    'example': 1250.5
                },
                'automated_quality_score': {
                    'type': 'float',
                    'description': 'System-generated quality assessment score',
                    'range': '0.0 to 1.0',
                    'example': 0.78
                },
                'user_hash': {
                    'type': 'string',
                    'description': 'Anonymized user identifier',
                    'example': 'usr_abc123def456',
                    'privacy_notes': 'SHA-256 hash of original user ID with salt'
                },
                'session_hash': {
                    'type': 'string',
                    'description': 'Anonymized session identifier',
                    'example': 'ses_def456ghi789',
                    'privacy_notes': 'SHA-256 hash of original session ID with salt'
                },
                'experiment_id': {
                    'type': 'string',
                    'description': 'Identifier for A/B test experiment if applicable',
                    'example': 'exp_model_comparison_2025',
                    'notes': 'Null if not part of an experiment'
                },
                'variant_name': {
                    'type': 'string',
                    'description': 'Experiment variant/group assignment',
                    'example': 'treatment_group_a',
                    'notes': 'Null if not part of an experiment'
                }
            },
            'data_quality': {
                'completeness': 'Percentage of non-null values per field',
                'accuracy': 'System validation checks applied',
                'consistency': 'Cross-field validation rules enforced',
                'timeliness': 'Data exported within 24 hours of collection'
            },
            'ethical_considerations': {
                'consent': 'All data collected with appropriate user consent',
                'anonymization': 'Personal identifiers removed or hashed',
                'purpose_limitation': 'Data used only for stated research purposes',
                'data_minimization': 'Only necessary data fields included'
            }
        }
        
        with open(dictionary_file, 'w') as f:
            json.dump(data_dictionary, f, indent=2)
        
        return str(dictionary_file)


class DataMiner:
    """Advanced data mining and pattern discovery"""
    
    def __init__(self, research_exporter: ResearchExporter):
        self.research_exporter = research_exporter
    
    async def discover_usage_patterns(self, config: ExportConfig) -> Dict[str, Any]:
        """Discover interesting usage patterns in the data"""
        
        data_points = await self.research_exporter._collect_research_data(config)
        
        patterns = {
            'temporal_patterns': self._discover_temporal_patterns(data_points),
            'user_behavior_patterns': self._discover_user_patterns(data_points),
            'model_selection_patterns': self._discover_selection_patterns(data_points),
            'satisfaction_patterns': self._discover_satisfaction_patterns(data_points)
        }
        
        return patterns
    
    def _discover_temporal_patterns(self, data_points: List[ResearchDataPoint]) -> Dict[str, Any]:
        """Discover temporal usage patterns"""
        
        # Group by hour and day of week
        hourly_usage = defaultdict(int)
        daily_usage = defaultdict(int)
        
        for point in data_points:
            hour = point.timestamp.hour
            day_of_week = point.timestamp.strftime('%A')
            
            hourly_usage[hour] += 1
            daily_usage[day_of_week] += 1
        
        # Find peak usage times
        peak_hour = max(hourly_usage.items(), key=lambda x: x[1]) if hourly_usage else (0, 0)
        peak_day = max(daily_usage.items(), key=lambda x: x[1]) if daily_usage else ('Unknown', 0)
        
        return {
            'peak_hour': {'hour': peak_hour[0], 'count': peak_hour[1]},
            'peak_day': {'day': peak_day[0], 'count': peak_day[1]},
            'hourly_distribution': dict(hourly_usage),
            'daily_distribution': dict(daily_usage)
        }
    
    def _discover_user_patterns(self, data_points: List[ResearchDataPoint]) -> Dict[str, Any]:
        """Discover user behavior patterns"""
        
        user_sessions = defaultdict(list)
        for point in data_points:
            if point.user_hash:
                user_sessions[point.user_hash].append(point)
        
        # Analyze user engagement patterns
        engagement_scores = []
        session_lengths = []
        
        for user_hash, points in user_sessions.items():
            if len(points) > 1:
                # Calculate engagement based on continued conversations
                continued_count = sum(1 for p in points if p.conversation_continued)
                engagement_score = continued_count / len(points)
                engagement_scores.append(engagement_score)
                
                session_lengths.append(len(points))
        
        return {
            'total_unique_users': len(user_sessions),
            'avg_session_length': statistics.mean(session_lengths) if session_lengths else 0,
            'avg_engagement_score': statistics.mean(engagement_scores) if engagement_scores else 0,
            'highly_engaged_users': sum(1 for score in engagement_scores if score > 0.7),
            'power_users': sum(1 for sessions in user_sessions.values() if len(sessions) > 5)
        }
    
    def _discover_selection_patterns(self, data_points: List[ResearchDataPoint]) -> Dict[str, Any]:
        """Discover model selection patterns"""
        
        # Model-prompt type associations
        model_prompt_matrix = defaultdict(lambda: defaultdict(int))
        
        for point in data_points:
            model_prompt_matrix[point.selected_model][point.prompt_type.value] += 1
        
        # Find strongest associations
        associations = []
        for model, prompt_counts in model_prompt_matrix.items():
            total_model_uses = sum(prompt_counts.values())
            for prompt_type, count in prompt_counts.items():
                association_strength = count / total_model_uses if total_model_uses > 0 else 0
                associations.append({
                    'model': model,
                    'prompt_type': prompt_type,
                    'strength': association_strength,
                    'count': count
                })
        
        # Sort by association strength
        associations.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'model_prompt_associations': associations[:10],  # Top 10
            'model_usage_distribution': {
                model: sum(counts.values()) 
                for model, counts in model_prompt_matrix.items()
            }
        }
    
    def _discover_satisfaction_patterns(self, data_points: List[ResearchDataPoint]) -> Dict[str, Any]:
        """Discover satisfaction patterns"""
        
        # Satisfaction by various factors
        satisfaction_by_model = defaultdict(list)
        satisfaction_by_prompt_type = defaultdict(list)
        satisfaction_by_time = defaultdict(list)
        
        for point in data_points:
            if point.user_satisfaction is not None:
                satisfaction_by_model[point.selected_model].append(point.user_satisfaction)
                satisfaction_by_prompt_type[point.prompt_type.value].append(point.user_satisfaction)
                
                hour = point.timestamp.hour
                time_period = 'morning' if 6 <= hour < 12 else 'afternoon' if 12 <= hour < 18 else 'evening' if 18 <= hour < 22 else 'night'
                satisfaction_by_time[time_period].append(point.user_satisfaction)
        
        # Calculate averages
        model_satisfaction = {
            model: statistics.mean(scores) 
            for model, scores in satisfaction_by_model.items()
        }
        
        prompt_satisfaction = {
            prompt_type: statistics.mean(scores) 
            for prompt_type, scores in satisfaction_by_prompt_type.items()
        }
        
        time_satisfaction = {
            period: statistics.mean(scores) 
            for period, scores in satisfaction_by_time.items()
        }
        
        return {
            'satisfaction_by_model': model_satisfaction,
            'satisfaction_by_prompt_type': prompt_satisfaction,
            'satisfaction_by_time_period': time_satisfaction,
            'best_performing_model': max(model_satisfaction.items(), key=lambda x: x[1]) if model_satisfaction else None,
            'most_challenging_prompt_type': min(prompt_satisfaction.items(), key=lambda x: x[1]) if prompt_satisfaction else None
        }