"""
Analytics Dashboard with Real-time Visualizations

Comprehensive dashboard system providing real-time insights into model selection
performance, user satisfaction, and system analytics with interactive visualizations.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import statistics
from collections import defaultdict, deque

from .feedback_system import FeedbackCollector, FeedbackAnalytics
from .ab_testing import ExperimentManager
from .smart_switching import SmartModelSwitcher
from .research_tools import ResearchExporter, StatisticalAnalyzer

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIME_SERIES = "time_series"


class RefreshInterval(Enum):
    """Dashboard refresh intervals"""
    REAL_TIME = 5      # 5 seconds
    FAST = 30          # 30 seconds
    NORMAL = 60        # 1 minute
    SLOW = 300         # 5 minutes


@dataclass
class ChartData:
    """Data structure for chart visualization"""
    chart_id: str
    title: str
    chart_type: ChartType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'chart_id': self.chart_id,
            'title': self.title,
            'chart_type': self.chart_type.value,
            'data': self.data,
            'metadata': self.metadata,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class DashboardConfig:
    """Configuration for dashboard behavior"""
    refresh_interval: RefreshInterval = RefreshInterval.NORMAL
    max_data_points: int = 100
    enable_real_time: bool = True
    
    # Chart configurations
    show_model_comparison: bool = True
    show_user_satisfaction: bool = True
    show_performance_metrics: bool = True
    show_temporal_patterns: bool = True
    show_experiment_results: bool = True
    
    # Data retention
    historical_days: int = 30
    cache_duration_minutes: int = 5
    
    # Alert thresholds
    low_satisfaction_threshold: float = 3.0
    high_response_time_threshold: float = 5000  # milliseconds
    low_cache_hit_threshold: float = 60.0  # percentage


class MetricsCollector:
    """Collects and aggregates metrics from various sources"""
    
    def __init__(self,
                 feedback_collector: FeedbackCollector = None,
                 experiment_manager: ExperimentManager = None,
                 model_switcher: SmartModelSwitcher = None):
        
        self.feedback_collector = feedback_collector
        self.experiment_manager = experiment_manager
        self.model_switcher = model_switcher
        
        # Real-time metrics cache
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        logger.info("MetricsCollector initialized")
    
    async def collect_real_time_metrics(self) -> Dict[str, Any]:
        """Collect current real-time metrics"""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system_status': await self._get_system_status(),
            'current_performance': await self._get_current_performance(),
            'recent_activity': await self._get_recent_activity(),
            'alerts': await self._get_active_alerts()
        }
        
        return metrics
    
    async def collect_historical_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Collect historical metrics for specified time period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        metrics = {
            'period': f'{days} days',
            'start_date': cutoff_date.isoformat(),
            'end_date': datetime.now().isoformat(),
            'user_satisfaction_trends': await self._get_satisfaction_trends(cutoff_date),
            'model_performance_history': await self._get_model_performance_history(cutoff_date),
            'usage_patterns': await self._get_usage_patterns(cutoff_date),
            'experiment_results': await self._get_experiment_results_history(cutoff_date)
        }
        
        return metrics
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status indicators"""
        
        status = {
            'overall_health': 'healthy',
            'components': {
                'feedback_system': 'operational' if self.feedback_collector else 'unavailable',
                'experiments': 'operational' if self.experiment_manager else 'unavailable',
                'model_switching': 'operational' if self.model_switcher else 'unavailable'
            },
            'uptime_hours': 24,  # Would be calculated from actual uptime
            'last_restart': (datetime.now() - timedelta(hours=24)).isoformat()
        }
        
        return status
    
    async def _get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        performance = {
            'response_time_ms': 1250.0,  # Would be from actual monitoring
            'throughput_rps': 2.5,
            'error_rate_percent': 1.2,
            'cache_hit_rate_percent': 78.5
        }
        
        # Add feedback-based metrics if available
        if self.feedback_collector:
            recent_metrics = self.feedback_collector.get_real_time_metrics()
            
            if recent_metrics.get('last_hour_avg_rating'):
                performance['user_satisfaction'] = recent_metrics['last_hour_avg_rating']
            if recent_metrics.get('last_hour_thumbs_up_pct'):
                performance['approval_rate_percent'] = recent_metrics['last_hour_thumbs_up_pct']
        
        return performance
    
    async def _get_recent_activity(self) -> Dict[str, Any]:
        """Get recent system activity"""
        
        activity = {
            'last_hour': {
                'total_requests': 150,
                'unique_users': 45,
                'feedback_received': 23,
                'experiments_active': 0
            },
            'last_24_hours': {
                'total_requests': 3240,
                'unique_users': 892,
                'feedback_received': 445,
                'model_switches': 12
            }
        }
        
        # Add real data if available
        if self.feedback_collector:
            recent_feedback = len([f for f in self.feedback_collector.recent_feedback 
                                 if datetime.now() - f.timestamp <= timedelta(hours=1)])
            activity['last_hour']['feedback_received'] = recent_feedback
        
        if self.experiment_manager:
            activity['last_hour']['experiments_active'] = len(self.experiment_manager.active_experiments)
        
        if self.model_switcher:
            recent_switches = len([s for s in self.model_switcher.switch_history 
                                 if datetime.now() - s.timestamp <= timedelta(hours=24)])
            activity['last_24_hours']['model_switches'] = recent_switches
        
        return activity
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get current active alerts"""
        
        alerts = []
        
        # Check performance alerts
        performance = await self._get_current_performance()
        
        if performance.get('response_time_ms', 0) > 5000:
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f"High response time: {performance['response_time_ms']:.0f}ms",
                'timestamp': datetime.now().isoformat()
            })
        
        if performance.get('user_satisfaction', 5.0) < 3.0:
            alerts.append({
                'type': 'satisfaction',
                'severity': 'critical',
                'message': f"Low user satisfaction: {performance['user_satisfaction']:.2f}/5.0",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    async def _get_satisfaction_trends(self, since_date: datetime) -> Dict[str, Any]:
        """Get user satisfaction trends over time"""
        
        if not self.feedback_collector:
            return {}
        
        # Group feedback by day
        daily_satisfaction = defaultdict(list)
        
        for feedback in self.feedback_collector.feedback_data:
            if feedback.timestamp >= since_date and feedback.overall_rating is not None:
                day_key = feedback.timestamp.strftime('%Y-%m-%d')
                daily_satisfaction[day_key].append(feedback.overall_rating)
        
        # Calculate daily averages
        trends = {}
        for day, ratings in daily_satisfaction.items():
            trends[day] = {
                'average_rating': statistics.mean(ratings),
                'total_responses': len(ratings),
                'thumbs_up_rate': sum(1 for r in ratings if r >= 3) / len(ratings) * 100
            }
        
        return trends
    
    async def _get_model_performance_history(self, since_date: datetime) -> Dict[str, Any]:
        """Get historical model performance data"""
        
        if not self.feedback_collector:
            return {}
        
        # Group by model and date
        model_performance = defaultdict(lambda: defaultdict(list))
        
        for feedback in self.feedback_collector.feedback_data:
            if (feedback.timestamp >= since_date and 
                feedback.overall_rating is not None and 
                feedback.selected_model):
                
                day_key = feedback.timestamp.strftime('%Y-%m-%d')
                model_performance[feedback.selected_model][day_key].append(feedback.overall_rating)
        
        # Calculate daily averages per model
        history = {}
        for model, daily_data in model_performance.items():
            history[model] = {}
            for day, ratings in daily_data.items():
                history[model][day] = {
                    'average_rating': statistics.mean(ratings),
                    'usage_count': len(ratings)
                }
        
        return history
    
    async def _get_usage_patterns(self, since_date: datetime) -> Dict[str, Any]:
        """Get usage pattern analysis"""
        
        patterns = {
            'hourly_distribution': defaultdict(int),
            'prompt_type_distribution': defaultdict(int),
            'model_usage_distribution': defaultdict(int)
        }
        
        if self.feedback_collector:
            for feedback in self.feedback_collector.feedback_data:
                if feedback.timestamp >= since_date:
                    hour = feedback.timestamp.hour
                    patterns['hourly_distribution'][hour] += 1
                    patterns['prompt_type_distribution'][feedback.prompt_type.value] += 1
                    if feedback.selected_model:
                        patterns['model_usage_distribution'][feedback.selected_model] += 1
        
        # Convert to regular dicts for JSON serialization
        return {
            'hourly_distribution': dict(patterns['hourly_distribution']),
            'prompt_type_distribution': dict(patterns['prompt_type_distribution']),
            'model_usage_distribution': dict(patterns['model_usage_distribution'])
        }
    
    async def _get_experiment_results_history(self, since_date: datetime) -> Dict[str, Any]:
        """Get experiment results history"""
        
        if not self.experiment_manager:
            return {}
        
        experiment_history = {}
        
        for exp_id, experiment in self.experiment_manager.experiments.items():
            if experiment.start_time and experiment.start_time >= since_date:
                experiment_history[exp_id] = {
                    'name': experiment.config.name,
                    'status': experiment.status.value,
                    'start_time': experiment.start_time.isoformat(),
                    'participants': len(set(r.user_id for r in experiment.results)),
                    'results_count': len(experiment.results),
                    'variants': list(experiment.config.strategies.keys())
                }
        
        return experiment_history


class VisualizationEngine:
    """Generates visualization data for various chart types"""
    
    def __init__(self):
        self.color_palette = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#16a085'
        ]
    
    def create_satisfaction_trend_chart(self, trends_data: Dict[str, Any]) -> ChartData:
        """Create user satisfaction trend chart"""
        
        if not trends_data:
            return ChartData(
                chart_id="satisfaction_trends",
                title="User Satisfaction Trends",
                chart_type=ChartType.LINE,
                data={'labels': [], 'datasets': []}
            )
        
        # Sort dates
        sorted_dates = sorted(trends_data.keys())
        
        chart_data = {
            'labels': sorted_dates,
            'datasets': [{
                'label': 'Average Rating',
                'data': [trends_data[date]['average_rating'] for date in sorted_dates],
                'borderColor': self.color_palette[0],
                'backgroundColor': self.color_palette[0] + '20',
                'tension': 0.4
            }, {
                'label': 'Approval Rate (%)',
                'data': [trends_data[date]['thumbs_up_rate'] for date in sorted_dates],
                'borderColor': self.color_palette[1],
                'backgroundColor': self.color_palette[1] + '20',
                'tension': 0.4,
                'yAxisID': 'y1'
            }]
        }
        
        return ChartData(
            chart_id="satisfaction_trends",
            title="User Satisfaction Trends",
            chart_type=ChartType.LINE,
            data=chart_data,
            metadata={
                'yAxes': {
                    'y': {'type': 'linear', 'display': True, 'position': 'left', 'min': 1, 'max': 5},
                    'y1': {'type': 'linear', 'display': True, 'position': 'right', 'min': 0, 'max': 100}
                }
            }
        )
    
    def create_model_comparison_chart(self, model_performance: Dict[str, Any]) -> ChartData:
        """Create model performance comparison chart"""
        
        if not model_performance:
            return ChartData(
                chart_id="model_comparison",
                title="Model Performance Comparison",
                chart_type=ChartType.BAR,
                data={'labels': [], 'datasets': []}
            )
        
        models = list(model_performance.keys())
        
        # Calculate average performance for each model
        model_averages = {}
        model_usage = {}
        
        for model, daily_data in model_performance.items():
            all_ratings = []
            total_usage = 0
            
            for day_data in daily_data.values():
                # Use usage_count as weight for the average
                usage = day_data['usage_count']
                rating = day_data['average_rating']
                all_ratings.extend([rating] * usage)
                total_usage += usage
            
            if all_ratings:
                model_averages[model] = statistics.mean(all_ratings)
                model_usage[model] = total_usage
        
        chart_data = {
            'labels': list(model_averages.keys()),
            'datasets': [{
                'label': 'Average Rating',
                'data': list(model_averages.values()),
                'backgroundColor': [self.color_palette[i % len(self.color_palette)] 
                                 for i in range(len(model_averages))],
                'borderColor': [self.color_palette[i % len(self.color_palette)] 
                              for i in range(len(model_averages))],
                'borderWidth': 2
            }]
        }
        
        return ChartData(
            chart_id="model_comparison",
            title="Model Performance Comparison",
            chart_type=ChartType.BAR,
            data=chart_data,
            metadata={
                'usage_counts': model_usage,
                'yAxis': {'min': 1, 'max': 5}
            }
        )
    
    def create_usage_distribution_pie(self, usage_data: Dict[str, int], title: str, chart_id: str) -> ChartData:
        """Create pie chart for usage distribution"""
        
        if not usage_data:
            return ChartData(
                chart_id=chart_id,
                title=title,
                chart_type=ChartType.PIE,
                data={'labels': [], 'datasets': []}
            )
        
        # Sort by usage count
        sorted_items = sorted(usage_data.items(), key=lambda x: x[1], reverse=True)
        
        chart_data = {
            'labels': [item[0] for item in sorted_items],
            'datasets': [{
                'data': [item[1] for item in sorted_items],
                'backgroundColor': [self.color_palette[i % len(self.color_palette)] 
                                 for i in range(len(sorted_items))],
                'borderWidth': 2
            }]
        }
        
        return ChartData(
            chart_id=chart_id,
            title=title,
            chart_type=ChartType.PIE,
            data=chart_data
        )
    
    def create_hourly_activity_heatmap(self, hourly_data: Dict[int, int]) -> ChartData:
        """Create heatmap of hourly activity"""
        
        # Create 24-hour grid
        hours = list(range(24))
        activity = [hourly_data.get(hour, 0) for hour in hours]
        
        # Group into time periods for better visualization
        time_periods = ['Night', 'Early Morning', 'Morning', 'Afternoon', 'Evening', 'Late Evening']
        period_activity = {
            'Night': sum(activity[22:24] + activity[0:6]),
            'Early Morning': sum(activity[6:9]),
            'Morning': sum(activity[9:12]),
            'Afternoon': sum(activity[12:15]),
            'Evening': sum(activity[15:19]),
            'Late Evening': sum(activity[19:22])
        }
        
        chart_data = {
            'labels': list(period_activity.keys()),
            'datasets': [{
                'label': 'Activity Level',
                'data': list(period_activity.values()),
                'backgroundColor': self.color_palette[2],
                'borderColor': self.color_palette[2],
                'borderWidth': 1
            }]
        }
        
        return ChartData(
            chart_id="hourly_activity",
            title="Activity by Time Period",
            chart_type=ChartType.BAR,
            data=chart_data,
            metadata={'raw_hourly_data': hourly_data}
        )
    
    def create_performance_gauge(self, current_performance: Dict[str, Any]) -> ChartData:
        """Create performance gauge chart"""
        
        # Normalize different metrics to 0-100 scale
        gauges = []
        
        if 'user_satisfaction' in current_performance:
            satisfaction_pct = (current_performance['user_satisfaction'] / 5.0) * 100
            gauges.append({
                'label': 'User Satisfaction',
                'value': satisfaction_pct,
                'color': self._get_gauge_color(satisfaction_pct, 70, 85)
            })
        
        if 'cache_hit_rate_percent' in current_performance:
            cache_rate = current_performance['cache_hit_rate_percent']
            gauges.append({
                'label': 'Cache Hit Rate',
                'value': cache_rate,
                'color': self._get_gauge_color(cache_rate, 60, 80)
            })
        
        if 'approval_rate_percent' in current_performance:
            approval_rate = current_performance['approval_rate_percent']
            gauges.append({
                'label': 'Approval Rate',
                'value': approval_rate,
                'color': self._get_gauge_color(approval_rate, 70, 85)
            })
        
        return ChartData(
            chart_id="performance_gauges",
            title="Key Performance Indicators",
            chart_type=ChartType.GAUGE,
            data={'gauges': gauges}
        )
    
    def create_experiment_results_chart(self, experiment_data: Dict[str, Any]) -> ChartData:
        """Create chart showing A/B test experiment results"""
        
        if not experiment_data:
            return ChartData(
                chart_id="experiment_results",
                title="A/B Test Results",
                chart_type=ChartType.BAR,
                data={'labels': [], 'datasets': []}
            )
        
        experiments = []
        for exp_id, exp_info in experiment_data.items():
            experiments.append({
                'name': exp_info['name'],
                'participants': exp_info['participants'],
                'status': exp_info['status']
            })
        
        # Sort by participant count
        experiments.sort(key=lambda x: x['participants'], reverse=True)
        
        chart_data = {
            'labels': [exp['name'] for exp in experiments],
            'datasets': [{
                'label': 'Participants',
                'data': [exp['participants'] for exp in experiments],
                'backgroundColor': [self.color_palette[3]] * len(experiments),
                'borderColor': self.color_palette[3],
                'borderWidth': 1
            }]
        }
        
        return ChartData(
            chart_id="experiment_results",
            title="Active Experiments",
            chart_type=ChartType.BAR,
            data=chart_data,
            metadata={'experiment_details': experiments}
        )
    
    def _get_gauge_color(self, value: float, warning_threshold: float, good_threshold: float) -> str:
        """Get color based on gauge value and thresholds"""
        if value >= good_threshold:
            return '#2ecc71'  # Green
        elif value >= warning_threshold:
            return '#f39c12'  # Orange
        else:
            return '#e74c3c'  # Red


class AnalyticsDashboard:
    """
    Comprehensive analytics dashboard with real-time visualizations
    
    Features:
    - Real-time performance monitoring
    - Interactive visualizations
    - Historical trend analysis
    - A/B test experiment tracking
    - User satisfaction metrics
    - System health monitoring
    """
    
    def __init__(self,
                 feedback_collector: FeedbackCollector = None,
                 experiment_manager: ExperimentManager = None,
                 model_switcher: SmartModelSwitcher = None,
                 config: DashboardConfig = None):
        
        self.config = config or DashboardConfig()
        self.metrics_collector = MetricsCollector(feedback_collector, experiment_manager, model_switcher)
        self.viz_engine = VisualizationEngine()
        
        # Dashboard state
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None
        self.charts: Dict[str, ChartData] = {}
        self.last_update: Optional[datetime] = None
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List = []
        
        logger.info("AnalyticsDashboard initialized")
    
    async def start_dashboard(self):
        """Start the dashboard with background updates"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Initial data load
        await self._update_all_charts()
        
        # Start background update task
        if self.config.enable_real_time:
            self._update_task = asyncio.create_task(self._background_update_loop())
        
        logger.info(f"Dashboard started with {self.config.refresh_interval.value}s refresh interval")
    
    async def stop_dashboard(self):
        """Stop the dashboard"""
        self.is_running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dashboard stopped")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        
        # Update charts if cache is stale
        if (not self.last_update or 
            datetime.now() - self.last_update > timedelta(minutes=self.config.cache_duration_minutes)):
            await self._update_all_charts()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'refresh_interval': self.config.refresh_interval.value,
            'charts': {chart_id: chart.to_dict() for chart_id, chart in self.charts.items()},
            'system_status': await self.metrics_collector._get_system_status(),
            'alerts': await self.metrics_collector._get_active_alerts()
        }
    
    async def get_chart_data(self, chart_id: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific chart"""
        
        chart = self.charts.get(chart_id)
        if not chart:
            return None
        
        return chart.to_dict()
    
    async def refresh_chart(self, chart_id: str) -> bool:
        """Refresh a specific chart"""
        
        try:
            if chart_id == "satisfaction_trends":
                await self._update_satisfaction_trends()
            elif chart_id == "model_comparison":
                await self._update_model_comparison()
            elif chart_id == "usage_distribution":
                await self._update_usage_charts()
            elif chart_id == "performance_gauges":
                await self._update_performance_gauges()
            elif chart_id == "experiment_results":
                await self._update_experiment_charts()
            else:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing chart {chart_id}: {e}")
            return False
    
    async def export_dashboard_data(self, format: str = "json") -> str:
        """Export dashboard data for analysis"""
        
        dashboard_data = await self.get_dashboard_data()
        
        # Add historical metrics
        historical_metrics = await self.metrics_collector.collect_historical_metrics(
            days=self.config.historical_days
        )
        dashboard_data['historical_data'] = historical_metrics
        
        # Export to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_dir = Path('results/dashboard_exports')
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            output_file = export_dir / f"dashboard_export_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Dashboard data exported to {output_file}")
        return str(output_file)
    
    async def _background_update_loop(self):
        """Background task for updating dashboard data"""
        
        while self.is_running:
            try:
                await asyncio.sleep(self.config.refresh_interval.value)
                
                if self.is_running:
                    await self._update_all_charts()
                    
                    # Notify WebSocket connections of updates
                    await self._notify_websocket_clients()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _update_all_charts(self):
        """Update all dashboard charts"""
        
        try:
            # Update each chart type
            if self.config.show_user_satisfaction:
                await self._update_satisfaction_trends()
            
            if self.config.show_model_comparison:
                await self._update_model_comparison()
            
            if self.config.show_temporal_patterns:
                await self._update_usage_charts()
            
            if self.config.show_performance_metrics:
                await self._update_performance_gauges()
            
            if self.config.show_experiment_results:
                await self._update_experiment_charts()
            
            self.last_update = datetime.now()
            logger.debug("Updated all dashboard charts")
            
        except Exception as e:
            logger.error(f"Error updating dashboard charts: {e}")
    
    async def _update_satisfaction_trends(self):
        """Update user satisfaction trend chart"""
        
        historical_metrics = await self.metrics_collector.collect_historical_metrics(days=7)
        trends_data = historical_metrics.get('user_satisfaction_trends', {})
        
        chart = self.viz_engine.create_satisfaction_trend_chart(trends_data)
        self.charts['satisfaction_trends'] = chart
    
    async def _update_model_comparison(self):
        """Update model performance comparison chart"""
        
        historical_metrics = await self.metrics_collector.collect_historical_metrics(days=7)
        model_performance = historical_metrics.get('model_performance_history', {})
        
        chart = self.viz_engine.create_model_comparison_chart(model_performance)
        self.charts['model_comparison'] = chart
    
    async def _update_usage_charts(self):
        """Update usage pattern charts"""
        
        historical_metrics = await self.metrics_collector.collect_historical_metrics(days=7)
        usage_patterns = historical_metrics.get('usage_patterns', {})
        
        # Prompt type distribution
        prompt_dist = usage_patterns.get('prompt_type_distribution', {})
        chart1 = self.viz_engine.create_usage_distribution_pie(
            prompt_dist, 
            "Prompt Type Distribution", 
            "prompt_type_distribution"
        )
        self.charts['prompt_type_distribution'] = chart1
        
        # Model usage distribution
        model_dist = usage_patterns.get('model_usage_distribution', {})
        chart2 = self.viz_engine.create_usage_distribution_pie(
            model_dist, 
            "Model Usage Distribution", 
            "model_usage_distribution"
        )
        self.charts['model_usage_distribution'] = chart2
        
        # Hourly activity heatmap
        hourly_dist = usage_patterns.get('hourly_distribution', {})
        # Convert string keys to integers
        hourly_int = {int(k): v for k, v in hourly_dist.items() if k.isdigit()}
        chart3 = self.viz_engine.create_hourly_activity_heatmap(hourly_int)
        self.charts['hourly_activity'] = chart3
    
    async def _update_performance_gauges(self):
        """Update performance gauge charts"""
        
        current_performance = await self.metrics_collector._get_current_performance()
        chart = self.viz_engine.create_performance_gauge(current_performance)
        self.charts['performance_gauges'] = chart
    
    async def _update_experiment_charts(self):
        """Update experiment results charts"""
        
        historical_metrics = await self.metrics_collector.collect_historical_metrics(days=30)
        experiment_data = historical_metrics.get('experiment_results', {})
        
        chart = self.viz_engine.create_experiment_results_chart(experiment_data)
        self.charts['experiment_results'] = chart
    
    async def _notify_websocket_clients(self):
        """Notify WebSocket clients of dashboard updates"""
        
        if not self.websocket_connections:
            return
        
        try:
            dashboard_data = await self.get_dashboard_data()
            message = json.dumps({
                'type': 'dashboard_update',
                'data': dashboard_data
            }, default=str)
            
            # Send to all connected clients
            for websocket in self.websocket_connections[:]:  # Copy list to avoid modification during iteration
                try:
                    await websocket.send(message)
                except Exception:
                    # Remove disconnected clients
                    self.websocket_connections.remove(websocket)
        
        except Exception as e:
            logger.error(f"Error notifying WebSocket clients: {e}")
    
    def add_websocket_connection(self, websocket):
        """Add WebSocket connection for real-time updates"""
        self.websocket_connections.append(websocket)
        logger.debug(f"Added WebSocket connection. Total: {len(self.websocket_connections)}")
    
    def remove_websocket_connection(self, websocket):
        """Remove WebSocket connection"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
            logger.debug(f"Removed WebSocket connection. Total: {len(self.websocket_connections)}")
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        
        current_performance = await self.metrics_collector._get_current_performance()
        system_status = await self.metrics_collector._get_system_status()
        alerts = await self.metrics_collector._get_active_alerts()
        recent_activity = await self.metrics_collector._get_recent_activity()
        
        # Calculate health score
        health_score = 100
        
        # Deduct points for issues
        if current_performance.get('user_satisfaction', 5.0) < self.config.low_satisfaction_threshold:
            health_score -= 20
        
        if current_performance.get('response_time_ms', 0) > self.config.high_response_time_threshold:
            health_score -= 15
        
        if current_performance.get('cache_hit_rate_percent', 100) < self.config.low_cache_hit_threshold:
            health_score -= 10
        
        if len(alerts) > 0:
            health_score -= len(alerts) * 5
        
        health_score = max(0, health_score)
        
        # Determine overall status
        if health_score >= 90:
            overall_status = "excellent"
        elif health_score >= 75:
            overall_status = "good"
        elif health_score >= 50:
            overall_status = "fair"
        else:
            overall_status = "poor"
        
        return {
            'health_score': health_score,
            'overall_status': overall_status,
            'system_status': system_status,
            'performance_metrics': current_performance,
            'active_alerts': alerts,
            'recent_activity': recent_activity,
            'recommendations': self._generate_health_recommendations(health_score, alerts, current_performance)
        }
    
    def _generate_health_recommendations(self, 
                                       health_score: int, 
                                       alerts: List[Dict[str, Any]], 
                                       performance: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations"""
        
        recommendations = []
        
        if health_score < 75:
            recommendations.append("System health is below optimal. Review alerts and performance metrics.")
        
        if len(alerts) > 0:
            recommendations.append(f"Address {len(alerts)} active alerts to improve system stability.")
        
        if performance.get('user_satisfaction', 5.0) < 3.5:
            recommendations.append("User satisfaction is low. Consider reviewing model selection algorithms.")
        
        if performance.get('cache_hit_rate_percent', 100) < 70:
            recommendations.append("Cache hit rate is low. Review caching strategies and cache warming.")
        
        if performance.get('response_time_ms', 0) > 3000:
            recommendations.append("Response times are high. Consider performance optimization or scaling.")
        
        if not recommendations:
            recommendations.append("System is performing well. Continue monitoring for any changes.")
        
        return recommendations