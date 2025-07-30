"""
Load Testing Framework for Dynamic Model Selection System

Comprehensive load testing scenarios to validate system performance under
various load conditions, including concurrent users, sustained load, and
stress testing scenarios.
"""

import asyncio
import aiohttp
import time
import statistics
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import psutil
import matplotlib.pyplot as plt
from pathlib import Path

from src.chat.dynamic_model_selector import PromptType


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios"""
    
    # Test parameters
    concurrent_users: int = 100
    test_duration_seconds: int = 300  # 5 minutes
    ramp_up_seconds: int = 60
    ramp_down_seconds: int = 60
    
    # Request patterns
    requests_per_user_per_minute: int = 5
    think_time_seconds: float = 2.0
    think_time_variance: float = 1.0
    
    # Endpoint configuration
    base_url: str = "http://localhost:8000"
    endpoints: Dict[str, str] = field(default_factory=lambda: {
        "model_selection": "/api/v1/select-model",
        "chat": "/api/v1/chat",
        "feedback": "/api/v1/feedback",
        "health": "/api/v1/health"
    })
    
    # Resource monitoring
    monitor_resources: bool = True
    monitor_interval_seconds: float = 1.0
    
    # Test scenarios
    user_behavior_profiles: List[str] = field(default_factory=lambda: [
        "casual_user",      # Low frequency, simple queries
        "active_user",      # Regular usage, varied queries  
        "crisis_user",      # Urgent, high-priority requests
        "information_seeker" # Research-focused queries
    ])


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing"""
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Response time metrics
    response_times: List[float] = field(default_factory=list)
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    peak_rps: float = 0.0
    
    # Error metrics
    error_rate_percent: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Resource metrics
    cpu_usage_percent: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    disk_io_mb: List[float] = field(default_factory=list)
    network_io_mb: List[float] = field(default_factory=list)
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0
    
    # Model selection metrics
    selection_latency_ms: List[float] = field(default_factory=list)
    model_usage_distribution: Dict[str, int] = field(default_factory=dict)
    
    def calculate_percentiles(self):
        """Calculate response time percentiles"""
        if self.response_times:
            sorted_times = sorted(self.response_times)
            n = len(sorted_times)
            
            self.p50_response_time_ms = sorted_times[int(n * 0.5)]
            self.p95_response_time_ms = sorted_times[int(n * 0.95)]
            self.p99_response_time_ms = sorted_times[int(n * 0.99)]
            self.max_response_time_ms = max(sorted_times)
            self.avg_response_time_ms = statistics.mean(sorted_times)
    
    def calculate_error_rate(self):
        """Calculate error rate percentage"""
        if self.total_requests > 0:
            self.error_rate_percent = (self.failed_requests / self.total_requests) * 100


class UserBehaviorSimulator:
    """Simulates different user behavior patterns"""
    
    def __init__(self, profile: str, config: LoadTestConfig):
        self.profile = profile
        self.config = config
        self.session = None
        
        # Define behavior patterns
        self.behavior_patterns = {
            "casual_user": {
                "request_frequency": 0.5,  # Requests per minute
                "prompt_types": [PromptType.GENERAL_WELLNESS, PromptType.INFORMATION_SEEKING],
                "session_duration_minutes": 5,
                "think_time_multiplier": 2.0
            },
            "active_user": {
                "request_frequency": 2.0,
                "prompt_types": [PromptType.ANXIETY, PromptType.DEPRESSION, PromptType.GENERAL_WELLNESS],
                "session_duration_minutes": 15,
                "think_time_multiplier": 1.0
            },
            "crisis_user": {
                "request_frequency": 5.0,
                "prompt_types": [PromptType.CRISIS, PromptType.ANXIETY],
                "session_duration_minutes": 30,
                "think_time_multiplier": 0.2
            },
            "information_seeker": {
                "request_frequency": 1.5,
                "prompt_types": [PromptType.INFORMATION_SEEKING, PromptType.GENERAL_WELLNESS],
                "session_duration_minutes": 10,
                "think_time_multiplier": 1.5
            }
        }
        
        self.pattern = self.behavior_patterns.get(profile, self.behavior_patterns["active_user"])
        
        # Sample prompts for each type
        self.sample_prompts = {
            PromptType.CRISIS: [
                "I'm having thoughts of hurting myself",
                "I feel like I can't go on anymore",
                "Everything feels hopeless right now",
                "I'm in crisis and need immediate help"
            ],
            PromptType.ANXIETY: [
                "I'm feeling really anxious about work",
                "My anxiety is getting worse lately",
                "I can't stop worrying about everything",
                "I'm having panic attacks frequently"
            ],
            PromptType.DEPRESSION: [
                "I feel sad and empty all the time",
                "Nothing brings me joy anymore",
                "I'm struggling with depression",
                "I feel worthless and hopeless"
            ],
            PromptType.GENERAL_WELLNESS: [
                "How can I improve my sleep?",
                "What are some stress management techniques?",
                "I want to feel more positive",
                "Can you help me with self-care tips?"
            ],
            PromptType.INFORMATION_SEEKING: [
                "What are the symptoms of anxiety?",
                "Can you explain different types of therapy?",
                "How does medication work for depression?",
                "What's the difference between anxiety and panic?"
            ]
        }
    
    async def create_session(self) -> aiohttp.ClientSession:
        """Create HTTP session for requests"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    def get_random_prompt(self) -> tuple[str, PromptType]:
        """Get random prompt based on user profile"""
        prompt_type = random.choice(self.pattern["prompt_types"])
        prompts = self.sample_prompts[prompt_type]
        prompt_text = random.choice(prompts)
        return prompt_text, prompt_type
    
    def calculate_think_time(self) -> float:
        """Calculate think time between requests"""
        base_time = self.config.think_time_seconds
        variance = self.config.think_time_variance
        multiplier = self.pattern["think_time_multiplier"]
        
        # Add random variance
        actual_time = base_time * multiplier
        actual_time += random.uniform(-variance, variance)
        
        return max(0.1, actual_time)  # Minimum 0.1 seconds


class ResourceMonitor:
    """Monitors system resources during load testing"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics = LoadTestMetrics()
        self._monitor_task = None
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.metrics.cpu_usage_percent.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.metrics.memory_usage_mb.append(memory_mb)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_mb = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)
                    self.metrics.disk_io_mb.append(disk_mb)
                
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    network_mb = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)
                    self.metrics.network_io_mb.append(network_mb)
                
                await asyncio.sleep(self.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.interval)


class LoadTestRunner:
    """Main load testing orchestrator"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics = LoadTestMetrics()
        self.resource_monitor = ResourceMonitor(config.monitor_interval_seconds)
        self.test_start_time = None
        self.test_end_time = None
        
    async def run_load_test(self) -> Dict[str, Any]:
        """Execute comprehensive load test"""
        
        print(f"Starting load test with {self.config.concurrent_users} concurrent users...")
        print(f"Test duration: {self.config.test_duration_seconds}s")
        print(f"Ramp-up: {self.config.ramp_up_seconds}s, Ramp-down: {self.config.ramp_down_seconds}s")
        
        self.test_start_time = datetime.now()
        
        # Start resource monitoring
        if self.config.monitor_resources:
            await self.resource_monitor.start_monitoring()
        
        try:
            # Execute load test phases
            await self._run_ramp_up_phase()
            await self._run_sustained_load_phase()
            await self._run_ramp_down_phase()
            
        finally:
            # Stop resource monitoring
            if self.config.monitor_resources:
                await self.resource_monitor.stop_monitoring()
            
            self.test_end_time = datetime.now()
        
        # Process and return results
        return await self._generate_test_report()
    
    async def _run_ramp_up_phase(self):
        """Gradually increase load to full capacity"""
        print("Phase 1: Ramp-up")
        
        ramp_interval = self.config.ramp_up_seconds / self.config.concurrent_users
        user_tasks = []
        
        for i in range(self.config.concurrent_users):
            # Start user simulation
            user_profile = random.choice(self.config.user_behavior_profiles)
            user_simulator = UserBehaviorSimulator(user_profile, self.config)
            
            task = asyncio.create_task(
                self._simulate_user_session(user_simulator, f"rampup_user_{i}")
            )
            user_tasks.append(task)
            
            # Stagger user starts
            if i < self.config.concurrent_users - 1:
                await asyncio.sleep(ramp_interval)
        
        # Wait a bit for ramp-up to stabilize
        await asyncio.sleep(10)
    
    async def _run_sustained_load_phase(self):
        """Run sustained load for the main test duration"""
        print("Phase 2: Sustained Load")
        
        # Create user simulators for sustained load
        user_tasks = []
        
        for i in range(self.config.concurrent_users):
            user_profile = random.choice(self.config.user_behavior_profiles)
            user_simulator = UserBehaviorSimulator(user_profile, self.config)
            
            task = asyncio.create_task(
                self._simulate_user_session(user_simulator, f"sustained_user_{i}")
            )
            user_tasks.append(task)
        
        # Run for the configured duration
        main_duration = (
            self.config.test_duration_seconds - 
            self.config.ramp_up_seconds - 
            self.config.ramp_down_seconds
        )
        
        await asyncio.sleep(main_duration)
        
        # Cancel user tasks
        for task in user_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*user_tasks, return_exceptions=True)
    
    async def _run_ramp_down_phase(self):
        """Gradually reduce load"""
        print("Phase 3: Ramp-down")
        
        # Gradual ramp-down is handled by natural completion of user sessions
        await asyncio.sleep(self.config.ramp_down_seconds)
    
    async def _simulate_user_session(self, user_simulator: UserBehaviorSimulator, user_id: str):
        """Simulate a single user session"""
        
        await user_simulator.create_session()
        
        try:
            session_duration = user_simulator.pattern["session_duration_minutes"] * 60
            session_start = time.time()
            
            while (time.time() - session_start) < session_duration:
                # Make a request
                await self._make_user_request(user_simulator, user_id)
                
                # Think time between requests
                think_time = user_simulator.calculate_think_time()
                await asyncio.sleep(think_time)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in user session {user_id}: {e}")
        finally:
            await user_simulator.close_session()
    
    async def _make_user_request(self, user_simulator: UserBehaviorSimulator, user_id: str):
        """Make a single request as a user"""
        
        prompt_text, prompt_type = user_simulator.get_random_prompt()
        
        # Prepare request data
        request_data = {
            "user_id": user_id,
            "message": prompt_text,
            "prompt_type": prompt_type.value,
            "session_id": f"session_{user_id}_{int(time.time())}"
        }
        
        # Model selection request
        await self._make_api_request(
            user_simulator.session,
            "model_selection",
            request_data,
            "model_selection"
        )
        
        # Chat request (simulate getting response)
        chat_data = {
            **request_data,
            "selected_model": "gpt-3.5-turbo"  # Would come from selection response
        }
        
        await self._make_api_request(
            user_simulator.session,
            "chat", 
            chat_data,
            "chat"
        )
        
        # Occasionally provide feedback
        if random.random() < 0.3:  # 30% chance of feedback
            feedback_data = {
                "user_id": user_id,
                "session_id": request_data["session_id"],
                "thumbs_up": random.choice([True, False]),
                "rating": random.uniform(1.0, 5.0)
            }
            
            await self._make_api_request(
                user_simulator.session,
                "feedback",
                feedback_data,
                "feedback"
            )
    
    async def _make_api_request(self, 
                              session: aiohttp.ClientSession, 
                              endpoint_name: str, 
                              data: Dict[str, Any], 
                              request_type: str):
        """Make API request and collect metrics"""
        
        url = f"{self.config.base_url}{self.config.endpoints[endpoint_name]}"
        
        start_time = time.time()
        
        try:
            async with session.post(url, json=data) as response:
                response_time_ms = (time.time() - start_time) * 1000
                
                self.metrics.total_requests += 1
                self.metrics.response_times.append(response_time_ms)
                
                if request_type == "model_selection":
                    self.metrics.selection_latency_ms.append(response_time_ms)
                
                if response.status == 200:
                    self.metrics.successful_requests += 1
                    
                    # Extract model selection info
                    if request_type == "model_selection":
                        response_data = await response.json()
                        selected_model = response_data.get("selected_model")
                        if selected_model:
                            if selected_model not in self.metrics.model_usage_distribution:
                                self.metrics.model_usage_distribution[selected_model] = 0
                            self.metrics.model_usage_distribution[selected_model] += 1
                
                else:
                    self.metrics.failed_requests += 1
                    error_type = f"HTTP_{response.status}"
                    if error_type not in self.metrics.errors_by_type:
                        self.metrics.errors_by_type[error_type] = 0
                    self.metrics.errors_by_type[error_type] += 1
                
        except asyncio.TimeoutError:
            self.metrics.failed_requests += 1
            self.metrics.total_requests += 1
            self.metrics.errors_by_type["timeout"] = self.metrics.errors_by_type.get("timeout", 0) + 1
            
        except Exception as e:
            self.metrics.failed_requests += 1
            self.metrics.total_requests += 1
            error_type = type(e).__name__
            self.metrics.errors_by_type[error_type] = self.metrics.errors_by_type.get(error_type, 0) + 1
    
    async def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Calculate final metrics
        self.metrics.calculate_percentiles()
        self.metrics.calculate_error_rate()
        
        # Calculate throughput
        if self.test_start_time and self.test_end_time:
            total_duration = (self.test_end_time - self.test_start_time).total_seconds()
            self.metrics.requests_per_second = self.metrics.total_requests / total_duration if total_duration > 0 else 0
        
        # Combine with resource metrics
        if self.config.monitor_resources:
            self.metrics.cpu_usage_percent = self.resource_monitor.metrics.cpu_usage_percent
            self.metrics.memory_usage_mb = self.resource_monitor.metrics.memory_usage_mb
            self.metrics.disk_io_mb = self.resource_monitor.metrics.disk_io_mb
            self.metrics.network_io_mb = self.resource_monitor.metrics.network_io_mb
        
        # Generate report
        report = {
            "test_configuration": {
                "concurrent_users": self.config.concurrent_users,
                "test_duration_seconds": self.config.test_duration_seconds,
                "total_duration_seconds": (self.test_end_time - self.test_start_time).total_seconds(),
                "user_behavior_profiles": self.config.user_behavior_profiles
            },
            "performance_metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "error_rate_percent": self.metrics.error_rate_percent,
                "requests_per_second": self.metrics.requests_per_second,
                "response_times": {
                    "average_ms": self.metrics.avg_response_time_ms,
                    "p50_ms": self.metrics.p50_response_time_ms,
                    "p95_ms": self.metrics.p95_response_time_ms,
                    "p99_ms": self.metrics.p99_response_time_ms,
                    "max_ms": self.metrics.max_response_time_ms
                }
            },
            "model_selection_metrics": {
                "avg_selection_latency_ms": statistics.mean(self.metrics.selection_latency_ms) if self.metrics.selection_latency_ms else 0,
                "model_usage_distribution": self.metrics.model_usage_distribution,
                "total_selections": sum(self.metrics.model_usage_distribution.values())
            },
            "error_analysis": {
                "errors_by_type": self.metrics.errors_by_type,
                "error_rate_threshold_exceeded": self.metrics.error_rate_percent > 5.0
            },
            "resource_utilization": {
                "cpu": {
                    "avg_percent": statistics.mean(self.metrics.cpu_usage_percent) if self.metrics.cpu_usage_percent else 0,
                    "max_percent": max(self.metrics.cpu_usage_percent) if self.metrics.cpu_usage_percent else 0
                },
                "memory": {
                    "avg_mb": statistics.mean(self.metrics.memory_usage_mb) if self.metrics.memory_usage_mb else 0,
                    "max_mb": max(self.metrics.memory_usage_mb) if self.metrics.memory_usage_mb else 0
                }
            },
            "test_verdict": self._determine_test_verdict()
        }
        
        return report
    
    def _determine_test_verdict(self) -> Dict[str, Any]:
        """Determine overall test pass/fail verdict"""
        
        verdict = {
            "overall_pass": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check error rate
        if self.metrics.error_rate_percent > 5.0:
            verdict["overall_pass"] = False
            verdict["issues"].append(f"High error rate: {self.metrics.error_rate_percent:.2f}%")
            verdict["recommendations"].append("Investigate error causes and improve error handling")
        
        # Check response times
        if self.metrics.p95_response_time_ms > 5000:  # 5 seconds
            verdict["overall_pass"] = False
            verdict["issues"].append(f"High P95 response time: {self.metrics.p95_response_time_ms:.0f}ms")
            verdict["recommendations"].append("Optimize response time or consider scaling")
        
        # Check throughput
        expected_min_rps = self.config.concurrent_users * 0.1  # Conservative estimate
        if self.metrics.requests_per_second < expected_min_rps:
            verdict["issues"].append(f"Low throughput: {self.metrics.requests_per_second:.2f} RPS")
            verdict["recommendations"].append("Investigate performance bottlenecks")
        
        # Check model selection latency
        if self.metrics.selection_latency_ms:
            avg_selection_latency = statistics.mean(self.metrics.selection_latency_ms)
            if avg_selection_latency > 3000:  # 3 seconds
                verdict["issues"].append(f"High model selection latency: {avg_selection_latency:.0f}ms")
                verdict["recommendations"].append("Optimize model selection algorithm or caching")
        
        if not verdict["issues"]:
            verdict["overall_pass"] = True
            verdict["recommendations"].append("System performance is within acceptable limits")
        
        return verdict
    
    async def generate_load_test_report(self, output_dir: str = "results/load_tests"):
        """Generate detailed load test report with visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        report = await self._generate_test_report()
        
        with open(output_path / f"load_test_report_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        await self._generate_visualizations(output_path, timestamp)
        
        print(f"Load test report saved to {output_path}")
        return report
    
    async def _generate_visualizations(self, output_path: Path, timestamp: str):
        """Generate performance visualization charts"""
        
        # Response time distribution
        if self.metrics.response_times:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(self.metrics.response_times, bins=50, alpha=0.7)
            plt.xlabel('Response Time (ms)')
            plt.ylabel('Frequency')
            plt.title('Response Time Distribution')
            
            # Model usage distribution
            plt.subplot(2, 2, 2)
            if self.metrics.model_usage_distribution:
                models = list(self.metrics.model_usage_distribution.keys())
                usage = list(self.metrics.model_usage_distribution.values())
                plt.pie(usage, labels=models, autopct='%1.1f%%')
                plt.title('Model Usage Distribution')
            
            # Resource utilization over time
            plt.subplot(2, 2, 3)
            if self.metrics.cpu_usage_percent:
                time_points = list(range(len(self.metrics.cpu_usage_percent)))
                plt.plot(time_points, self.metrics.cpu_usage_percent, label='CPU %')
                plt.plot(time_points, [m/100 for m in self.metrics.memory_usage_mb], label='Memory %')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Usage %')
                plt.title('Resource Utilization')
                plt.legend()
            
            # Response time over time
            plt.subplot(2, 2, 4)
            if len(self.metrics.response_times) > 100:
                # Sample data points for readability
                sample_size = min(1000, len(self.metrics.response_times))
                sample_indices = random.sample(range(len(self.metrics.response_times)), sample_size)
                sample_times = [self.metrics.response_times[i] for i in sorted(sample_indices)]
                
                plt.plot(sample_times)
                plt.xlabel('Request Number')
                plt.ylabel('Response Time (ms)')
                plt.title('Response Time Trend')
            
            plt.tight_layout()
            plt.savefig(output_path / f"load_test_charts_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()


# Load Test Scenarios

class LoadTestScenarios:
    """Predefined load testing scenarios"""
    
    @staticmethod
    def get_baseline_test() -> LoadTestConfig:
        """Basic load test configuration"""
        return LoadTestConfig(
            concurrent_users=50,
            test_duration_seconds=300,
            ramp_up_seconds=30,
            ramp_down_seconds=30
        )
    
    @staticmethod
    def get_stress_test() -> LoadTestConfig:
        """High-load stress test configuration"""
        return LoadTestConfig(
            concurrent_users=200,
            test_duration_seconds=600,
            ramp_up_seconds=120,
            ramp_down_seconds=60,
            requests_per_user_per_minute=10
        )
    
    @staticmethod
    def get_endurance_test() -> LoadTestConfig:
        """Long-duration endurance test"""
        return LoadTestConfig(
            concurrent_users=100,
            test_duration_seconds=3600,  # 1 hour
            ramp_up_seconds=300,
            ramp_down_seconds=300,
            requests_per_user_per_minute=3
        )
    
    @staticmethod
    def get_crisis_load_test() -> LoadTestConfig:
        """Crisis-heavy load test scenario"""
        config = LoadTestConfig(
            concurrent_users=75,
            test_duration_seconds=300,
            ramp_up_seconds=60,
            ramp_down_seconds=30
        )
        
        # Override user profiles to focus on crisis scenarios
        config.user_behavior_profiles = ["crisis_user"] * 3 + ["active_user"]
        
        return config


# Main execution
async def run_load_test_suite():
    """Run comprehensive load test suite"""
    
    scenarios = [
        ("baseline", LoadTestScenarios.get_baseline_test()),
        ("stress", LoadTestScenarios.get_stress_test()),
        ("crisis_load", LoadTestScenarios.get_crisis_load_test())
    ]
    
    results = {}
    
    for scenario_name, config in scenarios:
        print(f"\n{'='*60}")
        print(f"RUNNING {scenario_name.upper()} LOAD TEST")
        print(f"{'='*60}")
        
        runner = LoadTestRunner(config)
        
        try:
            result = await runner.run_load_test()
            results[scenario_name] = result
            
            # Save individual report
            await runner.generate_load_test_report(f"results/load_tests/{scenario_name}")
            
            # Print summary
            print(f"\n{scenario_name.upper()} RESULTS:")
            print(f"  Total Requests: {result['performance_metrics']['total_requests']}")
            print(f"  Error Rate: {result['performance_metrics']['error_rate_percent']:.2f}%")
            print(f"  Avg Response Time: {result['performance_metrics']['response_times']['average_ms']:.0f}ms")
            print(f"  P95 Response Time: {result['performance_metrics']['response_times']['p95_ms']:.0f}ms")
            print(f"  Throughput: {result['performance_metrics']['requests_per_second']:.2f} RPS")
            print(f"  Verdict: {'PASS' if result['test_verdict']['overall_pass'] else 'FAIL'}")
            
            if result['test_verdict']['issues']:
                print("  Issues:")
                for issue in result['test_verdict']['issues']:
                    print(f"    - {issue}")
            
        except Exception as e:
            print(f"Error running {scenario_name} test: {e}")
            results[scenario_name] = {"error": str(e)}
    
    # Generate combined report
    combined_report = {
        "test_suite_timestamp": datetime.now().isoformat(),
        "scenarios": results,
        "summary": {
            "total_scenarios": len(scenarios),
            "passed_scenarios": sum(1 for r in results.values() 
                                  if isinstance(r, dict) and 
                                     r.get('test_verdict', {}).get('overall_pass', False)),
            "failed_scenarios": sum(1 for r in results.values() 
                                  if isinstance(r, dict) and 
                                     not r.get('test_verdict', {}).get('overall_pass', True))
        }
    }
    
    # Save combined report
    output_path = Path("results/load_tests")
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_path / f"load_test_suite_{timestamp}.json", 'w') as f:
        json.dump(combined_report, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("LOAD TEST SUITE COMPLETE")
    print(f"{'='*60}")
    print(f"Total Scenarios: {combined_report['summary']['total_scenarios']}")
    print(f"Passed: {combined_report['summary']['passed_scenarios']}")
    print(f"Failed: {combined_report['summary']['failed_scenarios']}")
    
    return combined_report


if __name__ == "__main__":
    # Run load tests
    asyncio.run(run_load_test_suite())