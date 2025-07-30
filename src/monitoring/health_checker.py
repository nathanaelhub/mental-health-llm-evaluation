"""
Health Monitoring and Alerting System

Comprehensive health checking system that monitors all components of the
mental health LLM system and provides alerting capabilities.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthCheck:
    """Individual health check configuration"""
    name: str
    check_function: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True
    
    # State tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_time: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'alert_id': self.alert_id,
            'name': self.name,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class HealthMonitor:
    """Main health monitoring system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        
        # Monitoring state
        self.monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        logger.info("HealthMonitor initialized")
    
    def _setup_default_health_checks(self):
        """Setup default health checks for common components"""
        
        # Application health check
        self.add_health_check(
            name="app_health",
            check_function=self._check_app_health,
            interval_seconds=30,
            failure_threshold=3
        )
        
        # Database connectivity
        self.add_health_check(
            name="database_health",
            check_function=self._check_database_health,
            interval_seconds=60,
            failure_threshold=2
        )
        
        # Redis connectivity
        self.add_health_check(
            name="redis_health",
            check_function=self._check_redis_health,
            interval_seconds=30,
            failure_threshold=3
        )
        
        # System resources
        self.add_health_check(
            name="system_resources",
            check_function=self._check_system_resources,
            interval_seconds=30,
            failure_threshold=5
        )
        
        # Model selection performance
        self.add_health_check(
            name="model_selection_latency",
            check_function=self._check_model_selection_latency,
            interval_seconds=60,
            failure_threshold=3
        )
        
        # Cache performance
        self.add_health_check(
            name="cache_performance",
            check_function=self._check_cache_performance,
            interval_seconds=120,
            failure_threshold=3
        )
    
    def add_health_check(self, 
                        name: str, 
                        check_function: Callable, 
                        interval_seconds: int = 30,
                        failure_threshold: int = 3,
                        recovery_threshold: int = 2,
                        timeout_seconds: int = 10,
                        enabled: bool = True):
        """Add a health check"""
        
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            failure_threshold=failure_threshold,
            recovery_threshold=recovery_threshold,
            enabled=enabled
        )
        
        self.health_checks[name] = health_check
        logger.info(f"Added health check: {name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check"""
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Removed health check: {name}")
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    async def start_monitoring(self):
        """Start the health monitoring system"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop the health monitoring system"""
        self.monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Run health checks
                check_tasks = []
                
                for name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    # Check if it's time to run this check
                    if (health_check.last_check_time is None or 
                        datetime.now() - health_check.last_check_time >= 
                        timedelta(seconds=health_check.interval_seconds)):
                        
                        task = asyncio.create_task(
                            self._run_health_check(health_check)
                        )
                        check_tasks.append(task)
                
                # Wait for all checks to complete
                if check_tasks:
                    await asyncio.gather(*check_tasks, return_exceptions=True)
                
                # Sleep before next iteration
                await asyncio.sleep(10)  # Check every 10 seconds for due checks
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _run_health_check(self, health_check: HealthCheck):
        """Run an individual health check"""
        start_time = time.time()
        
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                health_check.check_function(),
                timeout=health_check.timeout_seconds
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Create result object
            if isinstance(result, HealthCheckResult):
                result.response_time_ms = response_time_ms
            else:
                # Convert simple result to HealthCheckResult
                if isinstance(result, bool):
                    status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                    message = "Check passed" if result else "Check failed"
                elif isinstance(result, dict):
                    status = HealthStatus(result.get('status', 'unknown'))
                    message = result.get('message', 'No message')
                else:
                    status = HealthStatus.HEALTHY
                    message = str(result)
                
                result = HealthCheckResult(
                    name=health_check.name,
                    status=status,
                    message=message,
                    response_time_ms=response_time_ms
                )
            
            # Update health check state
            health_check.last_check_time = datetime.now()
            health_check.last_error = None
            
            if result.status == HealthStatus.HEALTHY:
                health_check.consecutive_successes += 1
                health_check.consecutive_failures = 0
                
                # Check for recovery
                if (health_check.last_status in [HealthStatus.WARNING, HealthStatus.CRITICAL] and
                    health_check.consecutive_successes >= health_check.recovery_threshold):
                    await self._handle_recovery(health_check, result)
                
            else:
                health_check.consecutive_failures += 1
                health_check.consecutive_successes = 0
                
                # Check for failure threshold
                if health_check.consecutive_failures >= health_check.failure_threshold:
                    await self._handle_failure(health_check, result)
            
            health_check.last_status = result.status
            
        except asyncio.TimeoutError:
            health_check.consecutive_failures += 1
            health_check.consecutive_successes = 0
            health_check.last_check_time = datetime.now()
            health_check.last_error = "Timeout"
            
            result = HealthCheckResult(
                name=health_check.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {health_check.timeout_seconds}s",
                response_time_ms=(time.time() - start_time) * 1000
            )
            
            if health_check.consecutive_failures >= health_check.failure_threshold:
                await self._handle_failure(health_check, result)
        
        except Exception as e:
            health_check.consecutive_failures += 1
            health_check.consecutive_successes = 0
            health_check.last_check_time = datetime.now()
            health_check.last_error = str(e)
            
            result = HealthCheckResult(
                name=health_check.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
            
            if health_check.consecutive_failures >= health_check.failure_threshold:
                await self._handle_failure(health_check, result)
    
    async def _handle_failure(self, health_check: HealthCheck, result: HealthCheckResult):
        """Handle health check failure"""
        
        # Determine alert severity
        if result.status == HealthStatus.CRITICAL:
            severity = AlertSeverity.CRITICAL
        else:
            severity = AlertSeverity.WARNING
        
        # Create alert
        alert_id = f"{health_check.name}_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            name=f"Health Check Failed: {health_check.name}",
            severity=severity,
            message=f"{result.message} (Failed {health_check.consecutive_failures} times)",
            metadata={
                'health_check': health_check.name,
                'consecutive_failures': health_check.consecutive_failures,
                'last_error': health_check.last_error,
                'response_time_ms': result.response_time_ms
            }
        )
        
        self.alerts[alert_id] = alert
        
        # Send alert
        await self._send_alert(alert)
        
        logger.error(f"Health check failed: {health_check.name} - {result.message}")
    
    async def _handle_recovery(self, health_check: HealthCheck, result: HealthCheckResult):
        """Handle health check recovery"""
        
        # Create recovery alert
        alert_id = f"{health_check.name}_recovery_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            name=f"Health Check Recovered: {health_check.name}",
            severity=AlertSeverity.INFO,
            message=f"Health check is now healthy after {health_check.consecutive_successes} successful checks",
            metadata={
                'health_check': health_check.name,
                'consecutive_successes': health_check.consecutive_successes,
                'response_time_ms': result.response_time_ms
            }
        )
        
        self.alerts[alert_id] = alert
        
        # Send recovery alert
        await self._send_alert(alert)
        
        # Mark related failure alerts as resolved
        for existing_alert_id, existing_alert in self.alerts.items():
            if (existing_alert.metadata.get('health_check') == health_check.name and
                not existing_alert.resolved and
                existing_alert.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]):
                
                existing_alert.resolved = True
                existing_alert.resolved_at = datetime.now()
        
        logger.info(f"Health check recovered: {health_check.name}")
    
    async def _send_alert(self, alert: Alert):
        """Send alert to all registered handlers"""
        
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error sending alert via handler {handler.__name__}: {e}")
    
    # Default Health Check Functions
    
    async def _check_app_health(self) -> HealthCheckResult:
        """Check main application health"""
        
        app_url = self.config.get('app_url', 'http://localhost:8000')
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{app_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        return HealthCheckResult(
                            name="app_health",
                            status=HealthStatus.HEALTHY,
                            message="Application is healthy",
                            metadata=data
                        )
                    else:
                        return HealthCheckResult(
                            name="app_health",
                            status=HealthStatus.CRITICAL,
                            message=f"Application health check returned {response.status}"
                        )
        
        except Exception as e:
            return HealthCheckResult(
                name="app_health",
                status=HealthStatus.CRITICAL,
                message=f"Failed to connect to application: {str(e)}"
            )
    
    async def _check_database_health(self) -> HealthCheckResult:
        """Check database connectivity"""
        
        try:
            # This would use actual database connection in real implementation
            # For now, simulate the check
            postgres_url = self.config.get('postgres_url', 'postgresql://localhost:5432')
            
            # Simple connection test (would use actual DB library)
            import asyncpg
            
            conn = await asyncpg.connect(postgres_url)
            result = await conn.fetchval('SELECT 1')
            await conn.close()
            
            if result == 1:
                return HealthCheckResult(
                    name="database_health",
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful"
                )
            else:
                return HealthCheckResult(
                    name="database_health",
                    status=HealthStatus.CRITICAL,
                    message="Database query returned unexpected result"
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="database_health",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}"
            )
    
    async def _check_redis_health(self) -> HealthCheckResult:
        """Check Redis connectivity"""
        
        try:
            import redis.asyncio as redis
            
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            client = redis.from_url(redis_url)
            
            # Test ping
            result = await client.ping()
            await client.close()
            
            if result:
                return HealthCheckResult(
                    name="redis_health",
                    status=HealthStatus.HEALTHY,
                    message="Redis connection successful"
                )
            else:
                return HealthCheckResult(
                    name="redis_health",
                    status=HealthStatus.CRITICAL,
                    message="Redis ping failed"
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="redis_health",
                status=HealthStatus.CRITICAL,
                message=f"Redis connection failed: {str(e)}"
            )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage"""
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            messages = []
            
            cpu_threshold = self.config.get('cpu_threshold', 90)
            memory_threshold = self.config.get('memory_threshold', 90)
            disk_threshold = self.config.get('disk_threshold', 90)
            
            if cpu_percent > cpu_threshold:
                status = HealthStatus.CRITICAL
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > cpu_threshold * 0.8:
                status = HealthStatus.WARNING
                messages.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > memory_threshold:
                status = HealthStatus.CRITICAL
                messages.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > memory_threshold * 0.8:
                status = HealthStatus.WARNING
                messages.append(f"Elevated memory usage: {memory_percent:.1f}%")
            
            if disk_percent > disk_threshold:
                status = HealthStatus.CRITICAL
                messages.append(f"High disk usage: {disk_percent:.1f}%")
            elif disk_percent > disk_threshold * 0.8:
                status = HealthStatus.WARNING
                messages.append(f"Elevated disk usage: {disk_percent:.1f}%")
            
            if not messages:
                messages.append("System resources are healthy")
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message="; ".join(messages),
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent
                }
            )
        
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}"
            )
    
    async def _check_model_selection_latency(self) -> HealthCheckResult:
        """Check model selection performance"""
        
        try:
            app_url = self.config.get('app_url', 'http://localhost:8000')
            
            # Test model selection endpoint
            test_payload = {
                "user_id": "health_test",
                "message": "Test message for health check",
                "prompt_type": "general_wellness"
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(f"{app_url}/api/v1/select-model", json=test_payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        latency_ms = (time.time() - start_time) * 1000
                        
                        # Check latency threshold
                        latency_threshold = self.config.get('selection_latency_threshold', 3000)  # 3 seconds
                        
                        if latency_ms > latency_threshold:
                            status = HealthStatus.WARNING
                            message = f"Model selection latency high: {latency_ms:.0f}ms"
                        else:
                            status = HealthStatus.HEALTHY
                            message = f"Model selection latency normal: {latency_ms:.0f}ms"
                        
                        return HealthCheckResult(
                            name="model_selection_latency",
                            status=status,
                            message=message,
                            metadata={
                                'latency_ms': latency_ms,
                                'selected_model': data.get('selected_model'),
                                'confidence': data.get('confidence_score')
                            }
                        )
                    else:
                        return HealthCheckResult(
                            name="model_selection_latency",
                            status=HealthStatus.CRITICAL,
                            message=f"Model selection endpoint returned {response.status}"
                        )
        
        except Exception as e:
            return HealthCheckResult(
                name="model_selection_latency",
                status=HealthStatus.CRITICAL,
                message=f"Model selection health check failed: {str(e)}"
            )
    
    async def _check_cache_performance(self) -> HealthCheckResult:
        """Check cache performance"""
        
        try:
            app_url = self.config.get('app_url', 'http://localhost:8000')
            
            # Get cache metrics
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{app_url}/metrics") as response:
                    if response.status == 200:
                        metrics_text = await response.text()
                        
                        # Parse cache hit rate from metrics (simplified)
                        cache_hit_rate = 0.0
                        for line in metrics_text.split('\n'):
                            if 'cache_hit_rate' in line and not line.startswith('#'):
                                try:
                                    cache_hit_rate = float(line.split()[-1]) * 100
                                    break
                                except (ValueError, IndexError):
                                    pass
                        
                        # Check cache hit rate threshold
                        hit_rate_threshold = self.config.get('cache_hit_rate_threshold', 50.0)
                        
                        if cache_hit_rate < hit_rate_threshold:
                            status = HealthStatus.WARNING
                            message = f"Cache hit rate low: {cache_hit_rate:.1f}%"
                        else:
                            status = HealthStatus.HEALTHY
                            message = f"Cache hit rate normal: {cache_hit_rate:.1f}%"
                        
                        return HealthCheckResult(
                            name="cache_performance",
                            status=status,
                            message=message,
                            metadata={'cache_hit_rate': cache_hit_rate}
                        )
                    else:
                        return HealthCheckResult(
                            name="cache_performance",
                            status=HealthStatus.WARNING,
                            message=f"Could not retrieve cache metrics: {response.status}"
                        )
        
        except Exception as e:
            return HealthCheckResult(
                name="cache_performance",
                status=HealthStatus.WARNING,
                message=f"Cache performance check failed: {str(e)}"
            )
    
    # API Methods
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        
        overall_status = HealthStatus.HEALTHY
        health_checks_status = {}
        
        for name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            check_status = {
                'status': health_check.last_status.value,
                'last_check': health_check.last_check_time.isoformat() if health_check.last_check_time else None,
                'consecutive_failures': health_check.consecutive_failures,
                'consecutive_successes': health_check.consecutive_successes,
                'last_error': health_check.last_error
            }
            
            health_checks_status[name] = check_status
            
            # Update overall status
            if health_check.last_status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif health_check.last_status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'health_checks': health_checks_status,
            'active_alerts': len([a for a in self.alerts.values() if not a.resolved])
        }
    
    def get_alerts(self, resolved: bool = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by resolved status"""
        
        alerts = list(self.alerts.values())
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        return [alert.to_dict() for alert in alerts]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert"""
        
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            logger.info(f"Alert resolved manually: {alert_id}")
            return True
        
        return False
    
    async def run_manual_check(self, check_name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check manually"""
        
        if check_name not in self.health_checks:
            return None
        
        health_check = self.health_checks[check_name]
        await self._run_health_check(health_check)
        
        return HealthCheckResult(
            name=check_name,
            status=health_check.last_status,
            message=health_check.last_error or "Check completed",
            timestamp=health_check.last_check_time or datetime.now()
        )


# Alert Handlers

async def log_alert_handler(alert: Alert):
    """Simple log-based alert handler"""
    
    level = logging.ERROR if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else logging.WARNING
    logger.log(level, f"ALERT: {alert.name} - {alert.message}")


async def file_alert_handler(alert: Alert, alert_file: str = "results/alerts.json"):
    """File-based alert handler"""
    
    alert_path = Path(alert_file)
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing alerts
    alerts = []
    if alert_path.exists():
        try:
            with open(alert_path, 'r') as f:
                alerts = json.load(f)
        except Exception:
            pass
    
    # Add new alert
    alerts.append(alert.to_dict())
    
    # Keep only last 1000 alerts
    alerts = alerts[-1000:]
    
    # Save alerts
    with open(alert_path, 'w') as f:
        json.dump(alerts, f, indent=2)


async def webhook_alert_handler(alert: Alert, webhook_url: str):
    """Webhook-based alert handler"""
    
    payload = {
        'text': f"🚨 {alert.name}",
        'attachments': [{
            'color': 'danger' if alert.severity == AlertSeverity.CRITICAL else 'warning',
            'fields': [
                {'title': 'Severity', 'value': alert.severity.value, 'short': True},
                {'title': 'Time', 'value': alert.timestamp.isoformat(), 'short': True},
                {'title': 'Message', 'value': alert.message, 'short': False}
            ]
        }]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to send webhook alert: {response.status}")
    except Exception as e:
        logger.error(f"Error sending webhook alert: {e}")


# Main execution for testing
if __name__ == "__main__":
    async def main():
        # Create health monitor
        config = {
            'app_url': 'http://localhost:8000',
            'postgres_url': 'postgresql://postgres:postgres@localhost:5432/chat_sessions',
            'redis_url': 'redis://localhost:6379',
            'cpu_threshold': 80,
            'memory_threshold': 85,
            'disk_threshold': 90,
            'selection_latency_threshold': 3000,
            'cache_hit_rate_threshold': 60.0
        }
        
        monitor = HealthMonitor(config)
        
        # Add alert handlers
        monitor.add_alert_handler(log_alert_handler)
        monitor.add_alert_handler(lambda alert: file_alert_handler(alert, "results/development/alerts.json"))
        
        try:
            # Start monitoring
            await monitor.start_monitoring()
            
            print("Health monitoring started. Press Ctrl+C to stop.")
            
            # Keep running
            while True:
                await asyncio.sleep(30)
                
                # Print status every 30 seconds
                status = monitor.get_health_status()
                print(f"Overall Status: {status['overall_status']}")
                
                active_alerts = monitor.get_alerts(resolved=False)
                if active_alerts:
                    print(f"Active Alerts: {len(active_alerts)}")
        
        except KeyboardInterrupt:
            print("Stopping health monitoring...")
        finally:
            await monitor.stop_monitoring()
    
    asyncio.run(main())