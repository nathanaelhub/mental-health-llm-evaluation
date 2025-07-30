"""
Advanced Alerting System with Multiple Notification Channels

Comprehensive alerting system that supports multiple notification channels,
alert routing, escalation policies, and alert correlation.
"""

import asyncio
import logging
import smtplib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
from pathlib import Path

from .health_checker import Alert, AlertSeverity

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    LOG = "log"
    FILE = "file"


class EscalationLevel(Enum):
    """Escalation levels for alerts"""
    LEVEL_1 = "level_1"  # First responders
    LEVEL_2 = "level_2"  # Team leads
    LEVEL_3 = "level_3"  # Management
    LEVEL_4 = "level_4"  # Executive


@dataclass
class NotificationConfig:
    """Configuration for a notification channel"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Filtering
    min_severity: AlertSeverity = AlertSeverity.INFO
    alert_patterns: List[str] = field(default_factory=list)  # Regex patterns
    
    # Rate limiting
    rate_limit_minutes: int = 5
    max_alerts_per_period: int = 10


@dataclass
class EscalationPolicy:
    """Escalation policy configuration"""
    name: str
    alert_patterns: List[str] = field(default_factory=list)
    
    # Escalation levels
    levels: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    initial_delay_minutes: int = 0
    escalation_delay_minutes: int = 15
    max_escalations: int = 3
    
    # State
    active_escalations: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class AlertGroup:
    """Group of related alerts for correlation"""
    group_id: str
    pattern: str
    alerts: List[Alert] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    summary_sent: bool = False


class AlertManager:
    """Advanced alert management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Notification channels
        self.notification_configs: Dict[str, NotificationConfig] = {}
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}
        
        # Escalation policies
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        
        # Alert correlation
        self.alert_groups: Dict[str, AlertGroup] = {}
        self.correlation_rules: List[Dict[str, Any]] = []
        
        # Rate limiting
        self.notification_history: Dict[str, List[datetime]] = {}
        
        # Setup default handlers
        self._setup_notification_handlers()
        
        # Background tasks
        self._escalation_task: Optional[asyncio.Task] = None
        self._correlation_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("AlertManager initialized")
    
    def _setup_notification_handlers(self):
        """Setup default notification handlers"""
        
        self.notification_handlers = {
            NotificationChannel.EMAIL: self._send_email_notification,
            NotificationChannel.SLACK: self._send_slack_notification,
            NotificationChannel.WEBHOOK: self._send_webhook_notification,
            NotificationChannel.SMS: self._send_sms_notification,
            NotificationChannel.PAGERDUTY: self._send_pagerduty_notification,
            NotificationChannel.LOG: self._send_log_notification,
            NotificationChannel.FILE: self._send_file_notification
        }
    
    def add_notification_channel(self, 
                               name: str, 
                               channel: NotificationChannel,
                               config: Dict[str, Any],
                               min_severity: AlertSeverity = AlertSeverity.INFO,
                               alert_patterns: List[str] = None,
                               rate_limit_minutes: int = 5,
                               max_alerts_per_period: int = 10):
        """Add a notification channel"""
        
        notification_config = NotificationConfig(
            channel=channel,
            config=config,
            min_severity=min_severity,
            alert_patterns=alert_patterns or [],
            rate_limit_minutes=rate_limit_minutes,
            max_alerts_per_period=max_alerts_per_period
        )
        
        self.notification_configs[name] = notification_config
        logger.info(f"Added notification channel: {name} ({channel.value})")
    
    def add_escalation_policy(self,
                            name: str,
                            alert_patterns: List[str],
                            levels: List[Dict[str, Any]],
                            initial_delay_minutes: int = 0,
                            escalation_delay_minutes: int = 15,
                            max_escalations: int = 3):
        """Add an escalation policy"""
        
        policy = EscalationPolicy(
            name=name,
            alert_patterns=alert_patterns,
            levels=levels,
            initial_delay_minutes=initial_delay_minutes,
            escalation_delay_minutes=escalation_delay_minutes,
            max_escalations=max_escalations
        )
        
        self.escalation_policies[name] = policy
        logger.info(f"Added escalation policy: {name}")
    
    def add_correlation_rule(self,
                           pattern: str,
                           time_window_minutes: int = 5,
                           min_alerts: int = 2,
                           summary_template: str = None):
        """Add alert correlation rule"""
        
        rule = {
            'pattern': pattern,
            'time_window_minutes': time_window_minutes,
            'min_alerts': min_alerts,
            'summary_template': summary_template or f"Multiple alerts matching pattern: {pattern}"
        }
        
        self.correlation_rules.append(rule)
        logger.info(f"Added correlation rule: {pattern}")
    
    async def start(self):
        """Start the alert manager background tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._escalation_task = asyncio.create_task(self._escalation_loop())
        self._correlation_task = asyncio.create_task(self._correlation_loop())
        
        logger.info("AlertManager started")
    
    async def stop(self):
        """Stop the alert manager"""
        self._running = False
        
        if self._escalation_task:
            self._escalation_task.cancel()
            try:
                await self._escalation_task
            except asyncio.CancelledError:
                pass
        
        if self._correlation_task:
            self._correlation_task.cancel()
            try:
                await self._correlation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AlertManager stopped")
    
    async def process_alert(self, alert: Alert):
        """Process an incoming alert"""
        
        try:
            # Check for correlation
            await self._correlate_alert(alert)
            
            # Check for escalation policies
            await self._check_escalation_policies(alert)
            
            # Send notifications
            await self._send_notifications(alert)
            
            logger.info(f"Processed alert: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.alert_id}: {e}")
    
    async def _correlate_alert(self, alert: Alert):
        """Check if alert should be correlated with existing alerts"""
        
        import re
        
        for rule in self.correlation_rules:
            pattern = rule['pattern']
            
            # Check if alert matches pattern
            if re.search(pattern, alert.name) or re.search(pattern, alert.message):
                
                # Find or create alert group
                group_id = f"{pattern}_{int(datetime.now().timestamp() // (rule['time_window_minutes'] * 60))}"
                
                if group_id not in self.alert_groups:
                    self.alert_groups[group_id] = AlertGroup(
                        group_id=group_id,
                        pattern=pattern
                    )
                
                group = self.alert_groups[group_id]
                group.alerts.append(alert)
                group.last_updated = datetime.now()
                
                logger.debug(f"Added alert {alert.alert_id} to correlation group {group_id}")
                break
    
    async def _check_escalation_policies(self, alert: Alert):
        """Check if alert triggers any escalation policies"""
        
        import re
        
        for policy_name, policy in self.escalation_policies.items():
            
            # Check if alert matches any pattern
            matches = False
            for pattern in policy.alert_patterns:
                if re.search(pattern, alert.name) or re.search(pattern, alert.message):
                    matches = True
                    break
            
            if matches:
                # Start escalation
                escalation_id = f"{alert.alert_id}_{policy_name}"
                
                policy.active_escalations[escalation_id] = {
                    'alert': alert,
                    'policy': policy_name,
                    'current_level': 0,
                    'started_at': datetime.now(),
                    'last_escalated_at': datetime.now(),
                    'acknowledged': False
                }
                
                logger.info(f"Started escalation {escalation_id} for alert {alert.alert_id}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        
        import re
        
        for channel_name, config in self.notification_configs.items():
            
            if not config.enabled:
                continue
            
            # Check severity filter
            if alert.severity.value < config.min_severity.value:
                continue
            
            # Check pattern filters
            if config.alert_patterns:
                matches = False
                for pattern in config.alert_patterns:
                    if re.search(pattern, alert.name) or re.search(pattern, alert.message):
                        matches = True
                        break
                
                if not matches:
                    continue
            
            # Check rate limiting
            if not self._check_rate_limit(channel_name, config):
                logger.warning(f"Rate limit exceeded for channel {channel_name}")
                continue
            
            # Send notification
            try:
                handler = self.notification_handlers.get(config.channel)
                if handler:
                    await handler(alert, config.config)
                    self._record_notification(channel_name)
                    logger.debug(f"Sent notification via {channel_name}")
                else:
                    logger.warning(f"No handler for channel type {config.channel}")
            
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    def _check_rate_limit(self, channel_name: str, config: NotificationConfig) -> bool:
        """Check if notification is within rate limits"""
        
        now = datetime.now()
        cutoff = now - timedelta(minutes=config.rate_limit_minutes)
        
        # Clean old entries
        if channel_name in self.notification_history:
            self.notification_history[channel_name] = [
                ts for ts in self.notification_history[channel_name] if ts > cutoff
            ]
        else:
            self.notification_history[channel_name] = []
        
        # Check rate limit
        return len(self.notification_history[channel_name]) < config.max_alerts_per_period
    
    def _record_notification(self, channel_name: str):
        """Record that a notification was sent"""
        
        if channel_name not in self.notification_history:
            self.notification_history[channel_name] = []
        
        self.notification_history[channel_name].append(datetime.now())
    
    async def _escalation_loop(self):
        """Background task for handling escalations"""
        
        while self._running:
            try:
                now = datetime.now()
                
                for policy_name, policy in self.escalation_policies.items():
                    escalations_to_remove = []
                    
                    for escalation_id, escalation in policy.active_escalations.items():
                        
                        if escalation['acknowledged']:
                            escalations_to_remove.append(escalation_id)
                            continue
                        
                        # Check if it's time to escalate
                        time_since_start = now - escalation['started_at']
                        time_since_last = now - escalation['last_escalated_at']
                        
                        should_escalate = False
                        
                        # Initial escalation
                        if (escalation['current_level'] == 0 and 
                            time_since_start.total_seconds() >= policy.initial_delay_minutes * 60):
                            should_escalate = True
                        
                        # Subsequent escalations
                        elif (escalation['current_level'] > 0 and
                              time_since_last.total_seconds() >= policy.escalation_delay_minutes * 60):
                            should_escalate = True
                        
                        if should_escalate and escalation['current_level'] < policy.max_escalations:
                            await self._escalate(escalation_id, escalation, policy)
                        
                        # Remove if max escalations reached
                        elif escalation['current_level'] >= policy.max_escalations:
                            escalations_to_remove.append(escalation_id)
                    
                    # Clean up completed escalations
                    for escalation_id in escalations_to_remove:
                        del policy.active_escalations[escalation_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
                await asyncio.sleep(60)
    
    async def _escalate(self, escalation_id: str, escalation: Dict[str, Any], policy: EscalationPolicy):
        """Escalate an alert to the next level"""
        
        level = escalation['current_level']
        
        if level < len(policy.levels):
            level_config = policy.levels[level]
            alert = escalation['alert']
            
            # Create escalation alert
            escalation_alert = Alert(
                alert_id=f"{alert.alert_id}_escalation_{level}",
                name=f"ESCALATION L{level + 1}: {alert.name}",
                severity=AlertSeverity.CRITICAL,  # Escalations are always critical
                message=f"Alert escalated to level {level + 1}: {alert.message}",
                metadata={
                    **alert.metadata,
                    'escalation_level': level + 1,
                    'original_alert_id': alert.alert_id,
                    'escalation_policy': policy.name
                }
            )
            
            # Send to escalation recipients
            recipients = level_config.get('recipients', [])
            for recipient in recipients:
                try:
                    # Send notification based on recipient type
                    if recipient['type'] == 'email':
                        await self._send_email_notification(escalation_alert, {
                            'smtp_server': recipient.get('smtp_server', 'localhost'),
                            'smtp_port': recipient.get('smtp_port', 587),
                            'username': recipient.get('username'),
                            'password': recipient.get('password'),
                            'to_email': recipient['email'],
                            'from_email': recipient.get('from_email', 'alerts@system.com')
                        })
                    elif recipient['type'] == 'slack':
                        await self._send_slack_notification(escalation_alert, {
                            'webhook_url': recipient['webhook_url']
                        })
                    # Add more recipient types as needed
                
                except Exception as e:
                    logger.error(f"Failed to send escalation notification to {recipient}: {e}")
            
            # Update escalation state
            escalation['current_level'] = level + 1
            escalation['last_escalated_at'] = datetime.now()
            
            logger.info(f"Escalated {escalation_id} to level {level + 1}")
    
    async def _correlation_loop(self):
        """Background task for alert correlation and summary"""
        
        while self._running:
            try:
                now = datetime.now()
                
                groups_to_remove = []
                
                for group_id, group in self.alert_groups.items():
                    
                    # Find matching rule
                    rule = None
                    for r in self.correlation_rules:
                        if r['pattern'] in group.pattern:
                            rule = r
                            break
                    
                    if not rule:
                        continue
                    
                    # Check if group is ready for summary
                    time_since_last_update = now - group.last_updated
                    
                    if (not group.summary_sent and
                        len(group.alerts) >= rule['min_alerts'] and
                        time_since_last_update.total_seconds() >= rule['time_window_minutes'] * 60):
                        
                        # Send correlation summary
                        await self._send_correlation_summary(group, rule)
                        group.summary_sent = True
                    
                    # Remove old groups
                    if time_since_last_update.total_seconds() >= rule['time_window_minutes'] * 60 * 2:
                        groups_to_remove.append(group_id)
                
                # Clean up old groups
                for group_id in groups_to_remove:
                    del self.alert_groups[group_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in correlation loop: {e}")
                await asyncio.sleep(60)
    
    async def _send_correlation_summary(self, group: AlertGroup, rule: Dict[str, Any]):
        """Send a correlation summary alert"""
        
        summary_alert = Alert(
            alert_id=f"correlation_{group.group_id}",
            name=f"Alert Correlation: {group.pattern}",
            severity=AlertSeverity.WARNING,
            message=rule['summary_template'].format(
                count=len(group.alerts),
                pattern=group.pattern,
                time_window=rule['time_window_minutes']
            ),
            metadata={
                'correlation_group_id': group.group_id,
                'pattern': group.pattern,
                'alert_count': len(group.alerts),
                'alert_ids': [alert.alert_id for alert in group.alerts],
                'time_window_minutes': rule['time_window_minutes']
            }
        )
        
        await self._send_notifications(summary_alert)
        logger.info(f"Sent correlation summary for group {group.group_id}")
    
    # Notification Handlers
    
    async def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification"""
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config.get('from_email', 'alerts@system.com')
            msg['To'] = config['to_email']
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
            
            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange'};">
                    {alert.name}
                </h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Time:</strong> {alert.timestamp.isoformat()}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                
                {f'<h3>Metadata:</h3><pre>{json.dumps(alert.metadata, indent=2)}</pre>' if alert.metadata else ''}
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(config.get('smtp_server', 'localhost'), config.get('smtp_port', 587))
            
            if config.get('username') and config.get('password'):
                server.starttls()
                server.login(config['username'], config['password'])
            
            text = msg.as_string()
            server.sendmail(msg['From'], msg['To'], text)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            raise
    
    async def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack notification"""
        
        color = 'danger' if alert.severity == AlertSeverity.CRITICAL else 'warning'
        
        payload = {
            'text': f"🚨 {alert.name}",
            'attachments': [{
                'color': color,
                'fields': [
                    {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                    {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True},
                    {'title': 'Message', 'value': alert.message, 'short': False}
                ],
                'footer': 'Mental Health LLM Alert System',
                'ts': int(alert.timestamp.timestamp())
            }]
        }
        
        if alert.metadata:
            payload['attachments'][0]['fields'].append({
                'title': 'Details',
                'value': f"```{json.dumps(alert.metadata, indent=2)}```",
                'short': False
            })
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config['webhook_url'], json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Slack API returned {response.status}")
    
    async def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification"""
        
        payload = {
            'alert_id': alert.alert_id,
            'name': alert.name,
            'severity': alert.severity.value,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'metadata': alert.metadata
        }
        
        headers = config.get('headers', {})
        headers.setdefault('Content-Type', 'application/json')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config['url'], json=payload, headers=headers) as response:
                if response.status not in [200, 201, 202]:
                    raise Exception(f"Webhook returned {response.status}")
    
    async def _send_sms_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send SMS notification (via Twilio or similar service)"""
        
        # This would integrate with SMS service like Twilio
        # For now, just log
        message = f"ALERT: {alert.name} - {alert.message}"
        logger.info(f"SMS Alert to {config.get('phone_number')}: {message}")
    
    async def _send_pagerduty_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send PagerDuty notification"""
        
        payload = {
            'routing_key': config['integration_key'],
            'event_action': 'trigger',
            'payload': {
                'summary': alert.name,
                'severity': alert.severity.value,
                'source': 'mental-health-llm-system',
                'timestamp': alert.timestamp.isoformat(),
                'custom_details': {
                    'message': alert.message,
                    'metadata': alert.metadata
                }
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post('https://events.pagerduty.com/v2/enqueue', json=payload) as response:
                if response.status != 202:
                    raise Exception(f"PagerDuty API returned {response.status}")
    
    async def _send_log_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send log notification"""
        
        level = getattr(logging, config.get('level', 'ERROR').upper())
        logger.log(level, f"ALERT: {alert.name} - {alert.message} (ID: {alert.alert_id})")
    
    async def _send_file_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send file notification"""
        
        file_path = Path(config.get('file_path', 'results/alerts.json'))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing alerts
        alerts = []
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    alerts = json.load(f)
            except Exception:
                pass
        
        # Add new alert
        alerts.append(alert.to_dict())
        
        # Keep only last N alerts
        max_alerts = config.get('max_alerts', 1000)
        alerts = alerts[-max_alerts:]
        
        # Save alerts
        with open(file_path, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    # API Methods
    
    def acknowledge_escalation(self, escalation_id: str) -> bool:
        """Acknowledge an escalation to stop further escalations"""
        
        for policy in self.escalation_policies.values():
            if escalation_id in policy.active_escalations:
                policy.active_escalations[escalation_id]['acknowledged'] = True
                logger.info(f"Acknowledged escalation: {escalation_id}")
                return True
        
        return False
    
    def get_active_escalations(self) -> List[Dict[str, Any]]:
        """Get all active escalations"""
        
        escalations = []
        
        for policy_name, policy in self.escalation_policies.items():
            for escalation_id, escalation in policy.active_escalations.items():
                escalations.append({
                    'escalation_id': escalation_id,
                    'policy': policy_name,
                    'alert_id': escalation['alert'].alert_id,
                    'current_level': escalation['current_level'],
                    'started_at': escalation['started_at'].isoformat(),
                    'acknowledged': escalation['acknowledged']
                })
        
        return escalations
    
    def get_correlation_groups(self) -> List[Dict[str, Any]]:
        """Get all active correlation groups"""
        
        groups = []
        
        for group_id, group in self.alert_groups.items():
            groups.append({
                'group_id': group_id,
                'pattern': group.pattern,
                'alert_count': len(group.alerts),
                'created_at': group.created_at.isoformat(),
                'last_updated': group.last_updated.isoformat(),
                'summary_sent': group.summary_sent
            })
        
        return groups


# Example usage and configuration
async def setup_production_alerting() -> AlertManager:
    """Setup production alerting configuration"""
    
    alert_manager = AlertManager()
    
    # Email notifications for critical alerts
    alert_manager.add_notification_channel(
        name="critical_email",
        channel=NotificationChannel.EMAIL,
        config={
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'alerts@company.com',
            'password': 'app_password',
            'to_email': 'oncall@company.com',
            'from_email': 'alerts@company.com'
        },
        min_severity=AlertSeverity.CRITICAL,
        rate_limit_minutes=10,
        max_alerts_per_period=5
    )
    
    # Slack notifications for all alerts
    alert_manager.add_notification_channel(
        name="slack_general",
        channel=NotificationChannel.SLACK,
        config={
            'webhook_url': 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX'
        },
        min_severity=AlertSeverity.WARNING,
        rate_limit_minutes=5,
        max_alerts_per_period=10
    )
    
    # File logging for audit trail
    alert_manager.add_notification_channel(
        name="audit_log",
        channel=NotificationChannel.FILE,
        config={
            'file_path': 'results/alerts/audit.json',
            'max_alerts': 10000
        },
        min_severity=AlertSeverity.INFO
    )
    
    # Escalation policy for critical system issues
    alert_manager.add_escalation_policy(
        name="critical_system",
        alert_patterns=[r".*critical.*|.*system.*down.*|.*database.*failed.*"],
        levels=[
            {
                'recipients': [
                    {'type': 'email', 'email': 'level1-oncall@company.com'},
                    {'type': 'slack', 'webhook_url': 'https://hooks.slack.com/...'}
                ]
            },
            {
                'recipients': [
                    {'type': 'email', 'email': 'team-lead@company.com'},
                    {'type': 'email', 'email': 'manager@company.com'}
                ]
            },
            {
                'recipients': [
                    {'type': 'email', 'email': 'director@company.com'}
                ]
            }
        ],
        initial_delay_minutes=5,
        escalation_delay_minutes=15,
        max_escalations=3
    )
    
    # Correlation rules
    alert_manager.add_correlation_rule(
        pattern=r"model.*selection.*latency",
        time_window_minutes=10,
        min_alerts=3,
        summary_template="Multiple model selection latency alerts in the past 10 minutes"
    )
    
    alert_manager.add_correlation_rule(
        pattern=r"health.*check.*failed",
        time_window_minutes=5,
        min_alerts=2,
        summary_template="Multiple health check failures detected"
    )
    
    await alert_manager.start()
    
    return alert_manager


if __name__ == "__main__":
    async def main():
        alert_manager = await setup_production_alerting()
        
        # Test alert
        test_alert = Alert(
            alert_id="test_001",
            name="Test Critical Alert",
            severity=AlertSeverity.CRITICAL,
            message="This is a test critical alert",
            metadata={'component': 'test_system'}
        )
        
        await alert_manager.process_alert(test_alert)
        
        # Keep running
        try:
            await asyncio.sleep(3600)  # Run for 1 hour
        except KeyboardInterrupt:
            pass
        finally:
            await alert_manager.stop()
    
    asyncio.run(main())