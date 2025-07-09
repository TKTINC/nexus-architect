"""
Human Oversight and Monitoring System for Autonomous Decisions
Implements real-time monitoring, intervention capabilities, and approval workflows
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterventionType(Enum):
    """Types of human intervention"""
    APPROVAL_REQUEST = "approval_request"
    EMERGENCY_STOP = "emergency_stop"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    MANUAL_OVERRIDE = "manual_override"
    ESCALATION = "escalation"

class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBSOCKET = "websocket"
    SMS = "sms"

class OversightLevel(Enum):
    """Levels of oversight required"""
    NONE = "none"
    MONITORING = "monitoring"
    APPROVAL = "approval"
    CONTINUOUS = "continuous"

@dataclass
class OversightRule:
    """Rule for determining oversight requirements"""
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    oversight_level: OversightLevel
    notification_channels: List[NotificationChannel]
    approvers: List[str]
    timeout_minutes: int
    escalation_rules: Dict[str, Any]

@dataclass
class InterventionRequest:
    """Request for human intervention"""
    request_id: str
    decision_id: str
    intervention_type: InterventionType
    urgency: str
    description: str
    context: Dict[str, Any]
    requested_by: str
    requested_at: datetime
    timeout_at: datetime
    status: str  # pending, approved, rejected, expired
    approver: Optional[str] = None
    approved_at: Optional[datetime] = None
    response: Optional[Dict[str, Any]] = None

@dataclass
class MonitoringAlert:
    """Monitoring alert for autonomous decisions"""
    alert_id: str
    decision_id: str
    alert_type: str
    severity: str
    message: str
    metrics: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

class NotificationManager:
    """Manages notifications across different channels"""
    
    def __init__(self):
        self.email_config = {}
        self.slack_config = {}
        self.teams_config = {}
        self.websocket_clients = set()
        
    def configure_email(self, smtp_server: str, smtp_port: int, 
                       username: str, password: str):
        """Configure email notifications"""
        self.email_config = {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "username": username,
            "password": password
        }
    
    def configure_slack(self, webhook_url: str, channel: str):
        """Configure Slack notifications"""
        self.slack_config = {
            "webhook_url": webhook_url,
            "channel": channel
        }
    
    def configure_teams(self, webhook_url: str):
        """Configure Microsoft Teams notifications"""
        self.teams_config = {
            "webhook_url": webhook_url
        }
    
    async def send_notification(self, channel: NotificationChannel, 
                              recipients: List[str], subject: str, 
                              message: str, data: Dict[str, Any] = None):
        """Send notification through specified channel"""
        try:
            if channel == NotificationChannel.EMAIL:
                await self._send_email(recipients, subject, message, data)
            elif channel == NotificationChannel.SLACK:
                await self._send_slack(recipients, subject, message, data)
            elif channel == NotificationChannel.TEAMS:
                await self._send_teams(recipients, subject, message, data)
            elif channel == NotificationChannel.WEBSOCKET:
                await self._send_websocket(recipients, subject, message, data)
            
            logger.info(f"Notification sent via {channel.value} to {recipients}")
            
        except Exception as e:
            logger.error(f"Failed to send notification via {channel.value}: {e}")
    
    async def _send_email(self, recipients: List[str], subject: str, 
                         message: str, data: Dict[str, Any]):
        """Send email notification"""
        # Simplified email implementation
        logger.info(f"Email notification: {subject} to {recipients}")
    
    async def _send_slack(self, recipients: List[str], subject: str, 
                         message: str, data: Dict[str, Any]):
        """Send Slack notification"""
        logger.info(f"Slack notification: {subject} - {message}")
    
    async def _send_teams(self, recipients: List[str], subject: str, 
                         message: str, data: Dict[str, Any]):
        """Send Microsoft Teams notification"""
        logger.info(f"Teams notification: {subject} - {message}")
    
    async def _send_websocket(self, recipients: List[str], subject: str, 
                             message: str, data: Dict[str, Any]):
        """Send WebSocket notification"""
        notification = {
            "type": "notification",
            "subject": subject,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"WebSocket notification: {json.dumps(notification)}")

class ApprovalManager:
    """Manages approval workflows and requests"""
    
    def __init__(self, notification_manager: NotificationManager):
        self.notification_manager = notification_manager
        self.pending_requests = {}
        self.approval_history = []
        
    async def request_approval(self, decision_id: str, intervention_type: InterventionType,
                             context: Dict[str, Any], approvers: List[str],
                             timeout_minutes: int = 60) -> InterventionRequest:
        """Request approval for a decision"""
        
        request_id = f"APPROVAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        request = InterventionRequest(
            request_id=request_id,
            decision_id=decision_id,
            intervention_type=intervention_type,
            urgency=context.get("urgency", "medium"),
            description=context.get("description", "Approval required for autonomous decision"),
            context=context,
            requested_by="autonomous_system",
            requested_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(minutes=timeout_minutes),
            status="pending"
        )
        
        # Store request
        self.pending_requests[request_id] = request
        
        # Send notifications to approvers
        await self._notify_approvers(request, approvers)
        
        # Schedule timeout handling
        asyncio.create_task(self._handle_timeout(request_id, timeout_minutes))
        
        return request
    
    async def _notify_approvers(self, request: InterventionRequest, approvers: List[str]):
        """Notify approvers about pending request"""
        subject = f"Approval Required: {request.intervention_type.value}"
        message = f"""
        Decision ID: {request.decision_id}
        Request ID: {request.request_id}
        Urgency: {request.urgency}
        Description: {request.description}
        
        Please review and approve/reject this request.
        Timeout: {request.timeout_at.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Send notifications via multiple channels
        await self.notification_manager.send_notification(
            NotificationChannel.EMAIL, approvers, subject, message, request.context
        )
        await self.notification_manager.send_notification(
            NotificationChannel.WEBSOCKET, approvers, subject, message, request.context
        )
    
    async def _handle_timeout(self, request_id: str, timeout_minutes: int):
        """Handle request timeout"""
        await asyncio.sleep(timeout_minutes * 60)
        
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
            if request.status == "pending":
                request.status = "expired"
                
                # Notify about timeout
                await self.notification_manager.send_notification(
                    NotificationChannel.EMAIL,
                    ["system_admin"],
                    f"Approval Request Expired: {request_id}",
                    f"Approval request for decision {request.decision_id} has expired"
                )
                
                logger.warning(f"Approval request {request_id} expired")
    
    def approve_request(self, request_id: str, approver: str, 
                       response: Dict[str, Any] = None) -> bool:
        """Approve a pending request"""
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        if request.status != "pending":
            return False
        
        request.status = "approved"
        request.approver = approver
        request.approved_at = datetime.now()
        request.response = response or {}
        
        # Move to history
        self.approval_history.append(request)
        del self.pending_requests[request_id]
        
        logger.info(f"Request {request_id} approved by {approver}")
        return True
    
    def reject_request(self, request_id: str, approver: str, 
                      reason: str = None) -> bool:
        """Reject a pending request"""
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        if request.status != "pending":
            return False
        
        request.status = "rejected"
        request.approver = approver
        request.approved_at = datetime.now()
        request.response = {"reason": reason} if reason else {}
        
        # Move to history
        self.approval_history.append(request)
        del self.pending_requests[request_id]
        
        logger.info(f"Request {request_id} rejected by {approver}")
        return True
    
    def get_pending_requests(self, approver: str = None) -> List[InterventionRequest]:
        """Get pending approval requests"""
        return list(self.pending_requests.values())
    
    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get approval statistics"""
        total_requests = len(self.approval_history)
        if total_requests == 0:
            return {}
        
        approved = len([r for r in self.approval_history if r.status == "approved"])
        rejected = len([r for r in self.approval_history if r.status == "rejected"])
        expired = len([r for r in self.approval_history if r.status == "expired"])
        
        return {
            "total_requests": total_requests,
            "approval_rate": approved / total_requests,
            "rejection_rate": rejected / total_requests,
            "expiration_rate": expired / total_requests,
            "pending_requests": len(self.pending_requests)
        }

class MonitoringSystem:
    """Real-time monitoring system for autonomous decisions"""
    
    def __init__(self, notification_manager: NotificationManager):
        self.notification_manager = notification_manager
        self.active_monitors = {}
        self.alerts = []
        self.metrics_history = []
        
    def start_monitoring(self, decision_id: str, metrics_to_monitor: List[str],
                        thresholds: Dict[str, Any], duration_minutes: int = 60):
        """Start monitoring a decision"""
        monitor_config = {
            "decision_id": decision_id,
            "metrics": metrics_to_monitor,
            "thresholds": thresholds,
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(minutes=duration_minutes),
            "active": True
        }
        
        self.active_monitors[decision_id] = monitor_config
        
        # Start monitoring task
        asyncio.create_task(self._monitor_decision(decision_id))
        
        logger.info(f"Started monitoring decision {decision_id}")
    
    async def _monitor_decision(self, decision_id: str):
        """Monitor a specific decision"""
        monitor = self.active_monitors.get(decision_id)
        if not monitor:
            return
        
        while monitor["active"] and datetime.now() < monitor["end_time"]:
            try:
                # Collect metrics
                metrics = await self._collect_metrics(decision_id, monitor["metrics"])
                
                # Store metrics
                self.metrics_history.append({
                    "decision_id": decision_id,
                    "timestamp": datetime.now(),
                    "metrics": metrics
                })
                
                # Check thresholds
                alerts = self._check_thresholds(decision_id, metrics, monitor["thresholds"])
                
                # Process alerts
                for alert in alerts:
                    await self._process_alert(alert)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring decision {decision_id}: {e}")
                await asyncio.sleep(60)  # Wait longer on error
        
        # Mark monitoring as complete
        if decision_id in self.active_monitors:
            self.active_monitors[decision_id]["active"] = False
            logger.info(f"Completed monitoring decision {decision_id}")
    
    async def _collect_metrics(self, decision_id: str, metrics_to_collect: List[str]) -> Dict[str, float]:
        """Collect metrics for a decision"""
        metrics = {}
        
        # Simulate metric collection
        for metric in metrics_to_collect:
            if metric == "error_rate":
                metrics[metric] = 0.02  # 2% error rate
            elif metric == "response_time":
                metrics[metric] = 150.0  # 150ms response time
            elif metric == "cpu_usage":
                metrics[metric] = 45.0  # 45% CPU usage
            elif metric == "memory_usage":
                metrics[metric] = 60.0  # 60% memory usage
            elif metric == "throughput":
                metrics[metric] = 1000.0  # 1000 requests/minute
            else:
                metrics[metric] = 0.0
        
        return metrics
    
    def _check_thresholds(self, decision_id: str, metrics: Dict[str, float],
                         thresholds: Dict[str, Any]) -> List[MonitoringAlert]:
        """Check if metrics exceed thresholds"""
        alerts = []
        
        for metric, value in metrics.items():
            if metric in thresholds:
                threshold_config = thresholds[metric]
                
                # Check different threshold types
                if "max" in threshold_config and value > threshold_config["max"]:
                    alert = MonitoringAlert(
                        alert_id=f"ALERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        decision_id=decision_id,
                        alert_type="threshold_exceeded",
                        severity=threshold_config.get("severity", "medium"),
                        message=f"{metric} exceeded maximum threshold: {value} > {threshold_config['max']}",
                        metrics=metrics,
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
                
                if "min" in threshold_config and value < threshold_config["min"]:
                    alert = MonitoringAlert(
                        alert_id=f"ALERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        decision_id=decision_id,
                        alert_type="threshold_below",
                        severity=threshold_config.get("severity", "medium"),
                        message=f"{metric} below minimum threshold: {value} < {threshold_config['min']}",
                        metrics=metrics,
                        timestamp=datetime.now()
                    )
                    alerts.append(alert)
        
        return alerts
    
    async def _process_alert(self, alert: MonitoringAlert):
        """Process a monitoring alert"""
        self.alerts.append(alert)
        
        # Send notifications based on severity
        if alert.severity in ["high", "critical"]:
            await self.notification_manager.send_notification(
                NotificationChannel.EMAIL,
                ["system_admin", "on_call_engineer"],
                f"Critical Alert: {alert.alert_type}",
                alert.message,
                alert.metrics
            )
        
        # Send real-time notification
        await self.notification_manager.send_notification(
            NotificationChannel.WEBSOCKET,
            ["monitoring_dashboard"],
            f"Alert: {alert.alert_type}",
            alert.message,
            asdict(alert)
        )
        
        logger.warning(f"Alert generated: {alert.alert_id} - {alert.message}")
    
    def stop_monitoring(self, decision_id: str):
        """Stop monitoring a decision"""
        if decision_id in self.active_monitors:
            self.active_monitors[decision_id]["active"] = False
            logger.info(f"Stopped monitoring decision {decision_id}")
    
    def get_monitoring_status(self, decision_id: str = None) -> Dict[str, Any]:
        """Get monitoring status"""
        if decision_id:
            return self.active_monitors.get(decision_id, {})
        else:
            return {
                "active_monitors": len([m for m in self.active_monitors.values() if m["active"]]),
                "total_monitors": len(self.active_monitors),
                "total_alerts": len(self.alerts),
                "recent_alerts": len([a for a in self.alerts if 
                                    (datetime.now() - a.timestamp).total_seconds() < 3600])
            }

class OversightManager:
    """Main oversight manager coordinating all oversight activities"""
    
    def __init__(self):
        self.notification_manager = NotificationManager()
        self.approval_manager = ApprovalManager(self.notification_manager)
        self.monitoring_system = MonitoringSystem(self.notification_manager)
        self.oversight_rules = []
        self.intervention_callbacks = {}
        
    def add_oversight_rule(self, rule: OversightRule):
        """Add an oversight rule"""
        self.oversight_rules.append(rule)
        logger.info(f"Added oversight rule: {rule.name}")
    
    def determine_oversight_requirements(self, decision_context: Dict[str, Any]) -> OversightLevel:
        """Determine required oversight level for a decision"""
        max_oversight = OversightLevel.NONE
        
        for rule in self.oversight_rules:
            if self._rule_matches(rule, decision_context):
                if rule.oversight_level.value > max_oversight.value:
                    max_oversight = rule.oversight_level
        
        return max_oversight
    
    def _rule_matches(self, rule: OversightRule, context: Dict[str, Any]) -> bool:
        """Check if a rule matches the decision context"""
        for condition_key, condition_value in rule.conditions.items():
            context_value = context.get(condition_key)
            
            if isinstance(condition_value, dict):
                # Handle complex conditions
                if "min" in condition_value and context_value < condition_value["min"]:
                    return False
                if "max" in condition_value and context_value > condition_value["max"]:
                    return False
                if "equals" in condition_value and context_value != condition_value["equals"]:
                    return False
                if "in" in condition_value and context_value not in condition_value["in"]:
                    return False
            else:
                # Simple equality check
                if context_value != condition_value:
                    return False
        
        return True
    
    async def process_decision(self, decision_id: str, decision_context: Dict[str, Any],
                             decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a decision through the oversight system"""
        
        # Determine oversight requirements
        oversight_level = self.determine_oversight_requirements(decision_context)
        
        result = {
            "decision_id": decision_id,
            "oversight_level": oversight_level.value,
            "approved": False,
            "monitoring_started": False,
            "interventions": []
        }
        
        if oversight_level == OversightLevel.NONE:
            result["approved"] = True
            
        elif oversight_level == OversightLevel.MONITORING:
            # Start monitoring without requiring approval
            await self._start_decision_monitoring(decision_id, decision_context)
            result["approved"] = True
            result["monitoring_started"] = True
            
        elif oversight_level == OversightLevel.APPROVAL:
            # Request approval
            approval_request = await self.approval_manager.request_approval(
                decision_id, InterventionType.APPROVAL_REQUEST,
                decision_context, ["senior_engineer"], 30
            )
            result["interventions"].append(approval_request.request_id)
            
        elif oversight_level == OversightLevel.CONTINUOUS:
            # Request approval and start continuous monitoring
            approval_request = await self.approval_manager.request_approval(
                decision_id, InterventionType.APPROVAL_REQUEST,
                decision_context, ["senior_engineer", "security_team"], 15
            )
            await self._start_decision_monitoring(decision_id, decision_context)
            result["interventions"].append(approval_request.request_id)
            result["monitoring_started"] = True
        
        return result
    
    async def _start_decision_monitoring(self, decision_id: str, context: Dict[str, Any]):
        """Start monitoring for a decision"""
        # Define metrics to monitor based on decision type
        metrics = ["error_rate", "response_time", "cpu_usage", "memory_usage"]
        
        # Define thresholds
        thresholds = {
            "error_rate": {"max": 0.05, "severity": "high"},
            "response_time": {"max": 500, "severity": "medium"},
            "cpu_usage": {"max": 80, "severity": "medium"},
            "memory_usage": {"max": 85, "severity": "medium"}
        }
        
        # Start monitoring for 2 hours
        self.monitoring_system.start_monitoring(decision_id, metrics, thresholds, 120)
    
    def register_intervention_callback(self, intervention_type: InterventionType, 
                                     callback: Callable):
        """Register callback for intervention handling"""
        self.intervention_callbacks[intervention_type] = callback
    
    async def emergency_stop(self, decision_id: str, reason: str, operator: str):
        """Emergency stop for a decision"""
        logger.critical(f"Emergency stop requested for decision {decision_id} by {operator}: {reason}")
        
        # Stop monitoring
        self.monitoring_system.stop_monitoring(decision_id)
        
        # Send emergency notifications
        await self.notification_manager.send_notification(
            NotificationChannel.EMAIL,
            ["system_admin", "engineering_director"],
            f"EMERGENCY STOP: Decision {decision_id}",
            f"Emergency stop requested by {operator}. Reason: {reason}"
        )
        
        # Execute emergency stop callback if registered
        if InterventionType.EMERGENCY_STOP in self.intervention_callbacks:
            await self.intervention_callbacks[InterventionType.EMERGENCY_STOP](
                decision_id, reason, operator
            )
    
    def get_oversight_statistics(self) -> Dict[str, Any]:
        """Get comprehensive oversight statistics"""
        approval_stats = self.approval_manager.get_approval_statistics()
        monitoring_stats = self.monitoring_system.get_monitoring_status()
        
        return {
            "approval_statistics": approval_stats,
            "monitoring_statistics": monitoring_stats,
            "oversight_rules": len(self.oversight_rules),
            "active_interventions": len(self.approval_manager.pending_requests),
            "total_alerts": len(self.monitoring_system.alerts),
            "last_updated": datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize oversight manager
    oversight_manager = OversightManager()
    
    # Add oversight rules
    high_risk_rule = OversightRule(
        rule_id="high_risk",
        name="High Risk Decision Rule",
        description="Requires approval for high-risk decisions",
        conditions={
            "risk_score": {"min": 0.7},
            "impact": {"in": ["major", "severe"]}
        },
        oversight_level=OversightLevel.APPROVAL,
        notification_channels=[NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET],
        approvers=["senior_engineer", "security_team"],
        timeout_minutes=30,
        escalation_rules={"timeout_action": "escalate_to_director"}
    )
    
    oversight_manager.add_oversight_rule(high_risk_rule)
    
    # Example decision processing
    async def test_oversight():
        decision_context = {
            "decision_id": "TEST-001",
            "risk_score": 0.8,
            "impact": "major",
            "urgency": "high",
            "decision_type": "infrastructure_change"
        }
        
        decision_result = {
            "selected_alternative": "comprehensive_fix",
            "confidence": 0.85
        }
        
        # Process decision through oversight
        result = await oversight_manager.process_decision(
            "TEST-001", decision_context, decision_result
        )
        
        print(f"Oversight Result: {json.dumps(result, indent=2)}")
        
        # Get statistics
        stats = oversight_manager.get_oversight_statistics()
        print(f"Oversight Statistics: {json.dumps(stats, indent=2)}")
    
    # Run test
    asyncio.run(test_oversight())

