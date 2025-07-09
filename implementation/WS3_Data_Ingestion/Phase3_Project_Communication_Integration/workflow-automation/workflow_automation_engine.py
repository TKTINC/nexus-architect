"""
Workflow Automation and Analytics Engine for Nexus Architect
Intelligent automation and analytics for project management and communication workflows
"""

import os
import re
import json
import logging
import time
import asyncio
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import uuid
from collections import defaultdict, deque
from prometheus_client import Counter, Histogram, Gauge
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
WORKFLOW_EXECUTIONS = Counter('workflow_executions_total', 'Total workflow executions', ['workflow_type', 'status'])
WORKFLOW_LATENCY = Histogram('workflow_latency_seconds', 'Workflow execution latency', ['workflow_type'])
ACTIVE_WORKFLOWS = Gauge('active_workflows', 'Number of active workflows')
AUTOMATION_ACCURACY = Gauge('automation_accuracy', 'Automation accuracy score')

class WorkflowTriggerType(Enum):
    """Types of workflow triggers"""
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    MANUAL = "manual"

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ActionType(Enum):
    """Types of workflow actions"""
    CREATE_ISSUE = "create_issue"
    UPDATE_ISSUE = "update_issue"
    SEND_MESSAGE = "send_message"
    ASSIGN_TASK = "assign_task"
    CREATE_MEETING = "create_meeting"
    SEND_NOTIFICATION = "send_notification"
    UPDATE_STATUS = "update_status"
    GENERATE_REPORT = "generate_report"
    ESCALATE = "escalate"
    CUSTOM = "custom"

class AnalyticsType(Enum):
    """Types of analytics"""
    PRODUCTIVITY = "productivity"
    COLLABORATION = "collaboration"
    PROJECT_HEALTH = "project_health"
    TEAM_DYNAMICS = "team_dynamics"
    WORKFLOW_EFFICIENCY = "workflow_efficiency"
    PREDICTIVE = "predictive"

@dataclass
class WorkflowTrigger:
    """Workflow trigger definition"""
    trigger_id: str
    trigger_type: WorkflowTriggerType
    conditions: Dict[str, Any]
    schedule: Optional[str] = None  # Cron expression for time-based triggers
    event_filters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

@dataclass
class WorkflowAction:
    """Workflow action definition"""
    action_id: str
    action_type: ActionType
    parameters: Dict[str, Any]
    conditions: Dict[str, Any] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    triggers: List[WorkflowTrigger]
    actions: List[WorkflowAction]
    created_at: datetime
    updated_at: datetime
    created_by: str
    is_active: bool = True
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    triggered_by: Optional[str] = None
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class AnalyticsMetric:
    """Analytics metric definition"""
    metric_id: str
    name: str
    analytics_type: AnalyticsType
    value: float
    unit: str
    timestamp: datetime
    dimensions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectInsight:
    """Project insight from analytics"""
    insight_id: str
    project_id: str
    insight_type: str
    title: str
    description: str
    confidence: float
    impact_level: str  # low, medium, high, critical
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    generated_at: datetime

class EventProcessor:
    """Process events from various sources"""
    
    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue = deque()
        self.processing = False
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
    
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event"""
        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'data': event_data,
            'timestamp': datetime.now(timezone.utc)
        }
        
        self.event_queue.append(event)
        
        if not self.processing:
            await self._process_events()
    
    async def _process_events(self):
        """Process events in the queue"""
        self.processing = True
        
        try:
            while self.event_queue:
                event = self.event_queue.popleft()
                await self._handle_event(event)
        finally:
            self.processing = False
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle a single event"""
        event_type = event['event_type']
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Event handler failed for {event_type}: {e}")

class ConditionEvaluator:
    """Evaluate workflow conditions"""
    
    def __init__(self):
        self.operators = {
            'eq': lambda a, b: a == b,
            'ne': lambda a, b: a != b,
            'gt': lambda a, b: a > b,
            'gte': lambda a, b: a >= b,
            'lt': lambda a, b: a < b,
            'lte': lambda a, b: a <= b,
            'in': lambda a, b: a in b,
            'not_in': lambda a, b: a not in b,
            'contains': lambda a, b: b in a,
            'not_contains': lambda a, b: b not in a,
            'regex': lambda a, b: bool(re.search(b, str(a))),
            'exists': lambda a, b: a is not None,
            'not_exists': lambda a, b: a is None
        }
    
    def evaluate(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate conditions against context"""
        if not conditions:
            return True
        
        try:
            return self._evaluate_condition_group(conditions, context)
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    def _evaluate_condition_group(self, conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a group of conditions"""
        if 'and' in conditions:
            return all(self._evaluate_condition_group(cond, context) for cond in conditions['and'])
        
        if 'or' in conditions:
            return any(self._evaluate_condition_group(cond, context) for cond in conditions['or'])
        
        if 'not' in conditions:
            return not self._evaluate_condition_group(conditions['not'], context)
        
        # Single condition
        field = conditions.get('field')
        operator = conditions.get('operator', 'eq')
        value = conditions.get('value')
        
        if not field:
            return True
        
        context_value = self._get_nested_value(context, field)
        
        if operator not in self.operators:
            logger.warning(f"Unknown operator: {operator}")
            return False
        
        return self.operators[operator](context_value, value)
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current

class WorkflowEngine:
    """Core workflow execution engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.event_processor = EventProcessor()
        self.condition_evaluator = ConditionEvaluator()
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.running = False
        
        # Register default action handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default action handlers"""
        self.action_handlers[ActionType.SEND_NOTIFICATION] = self._handle_send_notification
        self.action_handlers[ActionType.UPDATE_STATUS] = self._handle_update_status
        self.action_handlers[ActionType.GENERATE_REPORT] = self._handle_generate_report
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow definition"""
        self.workflows[workflow.workflow_id] = workflow
        
        # Register event handlers for event-based triggers
        for trigger in workflow.triggers:
            if trigger.trigger_type == WorkflowTriggerType.EVENT_BASED:
                event_type = trigger.conditions.get('event_type')
                if event_type:
                    self.event_processor.register_handler(
                        event_type,
                        lambda event, wf_id=workflow.workflow_id, t=trigger: 
                        asyncio.create_task(self._handle_event_trigger(event, wf_id, t))
                    )
        
        logger.info(f"Registered workflow: {workflow.name}")
    
    def register_action_handler(self, action_type: ActionType, handler: Callable):
        """Register custom action handler"""
        self.action_handlers[action_type] = handler
        logger.info(f"Registered action handler for: {action_type.value}")
    
    async def start(self):
        """Start the workflow engine"""
        self.running = True
        logger.info("Workflow engine started")
        
        # Start time-based trigger scheduler
        asyncio.create_task(self._schedule_time_based_triggers())
    
    async def stop(self):
        """Stop the workflow engine"""
        self.running = False
        logger.info("Workflow engine stopped")
    
    async def execute_workflow(self, workflow_id: str, trigger_data: Dict[str, Any] = None, 
                             triggered_by: str = None) -> str:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        if not workflow.is_active:
            raise ValueError(f"Workflow is not active: {workflow_id}")
        
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            started_at=datetime.now(timezone.utc),
            triggered_by=triggered_by,
            trigger_data=trigger_data or {}
        )
        
        self.executions[execution_id] = execution
        ACTIVE_WORKFLOWS.inc()
        
        # Execute workflow asynchronously
        asyncio.create_task(self._execute_workflow_async(execution))
        
        return execution_id
    
    async def _execute_workflow_async(self, execution: WorkflowExecution):
        """Execute workflow asynchronously"""
        start_time = time.time()
        workflow = self.workflows[execution.workflow_id]
        
        try:
            execution.status = WorkflowStatus.RUNNING
            self._log_execution(execution, "Workflow execution started")
            
            # Execute actions in sequence
            for action in workflow.actions:
                await self._execute_action(execution, action)
                
                if execution.status == WorkflowStatus.FAILED:
                    break
            
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.now(timezone.utc)
                self._log_execution(execution, "Workflow execution completed successfully")
            
            WORKFLOW_EXECUTIONS.labels(
                workflow_type=workflow.name,
                status=execution.status.value
            ).inc()
        
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now(timezone.utc)
            self._log_execution(execution, f"Workflow execution failed: {e}")
            
            WORKFLOW_EXECUTIONS.labels(
                workflow_type=workflow.name,
                status='failed'
            ).inc()
        
        finally:
            ACTIVE_WORKFLOWS.dec()
            latency = time.time() - start_time
            WORKFLOW_LATENCY.labels(workflow_type=workflow.name).observe(latency)
    
    async def _execute_action(self, execution: WorkflowExecution, action: WorkflowAction):
        """Execute a single action"""
        try:
            # Evaluate action conditions
            context = {
                'execution': execution.__dict__,
                'trigger_data': execution.trigger_data,
                'results': execution.results
            }
            
            if not self.condition_evaluator.evaluate(action.conditions, context):
                self._log_execution(execution, f"Action {action.action_id} skipped due to conditions")
                return
            
            self._log_execution(execution, f"Executing action: {action.action_id}")
            
            # Get action handler
            handler = self.action_handlers.get(action.action_type)
            if not handler:
                raise ValueError(f"No handler for action type: {action.action_type}")
            
            # Execute action with timeout
            result = await asyncio.wait_for(
                handler(execution, action),
                timeout=action.timeout_seconds
            )
            
            # Store result
            execution.results[action.action_id] = result
            self._log_execution(execution, f"Action {action.action_id} completed successfully")
        
        except asyncio.TimeoutError:
            error_msg = f"Action {action.action_id} timed out after {action.timeout_seconds} seconds"
            self._log_execution(execution, error_msg)
            execution.status = WorkflowStatus.FAILED
            execution.error_message = error_msg
        
        except Exception as e:
            error_msg = f"Action {action.action_id} failed: {e}"
            self._log_execution(execution, error_msg)
            execution.status = WorkflowStatus.FAILED
            execution.error_message = error_msg
    
    def _log_execution(self, execution: WorkflowExecution, message: str):
        """Log execution event"""
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': message
        }
        execution.execution_log.append(log_entry)
        logger.info(f"Workflow {execution.workflow_id} [{execution.execution_id}]: {message}")
    
    async def _handle_event_trigger(self, event: Dict[str, Any], workflow_id: str, trigger: WorkflowTrigger):
        """Handle event-based trigger"""
        try:
            # Evaluate trigger conditions
            if self.condition_evaluator.evaluate(trigger.conditions, event['data']):
                await self.execute_workflow(
                    workflow_id,
                    trigger_data=event['data'],
                    triggered_by=f"event:{event['event_type']}"
                )
        except Exception as e:
            logger.error(f"Event trigger handling failed: {e}")
    
    async def _schedule_time_based_triggers(self):
        """Schedule time-based triggers"""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                
                for workflow in self.workflows.values():
                    if not workflow.is_active:
                        continue
                    
                    for trigger in workflow.triggers:
                        if trigger.trigger_type == WorkflowTriggerType.TIME_BASED and trigger.is_active:
                            # Simple time-based scheduling (would use proper cron library in production)
                            if self._should_trigger_now(trigger, current_time):
                                await self.execute_workflow(
                                    workflow.workflow_id,
                                    triggered_by="scheduler"
                                )
                
                await asyncio.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    def _should_trigger_now(self, trigger: WorkflowTrigger, current_time: datetime) -> bool:
        """Check if time-based trigger should fire now"""
        # Simplified implementation - would use proper cron parsing in production
        schedule = trigger.schedule
        if not schedule:
            return False
        
        # Example: "0 9 * * 1-5" (9 AM on weekdays)
        # This is a simplified check - real implementation would use croniter or similar
        if schedule == "hourly":
            return current_time.minute == 0
        elif schedule == "daily":
            return current_time.hour == 9 and current_time.minute == 0
        
        return False
    
    # Default action handlers
    async def _handle_send_notification(self, execution: WorkflowExecution, action: WorkflowAction) -> Dict[str, Any]:
        """Handle send notification action"""
        message = action.parameters.get('message', 'Workflow notification')
        recipients = action.parameters.get('recipients', [])
        
        # Simulate sending notification
        logger.info(f"Sending notification to {recipients}: {message}")
        
        return {
            'status': 'sent',
            'recipients': recipients,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _handle_update_status(self, execution: WorkflowExecution, action: WorkflowAction) -> Dict[str, Any]:
        """Handle update status action"""
        entity_type = action.parameters.get('entity_type')
        entity_id = action.parameters.get('entity_id')
        new_status = action.parameters.get('status')
        
        # Simulate status update
        logger.info(f"Updating {entity_type} {entity_id} status to {new_status}")
        
        return {
            'status': 'updated',
            'entity_type': entity_type,
            'entity_id': entity_id,
            'new_status': new_status,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _handle_generate_report(self, execution: WorkflowExecution, action: WorkflowAction) -> Dict[str, Any]:
        """Handle generate report action"""
        report_type = action.parameters.get('report_type')
        parameters = action.parameters.get('parameters', {})
        
        # Simulate report generation
        logger.info(f"Generating {report_type} report with parameters: {parameters}")
        
        return {
            'status': 'generated',
            'report_type': report_type,
            'report_id': str(uuid.uuid4()),
            'parameters': parameters,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

class AnalyticsEngine:
    """Advanced analytics engine for project and team insights"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_store: Dict[str, List[AnalyticsMetric]] = defaultdict(list)
        self.insights_store: Dict[str, List[ProjectInsight]] = defaultdict(list)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    async def analyze_project_health(self, project_data: Dict[str, Any]) -> List[ProjectInsight]:
        """Analyze project health and generate insights"""
        insights = []
        project_id = project_data.get('project_id')
        
        if not project_id:
            return insights
        
        try:
            # Analyze issue velocity
            velocity_insight = await self._analyze_issue_velocity(project_data)
            if velocity_insight:
                insights.append(velocity_insight)
            
            # Analyze team collaboration
            collaboration_insight = await self._analyze_team_collaboration(project_data)
            if collaboration_insight:
                insights.append(collaboration_insight)
            
            # Analyze project timeline
            timeline_insight = await self._analyze_project_timeline(project_data)
            if timeline_insight:
                insights.append(timeline_insight)
            
            # Analyze communication patterns
            communication_insight = await self._analyze_communication_patterns(project_data)
            if communication_insight:
                insights.append(communication_insight)
            
            # Store insights
            self.insights_store[project_id].extend(insights)
            
        except Exception as e:
            logger.error(f"Project health analysis failed: {e}")
        
        return insights
    
    async def _analyze_issue_velocity(self, project_data: Dict[str, Any]) -> Optional[ProjectInsight]:
        """Analyze issue completion velocity"""
        issues = project_data.get('issues', [])
        if not issues:
            return None
        
        # Calculate velocity metrics
        completed_issues = [issue for issue in issues if issue.get('status') == 'done']
        total_issues = len(issues)
        completion_rate = len(completed_issues) / total_issues if total_issues > 0 else 0
        
        # Analyze velocity trend
        velocity_trend = self._calculate_velocity_trend(completed_issues)
        
        # Generate insight
        if completion_rate < 0.3:
            impact_level = "high"
            title = "Low Issue Completion Rate Detected"
            description = f"Project has a completion rate of {completion_rate:.1%}, which is below the recommended threshold."
            recommendations = [
                "Review issue complexity and break down large tasks",
                "Identify and address team blockers",
                "Consider adjusting sprint capacity",
                "Improve task estimation accuracy"
            ]
        elif velocity_trend < -0.2:
            impact_level = "medium"
            title = "Declining Velocity Trend"
            description = "Issue completion velocity has been declining over recent periods."
            recommendations = [
                "Investigate causes of velocity decline",
                "Review team capacity and workload",
                "Address technical debt or process issues"
            ]
        else:
            return None  # No concerning patterns
        
        return ProjectInsight(
            insight_id=str(uuid.uuid4()),
            project_id=project_data['project_id'],
            insight_type="velocity",
            title=title,
            description=description,
            confidence=0.85,
            impact_level=impact_level,
            recommendations=recommendations,
            supporting_data={
                'completion_rate': completion_rate,
                'velocity_trend': velocity_trend,
                'total_issues': total_issues,
                'completed_issues': len(completed_issues)
            },
            generated_at=datetime.now(timezone.utc)
        )
    
    async def _analyze_team_collaboration(self, project_data: Dict[str, Any]) -> Optional[ProjectInsight]:
        """Analyze team collaboration patterns"""
        messages = project_data.get('messages', [])
        issues = project_data.get('issues', [])
        
        if not messages and not issues:
            return None
        
        # Analyze communication frequency
        user_activity = defaultdict(int)
        for message in messages:
            if message.get('author'):
                user_activity[message['author']] += 1
        
        # Analyze collaboration network
        collaboration_graph = nx.Graph()
        for issue in issues:
            assignee = issue.get('assignee')
            reporter = issue.get('reporter')
            if assignee and reporter and assignee != reporter:
                collaboration_graph.add_edge(assignee, reporter)
        
        # Calculate collaboration metrics
        if collaboration_graph.number_of_nodes() > 0:
            density = nx.density(collaboration_graph)
            centrality = nx.degree_centrality(collaboration_graph)
            avg_centrality = sum(centrality.values()) / len(centrality) if centrality else 0
        else:
            density = 0
            avg_centrality = 0
        
        # Generate insight
        if density < 0.3:
            impact_level = "medium"
            title = "Low Team Collaboration Detected"
            description = f"Team collaboration density is {density:.2f}, indicating limited cross-team interaction."
            recommendations = [
                "Encourage cross-functional collaboration",
                "Implement pair programming or code reviews",
                "Organize team building activities",
                "Create shared project goals"
            ]
        elif len(user_activity) > 0 and max(user_activity.values()) / sum(user_activity.values()) > 0.5:
            impact_level = "medium"
            title = "Communication Concentration Risk"
            description = "Communication is heavily concentrated among few team members."
            recommendations = [
                "Encourage broader team participation",
                "Rotate meeting facilitation",
                "Create inclusive communication channels",
                "Implement knowledge sharing sessions"
            ]
        else:
            return None
        
        return ProjectInsight(
            insight_id=str(uuid.uuid4()),
            project_id=project_data['project_id'],
            insight_type="collaboration",
            title=title,
            description=description,
            confidence=0.75,
            impact_level=impact_level,
            recommendations=recommendations,
            supporting_data={
                'collaboration_density': density,
                'avg_centrality': avg_centrality,
                'active_users': len(user_activity),
                'total_messages': len(messages)
            },
            generated_at=datetime.now(timezone.utc)
        )
    
    async def _analyze_project_timeline(self, project_data: Dict[str, Any]) -> Optional[ProjectInsight]:
        """Analyze project timeline and deadlines"""
        issues = project_data.get('issues', [])
        project_info = project_data.get('project_info', {})
        
        if not issues:
            return None
        
        # Analyze deadline adherence
        overdue_issues = []
        upcoming_deadlines = []
        current_time = datetime.now(timezone.utc)
        
        for issue in issues:
            due_date = issue.get('due_date')
            if due_date:
                due_datetime = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
                if due_datetime < current_time and issue.get('status') != 'done':
                    overdue_issues.append(issue)
                elif due_datetime < current_time + timedelta(days=7):
                    upcoming_deadlines.append(issue)
        
        # Calculate timeline metrics
        total_issues_with_deadlines = len([i for i in issues if i.get('due_date')])
        overdue_rate = len(overdue_issues) / total_issues_with_deadlines if total_issues_with_deadlines > 0 else 0
        
        # Generate insight
        if overdue_rate > 0.2:
            impact_level = "high"
            title = "High Overdue Issue Rate"
            description = f"{overdue_rate:.1%} of issues with deadlines are overdue."
            recommendations = [
                "Review and adjust project timeline",
                "Prioritize overdue issues",
                "Improve estimation accuracy",
                "Address resource constraints"
            ]
        elif len(upcoming_deadlines) > 5:
            impact_level = "medium"
            title = "Multiple Upcoming Deadlines"
            description = f"{len(upcoming_deadlines)} issues have deadlines in the next week."
            recommendations = [
                "Review upcoming deadline priorities",
                "Ensure adequate resource allocation",
                "Consider deadline adjustments if needed",
                "Increase team communication about priorities"
            ]
        else:
            return None
        
        return ProjectInsight(
            insight_id=str(uuid.uuid4()),
            project_id=project_data['project_id'],
            insight_type="timeline",
            title=title,
            description=description,
            confidence=0.90,
            impact_level=impact_level,
            recommendations=recommendations,
            supporting_data={
                'overdue_rate': overdue_rate,
                'overdue_count': len(overdue_issues),
                'upcoming_deadlines': len(upcoming_deadlines),
                'total_with_deadlines': total_issues_with_deadlines
            },
            generated_at=datetime.now(timezone.utc)
        )
    
    async def _analyze_communication_patterns(self, project_data: Dict[str, Any]) -> Optional[ProjectInsight]:
        """Analyze communication patterns and sentiment"""
        messages = project_data.get('messages', [])
        
        if not messages:
            return None
        
        # Analyze sentiment trends
        sentiments = [msg.get('sentiment_score', 0) for msg in messages if msg.get('sentiment_score') is not None]
        if not sentiments:
            return None
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        recent_sentiments = sentiments[-10:] if len(sentiments) > 10 else sentiments
        recent_avg_sentiment = sum(recent_sentiments) / len(recent_sentiments)
        
        # Analyze communication frequency
        message_dates = []
        for msg in messages:
            if msg.get('timestamp'):
                try:
                    msg_date = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                    message_dates.append(msg_date)
                except:
                    continue
        
        # Calculate communication frequency trend
        if len(message_dates) > 1:
            message_dates.sort()
            recent_messages = [d for d in message_dates if d > datetime.now(timezone.utc) - timedelta(days=7)]
            communication_frequency = len(recent_messages) / 7  # messages per day
        else:
            communication_frequency = 0
        
        # Generate insight
        if recent_avg_sentiment < -0.2:
            impact_level = "high"
            title = "Negative Communication Sentiment Detected"
            description = f"Recent team communication shows negative sentiment (score: {recent_avg_sentiment:.2f})."
            recommendations = [
                "Address team concerns and blockers",
                "Improve team morale and motivation",
                "Facilitate open communication sessions",
                "Review project challenges and support needs"
            ]
        elif communication_frequency < 1:
            impact_level = "medium"
            title = "Low Communication Frequency"
            description = f"Team communication frequency is low ({communication_frequency:.1f} messages/day)."
            recommendations = [
                "Encourage regular team check-ins",
                "Implement daily standups or status updates",
                "Create more collaborative work sessions",
                "Improve communication tools and processes"
            ]
        else:
            return None
        
        return ProjectInsight(
            insight_id=str(uuid.uuid4()),
            project_id=project_data['project_id'],
            insight_type="communication",
            title=title,
            description=description,
            confidence=0.80,
            impact_level=impact_level,
            recommendations=recommendations,
            supporting_data={
                'avg_sentiment': avg_sentiment,
                'recent_avg_sentiment': recent_avg_sentiment,
                'communication_frequency': communication_frequency,
                'total_messages': len(messages)
            },
            generated_at=datetime.now(timezone.utc)
        )
    
    def _calculate_velocity_trend(self, completed_issues: List[Dict[str, Any]]) -> float:
        """Calculate velocity trend over time"""
        if len(completed_issues) < 2:
            return 0
        
        # Group issues by completion week
        weekly_counts = defaultdict(int)
        for issue in completed_issues:
            completed_date = issue.get('completed_at') or issue.get('updated_at')
            if completed_date:
                try:
                    date = datetime.fromisoformat(completed_date.replace('Z', '+00:00'))
                    week = date.isocalendar()[1]  # Week number
                    weekly_counts[week] += 1
                except:
                    continue
        
        if len(weekly_counts) < 2:
            return 0
        
        # Calculate trend (simple linear regression slope)
        weeks = sorted(weekly_counts.keys())
        counts = [weekly_counts[week] for week in weeks]
        
        n = len(weeks)
        sum_x = sum(range(n))
        sum_y = sum(counts)
        sum_xy = sum(i * counts[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    async def generate_productivity_metrics(self, team_data: Dict[str, Any]) -> List[AnalyticsMetric]:
        """Generate productivity metrics for a team"""
        metrics = []
        
        try:
            # Issues completed per day
            issues = team_data.get('issues', [])
            completed_issues = [i for i in issues if i.get('status') == 'done']
            
            if completed_issues:
                # Calculate completion rate
                completion_rate = len(completed_issues) / len(issues) if issues else 0
                metrics.append(AnalyticsMetric(
                    metric_id=str(uuid.uuid4()),
                    name="Issue Completion Rate",
                    analytics_type=AnalyticsType.PRODUCTIVITY,
                    value=completion_rate,
                    unit="percentage",
                    timestamp=datetime.now(timezone.utc),
                    dimensions={'team_id': team_data.get('team_id')}
                ))
                
                # Calculate average resolution time
                resolution_times = []
                for issue in completed_issues:
                    created = issue.get('created_at')
                    completed = issue.get('completed_at') or issue.get('updated_at')
                    if created and completed:
                        try:
                            created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            completed_dt = datetime.fromisoformat(completed.replace('Z', '+00:00'))
                            resolution_time = (completed_dt - created_dt).total_seconds() / 3600  # hours
                            resolution_times.append(resolution_time)
                        except:
                            continue
                
                if resolution_times:
                    avg_resolution_time = sum(resolution_times) / len(resolution_times)
                    metrics.append(AnalyticsMetric(
                        metric_id=str(uuid.uuid4()),
                        name="Average Issue Resolution Time",
                        analytics_type=AnalyticsType.PRODUCTIVITY,
                        value=avg_resolution_time,
                        unit="hours",
                        timestamp=datetime.now(timezone.utc),
                        dimensions={'team_id': team_data.get('team_id')}
                    ))
            
            # Communication activity
            messages = team_data.get('messages', [])
            if messages:
                daily_message_count = len(messages) / 7  # Assuming last 7 days
                metrics.append(AnalyticsMetric(
                    metric_id=str(uuid.uuid4()),
                    name="Daily Communication Activity",
                    analytics_type=AnalyticsType.COLLABORATION,
                    value=daily_message_count,
                    unit="messages_per_day",
                    timestamp=datetime.now(timezone.utc),
                    dimensions={'team_id': team_data.get('team_id')}
                ))
        
        except Exception as e:
            logger.error(f"Productivity metrics generation failed: {e}")
        
        return metrics
    
    async def predict_project_completion(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict project completion timeline"""
        try:
            issues = project_data.get('issues', [])
            if not issues:
                return {'error': 'No issues data available'}
            
            # Calculate current velocity
            completed_issues = [i for i in issues if i.get('status') == 'done']
            remaining_issues = [i for i in issues if i.get('status') != 'done']
            
            if not completed_issues:
                return {'error': 'No completed issues for velocity calculation'}
            
            # Calculate velocity (issues per week)
            completion_dates = []
            for issue in completed_issues:
                completed_date = issue.get('completed_at') or issue.get('updated_at')
                if completed_date:
                    try:
                        date = datetime.fromisoformat(completed_date.replace('Z', '+00:00'))
                        completion_dates.append(date)
                    except:
                        continue
            
            if len(completion_dates) < 2:
                return {'error': 'Insufficient completion data'}
            
            # Calculate velocity over last 4 weeks
            four_weeks_ago = datetime.now(timezone.utc) - timedelta(weeks=4)
            recent_completions = [d for d in completion_dates if d > four_weeks_ago]
            velocity = len(recent_completions) / 4  # issues per week
            
            if velocity <= 0:
                return {'error': 'Zero velocity detected'}
            
            # Predict completion
            weeks_remaining = len(remaining_issues) / velocity
            estimated_completion = datetime.now(timezone.utc) + timedelta(weeks=weeks_remaining)
            
            # Calculate confidence based on velocity consistency
            confidence = min(0.9, len(recent_completions) / 10)  # Higher confidence with more data
            
            return {
                'estimated_completion_date': estimated_completion.isoformat(),
                'weeks_remaining': weeks_remaining,
                'current_velocity': velocity,
                'remaining_issues': len(remaining_issues),
                'confidence': confidence,
                'prediction_date': datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Project completion prediction failed: {e}")
            return {'error': str(e)}

class WorkflowAutomationManager:
    """Main manager for workflow automation and analytics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_engine = WorkflowEngine(config.get('workflow_engine', {}))
        self.analytics_engine = AnalyticsEngine(config.get('analytics_engine', {}))
        self.running = False
    
    async def start(self):
        """Start the automation manager"""
        await self.workflow_engine.start()
        self.running = True
        logger.info("Workflow automation manager started")
    
    async def stop(self):
        """Stop the automation manager"""
        await self.workflow_engine.stop()
        self.running = False
        logger.info("Workflow automation manager stopped")
    
    async def create_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create a new workflow"""
        # Parse workflow definition
        workflow = WorkflowDefinition(
            workflow_id=workflow_definition.get('workflow_id', str(uuid.uuid4())),
            name=workflow_definition['name'],
            description=workflow_definition.get('description', ''),
            triggers=[
                WorkflowTrigger(
                    trigger_id=trigger.get('trigger_id', str(uuid.uuid4())),
                    trigger_type=WorkflowTriggerType(trigger['trigger_type']),
                    conditions=trigger.get('conditions', {}),
                    schedule=trigger.get('schedule'),
                    event_filters=trigger.get('event_filters', {}),
                    is_active=trigger.get('is_active', True)
                )
                for trigger in workflow_definition.get('triggers', [])
            ],
            actions=[
                WorkflowAction(
                    action_id=action.get('action_id', str(uuid.uuid4())),
                    action_type=ActionType(action['action_type']),
                    parameters=action.get('parameters', {}),
                    conditions=action.get('conditions', {}),
                    retry_config=action.get('retry_config', {}),
                    timeout_seconds=action.get('timeout_seconds', 300)
                )
                for action in workflow_definition.get('actions', [])
            ],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by=workflow_definition.get('created_by', 'system'),
            is_active=workflow_definition.get('is_active', True),
            version=workflow_definition.get('version', '1.0'),
            tags=workflow_definition.get('tags', []),
            metadata=workflow_definition.get('metadata', {})
        )
        
        self.workflow_engine.register_workflow(workflow)
        return workflow.workflow_id
    
    async def trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger an event that may activate workflows"""
        await self.workflow_engine.event_processor.emit_event(event_type, event_data)
    
    async def analyze_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive project analysis"""
        try:
            # Generate insights
            insights = await self.analytics_engine.analyze_project_health(project_data)
            
            # Generate metrics
            metrics = await self.analytics_engine.generate_productivity_metrics(project_data)
            
            # Predict completion
            completion_prediction = await self.analytics_engine.predict_project_completion(project_data)
            
            return {
                'project_id': project_data.get('project_id'),
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'insights': [
                    {
                        'insight_id': insight.insight_id,
                        'type': insight.insight_type,
                        'title': insight.title,
                        'description': insight.description,
                        'confidence': insight.confidence,
                        'impact_level': insight.impact_level,
                        'recommendations': insight.recommendations,
                        'supporting_data': insight.supporting_data
                    }
                    for insight in insights
                ],
                'metrics': [
                    {
                        'metric_id': metric.metric_id,
                        'name': metric.name,
                        'type': metric.analytics_type.value,
                        'value': metric.value,
                        'unit': metric.unit,
                        'dimensions': metric.dimensions
                    }
                    for metric in metrics
                ],
                'completion_prediction': completion_prediction
            }
        
        except Exception as e:
            logger.error(f"Project analysis failed: {e}")
            return {'error': str(e)}
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status and recent executions"""
        if workflow_id not in self.workflow_engine.workflows:
            return {'error': 'Workflow not found'}
        
        workflow = self.workflow_engine.workflows[workflow_id]
        
        # Get recent executions
        recent_executions = [
            execution for execution in self.workflow_engine.executions.values()
            if execution.workflow_id == workflow_id
        ]
        recent_executions.sort(key=lambda x: x.started_at, reverse=True)
        recent_executions = recent_executions[:10]  # Last 10 executions
        
        return {
            'workflow_id': workflow.workflow_id,
            'name': workflow.name,
            'description': workflow.description,
            'is_active': workflow.is_active,
            'version': workflow.version,
            'created_at': workflow.created_at.isoformat(),
            'updated_at': workflow.updated_at.isoformat(),
            'triggers_count': len(workflow.triggers),
            'actions_count': len(workflow.actions),
            'recent_executions': [
                {
                    'execution_id': execution.execution_id,
                    'status': execution.status.value,
                    'started_at': execution.started_at.isoformat(),
                    'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
                    'triggered_by': execution.triggered_by,
                    'error_message': execution.error_message
                }
                for execution in recent_executions
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'status': 'healthy' if self.running else 'stopped',
            'workflow_engine': {
                'active_workflows': len(self.workflow_engine.workflows),
                'running_executions': len([
                    e for e in self.workflow_engine.executions.values()
                    if e.status == WorkflowStatus.RUNNING
                ])
            },
            'analytics_engine': {
                'metrics_stored': sum(len(metrics) for metrics in self.analytics_engine.metrics_store.values()),
                'insights_generated': sum(len(insights) for insights in self.analytics_engine.insights_store.values())
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example configuration
        config = {
            'workflow_engine': {
                'max_concurrent_executions': 10
            },
            'analytics_engine': {
                'metrics_retention_days': 90
            }
        }
        
        manager = WorkflowAutomationManager(config)
        await manager.start()
        
        # Example workflow definition
        workflow_def = {
            'name': 'Issue Escalation Workflow',
            'description': 'Automatically escalate high-priority issues',
            'triggers': [
                {
                    'trigger_type': 'event_based',
                    'conditions': {
                        'event_type': 'issue_created',
                        'field': 'priority',
                        'operator': 'eq',
                        'value': 'critical'
                    }
                }
            ],
            'actions': [
                {
                    'action_type': 'send_notification',
                    'parameters': {
                        'message': 'Critical issue requires immediate attention',
                        'recipients': ['team-lead@company.com']
                    }
                }
            ]
        }
        
        # Create workflow
        workflow_id = await manager.create_workflow(workflow_def)
        print(f"Created workflow: {workflow_id}")
        
        # Trigger event
        await manager.trigger_event('issue_created', {
            'issue_id': 'ISSUE-123',
            'priority': 'critical',
            'title': 'System outage'
        })
        
        # Wait for execution
        await asyncio.sleep(2)
        
        # Check workflow status
        status = await manager.get_workflow_status(workflow_id)
        print(f"Workflow status: {status}")
        
        # Example project analysis
        project_data = {
            'project_id': 'PROJECT-1',
            'issues': [
                {
                    'id': 'ISSUE-1',
                    'status': 'done',
                    'created_at': '2024-01-01T00:00:00Z',
                    'completed_at': '2024-01-05T00:00:00Z'
                },
                {
                    'id': 'ISSUE-2',
                    'status': 'in_progress',
                    'created_at': '2024-01-03T00:00:00Z',
                    'due_date': '2024-01-10T00:00:00Z'
                }
            ],
            'messages': [
                {
                    'id': 'MSG-1',
                    'author': 'user1',
                    'sentiment_score': 0.5,
                    'timestamp': '2024-01-06T00:00:00Z'
                }
            ]
        }
        
        analysis = await manager.analyze_project(project_data)
        print(f"Project analysis: {analysis}")
        
        # Health check
        health = await manager.health_check()
        print(f"Health status: {health['status']}")
        
        await manager.stop()
    
    asyncio.run(main())

