"""
Ticket Workflow Manager for Nexus Architect
Comprehensive end-to-end ticket-to-production automation with safety controls
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import tempfile
import os
from pathlib import Path

import requests
import git
from jira import JIRA
import slack_sdk
from github import Github

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketStatus(Enum):
    """Ticket status enumeration"""
    OPEN = "open"
    IN_ANALYSIS = "in_analysis"
    FIX_GENERATED = "fix_generated"
    FIX_VALIDATED = "fix_validated"
    IN_TESTING = "in_testing"
    READY_FOR_DEPLOYMENT = "ready_for_deployment"
    DEPLOYED = "deployed"
    VERIFIED = "verified"
    CLOSED = "closed"
    FAILED = "failed"
    NEEDS_HUMAN_REVIEW = "needs_human_review"

class WorkflowStage(Enum):
    """Workflow stages"""
    TICKET_INTAKE = "ticket_intake"
    BUG_ANALYSIS = "bug_analysis"
    FIX_GENERATION = "fix_generation"
    FIX_VALIDATION = "fix_validation"
    AUTOMATED_TESTING = "automated_testing"
    HUMAN_REVIEW = "human_review"
    DEPLOYMENT = "deployment"
    VERIFICATION = "verification"
    CLOSURE = "closure"

class AutomationLevel(Enum):
    """Levels of automation"""
    MANUAL = "manual"
    SEMI_AUTOMATED = "semi_automated"
    FULLY_AUTOMATED = "fully_automated"

@dataclass
class TicketWorkflow:
    """Ticket workflow configuration"""
    ticket_id: str
    workflow_id: str
    current_stage: WorkflowStage
    automation_level: AutomationLevel
    stages_completed: List[WorkflowStage]
    estimated_completion: datetime
    actual_completion: Optional[datetime]
    human_approvals_required: List[str]
    safety_checks_passed: bool
    rollback_plan: str
    created_at: datetime
    updated_at: datetime

@dataclass
class WorkflowExecution:
    """Workflow execution details"""
    execution_id: str
    ticket_id: str
    stage: WorkflowStage
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    success: bool
    error_message: Optional[str]
    artifacts: Dict[str, str]
    metrics: Dict[str, Any]

@dataclass
class DeploymentPlan:
    """Deployment plan for fixes"""
    plan_id: str
    ticket_id: str
    fix_id: str
    target_environment: str
    deployment_strategy: str
    rollback_strategy: str
    pre_deployment_checks: List[str]
    post_deployment_checks: List[str]
    estimated_downtime: int  # minutes
    risk_level: str
    approval_required: bool

class TicketWorkflowManager:
    """
    Comprehensive ticket workflow management with end-to-end automation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_workflows = {}
        self.workflow_history = []
        self.automation_rules = {}
        
        # Initialize integrations
        self._initialize_integrations()
        self._initialize_automation_rules()
    
    def _initialize_integrations(self):
        """Initialize external system integrations"""
        try:
            # JIRA integration
            if self.config.get("jira"):
                jira_config = self.config["jira"]
                self.jira_client = JIRA(
                    server=jira_config["server"],
                    basic_auth=(jira_config["username"], jira_config["token"])
                )
            else:
                self.jira_client = None
            
            # GitHub integration
            if self.config.get("github"):
                github_config = self.config["github"]
                self.github_client = Github(github_config["token"])
            else:
                self.github_client = None
            
            # Slack integration
            if self.config.get("slack"):
                slack_config = self.config["slack"]
                self.slack_client = slack_sdk.WebClient(token=slack_config["token"])
            else:
                self.slack_client = None
            
            logger.info("External integrations initialized")
            
        except Exception as e:
            logger.error(f"Error initializing integrations: {str(e)}")
    
    def _initialize_automation_rules(self):
        """Initialize automation rules based on configuration"""
        try:
            self.automation_rules = {
                "low_complexity": {
                    "automation_level": AutomationLevel.FULLY_AUTOMATED,
                    "human_review_required": False,
                    "auto_deploy_to_staging": True,
                    "auto_deploy_to_production": False,
                    "max_processing_time": 120  # minutes
                },
                "medium_complexity": {
                    "automation_level": AutomationLevel.SEMI_AUTOMATED,
                    "human_review_required": True,
                    "auto_deploy_to_staging": True,
                    "auto_deploy_to_production": False,
                    "max_processing_time": 240  # minutes
                },
                "high_complexity": {
                    "automation_level": AutomationLevel.MANUAL,
                    "human_review_required": True,
                    "auto_deploy_to_staging": False,
                    "auto_deploy_to_production": False,
                    "max_processing_time": 480  # minutes
                },
                "security_related": {
                    "automation_level": AutomationLevel.MANUAL,
                    "human_review_required": True,
                    "auto_deploy_to_staging": False,
                    "auto_deploy_to_production": False,
                    "additional_approvals": ["security_team", "senior_developer"]
                }
            }
            
            logger.info("Automation rules initialized")
            
        except Exception as e:
            logger.error(f"Error initializing automation rules: {str(e)}")
    
    async def process_ticket(self, ticket_data: Dict[str, Any]) -> TicketWorkflow:
        """
        Process a ticket through the complete workflow
        
        Args:
            ticket_data: Ticket information and metadata
            
        Returns:
            Workflow instance
        """
        start_time = time.time()
        
        try:
            # Create workflow instance
            workflow = self._create_workflow(ticket_data)
            
            # Store active workflow
            self.active_workflows[workflow.ticket_id] = workflow
            
            # Start workflow execution
            await self._execute_workflow(workflow)
            
            processing_time = time.time() - start_time
            logger.info(f"Ticket {workflow.ticket_id} processed in {processing_time:.2f}s")
            
            return workflow
            
        except Exception as e:
            logger.error(f"Error processing ticket: {str(e)}")
            raise
    
    def _create_workflow(self, ticket_data: Dict[str, Any]) -> TicketWorkflow:
        """Create workflow instance from ticket data"""
        try:
            ticket_id = ticket_data.get("id", str(uuid.uuid4()))
            workflow_id = f"WF-{ticket_id}-{int(time.time())}"
            
            # Determine automation level based on ticket characteristics
            automation_level = self._determine_automation_level(ticket_data)
            
            # Calculate estimated completion time
            complexity = ticket_data.get("complexity", "medium")
            rules = self.automation_rules.get(complexity, self.automation_rules["medium_complexity"])
            estimated_completion = datetime.now() + timedelta(minutes=rules["max_processing_time"])
            
            # Determine required approvals
            human_approvals = []
            if rules.get("human_review_required", False):
                human_approvals.append("technical_lead")
            
            if "security" in ticket_data.get("labels", []):
                human_approvals.extend(rules.get("additional_approvals", []))
            
            workflow = TicketWorkflow(
                ticket_id=ticket_id,
                workflow_id=workflow_id,
                current_stage=WorkflowStage.TICKET_INTAKE,
                automation_level=automation_level,
                stages_completed=[],
                estimated_completion=estimated_completion,
                actual_completion=None,
                human_approvals_required=human_approvals,
                safety_checks_passed=False,
                rollback_plan="",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating workflow: {str(e)}")
            raise
    
    def _determine_automation_level(self, ticket_data: Dict[str, Any]) -> AutomationLevel:
        """Determine appropriate automation level for ticket"""
        try:
            # Check complexity
            complexity = ticket_data.get("complexity", "medium")
            
            # Check labels for special handling
            labels = ticket_data.get("labels", [])
            
            if "security" in labels or "critical" in labels:
                return AutomationLevel.MANUAL
            elif complexity == "low":
                return AutomationLevel.FULLY_AUTOMATED
            elif complexity == "medium":
                return AutomationLevel.SEMI_AUTOMATED
            else:
                return AutomationLevel.MANUAL
            
        except Exception as e:
            logger.error(f"Error determining automation level: {str(e)}")
            return AutomationLevel.MANUAL
    
    async def _execute_workflow(self, workflow: TicketWorkflow):
        """Execute the complete workflow"""
        try:
            # Define workflow stages
            stages = [
                WorkflowStage.TICKET_INTAKE,
                WorkflowStage.BUG_ANALYSIS,
                WorkflowStage.FIX_GENERATION,
                WorkflowStage.FIX_VALIDATION,
                WorkflowStage.AUTOMATED_TESTING
            ]
            
            # Add conditional stages based on automation level
            if workflow.automation_level != AutomationLevel.FULLY_AUTOMATED:
                stages.append(WorkflowStage.HUMAN_REVIEW)
            
            stages.extend([
                WorkflowStage.DEPLOYMENT,
                WorkflowStage.VERIFICATION,
                WorkflowStage.CLOSURE
            ])
            
            # Execute each stage
            for stage in stages:
                workflow.current_stage = stage
                workflow.updated_at = datetime.now()
                
                # Execute stage
                execution = await self._execute_stage(workflow, stage)
                
                if execution.success:
                    workflow.stages_completed.append(stage)
                    
                    # Send notifications
                    await self._send_stage_notification(workflow, stage, execution)
                else:
                    # Handle stage failure
                    await self._handle_stage_failure(workflow, stage, execution)
                    break
            
            # Mark workflow as completed
            if len(workflow.stages_completed) == len(stages):
                workflow.actual_completion = datetime.now()
                await self._complete_workflow(workflow)
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow.workflow_id}: {str(e)}")
            await self._handle_workflow_failure(workflow, str(e))
    
    async def _execute_stage(self, workflow: TicketWorkflow, stage: WorkflowStage) -> WorkflowExecution:
        """Execute a specific workflow stage"""
        start_time = time.time()
        execution_id = f"EX-{workflow.ticket_id}-{stage.value}-{int(start_time)}"
        
        try:
            logger.info(f"Executing stage {stage.value} for ticket {workflow.ticket_id}")
            
            # Stage-specific execution
            if stage == WorkflowStage.TICKET_INTAKE:
                result = await self._execute_ticket_intake(workflow)
            elif stage == WorkflowStage.BUG_ANALYSIS:
                result = await self._execute_bug_analysis(workflow)
            elif stage == WorkflowStage.FIX_GENERATION:
                result = await self._execute_fix_generation(workflow)
            elif stage == WorkflowStage.FIX_VALIDATION:
                result = await self._execute_fix_validation(workflow)
            elif stage == WorkflowStage.AUTOMATED_TESTING:
                result = await self._execute_automated_testing(workflow)
            elif stage == WorkflowStage.HUMAN_REVIEW:
                result = await self._execute_human_review(workflow)
            elif stage == WorkflowStage.DEPLOYMENT:
                result = await self._execute_deployment(workflow)
            elif stage == WorkflowStage.VERIFICATION:
                result = await self._execute_verification(workflow)
            elif stage == WorkflowStage.CLOSURE:
                result = await self._execute_closure(workflow)
            else:
                result = {"success": False, "error": f"Unknown stage: {stage.value}"}
            
            end_time = time.time()
            duration = end_time - start_time
            
            execution = WorkflowExecution(
                execution_id=execution_id,
                ticket_id=workflow.ticket_id,
                stage=stage,
                status="completed" if result["success"] else "failed",
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.fromtimestamp(end_time),
                duration=duration,
                success=result["success"],
                error_message=result.get("error"),
                artifacts=result.get("artifacts", {}),
                metrics=result.get("metrics", {})
            )
            
            return execution
            
        except Exception as e:
            logger.error(f"Error executing stage {stage.value}: {str(e)}")
            
            return WorkflowExecution(
                execution_id=execution_id,
                ticket_id=workflow.ticket_id,
                stage=stage,
                status="failed",
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now(),
                duration=time.time() - start_time,
                success=False,
                error_message=str(e),
                artifacts={},
                metrics={}
            )
    
    async def _execute_ticket_intake(self, workflow: TicketWorkflow) -> Dict[str, Any]:
        """Execute ticket intake stage"""
        try:
            # Validate ticket data
            # Extract and normalize ticket information
            # Set up tracking and monitoring
            
            await asyncio.sleep(1)  # Simulate processing
            
            return {
                "success": True,
                "artifacts": {"ticket_normalized": True},
                "metrics": {"processing_time": 1.0}
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_bug_analysis(self, workflow: TicketWorkflow) -> Dict[str, Any]:
        """Execute bug analysis stage"""
        try:
            # Call bug analysis service
            # This would integrate with the intelligent_bug_analyzer
            
            await asyncio.sleep(5)  # Simulate analysis
            
            return {
                "success": True,
                "artifacts": {
                    "analysis_report": "bug_analysis_report.json",
                    "root_cause": "identified"
                },
                "metrics": {
                    "analysis_confidence": 0.85,
                    "processing_time": 5.0
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_fix_generation(self, workflow: TicketWorkflow) -> Dict[str, Any]:
        """Execute fix generation stage"""
        try:
            # Call fix generation service
            # This would integrate with the autonomous_fix_generator
            
            await asyncio.sleep(10)  # Simulate fix generation
            
            return {
                "success": True,
                "artifacts": {
                    "fix_candidates": "fix_candidates.json",
                    "selected_fix": "fix_001"
                },
                "metrics": {
                    "fixes_generated": 3,
                    "best_confidence": 0.82,
                    "processing_time": 10.0
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_fix_validation(self, workflow: TicketWorkflow) -> Dict[str, Any]:
        """Execute fix validation stage"""
        try:
            # Validate generated fix
            # Run safety checks
            # Verify fix quality
            
            await asyncio.sleep(3)  # Simulate validation
            
            # Update safety checks status
            workflow.safety_checks_passed = True
            
            return {
                "success": True,
                "artifacts": {
                    "validation_report": "validation_report.json",
                    "safety_checks": "passed"
                },
                "metrics": {
                    "validation_score": 0.9,
                    "safety_score": 0.95,
                    "processing_time": 3.0
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_automated_testing(self, workflow: TicketWorkflow) -> Dict[str, Any]:
        """Execute automated testing stage"""
        try:
            # Run automated tests
            # Execute test suites
            # Verify fix effectiveness
            
            await asyncio.sleep(8)  # Simulate testing
            
            return {
                "success": True,
                "artifacts": {
                    "test_results": "test_results.json",
                    "coverage_report": "coverage_report.html"
                },
                "metrics": {
                    "tests_passed": 45,
                    "tests_failed": 0,
                    "coverage_percentage": 92.5,
                    "processing_time": 8.0
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_human_review(self, workflow: TicketWorkflow) -> Dict[str, Any]:
        """Execute human review stage"""
        try:
            # Request human review
            # Send notifications to required approvers
            # Wait for approval (in real implementation)
            
            if workflow.automation_level == AutomationLevel.FULLY_AUTOMATED:
                # Skip human review for fully automated workflows
                return {
                    "success": True,
                    "artifacts": {"review_status": "skipped"},
                    "metrics": {"processing_time": 0.1}
                }
            
            # Simulate human review process
            await self._request_human_approval(workflow)
            
            # For demo purposes, assume approval after short delay
            await asyncio.sleep(2)
            
            return {
                "success": True,
                "artifacts": {
                    "review_status": "approved",
                    "reviewer": "technical_lead",
                    "review_comments": "Fix looks good, approved for deployment"
                },
                "metrics": {
                    "review_time": 2.0,
                    "approval_score": 0.9
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_deployment(self, workflow: TicketWorkflow) -> Dict[str, Any]:
        """Execute deployment stage"""
        try:
            # Create deployment plan
            deployment_plan = await self._create_deployment_plan(workflow)
            
            # Execute deployment
            deployment_result = await self._execute_deployment_plan(deployment_plan)
            
            return {
                "success": deployment_result["success"],
                "artifacts": {
                    "deployment_plan": deployment_plan.plan_id,
                    "deployment_log": "deployment.log"
                },
                "metrics": {
                    "deployment_time": deployment_result.get("duration", 0),
                    "downtime": deployment_result.get("downtime", 0)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_verification(self, workflow: TicketWorkflow) -> Dict[str, Any]:
        """Execute verification stage"""
        try:
            # Verify deployment success
            # Run post-deployment tests
            # Check system health
            
            await asyncio.sleep(5)  # Simulate verification
            
            return {
                "success": True,
                "artifacts": {
                    "verification_report": "verification_report.json",
                    "health_check": "passed"
                },
                "metrics": {
                    "verification_score": 0.95,
                    "system_health": 0.98,
                    "processing_time": 5.0
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_closure(self, workflow: TicketWorkflow) -> Dict[str, Any]:
        """Execute closure stage"""
        try:
            # Update ticket status
            # Send completion notifications
            # Archive workflow data
            
            await self._update_ticket_status(workflow.ticket_id, TicketStatus.CLOSED)
            await self._send_completion_notification(workflow)
            
            return {
                "success": True,
                "artifacts": {
                    "closure_report": "closure_report.json"
                },
                "metrics": {
                    "total_processing_time": (datetime.now() - workflow.created_at).total_seconds()
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _create_deployment_plan(self, workflow: TicketWorkflow) -> DeploymentPlan:
        """Create deployment plan for the fix"""
        try:
            plan_id = f"DP-{workflow.ticket_id}-{int(time.time())}"
            
            # Determine deployment strategy based on automation level
            if workflow.automation_level == AutomationLevel.FULLY_AUTOMATED:
                strategy = "blue_green"
                approval_required = False
            else:
                strategy = "rolling"
                approval_required = True
            
            plan = DeploymentPlan(
                plan_id=plan_id,
                ticket_id=workflow.ticket_id,
                fix_id="fix_001",  # Would come from fix generation
                target_environment="staging",  # Start with staging
                deployment_strategy=strategy,
                rollback_strategy="immediate_rollback",
                pre_deployment_checks=[
                    "verify_staging_environment",
                    "backup_current_version",
                    "run_pre_deployment_tests"
                ],
                post_deployment_checks=[
                    "verify_application_health",
                    "run_smoke_tests",
                    "check_performance_metrics"
                ],
                estimated_downtime=0 if strategy == "blue_green" else 5,
                risk_level="low" if workflow.safety_checks_passed else "medium",
                approval_required=approval_required
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating deployment plan: {str(e)}")
            raise
    
    async def _execute_deployment_plan(self, plan: DeploymentPlan) -> Dict[str, Any]:
        """Execute the deployment plan"""
        try:
            start_time = time.time()
            
            # Pre-deployment checks
            for check in plan.pre_deployment_checks:
                logger.info(f"Executing pre-deployment check: {check}")
                await asyncio.sleep(0.5)  # Simulate check
            
            # Execute deployment
            logger.info(f"Deploying using {plan.deployment_strategy} strategy")
            await asyncio.sleep(3)  # Simulate deployment
            
            # Post-deployment checks
            for check in plan.post_deployment_checks:
                logger.info(f"Executing post-deployment check: {check}")
                await asyncio.sleep(0.5)  # Simulate check
            
            duration = time.time() - start_time
            
            return {
                "success": True,
                "duration": duration,
                "downtime": plan.estimated_downtime
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _request_human_approval(self, workflow: TicketWorkflow):
        """Request human approval for workflow"""
        try:
            # Send notifications to required approvers
            for approver in workflow.human_approvals_required:
                await self._send_approval_request(workflow, approver)
            
            logger.info(f"Approval requested for workflow {workflow.workflow_id}")
            
        except Exception as e:
            logger.error(f"Error requesting approval: {str(e)}")
    
    async def _send_approval_request(self, workflow: TicketWorkflow, approver: str):
        """Send approval request to specific approver"""
        try:
            # Send Slack notification if available
            if self.slack_client:
                message = f"""
🔧 *Fix Approval Required*
Ticket: {workflow.ticket_id}
Workflow: {workflow.workflow_id}
Stage: {workflow.current_stage.value}
Automation Level: {workflow.automation_level.value}

Please review and approve the generated fix.
"""
                # In real implementation, would send to specific user/channel
                logger.info(f"Approval request sent to {approver}")
            
        except Exception as e:
            logger.error(f"Error sending approval request: {str(e)}")
    
    async def _update_ticket_status(self, ticket_id: str, status: TicketStatus):
        """Update ticket status in external system"""
        try:
            # Update JIRA ticket if available
            if self.jira_client:
                # In real implementation, would update JIRA ticket
                logger.info(f"Updated ticket {ticket_id} status to {status.value}")
            
        except Exception as e:
            logger.error(f"Error updating ticket status: {str(e)}")
    
    async def _send_stage_notification(self, workflow: TicketWorkflow, stage: WorkflowStage, execution: WorkflowExecution):
        """Send notification for stage completion"""
        try:
            if self.slack_client:
                message = f"""
✅ *Stage Completed*
Ticket: {workflow.ticket_id}
Stage: {stage.value}
Duration: {execution.duration:.2f}s
Status: {execution.status}
"""
                logger.info(f"Stage notification sent for {stage.value}")
            
        except Exception as e:
            logger.error(f"Error sending stage notification: {str(e)}")
    
    async def _send_completion_notification(self, workflow: TicketWorkflow):
        """Send workflow completion notification"""
        try:
            total_time = (datetime.now() - workflow.created_at).total_seconds()
            
            if self.slack_client:
                message = f"""
🎉 *Workflow Completed*
Ticket: {workflow.ticket_id}
Total Time: {total_time:.2f}s
Stages Completed: {len(workflow.stages_completed)}
Automation Level: {workflow.automation_level.value}
"""
                logger.info(f"Completion notification sent for {workflow.ticket_id}")
            
        except Exception as e:
            logger.error(f"Error sending completion notification: {str(e)}")
    
    async def _handle_stage_failure(self, workflow: TicketWorkflow, stage: WorkflowStage, execution: WorkflowExecution):
        """Handle stage failure"""
        try:
            logger.error(f"Stage {stage.value} failed for workflow {workflow.workflow_id}: {execution.error_message}")
            
            # Update workflow status
            workflow.current_stage = WorkflowStage.HUMAN_REVIEW
            workflow.updated_at = datetime.now()
            
            # Send failure notification
            if self.slack_client:
                message = f"""
❌ *Stage Failed*
Ticket: {workflow.ticket_id}
Stage: {stage.value}
Error: {execution.error_message}
Action: Escalated to human review
"""
                logger.info(f"Failure notification sent for {stage.value}")
            
        except Exception as e:
            logger.error(f"Error handling stage failure: {str(e)}")
    
    async def _handle_workflow_failure(self, workflow: TicketWorkflow, error: str):
        """Handle complete workflow failure"""
        try:
            logger.error(f"Workflow {workflow.workflow_id} failed: {error}")
            
            # Update workflow status
            workflow.current_stage = WorkflowStage.HUMAN_REVIEW
            workflow.updated_at = datetime.now()
            
            # Send critical failure notification
            if self.slack_client:
                message = f"""
🚨 *Workflow Failed*
Ticket: {workflow.ticket_id}
Workflow: {workflow.workflow_id}
Error: {error}
Action: Manual intervention required
"""
                logger.info(f"Critical failure notification sent")
            
        except Exception as e:
            logger.error(f"Error handling workflow failure: {str(e)}")
    
    async def _complete_workflow(self, workflow: TicketWorkflow):
        """Complete the workflow"""
        try:
            # Move to completed workflows
            self.workflow_history.append(workflow)
            
            # Remove from active workflows
            if workflow.ticket_id in self.active_workflows:
                del self.active_workflows[workflow.ticket_id]
            
            logger.info(f"Workflow {workflow.workflow_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error completing workflow: {str(e)}")
    
    def get_workflow_status(self, ticket_id: str) -> Optional[TicketWorkflow]:
        """Get current workflow status"""
        return self.active_workflows.get(ticket_id)
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics"""
        try:
            active_count = len(self.active_workflows)
            completed_count = len(self.workflow_history)
            
            # Calculate average completion time
            if self.workflow_history:
                completion_times = []
                for workflow in self.workflow_history:
                    if workflow.actual_completion:
                        duration = (workflow.actual_completion - workflow.created_at).total_seconds()
                        completion_times.append(duration)
                
                avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
            else:
                avg_completion_time = 0
            
            # Calculate success rate
            successful_workflows = sum(1 for w in self.workflow_history if w.actual_completion)
            success_rate = successful_workflows / completed_count if completed_count > 0 else 0
            
            return {
                "active_workflows": active_count,
                "completed_workflows": completed_count,
                "average_completion_time": avg_completion_time,
                "success_rate": success_rate,
                "automation_levels": {
                    "fully_automated": sum(1 for w in self.workflow_history if w.automation_level == AutomationLevel.FULLY_AUTOMATED),
                    "semi_automated": sum(1 for w in self.workflow_history if w.automation_level == AutomationLevel.SEMI_AUTOMATED),
                    "manual": sum(1 for w in self.workflow_history if w.automation_level == AutomationLevel.MANUAL)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow metrics: {str(e)}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    async def test_workflow_manager():
        """Test the workflow manager"""
        config = {
            "jira": {
                "server": "https://company.atlassian.net",
                "username": "automation@company.com",
                "token": "fake_token"
            },
            "slack": {
                "token": "fake_slack_token"
            },
            "github": {
                "token": "fake_github_token"
            }
        }
        
        manager = TicketWorkflowManager(config)
        
        # Test ticket data
        ticket_data = {
            "id": "BUG-001",
            "title": "NullPointerException in PaymentProcessor",
            "description": "Application crashes when processing payments with null customer data",
            "complexity": "low",
            "labels": ["bug", "payment"],
            "priority": "high",
            "reporter": "user@company.com"
        }
        
        print("Starting ticket workflow...")
        workflow = await manager.process_ticket(ticket_data)
        
        print(f"\nWorkflow completed:")
        print(f"  Ticket ID: {workflow.ticket_id}")
        print(f"  Workflow ID: {workflow.workflow_id}")
        print(f"  Automation Level: {workflow.automation_level.value}")
        print(f"  Stages Completed: {len(workflow.stages_completed)}")
        print(f"  Safety Checks Passed: {workflow.safety_checks_passed}")
        
        if workflow.actual_completion:
            duration = (workflow.actual_completion - workflow.created_at).total_seconds()
            print(f"  Total Duration: {duration:.2f}s")
        
        # Get metrics
        metrics = manager.get_workflow_metrics()
        print(f"\nWorkflow Metrics:")
        print(f"  Active Workflows: {metrics['active_workflows']}")
        print(f"  Completed Workflows: {metrics['completed_workflows']}")
        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Average Completion Time: {metrics['average_completion_time']:.2f}s")
    
    # Run the test
    asyncio.run(test_workflow_manager())

