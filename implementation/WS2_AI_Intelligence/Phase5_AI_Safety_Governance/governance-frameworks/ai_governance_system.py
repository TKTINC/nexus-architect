"""
Nexus Architect AI Governance System

This module implements comprehensive AI governance frameworks including human oversight,
approval workflows, escalation procedures, and compliance monitoring.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import time

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from celery import Celery
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of AI decisions requiring governance"""
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"
    AUTONOMOUS = "autonomous"
    HUMAN_ASSISTED = "human_assisted"

class ApprovalStatus(Enum):
    """Approval status for AI decisions"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class GovernanceRole(Enum):
    """Roles in AI governance system"""
    AI_OPERATOR = "ai_operator"
    DOMAIN_EXPERT = "domain_expert"
    ETHICS_REVIEWER = "ethics_reviewer"
    COMPLIANCE_OFFICER = "compliance_officer"
    SENIOR_MANAGER = "senior_manager"
    EXECUTIVE = "executive"
    SYSTEM_ADMIN = "system_admin"

class ComplianceFramework(Enum):
    """Compliance frameworks to monitor"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    AI_ETHICS = "ai_ethics"
    INTERNAL_POLICY = "internal_policy"

@dataclass
class GovernancePolicy:
    """AI governance policy definition"""
    policy_id: str
    name: str
    description: str
    decision_types: List[DecisionType]
    required_approvers: List[GovernanceRole]
    approval_threshold: int
    escalation_timeout: timedelta
    compliance_frameworks: List[ComplianceFramework]
    risk_thresholds: Dict[str, float]
    automated_approval_conditions: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_active: bool

@dataclass
class DecisionRequest:
    """AI decision request for governance review"""
    request_id: str
    decision_type: DecisionType
    description: str
    context: Dict[str, Any]
    risk_assessment: Dict[str, float]
    ai_confidence: float
    requested_by: str
    requested_at: datetime
    required_approvers: List[GovernanceRole]
    approval_deadline: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ApprovalRecord:
    """Record of approval/rejection"""
    approval_id: str
    request_id: str
    approver_id: str
    approver_role: GovernanceRole
    status: ApprovalStatus
    comments: str
    timestamp: datetime
    evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditTrail:
    """Audit trail entry for AI decisions"""
    audit_id: str
    request_id: str
    action: str
    actor: str
    timestamp: datetime
    details: Dict[str, Any]
    compliance_tags: List[str]

class RiskAssessment:
    """Risk assessment for AI decisions"""
    
    def __init__(self):
        self.risk_factors = {
            'data_sensitivity': {
                'personal_data': 0.8,
                'financial_data': 0.9,
                'health_data': 0.95,
                'public_data': 0.2
            },
            'decision_impact': {
                'individual_impact': 0.7,
                'group_impact': 0.8,
                'organizational_impact': 0.9,
                'societal_impact': 0.95
            },
            'automation_level': {
                'human_in_loop': 0.3,
                'human_on_loop': 0.6,
                'human_out_loop': 0.9
            },
            'reversibility': {
                'easily_reversible': 0.2,
                'moderately_reversible': 0.5,
                'difficult_to_reverse': 0.8,
                'irreversible': 0.95
            }
        }
    
    def assess_risk(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess risk level for AI decision
        
        Args:
            context: Decision context information
            
        Returns:
            Risk assessment scores
        """
        risk_scores = {}
        
        # Data sensitivity risk
        data_type = context.get('data_type', 'public_data')
        risk_scores['data_sensitivity'] = self.risk_factors['data_sensitivity'].get(data_type, 0.5)
        
        # Decision impact risk
        impact_level = context.get('impact_level', 'individual_impact')
        risk_scores['decision_impact'] = self.risk_factors['decision_impact'].get(impact_level, 0.5)
        
        # Automation level risk
        automation = context.get('automation_level', 'human_in_loop')
        risk_scores['automation_level'] = self.risk_factors['automation_level'].get(automation, 0.5)
        
        # Reversibility risk
        reversibility = context.get('reversibility', 'moderately_reversible')
        risk_scores['reversibility'] = self.risk_factors['reversibility'].get(reversibility, 0.5)
        
        # AI confidence risk (inverse relationship)
        ai_confidence = context.get('ai_confidence', 0.7)
        risk_scores['ai_confidence'] = 1.0 - ai_confidence
        
        # Bias risk
        bias_score = context.get('bias_score', 0.0)
        risk_scores['bias_risk'] = bias_score
        
        # Safety risk
        safety_score = context.get('safety_score', 0.9)
        risk_scores['safety_risk'] = 1.0 - safety_score
        
        # Calculate overall risk
        weights = {
            'data_sensitivity': 0.2,
            'decision_impact': 0.25,
            'automation_level': 0.15,
            'reversibility': 0.15,
            'ai_confidence': 0.1,
            'bias_risk': 0.1,
            'safety_risk': 0.05
        }
        
        overall_risk = sum(risk_scores[factor] * weight for factor, weight in weights.items())
        risk_scores['overall'] = overall_risk
        
        return risk_scores

class PolicyEngine:
    """AI governance policy engine"""
    
    def __init__(self):
        self.policies = {}
        self.risk_assessor = RiskAssessment()
        self._load_default_policies()
    
    def _load_default_policies(self):
        """Load default governance policies"""
        
        # Low risk policy
        self.policies['low_risk'] = GovernancePolicy(
            policy_id='low_risk_001',
            name='Low Risk AI Decisions',
            description='Automated approval for low-risk AI decisions',
            decision_types=[DecisionType.LOW_RISK],
            required_approvers=[GovernanceRole.AI_OPERATOR],
            approval_threshold=1,
            escalation_timeout=timedelta(hours=1),
            compliance_frameworks=[ComplianceFramework.INTERNAL_POLICY],
            risk_thresholds={'overall': 0.3},
            automated_approval_conditions={
                'max_risk': 0.3,
                'min_confidence': 0.8,
                'max_bias': 0.2
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True
        )
        
        # Medium risk policy
        self.policies['medium_risk'] = GovernancePolicy(
            policy_id='medium_risk_001',
            name='Medium Risk AI Decisions',
            description='Human review required for medium-risk AI decisions',
            decision_types=[DecisionType.MEDIUM_RISK],
            required_approvers=[GovernanceRole.DOMAIN_EXPERT, GovernanceRole.AI_OPERATOR],
            approval_threshold=2,
            escalation_timeout=timedelta(hours=4),
            compliance_frameworks=[ComplianceFramework.INTERNAL_POLICY, ComplianceFramework.AI_ETHICS],
            risk_thresholds={'overall': 0.6},
            automated_approval_conditions={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True
        )
        
        # High risk policy
        self.policies['high_risk'] = GovernancePolicy(
            policy_id='high_risk_001',
            name='High Risk AI Decisions',
            description='Multi-level approval required for high-risk AI decisions',
            decision_types=[DecisionType.HIGH_RISK],
            required_approvers=[
                GovernanceRole.DOMAIN_EXPERT,
                GovernanceRole.ETHICS_REVIEWER,
                GovernanceRole.COMPLIANCE_OFFICER,
                GovernanceRole.SENIOR_MANAGER
            ],
            approval_threshold=3,
            escalation_timeout=timedelta(hours=8),
            compliance_frameworks=[
                ComplianceFramework.GDPR,
                ComplianceFramework.AI_ETHICS,
                ComplianceFramework.INTERNAL_POLICY
            ],
            risk_thresholds={'overall': 0.8},
            automated_approval_conditions={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True
        )
        
        # Critical risk policy
        self.policies['critical'] = GovernancePolicy(
            policy_id='critical_001',
            name='Critical AI Decisions',
            description='Executive approval required for critical AI decisions',
            decision_types=[DecisionType.CRITICAL],
            required_approvers=[
                GovernanceRole.DOMAIN_EXPERT,
                GovernanceRole.ETHICS_REVIEWER,
                GovernanceRole.COMPLIANCE_OFFICER,
                GovernanceRole.SENIOR_MANAGER,
                GovernanceRole.EXECUTIVE
            ],
            approval_threshold=4,
            escalation_timeout=timedelta(hours=12),
            compliance_frameworks=[
                ComplianceFramework.GDPR,
                ComplianceFramework.HIPAA,
                ComplianceFramework.AI_ETHICS,
                ComplianceFramework.INTERNAL_POLICY
            ],
            risk_thresholds={'overall': 0.9},
            automated_approval_conditions={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True
        )
    
    def determine_decision_type(self, context: Dict[str, Any]) -> DecisionType:
        """Determine decision type based on context and risk assessment"""
        risk_scores = self.risk_assessor.assess_risk(context)
        overall_risk = risk_scores['overall']
        
        if overall_risk >= 0.9:
            return DecisionType.CRITICAL
        elif overall_risk >= 0.6:
            return DecisionType.HIGH_RISK
        elif overall_risk >= 0.3:
            return DecisionType.MEDIUM_RISK
        else:
            return DecisionType.LOW_RISK
    
    def get_applicable_policy(self, decision_type: DecisionType) -> Optional[GovernancePolicy]:
        """Get applicable policy for decision type"""
        for policy in self.policies.values():
            if decision_type in policy.decision_types and policy.is_active:
                return policy
        return None
    
    def check_automated_approval(self, policy: GovernancePolicy, 
                                context: Dict[str, Any]) -> bool:
        """Check if decision qualifies for automated approval"""
        if not policy.automated_approval_conditions:
            return False
        
        risk_scores = self.risk_assessor.assess_risk(context)
        
        # Check all automated approval conditions
        conditions = policy.automated_approval_conditions
        
        if 'max_risk' in conditions and risk_scores['overall'] > conditions['max_risk']:
            return False
        
        if 'min_confidence' in conditions and context.get('ai_confidence', 0) < conditions['min_confidence']:
            return False
        
        if 'max_bias' in conditions and context.get('bias_score', 0) > conditions['max_bias']:
            return False
        
        return True

class ApprovalWorkflow:
    """Approval workflow management"""
    
    def __init__(self, redis_client: redis.Redis, db_connection: str):
        self.redis_client = redis_client
        self.db_connection = db_connection
        self.policy_engine = PolicyEngine()
        self.pending_requests = {}
        self.approval_records = {}
        
    async def submit_decision_request(self, description: str, context: Dict[str, Any],
                                    requested_by: str) -> DecisionRequest:
        """
        Submit AI decision for governance review
        
        Args:
            description: Description of the decision
            context: Decision context and metadata
            requested_by: ID of the requester
            
        Returns:
            DecisionRequest object
        """
        # Assess risk and determine decision type
        decision_type = self.policy_engine.determine_decision_type(context)
        policy = self.policy_engine.get_applicable_policy(decision_type)
        
        if not policy:
            raise ValueError(f"No applicable policy found for decision type: {decision_type}")
        
        # Create decision request
        request_id = str(uuid.uuid4())
        risk_assessment = self.policy_engine.risk_assessor.assess_risk(context)
        
        decision_request = DecisionRequest(
            request_id=request_id,
            decision_type=decision_type,
            description=description,
            context=context,
            risk_assessment=risk_assessment,
            ai_confidence=context.get('ai_confidence', 0.7),
            requested_by=requested_by,
            requested_at=datetime.utcnow(),
            required_approvers=policy.required_approvers,
            approval_deadline=datetime.utcnow() + policy.escalation_timeout,
            metadata={'policy_id': policy.policy_id}
        )
        
        # Check for automated approval
        if self.policy_engine.check_automated_approval(policy, context):
            await self._auto_approve_request(decision_request)
        else:
            await self._initiate_human_review(decision_request, policy)
        
        # Store request
        self.pending_requests[request_id] = decision_request
        await self._store_request_in_db(decision_request)
        
        # Create audit trail
        await self._create_audit_entry(
            request_id=request_id,
            action='request_submitted',
            actor=requested_by,
            details={'decision_type': decision_type.value, 'risk_score': risk_assessment['overall']}
        )
        
        return decision_request
    
    async def submit_approval(self, request_id: str, approver_id: str, 
                            approver_role: GovernanceRole, status: ApprovalStatus,
                            comments: str = "", evidence: Dict[str, Any] = None) -> ApprovalRecord:
        """
        Submit approval or rejection for a decision request
        
        Args:
            request_id: ID of the decision request
            approver_id: ID of the approver
            approver_role: Role of the approver
            status: Approval status
            comments: Approval comments
            evidence: Supporting evidence
            
        Returns:
            ApprovalRecord object
        """
        if evidence is None:
            evidence = {}
        
        if request_id not in self.pending_requests:
            raise ValueError(f"Request {request_id} not found")
        
        decision_request = self.pending_requests[request_id]
        
        # Verify approver is authorized
        if approver_role not in decision_request.required_approvers:
            raise ValueError(f"Role {approver_role} not authorized to approve this request")
        
        # Create approval record
        approval_id = str(uuid.uuid4())
        approval_record = ApprovalRecord(
            approval_id=approval_id,
            request_id=request_id,
            approver_id=approver_id,
            approver_role=approver_role,
            status=status,
            comments=comments,
            timestamp=datetime.utcnow(),
            evidence=evidence
        )
        
        # Store approval record
        if request_id not in self.approval_records:
            self.approval_records[request_id] = []
        self.approval_records[request_id].append(approval_record)
        
        await self._store_approval_in_db(approval_record)
        
        # Check if decision is complete
        await self._check_decision_completion(request_id)
        
        # Create audit trail
        await self._create_audit_entry(
            request_id=request_id,
            action=f'approval_{status.value}',
            actor=approver_id,
            details={'approver_role': approver_role.value, 'comments': comments}
        )
        
        return approval_record
    
    async def _auto_approve_request(self, decision_request: DecisionRequest):
        """Automatically approve low-risk requests"""
        approval_record = ApprovalRecord(
            approval_id=str(uuid.uuid4()),
            request_id=decision_request.request_id,
            approver_id='system',
            approver_role=GovernanceRole.SYSTEM_ADMIN,
            status=ApprovalStatus.APPROVED,
            comments='Automatically approved based on policy conditions',
            timestamp=datetime.utcnow(),
            evidence={'automated': True, 'policy_id': decision_request.metadata.get('policy_id')}
        )
        
        if decision_request.request_id not in self.approval_records:
            self.approval_records[decision_request.request_id] = []
        self.approval_records[decision_request.request_id].append(approval_record)
        
        await self._store_approval_in_db(approval_record)
        
        # Notify stakeholders
        await self._notify_decision_completion(decision_request.request_id, ApprovalStatus.APPROVED)
    
    async def _initiate_human_review(self, decision_request: DecisionRequest, 
                                   policy: GovernancePolicy):
        """Initiate human review process"""
        # Notify required approvers
        for approver_role in policy.required_approvers:
            await self._notify_approver(decision_request, approver_role)
        
        # Schedule escalation if needed
        await self._schedule_escalation(decision_request.request_id, policy.escalation_timeout)
    
    async def _check_decision_completion(self, request_id: str):
        """Check if decision approval process is complete"""
        decision_request = self.pending_requests[request_id]
        approvals = self.approval_records.get(request_id, [])
        
        # Get applicable policy
        policy = self.policy_engine.get_applicable_policy(decision_request.decision_type)
        if not policy:
            return
        
        # Count approvals and rejections
        approved_count = sum(1 for approval in approvals if approval.status == ApprovalStatus.APPROVED)
        rejected_count = sum(1 for approval in approvals if approval.status == ApprovalStatus.REJECTED)
        
        # Check completion conditions
        if rejected_count > 0:
            # Any rejection completes the process
            await self._complete_decision(request_id, ApprovalStatus.REJECTED)
        elif approved_count >= policy.approval_threshold:
            # Sufficient approvals
            await self._complete_decision(request_id, ApprovalStatus.APPROVED)
        elif datetime.utcnow() > decision_request.approval_deadline:
            # Deadline exceeded
            await self._complete_decision(request_id, ApprovalStatus.EXPIRED)
    
    async def _complete_decision(self, request_id: str, final_status: ApprovalStatus):
        """Complete the decision approval process"""
        decision_request = self.pending_requests[request_id]
        
        # Update request status
        decision_request.metadata['final_status'] = final_status.value
        decision_request.metadata['completed_at'] = datetime.utcnow().isoformat()
        
        # Notify stakeholders
        await self._notify_decision_completion(request_id, final_status)
        
        # Create audit trail
        await self._create_audit_entry(
            request_id=request_id,
            action='decision_completed',
            actor='system',
            details={'final_status': final_status.value}
        )
        
        # Archive request
        await self._archive_request(request_id)
    
    async def _notify_approver(self, decision_request: DecisionRequest, 
                             approver_role: GovernanceRole):
        """Notify approver of pending decision"""
        notification = {
            'type': 'approval_request',
            'request_id': decision_request.request_id,
            'decision_type': decision_request.decision_type.value,
            'description': decision_request.description,
            'risk_score': decision_request.risk_assessment['overall'],
            'deadline': decision_request.approval_deadline.isoformat(),
            'approver_role': approver_role.value
        }
        
        # Store notification in Redis for real-time delivery
        await self._store_notification(approver_role.value, notification)
        
        # Send email notification (if configured)
        await self._send_email_notification(approver_role, notification)
    
    async def _notify_decision_completion(self, request_id: str, final_status: ApprovalStatus):
        """Notify stakeholders of decision completion"""
        decision_request = self.pending_requests[request_id]
        
        notification = {
            'type': 'decision_completed',
            'request_id': request_id,
            'final_status': final_status.value,
            'completed_at': datetime.utcnow().isoformat(),
            'description': decision_request.description
        }
        
        # Notify requester
        await self._store_notification(decision_request.requested_by, notification)
        
        # Notify all approvers
        for approver_role in decision_request.required_approvers:
            await self._store_notification(approver_role.value, notification)
    
    async def _schedule_escalation(self, request_id: str, timeout: timedelta):
        """Schedule escalation for overdue approvals"""
        # This would integrate with a task scheduler like Celery
        escalation_time = datetime.utcnow() + timeout
        
        escalation_task = {
            'request_id': request_id,
            'escalation_time': escalation_time.isoformat(),
            'action': 'escalate_approval'
        }
        
        # Store in Redis with expiration
        self.redis_client.setex(
            f"escalation:{request_id}",
            int(timeout.total_seconds()),
            json.dumps(escalation_task)
        )
    
    async def _store_request_in_db(self, decision_request: DecisionRequest):
        """Store decision request in database"""
        try:
            conn = psycopg2.connect(self.db_connection)
            cursor = conn.cursor()
            
            query = """
                INSERT INTO decision_requests 
                (request_id, decision_type, description, context, risk_assessment, 
                 ai_confidence, requested_by, requested_at, required_approvers, 
                 approval_deadline, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                decision_request.request_id,
                decision_request.decision_type.value,
                decision_request.description,
                json.dumps(decision_request.context),
                json.dumps(decision_request.risk_assessment),
                decision_request.ai_confidence,
                decision_request.requested_by,
                decision_request.requested_at,
                json.dumps([role.value for role in decision_request.required_approvers]),
                decision_request.approval_deadline,
                json.dumps(decision_request.metadata)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store request in database: {e}")
    
    async def _store_approval_in_db(self, approval_record: ApprovalRecord):
        """Store approval record in database"""
        try:
            conn = psycopg2.connect(self.db_connection)
            cursor = conn.cursor()
            
            query = """
                INSERT INTO approval_records 
                (approval_id, request_id, approver_id, approver_role, status, 
                 comments, timestamp, evidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                approval_record.approval_id,
                approval_record.request_id,
                approval_record.approver_id,
                approval_record.approver_role.value,
                approval_record.status.value,
                approval_record.comments,
                approval_record.timestamp,
                json.dumps(approval_record.evidence)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store approval in database: {e}")
    
    async def _create_audit_entry(self, request_id: str, action: str, actor: str,
                                details: Dict[str, Any]):
        """Create audit trail entry"""
        audit_entry = AuditTrail(
            audit_id=str(uuid.uuid4()),
            request_id=request_id,
            action=action,
            actor=actor,
            timestamp=datetime.utcnow(),
            details=details,
            compliance_tags=self._determine_compliance_tags(action, details)
        )
        
        # Store in database
        try:
            conn = psycopg2.connect(self.db_connection)
            cursor = conn.cursor()
            
            query = """
                INSERT INTO audit_trail 
                (audit_id, request_id, action, actor, timestamp, details, compliance_tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                audit_entry.audit_id,
                audit_entry.request_id,
                audit_entry.action,
                audit_entry.actor,
                audit_entry.timestamp,
                json.dumps(audit_entry.details),
                json.dumps(audit_entry.compliance_tags)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store audit entry: {e}")
    
    def _determine_compliance_tags(self, action: str, details: Dict[str, Any]) -> List[str]:
        """Determine compliance tags for audit entry"""
        tags = []
        
        # Add action-based tags
        if 'approval' in action:
            tags.append('approval_process')
        if 'decision' in action:
            tags.append('ai_decision')
        if 'escalation' in action:
            tags.append('escalation')
        
        # Add risk-based tags
        if 'risk_score' in details:
            risk_score = details['risk_score']
            if risk_score > 0.8:
                tags.append('high_risk')
            elif risk_score > 0.5:
                tags.append('medium_risk')
            else:
                tags.append('low_risk')
        
        # Add compliance framework tags
        tags.extend(['gdpr_compliance', 'ai_ethics', 'internal_policy'])
        
        return tags
    
    async def _store_notification(self, recipient: str, notification: Dict[str, Any]):
        """Store notification for delivery"""
        notification_key = f"notifications:{recipient}"
        self.redis_client.lpush(notification_key, json.dumps(notification))
        self.redis_client.ltrim(notification_key, 0, 99)  # Keep last 100 notifications
    
    async def _send_email_notification(self, approver_role: GovernanceRole, 
                                     notification: Dict[str, Any]):
        """Send email notification (placeholder implementation)"""
        # This would integrate with actual email service
        logger.info(f"Email notification sent to {approver_role.value}: {notification['type']}")
    
    async def _archive_request(self, request_id: str):
        """Archive completed request"""
        if request_id in self.pending_requests:
            # Move to archived requests (could be separate storage)
            archived_request = self.pending_requests.pop(request_id)
            # Store in archive table or move to cold storage
            logger.info(f"Request {request_id} archived")

class ComplianceMonitor:
    """Compliance monitoring and reporting"""
    
    def __init__(self, db_connection: str):
        self.db_connection = db_connection
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Load compliance rules for different frameworks"""
        return {
            ComplianceFramework.GDPR: {
                'data_retention_days': 365,
                'consent_required': True,
                'right_to_explanation': True,
                'data_minimization': True,
                'audit_frequency_days': 90
            },
            ComplianceFramework.HIPAA: {
                'data_encryption': True,
                'access_logging': True,
                'minimum_necessary': True,
                'audit_frequency_days': 30
            },
            ComplianceFramework.AI_ETHICS: {
                'bias_monitoring': True,
                'fairness_assessment': True,
                'transparency_required': True,
                'human_oversight': True
            }
        }
    
    async def check_compliance(self, request_id: str) -> Dict[str, Any]:
        """Check compliance for a specific request"""
        compliance_results = {}
        
        # Get request details
        request_data = await self._get_request_data(request_id)
        if not request_data:
            return {'error': 'Request not found'}
        
        # Check each applicable compliance framework
        applicable_frameworks = request_data.get('compliance_frameworks', [])
        
        for framework_name in applicable_frameworks:
            try:
                framework = ComplianceFramework(framework_name)
                compliance_results[framework_name] = await self._check_framework_compliance(
                    framework, request_data
                )
            except ValueError:
                logger.warning(f"Unknown compliance framework: {framework_name}")
        
        return compliance_results
    
    async def _check_framework_compliance(self, framework: ComplianceFramework,
                                        request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance for specific framework"""
        rules = self.compliance_rules.get(framework, {})
        compliance_result = {
            'framework': framework.value,
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        if framework == ComplianceFramework.GDPR:
            # Check GDPR compliance
            if rules.get('consent_required') and not request_data.get('consent_obtained'):
                compliance_result['compliant'] = False
                compliance_result['violations'].append('Missing user consent')
                compliance_result['recommendations'].append('Obtain explicit user consent')
            
            if rules.get('right_to_explanation') and not request_data.get('explanation_available'):
                compliance_result['violations'].append('No explanation mechanism available')
                compliance_result['recommendations'].append('Implement explanation capability')
        
        elif framework == ComplianceFramework.HIPAA:
            # Check HIPAA compliance
            if rules.get('data_encryption') and not request_data.get('data_encrypted'):
                compliance_result['compliant'] = False
                compliance_result['violations'].append('Data not encrypted')
                compliance_result['recommendations'].append('Implement data encryption')
        
        elif framework == ComplianceFramework.AI_ETHICS:
            # Check AI ethics compliance
            if rules.get('bias_monitoring'):
                bias_score = request_data.get('bias_score', 0)
                if bias_score > 0.3:
                    compliance_result['compliant'] = False
                    compliance_result['violations'].append(f'High bias score: {bias_score}')
                    compliance_result['recommendations'].append('Implement bias mitigation')
            
            if rules.get('human_oversight') and request_data.get('automation_level') == 'human_out_loop':
                compliance_result['violations'].append('Insufficient human oversight')
                compliance_result['recommendations'].append('Add human oversight mechanism')
        
        return compliance_result
    
    async def generate_compliance_report(self, start_date: datetime, 
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for date range"""
        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {},
            'framework_details': {},
            'violations': [],
            'recommendations': []
        }
        
        # Get all requests in date range
        requests = await self._get_requests_in_range(start_date, end_date)
        
        # Analyze compliance for each framework
        for framework in ComplianceFramework:
            framework_report = await self._analyze_framework_compliance(
                framework, requests, start_date, end_date
            )
            report['framework_details'][framework.value] = framework_report
        
        # Generate summary
        total_requests = len(requests)
        compliant_requests = sum(1 for req in requests if req.get('compliant', True))
        
        report['summary'] = {
            'total_requests': total_requests,
            'compliant_requests': compliant_requests,
            'compliance_rate': compliant_requests / max(total_requests, 1),
            'violation_count': total_requests - compliant_requests
        }
        
        return report
    
    async def _get_request_data(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get request data from database"""
        try:
            conn = psycopg2.connect(self.db_connection)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT dr.*, 
                       COALESCE(ar.approval_count, 0) as approval_count,
                       COALESCE(ar.rejection_count, 0) as rejection_count
                FROM decision_requests dr
                LEFT JOIN (
                    SELECT request_id,
                           SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approval_count,
                           SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejection_count
                    FROM approval_records
                    GROUP BY request_id
                ) ar ON dr.request_id = ar.request_id
                WHERE dr.request_id = %s
            """
            
            cursor.execute(query, (request_id,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to get request data: {e}")
            return None
    
    async def _get_requests_in_range(self, start_date: datetime, 
                                   end_date: datetime) -> List[Dict[str, Any]]:
        """Get all requests in date range"""
        try:
            conn = psycopg2.connect(self.db_connection)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT * FROM decision_requests
                WHERE requested_at BETWEEN %s AND %s
                ORDER BY requested_at
            """
            
            cursor.execute(query, (start_date, end_date))
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get requests in range: {e}")
            return []
    
    async def _analyze_framework_compliance(self, framework: ComplianceFramework,
                                          requests: List[Dict[str, Any]],
                                          start_date: datetime, 
                                          end_date: datetime) -> Dict[str, Any]:
        """Analyze compliance for specific framework"""
        framework_requests = [
            req for req in requests 
            if framework.value in req.get('compliance_frameworks', [])
        ]
        
        total_requests = len(framework_requests)
        violations = []
        
        for request in framework_requests:
            compliance_check = await self._check_framework_compliance(framework, request)
            if not compliance_check['compliant']:
                violations.extend(compliance_check['violations'])
        
        return {
            'framework': framework.value,
            'total_requests': total_requests,
            'violation_count': len(violations),
            'compliance_rate': (total_requests - len(violations)) / max(total_requests, 1),
            'common_violations': self._get_common_violations(violations)
        }
    
    def _get_common_violations(self, violations: List[str]) -> List[Dict[str, Any]]:
        """Get most common violations"""
        violation_counts = {}
        for violation in violations:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        return [
            {'violation': violation, 'count': count}
            for violation, count in sorted(violation_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
        ]

class AIGovernanceSystem:
    """Main AI Governance System"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379",
                 db_url: str = "postgresql://localhost/nexus_governance"):
        self.redis_client = redis.from_url(redis_url)
        self.db_url = db_url
        self.approval_workflow = ApprovalWorkflow(self.redis_client, db_url)
        self.compliance_monitor = ComplianceMonitor(db_url)
        
        # Initialize database tables
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_requests (
                    request_id VARCHAR(255) PRIMARY KEY,
                    decision_type VARCHAR(50) NOT NULL,
                    description TEXT NOT NULL,
                    context JSONB,
                    risk_assessment JSONB,
                    ai_confidence FLOAT,
                    requested_by VARCHAR(255) NOT NULL,
                    requested_at TIMESTAMP NOT NULL,
                    required_approvers JSONB,
                    approval_deadline TIMESTAMP NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS approval_records (
                    approval_id VARCHAR(255) PRIMARY KEY,
                    request_id VARCHAR(255) REFERENCES decision_requests(request_id),
                    approver_id VARCHAR(255) NOT NULL,
                    approver_role VARCHAR(50) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    comments TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    evidence JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    audit_id VARCHAR(255) PRIMARY KEY,
                    request_id VARCHAR(255) REFERENCES decision_requests(request_id),
                    action VARCHAR(100) NOT NULL,
                    actor VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    details JSONB,
                    compliance_tags JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision_requests_requested_at ON decision_requests(requested_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_approval_records_request_id ON approval_records(request_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_trail_request_id ON audit_trail(request_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON audit_trail(timestamp)")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    async def request_decision_approval(self, description: str, context: Dict[str, Any],
                                      requested_by: str) -> Dict[str, Any]:
        """Request approval for AI decision"""
        try:
            decision_request = await self.approval_workflow.submit_decision_request(
                description, context, requested_by
            )
            return {
                'request_id': decision_request.request_id,
                'decision_type': decision_request.decision_type.value,
                'risk_assessment': decision_request.risk_assessment,
                'required_approvers': [role.value for role in decision_request.required_approvers],
                'approval_deadline': decision_request.approval_deadline.isoformat(),
                'status': 'submitted'
            }
        except Exception as e:
            logger.error(f"Failed to request decision approval: {e}")
            raise HTTPException(status_code=500, detail="Failed to request approval")
    
    async def submit_approval_decision(self, request_id: str, approver_id: str,
                                     approver_role: str, approved: bool,
                                     comments: str = "") -> Dict[str, Any]:
        """Submit approval or rejection decision"""
        try:
            role = GovernanceRole(approver_role)
            status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            
            approval_record = await self.approval_workflow.submit_approval(
                request_id, approver_id, role, status, comments
            )
            
            return {
                'approval_id': approval_record.approval_id,
                'status': approval_record.status.value,
                'timestamp': approval_record.timestamp.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to submit approval decision: {e}")
            raise HTTPException(status_code=500, detail="Failed to submit approval")
    
    async def get_compliance_status(self, request_id: str) -> Dict[str, Any]:
        """Get compliance status for request"""
        try:
            return await self.compliance_monitor.check_compliance(request_id)
        except Exception as e:
            logger.error(f"Failed to get compliance status: {e}")
            raise HTTPException(status_code=500, detail="Failed to get compliance status")
    
    async def generate_governance_report(self, start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate governance and compliance report"""
        try:
            return await self.compliance_monitor.generate_compliance_report(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to generate governance report: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate report")

# FastAPI application
app = FastAPI(title="Nexus Architect AI Governance System", version="1.0.0")
security = HTTPBearer()

# Global governance system instance
governance_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize governance system on startup"""
    global governance_system
    governance_system = AIGovernanceSystem()
    logger.info("AI Governance System started successfully")

# Pydantic models for API
class DecisionApprovalRequest(BaseModel):
    description: str = Field(..., description="Description of the AI decision")
    context: Dict[str, Any] = Field(..., description="Decision context and metadata")
    requested_by: str = Field(..., description="ID of the requester")

class ApprovalSubmissionRequest(BaseModel):
    request_id: str = Field(..., description="ID of the decision request")
    approver_id: str = Field(..., description="ID of the approver")
    approver_role: str = Field(..., description="Role of the approver")
    approved: bool = Field(..., description="Whether the decision is approved")
    comments: str = Field(default="", description="Approval comments")

@app.post("/governance/request-approval")
async def request_approval(request: DecisionApprovalRequest,
                         credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Request approval for AI decision"""
    return await governance_system.request_decision_approval(
        request.description, request.context, request.requested_by
    )

@app.post("/governance/submit-approval")
async def submit_approval(request: ApprovalSubmissionRequest,
                        credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Submit approval or rejection decision"""
    return await governance_system.submit_approval_decision(
        request.request_id, request.approver_id, request.approver_role,
        request.approved, request.comments
    )

@app.get("/governance/compliance/{request_id}")
async def get_compliance(request_id: str,
                       credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get compliance status for request"""
    return await governance_system.get_compliance_status(request_id)

@app.get("/governance/report")
async def get_governance_report(start_date: str, end_date: str,
                              credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Generate governance and compliance report"""
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        return await governance_system.generate_governance_report(start_dt, end_dt)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

@app.get("/governance/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

