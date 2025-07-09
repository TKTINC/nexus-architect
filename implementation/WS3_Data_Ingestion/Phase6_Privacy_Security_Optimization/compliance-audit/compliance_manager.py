"""
Compliance Manager for Nexus Architect
Implements comprehensive compliance framework for GDPR, CCPA, HIPAA, and SOC 2
with automated audit procedures and reporting.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import secrets
import hashlib
from collections import defaultdict
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"

class AuditResult(Enum):
    """Audit result types"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    INFO = "info"

class DataProcessingLawfulness(Enum):
    """GDPR Article 6 lawful bases for processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    requirement_id: str
    framework: ComplianceFramework
    category: str
    title: str
    description: str
    mandatory: bool
    implementation_status: ComplianceStatus
    evidence_required: List[str]
    responsible_party: str
    due_date: Optional[datetime] = None
    last_reviewed: Optional[datetime] = None
    notes: str = ""

@dataclass
class AuditFinding:
    """Audit finding or issue"""
    finding_id: str
    audit_id: str
    requirement_id: str
    severity: str  # critical, high, medium, low
    result: AuditResult
    title: str
    description: str
    evidence: List[str]
    remediation_steps: List[str]
    responsible_party: str
    due_date: Optional[datetime] = None
    status: str = "open"  # open, in_progress, resolved, closed
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DataProcessingActivity:
    """GDPR Article 30 record of processing activities"""
    activity_id: str
    name: str
    description: str
    controller: str
    processor: Optional[str] = None
    data_subjects: Set[str] = field(default_factory=set)
    personal_data_categories: Set[str] = field(default_factory=set)
    processing_purposes: Set[str] = field(default_factory=set)
    lawful_basis: DataProcessingLawfulness = DataProcessingLawfulness.CONSENT
    recipients: Set[str] = field(default_factory=set)
    third_country_transfers: Set[str] = field(default_factory=set)
    retention_period: Optional[timedelta] = None
    security_measures: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ComplianceAudit:
    """Compliance audit session"""
    audit_id: str
    framework: ComplianceFramework
    audit_type: str  # internal, external, self_assessment
    auditor: str
    scope: str
    start_date: datetime
    end_date: Optional[datetime] = None
    status: str = "in_progress"  # planned, in_progress, completed, cancelled
    findings: List[AuditFinding] = field(default_factory=list)
    overall_score: Optional[float] = None
    compliance_percentage: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)

class ComplianceManager:
    """
    Comprehensive compliance manager implementing enterprise-grade
    compliance frameworks and automated audit procedures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the compliance manager"""
        self.config = config
        self.database_config = config.get('database', {})
        
        # Compliance requirements and audits
        self.compliance_requirements: Dict[str, ComplianceRequirement] = {}
        self.processing_activities: Dict[str, DataProcessingActivity] = {}
        self.audits: Dict[str, ComplianceAudit] = {}
        self.audit_findings: Dict[str, AuditFinding] = {}
        
        # Initialize compliance frameworks
        self._initialize_compliance_frameworks()
        
        logger.info("Compliance Manager initialized successfully")
    
    def _initialize_compliance_frameworks(self):
        """Initialize compliance requirements for supported frameworks"""
        
        # GDPR Requirements
        gdpr_requirements = [
            {
                "requirement_id": "gdpr_art_6",
                "framework": ComplianceFramework.GDPR,
                "category": "lawfulness",
                "title": "Lawfulness of processing",
                "description": "Processing must have a lawful basis under Article 6",
                "mandatory": True,
                "evidence_required": ["lawful_basis_documentation", "consent_records"],
                "responsible_party": "Data Protection Officer"
            },
            {
                "requirement_id": "gdpr_art_7",
                "framework": ComplianceFramework.GDPR,
                "category": "consent",
                "title": "Conditions for consent",
                "description": "Consent must be freely given, specific, informed and unambiguous",
                "mandatory": True,
                "evidence_required": ["consent_mechanism", "consent_records", "withdrawal_process"],
                "responsible_party": "Data Protection Officer"
            },
            {
                "requirement_id": "gdpr_art_13_14",
                "framework": ComplianceFramework.GDPR,
                "category": "transparency",
                "title": "Information to be provided",
                "description": "Provide transparent information about data processing",
                "mandatory": True,
                "evidence_required": ["privacy_notice", "data_collection_forms"],
                "responsible_party": "Legal Team"
            },
            {
                "requirement_id": "gdpr_art_15",
                "framework": ComplianceFramework.GDPR,
                "category": "rights",
                "title": "Right of access",
                "description": "Data subjects have the right to access their personal data",
                "mandatory": True,
                "evidence_required": ["access_request_process", "response_procedures"],
                "responsible_party": "Data Protection Officer"
            },
            {
                "requirement_id": "gdpr_art_17",
                "framework": ComplianceFramework.GDPR,
                "category": "rights",
                "title": "Right to erasure",
                "description": "Data subjects have the right to erasure of personal data",
                "mandatory": True,
                "evidence_required": ["erasure_procedures", "deletion_logs"],
                "responsible_party": "Technical Team"
            },
            {
                "requirement_id": "gdpr_art_25",
                "framework": ComplianceFramework.GDPR,
                "category": "security",
                "title": "Data protection by design and by default",
                "description": "Implement appropriate technical and organisational measures",
                "mandatory": True,
                "evidence_required": ["privacy_impact_assessments", "security_measures"],
                "responsible_party": "Technical Team"
            },
            {
                "requirement_id": "gdpr_art_30",
                "framework": ComplianceFramework.GDPR,
                "category": "accountability",
                "title": "Records of processing activities",
                "description": "Maintain records of all processing activities",
                "mandatory": True,
                "evidence_required": ["processing_records", "data_inventory"],
                "responsible_party": "Data Protection Officer"
            },
            {
                "requirement_id": "gdpr_art_32",
                "framework": ComplianceFramework.GDPR,
                "category": "security",
                "title": "Security of processing",
                "description": "Implement appropriate technical and organisational measures",
                "mandatory": True,
                "evidence_required": ["security_policies", "encryption_implementation"],
                "responsible_party": "Security Team"
            },
            {
                "requirement_id": "gdpr_art_33_34",
                "framework": ComplianceFramework.GDPR,
                "category": "breach",
                "title": "Personal data breach notification",
                "description": "Notify authorities and data subjects of personal data breaches",
                "mandatory": True,
                "evidence_required": ["breach_procedures", "notification_templates"],
                "responsible_party": "Incident Response Team"
            },
            {
                "requirement_id": "gdpr_art_35",
                "framework": ComplianceFramework.GDPR,
                "category": "assessment",
                "title": "Data protection impact assessment",
                "description": "Conduct DPIA for high-risk processing",
                "mandatory": True,
                "evidence_required": ["dpia_procedures", "risk_assessments"],
                "responsible_party": "Data Protection Officer"
            }
        ]
        
        # CCPA Requirements
        ccpa_requirements = [
            {
                "requirement_id": "ccpa_1798_100",
                "framework": ComplianceFramework.CCPA,
                "category": "transparency",
                "title": "Consumer right to know",
                "description": "Inform consumers about personal information collection and use",
                "mandatory": True,
                "evidence_required": ["privacy_policy", "collection_notices"],
                "responsible_party": "Legal Team"
            },
            {
                "requirement_id": "ccpa_1798_105",
                "framework": ComplianceFramework.CCPA,
                "category": "rights",
                "title": "Consumer right to delete",
                "description": "Allow consumers to request deletion of personal information",
                "mandatory": True,
                "evidence_required": ["deletion_procedures", "request_forms"],
                "responsible_party": "Technical Team"
            },
            {
                "requirement_id": "ccpa_1798_110",
                "framework": ComplianceFramework.CCPA,
                "category": "rights",
                "title": "Consumer right to access",
                "description": "Provide consumers access to their personal information",
                "mandatory": True,
                "evidence_required": ["access_procedures", "data_portability"],
                "responsible_party": "Data Protection Officer"
            },
            {
                "requirement_id": "ccpa_1798_120",
                "framework": ComplianceFramework.CCPA,
                "category": "rights",
                "title": "Consumer right to opt-out",
                "description": "Allow consumers to opt-out of sale of personal information",
                "mandatory": True,
                "evidence_required": ["opt_out_mechanisms", "do_not_sell_links"],
                "responsible_party": "Marketing Team"
            }
        ]
        
        # HIPAA Requirements
        hipaa_requirements = [
            {
                "requirement_id": "hipaa_164_502",
                "framework": ComplianceFramework.HIPAA,
                "category": "privacy",
                "title": "Uses and disclosures of PHI",
                "description": "Limit uses and disclosures of protected health information",
                "mandatory": True,
                "evidence_required": ["privacy_policies", "disclosure_logs"],
                "responsible_party": "Privacy Officer"
            },
            {
                "requirement_id": "hipaa_164_506",
                "framework": ComplianceFramework.HIPAA,
                "category": "consent",
                "title": "Consent for uses and disclosures",
                "description": "Obtain consent for uses and disclosures of PHI",
                "mandatory": True,
                "evidence_required": ["consent_forms", "authorization_procedures"],
                "responsible_party": "Privacy Officer"
            },
            {
                "requirement_id": "hipaa_164_308",
                "framework": ComplianceFramework.HIPAA,
                "category": "security",
                "title": "Administrative safeguards",
                "description": "Implement administrative safeguards for ePHI",
                "mandatory": True,
                "evidence_required": ["security_policies", "workforce_training"],
                "responsible_party": "Security Officer"
            },
            {
                "requirement_id": "hipaa_164_310",
                "framework": ComplianceFramework.HIPAA,
                "category": "security",
                "title": "Physical safeguards",
                "description": "Implement physical safeguards for ePHI",
                "mandatory": True,
                "evidence_required": ["facility_access_controls", "workstation_security"],
                "responsible_party": "Security Officer"
            },
            {
                "requirement_id": "hipaa_164_312",
                "framework": ComplianceFramework.HIPAA,
                "category": "security",
                "title": "Technical safeguards",
                "description": "Implement technical safeguards for ePHI",
                "mandatory": True,
                "evidence_required": ["access_controls", "encryption_implementation"],
                "responsible_party": "Technical Team"
            }
        ]
        
        # SOC 2 Requirements
        soc2_requirements = [
            {
                "requirement_id": "soc2_cc1",
                "framework": ComplianceFramework.SOC2,
                "category": "control_environment",
                "title": "Control Environment",
                "description": "Establish control environment for effective internal control",
                "mandatory": True,
                "evidence_required": ["governance_policies", "organizational_structure"],
                "responsible_party": "Management"
            },
            {
                "requirement_id": "soc2_cc2",
                "framework": ComplianceFramework.SOC2,
                "category": "communication",
                "title": "Communication and Information",
                "description": "Communicate information to support internal control",
                "mandatory": True,
                "evidence_required": ["communication_policies", "information_systems"],
                "responsible_party": "Management"
            },
            {
                "requirement_id": "soc2_cc3",
                "framework": ComplianceFramework.SOC2,
                "category": "risk_assessment",
                "title": "Risk Assessment",
                "description": "Specify objectives and identify and analyze risks",
                "mandatory": True,
                "evidence_required": ["risk_assessments", "risk_management_procedures"],
                "responsible_party": "Risk Management"
            },
            {
                "requirement_id": "soc2_cc4",
                "framework": ComplianceFramework.SOC2,
                "category": "monitoring",
                "title": "Monitoring Activities",
                "description": "Monitor system of internal control",
                "mandatory": True,
                "evidence_required": ["monitoring_procedures", "performance_metrics"],
                "responsible_party": "Internal Audit"
            },
            {
                "requirement_id": "soc2_cc5",
                "framework": ComplianceFramework.SOC2,
                "category": "control_activities",
                "title": "Control Activities",
                "description": "Select and develop control activities",
                "mandatory": True,
                "evidence_required": ["control_procedures", "segregation_of_duties"],
                "responsible_party": "Operations"
            }
        ]
        
        # Add all requirements
        all_requirements = gdpr_requirements + ccpa_requirements + hipaa_requirements + soc2_requirements
        
        for req_data in all_requirements:
            requirement = ComplianceRequirement(
                requirement_id=req_data["requirement_id"],
                framework=req_data["framework"],
                category=req_data["category"],
                title=req_data["title"],
                description=req_data["description"],
                mandatory=req_data["mandatory"],
                implementation_status=ComplianceStatus.UNDER_REVIEW,
                evidence_required=req_data["evidence_required"],
                responsible_party=req_data["responsible_party"]
            )
            
            self.compliance_requirements[requirement.requirement_id] = requirement
        
        logger.info(f"Initialized {len(all_requirements)} compliance requirements")
    
    async def create_processing_activity(self, activity_data: Dict[str, Any]) -> DataProcessingActivity:
        """
        Create a new data processing activity record (GDPR Article 30)
        
        Args:
            activity_data: Processing activity information
            
        Returns:
            Created processing activity
        """
        try:
            activity_id = activity_data.get('activity_id') or secrets.token_hex(8)
            
            activity = DataProcessingActivity(
                activity_id=activity_id,
                name=activity_data['name'],
                description=activity_data['description'],
                controller=activity_data['controller'],
                processor=activity_data.get('processor'),
                data_subjects=set(activity_data.get('data_subjects', [])),
                personal_data_categories=set(activity_data.get('personal_data_categories', [])),
                processing_purposes=set(activity_data.get('processing_purposes', [])),
                lawful_basis=DataProcessingLawfulness(activity_data.get('lawful_basis', 'consent')),
                recipients=set(activity_data.get('recipients', [])),
                third_country_transfers=set(activity_data.get('third_country_transfers', [])),
                retention_period=activity_data.get('retention_period'),
                security_measures=set(activity_data.get('security_measures', []))
            )
            
            self.processing_activities[activity_id] = activity
            logger.info(f"Created processing activity: {activity_id}")
            
            return activity
            
        except Exception as e:
            logger.error(f"Error creating processing activity: {str(e)}")
            raise
    
    async def conduct_compliance_audit(self, framework: ComplianceFramework, 
                                     auditor: str, scope: str = "full") -> ComplianceAudit:
        """
        Conduct a compliance audit for specified framework
        
        Args:
            framework: Compliance framework to audit
            auditor: Name of auditor
            scope: Audit scope
            
        Returns:
            Compliance audit results
        """
        try:
            audit_id = secrets.token_hex(16)
            
            audit = ComplianceAudit(
                audit_id=audit_id,
                framework=framework,
                audit_type="internal",
                auditor=auditor,
                scope=scope,
                start_date=datetime.utcnow()
            )
            
            # Get requirements for this framework
            framework_requirements = [
                req for req in self.compliance_requirements.values()
                if req.framework == framework
            ]
            
            findings = []
            compliant_count = 0
            total_count = len(framework_requirements)
            
            for requirement in framework_requirements:
                # Simulate audit check (in production, this would involve actual verification)
                finding = await self._audit_requirement(audit_id, requirement)
                findings.append(finding)
                
                if finding.result == AuditResult.PASS:
                    compliant_count += 1
            
            audit.findings = findings
            audit.compliance_percentage = (compliant_count / total_count * 100) if total_count > 0 else 0
            audit.overall_score = audit.compliance_percentage / 100
            audit.end_date = datetime.utcnow()
            audit.status = "completed"
            
            # Generate recommendations
            audit.recommendations = await self._generate_recommendations(findings)
            
            self.audits[audit_id] = audit
            
            # Store findings
            for finding in findings:
                self.audit_findings[finding.finding_id] = finding
            
            logger.info(f"Completed {framework.value} audit: {audit.compliance_percentage:.1f}% compliant")
            
            return audit
            
        except Exception as e:
            logger.error(f"Error conducting compliance audit: {str(e)}")
            raise
    
    async def _audit_requirement(self, audit_id: str, requirement: ComplianceRequirement) -> AuditFinding:
        """Audit a specific compliance requirement"""
        try:
            finding_id = secrets.token_hex(16)
            
            # Simulate audit logic (in production, this would involve actual checks)
            # For demonstration, we'll use some basic heuristics
            
            if requirement.implementation_status == ComplianceStatus.COMPLIANT:
                result = AuditResult.PASS
                severity = "low"
                description = f"Requirement {requirement.title} is properly implemented"
                remediation_steps = []
            elif requirement.implementation_status == ComplianceStatus.PARTIALLY_COMPLIANT:
                result = AuditResult.WARNING
                severity = "medium"
                description = f"Requirement {requirement.title} is partially implemented"
                remediation_steps = [
                    "Complete implementation of missing controls",
                    "Provide additional evidence of compliance",
                    "Update documentation and procedures"
                ]
            else:
                result = AuditResult.FAIL
                severity = "high" if requirement.mandatory else "medium"
                description = f"Requirement {requirement.title} is not implemented"
                remediation_steps = [
                    "Implement required controls and procedures",
                    "Assign responsible party for implementation",
                    "Set target completion date",
                    "Provide evidence of implementation"
                ]
            
            finding = AuditFinding(
                finding_id=finding_id,
                audit_id=audit_id,
                requirement_id=requirement.requirement_id,
                severity=severity,
                result=result,
                title=f"Audit of {requirement.title}",
                description=description,
                evidence=[],  # Would be populated with actual evidence
                remediation_steps=remediation_steps,
                responsible_party=requirement.responsible_party
            )
            
            return finding
            
        except Exception as e:
            logger.error(f"Error auditing requirement: {str(e)}")
            raise
    
    async def _generate_recommendations(self, findings: List[AuditFinding]) -> List[str]:
        """Generate recommendations based on audit findings"""
        try:
            recommendations = []
            
            # Count findings by severity
            critical_count = sum(1 for f in findings if f.severity == "critical")
            high_count = sum(1 for f in findings if f.severity == "high")
            medium_count = sum(1 for f in findings if f.severity == "medium")
            
            if critical_count > 0:
                recommendations.append(f"Address {critical_count} critical findings immediately")
                recommendations.append("Conduct emergency response procedures for critical issues")
            
            if high_count > 0:
                recommendations.append(f"Prioritize resolution of {high_count} high-severity findings")
                recommendations.append("Assign dedicated resources for high-priority remediation")
            
            if medium_count > 0:
                recommendations.append(f"Plan remediation for {medium_count} medium-severity findings")
            
            # Framework-specific recommendations
            failed_findings = [f for f in findings if f.result == AuditResult.FAIL]
            if failed_findings:
                categories = set(self.compliance_requirements[f.requirement_id].category 
                               for f in failed_findings)
                
                if "security" in categories:
                    recommendations.append("Strengthen security controls and technical safeguards")
                
                if "privacy" in categories:
                    recommendations.append("Enhance privacy protection measures and procedures")
                
                if "consent" in categories:
                    recommendations.append("Improve consent management and documentation")
                
                if "rights" in categories:
                    recommendations.append("Implement data subject rights fulfillment procedures")
            
            # General recommendations
            recommendations.extend([
                "Conduct regular compliance training for all staff",
                "Implement continuous monitoring and assessment procedures",
                "Establish clear accountability and governance structures",
                "Document all compliance activities and maintain evidence"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    async def update_requirement_status(self, requirement_id: str, 
                                      status: ComplianceStatus, notes: str = "") -> bool:
        """
        Update the implementation status of a compliance requirement
        
        Args:
            requirement_id: Requirement identifier
            status: New compliance status
            notes: Optional notes
            
        Returns:
            Success status
        """
        try:
            if requirement_id not in self.compliance_requirements:
                logger.error(f"Requirement not found: {requirement_id}")
                return False
            
            requirement = self.compliance_requirements[requirement_id]
            requirement.implementation_status = status
            requirement.last_reviewed = datetime.utcnow()
            requirement.notes = notes
            
            logger.info(f"Updated requirement {requirement_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating requirement status: {str(e)}")
            return False
    
    async def generate_compliance_report(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report for framework
        
        Args:
            framework: Compliance framework
            
        Returns:
            Compliance report
        """
        try:
            # Get requirements for framework
            framework_requirements = [
                req for req in self.compliance_requirements.values()
                if req.framework == framework
            ]
            
            # Calculate compliance statistics
            total_requirements = len(framework_requirements)
            compliant = sum(1 for req in framework_requirements 
                          if req.implementation_status == ComplianceStatus.COMPLIANT)
            partially_compliant = sum(1 for req in framework_requirements 
                                    if req.implementation_status == ComplianceStatus.PARTIALLY_COMPLIANT)
            non_compliant = sum(1 for req in framework_requirements 
                              if req.implementation_status == ComplianceStatus.NON_COMPLIANT)
            under_review = sum(1 for req in framework_requirements 
                             if req.implementation_status == ComplianceStatus.UNDER_REVIEW)
            
            compliance_percentage = (compliant / total_requirements * 100) if total_requirements > 0 else 0
            
            # Get recent audits
            recent_audits = [
                audit for audit in self.audits.values()
                if audit.framework == framework and 
                audit.start_date > datetime.utcnow() - timedelta(days=365)
            ]
            
            # Get processing activities (for GDPR)
            processing_activities_count = len(self.processing_activities) if framework == ComplianceFramework.GDPR else 0
            
            report = {
                "framework": framework.value,
                "report_date": datetime.utcnow().isoformat(),
                "compliance_summary": {
                    "total_requirements": total_requirements,
                    "compliant": compliant,
                    "partially_compliant": partially_compliant,
                    "non_compliant": non_compliant,
                    "under_review": under_review,
                    "compliance_percentage": round(compliance_percentage, 2)
                },
                "requirements_by_category": self._get_requirements_by_category(framework_requirements),
                "recent_audits": len(recent_audits),
                "processing_activities": processing_activities_count,
                "recommendations": await self._get_framework_recommendations(framework),
                "next_steps": await self._get_next_steps(framework_requirements)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {str(e)}")
            return {"error": str(e)}
    
    def _get_requirements_by_category(self, requirements: List[ComplianceRequirement]) -> Dict[str, Dict[str, int]]:
        """Group requirements by category and status"""
        categories = defaultdict(lambda: defaultdict(int))
        
        for req in requirements:
            categories[req.category][req.implementation_status.value] += 1
        
        return dict(categories)
    
    async def _get_framework_recommendations(self, framework: ComplianceFramework) -> List[str]:
        """Get framework-specific recommendations"""
        recommendations = []
        
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Conduct Data Protection Impact Assessments for high-risk processing",
                "Implement privacy by design and by default principles",
                "Establish clear lawful bases for all processing activities",
                "Provide comprehensive privacy notices to data subjects",
                "Implement data subject rights fulfillment procedures"
            ])
        elif framework == ComplianceFramework.CCPA:
            recommendations.extend([
                "Implement consumer rights request procedures",
                "Provide clear opt-out mechanisms for data sales",
                "Update privacy policies with CCPA-required disclosures",
                "Establish data inventory and mapping procedures"
            ])
        elif framework == ComplianceFramework.HIPAA:
            recommendations.extend([
                "Implement comprehensive administrative safeguards",
                "Establish physical security controls for PHI",
                "Deploy technical safeguards including encryption",
                "Conduct regular risk assessments and security evaluations"
            ])
        elif framework == ComplianceFramework.SOC2:
            recommendations.extend([
                "Establish strong control environment and governance",
                "Implement comprehensive risk management procedures",
                "Deploy continuous monitoring and assessment controls",
                "Maintain detailed documentation of all control activities"
            ])
        
        return recommendations
    
    async def _get_next_steps(self, requirements: List[ComplianceRequirement]) -> List[str]:
        """Get next steps based on requirement status"""
        next_steps = []
        
        non_compliant = [req for req in requirements 
                        if req.implementation_status == ComplianceStatus.NON_COMPLIANT]
        under_review = [req for req in requirements 
                       if req.implementation_status == ComplianceStatus.UNDER_REVIEW]
        
        if non_compliant:
            next_steps.append(f"Address {len(non_compliant)} non-compliant requirements")
            next_steps.append("Prioritize mandatory requirements for immediate implementation")
        
        if under_review:
            next_steps.append(f"Complete review of {len(under_review)} requirements")
            next_steps.append("Gather evidence and documentation for requirements under review")
        
        next_steps.extend([
            "Schedule regular compliance assessments and audits",
            "Provide compliance training to relevant staff",
            "Establish ongoing monitoring and maintenance procedures"
        ])
        
        return next_steps
    
    async def get_compliance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive compliance metrics
        
        Returns:
            Compliance metrics dictionary
        """
        try:
            metrics = {
                "total_requirements": len(self.compliance_requirements),
                "total_audits": len(self.audits),
                "total_findings": len(self.audit_findings),
                "processing_activities": len(self.processing_activities),
                "frameworks": {}
            }
            
            # Calculate metrics by framework
            for framework in ComplianceFramework:
                framework_requirements = [
                    req for req in self.compliance_requirements.values()
                    if req.framework == framework
                ]
                
                if framework_requirements:
                    compliant = sum(1 for req in framework_requirements 
                                  if req.implementation_status == ComplianceStatus.COMPLIANT)
                    total = len(framework_requirements)
                    
                    metrics["frameworks"][framework.value] = {
                        "total_requirements": total,
                        "compliant": compliant,
                        "compliance_percentage": round((compliant / total * 100), 2) if total > 0 else 0
                    }
            
            metrics["timestamp"] = datetime.utcnow().isoformat()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting compliance metrics: {str(e)}")
            return {}

def create_compliance_api(compliance_manager: ComplianceManager):
    """Create Flask API for compliance management"""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "service": "compliance_manager"})
    
    @app.route('/processing-activities', methods=['POST'])
    async def create_processing_activity():
        try:
            activity_data = request.get_json()
            activity = await compliance_manager.create_processing_activity(activity_data)
            
            return jsonify({
                "status": "success",
                "activity_id": activity.activity_id,
                "message": "Processing activity created successfully"
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/audit/<framework>', methods=['POST'])
    async def conduct_audit(framework):
        try:
            data = request.get_json()
            auditor = data.get('auditor', 'System')
            scope = data.get('scope', 'full')
            
            framework_enum = ComplianceFramework(framework)
            audit = await compliance_manager.conduct_compliance_audit(framework_enum, auditor, scope)
            
            return jsonify({
                "status": "success",
                "audit_id": audit.audit_id,
                "compliance_percentage": audit.compliance_percentage,
                "findings_count": len(audit.findings),
                "recommendations": audit.recommendations
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/requirements/<requirement_id>/status', methods=['PUT'])
    async def update_requirement_status(requirement_id):
        try:
            data = request.get_json()
            status = ComplianceStatus(data.get('status'))
            notes = data.get('notes', '')
            
            success = await compliance_manager.update_requirement_status(requirement_id, status, notes)
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "Requirement status updated successfully"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to update requirement status"
                }), 400
                
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/report/<framework>', methods=['GET'])
    async def generate_compliance_report(framework):
        try:
            framework_enum = ComplianceFramework(framework)
            report = await compliance_manager.generate_compliance_report(framework_enum)
            
            return jsonify(report)
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/metrics', methods=['GET'])
    async def get_compliance_metrics():
        try:
            metrics = await compliance_manager.get_compliance_metrics()
            return jsonify(metrics)
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    return app

if __name__ == "__main__":
    # Example configuration
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'nexus_architect',
            'user': 'postgres',
            'password': 'nexus_secure_password_2024'
        }
    }
    
    # Initialize compliance manager
    compliance_manager = ComplianceManager(config)
    
    # Create Flask API
    app = create_compliance_api(compliance_manager)
    
    print("Compliance Manager API starting...")
    print("Available endpoints:")
    print("  POST /processing-activities - Create processing activity")
    print("  POST /audit/<framework> - Conduct compliance audit")
    print("  PUT /requirements/<id>/status - Update requirement status")
    print("  GET /report/<framework> - Generate compliance report")
    print("  GET /metrics - Get compliance metrics")
    
    app.run(host='0.0.0.0', port=8012, debug=False)

