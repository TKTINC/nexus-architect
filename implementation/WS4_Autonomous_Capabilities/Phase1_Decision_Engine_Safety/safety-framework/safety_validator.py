"""
Safety Framework and Validator for Nexus Architect Autonomous Decisions
Implements comprehensive safety validation layers and approval workflows
"""

import ast
import re
import json
import hashlib
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import yaml
from pathlib import Path
import bandit
from bandit.core import manager as bandit_manager
from bandit.core import config as bandit_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Validation result types"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    REQUIRES_REVIEW = "requires_review"

class ApprovalLevel(Enum):
    """Approval levels for decisions"""
    AUTOMATIC = "automatic"
    PEER_REVIEW = "peer_review"
    SENIOR_APPROVAL = "senior_approval"
    EXECUTIVE_APPROVAL = "executive_approval"
    EMERGENCY_OVERRIDE = "emergency_override"

@dataclass
class ValidationCheck:
    """Individual validation check result"""
    check_name: str
    result: ValidationResult
    message: str
    details: Dict[str, Any]
    severity: str  # low, medium, high, critical
    timestamp: datetime

@dataclass
class SafetyAssessment:
    """Complete safety assessment result"""
    decision_id: str
    overall_result: ValidationResult
    approval_level: ApprovalLevel
    validation_checks: List[ValidationCheck]
    risk_score: float
    safety_score: float
    recommendations: List[str]
    blocking_issues: List[str]
    timestamp: datetime

@dataclass
class ApprovalWorkflow:
    """Approval workflow configuration"""
    workflow_id: str
    name: str
    description: str
    required_approvers: List[str]
    approval_timeout: int  # hours
    escalation_rules: Dict[str, Any]
    emergency_override_allowed: bool

class SyntaxValidator:
    """Validates syntax and semantic correctness of code changes"""
    
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'typescript', 'java', 'go']
        
    def validate_python_syntax(self, code: str) -> ValidationCheck:
        """Validate Python code syntax"""
        try:
            ast.parse(code)
            return ValidationCheck(
                check_name="python_syntax",
                result=ValidationResult.PASS,
                message="Python syntax is valid",
                details={"language": "python", "lines": len(code.split('\n'))},
                severity="low",
                timestamp=datetime.now()
            )
        except SyntaxError as e:
            return ValidationCheck(
                check_name="python_syntax",
                result=ValidationResult.FAIL,
                message=f"Python syntax error: {str(e)}",
                details={"error": str(e), "line": e.lineno, "offset": e.offset},
                severity="high",
                timestamp=datetime.now()
            )
    
    def validate_javascript_syntax(self, code: str) -> ValidationCheck:
        """Validate JavaScript/TypeScript syntax using Node.js"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Use Node.js to check syntax
            result = subprocess.run(
                ['node', '--check', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return ValidationCheck(
                    check_name="javascript_syntax",
                    result=ValidationResult.PASS,
                    message="JavaScript syntax is valid",
                    details={"language": "javascript", "lines": len(code.split('\n'))},
                    severity="low",
                    timestamp=datetime.now()
                )
            else:
                return ValidationCheck(
                    check_name="javascript_syntax",
                    result=ValidationResult.FAIL,
                    message=f"JavaScript syntax error: {result.stderr}",
                    details={"error": result.stderr},
                    severity="high",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return ValidationCheck(
                check_name="javascript_syntax",
                result=ValidationResult.WARNING,
                message=f"Could not validate JavaScript syntax: {str(e)}",
                details={"error": str(e)},
                severity="medium",
                timestamp=datetime.now()
            )
    
    def validate_code_quality(self, code: str, language: str) -> List[ValidationCheck]:
        """Validate code quality metrics"""
        checks = []
        
        # Check for common code smells
        if language == 'python':
            checks.extend(self._check_python_quality(code))
        elif language in ['javascript', 'typescript']:
            checks.extend(self._check_javascript_quality(code))
        
        return checks
    
    def _check_python_quality(self, code: str) -> List[ValidationCheck]:
        """Check Python-specific code quality"""
        checks = []
        lines = code.split('\n')
        
        # Check line length
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            checks.append(ValidationCheck(
                check_name="line_length",
                result=ValidationResult.WARNING,
                message=f"Lines exceed 120 characters: {long_lines}",
                details={"long_lines": long_lines, "max_length": 120},
                severity="low",
                timestamp=datetime.now()
            ))
        
        # Check for TODO/FIXME comments
        todo_pattern = re.compile(r'#.*(?:TODO|FIXME|XXX)', re.IGNORECASE)
        todo_lines = [i+1 for i, line in enumerate(lines) if todo_pattern.search(line)]
        if todo_lines:
            checks.append(ValidationCheck(
                check_name="todo_comments",
                result=ValidationResult.WARNING,
                message=f"TODO/FIXME comments found: {todo_lines}",
                details={"todo_lines": todo_lines},
                severity="low",
                timestamp=datetime.now()
            ))
        
        # Check for hardcoded credentials
        credential_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        for pattern in credential_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                checks.append(ValidationCheck(
                    check_name="hardcoded_credentials",
                    result=ValidationResult.FAIL,
                    message=f"Potential hardcoded credential on line {line_num}",
                    details={"line": line_num, "pattern": pattern},
                    severity="critical",
                    timestamp=datetime.now()
                ))
        
        return checks
    
    def _check_javascript_quality(self, code: str) -> List[ValidationCheck]:
        """Check JavaScript-specific code quality"""
        checks = []
        lines = code.split('\n')
        
        # Check for console.log statements
        console_lines = [i+1 for i, line in enumerate(lines) if 'console.log' in line]
        if console_lines:
            checks.append(ValidationCheck(
                check_name="console_statements",
                result=ValidationResult.WARNING,
                message=f"Console.log statements found: {console_lines}",
                details={"console_lines": console_lines},
                severity="low",
                timestamp=datetime.now()
            ))
        
        # Check for eval usage
        eval_lines = [i+1 for i, line in enumerate(lines) if re.search(r'\beval\s*\(', line)]
        if eval_lines:
            checks.append(ValidationCheck(
                check_name="eval_usage",
                result=ValidationResult.FAIL,
                message=f"Dangerous eval() usage found: {eval_lines}",
                details={"eval_lines": eval_lines},
                severity="critical",
                timestamp=datetime.now()
            ))
        
        return checks

class SecurityValidator:
    """Validates security aspects of code changes and configurations"""
    
    def __init__(self):
        self.bandit_manager = None
        self.security_patterns = self._load_security_patterns()
        
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load security vulnerability patterns"""
        return {
            "sql_injection": [
                r"execute\s*\(\s*[\"'].*%.*[\"']\s*%",
                r"cursor\.execute\s*\(\s*[\"'].*\+.*[\"']",
                r"query\s*=\s*[\"'].*\+.*[\"']"
            ],
            "xss_vulnerability": [
                r"innerHTML\s*=\s*.*\+",
                r"document\.write\s*\(\s*.*\+",
                r"eval\s*\(\s*.*user"
            ],
            "path_traversal": [
                r"open\s*\(\s*.*\+.*[\"']\.\.\/",
                r"file\s*=\s*.*\+.*[\"']\.\.\/",
                r"path\s*=\s*.*\+.*[\"']\.\.\/"
            ],
            "command_injection": [
                r"os\.system\s*\(\s*.*\+",
                r"subprocess\.\w+\s*\(\s*.*\+",
                r"exec\s*\(\s*.*user"
            ]
        }
    
    def validate_security(self, code: str, language: str) -> List[ValidationCheck]:
        """Perform comprehensive security validation"""
        checks = []
        
        # Pattern-based security checks
        checks.extend(self._check_security_patterns(code))
        
        # Language-specific security checks
        if language == 'python':
            checks.extend(self._check_python_security(code))
        elif language in ['javascript', 'typescript']:
            checks.extend(self._check_javascript_security(code))
        
        # Dependency security checks
        checks.extend(self._check_dependency_security(code, language))
        
        return checks
    
    def _check_security_patterns(self, code: str) -> List[ValidationCheck]:
        """Check for common security vulnerability patterns"""
        checks = []
        
        for vulnerability_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE))
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    checks.append(ValidationCheck(
                        check_name=f"security_{vulnerability_type}",
                        result=ValidationResult.FAIL,
                        message=f"Potential {vulnerability_type.replace('_', ' ')} vulnerability on line {line_num}",
                        details={
                            "vulnerability_type": vulnerability_type,
                            "line": line_num,
                            "pattern": pattern,
                            "match": match.group()
                        },
                        severity="critical",
                        timestamp=datetime.now()
                    ))
        
        return checks
    
    def _check_python_security(self, code: str) -> List[ValidationCheck]:
        """Python-specific security checks using Bandit"""
        checks = []
        
        try:
            # Create temporary file for Bandit analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run Bandit security analysis
            config = bandit_config.BanditConfig()
            manager = bandit_manager.BanditManager(config, 'file')
            manager.discover_files([temp_file])
            manager.run_tests()
            
            # Process Bandit results
            for issue in manager.get_issue_list():
                checks.append(ValidationCheck(
                    check_name=f"bandit_{issue.test}",
                    result=ValidationResult.FAIL if issue.severity == 'HIGH' else ValidationResult.WARNING,
                    message=f"Bandit security issue: {issue.text}",
                    details={
                        "test": issue.test,
                        "severity": issue.severity,
                        "confidence": issue.confidence,
                        "line": issue.lineno
                    },
                    severity="critical" if issue.severity == 'HIGH' else "medium",
                    timestamp=datetime.now()
                ))
            
            os.unlink(temp_file)
            
        except Exception as e:
            logger.warning(f"Bandit security analysis failed: {e}")
            checks.append(ValidationCheck(
                check_name="bandit_analysis",
                result=ValidationResult.WARNING,
                message=f"Security analysis could not be completed: {str(e)}",
                details={"error": str(e)},
                severity="medium",
                timestamp=datetime.now()
            ))
        
        return checks
    
    def _check_javascript_security(self, code: str) -> List[ValidationCheck]:
        """JavaScript-specific security checks"""
        checks = []
        
        # Check for dangerous functions
        dangerous_functions = ['eval', 'setTimeout', 'setInterval', 'Function']
        for func in dangerous_functions:
            pattern = rf'\b{func}\s*\('
            matches = list(re.finditer(pattern, code))
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                checks.append(ValidationCheck(
                    check_name=f"dangerous_function_{func}",
                    result=ValidationResult.WARNING,
                    message=f"Potentially dangerous function '{func}' used on line {line_num}",
                    details={"function": func, "line": line_num},
                    severity="medium",
                    timestamp=datetime.now()
                ))
        
        return checks
    
    def _check_dependency_security(self, code: str, language: str) -> List[ValidationCheck]:
        """Check for known vulnerable dependencies"""
        checks = []
        
        # Extract imports/requires
        if language == 'python':
            import_pattern = r'(?:from\s+(\S+)\s+import|import\s+(\S+))'
            imports = re.findall(import_pattern, code)
            dependencies = [imp[0] or imp[1] for imp in imports]
        elif language in ['javascript', 'typescript']:
            require_pattern = r'require\s*\(\s*["\']([^"\']+)["\']'
            import_pattern = r'from\s+["\']([^"\']+)["\']'
            dependencies = re.findall(require_pattern, code) + re.findall(import_pattern, code)
        else:
            dependencies = []
        
        # Check against known vulnerable packages (simplified)
        vulnerable_packages = {
            'python': ['pickle', 'cPickle', 'yaml.load'],
            'javascript': ['eval', 'vm', 'child_process']
        }
        
        for dep in dependencies:
            if dep in vulnerable_packages.get(language, []):
                checks.append(ValidationCheck(
                    check_name="vulnerable_dependency",
                    result=ValidationResult.WARNING,
                    message=f"Potentially vulnerable dependency: {dep}",
                    details={"dependency": dep, "language": language},
                    severity="medium",
                    timestamp=datetime.now()
                ))
        
        return checks

class PerformanceValidator:
    """Validates performance impact of changes"""
    
    def __init__(self):
        self.performance_thresholds = {
            "max_complexity": 10,
            "max_nesting_depth": 4,
            "max_function_length": 50,
            "max_class_length": 200
        }
    
    def validate_performance_impact(self, code: str, language: str) -> List[ValidationCheck]:
        """Validate potential performance impact"""
        checks = []
        
        if language == 'python':
            checks.extend(self._check_python_performance(code))
        elif language in ['javascript', 'typescript']:
            checks.extend(self._check_javascript_performance(code))
        
        return checks
    
    def _check_python_performance(self, code: str) -> List[ValidationCheck]:
        """Check Python performance patterns"""
        checks = []
        
        try:
            tree = ast.parse(code)
            
            # Check for nested loops
            nested_loops = self._find_nested_loops(tree)
            if nested_loops > 2:
                checks.append(ValidationCheck(
                    check_name="nested_loops",
                    result=ValidationResult.WARNING,
                    message=f"Deep nested loops detected (depth: {nested_loops})",
                    details={"nesting_depth": nested_loops},
                    severity="medium",
                    timestamp=datetime.now()
                ))
            
            # Check function complexity
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    if complexity > self.performance_thresholds["max_complexity"]:
                        checks.append(ValidationCheck(
                            check_name="high_complexity",
                            result=ValidationResult.WARNING,
                            message=f"Function '{node.name}' has high complexity: {complexity}",
                            details={"function": node.name, "complexity": complexity},
                            severity="medium",
                            timestamp=datetime.now()
                        ))
        
        except Exception as e:
            logger.warning(f"Performance analysis failed: {e}")
        
        return checks
    
    def _find_nested_loops(self, node: ast.AST, depth: int = 0) -> int:
        """Find maximum nesting depth of loops"""
        max_depth = depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                child_depth = self._find_nested_loops(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._find_nested_loops(child, depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _check_javascript_performance(self, code: str) -> List[ValidationCheck]:
        """Check JavaScript performance patterns"""
        checks = []
        
        # Check for synchronous operations that could block
        blocking_patterns = [
            r'\.sync\(',
            r'fs\.readFileSync',
            r'fs\.writeFileSync',
            r'while\s*\(\s*true\s*\)'
        ]
        
        for pattern in blocking_patterns:
            matches = list(re.finditer(pattern, code))
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                checks.append(ValidationCheck(
                    check_name="blocking_operation",
                    result=ValidationResult.WARNING,
                    message=f"Potentially blocking operation on line {line_num}",
                    details={"line": line_num, "pattern": pattern},
                    severity="medium",
                    timestamp=datetime.now()
                ))
        
        return checks

class ComplianceValidator:
    """Validates compliance with organizational policies"""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules from configuration"""
        return {
            "coding_standards": {
                "required_headers": ["copyright", "license"],
                "forbidden_patterns": ["print(", "console.log("],
                "required_documentation": True
            },
            "security_policies": {
                "encryption_required": True,
                "authentication_required": True,
                "audit_logging_required": True
            },
            "data_governance": {
                "pii_handling_required": True,
                "data_retention_policy": True,
                "gdpr_compliance": True
            }
        }
    
    def validate_compliance(self, code: str, metadata: Dict[str, Any]) -> List[ValidationCheck]:
        """Validate compliance with organizational policies"""
        checks = []
        
        # Check coding standards
        checks.extend(self._check_coding_standards(code))
        
        # Check security policies
        checks.extend(self._check_security_policies(code, metadata))
        
        # Check data governance
        checks.extend(self._check_data_governance(code, metadata))
        
        return checks
    
    def _check_coding_standards(self, code: str) -> List[ValidationCheck]:
        """Check coding standards compliance"""
        checks = []
        
        # Check for required headers
        required_headers = self.compliance_rules["coding_standards"]["required_headers"]
        for header in required_headers:
            if header.lower() not in code.lower():
                checks.append(ValidationCheck(
                    check_name=f"missing_{header}_header",
                    result=ValidationResult.WARNING,
                    message=f"Missing required {header} header",
                    details={"required_header": header},
                    severity="low",
                    timestamp=datetime.now()
                ))
        
        # Check for forbidden patterns
        forbidden_patterns = self.compliance_rules["coding_standards"]["forbidden_patterns"]
        for pattern in forbidden_patterns:
            if pattern in code:
                line_num = code[:code.find(pattern)].count('\n') + 1
                checks.append(ValidationCheck(
                    check_name="forbidden_pattern",
                    result=ValidationResult.WARNING,
                    message=f"Forbidden pattern '{pattern}' found on line {line_num}",
                    details={"pattern": pattern, "line": line_num},
                    severity="medium",
                    timestamp=datetime.now()
                ))
        
        return checks
    
    def _check_security_policies(self, code: str, metadata: Dict[str, Any]) -> List[ValidationCheck]:
        """Check security policy compliance"""
        checks = []
        
        # Check for encryption usage when handling sensitive data
        if metadata.get("handles_sensitive_data", False):
            encryption_patterns = ["encrypt", "decrypt", "cipher", "hash"]
            has_encryption = any(pattern in code.lower() for pattern in encryption_patterns)
            
            if not has_encryption:
                checks.append(ValidationCheck(
                    check_name="missing_encryption",
                    result=ValidationResult.FAIL,
                    message="Sensitive data handling requires encryption",
                    details={"policy": "encryption_required"},
                    severity="critical",
                    timestamp=datetime.now()
                ))
        
        return checks
    
    def _check_data_governance(self, code: str, metadata: Dict[str, Any]) -> List[ValidationCheck]:
        """Check data governance compliance"""
        checks = []
        
        # Check for PII handling compliance
        pii_patterns = ["email", "phone", "ssn", "credit_card", "personal"]
        has_pii = any(pattern in code.lower() for pattern in pii_patterns)
        
        if has_pii:
            # Check for proper PII handling
            pii_handling_patterns = ["anonymize", "pseudonymize", "encrypt", "mask"]
            has_proper_handling = any(pattern in code.lower() for pattern in pii_handling_patterns)
            
            if not has_proper_handling:
                checks.append(ValidationCheck(
                    check_name="pii_handling_violation",
                    result=ValidationResult.FAIL,
                    message="PII detected without proper handling mechanisms",
                    details={"policy": "pii_handling_required"},
                    severity="critical",
                    timestamp=datetime.now()
                ))
        
        return checks

class ApprovalWorkflowManager:
    """Manages approval workflows for different decision types"""
    
    def __init__(self):
        self.workflows = self._initialize_workflows()
        self.pending_approvals = {}
    
    def _initialize_workflows(self) -> Dict[str, ApprovalWorkflow]:
        """Initialize default approval workflows"""
        return {
            "low_risk": ApprovalWorkflow(
                workflow_id="low_risk",
                name="Low Risk Automatic Approval",
                description="Automatic approval for low-risk changes",
                required_approvers=[],
                approval_timeout=0,
                escalation_rules={},
                emergency_override_allowed=False
            ),
            "medium_risk": ApprovalWorkflow(
                workflow_id="medium_risk",
                name="Peer Review Required",
                description="Requires peer review for medium-risk changes",
                required_approvers=["peer_reviewer"],
                approval_timeout=4,  # 4 hours
                escalation_rules={"timeout_action": "escalate_to_senior"},
                emergency_override_allowed=True
            ),
            "high_risk": ApprovalWorkflow(
                workflow_id="high_risk",
                name="Senior Approval Required",
                description="Requires senior approval for high-risk changes",
                required_approvers=["senior_engineer", "security_team"],
                approval_timeout=8,  # 8 hours
                escalation_rules={"timeout_action": "escalate_to_executive"},
                emergency_override_allowed=True
            ),
            "critical_risk": ApprovalWorkflow(
                workflow_id="critical_risk",
                name="Executive Approval Required",
                description="Requires executive approval for critical changes",
                required_approvers=["engineering_director", "security_director", "cto"],
                approval_timeout=24,  # 24 hours
                escalation_rules={"timeout_action": "reject"},
                emergency_override_allowed=False
            )
        }
    
    def determine_approval_level(self, safety_assessment: SafetyAssessment) -> ApprovalLevel:
        """Determine required approval level based on safety assessment"""
        
        # Check for blocking issues
        if safety_assessment.blocking_issues:
            return ApprovalLevel.EXECUTIVE_APPROVAL
        
        # Check risk score
        if safety_assessment.risk_score >= 0.8:
            return ApprovalLevel.EXECUTIVE_APPROVAL
        elif safety_assessment.risk_score >= 0.6:
            return ApprovalLevel.SENIOR_APPROVAL
        elif safety_assessment.risk_score >= 0.3:
            return ApprovalLevel.PEER_REVIEW
        else:
            return ApprovalLevel.AUTOMATIC
    
    def get_workflow(self, approval_level: ApprovalLevel) -> ApprovalWorkflow:
        """Get workflow for approval level"""
        workflow_mapping = {
            ApprovalLevel.AUTOMATIC: "low_risk",
            ApprovalLevel.PEER_REVIEW: "medium_risk",
            ApprovalLevel.SENIOR_APPROVAL: "high_risk",
            ApprovalLevel.EXECUTIVE_APPROVAL: "critical_risk"
        }
        
        workflow_id = workflow_mapping.get(approval_level, "medium_risk")
        return self.workflows[workflow_id]

class SafetyFramework:
    """Main safety framework coordinator"""
    
    def __init__(self):
        self.syntax_validator = SyntaxValidator()
        self.security_validator = SecurityValidator()
        self.performance_validator = PerformanceValidator()
        self.compliance_validator = ComplianceValidator()
        self.approval_manager = ApprovalWorkflowManager()
        
    def assess_safety(self, decision_context: Dict[str, Any], 
                     proposed_changes: Dict[str, Any]) -> SafetyAssessment:
        """Perform comprehensive safety assessment"""
        
        validation_checks = []
        
        # Extract code and metadata
        code = proposed_changes.get("code", "")
        language = proposed_changes.get("language", "python")
        metadata = proposed_changes.get("metadata", {})
        
        # Syntax validation
        if language == "python":
            validation_checks.append(self.syntax_validator.validate_python_syntax(code))
        elif language in ["javascript", "typescript"]:
            validation_checks.append(self.syntax_validator.validate_javascript_syntax(code))
        
        # Code quality validation
        validation_checks.extend(self.syntax_validator.validate_code_quality(code, language))
        
        # Security validation
        validation_checks.extend(self.security_validator.validate_security(code, language))
        
        # Performance validation
        validation_checks.extend(self.performance_validator.validate_performance_impact(code, language))
        
        # Compliance validation
        validation_checks.extend(self.compliance_validator.validate_compliance(code, metadata))
        
        # Calculate overall scores
        risk_score = self._calculate_risk_score(validation_checks)
        safety_score = self._calculate_safety_score(validation_checks)
        
        # Determine overall result
        overall_result = self._determine_overall_result(validation_checks)
        
        # Identify blocking issues
        blocking_issues = [
            check.message for check in validation_checks 
            if check.result == ValidationResult.FAIL and check.severity == "critical"
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_checks)
        
        # Determine approval level
        assessment = SafetyAssessment(
            decision_id=decision_context.get("decision_id", "unknown"),
            overall_result=overall_result,
            approval_level=ApprovalLevel.AUTOMATIC,  # Will be updated
            validation_checks=validation_checks,
            risk_score=risk_score,
            safety_score=safety_score,
            recommendations=recommendations,
            blocking_issues=blocking_issues,
            timestamp=datetime.now()
        )
        
        # Update approval level
        assessment.approval_level = self.approval_manager.determine_approval_level(assessment)
        
        return assessment
    
    def _calculate_risk_score(self, checks: List[ValidationCheck]) -> float:
        """Calculate overall risk score from validation checks"""
        if not checks:
            return 0.0
        
        severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.7, "critical": 1.0}
        result_weights = {
            ValidationResult.PASS: 0.0,
            ValidationResult.WARNING: 0.3,
            ValidationResult.FAIL: 1.0,
            ValidationResult.REQUIRES_REVIEW: 0.5
        }
        
        total_risk = 0.0
        total_weight = 0.0
        
        for check in checks:
            severity_weight = severity_weights.get(check.severity, 0.5)
            result_weight = result_weights.get(check.result, 0.5)
            
            risk_contribution = severity_weight * result_weight
            total_risk += risk_contribution
            total_weight += severity_weight
        
        return total_risk / total_weight if total_weight > 0 else 0.0
    
    def _calculate_safety_score(self, checks: List[ValidationCheck]) -> float:
        """Calculate overall safety score (inverse of risk)"""
        risk_score = self._calculate_risk_score(checks)
        return 1.0 - risk_score
    
    def _determine_overall_result(self, checks: List[ValidationCheck]) -> ValidationResult:
        """Determine overall validation result"""
        if not checks:
            return ValidationResult.PASS
        
        # Check for any critical failures
        critical_failures = [c for c in checks if c.result == ValidationResult.FAIL and c.severity == "critical"]
        if critical_failures:
            return ValidationResult.FAIL
        
        # Check for any failures
        failures = [c for c in checks if c.result == ValidationResult.FAIL]
        if failures:
            return ValidationResult.REQUIRES_REVIEW
        
        # Check for warnings
        warnings = [c for c in checks if c.result == ValidationResult.WARNING]
        if warnings:
            return ValidationResult.WARNING
        
        return ValidationResult.PASS
    
    def _generate_recommendations(self, checks: List[ValidationCheck]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Group checks by type
        check_types = {}
        for check in checks:
            check_type = check.check_name.split('_')[0]
            if check_type not in check_types:
                check_types[check_type] = []
            check_types[check_type].append(check)
        
        # Generate type-specific recommendations
        if "security" in check_types:
            security_issues = len([c for c in check_types["security"] if c.result == ValidationResult.FAIL])
            if security_issues > 0:
                recommendations.append(f"Address {security_issues} security vulnerabilities before deployment")
        
        if "performance" in check_types:
            performance_issues = len([c for c in check_types["performance"] if c.result == ValidationResult.WARNING])
            if performance_issues > 0:
                recommendations.append(f"Consider optimizing {performance_issues} performance issues")
        
        if "compliance" in check_types:
            compliance_issues = len([c for c in check_types["compliance"] if c.result == ValidationResult.FAIL])
            if compliance_issues > 0:
                recommendations.append(f"Ensure compliance with {compliance_issues} organizational policies")
        
        # Add general recommendations
        total_issues = len([c for c in checks if c.result in [ValidationResult.FAIL, ValidationResult.WARNING]])
        if total_issues > 10:
            recommendations.append("Consider breaking down changes into smaller, more manageable pieces")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Initialize safety framework
    safety_framework = SafetyFramework()
    
    # Example decision context
    decision_context = {
        "decision_id": "CHANGE-2025-001",
        "decision_type": "code_change",
        "urgency": "medium",
        "requester": "developer_team"
    }
    
    # Example proposed changes
    proposed_changes = {
        "code": '''
import os
import subprocess

def process_user_input(user_input):
    # TODO: Add input validation
    password = "hardcoded_password_123"
    
    # Potential security issue
    command = "ls " + user_input
    result = os.system(command)
    
    return result

def complex_function(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                if data[i][j][k] > 100:
                    print(f"Processing {data[i][j][k]}")
    
    return data
        ''',
        "language": "python",
        "metadata": {
            "handles_sensitive_data": True,
            "component": "user_authentication",
            "impact": "high"
        }
    }
    
    # Perform safety assessment
    assessment = safety_framework.assess_safety(decision_context, proposed_changes)
    
    # Print results
    print(f"Decision ID: {assessment.decision_id}")
    print(f"Overall Result: {assessment.overall_result.value}")
    print(f"Approval Level: {assessment.approval_level.value}")
    print(f"Risk Score: {assessment.risk_score:.3f}")
    print(f"Safety Score: {assessment.safety_score:.3f}")
    print(f"Blocking Issues: {len(assessment.blocking_issues)}")
    print(f"Total Validation Checks: {len(assessment.validation_checks)}")
    
    print("\nValidation Checks:")
    for check in assessment.validation_checks:
        print(f"  - {check.check_name}: {check.result.value} ({check.severity}) - {check.message}")
    
    print(f"\nRecommendations:")
    for rec in assessment.recommendations:
        print(f"  - {rec}")
    
    if assessment.blocking_issues:
        print(f"\nBlocking Issues:")
        for issue in assessment.blocking_issues:
            print(f"  - {issue}")
    
    # Get workflow information
    workflow = safety_framework.approval_manager.get_workflow(assessment.approval_level)
    print(f"\nApproval Workflow: {workflow.name}")
    print(f"Required Approvers: {workflow.required_approvers}")
    print(f"Approval Timeout: {workflow.approval_timeout} hours")

