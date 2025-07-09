"""
Security Scanner for Nexus Architect
Comprehensive security vulnerability scanning for code repositories
"""

import os
import re
import json
import logging
import time
import asyncio
import hashlib
import subprocess
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import tempfile
from pathlib import Path
import aiofiles
import aiohttp
from prometheus_client import Counter, Histogram, Gauge

# Metrics
SECURITY_SCANS = Counter('security_scans_total', 'Total security scans', ['scan_type', 'status'])
SECURITY_SCAN_LATENCY = Histogram('security_scan_latency_seconds', 'Security scan latency', ['scan_type'])
VULNERABILITIES_FOUND = Counter('vulnerabilities_found_total', 'Total vulnerabilities found', ['severity', 'category'])
FALSE_POSITIVES = Counter('false_positives_total', 'Total false positives', ['scan_type'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class VulnerabilityCategory(Enum):
    """Vulnerability categories"""
    INJECTION = "injection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CRYPTOGRAPHY = "cryptography"
    INPUT_VALIDATION = "input_validation"
    CONFIGURATION = "configuration"
    SECRETS = "secrets"
    DEPENDENCIES = "dependencies"
    CODE_QUALITY = "code_quality"
    PRIVACY = "privacy"

class ScanType(Enum):
    """Types of security scans"""
    STATIC_ANALYSIS = "static_analysis"
    DEPENDENCY_SCAN = "dependency_scan"
    SECRET_SCAN = "secret_scan"
    CONFIGURATION_SCAN = "configuration_scan"
    CONTAINER_SCAN = "container_scan"

@dataclass
class SecurityVulnerability:
    """Security vulnerability finding"""
    id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    category: VulnerabilityCategory
    file_path: str
    line_number: int
    column: int = 0
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    confidence: float = 1.0
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    false_positive: bool = False
    suppressed: bool = False
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SecurityReport:
    """Comprehensive security report"""
    repository_id: str
    repository_name: str
    scan_timestamp: datetime
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    risk_score: float = 0.0
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    scan_coverage: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class StaticAnalysisScanner:
    """Static code analysis security scanner"""
    
    def __init__(self):
        self.rules = self._load_security_rules()
    
    def _load_security_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load security scanning rules"""
        return {
            'python': [
                {
                    'id': 'PY001',
                    'pattern': r'eval\s*\(',
                    'severity': VulnerabilitySeverity.CRITICAL,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'Use of eval() function',
                    'description': 'The eval() function can execute arbitrary code and is dangerous',
                    'cwe_id': 'CWE-95',
                    'remediation': 'Avoid using eval(). Use ast.literal_eval() for safe evaluation of literals.'
                },
                {
                    'id': 'PY002',
                    'pattern': r'exec\s*\(',
                    'severity': VulnerabilitySeverity.CRITICAL,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'Use of exec() function',
                    'description': 'The exec() function can execute arbitrary code and is dangerous',
                    'cwe_id': 'CWE-95',
                    'remediation': 'Avoid using exec(). Consider alternative approaches.'
                },
                {
                    'id': 'PY003',
                    'pattern': r'subprocess\.call\s*\([^)]*shell\s*=\s*True',
                    'severity': VulnerabilitySeverity.HIGH,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'Shell injection vulnerability',
                    'description': 'Using shell=True with subprocess can lead to command injection',
                    'cwe_id': 'CWE-78',
                    'remediation': 'Use shell=False and pass arguments as a list.'
                },
                {
                    'id': 'PY004',
                    'pattern': r'os\.system\s*\(',
                    'severity': VulnerabilitySeverity.HIGH,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'Command injection vulnerability',
                    'description': 'os.system() can lead to command injection vulnerabilities',
                    'cwe_id': 'CWE-78',
                    'remediation': 'Use subprocess module with proper argument handling.'
                },
                {
                    'id': 'PY005',
                    'pattern': r'pickle\.loads?\s*\(',
                    'severity': VulnerabilitySeverity.HIGH,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'Unsafe deserialization',
                    'description': 'pickle.load() can execute arbitrary code during deserialization',
                    'cwe_id': 'CWE-502',
                    'remediation': 'Use safe serialization formats like JSON.'
                },
                {
                    'id': 'PY006',
                    'pattern': r'random\.random\s*\(',
                    'severity': VulnerabilitySeverity.MEDIUM,
                    'category': VulnerabilityCategory.CRYPTOGRAPHY,
                    'title': 'Weak random number generation',
                    'description': 'random module is not cryptographically secure',
                    'cwe_id': 'CWE-338',
                    'remediation': 'Use secrets module for cryptographic purposes.'
                },
                {
                    'id': 'PY007',
                    'pattern': r'hashlib\.md5\s*\(',
                    'severity': VulnerabilitySeverity.MEDIUM,
                    'category': VulnerabilityCategory.CRYPTOGRAPHY,
                    'title': 'Weak cryptographic hash',
                    'description': 'MD5 is cryptographically broken and should not be used',
                    'cwe_id': 'CWE-327',
                    'remediation': 'Use SHA-256 or stronger hash functions.'
                },
                {
                    'id': 'PY008',
                    'pattern': r'hashlib\.sha1\s*\(',
                    'severity': VulnerabilitySeverity.MEDIUM,
                    'category': VulnerabilityCategory.CRYPTOGRAPHY,
                    'title': 'Weak cryptographic hash',
                    'description': 'SHA-1 is cryptographically weak and should be avoided',
                    'cwe_id': 'CWE-327',
                    'remediation': 'Use SHA-256 or stronger hash functions.'
                },
                {
                    'id': 'PY009',
                    'pattern': r'ssl\.create_default_context\s*\([^)]*check_hostname\s*=\s*False',
                    'severity': VulnerabilitySeverity.HIGH,
                    'category': VulnerabilityCategory.CRYPTOGRAPHY,
                    'title': 'Disabled SSL hostname verification',
                    'description': 'Disabling hostname verification makes SSL connections vulnerable',
                    'cwe_id': 'CWE-295',
                    'remediation': 'Enable hostname verification for SSL connections.'
                },
                {
                    'id': 'PY010',
                    'pattern': r'requests\.get\s*\([^)]*verify\s*=\s*False',
                    'severity': VulnerabilitySeverity.HIGH,
                    'category': VulnerabilityCategory.CRYPTOGRAPHY,
                    'title': 'Disabled SSL certificate verification',
                    'description': 'Disabling SSL verification makes connections vulnerable to MITM attacks',
                    'cwe_id': 'CWE-295',
                    'remediation': 'Enable SSL certificate verification.'
                }
            ],
            'javascript': [
                {
                    'id': 'JS001',
                    'pattern': r'eval\s*\(',
                    'severity': VulnerabilitySeverity.CRITICAL,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'Use of eval() function',
                    'description': 'eval() can execute arbitrary JavaScript code',
                    'cwe_id': 'CWE-95',
                    'remediation': 'Avoid using eval(). Use JSON.parse() for parsing JSON.'
                },
                {
                    'id': 'JS002',
                    'pattern': r'innerHTML\s*=',
                    'severity': VulnerabilitySeverity.HIGH,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'XSS vulnerability with innerHTML',
                    'description': 'Setting innerHTML with user data can lead to XSS',
                    'cwe_id': 'CWE-79',
                    'remediation': 'Use textContent or properly sanitize HTML content.'
                },
                {
                    'id': 'JS003',
                    'pattern': r'document\.write\s*\(',
                    'severity': VulnerabilitySeverity.HIGH,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'XSS vulnerability with document.write',
                    'description': 'document.write() with user data can lead to XSS',
                    'cwe_id': 'CWE-79',
                    'remediation': 'Use modern DOM manipulation methods.'
                },
                {
                    'id': 'JS004',
                    'pattern': r'setTimeout\s*\(\s*["\']',
                    'severity': VulnerabilitySeverity.MEDIUM,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'Code injection in setTimeout',
                    'description': 'Passing strings to setTimeout can lead to code injection',
                    'cwe_id': 'CWE-95',
                    'remediation': 'Pass functions instead of strings to setTimeout.'
                },
                {
                    'id': 'JS005',
                    'pattern': r'Math\.random\s*\(',
                    'severity': VulnerabilitySeverity.MEDIUM,
                    'category': VulnerabilityCategory.CRYPTOGRAPHY,
                    'title': 'Weak random number generation',
                    'description': 'Math.random() is not cryptographically secure',
                    'cwe_id': 'CWE-338',
                    'remediation': 'Use crypto.getRandomValues() for cryptographic purposes.'
                },
                {
                    'id': 'JS006',
                    'pattern': r'localStorage\.setItem\s*\([^)]*password',
                    'severity': VulnerabilitySeverity.HIGH,
                    'category': VulnerabilityCategory.PRIVACY,
                    'title': 'Sensitive data in localStorage',
                    'description': 'Storing passwords in localStorage is insecure',
                    'cwe_id': 'CWE-312',
                    'remediation': 'Never store sensitive data in localStorage.'
                }
            ],
            'sql': [
                {
                    'id': 'SQL001',
                    'pattern': r'SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*\+',
                    'severity': VulnerabilitySeverity.CRITICAL,
                    'category': VulnerabilityCategory.INJECTION,
                    'title': 'SQL injection vulnerability',
                    'description': 'String concatenation in SQL queries can lead to injection',
                    'cwe_id': 'CWE-89',
                    'remediation': 'Use parameterized queries or prepared statements.'
                }
            ]
        }
    
    async def scan_file(self, file_path: str, content: str, language: str) -> List[SecurityVulnerability]:
        """Scan single file for security vulnerabilities"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            rules = self.rules.get(language.lower(), [])
            lines = content.split('\n')
            
            for rule in rules:
                pattern = rule['pattern']
                
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    
                    for match in matches:
                        vuln = SecurityVulnerability(
                            id=f"{rule['id']}-{hashlib.md5(f'{file_path}:{line_num}:{match.start()}'.encode()).hexdigest()[:8]}",
                            title=rule['title'],
                            description=rule['description'],
                            severity=rule['severity'],
                            category=rule['category'],
                            file_path=file_path,
                            line_number=line_num,
                            column=match.start(),
                            code_snippet=line.strip(),
                            cwe_id=rule.get('cwe_id'),
                            remediation=rule.get('remediation'),
                            confidence=0.8  # Static analysis confidence
                        )
                        vulnerabilities.append(vuln)
                        
                        VULNERABILITIES_FOUND.labels(
                            severity=vuln.severity.value,
                            category=vuln.category.value
                        ).inc()
            
            SECURITY_SCANS.labels(scan_type='static_analysis', status='success').inc()
            
        except Exception as e:
            logger.error(f"Static analysis failed for {file_path}: {e}")
            SECURITY_SCANS.labels(scan_type='static_analysis', status='error').inc()
        
        finally:
            latency = time.time() - start_time
            SECURITY_SCAN_LATENCY.labels(scan_type='static_analysis').observe(latency)
        
        return vulnerabilities

class SecretScanner:
    """Scanner for hardcoded secrets and credentials"""
    
    def __init__(self):
        self.secret_patterns = self._load_secret_patterns()
    
    def _load_secret_patterns(self) -> List[Dict[str, Any]]:
        """Load secret detection patterns"""
        return [
            {
                'id': 'SECRET001',
                'name': 'AWS Access Key',
                'pattern': r'AKIA[0-9A-Z]{16}',
                'severity': VulnerabilitySeverity.CRITICAL,
                'description': 'AWS Access Key ID detected'
            },
            {
                'id': 'SECRET002',
                'name': 'AWS Secret Key',
                'pattern': r'[0-9a-zA-Z/+]{40}',
                'severity': VulnerabilitySeverity.CRITICAL,
                'description': 'Potential AWS Secret Access Key detected',
                'context_required': True
            },
            {
                'id': 'SECRET003',
                'name': 'GitHub Token',
                'pattern': r'ghp_[0-9a-zA-Z]{36}',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'GitHub Personal Access Token detected'
            },
            {
                'id': 'SECRET004',
                'name': 'Generic API Key',
                'pattern': r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([0-9a-zA-Z_-]{20,})["\']?',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'Generic API key detected'
            },
            {
                'id': 'SECRET005',
                'name': 'Database Password',
                'pattern': r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']([^"\']{8,})["\']',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'Hardcoded password detected'
            },
            {
                'id': 'SECRET006',
                'name': 'Private Key',
                'pattern': r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
                'severity': VulnerabilitySeverity.CRITICAL,
                'description': 'Private key detected'
            },
            {
                'id': 'SECRET007',
                'name': 'JWT Token',
                'pattern': r'eyJ[0-9a-zA-Z_-]*\.[0-9a-zA-Z_-]*\.[0-9a-zA-Z_-]*',
                'severity': VulnerabilitySeverity.MEDIUM,
                'description': 'JWT token detected'
            },
            {
                'id': 'SECRET008',
                'name': 'Database Connection String',
                'pattern': r'(?i)(mongodb|mysql|postgresql)://[^/\s]+:[^@\s]+@',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'Database connection string with credentials detected'
            },
            {
                'id': 'SECRET009',
                'name': 'Slack Token',
                'pattern': r'xox[baprs]-[0-9a-zA-Z-]{10,48}',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'Slack token detected'
            },
            {
                'id': 'SECRET010',
                'name': 'Google API Key',
                'pattern': r'AIza[0-9A-Za-z_-]{35}',
                'severity': VulnerabilitySeverity.HIGH,
                'description': 'Google API key detected'
            }
        ]
    
    async def scan_file(self, file_path: str, content: str) -> List[SecurityVulnerability]:
        """Scan file for hardcoded secrets"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            lines = content.split('\n')
            
            for pattern_info in self.secret_patterns:
                pattern = pattern_info['pattern']
                
                for line_num, line in enumerate(lines, 1):
                    # Skip comments in common formats
                    stripped_line = line.strip()
                    if (stripped_line.startswith('#') or 
                        stripped_line.startswith('//') or 
                        stripped_line.startswith('/*')):
                        continue
                    
                    matches = re.finditer(pattern, line)
                    
                    for match in matches:
                        # Additional validation for context-required patterns
                        if pattern_info.get('context_required'):
                            if not self._validate_secret_context(line, pattern_info['name']):
                                continue
                        
                        vuln = SecurityVulnerability(
                            id=f"{pattern_info['id']}-{hashlib.md5(f'{file_path}:{line_num}:{match.start()}'.encode()).hexdigest()[:8]}",
                            title=f"Hardcoded {pattern_info['name']}",
                            description=pattern_info['description'],
                            severity=pattern_info['severity'],
                            category=VulnerabilityCategory.SECRETS,
                            file_path=file_path,
                            line_number=line_num,
                            column=match.start(),
                            code_snippet=self._mask_secret(line.strip()),
                            cwe_id='CWE-798',
                            remediation='Remove hardcoded secrets and use environment variables or secure vaults.',
                            confidence=0.9
                        )
                        vulnerabilities.append(vuln)
                        
                        VULNERABILITIES_FOUND.labels(
                            severity=vuln.severity.value,
                            category=vuln.category.value
                        ).inc()
            
            SECURITY_SCANS.labels(scan_type='secret_scan', status='success').inc()
            
        except Exception as e:
            logger.error(f"Secret scanning failed for {file_path}: {e}")
            SECURITY_SCANS.labels(scan_type='secret_scan', status='error').inc()
        
        finally:
            latency = time.time() - start_time
            SECURITY_SCAN_LATENCY.labels(scan_type='secret_scan').observe(latency)
        
        return vulnerabilities
    
    def _validate_secret_context(self, line: str, secret_type: str) -> bool:
        """Validate secret based on context"""
        line_lower = line.lower()
        
        if secret_type == 'AWS Secret Key':
            # Check if line contains AWS-related keywords
            aws_keywords = ['aws', 'secret', 'access', 'key']
            return any(keyword in line_lower for keyword in aws_keywords)
        
        return True
    
    def _mask_secret(self, line: str) -> str:
        """Mask sensitive parts of the line"""
        # Replace potential secrets with asterisks
        masked_line = re.sub(r'["\'][^"\']{8,}["\']', '"***MASKED***"', line)
        masked_line = re.sub(r'[0-9a-zA-Z_-]{20,}', '***MASKED***', masked_line)
        return masked_line

class ConfigurationScanner:
    """Scanner for insecure configurations"""
    
    def __init__(self):
        self.config_rules = self._load_config_rules()
    
    def _load_config_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load configuration security rules"""
        return {
            'dockerfile': [
                {
                    'id': 'DOCKER001',
                    'pattern': r'FROM\s+.*:latest',
                    'severity': VulnerabilitySeverity.MEDIUM,
                    'title': 'Using latest tag in Docker image',
                    'description': 'Using :latest tag can lead to unpredictable builds',
                    'remediation': 'Use specific version tags for Docker images.'
                },
                {
                    'id': 'DOCKER002',
                    'pattern': r'USER\s+root',
                    'severity': VulnerabilitySeverity.HIGH,
                    'title': 'Running container as root',
                    'description': 'Running containers as root increases security risk',
                    'remediation': 'Create and use a non-root user in the container.'
                },
                {
                    'id': 'DOCKER003',
                    'pattern': r'ADD\s+http',
                    'severity': VulnerabilitySeverity.MEDIUM,
                    'title': 'Using ADD with HTTP URL',
                    'description': 'ADD with HTTP URLs can be insecure',
                    'remediation': 'Use COPY with local files or RUN with curl/wget.'
                }
            ],
            'yaml': [
                {
                    'id': 'K8S001',
                    'pattern': r'privileged:\s*true',
                    'severity': VulnerabilitySeverity.HIGH,
                    'title': 'Privileged container',
                    'description': 'Privileged containers have access to host resources',
                    'remediation': 'Avoid using privileged containers unless absolutely necessary.'
                },
                {
                    'id': 'K8S002',
                    'pattern': r'runAsUser:\s*0',
                    'severity': VulnerabilitySeverity.HIGH,
                    'title': 'Running as root user',
                    'description': 'Running pods as root increases security risk',
                    'remediation': 'Use a non-root user ID.'
                }
            ]
        }
    
    async def scan_file(self, file_path: str, content: str) -> List[SecurityVulnerability]:
        """Scan configuration file for security issues"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            # Determine file type
            file_type = self._determine_config_type(file_path)
            rules = self.config_rules.get(file_type, [])
            
            lines = content.split('\n')
            
            for rule in rules:
                pattern = rule['pattern']
                
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    
                    for match in matches:
                        vuln = SecurityVulnerability(
                            id=f"{rule['id']}-{hashlib.md5(f'{file_path}:{line_num}:{match.start()}'.encode()).hexdigest()[:8]}",
                            title=rule['title'],
                            description=rule['description'],
                            severity=rule['severity'],
                            category=VulnerabilityCategory.CONFIGURATION,
                            file_path=file_path,
                            line_number=line_num,
                            column=match.start(),
                            code_snippet=line.strip(),
                            remediation=rule.get('remediation'),
                            confidence=0.8
                        )
                        vulnerabilities.append(vuln)
                        
                        VULNERABILITIES_FOUND.labels(
                            severity=vuln.severity.value,
                            category=vuln.category.value
                        ).inc()
            
            SECURITY_SCANS.labels(scan_type='configuration_scan', status='success').inc()
            
        except Exception as e:
            logger.error(f"Configuration scanning failed for {file_path}: {e}")
            SECURITY_SCANS.labels(scan_type='configuration_scan', status='error').inc()
        
        finally:
            latency = time.time() - start_time
            SECURITY_SCAN_LATENCY.labels(scan_type='configuration_scan').observe(latency)
        
        return vulnerabilities
    
    def _determine_config_type(self, file_path: str) -> str:
        """Determine configuration file type"""
        file_name = os.path.basename(file_path).lower()
        
        if 'dockerfile' in file_name:
            return 'dockerfile'
        elif file_name.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif file_name.endswith('.json'):
            return 'json'
        elif file_name.endswith(('.conf', '.config', '.ini')):
            return 'config'
        
        return 'unknown'

class DependencyVulnerabilityScanner:
    """Scanner for vulnerabilities in dependencies"""
    
    def __init__(self):
        self.vulnerability_db = {}
    
    async def scan_dependencies(self, dependencies: List[Dict[str, Any]]) -> List[SecurityVulnerability]:
        """Scan dependencies for known vulnerabilities"""
        start_time = time.time()
        vulnerabilities = []
        
        try:
            for dep in dependencies:
                dep_vulns = await self._check_dependency_vulnerabilities(dep)
                vulnerabilities.extend(dep_vulns)
            
            SECURITY_SCANS.labels(scan_type='dependency_scan', status='success').inc()
            
        except Exception as e:
            logger.error(f"Dependency vulnerability scanning failed: {e}")
            SECURITY_SCANS.labels(scan_type='dependency_scan', status='error').inc()
        
        finally:
            latency = time.time() - start_time
            SECURITY_SCAN_LATENCY.labels(scan_type='dependency_scan').observe(latency)
        
        return vulnerabilities
    
    async def _check_dependency_vulnerabilities(self, dependency: Dict[str, Any]) -> List[SecurityVulnerability]:
        """Check single dependency for vulnerabilities"""
        vulnerabilities = []
        
        try:
            # This would integrate with vulnerability databases like OSV, NVD, etc.
            # For demonstration, return empty list
            pass
            
        except Exception as e:
            logger.error(f"Failed to check vulnerabilities for {dependency.get('name')}: {e}")
        
        return vulnerabilities

class SecurityScanner:
    """Main security scanner that coordinates all scan types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.static_scanner = StaticAnalysisScanner()
        self.secret_scanner = SecretScanner()
        self.config_scanner = ConfigurationScanner()
        self.dependency_scanner = DependencyVulnerabilityScanner()
        
        # Start metrics server
        start_http_server(8006)
    
    async def scan_repository(self, repo_path: str, repository_id: str, 
                            repository_name: str) -> SecurityReport:
        """Perform comprehensive security scan of repository"""
        start_time = time.time()
        
        logger.info(f"Starting security scan of repository: {repository_name}")
        
        report = SecurityReport(
            repository_id=repository_id,
            repository_name=repository_name,
            scan_timestamp=datetime.utcnow()
        )
        
        # Find files to scan
        files_to_scan = []
        for root, dirs, files in os.walk(repo_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                if self._should_scan_file(file_path):
                    files_to_scan.append((file_path, relative_path))
        
        # Scan files
        scan_tasks = []
        for file_path, relative_path in files_to_scan:
            task = self._scan_file_comprehensive(file_path, relative_path)
            scan_tasks.append(task)
        
        # Process scan results
        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        for result in scan_results:
            if isinstance(result, list):
                report.vulnerabilities.extend(result)
        
        # Generate summary
        report.summary = self._generate_summary(report.vulnerabilities)
        
        # Calculate risk score
        report.risk_score = self._calculate_risk_score(report.vulnerabilities)
        
        # Check compliance
        report.compliance_status = self._check_compliance(report.vulnerabilities)
        
        # Calculate scan coverage
        report.scan_coverage = self._calculate_scan_coverage(files_to_scan, scan_results)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report.vulnerabilities)
        
        logger.info(f"Security scan completed: {len(report.vulnerabilities)} vulnerabilities found")
        
        return report
    
    def _should_scan_file(self, file_path: str) -> bool:
        """Determine if file should be scanned"""
        # Skip binary files
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:
                    return False
        except:
            return False
        
        # Skip very large files (>1MB)
        try:
            if os.path.getsize(file_path) > 1024 * 1024:
                return False
        except:
            return False
        
        return True
    
    async def _scan_file_comprehensive(self, file_path: str, relative_path: str) -> List[SecurityVulnerability]:
        """Perform comprehensive scan of single file"""
        vulnerabilities = []
        
        try:
            # Read file content
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            
            # Determine file type and language
            language = self._detect_language(relative_path)
            
            # Static analysis scan
            if language in ['python', 'javascript', 'sql']:
                static_vulns = await self.static_scanner.scan_file(relative_path, content, language)
                vulnerabilities.extend(static_vulns)
            
            # Secret scan (all files)
            secret_vulns = await self.secret_scanner.scan_file(relative_path, content)
            vulnerabilities.extend(secret_vulns)
            
            # Configuration scan
            if self._is_config_file(relative_path):
                config_vulns = await self.config_scanner.scan_file(relative_path, content)
                vulnerabilities.extend(config_vulns)
            
        except Exception as e:
            logger.error(f"Failed to scan file {relative_path}: {e}")
        
        return vulnerabilities
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript',
            '.sql': 'sql',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        return language_map.get(ext, 'unknown')
    
    def _is_config_file(self, file_path: str) -> bool:
        """Check if file is a configuration file"""
        file_name = os.path.basename(file_path).lower()
        
        config_patterns = [
            'dockerfile', '.dockerignore',
            '.yml', '.yaml',
            '.json',
            '.conf', '.config', '.ini',
            '.env', '.environment'
        ]
        
        return any(file_name.endswith(pattern) or pattern in file_name for pattern in config_patterns)
    
    def _generate_summary(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, int]:
        """Generate vulnerability summary"""
        summary = {
            'total': len(vulnerabilities),
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }
        
        for vuln in vulnerabilities:
            if not vuln.suppressed:
                summary[vuln.severity.value] += 1
        
        return summary
    
    def _calculate_risk_score(self, vulnerabilities: List[SecurityVulnerability]) -> float:
        """Calculate overall risk score (0-100)"""
        if not vulnerabilities:
            return 0.0
        
        severity_weights = {
            VulnerabilitySeverity.CRITICAL: 10.0,
            VulnerabilitySeverity.HIGH: 7.0,
            VulnerabilitySeverity.MEDIUM: 4.0,
            VulnerabilitySeverity.LOW: 2.0,
            VulnerabilitySeverity.INFO: 1.0
        }
        
        total_score = 0.0
        for vuln in vulnerabilities:
            if not vuln.suppressed:
                weight = severity_weights.get(vuln.severity, 1.0)
                confidence_factor = vuln.confidence
                total_score += weight * confidence_factor
        
        # Normalize to 0-100 scale
        max_possible_score = len(vulnerabilities) * 10.0
        if max_possible_score > 0:
            risk_score = min(100.0, (total_score / max_possible_score) * 100.0)
        else:
            risk_score = 0.0
        
        return round(risk_score, 2)
    
    def _check_compliance(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, bool]:
        """Check compliance with security standards"""
        compliance = {
            'owasp_top_10': True,
            'cwe_top_25': True,
            'pci_dss': True,
            'hipaa': True
        }
        
        # Check for critical vulnerabilities that affect compliance
        for vuln in vulnerabilities:
            if vuln.severity == VulnerabilitySeverity.CRITICAL and not vuln.suppressed:
                compliance['owasp_top_10'] = False
                compliance['cwe_top_25'] = False
                
                # Specific compliance checks
                if vuln.category in [VulnerabilityCategory.CRYPTOGRAPHY, VulnerabilityCategory.SECRETS]:
                    compliance['pci_dss'] = False
                    compliance['hipaa'] = False
        
        return compliance
    
    def _calculate_scan_coverage(self, files_to_scan: List[Tuple[str, str]], 
                                scan_results: List[Any]) -> Dict[str, float]:
        """Calculate scan coverage metrics"""
        total_files = len(files_to_scan)
        successful_scans = sum(1 for result in scan_results if not isinstance(result, Exception))
        
        coverage = {
            'files_scanned': successful_scans / total_files if total_files > 0 else 0.0,
            'static_analysis': 0.8,  # Estimated coverage
            'secret_detection': 0.95,  # High coverage for secret detection
            'configuration': 0.7  # Estimated coverage for config files
        }
        
        return coverage
    
    def _generate_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Count vulnerabilities by category
        category_counts = {}
        for vuln in vulnerabilities:
            if not vuln.suppressed:
                category = vuln.category
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Generate recommendations based on most common issues
        if category_counts.get(VulnerabilityCategory.SECRETS, 0) > 0:
            recommendations.append("Implement secret management using environment variables or secure vaults")
        
        if category_counts.get(VulnerabilityCategory.INJECTION, 0) > 0:
            recommendations.append("Use parameterized queries and input validation to prevent injection attacks")
        
        if category_counts.get(VulnerabilityCategory.CRYPTOGRAPHY, 0) > 0:
            recommendations.append("Update cryptographic implementations to use strong algorithms")
        
        if category_counts.get(VulnerabilityCategory.CONFIGURATION, 0) > 0:
            recommendations.append("Review and harden configuration files following security best practices")
        
        if category_counts.get(VulnerabilityCategory.DEPENDENCIES, 0) > 0:
            recommendations.append("Update dependencies to latest secure versions")
        
        # General recommendations
        if len(vulnerabilities) > 10:
            recommendations.append("Implement automated security scanning in CI/CD pipeline")
        
        if any(v.severity == VulnerabilitySeverity.CRITICAL for v in vulnerabilities):
            recommendations.append("Address critical vulnerabilities immediately")
        
        return recommendations
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'scanners': {
                'static_analysis': 'available',
                'secret_detection': 'available',
                'configuration': 'available',
                'dependency': 'available'
            }
        }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        config = {}
        scanner = SecurityScanner(config)
        
        # Example usage
        repo_path = "/path/to/repository"
        if os.path.exists(repo_path):
            report = await scanner.scan_repository(repo_path, "repo-001", "test-repo")
            print(f"Security scan completed:")
            print(f"  Total vulnerabilities: {report.summary['total']}")
            print(f"  Critical: {report.summary['critical']}")
            print(f"  High: {report.summary['high']}")
            print(f"  Risk score: {report.risk_score}")
            print(f"  Compliance: {report.compliance_status}")
        
        # Health check
        health = await scanner.health_check()
        print(f"Health: {health}")
    
    asyncio.run(main())

