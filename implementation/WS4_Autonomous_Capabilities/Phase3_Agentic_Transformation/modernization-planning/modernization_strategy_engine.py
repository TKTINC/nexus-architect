"""
Modernization Strategy Engine for Nexus Architect
Comprehensive modernization planning, migration strategies, and risk assessment
"""

import os
import json
import logging
import asyncio
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import networkx as nx

import aiohttp
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Strategy Models
@dataclass
class ModernizationStrategy:
    """Represents a modernization strategy"""
    strategy_id: str
    name: str
    description: str
    target_architecture: str
    estimated_duration: int  # in weeks
    estimated_cost: float
    risk_level: str
    success_probability: float
    prerequisites: List[str]
    phases: List[str]

@dataclass
class MigrationPlan:
    """Represents a detailed migration plan"""
    plan_id: str
    strategy_id: str
    phases: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    timeline: Dict[str, str]
    resource_requirements: Dict[str, Any]
    risk_mitigation: List[str]
    rollback_plan: List[str]

@dataclass
class ArchitectureAssessment:
    """Represents architecture assessment results"""
    current_architecture: str
    complexity_score: float
    technical_debt_score: float
    maintainability_index: float
    scalability_issues: List[str]
    security_concerns: List[str]
    performance_bottlenecks: List[str]
    modernization_opportunities: List[str]

@dataclass
class RiskAssessment:
    """Represents risk assessment for modernization"""
    risk_id: str
    category: str
    description: str
    probability: float
    impact: str
    severity: str
    mitigation_strategies: List[str]
    contingency_plans: List[str]

# Request/Response Models
class ModernizationRequest(BaseModel):
    repository_url: str
    branch: str = "main"
    target_architecture: str = Field(..., regex="^(microservices|serverless|cloud_native|containerized)$")
    constraints: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}

class ModernizationResponse(BaseModel):
    assessment_id: str
    status: str
    progress: float
    strategy: Optional[Dict[str, Any]] = None
    migration_plan: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class ModernizationStrategyEngine:
    """
    Comprehensive modernization strategy engine with intelligent planning,
    risk assessment, and migration roadmap generation.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategy templates
        self.strategy_templates = self._initialize_strategy_templates()
        
        # Architecture patterns
        self.architecture_patterns = self._initialize_architecture_patterns()
        
        # Risk assessment framework
        self.risk_framework = self._initialize_risk_framework()
        
        # Migration patterns
        self.migration_patterns = self._initialize_migration_patterns()
    
    def _initialize_strategy_templates(self) -> Dict[str, ModernizationStrategy]:
        """Initialize modernization strategy templates"""
        return {
            "monolith_to_microservices": ModernizationStrategy(
                strategy_id="monolith_to_microservices",
                name="Monolith to Microservices",
                description="Decompose monolithic application into microservices architecture",
                target_architecture="microservices",
                estimated_duration=24,
                estimated_cost=150000.0,
                risk_level="high",
                success_probability=0.75,
                prerequisites=[
                    "Comprehensive test coverage",
                    "CI/CD pipeline",
                    "Monitoring infrastructure",
                    "Team training on microservices"
                ],
                phases=[
                    "Domain analysis and service identification",
                    "Data decomposition and database per service",
                    "API design and service contracts",
                    "Gradual service extraction",
                    "Infrastructure setup and deployment",
                    "Monitoring and observability",
                    "Performance optimization"
                ]
            ),
            "legacy_to_cloud_native": ModernizationStrategy(
                strategy_id="legacy_to_cloud_native",
                name="Legacy to Cloud Native",
                description="Transform legacy application to cloud-native architecture",
                target_architecture="cloud_native",
                estimated_duration=18,
                estimated_cost=120000.0,
                risk_level="medium",
                success_probability=0.80,
                prerequisites=[
                    "Cloud platform selection",
                    "Security assessment",
                    "Data migration strategy",
                    "Team cloud training"
                ],
                phases=[
                    "Cloud readiness assessment",
                    "Application containerization",
                    "Infrastructure as code setup",
                    "Service mesh implementation",
                    "Security hardening",
                    "Performance optimization",
                    "Operational procedures"
                ]
            ),
            "framework_modernization": ModernizationStrategy(
                strategy_id="framework_modernization",
                name="Framework Modernization",
                description="Upgrade to modern frameworks and libraries",
                target_architecture="modernized",
                estimated_duration=12,
                estimated_cost=80000.0,
                risk_level="medium",
                success_probability=0.85,
                prerequisites=[
                    "Dependency analysis",
                    "Compatibility testing",
                    "Migration tooling",
                    "Rollback procedures"
                ],
                phases=[
                    "Framework compatibility analysis",
                    "Incremental migration planning",
                    "Automated migration execution",
                    "Testing and validation",
                    "Performance benchmarking",
                    "Documentation updates"
                ]
            ),
            "serverless_transformation": ModernizationStrategy(
                strategy_id="serverless_transformation",
                name="Serverless Transformation",
                description="Transform application to serverless architecture",
                target_architecture="serverless",
                estimated_duration=16,
                estimated_cost=100000.0,
                risk_level="medium",
                success_probability=0.78,
                prerequisites=[
                    "Function decomposition analysis",
                    "State management strategy",
                    "Cold start optimization",
                    "Monitoring setup"
                ],
                phases=[
                    "Function identification and design",
                    "State externalization",
                    "Event-driven architecture setup",
                    "Function implementation and testing",
                    "Performance optimization",
                    "Cost optimization"
                ]
            )
        }
    
    def _initialize_architecture_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize architecture pattern definitions"""
        return {
            "microservices": {
                "characteristics": [
                    "Service decomposition",
                    "Independent deployment",
                    "Database per service",
                    "API-first design",
                    "Fault tolerance"
                ],
                "benefits": [
                    "Scalability",
                    "Technology diversity",
                    "Team autonomy",
                    "Fault isolation"
                ],
                "challenges": [
                    "Distributed system complexity",
                    "Data consistency",
                    "Network latency",
                    "Operational overhead"
                ],
                "best_practices": [
                    "Domain-driven design",
                    "API versioning",
                    "Circuit breakers",
                    "Distributed tracing"
                ]
            },
            "cloud_native": {
                "characteristics": [
                    "Containerization",
                    "Orchestration",
                    "Service mesh",
                    "Infrastructure as code",
                    "Observability"
                ],
                "benefits": [
                    "Portability",
                    "Scalability",
                    "Resilience",
                    "Cost efficiency"
                ],
                "challenges": [
                    "Complexity",
                    "Security",
                    "Vendor lock-in",
                    "Skills gap"
                ],
                "best_practices": [
                    "12-factor app principles",
                    "Immutable infrastructure",
                    "GitOps",
                    "Zero-trust security"
                ]
            },
            "serverless": {
                "characteristics": [
                    "Function as a service",
                    "Event-driven",
                    "Stateless",
                    "Auto-scaling",
                    "Pay-per-use"
                ],
                "benefits": [
                    "No server management",
                    "Automatic scaling",
                    "Cost optimization",
                    "Faster time to market"
                ],
                "challenges": [
                    "Cold starts",
                    "Vendor lock-in",
                    "Debugging complexity",
                    "State management"
                ],
                "best_practices": [
                    "Function composition",
                    "Event sourcing",
                    "Monitoring and alerting",
                    "Security best practices"
                ]
            }
        }
    
    def _initialize_risk_framework(self) -> Dict[str, List[RiskAssessment]]:
        """Initialize risk assessment framework"""
        return {
            "technical": [
                RiskAssessment(
                    risk_id="data_loss",
                    category="technical",
                    description="Risk of data loss during migration",
                    probability=0.15,
                    impact="high",
                    severity="critical",
                    mitigation_strategies=[
                        "Comprehensive backup strategy",
                        "Incremental migration",
                        "Data validation checks",
                        "Rollback procedures"
                    ],
                    contingency_plans=[
                        "Immediate rollback to previous version",
                        "Data recovery from backups",
                        "Emergency data reconstruction"
                    ]
                ),
                RiskAssessment(
                    risk_id="performance_degradation",
                    category="technical",
                    description="Performance degradation after modernization",
                    probability=0.25,
                    impact="medium",
                    severity="high",
                    mitigation_strategies=[
                        "Performance testing",
                        "Load testing",
                        "Gradual rollout",
                        "Performance monitoring"
                    ],
                    contingency_plans=[
                        "Performance tuning",
                        "Infrastructure scaling",
                        "Code optimization"
                    ]
                )
            ],
            "operational": [
                RiskAssessment(
                    risk_id="downtime",
                    category="operational",
                    description="Extended downtime during migration",
                    probability=0.20,
                    impact="high",
                    severity="high",
                    mitigation_strategies=[
                        "Blue-green deployment",
                        "Canary releases",
                        "Feature flags",
                        "Maintenance windows"
                    ],
                    contingency_plans=[
                        "Quick rollback procedures",
                        "Emergency communication plan",
                        "Service degradation handling"
                    ]
                )
            ],
            "business": [
                RiskAssessment(
                    risk_id="budget_overrun",
                    category="business",
                    description="Project exceeding allocated budget",
                    probability=0.30,
                    impact="medium",
                    severity="medium",
                    mitigation_strategies=[
                        "Detailed cost estimation",
                        "Regular budget reviews",
                        "Scope management",
                        "Vendor negotiations"
                    ],
                    contingency_plans=[
                        "Scope reduction",
                        "Phased implementation",
                        "Additional funding approval"
                    ]
                )
            ]
        }
    
    def _initialize_migration_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize migration pattern templates"""
        return {
            "strangler_fig": {
                "name": "Strangler Fig Pattern",
                "description": "Gradually replace legacy system by intercepting calls",
                "use_cases": ["Large monoliths", "Critical systems", "Risk-averse organizations"],
                "steps": [
                    "Identify service boundaries",
                    "Create facade layer",
                    "Implement new services",
                    "Route traffic gradually",
                    "Retire legacy components"
                ],
                "duration_factor": 1.5,
                "risk_factor": 0.7
            },
            "database_decomposition": {
                "name": "Database Decomposition",
                "description": "Split shared database into service-specific databases",
                "use_cases": ["Microservices migration", "Data isolation", "Scalability"],
                "steps": [
                    "Analyze data dependencies",
                    "Design service databases",
                    "Implement data synchronization",
                    "Migrate data incrementally",
                    "Remove shared dependencies"
                ],
                "duration_factor": 1.3,
                "risk_factor": 0.8
            },
            "api_first_migration": {
                "name": "API-First Migration",
                "description": "Design APIs before implementing services",
                "use_cases": ["New service development", "Integration planning", "Team coordination"],
                "steps": [
                    "Design API contracts",
                    "Create API mocks",
                    "Implement services",
                    "Test integrations",
                    "Deploy and monitor"
                ],
                "duration_factor": 1.1,
                "risk_factor": 0.6
            }
        }
    
    async def start_modernization_assessment(self, request: ModernizationRequest) -> str:
        """Start comprehensive modernization assessment"""
        assessment_id = f"assess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.repository_url) % 10000}"
        
        # Store assessment request
        await self.redis_client.hset(
            f"assessment:{assessment_id}",
            mapping={
                "status": "started",
                "progress": "0",
                "repository_url": request.repository_url,
                "branch": request.branch,
                "target_architecture": request.target_architecture,
                "started_at": datetime.now().isoformat()
            }
        )
        
        # Start background assessment
        asyncio.create_task(self._perform_modernization_assessment(assessment_id, request))
        
        return assessment_id
    
    async def _perform_modernization_assessment(self, assessment_id: str, request: ModernizationRequest):
        """Perform comprehensive modernization assessment"""
        try:
            # Update progress
            await self._update_progress(assessment_id, 10, "Cloning repository")
            
            # Clone repository
            repo_path = await self._clone_repository(request.repository_url, request.branch)
            
            # Update progress
            await self._update_progress(assessment_id, 20, "Analyzing architecture")
            
            # Analyze current architecture
            architecture_assessment = await self._analyze_current_architecture(repo_path)
            
            # Update progress
            await self._update_progress(assessment_id, 40, "Generating modernization strategy")
            
            # Generate modernization strategy
            strategy = await self._generate_modernization_strategy(
                architecture_assessment, request.target_architecture, request.constraints, request.preferences
            )
            
            # Update progress
            await self._update_progress(assessment_id, 60, "Creating migration plan")
            
            # Create detailed migration plan
            migration_plan = await self._create_migration_plan(strategy, architecture_assessment)
            
            # Update progress
            await self._update_progress(assessment_id, 80, "Assessing risks")
            
            # Perform risk assessment
            risk_assessment = await self._perform_risk_assessment(strategy, migration_plan, architecture_assessment)
            
            # Update progress
            await self._update_progress(assessment_id, 90, "Generating recommendations")
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                strategy, migration_plan, risk_assessment, architecture_assessment
            )
            
            # Store results
            await self._store_assessment_results(assessment_id, {
                "architecture_assessment": asdict(architecture_assessment),
                "modernization_strategy": asdict(strategy),
                "migration_plan": asdict(migration_plan),
                "risk_assessment": [asdict(risk) for risk in risk_assessment],
                "recommendations": recommendations
            })
            
            # Update final status
            await self._update_progress(assessment_id, 100, "Assessment completed")
            
            # Cleanup
            await self._cleanup_repository(repo_path)
            
        except Exception as e:
            self.logger.error(f"Assessment failed for {assessment_id}: {str(e)}")
            await self.redis_client.hset(
                f"assessment:{assessment_id}",
                mapping={
                    "status": "failed",
                    "error_message": str(e),
                    "completed_at": datetime.now().isoformat()
                }
            )
    
    async def _clone_repository(self, repo_url: str, branch: str) -> str:
        """Clone repository for analysis"""
        temp_dir = tempfile.mkdtemp()
        repo_path = os.path.join(temp_dir, "repo")
        
        # Clone repository
        process = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", "--branch", branch, repo_url, repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Failed to clone repository: {stderr.decode()}")
        
        return repo_path
    
    async def _analyze_current_architecture(self, repo_path: str) -> ArchitectureAssessment:
        """Analyze current architecture and identify modernization opportunities"""
        
        # Analyze codebase structure
        structure_analysis = await self._analyze_codebase_structure(repo_path)
        
        # Calculate complexity metrics
        complexity_score = await self._calculate_complexity_score(repo_path)
        
        # Assess technical debt
        technical_debt_score = await self._assess_technical_debt(repo_path)
        
        # Calculate maintainability index
        maintainability_index = await self._calculate_maintainability_index(repo_path)
        
        # Identify scalability issues
        scalability_issues = await self._identify_scalability_issues(repo_path)
        
        # Identify security concerns
        security_concerns = await self._identify_security_concerns(repo_path)
        
        # Identify performance bottlenecks
        performance_bottlenecks = await self._identify_performance_bottlenecks(repo_path)
        
        # Identify modernization opportunities
        modernization_opportunities = await self._identify_modernization_opportunities(repo_path)
        
        # Determine current architecture type
        current_architecture = await self._determine_architecture_type(structure_analysis)
        
        return ArchitectureAssessment(
            current_architecture=current_architecture,
            complexity_score=complexity_score,
            technical_debt_score=technical_debt_score,
            maintainability_index=maintainability_index,
            scalability_issues=scalability_issues,
            security_concerns=security_concerns,
            performance_bottlenecks=performance_bottlenecks,
            modernization_opportunities=modernization_opportunities
        )
    
    async def _analyze_codebase_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze codebase structure and organization"""
        structure = {
            "total_files": 0,
            "languages": defaultdict(int),
            "frameworks": [],
            "dependencies": [],
            "modules": [],
            "services": [],
            "databases": [],
            "apis": []
        }
        
        # Language file extensions
        language_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cs': 'C#',
            '.go': 'Go',
            '.rs': 'Rust'
        }
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'vendor', 'target', 'build', 'dist']]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                structure["total_files"] += 1
                
                if file_ext in language_extensions:
                    language = language_extensions[file_ext]
                    structure["languages"][language] += 1
                    
                    # Analyze file content for frameworks and patterns
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Detect frameworks
                        frameworks = self._detect_frameworks_in_content(content, language)
                        structure["frameworks"].extend(frameworks)
                        
                        # Detect API patterns
                        apis = self._detect_api_patterns(content, language)
                        structure["apis"].extend(apis)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze file {file_path}: {str(e)}")
                
                # Check for configuration files
                if file in ['requirements.txt', 'package.json', 'pom.xml', 'Cargo.toml']:
                    dependencies = await self._extract_dependencies(file_path)
                    structure["dependencies"].extend(dependencies)
        
        # Remove duplicates
        structure["frameworks"] = list(set(structure["frameworks"]))
        structure["apis"] = list(set(structure["apis"]))
        
        return structure
    
    def _detect_frameworks_in_content(self, content: str, language: str) -> List[str]:
        """Detect frameworks in file content"""
        frameworks = []
        
        if language == "Python":
            framework_patterns = {
                'Flask': [r'from flask import', r'import flask'],
                'Django': [r'from django', r'import django'],
                'FastAPI': [r'from fastapi import', r'import fastapi'],
                'SQLAlchemy': [r'from sqlalchemy import'],
                'Celery': [r'from celery import'],
                'Pandas': [r'import pandas', r'from pandas import']
            }
        elif language == "JavaScript":
            framework_patterns = {
                'Express': [r'require\([\'"]express[\'"]\)', r'from [\'"]express[\'"]'],
                'React': [r'require\([\'"]react[\'"]\)', r'from [\'"]react[\'"]'],
                'Vue': [r'require\([\'"]vue[\'"]\)', r'from [\'"]vue[\'"]'],
                'Angular': [r'@angular/', r'from [\'"]@angular'],
                'Node.js': [r'require\([\'"]http[\'"]\)', r'require\([\'"]fs[\'"]']
            }
        else:
            return frameworks
        
        for framework, patterns in framework_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    frameworks.append(framework)
                    break
        
        return frameworks
    
    def _detect_api_patterns(self, content: str, language: str) -> List[str]:
        """Detect API patterns in code"""
        apis = []
        
        # REST API patterns
        if re.search(r'@app\.(get|post|put|delete)', content):
            apis.append("REST API")
        
        # GraphQL patterns
        if re.search(r'graphql|GraphQL', content):
            apis.append("GraphQL")
        
        # gRPC patterns
        if re.search(r'grpc|protobuf', content):
            apis.append("gRPC")
        
        return apis
    
    async def _extract_dependencies(self, file_path: str) -> List[str]:
        """Extract dependencies from configuration files"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = os.path.basename(file_path)
            
            if filename == 'requirements.txt':
                # Python dependencies
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep = line.split('==')[0].split('>=')[0].split('<=')[0]
                        dependencies.append(dep)
            
            elif filename == 'package.json':
                # Node.js dependencies
                try:
                    package_data = json.loads(content)
                    deps = package_data.get('dependencies', {})
                    dev_deps = package_data.get('devDependencies', {})
                    dependencies.extend(list(deps.keys()))
                    dependencies.extend(list(dev_deps.keys()))
                except json.JSONDecodeError:
                    pass
        
        except Exception as e:
            self.logger.warning(f"Failed to extract dependencies from {file_path}: {str(e)}")
        
        return dependencies
    
    async def _calculate_complexity_score(self, repo_path: str) -> float:
        """Calculate overall complexity score"""
        complexity_metrics = {
            "cyclomatic_complexity": 0,
            "cognitive_complexity": 0,
            "nesting_depth": 0,
            "file_count": 0
        }
        
        # Analyze Python files for complexity
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_complexity = await self._calculate_file_complexity(file_path)
                    
                    complexity_metrics["cyclomatic_complexity"] += file_complexity.get("cyclomatic", 0)
                    complexity_metrics["cognitive_complexity"] += file_complexity.get("cognitive", 0)
                    complexity_metrics["nesting_depth"] += file_complexity.get("nesting", 0)
                    complexity_metrics["file_count"] += 1
        
        # Calculate normalized complexity score (0-100)
        if complexity_metrics["file_count"] > 0:
            avg_cyclomatic = complexity_metrics["cyclomatic_complexity"] / complexity_metrics["file_count"]
            avg_cognitive = complexity_metrics["cognitive_complexity"] / complexity_metrics["file_count"]
            avg_nesting = complexity_metrics["nesting_depth"] / complexity_metrics["file_count"]
            
            # Normalize to 0-100 scale
            complexity_score = min(100, (avg_cyclomatic * 2 + avg_cognitive * 1.5 + avg_nesting * 3))
        else:
            complexity_score = 0
        
        return complexity_score
    
    async def _calculate_file_complexity(self, file_path: str) -> Dict[str, int]:
        """Calculate complexity metrics for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            complexity = {
                "cyclomatic": 1,  # Base complexity
                "cognitive": 0,
                "nesting": 0
            }
            
            # Calculate complexity metrics
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    complexity["cyclomatic"] += 1
                    complexity["cognitive"] += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity["cyclomatic"] += 1
                
                # Calculate nesting depth (simplified)
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    depth = self._calculate_nesting_depth(node)
                    complexity["nesting"] = max(complexity["nesting"], depth)
            
            return complexity
        
        except Exception:
            return {"cyclomatic": 0, "cognitive": 0, "nesting": 0}
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 1) -> int:
        """Calculate maximum nesting depth"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    async def _assess_technical_debt(self, repo_path: str) -> float:
        """Assess technical debt in the codebase"""
        debt_indicators = {
            "code_smells": 0,
            "duplicated_code": 0,
            "long_methods": 0,
            "large_classes": 0,
            "deprecated_usage": 0
        }
        
        # Analyze files for technical debt indicators
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_debt = await self._analyze_file_technical_debt(file_path)
                    
                    for key, value in file_debt.items():
                        debt_indicators[key] += value
        
        # Calculate technical debt score (0-100)
        total_indicators = sum(debt_indicators.values())
        total_files = debt_indicators.get("file_count", 1)
        
        # Normalize to 0-100 scale
        debt_score = min(100, (total_indicators / max(total_files, 1)) * 10)
        
        return debt_score
    
    async def _analyze_file_technical_debt(self, file_path: str) -> Dict[str, int]:
        """Analyze technical debt indicators in a single file"""
        debt = {
            "code_smells": 0,
            "duplicated_code": 0,
            "long_methods": 0,
            "large_classes": 0,
            "deprecated_usage": 0
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check for long methods
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        method_lines = node.end_lineno - node.lineno
                        if method_lines > 50:
                            debt["long_methods"] += 1
                
                elif isinstance(node, ast.ClassDef):
                    # Check for large classes
                    method_count = len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
                    if method_count > 20:
                        debt["large_classes"] += 1
            
            # Check for deprecated usage patterns
            deprecated_patterns = [
                r'from imp import',
                r'%[sd]',  # Old string formatting
                r'assertEquals',  # Deprecated unittest method
            ]
            
            for pattern in deprecated_patterns:
                if re.search(pattern, content):
                    debt["deprecated_usage"] += 1
            
            # Simple code smell detection
            if 'TODO' in content or 'FIXME' in content or 'HACK' in content:
                debt["code_smells"] += 1
        
        except Exception:
            pass
        
        return debt
    
    async def _calculate_maintainability_index(self, repo_path: str) -> float:
        """Calculate maintainability index"""
        # Simplified maintainability calculation
        # In practice, would use more sophisticated metrics
        
        complexity_score = await self._calculate_complexity_score(repo_path)
        debt_score = await self._assess_technical_debt(repo_path)
        
        # Calculate maintainability (higher is better)
        maintainability = max(0, 100 - (complexity_score * 0.4 + debt_score * 0.6))
        
        return maintainability
    
    async def _identify_scalability_issues(self, repo_path: str) -> List[str]:
        """Identify potential scalability issues"""
        issues = []
        
        # Check for common scalability anti-patterns
        scalability_patterns = {
            "Synchronous processing": [r'time\.sleep', r'requests\.get', r'urllib\.request'],
            "Database N+1 queries": [r'for.*in.*:.*\.get\(', r'for.*in.*:.*\.filter\('],
            "Memory leaks": [r'global\s+\w+\s*=\s*\[\]', r'cache\s*=\s*\{\}'],
            "Blocking operations": [r'input\(', r'raw_input\('],
            "Inefficient algorithms": [r'for.*in.*for.*in', r'while.*while']
        }
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        for issue_type, patterns in scalability_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, content):
                                    issues.append(f"{issue_type} detected in {os.path.basename(file_path)}")
                                    break
                    
                    except Exception:
                        continue
        
        return list(set(issues))
    
    async def _identify_security_concerns(self, repo_path: str) -> List[str]:
        """Identify potential security concerns"""
        concerns = []
        
        # Check for common security issues
        security_patterns = {
            "Hardcoded secrets": [r'password\s*=\s*[\'"][^\'"]+[\'"]', r'api_key\s*=\s*[\'"][^\'"]+[\'"]'],
            "SQL injection": [r'execute\([\'"].*%.*[\'"]', r'query\([\'"].*\+.*[\'"]'],
            "Command injection": [r'os\.system\(.*\+', r'subprocess\.call\(.*\+'],
            "Insecure random": [r'random\.random\(\)', r'random\.choice\('],
            "Debug mode enabled": [r'debug\s*=\s*True', r'DEBUG\s*=\s*True']
        }
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        for concern_type, patterns in security_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, content):
                                    concerns.append(f"{concern_type} detected in {os.path.basename(file_path)}")
                                    break
                    
                    except Exception:
                        continue
        
        return list(set(concerns))
    
    async def _identify_performance_bottlenecks(self, repo_path: str) -> List[str]:
        """Identify potential performance bottlenecks"""
        bottlenecks = []
        
        # Check for common performance issues
        performance_patterns = {
            "Inefficient loops": [r'for.*in.*range\(len\(', r'while.*len\(.*\)\s*>'],
            "Repeated calculations": [r'for.*in.*:.*math\.|for.*in.*:.*len\('],
            "Large data loading": [r'\.read\(\)', r'\.readlines\(\)'],
            "Synchronous I/O": [r'requests\.get', r'urllib\.request'],
            "String concatenation": [r'\+.*[\'"].*\+.*[\'"]']
        }
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        for bottleneck_type, patterns in performance_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, content):
                                    bottlenecks.append(f"{bottleneck_type} detected in {os.path.basename(file_path)}")
                                    break
                    
                    except Exception:
                        continue
        
        return list(set(bottlenecks))
    
    async def _identify_modernization_opportunities(self, repo_path: str) -> List[str]:
        """Identify modernization opportunities"""
        opportunities = []
        
        # Check for modernization opportunities
        modernization_patterns = {
            "Legacy Python version": [r'print\s+[^(]', r'raw_input\('],
            "Old string formatting": [r'%[sd]', r'\.format\('],
            "Deprecated imports": [r'from imp import', r'import imp'],
            "Missing type hints": [r'def\s+\w+\([^)]*\):(?!\s*->)'],
            "Old exception handling": [r'except.*,.*:'],
            "Legacy async patterns": [r'yield from', r'@asyncio\.coroutine']
        }
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        for opportunity_type, patterns in modernization_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, content):
                                    opportunities.append(f"{opportunity_type} in {os.path.basename(file_path)}")
                                    break
                    
                    except Exception:
                        continue
        
        return list(set(opportunities))
    
    async def _determine_architecture_type(self, structure_analysis: Dict[str, Any]) -> str:
        """Determine current architecture type"""
        
        # Simple heuristics for architecture detection
        total_files = structure_analysis["total_files"]
        frameworks = structure_analysis["frameworks"]
        apis = structure_analysis["apis"]
        
        # Check for microservices indicators
        if len(apis) > 3 and "Docker" in str(structure_analysis) or "Kubernetes" in str(structure_analysis):
            return "microservices"
        
        # Check for monolithic indicators
        elif total_files > 100 and len(frameworks) <= 2:
            return "monolithic"
        
        # Check for serverless indicators
        elif "AWS Lambda" in str(structure_analysis) or "Azure Functions" in str(structure_analysis):
            return "serverless"
        
        # Default to traditional
        else:
            return "traditional"
    
    async def _generate_modernization_strategy(self, architecture_assessment: ArchitectureAssessment,
                                             target_architecture: str,
                                             constraints: Dict[str, Any],
                                             preferences: Dict[str, Any]) -> ModernizationStrategy:
        """Generate customized modernization strategy"""
        
        # Select base strategy template
        strategy_key = f"{architecture_assessment.current_architecture}_to_{target_architecture}"
        
        # Find best matching strategy
        base_strategy = None
        for key, strategy in self.strategy_templates.items():
            if target_architecture in strategy.target_architecture:
                base_strategy = strategy
                break
        
        if not base_strategy:
            # Create custom strategy
            base_strategy = self.strategy_templates["framework_modernization"]
        
        # Customize strategy based on assessment
        customized_strategy = ModernizationStrategy(
            strategy_id=f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"Custom {base_strategy.name}",
            description=f"Customized modernization strategy for {architecture_assessment.current_architecture} to {target_architecture}",
            target_architecture=target_architecture,
            estimated_duration=self._adjust_duration(base_strategy.estimated_duration, architecture_assessment, constraints),
            estimated_cost=self._adjust_cost(base_strategy.estimated_cost, architecture_assessment, constraints),
            risk_level=self._assess_risk_level(architecture_assessment),
            success_probability=self._calculate_success_probability(architecture_assessment, base_strategy),
            prerequisites=self._customize_prerequisites(base_strategy.prerequisites, architecture_assessment),
            phases=self._customize_phases(base_strategy.phases, architecture_assessment, target_architecture)
        )
        
        return customized_strategy
    
    def _adjust_duration(self, base_duration: int, assessment: ArchitectureAssessment, constraints: Dict[str, Any]) -> int:
        """Adjust estimated duration based on assessment"""
        duration_factor = 1.0
        
        # Adjust based on complexity
        if assessment.complexity_score > 70:
            duration_factor *= 1.3
        elif assessment.complexity_score < 30:
            duration_factor *= 0.8
        
        # Adjust based on technical debt
        if assessment.technical_debt_score > 60:
            duration_factor *= 1.2
        
        # Adjust based on team size constraint
        team_size = constraints.get("team_size", 5)
        if team_size < 3:
            duration_factor *= 1.4
        elif team_size > 8:
            duration_factor *= 0.9
        
        return int(base_duration * duration_factor)
    
    def _adjust_cost(self, base_cost: float, assessment: ArchitectureAssessment, constraints: Dict[str, Any]) -> float:
        """Adjust estimated cost based on assessment"""
        cost_factor = 1.0
        
        # Adjust based on complexity and technical debt
        complexity_factor = assessment.complexity_score / 100
        debt_factor = assessment.technical_debt_score / 100
        
        cost_factor *= (1 + complexity_factor * 0.5 + debt_factor * 0.3)
        
        # Adjust based on budget constraint
        max_budget = constraints.get("max_budget")
        if max_budget and base_cost * cost_factor > max_budget:
            cost_factor = max_budget / base_cost
        
        return base_cost * cost_factor
    
    def _assess_risk_level(self, assessment: ArchitectureAssessment) -> str:
        """Assess overall risk level for modernization"""
        risk_score = 0
        
        # Factor in complexity
        if assessment.complexity_score > 70:
            risk_score += 3
        elif assessment.complexity_score > 40:
            risk_score += 2
        else:
            risk_score += 1
        
        # Factor in technical debt
        if assessment.technical_debt_score > 60:
            risk_score += 2
        elif assessment.technical_debt_score > 30:
            risk_score += 1
        
        # Factor in security concerns
        if len(assessment.security_concerns) > 5:
            risk_score += 2
        elif len(assessment.security_concerns) > 2:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return "high"
        elif risk_score >= 4:
            return "medium"
        else:
            return "low"
    
    def _calculate_success_probability(self, assessment: ArchitectureAssessment, base_strategy: ModernizationStrategy) -> float:
        """Calculate success probability based on assessment"""
        base_probability = base_strategy.success_probability
        
        # Adjust based on maintainability
        maintainability_factor = assessment.maintainability_index / 100
        
        # Adjust based on modernization opportunities
        opportunity_factor = min(1.0, len(assessment.modernization_opportunities) / 10)
        
        # Calculate adjusted probability
        adjusted_probability = base_probability * (0.7 + maintainability_factor * 0.2 + opportunity_factor * 0.1)
        
        return min(0.95, max(0.3, adjusted_probability))
    
    def _customize_prerequisites(self, base_prerequisites: List[str], assessment: ArchitectureAssessment) -> List[str]:
        """Customize prerequisites based on assessment"""
        prerequisites = base_prerequisites.copy()
        
        # Add security-related prerequisites if needed
        if len(assessment.security_concerns) > 3:
            prerequisites.append("Security audit and remediation")
        
        # Add performance prerequisites if needed
        if len(assessment.performance_bottlenecks) > 5:
            prerequisites.append("Performance optimization")
        
        # Add technical debt prerequisites if needed
        if assessment.technical_debt_score > 70:
            prerequisites.append("Technical debt reduction")
        
        return prerequisites
    
    def _customize_phases(self, base_phases: List[str], assessment: ArchitectureAssessment, target_architecture: str) -> List[str]:
        """Customize phases based on assessment and target architecture"""
        phases = base_phases.copy()
        
        # Add preparatory phases if needed
        if assessment.technical_debt_score > 60:
            phases.insert(0, "Technical debt reduction and code cleanup")
        
        if len(assessment.security_concerns) > 3:
            phases.insert(-1, "Security hardening and vulnerability remediation")
        
        # Add architecture-specific phases
        if target_architecture == "microservices":
            phases.append("Service mesh implementation")
            phases.append("Distributed tracing setup")
        elif target_architecture == "serverless":
            phases.append("Function optimization and cold start reduction")
            phases.append("Event-driven architecture implementation")
        
        return phases
    
    async def _create_migration_plan(self, strategy: ModernizationStrategy, assessment: ArchitectureAssessment) -> MigrationPlan:
        """Create detailed migration plan"""
        
        plan_id = f"plan_{strategy.strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create detailed phases with tasks
        detailed_phases = []
        for i, phase_name in enumerate(strategy.phases):
            phase = {
                "phase_id": f"phase_{i+1}",
                "name": phase_name,
                "duration_weeks": strategy.estimated_duration // len(strategy.phases),
                "tasks": self._generate_phase_tasks(phase_name, assessment),
                "deliverables": self._generate_phase_deliverables(phase_name),
                "success_criteria": self._generate_success_criteria(phase_name)
            }
            detailed_phases.append(phase)
        
        # Create dependency graph
        dependencies = self._create_dependency_graph(detailed_phases)
        
        # Create timeline
        timeline = self._create_timeline(detailed_phases, dependencies)
        
        # Define resource requirements
        resource_requirements = {
            "team_size": max(3, min(8, int(strategy.estimated_duration / 4))),
            "skills_required": self._identify_required_skills(strategy.target_architecture),
            "infrastructure": self._identify_infrastructure_needs(strategy.target_architecture),
            "tools": self._identify_required_tools(strategy.target_architecture)
        }
        
        # Create risk mitigation strategies
        risk_mitigation = [
            "Implement comprehensive backup and rollback procedures",
            "Use feature flags for gradual rollout",
            "Establish monitoring and alerting",
            "Create detailed testing procedures",
            "Plan for team training and knowledge transfer"
        ]
        
        # Create rollback plan
        rollback_plan = [
            "Immediate traffic routing to previous version",
            "Database rollback using backup procedures",
            "Configuration rollback to previous state",
            "Team notification and incident response",
            "Post-incident analysis and lessons learned"
        ]
        
        return MigrationPlan(
            plan_id=plan_id,
            strategy_id=strategy.strategy_id,
            phases=detailed_phases,
            dependencies=dependencies,
            timeline=timeline,
            resource_requirements=resource_requirements,
            risk_mitigation=risk_mitigation,
            rollback_plan=rollback_plan
        )
    
    def _generate_phase_tasks(self, phase_name: str, assessment: ArchitectureAssessment) -> List[str]:
        """Generate specific tasks for a phase"""
        task_templates = {
            "Domain analysis and service identification": [
                "Analyze business domains and bounded contexts",
                "Identify service boundaries using domain-driven design",
                "Map data flows and dependencies",
                "Define service contracts and APIs",
                "Create service decomposition plan"
            ],
            "Data decomposition and database per service": [
                "Analyze current data model and dependencies",
                "Design service-specific data models",
                "Plan data migration strategy",
                "Implement data synchronization mechanisms",
                "Test data consistency and integrity"
            ],
            "Technical debt reduction and code cleanup": [
                "Identify and prioritize technical debt items",
                "Refactor complex and tightly coupled code",
                "Update deprecated dependencies",
                "Improve code documentation",
                "Implement automated code quality checks"
            ],
            "Security hardening and vulnerability remediation": [
                "Conduct security audit and vulnerability assessment",
                "Implement security best practices",
                "Update authentication and authorization mechanisms",
                "Encrypt sensitive data and communications",
                "Implement security monitoring and alerting"
            ]
        }
        
        return task_templates.get(phase_name, [
            f"Plan and design {phase_name.lower()}",
            f"Implement {phase_name.lower()}",
            f"Test {phase_name.lower()}",
            f"Deploy {phase_name.lower()}",
            f"Monitor and validate {phase_name.lower()}"
        ])
    
    def _generate_phase_deliverables(self, phase_name: str) -> List[str]:
        """Generate deliverables for a phase"""
        deliverable_templates = {
            "Domain analysis and service identification": [
                "Service boundary documentation",
                "API specifications",
                "Data flow diagrams",
                "Service dependency map"
            ],
            "Technical debt reduction and code cleanup": [
                "Refactored codebase",
                "Updated documentation",
                "Code quality reports",
                "Automated quality gates"
            ]
        }
        
        return deliverable_templates.get(phase_name, [
            f"{phase_name} implementation",
            f"{phase_name} documentation",
            f"{phase_name} test results"
        ])
    
    def _generate_success_criteria(self, phase_name: str) -> List[str]:
        """Generate success criteria for a phase"""
        criteria_templates = {
            "Domain analysis and service identification": [
                "All service boundaries clearly defined",
                "API contracts approved by stakeholders",
                "Data dependencies mapped and documented",
                "Service decomposition plan validated"
            ],
            "Technical debt reduction and code cleanup": [
                "Technical debt score reduced by 30%",
                "Code quality metrics improved",
                "All deprecated dependencies updated",
                "Automated quality checks passing"
            ]
        }
        
        return criteria_templates.get(phase_name, [
            f"{phase_name} completed successfully",
            f"All tests passing",
            f"Documentation updated",
            f"Stakeholder approval received"
        ])
    
    def _create_dependency_graph(self, phases: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Create dependency graph between phases"""
        dependencies = {}
        
        for i, phase in enumerate(phases):
            phase_id = phase["phase_id"]
            
            # Most phases depend on the previous phase
            if i > 0:
                dependencies[phase_id] = [phases[i-1]["phase_id"]]
            else:
                dependencies[phase_id] = []
            
            # Add specific dependencies based on phase content
            if "security" in phase["name"].lower() and i > 1:
                # Security phases can run in parallel with some other phases
                dependencies[phase_id] = [phases[0]["phase_id"]]
        
        return dependencies
    
    def _create_timeline(self, phases: List[Dict[str, Any]], dependencies: Dict[str, List[str]]) -> Dict[str, str]:
        """Create project timeline"""
        timeline = {}
        current_date = datetime.now()
        
        for phase in phases:
            phase_id = phase["phase_id"]
            duration_weeks = phase["duration_weeks"]
            
            # Calculate start date based on dependencies
            start_date = current_date
            for dep_phase_id in dependencies.get(phase_id, []):
                if dep_phase_id in timeline:
                    dep_end_date = datetime.fromisoformat(timeline[dep_phase_id].split(" - ")[1])
                    start_date = max(start_date, dep_end_date + timedelta(days=1))
            
            end_date = start_date + timedelta(weeks=duration_weeks)
            
            timeline[phase_id] = f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
            current_date = end_date
        
        return timeline
    
    def _identify_required_skills(self, target_architecture: str) -> List[str]:
        """Identify required skills for target architecture"""
        skill_requirements = {
            "microservices": [
                "Microservices architecture",
                "API design and development",
                "Container technologies (Docker, Kubernetes)",
                "Service mesh (Istio, Linkerd)",
                "Distributed systems",
                "Event-driven architecture"
            ],
            "cloud_native": [
                "Cloud platforms (AWS, Azure, GCP)",
                "Container orchestration",
                "Infrastructure as code",
                "DevOps practices",
                "Cloud security",
                "Monitoring and observability"
            ],
            "serverless": [
                "Serverless platforms",
                "Function-as-a-Service",
                "Event-driven architecture",
                "NoSQL databases",
                "API Gateway",
                "Performance optimization"
            ]
        }
        
        return skill_requirements.get(target_architecture, [
            "Software architecture",
            "Modern development practices",
            "Testing and quality assurance",
            "DevOps and deployment"
        ])
    
    def _identify_infrastructure_needs(self, target_architecture: str) -> List[str]:
        """Identify infrastructure needs for target architecture"""
        infrastructure_needs = {
            "microservices": [
                "Container orchestration platform",
                "Service discovery",
                "Load balancers",
                "Message queues",
                "Distributed databases",
                "Monitoring and logging"
            ],
            "cloud_native": [
                "Cloud platform subscription",
                "Container registry",
                "CI/CD pipeline",
                "Monitoring and alerting",
                "Security tools",
                "Backup and disaster recovery"
            ],
            "serverless": [
                "Serverless platform",
                "API Gateway",
                "Event streaming",
                "NoSQL databases",
                "Monitoring and tracing",
                "Security and compliance tools"
            ]
        }
        
        return infrastructure_needs.get(target_architecture, [
            "Development environment",
            "Testing infrastructure",
            "Deployment pipeline",
            "Monitoring tools"
        ])
    
    def _identify_required_tools(self, target_architecture: str) -> List[str]:
        """Identify required tools for target architecture"""
        tool_requirements = {
            "microservices": [
                "Docker",
                "Kubernetes",
                "Helm",
                "Istio/Linkerd",
                "Prometheus",
                "Grafana",
                "Jaeger/Zipkin"
            ],
            "cloud_native": [
                "Terraform/CloudFormation",
                "Jenkins/GitLab CI",
                "Docker",
                "Kubernetes",
                "Prometheus",
                "ELK Stack"
            ],
            "serverless": [
                "Serverless Framework",
                "AWS SAM/Azure Functions",
                "API Gateway",
                "CloudWatch/Application Insights",
                "X-Ray/Application Map"
            ]
        }
        
        return tool_requirements.get(target_architecture, [
            "Version control system",
            "CI/CD tools",
            "Testing frameworks",
            "Monitoring tools"
        ])
    
    async def _perform_risk_assessment(self, strategy: ModernizationStrategy,
                                     migration_plan: MigrationPlan,
                                     assessment: ArchitectureAssessment) -> List[RiskAssessment]:
        """Perform comprehensive risk assessment"""
        risks = []
        
        # Add base risks from framework
        for category, category_risks in self.risk_framework.items():
            risks.extend(category_risks)
        
        # Add strategy-specific risks
        if strategy.target_architecture == "microservices":
            risks.append(RiskAssessment(
                risk_id="service_communication_failure",
                category="technical",
                description="Service-to-service communication failures",
                probability=0.20,
                impact="high",
                severity="high",
                mitigation_strategies=[
                    "Implement circuit breakers",
                    "Use service mesh for resilience",
                    "Design for graceful degradation",
                    "Implement retry mechanisms"
                ],
                contingency_plans=[
                    "Fallback to monolithic deployment",
                    "Service isolation and recovery",
                    "Manual service restart procedures"
                ]
            ))
        
        # Add assessment-specific risks
        if assessment.complexity_score > 70:
            risks.append(RiskAssessment(
                risk_id="high_complexity_migration",
                category="technical",
                description="High complexity may lead to migration failures",
                probability=0.35,
                impact="high",
                severity="high",
                mitigation_strategies=[
                    "Incremental migration approach",
                    "Comprehensive testing strategy",
                    "Expert consultation",
                    "Prototype development"
                ],
                contingency_plans=[
                    "Scope reduction",
                    "Extended timeline",
                    "Additional resources"
                ]
            ))
        
        if len(assessment.security_concerns) > 5:
            risks.append(RiskAssessment(
                risk_id="security_vulnerabilities",
                category="security",
                description="Existing security vulnerabilities may be exposed",
                probability=0.40,
                impact="high",
                severity="critical",
                mitigation_strategies=[
                    "Security audit before migration",
                    "Implement security best practices",
                    "Regular security testing",
                    "Security training for team"
                ],
                contingency_plans=[
                    "Immediate security patching",
                    "Temporary service isolation",
                    "Emergency rollback procedures"
                ]
            ))
        
        return risks
    
    async def _generate_recommendations(self, strategy: ModernizationStrategy,
                                      migration_plan: MigrationPlan,
                                      risk_assessment: List[RiskAssessment],
                                      architecture_assessment: ArchitectureAssessment) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Strategy-based recommendations
        if strategy.risk_level == "high":
            recommendations.append("Consider a phased approach with pilot projects to reduce risk")
            recommendations.append("Invest in comprehensive team training before starting migration")
        
        if strategy.success_probability < 0.7:
            recommendations.append("Address technical debt and complexity issues before migration")
            recommendations.append("Consider hiring external experts or consultants")
        
        # Assessment-based recommendations
        if architecture_assessment.complexity_score > 70:
            recommendations.append("Prioritize code refactoring and simplification")
            recommendations.append("Implement comprehensive automated testing")
        
        if architecture_assessment.technical_debt_score > 60:
            recommendations.append("Allocate dedicated time for technical debt reduction")
            recommendations.append("Establish code quality gates and standards")
        
        if len(architecture_assessment.security_concerns) > 3:
            recommendations.append("Conduct thorough security audit and remediation")
            recommendations.append("Implement security-first development practices")
        
        # Risk-based recommendations
        high_risk_count = len([r for r in risk_assessment if r.severity in ["high", "critical"]])
        if high_risk_count > 3:
            recommendations.append("Develop comprehensive risk mitigation strategies")
            recommendations.append("Establish clear rollback procedures and criteria")
        
        # Timeline and resource recommendations
        if strategy.estimated_duration > 20:
            recommendations.append("Consider breaking the project into smaller, manageable phases")
            recommendations.append("Establish regular milestone reviews and checkpoints")
        
        # General best practices
        recommendations.extend([
            "Establish clear success metrics and monitoring",
            "Implement continuous integration and deployment practices",
            "Plan for comprehensive documentation and knowledge transfer",
            "Engage stakeholders throughout the migration process"
        ])
        
        return recommendations
    
    async def _update_progress(self, assessment_id: str, progress: float, status: str):
        """Update assessment progress"""
        await self.redis_client.hset(
            f"assessment:{assessment_id}",
            mapping={
                "progress": str(progress),
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
        )
    
    async def _store_assessment_results(self, assessment_id: str, results: Dict[str, Any]):
        """Store assessment results in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO modernization_assessments (assessment_id, results, created_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (assessment_id) DO UPDATE SET
                    results = $2,
                    updated_at = $3
            """, assessment_id, json.dumps(results), datetime.now())
        
        # Also store in Redis for quick access
        await self.redis_client.hset(
            f"assessment:{assessment_id}",
            mapping={
                "status": "completed",
                "progress": "100",
                "results": json.dumps(results),
                "completed_at": datetime.now().isoformat()
            }
        )
    
    async def _cleanup_repository(self, repo_path: str):
        """Clean up cloned repository"""
        try:
            import shutil
            shutil.rmtree(os.path.dirname(repo_path))
        except Exception as e:
            self.logger.warning(f"Failed to cleanup repository: {str(e)}")
    
    async def get_assessment_status(self, assessment_id: str) -> Dict[str, Any]:
        """Get assessment status and results"""
        data = await self.redis_client.hgetall(f"assessment:{assessment_id}")
        
        if not data:
            raise HTTPException(status_code=404, detail="Assessment not found")
        
        result = {
            "assessment_id": assessment_id,
            "status": data.get("status", "unknown"),
            "progress": float(data.get("progress", 0)),
            "started_at": data.get("started_at"),
            "updated_at": data.get("updated_at"),
            "completed_at": data.get("completed_at")
        }
        
        if data.get("results"):
            result["results"] = json.loads(data["results"])
        
        if data.get("error_message"):
            result["error_message"] = data["error_message"]
        
        return result

# FastAPI Application
app = FastAPI(title="Modernization Strategy Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global strategy engine instance
strategy_engine = None

@app.on_event("startup")
async def startup_event():
    global strategy_engine
    
    # Database connection
    db_pool = await asyncpg.create_pool(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        user=os.getenv("DB_USER", "nexus_user"),
        password=os.getenv("DB_PASSWORD", "nexus_password"),
        database=os.getenv("DB_NAME", "nexus_architect"),
        min_size=5,
        max_size=20
    )
    
    # Redis connection
    redis_client = redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True
    )
    
    # Create tables
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS modernization_assessments (
                assessment_id VARCHAR(64) PRIMARY KEY,
                results JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
    
    strategy_engine = ModernizationStrategyEngine(db_pool, redis_client)

@app.post("/api/modernization/assess", response_model=ModernizationResponse)
async def start_modernization_assessment(request: ModernizationRequest):
    """Start modernization assessment"""
    assessment_id = await strategy_engine.start_modernization_assessment(request)
    
    return ModernizationResponse(
        assessment_id=assessment_id,
        status="started",
        progress=0.0
    )

@app.get("/api/modernization/status/{assessment_id}", response_model=ModernizationResponse)
async def get_assessment_status(assessment_id: str):
    """Get assessment status and results"""
    status_data = await strategy_engine.get_assessment_status(assessment_id)
    
    return ModernizationResponse(
        assessment_id=assessment_id,
        status=status_data["status"],
        progress=status_data["progress"],
        strategy=status_data.get("results", {}).get("modernization_strategy"),
        migration_plan=status_data.get("results", {}).get("migration_plan"),
        error_message=status_data.get("error_message")
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "modernization-strategy-engine"}

if __name__ == "__main__":
    uvicorn.run(
        "modernization_strategy_engine:app",
        host="0.0.0.0",
        port=8042,
        reload=True,
        log_level="info"
    )

