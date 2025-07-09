"""
Legacy System Analyzer for Nexus Architect
Comprehensive legacy code analysis, technical debt assessment, and modernization planning
"""

import ast
import os
import re
import json
import logging
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import subprocess
import tempfile

import aiohttp
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Analysis Models
@dataclass
class LegacyPattern:
    """Represents a detected legacy pattern in code"""
    pattern_type: str
    severity: str  # low, medium, high, critical
    location: str
    description: str
    recommendation: str
    effort_estimate: int  # hours
    risk_level: str
    modernization_priority: int

@dataclass
class TechnicalDebt:
    """Represents technical debt in the codebase"""
    debt_type: str
    file_path: str
    line_number: int
    description: str
    severity: str
    estimated_hours: float
    interest_rate: float  # how much it slows development
    principal: float  # initial cost to fix
    compound_factor: float  # how debt grows over time

@dataclass
class DependencyAnalysis:
    """Represents dependency analysis results"""
    dependency_name: str
    current_version: str
    latest_version: str
    security_vulnerabilities: List[str]
    breaking_changes: List[str]
    migration_complexity: str
    update_recommendation: str

@dataclass
class ArchitectureAssessment:
    """Represents architecture assessment results"""
    architecture_type: str  # monolith, microservices, hybrid
    complexity_score: float
    coupling_score: float
    cohesion_score: float
    maintainability_index: float
    decomposition_opportunities: List[str]
    modernization_recommendations: List[str]

# Request/Response Models
class AnalysisRequest(BaseModel):
    repository_url: str
    branch: str = "main"
    analysis_type: str = Field(..., regex="^(full|quick|focused)$")
    focus_areas: List[str] = []
    exclude_patterns: List[str] = []

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class LegacySystemAnalyzer:
    """
    Comprehensive legacy system analysis engine with pattern detection,
    technical debt assessment, and modernization planning capabilities.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection rules
        self.legacy_patterns = {
            'god_class': {
                'description': 'Class with too many responsibilities',
                'detection': self._detect_god_class,
                'severity': 'high',
                'recommendation': 'Split into smaller, focused classes'
            },
            'long_method': {
                'description': 'Method with excessive lines of code',
                'detection': self._detect_long_method,
                'severity': 'medium',
                'recommendation': 'Extract smaller methods'
            },
            'duplicate_code': {
                'description': 'Duplicated code blocks',
                'detection': self._detect_duplicate_code,
                'severity': 'medium',
                'recommendation': 'Extract common functionality'
            },
            'magic_numbers': {
                'description': 'Hard-coded numeric values',
                'detection': self._detect_magic_numbers,
                'severity': 'low',
                'recommendation': 'Use named constants'
            },
            'deprecated_apis': {
                'description': 'Usage of deprecated APIs',
                'detection': self._detect_deprecated_apis,
                'severity': 'high',
                'recommendation': 'Migrate to modern APIs'
            },
            'security_vulnerabilities': {
                'description': 'Potential security issues',
                'detection': self._detect_security_vulnerabilities,
                'severity': 'critical',
                'recommendation': 'Apply security fixes'
            }
        }
        
        # Technical debt calculation factors
        self.debt_factors = {
            'complexity_multiplier': 1.5,
            'age_multiplier': 1.2,
            'change_frequency_multiplier': 2.0,
            'bug_density_multiplier': 3.0
        }
    
    async def analyze_repository(self, request: AnalysisRequest) -> str:
        """Start comprehensive repository analysis"""
        analysis_id = hashlib.md5(
            f"{request.repository_url}_{request.branch}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        # Store analysis request
        await self.redis_client.hset(
            f"analysis:{analysis_id}",
            mapping={
                "status": "started",
                "progress": "0",
                "repository_url": request.repository_url,
                "branch": request.branch,
                "analysis_type": request.analysis_type,
                "started_at": datetime.now().isoformat()
            }
        )
        
        # Start background analysis
        asyncio.create_task(self._perform_analysis(analysis_id, request))
        
        return analysis_id
    
    async def _perform_analysis(self, analysis_id: str, request: AnalysisRequest):
        """Perform comprehensive legacy system analysis"""
        try:
            # Update progress
            await self._update_progress(analysis_id, 10, "Cloning repository")
            
            # Clone repository
            repo_path = await self._clone_repository(request.repository_url, request.branch)
            
            # Update progress
            await self._update_progress(analysis_id, 20, "Analyzing code structure")
            
            # Analyze code structure
            code_analysis = await self._analyze_code_structure(repo_path)
            
            # Update progress
            await self._update_progress(analysis_id, 40, "Detecting legacy patterns")
            
            # Detect legacy patterns
            legacy_patterns = await self._detect_legacy_patterns(repo_path, code_analysis)
            
            # Update progress
            await self._update_progress(analysis_id, 60, "Calculating technical debt")
            
            # Calculate technical debt
            technical_debt = await self._calculate_technical_debt(repo_path, code_analysis, legacy_patterns)
            
            # Update progress
            await self._update_progress(analysis_id, 80, "Analyzing dependencies")
            
            # Analyze dependencies
            dependency_analysis = await self._analyze_dependencies(repo_path)
            
            # Update progress
            await self._update_progress(analysis_id, 90, "Assessing architecture")
            
            # Assess architecture
            architecture_assessment = await self._assess_architecture(repo_path, code_analysis)
            
            # Generate modernization recommendations
            modernization_plan = await self._generate_modernization_plan(
                legacy_patterns, technical_debt, dependency_analysis, architecture_assessment
            )
            
            # Compile results
            results = {
                "analysis_summary": {
                    "total_files": code_analysis["total_files"],
                    "total_lines": code_analysis["total_lines"],
                    "languages": code_analysis["languages"],
                    "complexity_score": code_analysis["complexity_score"],
                    "maintainability_index": code_analysis["maintainability_index"]
                },
                "legacy_patterns": [asdict(pattern) for pattern in legacy_patterns],
                "technical_debt": {
                    "total_debt_hours": sum(debt.estimated_hours for debt in technical_debt),
                    "debt_by_type": self._group_debt_by_type(technical_debt),
                    "high_priority_items": [asdict(debt) for debt in technical_debt if debt.severity in ['high', 'critical']]
                },
                "dependency_analysis": [asdict(dep) for dep in dependency_analysis],
                "architecture_assessment": asdict(architecture_assessment),
                "modernization_plan": modernization_plan,
                "generated_at": datetime.now().isoformat()
            }
            
            # Store results
            await self._store_analysis_results(analysis_id, results)
            
            # Update final status
            await self._update_progress(analysis_id, 100, "Analysis completed")
            
            # Cleanup
            await self._cleanup_repository(repo_path)
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {analysis_id}: {str(e)}")
            await self.redis_client.hset(
                f"analysis:{analysis_id}",
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
    
    async def _analyze_code_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze code structure and metrics"""
        analysis = {
            "total_files": 0,
            "total_lines": 0,
            "languages": defaultdict(int),
            "file_sizes": [],
            "complexity_scores": [],
            "function_counts": [],
            "class_counts": []
        }
        
        # Language file extensions
        language_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cs': 'C#',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.php': 'PHP',
            '.rb': 'Ruby'
        }
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common build/dependency directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'vendor', 'target', 'build', 'dist']]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in language_extensions:
                    language = language_extensions[file_ext]
                    analysis["languages"][language] += 1
                    analysis["total_files"] += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            lines = len(content.splitlines())
                            analysis["total_lines"] += lines
                            analysis["file_sizes"].append(lines)
                            
                            # Language-specific analysis
                            if file_ext == '.py':
                                file_analysis = await self._analyze_python_file(content)
                                analysis["complexity_scores"].extend(file_analysis["complexity_scores"])
                                analysis["function_counts"].append(file_analysis["function_count"])
                                analysis["class_counts"].append(file_analysis["class_count"])
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze file {file_path}: {str(e)}")
        
        # Calculate aggregate metrics
        analysis["complexity_score"] = sum(analysis["complexity_scores"]) / max(len(analysis["complexity_scores"]), 1)
        analysis["average_file_size"] = sum(analysis["file_sizes"]) / max(len(analysis["file_sizes"]), 1)
        analysis["maintainability_index"] = self._calculate_maintainability_index(analysis)
        
        return analysis
    
    async def _analyze_python_file(self, content: str) -> Dict[str, Any]:
        """Analyze Python file for metrics"""
        try:
            tree = ast.parse(content)
            
            function_count = 0
            class_count = 0
            complexity_scores = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1
                    complexity = self._calculate_cyclomatic_complexity(node)
                    complexity_scores.append(complexity)
                elif isinstance(node, ast.ClassDef):
                    class_count += 1
            
            return {
                "function_count": function_count,
                "class_count": class_count,
                "complexity_scores": complexity_scores
            }
        
        except SyntaxError:
            return {
                "function_count": 0,
                "class_count": 0,
                "complexity_scores": []
            }
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _calculate_maintainability_index(self, analysis: Dict[str, Any]) -> float:
        """Calculate maintainability index"""
        # Simplified maintainability index calculation
        avg_complexity = analysis["complexity_score"]
        avg_file_size = analysis["average_file_size"]
        
        # Higher complexity and file size reduce maintainability
        maintainability = max(0, 100 - (avg_complexity * 5) - (avg_file_size / 10))
        
        return min(100, maintainability)
    
    async def _detect_legacy_patterns(self, repo_path: str, code_analysis: Dict[str, Any]) -> List[LegacyPattern]:
        """Detect legacy patterns in the codebase"""
        patterns = []
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'vendor', 'target', 'build', 'dist']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_patterns = await self._detect_file_patterns(file_path)
                    patterns.extend(file_patterns)
        
        return patterns
    
    async def _detect_file_patterns(self, file_path: str) -> List[LegacyPattern]:
        """Detect patterns in a single file"""
        patterns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Run pattern detection methods
            for pattern_name, pattern_config in self.legacy_patterns.items():
                detected_patterns = await pattern_config['detection'](file_path, content)
                patterns.extend(detected_patterns)
        
        except Exception as e:
            self.logger.warning(f"Failed to detect patterns in {file_path}: {str(e)}")
        
        return patterns
    
    async def _detect_god_class(self, file_path: str, content: str) -> List[LegacyPattern]:
        """Detect god classes (classes with too many responsibilities)"""
        patterns = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    lines_of_code = len(content.splitlines())
                    
                    if method_count > 20 or lines_of_code > 500:
                        patterns.append(LegacyPattern(
                            pattern_type="god_class",
                            severity="high",
                            location=f"{file_path}:{node.lineno}",
                            description=f"Class '{node.name}' has {method_count} methods and {lines_of_code} lines",
                            recommendation="Split into smaller, focused classes using Single Responsibility Principle",
                            effort_estimate=method_count * 2,
                            risk_level="medium",
                            modernization_priority=8
                        ))
        
        except SyntaxError:
            pass
        
        return patterns
    
    async def _detect_long_method(self, file_path: str, content: str) -> List[LegacyPattern]:
        """Detect long methods"""
        patterns = []
        
        try:
            tree = ast.parse(content)
            lines = content.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method_lines = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                    
                    if method_lines > 50:
                        patterns.append(LegacyPattern(
                            pattern_type="long_method",
                            severity="medium",
                            location=f"{file_path}:{node.lineno}",
                            description=f"Method '{node.name}' has {method_lines} lines",
                            recommendation="Extract smaller methods and improve readability",
                            effort_estimate=method_lines // 10,
                            risk_level="low",
                            modernization_priority=5
                        ))
        
        except SyntaxError:
            pass
        
        return patterns
    
    async def _detect_duplicate_code(self, file_path: str, content: str) -> List[LegacyPattern]:
        """Detect duplicate code blocks"""
        patterns = []
        lines = content.splitlines()
        
        # Simple duplicate detection based on line similarity
        line_hashes = defaultdict(list)
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 10 and not stripped.startswith('#'):
                line_hash = hashlib.md5(stripped.encode()).hexdigest()
                line_hashes[line_hash].append(i + 1)
        
        for line_hash, line_numbers in line_hashes.items():
            if len(line_numbers) > 1:
                patterns.append(LegacyPattern(
                    pattern_type="duplicate_code",
                    severity="medium",
                    location=f"{file_path}:{','.join(map(str, line_numbers))}",
                    description=f"Duplicate code found at lines {line_numbers}",
                    recommendation="Extract common functionality into reusable methods",
                    effort_estimate=len(line_numbers) * 1,
                    risk_level="low",
                    modernization_priority=4
                ))
        
        return patterns
    
    async def _detect_magic_numbers(self, file_path: str, content: str) -> List[LegacyPattern]:
        """Detect magic numbers in code"""
        patterns = []
        
        # Regex to find numeric literals (excluding common values like 0, 1, -1)
        magic_number_pattern = re.compile(r'\b(?!0\b|1\b|-1\b)\d+\b')
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            matches = magic_number_pattern.findall(line)
            if matches:
                patterns.append(LegacyPattern(
                    pattern_type="magic_numbers",
                    severity="low",
                    location=f"{file_path}:{i + 1}",
                    description=f"Magic numbers found: {', '.join(matches)}",
                    recommendation="Replace with named constants for better readability",
                    effort_estimate=len(matches) * 0.5,
                    risk_level="low",
                    modernization_priority=2
                ))
        
        return patterns
    
    async def _detect_deprecated_apis(self, file_path: str, content: str) -> List[LegacyPattern]:
        """Detect usage of deprecated APIs"""
        patterns = []
        
        # Common deprecated patterns
        deprecated_patterns = [
            (r'urllib\.urlopen', 'Use urllib.request.urlopen instead'),
            (r'os\.popen', 'Use subprocess module instead'),
            (r'string\.split', 'Use str.split() method instead'),
            (r'apply\(', 'Use direct function calls instead'),
        ]
        
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            for pattern, recommendation in deprecated_patterns:
                if re.search(pattern, line):
                    patterns.append(LegacyPattern(
                        pattern_type="deprecated_apis",
                        severity="high",
                        location=f"{file_path}:{i + 1}",
                        description=f"Deprecated API usage: {pattern}",
                        recommendation=recommendation,
                        effort_estimate=2,
                        risk_level="medium",
                        modernization_priority=7
                    ))
        
        return patterns
    
    async def _detect_security_vulnerabilities(self, file_path: str, content: str) -> List[LegacyPattern]:
        """Detect potential security vulnerabilities"""
        patterns = []
        
        # Common security anti-patterns
        security_patterns = [
            (r'eval\(', 'Avoid eval() - use safer alternatives'),
            (r'exec\(', 'Avoid exec() - use safer alternatives'),
            (r'pickle\.loads\(', 'Pickle is unsafe - use json or safer serialization'),
            (r'shell=True', 'Avoid shell=True in subprocess calls'),
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
        ]
        
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            for pattern, recommendation in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    patterns.append(LegacyPattern(
                        pattern_type="security_vulnerabilities",
                        severity="critical",
                        location=f"{file_path}:{i + 1}",
                        description=f"Security vulnerability: {pattern}",
                        recommendation=recommendation,
                        effort_estimate=4,
                        risk_level="high",
                        modernization_priority=10
                    ))
        
        return patterns
    
    async def _calculate_technical_debt(self, repo_path: str, code_analysis: Dict[str, Any], 
                                      legacy_patterns: List[LegacyPattern]) -> List[TechnicalDebt]:
        """Calculate technical debt based on analysis results"""
        debt_items = []
        
        # Convert legacy patterns to technical debt
        for pattern in legacy_patterns:
            debt = TechnicalDebt(
                debt_type=pattern.pattern_type,
                file_path=pattern.location.split(':')[0],
                line_number=int(pattern.location.split(':')[1]) if ':' in pattern.location else 0,
                description=pattern.description,
                severity=pattern.severity,
                estimated_hours=pattern.effort_estimate,
                interest_rate=self._calculate_interest_rate(pattern),
                principal=pattern.effort_estimate * 50,  # $50/hour
                compound_factor=1.1  # 10% compound growth
            )
            debt_items.append(debt)
        
        # Add complexity-based debt
        if code_analysis["complexity_score"] > 10:
            debt_items.append(TechnicalDebt(
                debt_type="high_complexity",
                file_path="overall",
                line_number=0,
                description=f"High overall complexity score: {code_analysis['complexity_score']:.2f}",
                severity="medium",
                estimated_hours=code_analysis["complexity_score"] * 2,
                interest_rate=0.15,
                principal=code_analysis["complexity_score"] * 100,
                compound_factor=1.15
            ))
        
        # Add maintainability-based debt
        if code_analysis["maintainability_index"] < 50:
            debt_items.append(TechnicalDebt(
                debt_type="low_maintainability",
                file_path="overall",
                line_number=0,
                description=f"Low maintainability index: {code_analysis['maintainability_index']:.2f}",
                severity="high",
                estimated_hours=(100 - code_analysis["maintainability_index"]) * 0.5,
                interest_rate=0.20,
                principal=(100 - code_analysis["maintainability_index"]) * 25,
                compound_factor=1.20
            ))
        
        return debt_items
    
    def _calculate_interest_rate(self, pattern: LegacyPattern) -> float:
        """Calculate interest rate for technical debt"""
        severity_rates = {
            "low": 0.05,
            "medium": 0.10,
            "high": 0.15,
            "critical": 0.25
        }
        return severity_rates.get(pattern.severity, 0.10)
    
    def _group_debt_by_type(self, technical_debt: List[TechnicalDebt]) -> Dict[str, Dict[str, Any]]:
        """Group technical debt by type"""
        grouped = defaultdict(lambda: {"count": 0, "total_hours": 0, "total_cost": 0})
        
        for debt in technical_debt:
            grouped[debt.debt_type]["count"] += 1
            grouped[debt.debt_type]["total_hours"] += debt.estimated_hours
            grouped[debt.debt_type]["total_cost"] += debt.principal
        
        return dict(grouped)
    
    async def _analyze_dependencies(self, repo_path: str) -> List[DependencyAnalysis]:
        """Analyze project dependencies"""
        dependencies = []
        
        # Check for Python requirements
        requirements_files = ['requirements.txt', 'Pipfile', 'pyproject.toml']
        
        for req_file in requirements_files:
            req_path = os.path.join(repo_path, req_file)
            if os.path.exists(req_path):
                deps = await self._analyze_python_dependencies(req_path)
                dependencies.extend(deps)
        
        # Check for Node.js dependencies
        package_json_path = os.path.join(repo_path, 'package.json')
        if os.path.exists(package_json_path):
            deps = await self._analyze_nodejs_dependencies(package_json_path)
            dependencies.extend(deps)
        
        return dependencies
    
    async def _analyze_python_dependencies(self, requirements_path: str) -> List[DependencyAnalysis]:
        """Analyze Python dependencies"""
        dependencies = []
        
        try:
            with open(requirements_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse dependency name and version
                    if '==' in line:
                        name, version = line.split('==', 1)
                    elif '>=' in line:
                        name, version = line.split('>=', 1)
                    else:
                        name, version = line, "unknown"
                    
                    # Mock dependency analysis (in real implementation, use PyPI API)
                    dependencies.append(DependencyAnalysis(
                        dependency_name=name.strip(),
                        current_version=version.strip(),
                        latest_version="latest",
                        security_vulnerabilities=[],
                        breaking_changes=[],
                        migration_complexity="low",
                        update_recommendation="safe_to_update"
                    ))
        
        except Exception as e:
            self.logger.warning(f"Failed to analyze Python dependencies: {str(e)}")
        
        return dependencies
    
    async def _analyze_nodejs_dependencies(self, package_json_path: str) -> List[DependencyAnalysis]:
        """Analyze Node.js dependencies"""
        dependencies = []
        
        try:
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
            
            deps = package_data.get('dependencies', {})
            dev_deps = package_data.get('devDependencies', {})
            
            all_deps = {**deps, **dev_deps}
            
            for name, version in all_deps.items():
                dependencies.append(DependencyAnalysis(
                    dependency_name=name,
                    current_version=version,
                    latest_version="latest",
                    security_vulnerabilities=[],
                    breaking_changes=[],
                    migration_complexity="medium",
                    update_recommendation="review_required"
                ))
        
        except Exception as e:
            self.logger.warning(f"Failed to analyze Node.js dependencies: {str(e)}")
        
        return dependencies
    
    async def _assess_architecture(self, repo_path: str, code_analysis: Dict[str, Any]) -> ArchitectureAssessment:
        """Assess overall architecture"""
        # Simplified architecture assessment
        total_files = code_analysis["total_files"]
        complexity_score = code_analysis["complexity_score"]
        
        # Determine architecture type based on structure
        architecture_type = "monolith"  # Default assumption
        
        # Check for microservices indicators
        if os.path.exists(os.path.join(repo_path, 'docker-compose.yml')):
            architecture_type = "microservices"
        elif os.path.exists(os.path.join(repo_path, 'Dockerfile')):
            architecture_type = "containerized_monolith"
        
        # Calculate metrics
        coupling_score = min(100, complexity_score * 2)  # Higher complexity suggests higher coupling
        cohesion_score = max(0, 100 - coupling_score)
        
        # Generate recommendations
        recommendations = []
        if complexity_score > 15:
            recommendations.append("Consider breaking down complex components")
        if total_files > 100:
            recommendations.append("Consider modular architecture")
        if coupling_score > 70:
            recommendations.append("Reduce coupling between components")
        
        decomposition_opportunities = []
        if architecture_type == "monolith" and total_files > 50:
            decomposition_opportunities.append("User management module")
            decomposition_opportunities.append("Data processing module")
            decomposition_opportunities.append("API gateway")
        
        return ArchitectureAssessment(
            architecture_type=architecture_type,
            complexity_score=complexity_score,
            coupling_score=coupling_score,
            cohesion_score=cohesion_score,
            maintainability_index=code_analysis["maintainability_index"],
            decomposition_opportunities=decomposition_opportunities,
            modernization_recommendations=recommendations
        )
    
    async def _generate_modernization_plan(self, legacy_patterns: List[LegacyPattern],
                                         technical_debt: List[TechnicalDebt],
                                         dependency_analysis: List[DependencyAnalysis],
                                         architecture_assessment: ArchitectureAssessment) -> Dict[str, Any]:
        """Generate comprehensive modernization plan"""
        
        # Prioritize modernization tasks
        high_priority_patterns = [p for p in legacy_patterns if p.modernization_priority >= 7]
        critical_debt = [d for d in technical_debt if d.severity == "critical"]
        vulnerable_deps = [d for d in dependency_analysis if d.security_vulnerabilities]
        
        # Calculate effort estimates
        total_effort_hours = sum(p.effort_estimate for p in legacy_patterns) + sum(d.estimated_hours for d in technical_debt)
        
        # Generate phases
        phases = []
        
        # Phase 1: Critical security and stability issues
        if critical_debt or vulnerable_deps:
            phases.append({
                "phase": 1,
                "title": "Critical Security and Stability",
                "duration_weeks": 2,
                "effort_hours": sum(d.estimated_hours for d in critical_debt) + len(vulnerable_deps) * 4,
                "tasks": [
                    "Fix critical security vulnerabilities",
                    "Update vulnerable dependencies",
                    "Address critical technical debt"
                ]
            })
        
        # Phase 2: High-priority legacy patterns
        if high_priority_patterns:
            phases.append({
                "phase": 2,
                "title": "Legacy Pattern Remediation",
                "duration_weeks": 4,
                "effort_hours": sum(p.effort_estimate for p in high_priority_patterns),
                "tasks": [
                    "Refactor god classes",
                    "Modernize deprecated APIs",
                    "Improve code structure"
                ]
            })
        
        # Phase 3: Architecture modernization
        if architecture_assessment.decomposition_opportunities:
            phases.append({
                "phase": 3,
                "title": "Architecture Modernization",
                "duration_weeks": 8,
                "effort_hours": len(architecture_assessment.decomposition_opportunities) * 40,
                "tasks": [
                    "Implement microservices decomposition",
                    "Modernize data layer",
                    "Implement API gateway"
                ]
            })
        
        # Phase 4: Remaining improvements
        phases.append({
            "phase": 4,
            "title": "Quality and Performance Improvements",
            "duration_weeks": 6,
            "effort_hours": total_effort_hours * 0.3,
            "tasks": [
                "Code quality improvements",
                "Performance optimizations",
                "Documentation updates"
            ]
        })
        
        return {
            "total_effort_hours": total_effort_hours,
            "estimated_duration_weeks": sum(p["duration_weeks"] for p in phases),
            "estimated_cost": total_effort_hours * 75,  # $75/hour
            "risk_level": "medium" if critical_debt else "low",
            "phases": phases,
            "success_metrics": [
                "Reduce technical debt by 70%",
                "Improve maintainability index to >80",
                "Achieve 95% test coverage",
                "Reduce security vulnerabilities to zero"
            ],
            "recommendations": [
                "Start with critical security issues",
                "Implement comprehensive testing",
                "Use incremental migration approach",
                "Establish code quality gates"
            ]
        }
    
    async def _update_progress(self, analysis_id: str, progress: float, status: str):
        """Update analysis progress"""
        await self.redis_client.hset(
            f"analysis:{analysis_id}",
            mapping={
                "progress": str(progress),
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
        )
    
    async def _store_analysis_results(self, analysis_id: str, results: Dict[str, Any]):
        """Store analysis results in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO legacy_analysis_results (analysis_id, results, created_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (analysis_id) DO UPDATE SET
                    results = $2,
                    updated_at = $3
            """, analysis_id, json.dumps(results), datetime.now())
        
        # Also store in Redis for quick access
        await self.redis_client.hset(
            f"analysis:{analysis_id}",
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
    
    async def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis status and results"""
        data = await self.redis_client.hgetall(f"analysis:{analysis_id}")
        
        if not data:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        result = {
            "analysis_id": analysis_id,
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
app = FastAPI(title="Legacy System Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    
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
            CREATE TABLE IF NOT EXISTS legacy_analysis_results (
                analysis_id VARCHAR(64) PRIMARY KEY,
                results JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
    
    analyzer = LegacySystemAnalyzer(db_pool, redis_client)

@app.post("/api/legacy-analysis/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest):
    """Start legacy system analysis"""
    analysis_id = await analyzer.analyze_repository(request)
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status="started",
        progress=0.0
    )

@app.get("/api/legacy-analysis/status/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_status(analysis_id: str):
    """Get analysis status and results"""
    status_data = await analyzer.get_analysis_status(analysis_id)
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status=status_data["status"],
        progress=status_data["progress"],
        results=status_data.get("results"),
        error_message=status_data.get("error_message")
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "legacy-system-analyzer"}

if __name__ == "__main__":
    uvicorn.run(
        "legacy_system_analyzer:app",
        host="0.0.0.0",
        port=8040,
        reload=True,
        log_level="info"
    )

