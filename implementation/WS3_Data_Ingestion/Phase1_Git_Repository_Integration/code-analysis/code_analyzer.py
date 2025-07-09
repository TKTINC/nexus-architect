"""
Code Analyzer for Nexus Architect
Provides comprehensive code analysis including AST parsing, complexity metrics, and quality assessment
"""

import ast
import os
import re
import json
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import subprocess
import tempfile
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from prometheus_client import Counter, Histogram, Gauge

# Language-specific parsers
try:
    import javalang
    JAVA_AVAILABLE = True
except ImportError:
    JAVA_AVAILABLE = False

try:
    import esprima
    JAVASCRIPT_AVAILABLE = True
except ImportError:
    JAVASCRIPT_AVAILABLE = False

try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Metrics
CODE_ANALYSIS_REQUESTS = Counter('code_analysis_requests_total', 'Total code analysis requests', ['language', 'status'])
CODE_ANALYSIS_LATENCY = Histogram('code_analysis_latency_seconds', 'Code analysis latency', ['language'])
FILES_ANALYZED = Counter('files_analyzed_total', 'Total files analyzed', ['language', 'file_type'])
COMPLEXITY_SCORE = Gauge('code_complexity_score', 'Code complexity score', ['repository', 'language'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    PHP = "php"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    SCALA = "scala"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    SHELL = "shell"
    DOCKERFILE = "dockerfile"
    MARKDOWN = "markdown"

class FileType(Enum):
    """File type categories"""
    SOURCE_CODE = "source_code"
    CONFIGURATION = "configuration"
    DOCUMENTATION = "documentation"
    BUILD_SCRIPT = "build_script"
    TEST = "test"
    RESOURCE = "resource"

@dataclass
class CodeMetrics:
    """Code quality and complexity metrics"""
    lines_of_code: int = 0
    lines_of_comments: int = 0
    blank_lines: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    technical_debt_ratio: float = 0.0
    duplication_ratio: float = 0.0
    test_coverage: float = 0.0

@dataclass
class SecurityIssue:
    """Security vulnerability or issue"""
    severity: str  # critical, high, medium, low
    type: str
    description: str
    line_number: int
    column: int = 0
    cwe_id: Optional[str] = None
    confidence: float = 1.0

@dataclass
class CodeEntity:
    """Code entity (function, class, variable, etc.)"""
    name: str
    type: str  # function, class, variable, import, etc.
    line_start: int
    line_end: int
    complexity: int = 0
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Dependency:
    """Code dependency"""
    name: str
    type: str  # import, require, include, etc.
    source: str  # local, external, standard_library
    version: Optional[str] = None
    line_number: int = 0

@dataclass
class FileAnalysis:
    """Complete analysis of a single file"""
    file_path: str
    language: Language
    file_type: FileType
    size_bytes: int
    metrics: CodeMetrics
    entities: List[CodeEntity] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    security_issues: List[SecurityIssue] = field(default_factory=list)
    documentation_coverage: float = 0.0
    quality_score: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RepositoryAnalysis:
    """Complete analysis of a repository"""
    repository_id: str
    repository_name: str
    total_files: int
    analyzed_files: int
    languages: Dict[Language, int] = field(default_factory=dict)
    file_types: Dict[FileType, int] = field(default_factory=dict)
    overall_metrics: CodeMetrics = field(default_factory=CodeMetrics)
    file_analyses: List[FileAnalysis] = field(default_factory=list)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    security_summary: Dict[str, int] = field(default_factory=dict)
    quality_trends: List[Dict[str, Any]] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

class LanguageDetector:
    """Detect programming language from file extension and content"""
    
    EXTENSION_MAP = {
        '.py': Language.PYTHON,
        '.js': Language.JAVASCRIPT,
        '.jsx': Language.JAVASCRIPT,
        '.ts': Language.TYPESCRIPT,
        '.tsx': Language.TYPESCRIPT,
        '.java': Language.JAVA,
        '.cs': Language.CSHARP,
        '.go': Language.GO,
        '.rs': Language.RUST,
        '.cpp': Language.CPP,
        '.cc': Language.CPP,
        '.cxx': Language.CPP,
        '.c': Language.C,
        '.h': Language.C,
        '.hpp': Language.CPP,
        '.php': Language.PHP,
        '.rb': Language.RUBY,
        '.kt': Language.KOTLIN,
        '.swift': Language.SWIFT,
        '.scala': Language.SCALA,
        '.sql': Language.SQL,
        '.html': Language.HTML,
        '.htm': Language.HTML,
        '.css': Language.CSS,
        '.scss': Language.CSS,
        '.sass': Language.CSS,
        '.json': Language.JSON,
        '.yaml': Language.YAML,
        '.yml': Language.YAML,
        '.xml': Language.XML,
        '.sh': Language.SHELL,
        '.bash': Language.SHELL,
        '.zsh': Language.SHELL,
        '.dockerfile': Language.DOCKERFILE,
        '.md': Language.MARKDOWN,
        '.markdown': Language.MARKDOWN,
        '.rst': Language.MARKDOWN
    }
    
    @classmethod
    def detect_language(cls, file_path: str, content: Optional[str] = None) -> Optional[Language]:
        """Detect language from file path and content"""
        # Check by extension first
        ext = Path(file_path).suffix.lower()
        if ext in cls.EXTENSION_MAP:
            return cls.EXTENSION_MAP[ext]
        
        # Check by filename
        filename = Path(file_path).name.lower()
        if filename == 'dockerfile':
            return Language.DOCKERFILE
        elif filename == 'makefile':
            return Language.SHELL
        elif filename.endswith('.config') or filename.endswith('.conf'):
            return Language.JSON  # Treat as configuration
        
        # Check by content if available
        if content:
            return cls._detect_by_content(content)
        
        return None
    
    @classmethod
    def _detect_by_content(cls, content: str) -> Optional[Language]:
        """Detect language by analyzing content"""
        content_lower = content.lower().strip()
        
        # Check for shebangs
        if content_lower.startswith('#!/usr/bin/env python') or content_lower.startswith('#!/usr/bin/python'):
            return Language.PYTHON
        elif content_lower.startswith('#!/bin/bash') or content_lower.startswith('#!/bin/sh'):
            return Language.SHELL
        elif content_lower.startswith('#!/usr/bin/env node'):
            return Language.JAVASCRIPT
        
        # Check for language-specific patterns
        if re.search(r'\bdef\s+\w+\s*\(', content) and re.search(r'\bimport\s+\w+', content):
            return Language.PYTHON
        elif re.search(r'\bfunction\s+\w+\s*\(', content) and re.search(r'\bvar\s+\w+', content):
            return Language.JAVASCRIPT
        elif re.search(r'\bpublic\s+class\s+\w+', content):
            return Language.JAVA
        elif re.search(r'\bpackage\s+main', content) and re.search(r'\bfunc\s+\w+', content):
            return Language.GO
        
        return None

class FileTypeClassifier:
    """Classify file types based on path and content"""
    
    TEST_PATTERNS = [
        r'test_.*\.py$',
        r'.*_test\.py$',
        r'.*\.test\.js$',
        r'.*\.spec\.js$',
        r'test/.*',
        r'tests/.*',
        r'spec/.*'
    ]
    
    CONFIG_PATTERNS = [
        r'.*\.config\..*',
        r'.*\.conf$',
        r'.*\.ini$',
        r'.*\.toml$',
        r'.*\.properties$',
        r'package\.json$',
        r'requirements\.txt$',
        r'Cargo\.toml$',
        r'pom\.xml$',
        r'build\.gradle$'
    ]
    
    BUILD_PATTERNS = [
        r'Makefile$',
        r'.*\.mk$',
        r'build\..*',
        r'.*\.gradle$',
        r'.*\.maven$',
        r'CMakeLists\.txt$',
        r'setup\.py$',
        r'Dockerfile.*'
    ]
    
    DOC_PATTERNS = [
        r'README.*',
        r'.*\.md$',
        r'.*\.rst$',
        r'.*\.txt$',
        r'docs/.*',
        r'documentation/.*'
    ]
    
    @classmethod
    def classify_file_type(cls, file_path: str, language: Optional[Language] = None) -> FileType:
        """Classify file type based on path and language"""
        path_lower = file_path.lower()
        
        # Check test files
        for pattern in cls.TEST_PATTERNS:
            if re.search(pattern, path_lower):
                return FileType.TEST
        
        # Check configuration files
        for pattern in cls.CONFIG_PATTERNS:
            if re.search(pattern, path_lower):
                return FileType.CONFIGURATION
        
        # Check build scripts
        for pattern in cls.BUILD_PATTERNS:
            if re.search(pattern, path_lower):
                return FileType.BUILD_SCRIPT
        
        # Check documentation
        for pattern in cls.DOC_PATTERNS:
            if re.search(pattern, path_lower):
                return FileType.DOCUMENTATION
        
        # Default to source code for programming languages
        if language and language in [Language.PYTHON, Language.JAVASCRIPT, Language.JAVA, 
                                    Language.CSHARP, Language.GO, Language.RUST, Language.CPP, Language.C]:
            return FileType.SOURCE_CODE
        
        return FileType.RESOURCE

class PythonAnalyzer:
    """Python-specific code analyzer"""
    
    def analyze_file(self, content: str, file_path: str) -> FileAnalysis:
        """Analyze Python file"""
        start_time = time.time()
        
        try:
            tree = ast.parse(content)
            
            analysis = FileAnalysis(
                file_path=file_path,
                language=Language.PYTHON,
                file_type=FileTypeClassifier.classify_file_type(file_path, Language.PYTHON),
                size_bytes=len(content.encode('utf-8'))
            )
            
            # Calculate basic metrics
            analysis.metrics = self._calculate_metrics(content, tree)
            
            # Extract entities
            analysis.entities = self._extract_entities(tree)
            
            # Extract dependencies
            analysis.dependencies = self._extract_dependencies(tree)
            
            # Security analysis
            analysis.security_issues = self._analyze_security(tree, content)
            
            # Documentation coverage
            analysis.documentation_coverage = self._calculate_doc_coverage(tree)
            
            # Quality score
            analysis.quality_score = self._calculate_quality_score(analysis)
            
            CODE_ANALYSIS_REQUESTS.labels(language='python', status='success').inc()
            FILES_ANALYZED.labels(language='python', file_type=analysis.file_type.value).inc()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Python analysis failed for {file_path}: {e}")
            CODE_ANALYSIS_REQUESTS.labels(language='python', status='error').inc()
            
            # Return basic analysis
            return FileAnalysis(
                file_path=file_path,
                language=Language.PYTHON,
                file_type=FileTypeClassifier.classify_file_type(file_path, Language.PYTHON),
                size_bytes=len(content.encode('utf-8')),
                metrics=self._calculate_basic_metrics(content)
            )
        
        finally:
            latency = time.time() - start_time
            CODE_ANALYSIS_LATENCY.labels(language='python').observe(latency)
    
    def _calculate_metrics(self, content: str, tree: ast.AST) -> CodeMetrics:
        """Calculate code metrics for Python"""
        lines = content.split('\n')
        
        metrics = CodeMetrics()
        metrics.lines_of_code = len(lines)
        metrics.blank_lines = sum(1 for line in lines if not line.strip())
        metrics.lines_of_comments = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Calculate cyclomatic complexity
        metrics.cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
        
        # Calculate cognitive complexity (simplified)
        metrics.cognitive_complexity = self._calculate_cognitive_complexity(tree)
        
        # Calculate maintainability index (simplified)
        if metrics.lines_of_code > 0:
            halstead_volume = metrics.lines_of_code * 2  # Simplified
            metrics.maintainability_index = max(0, 171 - 5.2 * math.log(halstead_volume) - 
                                               0.23 * metrics.cyclomatic_complexity - 
                                               16.2 * math.log(metrics.lines_of_code))
        
        return metrics
    
    def _calculate_basic_metrics(self, content: str) -> CodeMetrics:
        """Calculate basic metrics when AST parsing fails"""
        lines = content.split('\n')
        
        return CodeMetrics(
            lines_of_code=len(lines),
            blank_lines=sum(1 for line in lines if not line.strip()),
            lines_of_comments=sum(1 for line in lines if line.strip().startswith('#'))
        )
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (simplified)"""
        complexity = 0
        nesting_level = 0
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting = 0
            
            def visit_If(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
            
            def visit_While(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
            
            def visit_For(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        return visitor.complexity
    
    def _extract_entities(self, tree: ast.AST) -> List[CodeEntity]:
        """Extract code entities from AST"""
        entities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                entity = CodeEntity(
                    name=node.name,
                    type='function',
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    parameters=[arg.arg for arg in node.args.args],
                    docstring=ast.get_docstring(node)
                )
                entities.append(entity)
            
            elif isinstance(node, ast.ClassDef):
                entity = CodeEntity(
                    name=node.name,
                    type='class',
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    docstring=ast.get_docstring(node)
                )
                entities.append(entity)
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        entity = CodeEntity(
                            name=target.id,
                            type='variable',
                            line_start=node.lineno,
                            line_end=node.lineno
                        )
                        entities.append(entity)
        
        return entities
    
    def _extract_dependencies(self, tree: ast.AST) -> List[Dependency]:
        """Extract dependencies from imports"""
        dependencies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dep = Dependency(
                        name=alias.name,
                        type='import',
                        source=self._classify_import_source(alias.name),
                        line_number=node.lineno
                    )
                    dependencies.append(dep)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    dep = Dependency(
                        name=f"{module}.{alias.name}" if module else alias.name,
                        type='from_import',
                        source=self._classify_import_source(module),
                        line_number=node.lineno
                    )
                    dependencies.append(dep)
        
        return dependencies
    
    def _classify_import_source(self, module_name: str) -> str:
        """Classify import source as standard library, external, or local"""
        if not module_name:
            return 'local'
        
        # Python standard library modules (partial list)
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'time', 'math', 'random', 're',
            'collections', 'itertools', 'functools', 'typing', 'pathlib',
            'urllib', 'http', 'email', 'html', 'xml', 'sqlite3', 'csv',
            'logging', 'unittest', 'asyncio', 'threading', 'multiprocessing'
        }
        
        root_module = module_name.split('.')[0]
        
        if root_module in stdlib_modules:
            return 'standard_library'
        elif '.' in module_name or module_name.startswith('_'):
            return 'local'
        else:
            return 'external'
    
    def _analyze_security(self, tree: ast.AST, content: str) -> List[SecurityIssue]:
        """Analyze security issues in Python code"""
        issues = []
        
        # Check for common security issues
        for node in ast.walk(tree):
            # SQL injection potential
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr in ['execute', 'executemany']):
                    # Check if query is constructed with string formatting
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                            issues.append(SecurityIssue(
                                severity='high',
                                type='sql_injection',
                                description='Potential SQL injection vulnerability',
                                line_number=node.lineno,
                                cwe_id='CWE-89'
                            ))
            
            # eval() usage
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id == 'eval'):
                issues.append(SecurityIssue(
                    severity='critical',
                    type='code_injection',
                    description='Use of eval() function',
                    line_number=node.lineno,
                    cwe_id='CWE-95'
                ))
            
            # exec() usage
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id == 'exec'):
                issues.append(SecurityIssue(
                    severity='critical',
                    type='code_injection',
                    description='Use of exec() function',
                    line_number=node.lineno,
                    cwe_id='CWE-95'
                ))
        
        # Check for hardcoded secrets in content
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded_password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'hardcoded_api_key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'hardcoded_secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'hardcoded_token')
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, issue_type in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        severity='medium',
                        type=issue_type,
                        description=f'Potential hardcoded secret: {issue_type}',
                        line_number=i,
                        cwe_id='CWE-798'
                    ))
        
        return issues
    
    def _calculate_doc_coverage(self, tree: ast.AST) -> float:
        """Calculate documentation coverage"""
        total_functions = 0
        documented_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                if ast.get_docstring(node):
                    documented_functions += 1
        
        if total_functions == 0:
            return 1.0
        
        return documented_functions / total_functions
    
    def _calculate_quality_score(self, analysis: FileAnalysis) -> float:
        """Calculate overall quality score"""
        score = 100.0
        
        # Penalize high complexity
        if analysis.metrics.cyclomatic_complexity > 10:
            score -= (analysis.metrics.cyclomatic_complexity - 10) * 2
        
        # Penalize low documentation coverage
        score -= (1.0 - analysis.documentation_coverage) * 20
        
        # Penalize security issues
        for issue in analysis.security_issues:
            if issue.severity == 'critical':
                score -= 30
            elif issue.severity == 'high':
                score -= 20
            elif issue.severity == 'medium':
                score -= 10
            elif issue.severity == 'low':
                score -= 5
        
        # Penalize very long files
        if analysis.metrics.lines_of_code > 1000:
            score -= (analysis.metrics.lines_of_code - 1000) / 100
        
        return max(0.0, min(100.0, score))

class JavaScriptAnalyzer:
    """JavaScript/TypeScript analyzer"""
    
    def analyze_file(self, content: str, file_path: str) -> FileAnalysis:
        """Analyze JavaScript/TypeScript file"""
        start_time = time.time()
        
        try:
            language = Language.TYPESCRIPT if file_path.endswith(('.ts', '.tsx')) else Language.JAVASCRIPT
            
            analysis = FileAnalysis(
                file_path=file_path,
                language=language,
                file_type=FileTypeClassifier.classify_file_type(file_path, language),
                size_bytes=len(content.encode('utf-8'))
            )
            
            # Calculate basic metrics
            analysis.metrics = self._calculate_metrics(content)
            
            # Extract entities (simplified)
            analysis.entities = self._extract_entities(content)
            
            # Extract dependencies
            analysis.dependencies = self._extract_dependencies(content)
            
            # Security analysis
            analysis.security_issues = self._analyze_security(content)
            
            # Quality score
            analysis.quality_score = self._calculate_quality_score(analysis)
            
            CODE_ANALYSIS_REQUESTS.labels(language=language.value, status='success').inc()
            FILES_ANALYZED.labels(language=language.value, file_type=analysis.file_type.value).inc()
            
            return analysis
            
        except Exception as e:
            logger.error(f"JavaScript analysis failed for {file_path}: {e}")
            CODE_ANALYSIS_REQUESTS.labels(language='javascript', status='error').inc()
            
            return FileAnalysis(
                file_path=file_path,
                language=Language.JAVASCRIPT,
                file_type=FileTypeClassifier.classify_file_type(file_path, Language.JAVASCRIPT),
                size_bytes=len(content.encode('utf-8')),
                metrics=self._calculate_basic_metrics(content)
            )
        
        finally:
            latency = time.time() - start_time
            CODE_ANALYSIS_LATENCY.labels(language='javascript').observe(latency)
    
    def _calculate_metrics(self, content: str) -> CodeMetrics:
        """Calculate metrics for JavaScript"""
        lines = content.split('\n')
        
        metrics = CodeMetrics()
        metrics.lines_of_code = len(lines)
        metrics.blank_lines = sum(1 for line in lines if not line.strip())
        metrics.lines_of_comments = sum(1 for line in lines 
                                      if line.strip().startswith('//') or 
                                         line.strip().startswith('/*') or
                                         line.strip().startswith('*'))
        
        # Simple complexity calculation
        complexity_keywords = ['if', 'else', 'while', 'for', 'switch', 'case', 'catch', 'try']
        metrics.cyclomatic_complexity = 1
        
        for line in lines:
            for keyword in complexity_keywords:
                metrics.cyclomatic_complexity += line.count(keyword)
        
        return metrics
    
    def _calculate_basic_metrics(self, content: str) -> CodeMetrics:
        """Calculate basic metrics"""
        return self._calculate_metrics(content)
    
    def _extract_entities(self, content: str) -> List[CodeEntity]:
        """Extract entities using regex patterns"""
        entities = []
        lines = content.split('\n')
        
        # Function patterns
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'(\w+)\s*=\s*function\s*\(',
            r'(\w+)\s*=>\s*',
            r'async\s+function\s+(\w+)\s*\('
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in function_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    entities.append(CodeEntity(
                        name=match.group(1),
                        type='function',
                        line_start=i,
                        line_end=i
                    ))
        
        # Class patterns
        class_pattern = r'class\s+(\w+)'
        for i, line in enumerate(lines, 1):
            matches = re.finditer(class_pattern, line)
            for match in matches:
                entities.append(CodeEntity(
                    name=match.group(1),
                    type='class',
                    line_start=i,
                    line_end=i
                ))
        
        return entities
    
    def _extract_dependencies(self, content: str) -> List[Dependency]:
        """Extract dependencies from imports/requires"""
        dependencies = []
        lines = content.split('\n')
        
        # Import patterns
        import_patterns = [
            r'import\s+.*\s+from\s+["\']([^"\']+)["\']',
            r'import\s+["\']([^"\']+)["\']',
            r'require\s*\(\s*["\']([^"\']+)["\']\s*\)',
            r'import\s*\(\s*["\']([^"\']+)["\']\s*\)'
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in import_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    module_name = match.group(1)
                    dependencies.append(Dependency(
                        name=module_name,
                        type='import',
                        source=self._classify_import_source(module_name),
                        line_number=i
                    ))
        
        return dependencies
    
    def _classify_import_source(self, module_name: str) -> str:
        """Classify import source"""
        if module_name.startswith('.'):
            return 'local'
        elif module_name.startswith('@') or '/' in module_name:
            return 'external'
        else:
            # Check if it's a Node.js built-in module
            builtin_modules = {
                'fs', 'path', 'http', 'https', 'url', 'crypto', 'os', 'util',
                'events', 'stream', 'buffer', 'child_process', 'cluster'
            }
            if module_name in builtin_modules:
                return 'standard_library'
            return 'external'
    
    def _analyze_security(self, content: str) -> List[SecurityIssue]:
        """Analyze security issues"""
        issues = []
        lines = content.split('\n')
        
        # Check for dangerous functions
        dangerous_patterns = [
            (r'eval\s*\(', 'code_injection', 'Use of eval() function'),
            (r'innerHTML\s*=', 'xss', 'Potential XSS vulnerability with innerHTML'),
            (r'document\.write\s*\(', 'xss', 'Potential XSS vulnerability with document.write'),
            (r'setTimeout\s*\(\s*["\']', 'code_injection', 'String passed to setTimeout'),
            (r'setInterval\s*\(\s*["\']', 'code_injection', 'String passed to setInterval')
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, issue_type, description in dangerous_patterns:
                if re.search(pattern, line):
                    issues.append(SecurityIssue(
                        severity='high',
                        type=issue_type,
                        description=description,
                        line_number=i
                    ))
        
        return issues
    
    def _calculate_quality_score(self, analysis: FileAnalysis) -> float:
        """Calculate quality score"""
        score = 100.0
        
        # Penalize high complexity
        if analysis.metrics.cyclomatic_complexity > 15:
            score -= (analysis.metrics.cyclomatic_complexity - 15) * 2
        
        # Penalize security issues
        for issue in analysis.security_issues:
            if issue.severity == 'critical':
                score -= 30
            elif issue.severity == 'high':
                score -= 20
            elif issue.severity == 'medium':
                score -= 10
        
        return max(0.0, min(100.0, score))

class CodeAnalyzer:
    """Main code analyzer that coordinates language-specific analyzers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.python_analyzer = PythonAnalyzer()
        self.javascript_analyzer = JavaScriptAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # Start metrics server
        start_http_server(8004)
    
    async def analyze_repository(self, repo_path: str, repository_id: str, 
                               repository_name: str) -> RepositoryAnalysis:
        """Analyze entire repository"""
        start_time = time.time()
        
        logger.info(f"Starting analysis of repository: {repository_name}")
        
        analysis = RepositoryAnalysis(
            repository_id=repository_id,
            repository_name=repository_name,
            total_files=0,
            analyzed_files=0
        )
        
        # Find all files to analyze
        files_to_analyze = []
        for root, dirs, files in os.walk(repo_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                # Skip binary files and very large files
                if self._should_analyze_file(file_path):
                    files_to_analyze.append((file_path, relative_path))
        
        analysis.total_files = len(files_to_analyze)
        
        # Analyze files in parallel
        tasks = []
        for file_path, relative_path in files_to_analyze:
            task = self._analyze_file_async(file_path, relative_path)
            tasks.append(task)
        
        # Process results
        file_analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in file_analyses:
            if isinstance(result, FileAnalysis):
                analysis.file_analyses.append(result)
                analysis.analyzed_files += 1
                
                # Update language counts
                if result.language in analysis.languages:
                    analysis.languages[result.language] += 1
                else:
                    analysis.languages[result.language] = 1
                
                # Update file type counts
                if result.file_type in analysis.file_types:
                    analysis.file_types[result.file_type] += 1
                else:
                    analysis.file_types[result.file_type] = 1
        
        # Calculate overall metrics
        analysis.overall_metrics = self._calculate_overall_metrics(analysis.file_analyses)
        
        # Build dependency graph
        analysis.dependency_graph = self._build_dependency_graph(analysis.file_analyses)
        
        # Security summary
        analysis.security_summary = self._calculate_security_summary(analysis.file_analyses)
        
        # Update complexity metrics
        for language, count in analysis.languages.items():
            avg_complexity = sum(fa.metrics.cyclomatic_complexity for fa in analysis.file_analyses 
                               if fa.language == language) / max(count, 1)
            COMPLEXITY_SCORE.labels(repository=repository_name, language=language.value).set(avg_complexity)
        
        logger.info(f"Repository analysis completed: {analysis.analyzed_files}/{analysis.total_files} files analyzed")
        
        return analysis
    
    def _should_analyze_file(self, file_path: str) -> bool:
        """Determine if file should be analyzed"""
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
        
        # Check if we can detect the language
        language = LanguageDetector.detect_language(file_path)
        return language is not None
    
    async def _analyze_file_async(self, file_path: str, relative_path: str) -> Optional[FileAnalysis]:
        """Analyze single file asynchronously"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Detect language
            language = LanguageDetector.detect_language(relative_path, content)
            if not language:
                return None
            
            # Run analysis in thread pool
            loop = asyncio.get_event_loop()
            
            if language == Language.PYTHON:
                analysis = await loop.run_in_executor(
                    self.executor, 
                    self.python_analyzer.analyze_file, 
                    content, 
                    relative_path
                )
            elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                analysis = await loop.run_in_executor(
                    self.executor, 
                    self.javascript_analyzer.analyze_file, 
                    content, 
                    relative_path
                )
            else:
                # Basic analysis for other languages
                analysis = FileAnalysis(
                    file_path=relative_path,
                    language=language,
                    file_type=FileTypeClassifier.classify_file_type(relative_path, language),
                    size_bytes=len(content.encode('utf-8')),
                    metrics=self._calculate_basic_metrics(content)
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze file {relative_path}: {e}")
            return None
    
    def _calculate_basic_metrics(self, content: str) -> CodeMetrics:
        """Calculate basic metrics for unsupported languages"""
        lines = content.split('\n')
        
        return CodeMetrics(
            lines_of_code=len(lines),
            blank_lines=sum(1 for line in lines if not line.strip()),
            lines_of_comments=sum(1 for line in lines if line.strip().startswith('#') or 
                                line.strip().startswith('//') or 
                                line.strip().startswith('/*'))
        )
    
    def _calculate_overall_metrics(self, file_analyses: List[FileAnalysis]) -> CodeMetrics:
        """Calculate overall repository metrics"""
        if not file_analyses:
            return CodeMetrics()
        
        total_metrics = CodeMetrics()
        
        for analysis in file_analyses:
            total_metrics.lines_of_code += analysis.metrics.lines_of_code
            total_metrics.lines_of_comments += analysis.metrics.lines_of_comments
            total_metrics.blank_lines += analysis.metrics.blank_lines
            total_metrics.cyclomatic_complexity += analysis.metrics.cyclomatic_complexity
        
        # Calculate averages
        num_files = len(file_analyses)
        total_metrics.cyclomatic_complexity = total_metrics.cyclomatic_complexity / num_files
        
        return total_metrics
    
    def _build_dependency_graph(self, file_analyses: List[FileAnalysis]) -> Dict[str, List[str]]:
        """Build dependency graph from file analyses"""
        graph = {}
        
        for analysis in file_analyses:
            dependencies = []
            for dep in analysis.dependencies:
                if dep.source == 'local':
                    dependencies.append(dep.name)
            
            if dependencies:
                graph[analysis.file_path] = dependencies
        
        return graph
    
    def _calculate_security_summary(self, file_analyses: List[FileAnalysis]) -> Dict[str, int]:
        """Calculate security issue summary"""
        summary = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for analysis in file_analyses:
            for issue in analysis.security_issues:
                if issue.severity in summary:
                    summary[issue.severity] += 1
        
        return summary
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'analyzers': {
                'python': 'available',
                'javascript': 'available',
                'java': 'available' if JAVA_AVAILABLE else 'unavailable',
                'tree_sitter': 'available' if TREE_SITTER_AVAILABLE else 'unavailable'
            }
        }

# Import math for maintainability index calculation
import math

if __name__ == "__main__":
    import asyncio
    
    async def main():
        config = {'max_workers': 4}
        analyzer = CodeAnalyzer(config)
        
        # Example usage
        repo_path = "/path/to/repository"
        if os.path.exists(repo_path):
            analysis = await analyzer.analyze_repository(repo_path, "repo-001", "test-repo")
            print(f"Analysis completed: {analysis.analyzed_files} files analyzed")
            print(f"Languages: {analysis.languages}")
            print(f"Security issues: {analysis.security_summary}")
        
        # Health check
        health = await analyzer.health_check()
        print(f"Health: {health}")
    
    asyncio.run(main())

