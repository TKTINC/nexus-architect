"""
Agentic Transformation Engine for Nexus Architect
Automated refactoring, framework migration, and code transformation capabilities
"""

import ast
import os
import re
import json
import logging
import asyncio
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import difflib

import aiohttp
import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Transformation Models
@dataclass
class RefactoringRule:
    """Represents a refactoring rule"""
    rule_id: str
    name: str
    description: str
    pattern: str
    replacement: str
    confidence: float
    risk_level: str
    applicable_languages: List[str]

@dataclass
class TransformationResult:
    """Represents the result of a transformation"""
    file_path: str
    original_content: str
    transformed_content: str
    applied_rules: List[str]
    confidence_score: float
    risk_assessment: str
    validation_status: str
    backup_created: bool

@dataclass
class FrameworkMigration:
    """Represents a framework migration plan"""
    source_framework: str
    target_framework: str
    migration_steps: List[str]
    compatibility_issues: List[str]
    estimated_effort: int
    risk_level: str
    success_probability: float

@dataclass
class CodeGeneration:
    """Represents generated code"""
    generation_type: str
    template_used: str
    generated_content: str
    target_location: str
    dependencies: List[str]
    validation_tests: List[str]

# Request/Response Models
class TransformationRequest(BaseModel):
    repository_url: str
    branch: str = "main"
    transformation_type: str = Field(..., regex="^(refactor|migrate|modernize|generate)$")
    target_framework: Optional[str] = None
    rules: List[str] = []
    options: Dict[str, Any] = {}

class TransformationResponse(BaseModel):
    transformation_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class AgenticTransformationEngine:
    """
    Comprehensive agentic transformation engine with automated refactoring,
    framework migration, and intelligent code generation capabilities.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize refactoring rules
        self.refactoring_rules = self._initialize_refactoring_rules()
        
        # Framework migration mappings
        self.framework_migrations = self._initialize_framework_migrations()
        
        # Code generation templates
        self.code_templates = self._initialize_code_templates()
        
        # AST transformers
        self.ast_transformers = self._initialize_ast_transformers()
    
    def _initialize_refactoring_rules(self) -> Dict[str, RefactoringRule]:
        """Initialize comprehensive refactoring rules"""
        return {
            "extract_method": RefactoringRule(
                rule_id="extract_method",
                name="Extract Method",
                description="Extract long methods into smaller, focused methods",
                pattern=r"def\s+(\w+)\(.*?\):\s*\n((?:\s{4,}.*\n){20,})",
                replacement="# Method extracted - see generated methods",
                confidence=0.85,
                risk_level="low",
                applicable_languages=["python"]
            ),
            "rename_variable": RefactoringRule(
                rule_id="rename_variable",
                name="Rename Variable",
                description="Rename variables to follow naming conventions",
                pattern=r"\b([a-z])([A-Z])",
                replacement=r"\1_\2",
                confidence=0.95,
                risk_level="low",
                applicable_languages=["python"]
            ),
            "remove_dead_code": RefactoringRule(
                rule_id="remove_dead_code",
                name="Remove Dead Code",
                description="Remove unreachable or unused code",
                pattern=r"if\s+False:\s*\n((?:\s{4,}.*\n)*)",
                replacement="",
                confidence=0.90,
                risk_level="medium",
                applicable_languages=["python", "javascript"]
            ),
            "simplify_conditionals": RefactoringRule(
                rule_id="simplify_conditionals",
                name="Simplify Conditionals",
                description="Simplify complex conditional expressions",
                pattern=r"if\s+(.+)\s+==\s+True:",
                replacement=r"if \1:",
                confidence=0.95,
                risk_level="low",
                applicable_languages=["python"]
            ),
            "extract_constant": RefactoringRule(
                rule_id="extract_constant",
                name="Extract Constant",
                description="Extract magic numbers into named constants",
                pattern=r"\b(\d{2,})\b",
                replacement="CONSTANT_VALUE",
                confidence=0.70,
                risk_level="low",
                applicable_languages=["python", "javascript", "java"]
            ),
            "modernize_string_formatting": RefactoringRule(
                rule_id="modernize_string_formatting",
                name="Modernize String Formatting",
                description="Convert old string formatting to f-strings",
                pattern=r'"([^"]*?)%[sd]"\s*%\s*([^,\n]+)',
                replacement=r'f"\1{\2}"',
                confidence=0.85,
                risk_level="low",
                applicable_languages=["python"]
            ),
            "replace_deprecated_imports": RefactoringRule(
                rule_id="replace_deprecated_imports",
                name="Replace Deprecated Imports",
                description="Replace deprecated import statements",
                pattern=r"from\s+imp\s+import",
                replacement="from importlib import",
                confidence=0.90,
                risk_level="medium",
                applicable_languages=["python"]
            ),
            "add_type_hints": RefactoringRule(
                rule_id="add_type_hints",
                name="Add Type Hints",
                description="Add type hints to function parameters and returns",
                pattern=r"def\s+(\w+)\(([^)]*)\):",
                replacement=r"def \1(\2) -> None:",
                confidence=0.60,
                risk_level="low",
                applicable_languages=["python"]
            )
        }
    
    def _initialize_framework_migrations(self) -> Dict[str, FrameworkMigration]:
        """Initialize framework migration configurations"""
        return {
            "flask_to_fastapi": FrameworkMigration(
                source_framework="flask",
                target_framework="fastapi",
                migration_steps=[
                    "Convert Flask app to FastAPI app",
                    "Replace @app.route with @app.get/@app.post",
                    "Convert request handling to Pydantic models",
                    "Update error handling to HTTPException",
                    "Migrate middleware to FastAPI middleware",
                    "Update testing framework"
                ],
                compatibility_issues=[
                    "Session handling differences",
                    "Template rendering changes",
                    "Blueprint to router conversion"
                ],
                estimated_effort=40,
                risk_level="medium",
                success_probability=0.85
            ),
            "django_to_fastapi": FrameworkMigration(
                source_framework="django",
                target_framework="fastapi",
                migration_steps=[
                    "Extract business logic from views",
                    "Convert Django models to Pydantic models",
                    "Migrate URL patterns to FastAPI routes",
                    "Convert Django ORM to SQLAlchemy",
                    "Update authentication system",
                    "Migrate admin interface"
                ],
                compatibility_issues=[
                    "ORM differences",
                    "Admin interface replacement",
                    "Middleware system changes"
                ],
                estimated_effort=80,
                risk_level="high",
                success_probability=0.70
            ),
            "express_to_fastify": FrameworkMigration(
                source_framework="express",
                target_framework="fastify",
                migration_steps=[
                    "Convert Express app to Fastify app",
                    "Update middleware registration",
                    "Convert route handlers",
                    "Update request/response handling",
                    "Migrate plugins and decorators"
                ],
                compatibility_issues=[
                    "Middleware API differences",
                    "Plugin system changes",
                    "Request/response object differences"
                ],
                estimated_effort=30,
                risk_level="medium",
                success_probability=0.90
            )
        }
    
    def _initialize_code_templates(self) -> Dict[str, str]:
        """Initialize code generation templates"""
        return {
            "fastapi_crud": '''
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import asyncpg

class {model_name}Base(BaseModel):
    {fields}

class {model_name}Create({model_name}Base):
    pass

class {model_name}Update({model_name}Base):
    pass

class {model_name}({model_name}Base):
    id: int
    
    class Config:
        orm_mode = True

@app.post("/{endpoint}/", response_model={model_name})
async def create_{endpoint}(item: {model_name}Create, db: asyncpg.Pool = Depends(get_db)):
    # Implementation here
    pass

@app.get("/{endpoint}/", response_model=List[{model_name}])
async def read_{endpoint}s(skip: int = 0, limit: int = 100, db: asyncpg.Pool = Depends(get_db)):
    # Implementation here
    pass

@app.get("/{endpoint}/{{item_id}}", response_model={model_name})
async def read_{endpoint}(item_id: int, db: asyncpg.Pool = Depends(get_db)):
    # Implementation here
    pass

@app.put("/{endpoint}/{{item_id}}", response_model={model_name})
async def update_{endpoint}(item_id: int, item: {model_name}Update, db: asyncpg.Pool = Depends(get_db)):
    # Implementation here
    pass

@app.delete("/{endpoint}/{{item_id}}")
async def delete_{endpoint}(item_id: int, db: asyncpg.Pool = Depends(get_db)):
    # Implementation here
    pass
''',
            "microservice_template": '''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="{service_name}",
    description="{service_description}",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "service": "{service_name}"}}

@app.get("/")
async def root():
    return {{"message": "Welcome to {service_name}"}}

if __name__ == "__main__":
    uvicorn.run(
        "{module_name}:app",
        host="0.0.0.0",
        port={port},
        reload=True
    )
''',
            "database_model": '''
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class {model_name}(Base):
    __tablename__ = "{table_name}"
    
    id = Column(Integer, primary_key=True, index=True)
    {columns}
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<{model_name}(id={{self.id}})>"
''',
            "test_template": '''
import pytest
import asyncio
from httpx import AsyncClient
from {module_name} import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_{function_name}(client: AsyncClient):
    response = await client.{method}("{endpoint}")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.asyncio
async def test_{function_name}_error_handling(client: AsyncClient):
    response = await client.{method}("{endpoint}/invalid")
    assert response.status_code == 404
'''
        }
    
    def _initialize_ast_transformers(self) -> Dict[str, Any]:
        """Initialize AST transformers for code refactoring"""
        return {
            "method_extractor": self._extract_long_methods,
            "variable_renamer": self._rename_variables,
            "import_modernizer": self._modernize_imports,
            "type_hint_adder": self._add_type_hints,
            "constant_extractor": self._extract_constants
        }
    
    async def start_transformation(self, request: TransformationRequest) -> str:
        """Start agentic transformation process"""
        transformation_id = f"transform_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.repository_url) % 10000}"
        
        # Store transformation request
        await self.redis_client.hset(
            f"transformation:{transformation_id}",
            mapping={
                "status": "started",
                "progress": "0",
                "repository_url": request.repository_url,
                "branch": request.branch,
                "transformation_type": request.transformation_type,
                "target_framework": request.target_framework or "",
                "started_at": datetime.now().isoformat()
            }
        )
        
        # Start background transformation
        asyncio.create_task(self._perform_transformation(transformation_id, request))
        
        return transformation_id
    
    async def _perform_transformation(self, transformation_id: str, request: TransformationRequest):
        """Perform comprehensive agentic transformation"""
        try:
            # Update progress
            await self._update_progress(transformation_id, 10, "Cloning repository")
            
            # Clone repository
            repo_path = await self._clone_repository(request.repository_url, request.branch)
            
            # Update progress
            await self._update_progress(transformation_id, 20, "Analyzing codebase")
            
            # Analyze codebase
            analysis_results = await self._analyze_codebase(repo_path)
            
            # Update progress
            await self._update_progress(transformation_id, 40, "Applying transformations")
            
            # Apply transformations based on type
            transformation_results = []
            
            if request.transformation_type == "refactor":
                transformation_results = await self._apply_refactoring(repo_path, request.rules, analysis_results)
            elif request.transformation_type == "migrate":
                transformation_results = await self._apply_framework_migration(repo_path, request.target_framework, analysis_results)
            elif request.transformation_type == "modernize":
                transformation_results = await self._apply_modernization(repo_path, analysis_results)
            elif request.transformation_type == "generate":
                transformation_results = await self._generate_code(repo_path, request.options, analysis_results)
            
            # Update progress
            await self._update_progress(transformation_id, 70, "Validating transformations")
            
            # Validate transformations
            validation_results = await self._validate_transformations(repo_path, transformation_results)
            
            # Update progress
            await self._update_progress(transformation_id, 90, "Generating reports")
            
            # Generate transformation report
            report = await self._generate_transformation_report(
                transformation_results, validation_results, analysis_results
            )
            
            # Store results
            await self._store_transformation_results(transformation_id, {
                "transformation_results": [asdict(result) for result in transformation_results],
                "validation_results": validation_results,
                "report": report,
                "analysis_results": analysis_results
            })
            
            # Update final status
            await self._update_progress(transformation_id, 100, "Transformation completed")
            
            # Cleanup
            await self._cleanup_repository(repo_path)
            
        except Exception as e:
            self.logger.error(f"Transformation failed for {transformation_id}: {str(e)}")
            await self.redis_client.hset(
                f"transformation:{transformation_id}",
                mapping={
                    "status": "failed",
                    "error_message": str(e),
                    "completed_at": datetime.now().isoformat()
                }
            )
    
    async def _clone_repository(self, repo_url: str, branch: str) -> str:
        """Clone repository for transformation"""
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
    
    async def _analyze_codebase(self, repo_path: str) -> Dict[str, Any]:
        """Analyze codebase for transformation planning"""
        analysis = {
            "files": [],
            "languages": defaultdict(int),
            "frameworks": [],
            "dependencies": [],
            "complexity_metrics": {},
            "refactoring_opportunities": []
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
                
                if file_ext in language_extensions:
                    language = language_extensions[file_ext]
                    analysis["languages"][language] += 1
                    
                    # Analyze file content
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        file_analysis = await self._analyze_file_content(file_path, content, language)
                        analysis["files"].append(file_analysis)
                        
                        # Detect frameworks
                        frameworks = self._detect_frameworks(content, language)
                        analysis["frameworks"].extend(frameworks)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze file {file_path}: {str(e)}")
        
        # Remove duplicates from frameworks
        analysis["frameworks"] = list(set(analysis["frameworks"]))
        
        return analysis
    
    async def _analyze_file_content(self, file_path: str, content: str, language: str) -> Dict[str, Any]:
        """Analyze individual file content"""
        analysis = {
            "file_path": file_path,
            "language": language,
            "lines_of_code": len(content.splitlines()),
            "complexity_score": 0,
            "refactoring_opportunities": [],
            "modernization_suggestions": []
        }
        
        if language == "Python":
            try:
                tree = ast.parse(content)
                analysis["complexity_score"] = self._calculate_file_complexity(tree)
                analysis["refactoring_opportunities"] = self._identify_refactoring_opportunities(tree, content)
                analysis["modernization_suggestions"] = self._identify_modernization_opportunities(content)
            except SyntaxError:
                analysis["complexity_score"] = 0
        
        return analysis
    
    def _calculate_file_complexity(self, tree: ast.AST) -> int:
        """Calculate file complexity score"""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += self._calculate_function_complexity(node)
            elif isinstance(node, ast.ClassDef):
                complexity += len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
        
        return complexity
    
    def _calculate_function_complexity(self, node: ast.AST) -> int:
        """Calculate function complexity"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        
        return complexity
    
    def _identify_refactoring_opportunities(self, tree: ast.AST, content: str) -> List[str]:
        """Identify refactoring opportunities"""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check for long methods
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    method_lines = node.end_lineno - node.lineno
                    if method_lines > 30:
                        opportunities.append(f"Long method '{node.name}' ({method_lines} lines)")
                
                # Check for high complexity
                complexity = self._calculate_function_complexity(node)
                if complexity > 10:
                    opportunities.append(f"High complexity method '{node.name}' (complexity: {complexity})")
            
            elif isinstance(node, ast.ClassDef):
                # Check for god classes
                method_count = len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
                if method_count > 15:
                    opportunities.append(f"God class '{node.name}' ({method_count} methods)")
        
        return opportunities
    
    def _identify_modernization_opportunities(self, content: str) -> List[str]:
        """Identify modernization opportunities"""
        opportunities = []
        
        # Check for old string formatting
        if re.search(r'%[sd]', content):
            opportunities.append("Old string formatting detected - consider f-strings")
        
        # Check for deprecated imports
        if re.search(r'from imp import', content):
            opportunities.append("Deprecated 'imp' module - use 'importlib'")
        
        # Check for missing type hints
        if re.search(r'def\s+\w+\([^)]*\):', content) and not re.search(r'->', content):
            opportunities.append("Missing type hints - consider adding them")
        
        return opportunities
    
    def _detect_frameworks(self, content: str, language: str) -> List[str]:
        """Detect frameworks used in the code"""
        frameworks = []
        
        if language == "Python":
            if "from flask import" in content or "import flask" in content:
                frameworks.append("Flask")
            if "from django" in content or "import django" in content:
                frameworks.append("Django")
            if "from fastapi import" in content or "import fastapi" in content:
                frameworks.append("FastAPI")
            if "from sqlalchemy import" in content:
                frameworks.append("SQLAlchemy")
        
        elif language == "JavaScript":
            if "require('express')" in content or "from 'express'" in content:
                frameworks.append("Express")
            if "require('react')" in content or "from 'react'" in content:
                frameworks.append("React")
            if "require('vue')" in content or "from 'vue'" in content:
                frameworks.append("Vue")
        
        return frameworks
    
    async def _apply_refactoring(self, repo_path: str, rules: List[str], analysis_results: Dict[str, Any]) -> List[TransformationResult]:
        """Apply refactoring transformations"""
        results = []
        
        # If no specific rules provided, apply all applicable rules
        if not rules:
            rules = list(self.refactoring_rules.keys())
        
        for file_info in analysis_results["files"]:
            if file_info["language"] == "Python":
                file_path = file_info["file_path"]
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    transformed_content = original_content
                    applied_rules = []
                    
                    # Apply each refactoring rule
                    for rule_name in rules:
                        if rule_name in self.refactoring_rules:
                            rule = self.refactoring_rules[rule_name]
                            
                            if file_info["language"].lower() in rule.applicable_languages:
                                new_content = await self._apply_refactoring_rule(transformed_content, rule)
                                
                                if new_content != transformed_content:
                                    transformed_content = new_content
                                    applied_rules.append(rule_name)
                    
                    # Only create result if transformations were applied
                    if applied_rules:
                        # Create backup
                        backup_path = f"{file_path}.backup"
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            f.write(original_content)
                        
                        # Write transformed content
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(transformed_content)
                        
                        results.append(TransformationResult(
                            file_path=file_path,
                            original_content=original_content,
                            transformed_content=transformed_content,
                            applied_rules=applied_rules,
                            confidence_score=sum(self.refactoring_rules[rule].confidence for rule in applied_rules) / len(applied_rules),
                            risk_assessment="low",
                            validation_status="pending",
                            backup_created=True
                        ))
                
                except Exception as e:
                    self.logger.error(f"Failed to refactor file {file_path}: {str(e)}")
        
        return results
    
    async def _apply_refactoring_rule(self, content: str, rule: RefactoringRule) -> str:
        """Apply a specific refactoring rule"""
        if rule.rule_id == "extract_method":
            return await self._extract_long_methods(content)
        elif rule.rule_id == "rename_variable":
            return self._rename_variables(content)
        elif rule.rule_id == "modernize_string_formatting":
            return self._modernize_string_formatting(content)
        elif rule.rule_id == "replace_deprecated_imports":
            return self._replace_deprecated_imports(content)
        else:
            # Apply regex-based transformation
            return re.sub(rule.pattern, rule.replacement, content)
    
    async def _extract_long_methods(self, content: str) -> str:
        """Extract long methods into smaller methods"""
        try:
            tree = ast.parse(content)
            transformer = MethodExtractorTransformer()
            new_tree = transformer.visit(tree)
            
            # Convert back to code (simplified - in practice, use ast.unparse or similar)
            return content  # Placeholder - would need proper AST to code conversion
        
        except SyntaxError:
            return content
    
    def _rename_variables(self, content: str) -> str:
        """Rename variables to follow naming conventions"""
        # Convert camelCase to snake_case for Python
        def camel_to_snake(match):
            return match.group(1) + '_' + match.group(2).lower()
        
        # Only rename in variable assignments and function parameters
        lines = content.splitlines()
        transformed_lines = []
        
        for line in lines:
            # Simple variable assignment pattern
            if '=' in line and not line.strip().startswith('#'):
                line = re.sub(r'([a-z])([A-Z])', camel_to_snake, line)
            transformed_lines.append(line)
        
        return '\n'.join(transformed_lines)
    
    def _modernize_string_formatting(self, content: str) -> str:
        """Convert old string formatting to f-strings"""
        # Convert % formatting to f-strings
        def convert_percent_formatting(match):
            format_str = match.group(1)
            value = match.group(2)
            return f'f"{format_str}{{{value}}}"'
        
        # Simple conversion for basic cases
        content = re.sub(r'"([^"]*?)%s"\s*%\s*([^,\n]+)', convert_percent_formatting, content)
        content = re.sub(r'"([^"]*?)%d"\s*%\s*([^,\n]+)', convert_percent_formatting, content)
        
        return content
    
    def _replace_deprecated_imports(self, content: str) -> str:
        """Replace deprecated import statements"""
        replacements = {
            'from imp import': 'from importlib import',
            'import imp': 'import importlib',
            'from string import': '# string module methods are now str methods',
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        return content
    
    async def _apply_framework_migration(self, repo_path: str, target_framework: str, analysis_results: Dict[str, Any]) -> List[TransformationResult]:
        """Apply framework migration transformations"""
        results = []
        
        # Detect source framework
        source_framework = self._detect_primary_framework(analysis_results)
        
        if not source_framework:
            raise Exception("Could not detect source framework")
        
        migration_key = f"{source_framework.lower()}_to_{target_framework.lower()}"
        
        if migration_key not in self.framework_migrations:
            raise Exception(f"Migration from {source_framework} to {target_framework} not supported")
        
        migration = self.framework_migrations[migration_key]
        
        # Apply migration transformations
        for file_info in analysis_results["files"]:
            if source_framework.lower() in file_info["file_path"].lower() or self._contains_framework_code(file_info, source_framework):
                file_path = file_info["file_path"]
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    transformed_content = await self._migrate_framework_code(original_content, migration)
                    
                    if transformed_content != original_content:
                        # Create backup
                        backup_path = f"{file_path}.backup"
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            f.write(original_content)
                        
                        # Write transformed content
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(transformed_content)
                        
                        results.append(TransformationResult(
                            file_path=file_path,
                            original_content=original_content,
                            transformed_content=transformed_content,
                            applied_rules=[migration_key],
                            confidence_score=migration.success_probability,
                            risk_assessment=migration.risk_level,
                            validation_status="pending",
                            backup_created=True
                        ))
                
                except Exception as e:
                    self.logger.error(f"Failed to migrate file {file_path}: {str(e)}")
        
        return results
    
    def _detect_primary_framework(self, analysis_results: Dict[str, Any]) -> Optional[str]:
        """Detect the primary framework used in the codebase"""
        framework_counts = defaultdict(int)
        
        for framework in analysis_results["frameworks"]:
            framework_counts[framework] += 1
        
        if framework_counts:
            return max(framework_counts, key=framework_counts.get)
        
        return None
    
    def _contains_framework_code(self, file_info: Dict[str, Any], framework: str) -> bool:
        """Check if file contains framework-specific code"""
        # Simple heuristic - in practice, would need more sophisticated detection
        return framework.lower() in str(file_info).lower()
    
    async def _migrate_framework_code(self, content: str, migration: FrameworkMigration) -> str:
        """Migrate code from source to target framework"""
        transformed_content = content
        
        if migration.source_framework == "flask" and migration.target_framework == "fastapi":
            # Flask to FastAPI migration
            transformed_content = self._migrate_flask_to_fastapi(transformed_content)
        
        return transformed_content
    
    def _migrate_flask_to_fastapi(self, content: str) -> str:
        """Migrate Flask code to FastAPI"""
        # Basic Flask to FastAPI transformations
        transformations = [
            (r'from flask import Flask', 'from fastapi import FastAPI'),
            (r'app = Flask\(__name__\)', 'app = FastAPI()'),
            (r'@app\.route\([\'"]([^\'"]+)[\'"]\)', r'@app.get("\1")'),
            (r'@app\.route\([\'"]([^\'"]+)[\'"], methods=\[\'POST\'\]\)', r'@app.post("\1")'),
            (r'from flask import request', 'from fastapi import Request'),
            (r'from flask import jsonify', '# FastAPI automatically handles JSON responses'),
            (r'return jsonify\(([^)]+)\)', r'return \1'),
        ]
        
        for pattern, replacement in transformations:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    async def _apply_modernization(self, repo_path: str, analysis_results: Dict[str, Any]) -> List[TransformationResult]:
        """Apply modernization transformations"""
        results = []
        
        # Apply all applicable modernization rules
        modernization_rules = [
            "modernize_string_formatting",
            "replace_deprecated_imports",
            "add_type_hints"
        ]
        
        return await self._apply_refactoring(repo_path, modernization_rules, analysis_results)
    
    async def _generate_code(self, repo_path: str, options: Dict[str, Any], analysis_results: Dict[str, Any]) -> List[TransformationResult]:
        """Generate new code based on templates and options"""
        results = []
        
        generation_type = options.get("type", "microservice")
        
        if generation_type == "microservice":
            result = await self._generate_microservice(repo_path, options)
            results.append(result)
        elif generation_type == "crud":
            result = await self._generate_crud_api(repo_path, options)
            results.append(result)
        elif generation_type == "tests":
            test_results = await self._generate_tests(repo_path, analysis_results)
            results.extend(test_results)
        
        return results
    
    async def _generate_microservice(self, repo_path: str, options: Dict[str, Any]) -> TransformationResult:
        """Generate a new microservice"""
        service_name = options.get("service_name", "new_service")
        service_description = options.get("description", "Generated microservice")
        port = options.get("port", 8000)
        
        template = self.code_templates["microservice_template"]
        generated_content = template.format(
            service_name=service_name,
            service_description=service_description,
            module_name=service_name.lower(),
            port=port
        )
        
        # Create service file
        service_file = os.path.join(repo_path, f"{service_name.lower()}.py")
        with open(service_file, 'w', encoding='utf-8') as f:
            f.write(generated_content)
        
        return TransformationResult(
            file_path=service_file,
            original_content="",
            transformed_content=generated_content,
            applied_rules=["generate_microservice"],
            confidence_score=0.95,
            risk_assessment="low",
            validation_status="pending",
            backup_created=False
        )
    
    async def _generate_crud_api(self, repo_path: str, options: Dict[str, Any]) -> TransformationResult:
        """Generate CRUD API endpoints"""
        model_name = options.get("model_name", "Item")
        endpoint = options.get("endpoint", "items")
        fields = options.get("fields", ["name: str", "description: str"])
        
        template = self.code_templates["fastapi_crud"]
        generated_content = template.format(
            model_name=model_name,
            endpoint=endpoint,
            fields="\n    ".join(fields)
        )
        
        # Create CRUD file
        crud_file = os.path.join(repo_path, f"{endpoint}_crud.py")
        with open(crud_file, 'w', encoding='utf-8') as f:
            f.write(generated_content)
        
        return TransformationResult(
            file_path=crud_file,
            original_content="",
            transformed_content=generated_content,
            applied_rules=["generate_crud"],
            confidence_score=0.90,
            risk_assessment="low",
            validation_status="pending",
            backup_created=False
        )
    
    async def _generate_tests(self, repo_path: str, analysis_results: Dict[str, Any]) -> List[TransformationResult]:
        """Generate test files for existing code"""
        results = []
        
        for file_info in analysis_results["files"]:
            if file_info["language"] == "Python" and not file_info["file_path"].endswith("test.py"):
                test_result = await self._generate_test_file(file_info)
                if test_result:
                    results.append(test_result)
        
        return results
    
    async def _generate_test_file(self, file_info: Dict[str, Any]) -> Optional[TransformationResult]:
        """Generate test file for a specific source file"""
        source_file = file_info["file_path"]
        module_name = os.path.splitext(os.path.basename(source_file))[0]
        
        template = self.code_templates["test_template"]
        generated_content = template.format(
            module_name=module_name,
            function_name="example_function",
            method="get",
            endpoint="/test"
        )
        
        # Create test file
        test_file = source_file.replace(".py", "_test.py")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(generated_content)
        
        return TransformationResult(
            file_path=test_file,
            original_content="",
            transformed_content=generated_content,
            applied_rules=["generate_tests"],
            confidence_score=0.80,
            risk_assessment="low",
            validation_status="pending",
            backup_created=False
        )
    
    async def _validate_transformations(self, repo_path: str, transformation_results: List[TransformationResult]) -> Dict[str, Any]:
        """Validate transformation results"""
        validation_results = {
            "syntax_valid": 0,
            "syntax_errors": 0,
            "test_results": {},
            "performance_impact": {},
            "security_analysis": {}
        }
        
        for result in transformation_results:
            # Syntax validation
            if result.file_path.endswith('.py'):
                try:
                    with open(result.file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    ast.parse(content)
                    validation_results["syntax_valid"] += 1
                    result.validation_status = "valid"
                
                except SyntaxError as e:
                    validation_results["syntax_errors"] += 1
                    result.validation_status = f"syntax_error: {str(e)}"
        
        # Run tests if available
        test_command = "python -m pytest"
        try:
            process = await asyncio.create_subprocess_exec(
                *test_command.split(),
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            validation_results["test_results"] = {
                "exit_code": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode()
            }
        
        except Exception as e:
            validation_results["test_results"] = {"error": str(e)}
        
        return validation_results
    
    async def _generate_transformation_report(self, transformation_results: List[TransformationResult],
                                            validation_results: Dict[str, Any],
                                            analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive transformation report"""
        
        total_files_transformed = len(transformation_results)
        successful_transformations = len([r for r in transformation_results if r.validation_status == "valid"])
        
        # Calculate metrics
        average_confidence = sum(r.confidence_score for r in transformation_results) / max(total_files_transformed, 1)
        
        # Group by transformation type
        transformations_by_type = defaultdict(int)
        for result in transformation_results:
            for rule in result.applied_rules:
                transformations_by_type[rule] += 1
        
        return {
            "summary": {
                "total_files_analyzed": len(analysis_results["files"]),
                "total_files_transformed": total_files_transformed,
                "successful_transformations": successful_transformations,
                "failed_transformations": total_files_transformed - successful_transformations,
                "average_confidence_score": average_confidence,
                "syntax_validation": {
                    "valid": validation_results["syntax_valid"],
                    "errors": validation_results["syntax_errors"]
                }
            },
            "transformations_applied": dict(transformations_by_type),
            "validation_summary": validation_results,
            "recommendations": [
                "Review all transformed files before committing",
                "Run comprehensive tests to ensure functionality",
                "Update documentation to reflect changes",
                "Consider gradual rollout for critical systems"
            ],
            "next_steps": [
                "Commit transformed code to version control",
                "Deploy to staging environment for testing",
                "Update CI/CD pipelines if necessary",
                "Train team on new patterns and frameworks"
            ]
        }
    
    async def _update_progress(self, transformation_id: str, progress: float, status: str):
        """Update transformation progress"""
        await self.redis_client.hset(
            f"transformation:{transformation_id}",
            mapping={
                "progress": str(progress),
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
        )
    
    async def _store_transformation_results(self, transformation_id: str, results: Dict[str, Any]):
        """Store transformation results in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO transformation_results (transformation_id, results, created_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (transformation_id) DO UPDATE SET
                    results = $2,
                    updated_at = $3
            """, transformation_id, json.dumps(results), datetime.now())
        
        # Also store in Redis for quick access
        await self.redis_client.hset(
            f"transformation:{transformation_id}",
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
    
    async def get_transformation_status(self, transformation_id: str) -> Dict[str, Any]:
        """Get transformation status and results"""
        data = await self.redis_client.hgetall(f"transformation:{transformation_id}")
        
        if not data:
            raise HTTPException(status_code=404, detail="Transformation not found")
        
        result = {
            "transformation_id": transformation_id,
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

# AST Transformer Classes
class MethodExtractorTransformer(ast.NodeTransformer):
    """AST transformer for extracting long methods"""
    
    def visit_FunctionDef(self, node):
        # Check if method is too long
        if hasattr(node, 'end_lineno') and node.end_lineno:
            method_lines = node.end_lineno - node.lineno
            if method_lines > 30:
                # Extract method logic (simplified)
                # In practice, would need sophisticated analysis
                pass
        
        return node

# FastAPI Application
app = FastAPI(title="Agentic Transformation Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global transformation engine instance
transformation_engine = None

@app.on_event("startup")
async def startup_event():
    global transformation_engine
    
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
            CREATE TABLE IF NOT EXISTS transformation_results (
                transformation_id VARCHAR(64) PRIMARY KEY,
                results JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
    
    transformation_engine = AgenticTransformationEngine(db_pool, redis_client)

@app.post("/api/transformation/start", response_model=TransformationResponse)
async def start_transformation(request: TransformationRequest):
    """Start agentic transformation process"""
    transformation_id = await transformation_engine.start_transformation(request)
    
    return TransformationResponse(
        transformation_id=transformation_id,
        status="started",
        progress=0.0
    )

@app.get("/api/transformation/status/{transformation_id}", response_model=TransformationResponse)
async def get_transformation_status(transformation_id: str):
    """Get transformation status and results"""
    status_data = await transformation_engine.get_transformation_status(transformation_id)
    
    return TransformationResponse(
        transformation_id=transformation_id,
        status=status_data["status"],
        progress=status_data["progress"],
        results=status_data.get("results"),
        error_message=status_data.get("error_message")
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agentic-transformation-engine"}

if __name__ == "__main__":
    uvicorn.run(
        "agentic_transformation_engine:app",
        host="0.0.0.0",
        port=8041,
        reload=True,
        log_level="info"
    )

