"""
Intelligent Test Generator for Nexus Architect QA Automation
Implements comprehensive test case generation from code analysis and requirements
"""

import ast
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum

import aiohttp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import openai
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Test type enumeration"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    API = "api"
    UI = "ui"
    REGRESSION = "regression"

class TestPriority(Enum):
    """Test priority enumeration"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class TestCase:
    """Test case data structure"""
    id: str
    name: str
    description: str
    test_type: TestType
    priority: TestPriority
    function_name: str
    class_name: Optional[str]
    module_name: str
    test_code: str
    setup_code: Optional[str]
    teardown_code: Optional[str]
    expected_result: str
    test_data: Dict[str, Any]
    coverage_targets: List[str]
    dependencies: List[str]
    tags: List[str]
    estimated_execution_time: float
    confidence_score: float
    created_at: datetime
    updated_at: datetime

@dataclass
class CodeAnalysis:
    """Code analysis results"""
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[str]
    complexity_metrics: Dict[str, float]
    control_flow_paths: List[List[str]]
    data_flow_analysis: Dict[str, List[str]]
    security_patterns: List[Dict[str, Any]]
    performance_patterns: List[Dict[str, Any]]

class ASTAnalyzer:
    """Abstract Syntax Tree analyzer for code analysis"""
    
    def __init__(self):
        self.complexity_threshold = 10
        self.depth_threshold = 5
        
    def analyze_file(self, file_path: str) -> CodeAnalysis:
        """Analyze Python file and extract code structure"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            functions = self._extract_functions(tree)
            classes = self._extract_classes(tree)
            imports = self._extract_imports(tree)
            complexity_metrics = self._calculate_complexity(tree)
            control_flow_paths = self._analyze_control_flow(tree)
            data_flow_analysis = self._analyze_data_flow(tree)
            security_patterns = self._detect_security_patterns(tree)
            performance_patterns = self._detect_performance_patterns(tree)
            
            return CodeAnalysis(
                functions=functions,
                classes=classes,
                imports=imports,
                complexity_metrics=complexity_metrics,
                control_flow_paths=control_flow_paths,
                data_flow_analysis=data_flow_analysis,
                security_patterns=security_patterns,
                performance_patterns=performance_patterns
            )
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return CodeAnalysis([], [], [], {}, [], {}, [], [])
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function definitions and metadata"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_info = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'defaults': len(node.args.defaults),
                    'returns': ast.unparse(node.returns) if node.returns else None,
                    'docstring': ast.get_docstring(node),
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'line_number': node.lineno,
                    'complexity': self._calculate_cyclomatic_complexity(node),
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'has_exceptions': self._has_exception_handling(node),
                    'calls_external': self._calls_external_functions(node)
                }
                functions.append(function_info)
        
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class definitions and metadata"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'is_private': item.name.startswith('_'),
                            'is_property': any(
                                ast.unparse(dec) == 'property' 
                                for dec in item.decorator_list
                            ),
                            'complexity': self._calculate_cyclomatic_complexity(item)
                        })
                
                class_info = {
                    'name': node.name,
                    'bases': [ast.unparse(base) for base in node.bases],
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'methods': methods,
                    'line_number': node.lineno,
                    'docstring': ast.get_docstring(node),
                    'is_abstract': self._is_abstract_class(node),
                    'inheritance_depth': len(node.bases)
                }
                classes.append(class_info)
        
        return classes
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return imports
    
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate various complexity metrics"""
        metrics = {
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'halstead_difficulty': 0,
            'maintainability_index': 0,
            'lines_of_code': 0,
            'comment_ratio': 0
        }
        
        # Calculate cyclomatic complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, 
                               ast.ExceptHandler, ast.With, ast.Assert)):
                metrics['cyclomatic_complexity'] += 1
            elif isinstance(node, ast.BoolOp):
                metrics['cyclomatic_complexity'] += len(node.values) - 1
        
        # Calculate lines of code
        if hasattr(tree, 'body') and tree.body:
            last_node = tree.body[-1]
            if hasattr(last_node, 'lineno'):
                metrics['lines_of_code'] = last_node.lineno
        
        # Calculate maintainability index (simplified)
        loc = metrics['lines_of_code']
        cc = metrics['cyclomatic_complexity']
        if loc > 0:
            metrics['maintainability_index'] = max(0, 
                171 - 5.2 * np.log(loc) - 0.23 * cc - 16.2 * np.log(loc/10)
            )
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try,
                                ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _analyze_control_flow(self, tree: ast.AST) -> List[List[str]]:
        """Analyze control flow paths for test coverage"""
        paths = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_paths = self._extract_function_paths(node)
                paths.extend(function_paths)
        
        return paths
    
    def _extract_function_paths(self, func_node: ast.FunctionDef) -> List[List[str]]:
        """Extract execution paths from a function"""
        paths = []
        current_path = [func_node.name]
        
        def traverse_node(node, path):
            if isinstance(node, ast.If):
                # True branch
                true_path = path + ['if_true']
                for child in node.body:
                    traverse_node(child, true_path)
                
                # False branch
                if node.orelse:
                    false_path = path + ['if_false']
                    for child in node.orelse:
                        traverse_node(child, false_path)
                else:
                    paths.append(path + ['if_false'])
            
            elif isinstance(node, ast.For):
                loop_path = path + ['for_loop']
                for child in node.body:
                    traverse_node(child, loop_path)
                paths.append(loop_path)
            
            elif isinstance(node, ast.While):
                loop_path = path + ['while_loop']
                for child in node.body:
                    traverse_node(child, loop_path)
                paths.append(loop_path)
            
            elif isinstance(node, ast.Try):
                try_path = path + ['try_block']
                for child in node.body:
                    traverse_node(child, try_path)
                
                for handler in node.handlers:
                    except_path = path + [f'except_{handler.type.id if handler.type else "all"}']
                    for child in handler.body:
                        traverse_node(child, except_path)
                    paths.append(except_path)
            
            else:
                if hasattr(node, 'body'):
                    for child in node.body:
                        traverse_node(child, path)
                elif not paths or path != paths[-1]:
                    paths.append(path)
        
        for stmt in func_node.body:
            traverse_node(stmt, current_path)
        
        return paths if paths else [current_path]
    
    def _analyze_data_flow(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analyze data flow for variable dependencies"""
        data_flow = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                variables = self._extract_variable_usage(node)
                data_flow[node.name] = variables
        
        return data_flow
    
    def _extract_variable_usage(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract variable usage patterns in a function"""
        variables = set()
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name):
                variables.add(node.id)
            elif isinstance(node, ast.Attribute):
                variables.add(f"{ast.unparse(node.value)}.{node.attr}")
        
        return list(variables)
    
    def _detect_security_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect security-related patterns in code"""
        security_patterns = []
        
        for node in ast.walk(tree):
            # SQL injection patterns
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'attr') and node.func.attr in ['execute', 'query']:
                    if any(isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add) 
                          for arg in node.args):
                        security_patterns.append({
                            'type': 'sql_injection_risk',
                            'line': node.lineno,
                            'description': 'Potential SQL injection vulnerability'
                        })
            
            # Command injection patterns
            if isinstance(node, ast.Call):
                if (hasattr(node.func, 'id') and node.func.id in ['eval', 'exec'] or
                    hasattr(node.func, 'attr') and node.func.attr in ['system', 'popen']):
                    security_patterns.append({
                        'type': 'command_injection_risk',
                        'line': node.lineno,
                        'description': 'Potential command injection vulnerability'
                    })
            
            # Hardcoded secrets
            if isinstance(node, ast.Str):
                if re.search(r'(password|secret|key|token).*=.*["\'][^"\']{8,}["\']', 
                           ast.unparse(node), re.IGNORECASE):
                    security_patterns.append({
                        'type': 'hardcoded_secret',
                        'line': node.lineno,
                        'description': 'Potential hardcoded secret'
                    })
        
        return security_patterns
    
    def _detect_performance_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect performance-related patterns in code"""
        performance_patterns = []
        
        for node in ast.walk(tree):
            # Nested loops
            if isinstance(node, (ast.For, ast.While)):
                nested_loops = sum(1 for child in ast.walk(node) 
                                 if isinstance(child, (ast.For, ast.While)) and child != node)
                if nested_loops > 0:
                    performance_patterns.append({
                        'type': 'nested_loops',
                        'line': node.lineno,
                        'depth': nested_loops + 1,
                        'description': f'Nested loops detected (depth: {nested_loops + 1})'
                    })
            
            # Large list comprehensions
            if isinstance(node, ast.ListComp):
                generators = len(node.generators)
                if generators > 2:
                    performance_patterns.append({
                        'type': 'complex_comprehension',
                        'line': node.lineno,
                        'generators': generators,
                        'description': f'Complex list comprehension ({generators} generators)'
                    })
        
        return performance_patterns
    
    def _has_exception_handling(self, node: ast.FunctionDef) -> bool:
        """Check if function has exception handling"""
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
        return False
    
    def _calls_external_functions(self, node: ast.FunctionDef) -> bool:
        """Check if function calls external functions"""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if hasattr(child.func, 'id') and not child.func.id.startswith('_'):
                    return True
                elif hasattr(child.func, 'attr'):
                    return True
        return False
    
    def _is_abstract_class(self, node: ast.ClassDef) -> bool:
        """Check if class is abstract"""
        for decorator in node.decorator_list:
            if ast.unparse(decorator) == 'abstractmethod':
                return True
        
        for method in node.body:
            if isinstance(method, ast.FunctionDef):
                for decorator in method.decorator_list:
                    if ast.unparse(decorator) == 'abstractmethod':
                        return True
        
        return False

class RequirementAnalyzer:
    """Analyzer for requirements and user stories"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def analyze_requirements(self, requirements_text: str) -> Dict[str, Any]:
        """Analyze requirements document and extract test scenarios"""
        doc = self.nlp(requirements_text)
        
        # Extract entities and relationships
        entities = self._extract_entities(doc)
        actions = self._extract_actions(doc)
        conditions = self._extract_conditions(doc)
        acceptance_criteria = self._extract_acceptance_criteria(requirements_text)
        
        return {
            'entities': entities,
            'actions': actions,
            'conditions': conditions,
            'acceptance_criteria': acceptance_criteria,
            'test_scenarios': self._generate_test_scenarios(entities, actions, conditions)
        }
    
    def _extract_entities(self, doc) -> List[Dict[str, str]]:
        """Extract named entities from requirements"""
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            })
        return entities
    
    def _extract_actions(self, doc) -> List[str]:
        """Extract action verbs from requirements"""
        actions = []
        for token in doc:
            if token.pos_ == 'VERB' and not token.is_stop:
                actions.append(token.lemma_)
        return list(set(actions))
    
    def _extract_conditions(self, doc) -> List[str]:
        """Extract conditional statements from requirements"""
        conditions = []
        for sent in doc.sents:
            if any(word in sent.text.lower() for word in ['if', 'when', 'unless', 'provided']):
                conditions.append(sent.text.strip())
        return conditions
    
    def _extract_acceptance_criteria(self, text: str) -> List[str]:
        """Extract acceptance criteria from requirements"""
        criteria = []
        
        # Look for common acceptance criteria patterns
        patterns = [
            r'Given.*When.*Then.*',
            r'As a.*I want.*So that.*',
            r'The system should.*',
            r'The user must be able to.*'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            criteria.extend(matches)
        
        return criteria
    
    def _generate_test_scenarios(self, entities: List[Dict], actions: List[str], 
                                conditions: List[str]) -> List[Dict[str, str]]:
        """Generate test scenarios from extracted components"""
        scenarios = []
        
        for action in actions[:5]:  # Limit to top 5 actions
            for entity in entities[:3]:  # Limit to top 3 entities
                scenario = {
                    'name': f"Test {action} for {entity['text']}",
                    'description': f"Verify that {action} works correctly for {entity['text']}",
                    'type': 'functional',
                    'priority': 'medium'
                }
                scenarios.append(scenario)
        
        return scenarios

class AITestGenerator:
    """AI-powered test generation using language models"""
    
    def __init__(self):
        self.openai_client = None
        self.code_model = None
        self.test_templates = self._load_test_templates()
        
    def _load_test_templates(self) -> Dict[str, str]:
        """Load test code templates"""
        return {
            'unit_test': '''
def test_{function_name}():
    """Test {function_name} function"""
    # Arrange
    {setup_code}
    
    # Act
    result = {function_call}
    
    # Assert
    {assertions}
''',
            'integration_test': '''
def test_{function_name}_integration():
    """Integration test for {function_name}"""
    # Setup test environment
    {setup_code}
    
    try:
        # Execute integration scenario
        {test_scenario}
        
        # Verify results
        {assertions}
    finally:
        # Cleanup
        {cleanup_code}
''',
            'api_test': '''
def test_{endpoint_name}_api():
    """Test {endpoint_name} API endpoint"""
    # Prepare request data
    {request_data}
    
    # Make API call
    response = client.{method}('{endpoint}', {parameters})
    
    # Verify response
    assert response.status_code == {expected_status}
    {response_assertions}
'''
        }
    
    async def generate_test_from_function(self, function_info: Dict[str, Any], 
                                        test_type: TestType = TestType.UNIT) -> TestCase:
        """Generate test case for a specific function"""
        function_name = function_info['name']
        
        # Generate test data
        test_data = self._generate_test_data(function_info)
        
        # Generate test code
        test_code = self._generate_test_code(function_info, test_type, test_data)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(function_info, test_type)
        
        return TestCase(
            id=f"test_{function_name}_{int(time.time())}",
            name=f"test_{function_name}",
            description=f"Test case for {function_name} function",
            test_type=test_type,
            priority=self._determine_priority(function_info),
            function_name=function_name,
            class_name=function_info.get('class_name'),
            module_name=function_info.get('module_name', 'unknown'),
            test_code=test_code,
            setup_code=self._generate_setup_code(function_info),
            teardown_code=self._generate_teardown_code(function_info),
            expected_result=self._generate_expected_result(function_info, test_data),
            test_data=test_data,
            coverage_targets=[function_name],
            dependencies=function_info.get('dependencies', []),
            tags=self._generate_tags(function_info, test_type),
            estimated_execution_time=self._estimate_execution_time(function_info, test_type),
            confidence_score=confidence_score,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _generate_test_data(self, function_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test data for function parameters"""
        test_data = {}
        
        args = function_info.get('args', [])
        for arg in args:
            # Generate different types of test data
            test_data[arg] = {
                'valid_values': self._generate_valid_values(arg),
                'invalid_values': self._generate_invalid_values(arg),
                'edge_cases': self._generate_edge_cases(arg)
            }
        
        return test_data
    
    def _generate_valid_values(self, arg_name: str) -> List[Any]:
        """Generate valid test values for an argument"""
        # Simple heuristic-based generation
        if 'id' in arg_name.lower():
            return [1, 42, 999]
        elif 'name' in arg_name.lower():
            return ['test_name', 'example', 'valid_name']
        elif 'email' in arg_name.lower():
            return ['test@example.com', 'user@domain.org']
        elif 'count' in arg_name.lower() or 'num' in arg_name.lower():
            return [0, 1, 10, 100]
        else:
            return ['test_value', 42, True, [1, 2, 3]]
    
    def _generate_invalid_values(self, arg_name: str) -> List[Any]:
        """Generate invalid test values for an argument"""
        return [None, '', -1, 'invalid', [], {}]
    
    def _generate_edge_cases(self, arg_name: str) -> List[Any]:
        """Generate edge case values for an argument"""
        return [0, -1, float('inf'), '', ' ', '\n', '\t']
    
    def _generate_test_code(self, function_info: Dict[str, Any], 
                          test_type: TestType, test_data: Dict[str, Any]) -> str:
        """Generate test code based on function info and test type"""
        function_name = function_info['name']
        args = function_info.get('args', [])
        
        if test_type == TestType.UNIT:
            template = self.test_templates['unit_test']
            
            # Generate function call
            if args:
                arg_values = []
                for arg in args:
                    valid_values = test_data.get(arg, {}).get('valid_values', ['test_value'])
                    arg_values.append(repr(valid_values[0]))
                function_call = f"{function_name}({', '.join(arg_values)})"
            else:
                function_call = f"{function_name}()"
            
            # Generate assertions
            assertions = [
                "assert result is not None",
                "assert isinstance(result, (str, int, float, bool, list, dict))"
            ]
            
            return template.format(
                function_name=function_name,
                setup_code="# Setup test data",
                function_call=function_call,
                assertions='\n    '.join(assertions)
            )
        
        elif test_type == TestType.INTEGRATION:
            template = self.test_templates['integration_test']
            return template.format(
                function_name=function_name,
                setup_code="# Setup integration environment",
                test_scenario=f"result = {function_name}()",
                assertions="assert result is not None",
                cleanup_code="# Cleanup resources"
            )
        
        return f"def test_{function_name}():\n    pass  # TODO: Implement test"
    
    def _generate_setup_code(self, function_info: Dict[str, Any]) -> Optional[str]:
        """Generate setup code for test"""
        if function_info.get('calls_external'):
            return "# Mock external dependencies\n# Setup test environment"
        return None
    
    def _generate_teardown_code(self, function_info: Dict[str, Any]) -> Optional[str]:
        """Generate teardown code for test"""
        if function_info.get('calls_external'):
            return "# Cleanup resources\n# Reset mocks"
        return None
    
    def _generate_expected_result(self, function_info: Dict[str, Any], 
                                test_data: Dict[str, Any]) -> str:
        """Generate expected result description"""
        returns = function_info.get('returns')
        if returns:
            return f"Expected return type: {returns}"
        return "Function should execute without errors"
    
    def _determine_priority(self, function_info: Dict[str, Any]) -> TestPriority:
        """Determine test priority based on function characteristics"""
        complexity = function_info.get('complexity', 1)
        has_exceptions = function_info.get('has_exceptions', False)
        calls_external = function_info.get('calls_external', False)
        
        if complexity > 10 or calls_external:
            return TestPriority.HIGH
        elif complexity > 5 or has_exceptions:
            return TestPriority.MEDIUM
        else:
            return TestPriority.LOW
    
    def _generate_tags(self, function_info: Dict[str, Any], test_type: TestType) -> List[str]:
        """Generate tags for test case"""
        tags = [test_type.value]
        
        if function_info.get('is_async'):
            tags.append('async')
        if function_info.get('has_exceptions'):
            tags.append('exception_handling')
        if function_info.get('calls_external'):
            tags.append('external_dependencies')
        if function_info.get('complexity', 1) > 10:
            tags.append('high_complexity')
        
        return tags
    
    def _estimate_execution_time(self, function_info: Dict[str, Any], 
                               test_type: TestType) -> float:
        """Estimate test execution time in seconds"""
        base_time = {
            TestType.UNIT: 0.1,
            TestType.INTEGRATION: 1.0,
            TestType.FUNCTIONAL: 2.0,
            TestType.PERFORMANCE: 10.0,
            TestType.SECURITY: 5.0
        }.get(test_type, 1.0)
        
        complexity_multiplier = 1 + (function_info.get('complexity', 1) / 10)
        external_multiplier = 2.0 if function_info.get('calls_external') else 1.0
        
        return base_time * complexity_multiplier * external_multiplier
    
    def _calculate_confidence_score(self, function_info: Dict[str, Any], 
                                  test_type: TestType) -> float:
        """Calculate confidence score for generated test"""
        score = 0.5  # Base score
        
        # Increase score based on available information
        if function_info.get('docstring'):
            score += 0.2
        if function_info.get('args'):
            score += 0.1
        if function_info.get('returns'):
            score += 0.1
        if function_info.get('complexity', 1) <= 5:
            score += 0.1
        
        return min(score, 1.0)

class IntelligentTestGenerator:
    """Main test generator orchestrating all analysis components"""
    
    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()
        self.requirement_analyzer = RequirementAnalyzer()
        self.ai_generator = AITestGenerator()
        self.generated_tests = []
        
    async def generate_tests_from_code(self, file_path: str, 
                                     test_types: List[TestType] = None) -> List[TestCase]:
        """Generate tests from code file analysis"""
        if test_types is None:
            test_types = [TestType.UNIT, TestType.INTEGRATION]
        
        # Analyze code structure
        code_analysis = self.ast_analyzer.analyze_file(file_path)
        
        generated_tests = []
        
        # Generate tests for each function
        for function_info in code_analysis.functions:
            function_info['module_name'] = Path(file_path).stem
            
            for test_type in test_types:
                try:
                    test_case = await self.ai_generator.generate_test_from_function(
                        function_info, test_type
                    )
                    generated_tests.append(test_case)
                except Exception as e:
                    logger.error(f"Error generating test for {function_info['name']}: {str(e)}")
        
        # Generate tests for classes
        for class_info in code_analysis.classes:
            for method in class_info['methods']:
                method_info = {
                    **method,
                    'class_name': class_info['name'],
                    'module_name': Path(file_path).stem
                }
                
                try:
                    test_case = await self.ai_generator.generate_test_from_function(
                        method_info, TestType.UNIT
                    )
                    generated_tests.append(test_case)
                except Exception as e:
                    logger.error(f"Error generating test for {method['name']}: {str(e)}")
        
        self.generated_tests.extend(generated_tests)
        return generated_tests
    
    async def generate_tests_from_requirements(self, requirements_text: str) -> List[TestCase]:
        """Generate tests from requirements analysis"""
        # Analyze requirements
        req_analysis = self.requirement_analyzer.analyze_requirements(requirements_text)
        
        generated_tests = []
        
        # Generate tests from scenarios
        for scenario in req_analysis['test_scenarios']:
            test_case = TestCase(
                id=f"req_test_{int(time.time())}_{len(generated_tests)}",
                name=scenario['name'].lower().replace(' ', '_'),
                description=scenario['description'],
                test_type=TestType.FUNCTIONAL,
                priority=TestPriority[scenario['priority'].upper()],
                function_name='requirement_test',
                class_name=None,
                module_name='requirements',
                test_code=self._generate_requirement_test_code(scenario),
                setup_code=None,
                teardown_code=None,
                expected_result=scenario['description'],
                test_data={'scenario': scenario},
                coverage_targets=[],
                dependencies=[],
                tags=['requirement', 'functional'],
                estimated_execution_time=2.0,
                confidence_score=0.7,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            generated_tests.append(test_case)
        
        self.generated_tests.extend(generated_tests)
        return generated_tests
    
    def _generate_requirement_test_code(self, scenario: Dict[str, str]) -> str:
        """Generate test code from requirement scenario"""
        return f'''
def test_{scenario['name'].lower().replace(' ', '_')}():
    """
    {scenario['description']}
    """
    # TODO: Implement requirement test
    # Scenario: {scenario['name']}
    # Description: {scenario['description']}
    pass
'''
    
    async def generate_performance_tests(self, code_analysis: CodeAnalysis) -> List[TestCase]:
        """Generate performance tests based on code analysis"""
        performance_tests = []
        
        for pattern in code_analysis.performance_patterns:
            test_case = TestCase(
                id=f"perf_test_{int(time.time())}_{len(performance_tests)}",
                name=f"test_performance_{pattern['type']}",
                description=f"Performance test for {pattern['description']}",
                test_type=TestType.PERFORMANCE,
                priority=TestPriority.MEDIUM,
                function_name='performance_test',
                class_name=None,
                module_name='performance',
                test_code=self._generate_performance_test_code(pattern),
                setup_code="import time\nimport psutil",
                teardown_code=None,
                expected_result="Performance within acceptable limits",
                test_data={'pattern': pattern},
                coverage_targets=[],
                dependencies=['psutil'],
                tags=['performance', pattern['type']],
                estimated_execution_time=10.0,
                confidence_score=0.6,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            performance_tests.append(test_case)
        
        return performance_tests
    
    def _generate_performance_test_code(self, pattern: Dict[str, Any]) -> str:
        """Generate performance test code"""
        return f'''
def test_performance_{pattern['type']}():
    """
    Performance test for {pattern['description']}
    """
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # TODO: Execute performance-critical code
    # Pattern: {pattern['type']}
    # Line: {pattern['line']}
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    execution_time = end_time - start_time
    memory_usage = end_memory - start_memory
    
    # Assert performance criteria
    assert execution_time < 1.0, f"Execution time too high: {{execution_time}}s"
    assert memory_usage < 100 * 1024 * 1024, f"Memory usage too high: {{memory_usage}} bytes"
'''
    
    async def generate_security_tests(self, code_analysis: CodeAnalysis) -> List[TestCase]:
        """Generate security tests based on code analysis"""
        security_tests = []
        
        for pattern in code_analysis.security_patterns:
            test_case = TestCase(
                id=f"sec_test_{int(time.time())}_{len(security_tests)}",
                name=f"test_security_{pattern['type']}",
                description=f"Security test for {pattern['description']}",
                test_type=TestType.SECURITY,
                priority=TestPriority.HIGH,
                function_name='security_test',
                class_name=None,
                module_name='security',
                test_code=self._generate_security_test_code(pattern),
                setup_code="import pytest",
                teardown_code=None,
                expected_result="No security vulnerabilities detected",
                test_data={'pattern': pattern},
                coverage_targets=[],
                dependencies=['pytest'],
                tags=['security', pattern['type']],
                estimated_execution_time=5.0,
                confidence_score=0.8,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            security_tests.append(test_case)
        
        return security_tests
    
    def _generate_security_test_code(self, pattern: Dict[str, Any]) -> str:
        """Generate security test code"""
        return f'''
def test_security_{pattern['type']}():
    """
    Security test for {pattern['description']}
    """
    # TODO: Implement security test
    # Pattern: {pattern['type']}
    # Line: {pattern['line']}
    # Description: {pattern['description']}
    
    # Test for security vulnerability
    with pytest.raises(SecurityError):
        # Execute potentially vulnerable code with malicious input
        pass
'''
    
    def get_test_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated tests"""
        if not self.generated_tests:
            return {}
        
        total_tests = len(self.generated_tests)
        test_types = {}
        priorities = {}
        avg_confidence = 0
        total_execution_time = 0
        
        for test in self.generated_tests:
            # Count test types
            test_type = test.test_type.value
            test_types[test_type] = test_types.get(test_type, 0) + 1
            
            # Count priorities
            priority = test.priority.value
            priorities[priority] = priorities.get(priority, 0) + 1
            
            # Sum confidence and execution time
            avg_confidence += test.confidence_score
            total_execution_time += test.estimated_execution_time
        
        avg_confidence /= total_tests
        
        return {
            'total_tests': total_tests,
            'test_types': test_types,
            'priorities': priorities,
            'average_confidence': avg_confidence,
            'total_execution_time': total_execution_time,
            'estimated_completion_time': f"{total_execution_time / 60:.1f} minutes"
        }
    
    def export_tests(self, format_type: str = 'json') -> str:
        """Export generated tests in specified format"""
        if format_type == 'json':
            return json.dumps([asdict(test) for test in self.generated_tests], 
                            default=str, indent=2)
        elif format_type == 'pytest':
            return self._export_as_pytest()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_as_pytest(self) -> str:
        """Export tests as pytest-compatible code"""
        pytest_code = "import pytest\nimport time\nimport psutil\n\n"
        
        for test in self.generated_tests:
            pytest_code += f"# {test.description}\n"
            pytest_code += f"# Priority: {test.priority.value}\n"
            pytest_code += f"# Estimated execution time: {test.estimated_execution_time}s\n"
            pytest_code += f"# Confidence score: {test.confidence_score}\n"
            
            if test.setup_code:
                pytest_code += f"\n{test.setup_code}\n"
            
            pytest_code += f"\n{test.test_code}\n\n"
        
        return pytest_code

# Example usage and testing
async def main():
    """Example usage of the intelligent test generator"""
    generator = IntelligentTestGenerator()
    
    # Example: Generate tests from a Python file
    # tests = await generator.generate_tests_from_code('example.py')
    
    # Example: Generate tests from requirements
    requirements = """
    As a user, I want to be able to login to the system
    so that I can access my personal dashboard.
    
    The system should validate user credentials
    and redirect to the dashboard upon successful login.
    """
    
    req_tests = await generator.generate_tests_from_requirements(requirements)
    
    # Get statistics
    stats = generator.get_test_statistics()
    print("Test Generation Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Export tests
    pytest_code = generator.export_tests('pytest')
    print("\nGenerated pytest code:")
    print(pytest_code[:500] + "..." if len(pytest_code) > 500 else pytest_code)

if __name__ == "__main__":
    asyncio.run(main())

