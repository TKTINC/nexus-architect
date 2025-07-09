"""
Dependency Analyzer for Nexus Architect
Analyzes code dependencies and builds comprehensive dependency graphs
"""

import os
import re
import json
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import subprocess
import tempfile
from pathlib import Path
import networkx as nx
import aiofiles
import aiohttp
from prometheus_client import Counter, Histogram, Gauge

# Metrics
DEPENDENCY_ANALYSIS_REQUESTS = Counter('dependency_analysis_requests_total', 'Total dependency analysis requests', ['language', 'status'])
DEPENDENCY_ANALYSIS_LATENCY = Histogram('dependency_analysis_latency_seconds', 'Dependency analysis latency', ['language'])
DEPENDENCIES_FOUND = Counter('dependencies_found_total', 'Total dependencies found', ['language', 'source_type'])
VULNERABILITY_SCANS = Counter('vulnerability_scans_total', 'Total vulnerability scans', ['language', 'status'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependencyType(Enum):
    """Types of dependencies"""
    DIRECT = "direct"
    TRANSITIVE = "transitive"
    DEV = "dev"
    PEER = "peer"
    OPTIONAL = "optional"

class DependencySource(Enum):
    """Source of dependency"""
    PACKAGE_MANAGER = "package_manager"
    IMPORT_STATEMENT = "import_statement"
    CONFIGURATION = "configuration"
    DOCKERFILE = "dockerfile"
    REQUIREMENTS = "requirements"

class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class Dependency:
    """Dependency information"""
    name: str
    version: Optional[str] = None
    version_constraint: Optional[str] = None
    dependency_type: DependencyType = DependencyType.DIRECT
    source: DependencySource = DependencySource.PACKAGE_MANAGER
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    description: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    author: Optional[str] = None
    size: Optional[int] = None
    last_updated: Optional[datetime] = None
    is_deprecated: bool = False
    alternatives: List[str] = field(default_factory=list)

@dataclass
class Vulnerability:
    """Security vulnerability in dependency"""
    id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    cvss_score: Optional[float] = None
    cve_id: Optional[str] = None
    cwe_id: Optional[str] = None
    affected_versions: List[str] = field(default_factory=list)
    patched_versions: List[str] = field(default_factory=list)
    published_date: Optional[datetime] = None
    references: List[str] = field(default_factory=list)
    exploit_available: bool = False

@dataclass
class DependencyGraph:
    """Dependency graph for a project"""
    project_name: str
    language: str
    dependencies: List[Dependency] = field(default_factory=list)
    graph: Dict[str, List[str]] = field(default_factory=dict)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    outdated_dependencies: List[Dependency] = field(default_factory=list)
    license_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

class PythonDependencyAnalyzer:
    """Python dependency analyzer"""
    
    def __init__(self):
        self.package_cache = {}
    
    async def analyze_dependencies(self, project_path: str) -> DependencyGraph:
        """Analyze Python dependencies"""
        start_time = time.time()
        
        try:
            graph = DependencyGraph(
                project_name=os.path.basename(project_path),
                language="python"
            )
            
            # Find dependency files
            dependency_files = await self._find_dependency_files(project_path)
            
            # Parse each dependency file
            for file_path in dependency_files:
                file_deps = await self._parse_dependency_file(file_path)
                graph.dependencies.extend(file_deps)
            
            # Parse import statements
            import_deps = await self._parse_import_statements(project_path)
            graph.dependencies.extend(import_deps)
            
            # Build dependency graph
            graph.graph = await self._build_dependency_graph(graph.dependencies)
            
            # Detect circular dependencies
            graph.circular_dependencies = self._detect_circular_dependencies(graph.graph)
            
            # Enrich with package information
            await self._enrich_package_info(graph.dependencies)
            
            # Check for vulnerabilities
            graph.vulnerabilities = await self._check_vulnerabilities(graph.dependencies)
            
            # Find outdated dependencies
            graph.outdated_dependencies = await self._find_outdated_dependencies(graph.dependencies)
            
            # Check license conflicts
            graph.license_conflicts = self._check_license_conflicts(graph.dependencies)
            
            DEPENDENCY_ANALYSIS_REQUESTS.labels(language='python', status='success').inc()
            DEPENDENCIES_FOUND.labels(language='python', source_type='total').inc(len(graph.dependencies))
            
            return graph
            
        except Exception as e:
            logger.error(f"Python dependency analysis failed: {e}")
            DEPENDENCY_ANALYSIS_REQUESTS.labels(language='python', status='error').inc()
            raise
        
        finally:
            latency = time.time() - start_time
            DEPENDENCY_ANALYSIS_LATENCY.labels(language='python').observe(latency)
    
    async def _find_dependency_files(self, project_path: str) -> List[str]:
        """Find Python dependency files"""
        dependency_files = []
        
        # Common Python dependency files
        patterns = [
            'requirements*.txt',
            'Pipfile',
            'pyproject.toml',
            'setup.py',
            'setup.cfg',
            'environment.yml',
            'conda.yml'
        ]
        
        for root, dirs, files in os.walk(project_path):
            # Skip virtual environments and cache directories
            dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '__pycache__', '.git'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                for pattern in patterns:
                    if file.startswith(pattern.replace('*', '')):
                        dependency_files.append(file_path)
                        break
        
        return dependency_files
    
    async def _parse_dependency_file(self, file_path: str) -> List[Dependency]:
        """Parse dependency file"""
        dependencies = []
        file_name = os.path.basename(file_path)
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if file_name.startswith('requirements'):
                dependencies = self._parse_requirements_txt(content, file_path)
            elif file_name == 'Pipfile':
                dependencies = self._parse_pipfile(content, file_path)
            elif file_name == 'pyproject.toml':
                dependencies = self._parse_pyproject_toml(content, file_path)
            elif file_name == 'setup.py':
                dependencies = self._parse_setup_py(content, file_path)
            elif file_name in ['environment.yml', 'conda.yml']:
                dependencies = self._parse_conda_yml(content, file_path)
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
        
        return dependencies
    
    def _parse_requirements_txt(self, content: str, file_path: str) -> List[Dependency]:
        """Parse requirements.txt file"""
        dependencies = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Skip -r includes and other options
            if line.startswith('-'):
                continue
            
            # Parse dependency
            dep = self._parse_requirement_line(line, file_path, i)
            if dep:
                dependencies.append(dep)
        
        return dependencies
    
    def _parse_requirement_line(self, line: str, file_path: str, line_number: int) -> Optional[Dependency]:
        """Parse single requirement line"""
        # Remove inline comments
        line = line.split('#')[0].strip()
        
        # Parse package name and version constraint
        match = re.match(r'^([a-zA-Z0-9_-]+)([><=!~\s].*)?$', line)
        if not match:
            return None
        
        name = match.group(1)
        version_constraint = match.group(2).strip() if match.group(2) else None
        
        # Determine dependency type
        dep_type = DependencyType.DEV if 'dev' in file_path.lower() else DependencyType.DIRECT
        
        return Dependency(
            name=name,
            version_constraint=version_constraint,
            dependency_type=dep_type,
            source=DependencySource.REQUIREMENTS,
            file_path=file_path,
            line_number=line_number
        )
    
    def _parse_pipfile(self, content: str, file_path: str) -> List[Dependency]:
        """Parse Pipfile"""
        dependencies = []
        
        try:
            import toml
            data = toml.loads(content)
            
            # Parse packages
            packages = data.get('packages', {})
            for name, version in packages.items():
                dep = Dependency(
                    name=name,
                    version_constraint=version if isinstance(version, str) else None,
                    dependency_type=DependencyType.DIRECT,
                    source=DependencySource.PACKAGE_MANAGER,
                    file_path=file_path
                )
                dependencies.append(dep)
            
            # Parse dev-packages
            dev_packages = data.get('dev-packages', {})
            for name, version in dev_packages.items():
                dep = Dependency(
                    name=name,
                    version_constraint=version if isinstance(version, str) else None,
                    dependency_type=DependencyType.DEV,
                    source=DependencySource.PACKAGE_MANAGER,
                    file_path=file_path
                )
                dependencies.append(dep)
                
        except Exception as e:
            logger.error(f"Failed to parse Pipfile: {e}")
        
        return dependencies
    
    def _parse_pyproject_toml(self, content: str, file_path: str) -> List[Dependency]:
        """Parse pyproject.toml file"""
        dependencies = []
        
        try:
            import toml
            data = toml.loads(content)
            
            # Parse dependencies from different sections
            sections = [
                ('project.dependencies', DependencyType.DIRECT),
                ('project.optional-dependencies', DependencyType.OPTIONAL),
                ('build-system.requires', DependencyType.DIRECT),
                ('tool.poetry.dependencies', DependencyType.DIRECT),
                ('tool.poetry.dev-dependencies', DependencyType.DEV)
            ]
            
            for section_path, dep_type in sections:
                section_data = data
                for key in section_path.split('.'):
                    section_data = section_data.get(key, {})
                
                if isinstance(section_data, list):
                    # List format (PEP 621)
                    for dep_str in section_data:
                        dep = self._parse_requirement_line(dep_str, file_path, 0)
                        if dep:
                            dep.dependency_type = dep_type
                            dependencies.append(dep)
                
                elif isinstance(section_data, dict):
                    # Dict format (Poetry)
                    for name, version in section_data.items():
                        if name == 'python':  # Skip Python version constraint
                            continue
                        
                        dep = Dependency(
                            name=name,
                            version_constraint=version if isinstance(version, str) else None,
                            dependency_type=dep_type,
                            source=DependencySource.PACKAGE_MANAGER,
                            file_path=file_path
                        )
                        dependencies.append(dep)
                        
        except Exception as e:
            logger.error(f"Failed to parse pyproject.toml: {e}")
        
        return dependencies
    
    def _parse_setup_py(self, content: str, file_path: str) -> List[Dependency]:
        """Parse setup.py file"""
        dependencies = []
        
        # Extract install_requires and extras_require using regex
        install_requires_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if install_requires_match:
            deps_str = install_requires_match.group(1)
            deps = re.findall(r'["\']([^"\']+)["\']', deps_str)
            
            for dep_str in deps:
                dep = self._parse_requirement_line(dep_str, file_path, 0)
                if dep:
                    dependencies.append(dep)
        
        return dependencies
    
    def _parse_conda_yml(self, content: str, file_path: str) -> List[Dependency]:
        """Parse conda environment.yml file"""
        dependencies = []
        
        try:
            import yaml
            data = yaml.safe_load(content)
            
            deps = data.get('dependencies', [])
            for dep in deps:
                if isinstance(dep, str):
                    # Parse conda package specification
                    parts = dep.split('=')
                    name = parts[0]
                    version = parts[1] if len(parts) > 1 else None
                    
                    dependency = Dependency(
                        name=name,
                        version=version,
                        dependency_type=DependencyType.DIRECT,
                        source=DependencySource.PACKAGE_MANAGER,
                        file_path=file_path
                    )
                    dependencies.append(dependency)
                
                elif isinstance(dep, dict) and 'pip' in dep:
                    # Parse pip dependencies in conda file
                    pip_deps = dep['pip']
                    for pip_dep in pip_deps:
                        dependency = self._parse_requirement_line(pip_dep, file_path, 0)
                        if dependency:
                            dependencies.append(dependency)
                            
        except Exception as e:
            logger.error(f"Failed to parse conda yml: {e}")
        
        return dependencies
    
    async def _parse_import_statements(self, project_path: str) -> List[Dependency]:
        """Parse import statements from Python files"""
        dependencies = []
        
        for root, dirs, files in os.walk(project_path):
            # Skip virtual environments and cache directories
            dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '__pycache__', '.git'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_deps = await self._extract_imports_from_file(file_path)
                    dependencies.extend(file_deps)
        
        # Remove duplicates and filter standard library modules
        unique_deps = {}
        for dep in dependencies:
            if dep.name not in unique_deps and not self._is_standard_library(dep.name):
                unique_deps[dep.name] = dep
        
        return list(unique_deps.values())
    
    async def _extract_imports_from_file(self, file_path: str) -> List[Dependency]:
        """Extract imports from Python file"""
        dependencies = []
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Parse import statements
                if line.startswith('import '):
                    modules = line[7:].split(',')
                    for module in modules:
                        module = module.strip().split('.')[0]  # Get root module
                        if module:
                            dep = Dependency(
                                name=module,
                                source=DependencySource.IMPORT_STATEMENT,
                                file_path=file_path,
                                line_number=i
                            )
                            dependencies.append(dep)
                
                elif line.startswith('from '):
                    match = re.match(r'from\s+([^\s]+)', line)
                    if match:
                        module = match.group(1).split('.')[0]  # Get root module
                        if module:
                            dep = Dependency(
                                name=module,
                                source=DependencySource.IMPORT_STATEMENT,
                                file_path=file_path,
                                line_number=i
                            )
                            dependencies.append(dep)
                            
        except Exception as e:
            logger.error(f"Failed to extract imports from {file_path}: {e}")
        
        return dependencies
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Check if module is part of Python standard library"""
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'time', 'math', 'random', 're',
            'collections', 'itertools', 'functools', 'typing', 'pathlib',
            'urllib', 'http', 'email', 'html', 'xml', 'sqlite3', 'csv',
            'logging', 'unittest', 'asyncio', 'threading', 'multiprocessing',
            'subprocess', 'tempfile', 'shutil', 'glob', 'pickle', 'base64',
            'hashlib', 'hmac', 'secrets', 'uuid', 'copy', 'pprint', 'textwrap',
            'string', 'io', 'struct', 'codecs', 'locale', 'calendar', 'heapq',
            'bisect', 'array', 'weakref', 'types', 'gc', 'inspect', 'dis'
        }
        
        return module_name in stdlib_modules
    
    async def _build_dependency_graph(self, dependencies: List[Dependency]) -> Dict[str, List[str]]:
        """Build dependency graph"""
        graph = {}
        
        # For now, build a simple graph based on package relationships
        # In a more sophisticated implementation, this would analyze actual dependencies
        for dep in dependencies:
            if dep.name not in graph:
                graph[dep.name] = []
        
        return graph
    
    def _detect_circular_dependencies(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies"""
        circular_deps = []
        
        try:
            # Use NetworkX to detect cycles
            G = nx.DiGraph(graph)
            cycles = list(nx.simple_cycles(G))
            circular_deps = cycles
        except Exception as e:
            logger.error(f"Failed to detect circular dependencies: {e}")
        
        return circular_deps
    
    async def _enrich_package_info(self, dependencies: List[Dependency]):
        """Enrich dependencies with package information from PyPI"""
        for dep in dependencies:
            if dep.source == DependencySource.IMPORT_STATEMENT:
                continue  # Skip import-only dependencies
            
            try:
                package_info = await self._get_pypi_info(dep.name)
                if package_info:
                    dep.description = package_info.get('summary')
                    dep.homepage = package_info.get('home_page')
                    dep.author = package_info.get('author')
                    dep.license = package_info.get('license')
                    
                    # Get latest version
                    releases = package_info.get('releases', {})
                    if releases:
                        latest_version = max(releases.keys(), key=lambda v: self._version_key(v))
                        if not dep.version:
                            dep.version = latest_version
                        
                        # Check if current version is outdated
                        if dep.version and dep.version != latest_version:
                            dep.is_deprecated = True
                            
            except Exception as e:
                logger.error(f"Failed to enrich package info for {dep.name}: {e}")
    
    async def _get_pypi_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Get package information from PyPI"""
        if package_name in self.package_cache:
            return self.package_cache[package_name]
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://pypi.org/pypi/{package_name}/json"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        package_info = data.get('info', {})
                        package_info['releases'] = data.get('releases', {})
                        
                        self.package_cache[package_name] = package_info
                        return package_info
                        
        except Exception as e:
            logger.error(f"Failed to get PyPI info for {package_name}: {e}")
        
        return None
    
    def _version_key(self, version: str) -> Tuple:
        """Convert version string to tuple for comparison"""
        try:
            return tuple(map(int, version.split('.')))
        except:
            return (0, 0, 0)
    
    async def _check_vulnerabilities(self, dependencies: List[Dependency]) -> List[Vulnerability]:
        """Check for vulnerabilities in dependencies"""
        vulnerabilities = []
        
        try:
            # Use safety database or similar service
            # For now, implement a basic check
            for dep in dependencies:
                if dep.name and dep.version:
                    vulns = await self._check_package_vulnerabilities(dep.name, dep.version)
                    vulnerabilities.extend(vulns)
                    
            VULNERABILITY_SCANS.labels(language='python', status='success').inc()
            
        except Exception as e:
            logger.error(f"Vulnerability check failed: {e}")
            VULNERABILITY_SCANS.labels(language='python', status='error').inc()
        
        return vulnerabilities
    
    async def _check_package_vulnerabilities(self, package_name: str, version: str) -> List[Vulnerability]:
        """Check vulnerabilities for specific package version"""
        vulnerabilities = []
        
        try:
            # This would integrate with a vulnerability database like OSV, Safety DB, etc.
            # For demonstration, return empty list
            pass
            
        except Exception as e:
            logger.error(f"Failed to check vulnerabilities for {package_name}: {e}")
        
        return vulnerabilities
    
    async def _find_outdated_dependencies(self, dependencies: List[Dependency]) -> List[Dependency]:
        """Find outdated dependencies"""
        outdated = []
        
        for dep in dependencies:
            if dep.is_deprecated or (dep.version and dep.version_constraint and 
                                   self._is_version_outdated(dep.version, dep.version_constraint)):
                outdated.append(dep)
        
        return outdated
    
    def _is_version_outdated(self, current_version: str, constraint: str) -> bool:
        """Check if version is outdated based on constraint"""
        # Simplified version comparison
        # In practice, use packaging.version or similar
        try:
            if constraint.startswith('>='):
                min_version = constraint[2:].strip()
                return self._version_key(current_version) < self._version_key(min_version)
            elif constraint.startswith('>'):
                min_version = constraint[1:].strip()
                return self._version_key(current_version) <= self._version_key(min_version)
        except:
            pass
        
        return False
    
    def _check_license_conflicts(self, dependencies: List[Dependency]) -> List[Dict[str, Any]]:
        """Check for license conflicts"""
        conflicts = []
        
        # Define incompatible license combinations
        incompatible_licenses = [
            (['GPL-3.0', 'GPL-2.0'], ['MIT', 'BSD', 'Apache-2.0']),
            (['AGPL-3.0'], ['MIT', 'BSD', 'Apache-2.0', 'GPL-2.0'])
        ]
        
        licenses = [dep.license for dep in dependencies if dep.license]
        
        for restrictive, permissive in incompatible_licenses:
            has_restrictive = any(lic in restrictive for lic in licenses)
            has_permissive = any(lic in permissive for lic in licenses)
            
            if has_restrictive and has_permissive:
                conflicts.append({
                    'type': 'license_conflict',
                    'restrictive_licenses': [lic for lic in licenses if lic in restrictive],
                    'permissive_licenses': [lic for lic in licenses if lic in permissive],
                    'description': 'Potential license conflict detected'
                })
        
        return conflicts

class JavaScriptDependencyAnalyzer:
    """JavaScript/Node.js dependency analyzer"""
    
    async def analyze_dependencies(self, project_path: str) -> DependencyGraph:
        """Analyze JavaScript dependencies"""
        start_time = time.time()
        
        try:
            graph = DependencyGraph(
                project_name=os.path.basename(project_path),
                language="javascript"
            )
            
            # Parse package.json
            package_json_path = os.path.join(project_path, 'package.json')
            if os.path.exists(package_json_path):
                deps = await self._parse_package_json(package_json_path)
                graph.dependencies.extend(deps)
            
            # Parse package-lock.json for exact versions
            lock_file_path = os.path.join(project_path, 'package-lock.json')
            if os.path.exists(lock_file_path):
                await self._enrich_with_lock_file(graph.dependencies, lock_file_path)
            
            # Parse yarn.lock if present
            yarn_lock_path = os.path.join(project_path, 'yarn.lock')
            if os.path.exists(yarn_lock_path):
                await self._enrich_with_yarn_lock(graph.dependencies, yarn_lock_path)
            
            # Build dependency graph
            graph.graph = await self._build_js_dependency_graph(project_path, graph.dependencies)
            
            # Check for vulnerabilities
            graph.vulnerabilities = await self._check_js_vulnerabilities(project_path)
            
            DEPENDENCY_ANALYSIS_REQUESTS.labels(language='javascript', status='success').inc()
            DEPENDENCIES_FOUND.labels(language='javascript', source_type='total').inc(len(graph.dependencies))
            
            return graph
            
        except Exception as e:
            logger.error(f"JavaScript dependency analysis failed: {e}")
            DEPENDENCY_ANALYSIS_REQUESTS.labels(language='javascript', status='error').inc()
            raise
        
        finally:
            latency = time.time() - start_time
            DEPENDENCY_ANALYSIS_LATENCY.labels(language='javascript').observe(latency)
    
    async def _parse_package_json(self, file_path: str) -> List[Dependency]:
        """Parse package.json file"""
        dependencies = []
        
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            data = json.loads(content)
            
            # Parse dependencies
            deps = data.get('dependencies', {})
            for name, version in deps.items():
                dep = Dependency(
                    name=name,
                    version_constraint=version,
                    dependency_type=DependencyType.DIRECT,
                    source=DependencySource.PACKAGE_MANAGER,
                    file_path=file_path
                )
                dependencies.append(dep)
            
            # Parse devDependencies
            dev_deps = data.get('devDependencies', {})
            for name, version in dev_deps.items():
                dep = Dependency(
                    name=name,
                    version_constraint=version,
                    dependency_type=DependencyType.DEV,
                    source=DependencySource.PACKAGE_MANAGER,
                    file_path=file_path
                )
                dependencies.append(dep)
            
            # Parse peerDependencies
            peer_deps = data.get('peerDependencies', {})
            for name, version in peer_deps.items():
                dep = Dependency(
                    name=name,
                    version_constraint=version,
                    dependency_type=DependencyType.PEER,
                    source=DependencySource.PACKAGE_MANAGER,
                    file_path=file_path
                )
                dependencies.append(dep)
                
        except Exception as e:
            logger.error(f"Failed to parse package.json: {e}")
        
        return dependencies
    
    async def _enrich_with_lock_file(self, dependencies: List[Dependency], lock_file_path: str):
        """Enrich dependencies with exact versions from package-lock.json"""
        try:
            async with aiofiles.open(lock_file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            data = json.loads(content)
            packages = data.get('packages', {})
            
            # Create mapping of package names to exact versions
            version_map = {}
            for package_path, package_info in packages.items():
                if package_path.startswith('node_modules/'):
                    package_name = package_path.replace('node_modules/', '')
                    version_map[package_name] = package_info.get('version')
            
            # Update dependencies with exact versions
            for dep in dependencies:
                if dep.name in version_map:
                    dep.version = version_map[dep.name]
                    
        except Exception as e:
            logger.error(f"Failed to parse package-lock.json: {e}")
    
    async def _enrich_with_yarn_lock(self, dependencies: List[Dependency], yarn_lock_path: str):
        """Enrich dependencies with exact versions from yarn.lock"""
        try:
            async with aiofiles.open(yarn_lock_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Parse yarn.lock format (simplified)
            version_map = {}
            current_package = None
            
            for line in content.split('\n'):
                line = line.strip()
                
                if line and not line.startswith('#') and ':' in line and not line.startswith(' '):
                    # Package declaration line
                    package_spec = line.split(':')[0].strip().strip('"')
                    package_name = package_spec.split('@')[0]
                    current_package = package_name
                
                elif line.startswith('  version ') and current_package:
                    # Version line
                    version = line.split('"')[1]
                    version_map[current_package] = version
                    current_package = None
            
            # Update dependencies with exact versions
            for dep in dependencies:
                if dep.name in version_map:
                    dep.version = version_map[dep.name]
                    
        except Exception as e:
            logger.error(f"Failed to parse yarn.lock: {e}")
    
    async def _build_js_dependency_graph(self, project_path: str, dependencies: List[Dependency]) -> Dict[str, List[str]]:
        """Build JavaScript dependency graph"""
        graph = {}
        
        try:
            # Use npm ls to get dependency tree
            result = subprocess.run(
                ['npm', 'ls', '--json', '--all'],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                graph = self._parse_npm_tree(data)
                
        except Exception as e:
            logger.error(f"Failed to build JS dependency graph: {e}")
        
        return graph
    
    def _parse_npm_tree(self, npm_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Parse npm dependency tree"""
        graph = {}
        
        def parse_dependencies(deps: Dict[str, Any], parent: Optional[str] = None):
            for name, info in deps.items():
                if parent:
                    if parent not in graph:
                        graph[parent] = []
                    graph[parent].append(name)
                
                if name not in graph:
                    graph[name] = []
                
                # Recursively parse nested dependencies
                nested_deps = info.get('dependencies', {})
                if nested_deps:
                    parse_dependencies(nested_deps, name)
        
        dependencies = npm_data.get('dependencies', {})
        parse_dependencies(dependencies)
        
        return graph
    
    async def _check_js_vulnerabilities(self, project_path: str) -> List[Vulnerability]:
        """Check JavaScript vulnerabilities using npm audit"""
        vulnerabilities = []
        
        try:
            result = subprocess.run(
                ['npm', 'audit', '--json'],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                vulnerabilities = self._parse_npm_audit(data)
                
            VULNERABILITY_SCANS.labels(language='javascript', status='success').inc()
            
        except Exception as e:
            logger.error(f"JavaScript vulnerability check failed: {e}")
            VULNERABILITY_SCANS.labels(language='javascript', status='error').inc()
        
        return vulnerabilities
    
    def _parse_npm_audit(self, audit_data: Dict[str, Any]) -> List[Vulnerability]:
        """Parse npm audit results"""
        vulnerabilities = []
        
        advisories = audit_data.get('advisories', {})
        for advisory_id, advisory in advisories.items():
            vuln = Vulnerability(
                id=str(advisory_id),
                title=advisory.get('title', ''),
                description=advisory.get('overview', ''),
                severity=VulnerabilitySeverity(advisory.get('severity', 'low')),
                cvss_score=advisory.get('cvss_score'),
                cve_id=advisory.get('cve'),
                affected_versions=advisory.get('vulnerable_versions', []),
                patched_versions=advisory.get('patched_versions', []),
                references=advisory.get('references', [])
            )
            vulnerabilities.append(vuln)
        
        return vulnerabilities

class DependencyAnalyzer:
    """Main dependency analyzer that coordinates language-specific analyzers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.python_analyzer = PythonDependencyAnalyzer()
        self.javascript_analyzer = JavaScriptDependencyAnalyzer()
    
    async def analyze_project_dependencies(self, project_path: str, language: str) -> DependencyGraph:
        """Analyze project dependencies based on language"""
        if language.lower() == 'python':
            return await self.python_analyzer.analyze_dependencies(project_path)
        elif language.lower() in ['javascript', 'typescript', 'node.js']:
            return await self.javascript_analyzer.analyze_dependencies(project_path)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    async def analyze_multi_language_project(self, project_path: str) -> Dict[str, DependencyGraph]:
        """Analyze project with multiple languages"""
        results = {}
        
        # Detect languages in project
        languages = self._detect_project_languages(project_path)
        
        for language in languages:
            try:
                graph = await self.analyze_project_dependencies(project_path, language)
                results[language] = graph
            except Exception as e:
                logger.error(f"Failed to analyze {language} dependencies: {e}")
        
        return results
    
    def _detect_project_languages(self, project_path: str) -> List[str]:
        """Detect programming languages in project"""
        languages = []
        
        # Check for Python
        python_files = ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile']
        if any(os.path.exists(os.path.join(project_path, f)) for f in python_files):
            languages.append('python')
        
        # Check for JavaScript/Node.js
        js_files = ['package.json', 'package-lock.json', 'yarn.lock']
        if any(os.path.exists(os.path.join(project_path, f)) for f in js_files):
            languages.append('javascript')
        
        return languages
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'analyzers': {
                'python': 'available',
                'javascript': 'available'
            }
        }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        config = {}
        analyzer = DependencyAnalyzer(config)
        
        # Example usage
        project_path = "/path/to/project"
        if os.path.exists(project_path):
            # Analyze single language
            try:
                graph = await analyzer.analyze_project_dependencies(project_path, 'python')
                print(f"Found {len(graph.dependencies)} dependencies")
                print(f"Vulnerabilities: {len(graph.vulnerabilities)}")
            except Exception as e:
                print(f"Analysis failed: {e}")
            
            # Analyze multi-language project
            results = await analyzer.analyze_multi_language_project(project_path)
            print(f"Analyzed {len(results)} languages")
        
        # Health check
        health = await analyzer.health_check()
        print(f"Health: {health}")
    
    asyncio.run(main())

