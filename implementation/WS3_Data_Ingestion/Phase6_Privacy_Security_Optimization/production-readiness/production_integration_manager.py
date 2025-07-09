"""
Production Integration Manager for Nexus Architect
Implements final system integration, production readiness validation,
and comprehensive deployment orchestration.
"""

import logging
import asyncio
import time
import subprocess
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import secrets
import requests
import psutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentStatus(Enum):
    """Component status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

class DeploymentStage(Enum):
    """Deployment stages"""
    PREPARATION = "preparation"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    VERIFICATION = "verification"
    COMPLETION = "completion"

class IntegrationTest(Enum):
    """Types of integration tests"""
    HEALTH_CHECK = "health_check"
    API_CONNECTIVITY = "api_connectivity"
    DATABASE_CONNECTION = "database_connection"
    CACHE_CONNECTION = "cache_connection"
    SECURITY_VALIDATION = "security_validation"
    PERFORMANCE_BASELINE = "performance_baseline"
    COMPLIANCE_CHECK = "compliance_check"

@dataclass
class SystemComponent:
    """System component definition"""
    component_id: str
    name: str
    service_type: str  # api, database, cache, monitoring
    host: str
    port: int
    health_endpoint: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_check: Optional[datetime] = None
    response_time: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class IntegrationTestResult:
    """Integration test result"""
    test_id: str
    test_type: IntegrationTest
    component_id: str
    status: str  # pass, fail, warning
    duration: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DeploymentPlan:
    """Production deployment plan"""
    plan_id: str
    name: str
    description: str
    components: List[str]
    deployment_order: List[str]
    rollback_plan: List[str]
    validation_tests: List[IntegrationTest]
    estimated_duration: timedelta
    created_at: datetime = field(default_factory=datetime.utcnow)

class ProductionIntegrationManager:
    """
    Comprehensive production integration manager implementing enterprise-grade
    system integration, validation, and deployment orchestration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the production integration manager"""
        self.config = config
        self.database_config = config.get('database', {})
        
        # System components
        self.components: Dict[str, SystemComponent] = {}
        self.test_results: List[IntegrationTestResult] = []
        self.deployment_plans: Dict[str, DeploymentPlan] = {}
        
        # Integration status
        self.integration_status = "initializing"
        self.last_full_validation = None
        
        # Initialize system components
        self._initialize_system_components()
        
        logger.info("Production Integration Manager initialized successfully")
    
    def _initialize_system_components(self):
        """Initialize system components from configuration"""
        try:
            # WS1 Core Foundation Components
            self.components["auth_service"] = SystemComponent(
                component_id="auth_service",
                name="Authentication Service",
                service_type="api",
                host="localhost",
                port=8001,
                health_endpoint="/health",
                dependencies=["postgres", "redis"]
            )
            
            self.components["api_gateway"] = SystemComponent(
                component_id="api_gateway",
                name="API Gateway (Kong)",
                service_type="api",
                host="localhost",
                port=8000,
                health_endpoint="/health",
                dependencies=["auth_service"]
            )
            
            # WS2 AI Intelligence Components
            self.components["ai_orchestrator"] = SystemComponent(
                component_id="ai_orchestrator",
                name="AI Orchestrator",
                service_type="api",
                host="localhost",
                port=8002,
                health_endpoint="/health",
                dependencies=["postgres", "redis"]
            )
            
            self.components["knowledge_graph"] = SystemComponent(
                component_id="knowledge_graph",
                name="Knowledge Graph (Neo4j)",
                service_type="database",
                host="localhost",
                port=7474,
                health_endpoint="/db/data/",
                dependencies=[]
            )
            
            # WS3 Data Ingestion Components
            self.components["git_connector"] = SystemComponent(
                component_id="git_connector",
                name="Git Platform Manager",
                service_type="api",
                host="localhost",
                port=8003,
                health_endpoint="/health",
                dependencies=["postgres"]
            )
            
            self.components["doc_processor"] = SystemComponent(
                component_id="doc_processor",
                name="Document Processor",
                service_type="api",
                host="localhost",
                port=8004,
                health_endpoint="/health",
                dependencies=["postgres", "redis"]
            )
            
            self.components["stream_processor"] = SystemComponent(
                component_id="stream_processor",
                name="Stream Processor (Kafka)",
                service_type="api",
                host="localhost",
                port=9092,
                dependencies=[]
            )
            
            # WS3 Phase 6 Components
            self.components["privacy_manager"] = SystemComponent(
                component_id="privacy_manager",
                name="Data Privacy Manager",
                service_type="api",
                host="localhost",
                port=8010,
                health_endpoint="/health",
                dependencies=["postgres"]
            )
            
            self.components["security_manager"] = SystemComponent(
                component_id="security_manager",
                name="Security Manager",
                service_type="api",
                host="localhost",
                port=8011,
                health_endpoint="/health",
                dependencies=["postgres", "redis"]
            )
            
            self.components["compliance_manager"] = SystemComponent(
                component_id="compliance_manager",
                name="Compliance Manager",
                service_type="api",
                host="localhost",
                port=8012,
                health_endpoint="/health",
                dependencies=["postgres"]
            )
            
            self.components["performance_optimizer"] = SystemComponent(
                component_id="performance_optimizer",
                name="Performance Optimizer",
                service_type="api",
                host="localhost",
                port=8013,
                health_endpoint="/health",
                dependencies=["redis"]
            )
            
            # Infrastructure Components
            self.components["postgres"] = SystemComponent(
                component_id="postgres",
                name="PostgreSQL Database",
                service_type="database",
                host="localhost",
                port=5432,
                dependencies=[]
            )
            
            self.components["redis"] = SystemComponent(
                component_id="redis",
                name="Redis Cache",
                service_type="cache",
                host="localhost",
                port=6379,
                dependencies=[]
            )
            
            self.components["prometheus"] = SystemComponent(
                component_id="prometheus",
                name="Prometheus Monitoring",
                service_type="monitoring",
                host="localhost",
                port=9090,
                health_endpoint="/-/healthy",
                dependencies=[]
            )
            
            logger.info(f"Initialized {len(self.components)} system components")
            
        except Exception as e:
            logger.error(f"Error initializing system components: {str(e)}")
    
    async def check_component_health(self, component_id: str) -> ComponentStatus:
        """
        Check health status of a specific component
        
        Args:
            component_id: Component identifier
            
        Returns:
            Component status
        """
        try:
            if component_id not in self.components:
                return ComponentStatus.UNKNOWN
            
            component = self.components[component_id]
            start_time = time.time()
            
            try:
                if component.health_endpoint:
                    # HTTP health check
                    url = f"http://{component.host}:{component.port}{component.health_endpoint}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        component.status = ComponentStatus.HEALTHY
                        component.response_time = time.time() - start_time
                        component.error_message = None
                    else:
                        component.status = ComponentStatus.WARNING
                        component.error_message = f"HTTP {response.status_code}"
                else:
                    # TCP port check
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((component.host, component.port))
                    sock.close()
                    
                    if result == 0:
                        component.status = ComponentStatus.HEALTHY
                        component.response_time = time.time() - start_time
                        component.error_message = None
                    else:
                        component.status = ComponentStatus.OFFLINE
                        component.error_message = "Port not accessible"
                        
            except Exception as e:
                component.status = ComponentStatus.CRITICAL
                component.error_message = str(e)
            
            component.last_check = datetime.utcnow()
            
            return component.status
            
        except Exception as e:
            logger.error(f"Error checking component health: {str(e)}")
            return ComponentStatus.UNKNOWN
    
    async def run_integration_test(self, test_type: IntegrationTest, 
                                 component_id: str) -> IntegrationTestResult:
        """
        Run a specific integration test
        
        Args:
            test_type: Type of integration test
            component_id: Component to test
            
        Returns:
            Test result
        """
        try:
            test_id = secrets.token_hex(8)
            start_time = time.time()
            
            if test_type == IntegrationTest.HEALTH_CHECK:
                status = await self.check_component_health(component_id)
                test_status = "pass" if status == ComponentStatus.HEALTHY else "fail"
                details = {
                    "component_status": status.value,
                    "response_time": self.components[component_id].response_time
                }
                error_message = self.components[component_id].error_message
                
            elif test_type == IntegrationTest.API_CONNECTIVITY:
                test_status, details, error_message = await self._test_api_connectivity(component_id)
                
            elif test_type == IntegrationTest.DATABASE_CONNECTION:
                test_status, details, error_message = await self._test_database_connection(component_id)
                
            elif test_type == IntegrationTest.CACHE_CONNECTION:
                test_status, details, error_message = await self._test_cache_connection(component_id)
                
            elif test_type == IntegrationTest.SECURITY_VALIDATION:
                test_status, details, error_message = await self._test_security_validation(component_id)
                
            elif test_type == IntegrationTest.PERFORMANCE_BASELINE:
                test_status, details, error_message = await self._test_performance_baseline(component_id)
                
            elif test_type == IntegrationTest.COMPLIANCE_CHECK:
                test_status, details, error_message = await self._test_compliance_check(component_id)
                
            else:
                test_status = "fail"
                details = {}
                error_message = f"Unknown test type: {test_type.value}"
            
            duration = time.time() - start_time
            
            result = IntegrationTestResult(
                test_id=test_id,
                test_type=test_type,
                component_id=component_id,
                status=test_status,
                duration=duration,
                details=details,
                error_message=error_message
            )
            
            self.test_results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running integration test: {str(e)}")
            return IntegrationTestResult(
                test_id=secrets.token_hex(8),
                test_type=test_type,
                component_id=component_id,
                status="fail",
                duration=0,
                details={},
                error_message=str(e)
            )
    
    async def _test_api_connectivity(self, component_id: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Test API connectivity"""
        try:
            component = self.components[component_id]
            
            if component.service_type != "api":
                return "fail", {}, "Component is not an API service"
            
            # Test basic API endpoints
            base_url = f"http://{component.host}:{component.port}"
            
            # Test health endpoint
            health_response = requests.get(f"{base_url}/health", timeout=10)
            
            details = {
                "health_status": health_response.status_code,
                "health_response_time": health_response.elapsed.total_seconds()
            }
            
            if health_response.status_code == 200:
                return "pass", details, None
            else:
                return "fail", details, f"Health check failed: {health_response.status_code}"
                
        except Exception as e:
            return "fail", {}, str(e)
    
    async def _test_database_connection(self, component_id: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Test database connection"""
        try:
            component = self.components[component_id]
            
            if component.service_type != "database":
                return "fail", {}, "Component is not a database service"
            
            if component_id == "postgres":
                import psycopg2
                conn = psycopg2.connect(
                    host=component.host,
                    port=component.port,
                    database=self.database_config.get('database', 'nexus_architect'),
                    user=self.database_config.get('user', 'postgres'),
                    password=self.database_config.get('password', 'password')
                )
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                
                details = {"connection_test": "success", "query_result": result[0]}
                return "pass", details, None
                
            elif component_id == "knowledge_graph":
                # Test Neo4j connection
                response = requests.get(f"http://{component.host}:{component.port}/db/data/", timeout=10)
                
                details = {"neo4j_status": response.status_code}
                
                if response.status_code == 200:
                    return "pass", details, None
                else:
                    return "fail", details, f"Neo4j connection failed: {response.status_code}"
            
            return "fail", {}, "Unknown database type"
            
        except Exception as e:
            return "fail", {}, str(e)
    
    async def _test_cache_connection(self, component_id: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Test cache connection"""
        try:
            component = self.components[component_id]
            
            if component.service_type != "cache":
                return "fail", {}, "Component is not a cache service"
            
            if component_id == "redis":
                import redis
                r = redis.Redis(host=component.host, port=component.port, db=0)
                
                # Test basic operations
                test_key = f"integration_test_{int(time.time())}"
                r.set(test_key, "test_value", ex=60)
                value = r.get(test_key)
                r.delete(test_key)
                
                details = {
                    "redis_ping": r.ping(),
                    "set_get_test": value.decode() if value else None
                }
                
                if details["redis_ping"] and details["set_get_test"] == "test_value":
                    return "pass", details, None
                else:
                    return "fail", details, "Redis operations failed"
            
            return "fail", {}, "Unknown cache type"
            
        except Exception as e:
            return "fail", {}, str(e)
    
    async def _test_security_validation(self, component_id: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Test security validation"""
        try:
            component = self.components[component_id]
            
            if component_id == "security_manager":
                # Test security manager endpoints
                base_url = f"http://{component.host}:{component.port}"
                
                # Test security scan
                scan_response = requests.post(f"{base_url}/security-scan", timeout=30)
                
                details = {
                    "security_scan_status": scan_response.status_code,
                    "scan_results": scan_response.json() if scan_response.status_code == 200 else None
                }
                
                if scan_response.status_code == 200:
                    scan_data = scan_response.json()
                    risk_score = scan_data.get("risk_score", 100)
                    
                    if risk_score < 50:
                        return "pass", details, None
                    else:
                        return "warning", details, f"High risk score: {risk_score}"
                else:
                    return "fail", details, "Security scan failed"
            
            return "pass", {"message": "No specific security tests for this component"}, None
            
        except Exception as e:
            return "fail", {}, str(e)
    
    async def _test_performance_baseline(self, component_id: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Test performance baseline"""
        try:
            component = self.components[component_id]
            
            if component.service_type == "api":
                # Test API response times
                base_url = f"http://{component.host}:{component.port}"
                
                response_times = []
                for _ in range(5):
                    start_time = time.time()
                    response = requests.get(f"{base_url}/health", timeout=10)
                    response_time = time.time() - start_time
                    response_times.append(response_time)
                
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                
                details = {
                    "average_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "response_times": response_times
                }
                
                if avg_response_time < 1.0:  # Less than 1 second
                    return "pass", details, None
                elif avg_response_time < 2.0:
                    return "warning", details, "Response time above optimal"
                else:
                    return "fail", details, "Response time too high"
            
            return "pass", {"message": "No specific performance tests for this component"}, None
            
        except Exception as e:
            return "fail", {}, str(e)
    
    async def _test_compliance_check(self, component_id: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Test compliance check"""
        try:
            component = self.components[component_id]
            
            if component_id == "compliance_manager":
                # Test compliance manager
                base_url = f"http://{component.host}:{component.port}"
                
                # Get compliance metrics
                metrics_response = requests.get(f"{base_url}/metrics", timeout=30)
                
                details = {
                    "metrics_status": metrics_response.status_code,
                    "compliance_data": metrics_response.json() if metrics_response.status_code == 200 else None
                }
                
                if metrics_response.status_code == 200:
                    return "pass", details, None
                else:
                    return "fail", details, "Compliance metrics unavailable"
            
            return "pass", {"message": "No specific compliance tests for this component"}, None
            
        except Exception as e:
            return "fail", {}, str(e)
    
    async def run_full_integration_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive integration validation across all components
        
        Returns:
            Validation results
        """
        try:
            validation_id = secrets.token_hex(16)
            start_time = datetime.utcnow()
            
            validation_results = {
                "validation_id": validation_id,
                "start_time": start_time.isoformat(),
                "component_results": {},
                "overall_status": "unknown",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "warning_tests": 0
            }
            
            # Test all components
            for component_id, component in self.components.items():
                component_results = {
                    "component_name": component.name,
                    "component_type": component.service_type,
                    "tests": []
                }
                
                # Run health check for all components
                health_result = await self.run_integration_test(
                    IntegrationTest.HEALTH_CHECK, component_id
                )
                component_results["tests"].append({
                    "test_type": health_result.test_type.value,
                    "status": health_result.status,
                    "duration": health_result.duration,
                    "details": health_result.details,
                    "error_message": health_result.error_message
                })
                
                # Run specific tests based on component type
                if component.service_type == "api":
                    api_result = await self.run_integration_test(
                        IntegrationTest.API_CONNECTIVITY, component_id
                    )
                    component_results["tests"].append({
                        "test_type": api_result.test_type.value,
                        "status": api_result.status,
                        "duration": api_result.duration,
                        "details": api_result.details,
                        "error_message": api_result.error_message
                    })
                    
                    perf_result = await self.run_integration_test(
                        IntegrationTest.PERFORMANCE_BASELINE, component_id
                    )
                    component_results["tests"].append({
                        "test_type": perf_result.test_type.value,
                        "status": perf_result.status,
                        "duration": perf_result.duration,
                        "details": perf_result.details,
                        "error_message": perf_result.error_message
                    })
                
                elif component.service_type == "database":
                    db_result = await self.run_integration_test(
                        IntegrationTest.DATABASE_CONNECTION, component_id
                    )
                    component_results["tests"].append({
                        "test_type": db_result.test_type.value,
                        "status": db_result.status,
                        "duration": db_result.duration,
                        "details": db_result.details,
                        "error_message": db_result.error_message
                    })
                
                elif component.service_type == "cache":
                    cache_result = await self.run_integration_test(
                        IntegrationTest.CACHE_CONNECTION, component_id
                    )
                    component_results["tests"].append({
                        "test_type": cache_result.test_type.value,
                        "status": cache_result.status,
                        "duration": cache_result.duration,
                        "details": cache_result.details,
                        "error_message": cache_result.error_message
                    })
                
                # Special tests for specific components
                if component_id == "security_manager":
                    security_result = await self.run_integration_test(
                        IntegrationTest.SECURITY_VALIDATION, component_id
                    )
                    component_results["tests"].append({
                        "test_type": security_result.test_type.value,
                        "status": security_result.status,
                        "duration": security_result.duration,
                        "details": security_result.details,
                        "error_message": security_result.error_message
                    })
                
                if component_id == "compliance_manager":
                    compliance_result = await self.run_integration_test(
                        IntegrationTest.COMPLIANCE_CHECK, component_id
                    )
                    component_results["tests"].append({
                        "test_type": compliance_result.test_type.value,
                        "status": compliance_result.status,
                        "duration": compliance_result.duration,
                        "details": compliance_result.details,
                        "error_message": compliance_result.error_message
                    })
                
                validation_results["component_results"][component_id] = component_results
            
            # Calculate overall statistics
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            warning_tests = 0
            
            for component_results in validation_results["component_results"].values():
                for test in component_results["tests"]:
                    total_tests += 1
                    if test["status"] == "pass":
                        passed_tests += 1
                    elif test["status"] == "fail":
                        failed_tests += 1
                    elif test["status"] == "warning":
                        warning_tests += 1
            
            validation_results.update({
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "warning_tests": warning_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            })
            
            # Determine overall status
            if failed_tests == 0 and warning_tests == 0:
                validation_results["overall_status"] = "excellent"
            elif failed_tests == 0:
                validation_results["overall_status"] = "good"
            elif failed_tests < total_tests * 0.2:  # Less than 20% failures
                validation_results["overall_status"] = "warning"
            else:
                validation_results["overall_status"] = "critical"
            
            validation_results["end_time"] = datetime.utcnow().isoformat()
            validation_results["duration"] = (datetime.utcnow() - start_time).total_seconds()
            
            self.last_full_validation = datetime.utcnow()
            self.integration_status = validation_results["overall_status"]
            
            logger.info(f"Full integration validation completed: {validation_results['overall_status']}")
            logger.info(f"Tests: {passed_tests}/{total_tests} passed, {failed_tests} failed, {warning_tests} warnings")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error running full integration validation: {str(e)}")
            return {"error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            System status dictionary
        """
        try:
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "integration_status": self.integration_status,
                "last_validation": self.last_full_validation.isoformat() if self.last_full_validation else None,
                "components": {},
                "summary": {
                    "total_components": len(self.components),
                    "healthy_components": 0,
                    "warning_components": 0,
                    "critical_components": 0,
                    "offline_components": 0
                }
            }
            
            # Get status for each component
            for component_id, component in self.components.items():
                component_status = await self.check_component_health(component_id)
                
                status["components"][component_id] = {
                    "name": component.name,
                    "type": component.service_type,
                    "status": component_status.value,
                    "host": component.host,
                    "port": component.port,
                    "last_check": component.last_check.isoformat() if component.last_check else None,
                    "response_time": component.response_time,
                    "error_message": component.error_message,
                    "dependencies": component.dependencies
                }
                
                # Update summary counts
                if component_status == ComponentStatus.HEALTHY:
                    status["summary"]["healthy_components"] += 1
                elif component_status == ComponentStatus.WARNING:
                    status["summary"]["warning_components"] += 1
                elif component_status == ComponentStatus.CRITICAL:
                    status["summary"]["critical_components"] += 1
                elif component_status == ComponentStatus.OFFLINE:
                    status["summary"]["offline_components"] += 1
            
            # Calculate health percentage
            total = status["summary"]["total_components"]
            healthy = status["summary"]["healthy_components"]
            status["summary"]["health_percentage"] = (healthy / total * 100) if total > 0 else 0
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {"error": str(e)}

def create_integration_api(integration_manager: ProductionIntegrationManager):
    """Create Flask API for production integration management"""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "service": "production_integration_manager"})
    
    @app.route('/system/status', methods=['GET'])
    async def get_system_status():
        try:
            status = await integration_manager.get_system_status()
            return jsonify(status)
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/component/<component_id>/health', methods=['GET'])
    async def check_component_health(component_id):
        try:
            status = await integration_manager.check_component_health(component_id)
            
            component = integration_manager.components.get(component_id)
            if not component:
                return jsonify({"status": "error", "message": "Component not found"}), 404
            
            return jsonify({
                "component_id": component_id,
                "name": component.name,
                "status": status.value,
                "last_check": component.last_check.isoformat() if component.last_check else None,
                "response_time": component.response_time,
                "error_message": component.error_message
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/test/<test_type>/<component_id>', methods=['POST'])
    async def run_integration_test(test_type, component_id):
        try:
            test_enum = IntegrationTest(test_type)
            result = await integration_manager.run_integration_test(test_enum, component_id)
            
            return jsonify({
                "test_id": result.test_id,
                "test_type": result.test_type.value,
                "component_id": result.component_id,
                "status": result.status,
                "duration": result.duration,
                "details": result.details,
                "error_message": result.error_message,
                "timestamp": result.timestamp.isoformat()
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/validate/full', methods=['POST'])
    async def run_full_validation():
        try:
            results = await integration_manager.run_full_integration_validation()
            return jsonify(results)
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/test-results', methods=['GET'])
    def get_test_results():
        try:
            limit = request.args.get('limit', 100, type=int)
            
            recent_results = integration_manager.test_results[-limit:]
            
            results = [
                {
                    "test_id": result.test_id,
                    "test_type": result.test_type.value,
                    "component_id": result.component_id,
                    "status": result.status,
                    "duration": result.duration,
                    "timestamp": result.timestamp.isoformat(),
                    "error_message": result.error_message
                }
                for result in recent_results
            ]
            
            return jsonify({
                "status": "success",
                "results": results,
                "count": len(results)
            })
            
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
    
    # Initialize production integration manager
    integration_manager = ProductionIntegrationManager(config)
    
    # Create Flask API
    app = create_integration_api(integration_manager)
    
    print("Production Integration Manager API starting...")
    print("Available endpoints:")
    print("  GET /system/status - Get comprehensive system status")
    print("  GET /component/<id>/health - Check component health")
    print("  POST /test/<type>/<component_id> - Run integration test")
    print("  POST /validate/full - Run full system validation")
    print("  GET /test-results - Get recent test results")
    
    app.run(host='0.0.0.0', port=8014, debug=False)

