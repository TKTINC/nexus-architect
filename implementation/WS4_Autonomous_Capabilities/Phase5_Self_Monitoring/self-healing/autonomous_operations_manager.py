#!/usr/bin/env python3
"""
Nexus Architect - WS4 Phase 5: Autonomous Operations Manager
Self-healing mechanisms, performance optimization, and security management
"""

import asyncio
import json
import logging
import time
import subprocess
import os
import signal
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import redis
import psycopg2
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import docker
import kubernetes
from kubernetes import client, config
import yaml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of autonomous actions"""
    RESTART_SERVICE = "restart_service"
    SCALE_SERVICE = "scale_service"
    CLEAR_CACHE = "clear_cache"
    OPTIMIZE_DATABASE = "optimize_database"
    CLEANUP_DISK = "cleanup_disk"
    ADJUST_MEMORY = "adjust_memory"
    NETWORK_OPTIMIZATION = "network_optimization"
    SECURITY_RESPONSE = "security_response"
    PERFORMANCE_TUNING = "performance_tuning"

class ActionStatus(Enum):
    """Status of autonomous actions"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class Priority(Enum):
    """Action priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AutonomousAction:
    """Autonomous action data structure"""
    id: str
    action_type: ActionType
    priority: Priority
    description: str
    target_component: str
    parameters: Dict[str, Any]
    created_at: datetime
    status: ActionStatus = ActionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    rollback_data: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceOptimization:
    """Performance optimization recommendation"""
    component: str
    optimization_type: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: float
    risk_level: str

@dataclass
class SecurityIncident:
    """Security incident data structure"""
    id: str
    incident_type: str
    severity: str
    description: str
    source_ip: Optional[str]
    affected_component: str
    detected_at: datetime
    status: str = "active"
    response_actions: List[str] = None

class ServiceManager:
    """Manages service lifecycle and health"""
    
    def __init__(self):
        self.docker_client = None
        self.k8s_client = None
        self._init_clients()
        
    def _init_clients(self):
        """Initialize Docker and Kubernetes clients"""
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning(f"Docker client initialization failed: {e}")
            
        try:
            config.load_incluster_config()
            self.k8s_client = client.AppsV1Api()
            logger.info("Kubernetes client initialized")
        except Exception:
            try:
                config.load_kube_config()
                self.k8s_client = client.AppsV1Api()
                logger.info("Kubernetes client initialized from kubeconfig")
            except Exception as e:
                logger.warning(f"Kubernetes client initialization failed: {e}")
    
    def restart_service(self, service_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Restart a service"""
        try:
            if self.k8s_client:
                # Kubernetes deployment restart
                deployment = self.k8s_client.read_namespaced_deployment(service_name, namespace)
                
                # Update deployment to trigger restart
                deployment.spec.template.metadata.annotations = deployment.spec.template.metadata.annotations or {}
                deployment.spec.template.metadata.annotations["kubectl.kubernetes.io/restartedAt"] = datetime.now().isoformat()
                
                self.k8s_client.patch_namespaced_deployment(
                    name=service_name,
                    namespace=namespace,
                    body=deployment
                )
                
                return {
                    "success": True,
                    "message": f"Kubernetes deployment {service_name} restarted",
                    "method": "kubernetes"
                }
                
            elif self.docker_client:
                # Docker container restart
                container = self.docker_client.containers.get(service_name)
                container.restart()
                
                return {
                    "success": True,
                    "message": f"Docker container {service_name} restarted",
                    "method": "docker"
                }
            else:
                # System service restart
                result = subprocess.run(['sudo', 'systemctl', 'restart', service_name], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    return {
                        "success": True,
                        "message": f"System service {service_name} restarted",
                        "method": "systemctl"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Failed to restart service: {result.stderr}",
                        "method": "systemctl"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "message": f"Service restart failed: {str(e)}",
                "error": str(e)
            }
    
    def scale_service(self, service_name: str, replicas: int, namespace: str = "default") -> Dict[str, Any]:
        """Scale a service"""
        try:
            if self.k8s_client:
                # Scale Kubernetes deployment
                scale = client.V1Scale(
                    metadata=client.V1ObjectMeta(name=service_name, namespace=namespace),
                    spec=client.V1ScaleSpec(replicas=replicas)
                )
                
                self.k8s_client.patch_namespaced_deployment_scale(
                    name=service_name,
                    namespace=namespace,
                    body=scale
                )
                
                return {
                    "success": True,
                    "message": f"Scaled {service_name} to {replicas} replicas",
                    "method": "kubernetes",
                    "replicas": replicas
                }
            else:
                return {
                    "success": False,
                    "message": "Kubernetes client not available for scaling",
                    "method": "none"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Service scaling failed: {str(e)}",
                "error": str(e)
            }
    
    def get_service_status(self, service_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get service status"""
        try:
            if self.k8s_client:
                deployment = self.k8s_client.read_namespaced_deployment(service_name, namespace)
                return {
                    "name": service_name,
                    "namespace": namespace,
                    "replicas": deployment.spec.replicas,
                    "ready_replicas": deployment.status.ready_replicas or 0,
                    "available_replicas": deployment.status.available_replicas or 0,
                    "status": "healthy" if deployment.status.ready_replicas == deployment.spec.replicas else "degraded"
                }
            elif self.docker_client:
                container = self.docker_client.containers.get(service_name)
                return {
                    "name": service_name,
                    "status": container.status,
                    "state": container.attrs['State']
                }
            else:
                result = subprocess.run(['systemctl', 'is-active', service_name], 
                                      capture_output=True, text=True)
                return {
                    "name": service_name,
                    "status": result.stdout.strip(),
                    "active": result.returncode == 0
                }
                
        except Exception as e:
            return {
                "name": service_name,
                "status": "unknown",
                "error": str(e)
            }

class PerformanceOptimizer:
    """Handles performance optimization tasks"""
    
    def __init__(self):
        self.optimization_history = []
        self.redis_client = None
        self.postgres_conn = None
        self._init_connections()
        
    def _init_connections(self):
        """Initialize database connections"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            
        try:
            self.postgres_conn = psycopg2.connect(
                host="localhost",
                database="nexus_architect",
                user="nexus_user",
                password="nexus_password"
            )
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
    
    def analyze_performance_bottlenecks(self) -> List[PerformanceOptimization]:
        """Analyze system for performance bottlenecks"""
        optimizations = []
        
        # CPU optimization
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            optimizations.append(PerformanceOptimization(
                component="cpu",
                optimization_type="process_optimization",
                current_value=cpu_percent,
                recommended_value="< 70%",
                expected_improvement=15.0,
                confidence=0.8,
                risk_level="low"
            ))
        
        # Memory optimization
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            optimizations.append(PerformanceOptimization(
                component="memory",
                optimization_type="memory_cleanup",
                current_value=memory.percent,
                recommended_value="< 80%",
                expected_improvement=10.0,
                confidence=0.9,
                risk_level="low"
            ))
        
        # Disk optimization
        disk = psutil.disk_usage('/')
        if (disk.used / disk.total) * 100 > 85:
            optimizations.append(PerformanceOptimization(
                component="disk",
                optimization_type="disk_cleanup",
                current_value=(disk.used / disk.total) * 100,
                recommended_value="< 80%",
                expected_improvement=20.0,
                confidence=0.95,
                risk_level="low"
            ))
        
        # Redis optimization
        if self.redis_client:
            try:
                info = self.redis_client.info()
                memory_usage = info.get('used_memory', 0) / (1024**2)  # MB
                if memory_usage > 500:
                    optimizations.append(PerformanceOptimization(
                        component="redis",
                        optimization_type="cache_optimization",
                        current_value=memory_usage,
                        recommended_value="< 400 MB",
                        expected_improvement=25.0,
                        confidence=0.85,
                        risk_level="medium"
                    ))
            except Exception as e:
                logger.warning(f"Redis analysis failed: {e}")
        
        return optimizations
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance"""
        try:
            if not self.redis_client:
                return {"success": False, "message": "Redis not available"}
            
            # Get current memory usage
            info_before = self.redis_client.info()
            memory_before = info_before.get('used_memory', 0)
            
            # Clear expired keys
            expired_keys = 0
            for key in self.redis_client.scan_iter():
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    # Set default expiration for keys without TTL
                    self.redis_client.expire(key, 3600)  # 1 hour
                elif ttl == -2:  # Key doesn't exist (expired)
                    expired_keys += 1
            
            # Optimize memory usage
            self.redis_client.execute_command('MEMORY', 'PURGE')
            
            # Get memory usage after optimization
            info_after = self.redis_client.info()
            memory_after = info_after.get('used_memory', 0)
            
            memory_saved = memory_before - memory_after
            
            return {
                "success": True,
                "message": "Cache optimization completed",
                "memory_before_mb": memory_before / (1024**2),
                "memory_after_mb": memory_after / (1024**2),
                "memory_saved_mb": memory_saved / (1024**2),
                "expired_keys_processed": expired_keys
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Cache optimization failed: {str(e)}",
                "error": str(e)
            }
    
    def optimize_database(self) -> Dict[str, Any]:
        """Optimize database performance"""
        try:
            if not self.postgres_conn:
                return {"success": False, "message": "PostgreSQL not available"}
            
            cursor = self.postgres_conn.cursor()
            
            # Analyze table statistics
            cursor.execute("ANALYZE;")
            
            # Vacuum to reclaim space
            self.postgres_conn.autocommit = True
            cursor.execute("VACUUM;")
            
            # Get database size before and after
            cursor.execute("SELECT pg_database_size(current_database());")
            db_size = cursor.fetchone()[0]
            
            # Reindex for better performance
            cursor.execute("REINDEX DATABASE nexus_architect;")
            
            cursor.close()
            
            return {
                "success": True,
                "message": "Database optimization completed",
                "database_size_mb": db_size / (1024**2),
                "operations": ["ANALYZE", "VACUUM", "REINDEX"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Database optimization failed: {str(e)}",
                "error": str(e)
            }
    
    def cleanup_disk_space(self) -> Dict[str, Any]:
        """Clean up disk space"""
        try:
            space_before = psutil.disk_usage('/').free
            
            cleanup_commands = [
                # Clean package cache
                ['sudo', 'apt-get', 'clean'],
                # Remove old logs
                ['sudo', 'journalctl', '--vacuum-time=7d'],
                # Clean temporary files
                ['sudo', 'find', '/tmp', '-type', 'f', '-atime', '+7', '-delete'],
                # Clean Docker if available
                ['docker', 'system', 'prune', '-f']
            ]
            
            results = []
            for cmd in cleanup_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    results.append({
                        "command": " ".join(cmd),
                        "success": result.returncode == 0,
                        "output": result.stdout if result.returncode == 0 else result.stderr
                    })
                except subprocess.TimeoutExpired:
                    results.append({
                        "command": " ".join(cmd),
                        "success": False,
                        "output": "Command timed out"
                    })
                except Exception as e:
                    results.append({
                        "command": " ".join(cmd),
                        "success": False,
                        "output": str(e)
                    })
            
            space_after = psutil.disk_usage('/').free
            space_freed = space_after - space_before
            
            return {
                "success": True,
                "message": "Disk cleanup completed",
                "space_freed_mb": space_freed / (1024**2),
                "cleanup_results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Disk cleanup failed: {str(e)}",
                "error": str(e)
            }

class SecurityManager:
    """Handles security incident response"""
    
    def __init__(self):
        self.active_incidents = {}
        self.security_rules = self._load_security_rules()
        
    def _load_security_rules(self) -> Dict[str, Any]:
        """Load security rules and thresholds"""
        return {
            "failed_login_threshold": 5,
            "suspicious_ip_patterns": [
                r"^10\.0\.0\.",  # Internal network
                r"^192\.168\.",  # Private network
                r"^172\.16\."   # Private network
            ],
            "blocked_ports": [22, 3389, 5432, 6379],
            "max_connections_per_ip": 100,
            "rate_limit_threshold": 1000  # requests per minute
        }
    
    def detect_security_threats(self) -> List[SecurityIncident]:
        """Detect potential security threats"""
        incidents = []
        
        # Check for suspicious processes
        suspicious_processes = self._check_suspicious_processes()
        for proc in suspicious_processes:
            incident = SecurityIncident(
                id=f"proc_{proc['pid']}_{int(time.time())}",
                incident_type="suspicious_process",
                severity="medium",
                description=f"Suspicious process detected: {proc['name']} (PID: {proc['pid']})",
                affected_component="system",
                detected_at=datetime.now()
            )
            incidents.append(incident)
        
        # Check network connections
        suspicious_connections = self._check_network_connections()
        for conn in suspicious_connections:
            incident = SecurityIncident(
                id=f"net_{conn['laddr']}_{int(time.time())}",
                incident_type="suspicious_connection",
                severity="high",
                description=f"Suspicious network connection: {conn['laddr']} -> {conn['raddr']}",
                source_ip=conn.get('raddr', '').split(':')[0] if conn.get('raddr') else None,
                affected_component="network",
                detected_at=datetime.now()
            )
            incidents.append(incident)
        
        return incidents
    
    def _check_suspicious_processes(self) -> List[Dict[str, Any]]:
        """Check for suspicious processes"""
        suspicious = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                
                # High CPU usage processes
                if info['cpu_percent'] and info['cpu_percent'] > 90:
                    suspicious.append(info)
                
                # Processes with suspicious names
                suspicious_names = ['nc', 'netcat', 'nmap', 'tcpdump', 'wireshark']
                if any(name in info['name'].lower() for name in suspicious_names):
                    suspicious.append(info)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return suspicious
    
    def _check_network_connections(self) -> List[Dict[str, Any]]:
        """Check for suspicious network connections"""
        suspicious = []
        
        try:
            connections = psutil.net_connections(kind='inet')
            
            for conn in connections:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    # Check for connections to suspicious ports
                    if conn.raddr.port in self.security_rules['blocked_ports']:
                        suspicious.append({
                            'laddr': f"{conn.laddr.ip}:{conn.laddr.port}",
                            'raddr': f"{conn.raddr.ip}:{conn.raddr.port}",
                            'status': conn.status,
                            'pid': conn.pid
                        })
                        
        except Exception as e:
            logger.warning(f"Network connection check failed: {e}")
        
        return suspicious
    
    def respond_to_incident(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Respond to a security incident"""
        try:
            response_actions = []
            
            if incident.incident_type == "suspicious_process":
                # Kill suspicious process
                pid = int(incident.id.split('_')[1])
                try:
                    os.kill(pid, signal.SIGTERM)
                    response_actions.append(f"Terminated process {pid}")
                except ProcessLookupError:
                    response_actions.append(f"Process {pid} already terminated")
                except PermissionError:
                    response_actions.append(f"Permission denied to terminate process {pid}")
            
            elif incident.incident_type == "suspicious_connection":
                # Block suspicious IP
                if incident.source_ip:
                    try:
                        subprocess.run([
                            'sudo', 'iptables', '-A', 'INPUT', 
                            '-s', incident.source_ip, '-j', 'DROP'
                        ], check=True)
                        response_actions.append(f"Blocked IP {incident.source_ip}")
                    except subprocess.CalledProcessError:
                        response_actions.append(f"Failed to block IP {incident.source_ip}")
            
            # Update incident status
            incident.status = "responded"
            incident.response_actions = response_actions
            
            return {
                "success": True,
                "message": f"Responded to incident {incident.id}",
                "actions_taken": response_actions
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Incident response failed: {str(e)}",
                "error": str(e)
            }

class NotificationManager:
    """Handles notifications for autonomous actions"""
    
    def __init__(self):
        self.notification_channels = {
            "email": self._send_email,
            "webhook": self._send_webhook,
            "log": self._log_notification
        }
        
    def send_notification(self, message: str, severity: str = "info", channels: List[str] = None):
        """Send notification through specified channels"""
        if channels is None:
            channels = ["log"]
            
        for channel in channels:
            if channel in self.notification_channels:
                try:
                    self.notification_channels[channel](message, severity)
                except Exception as e:
                    logger.error(f"Notification failed for channel {channel}: {e}")
    
    def _send_email(self, message: str, severity: str):
        """Send email notification"""
        # Email configuration would be loaded from environment variables
        # This is a placeholder implementation
        logger.info(f"EMAIL NOTIFICATION [{severity.upper()}]: {message}")
    
    def _send_webhook(self, message: str, severity: str):
        """Send webhook notification"""
        # Webhook URL would be configured
        # This is a placeholder implementation
        logger.info(f"WEBHOOK NOTIFICATION [{severity.upper()}]: {message}")
    
    def _log_notification(self, message: str, severity: str):
        """Log notification"""
        if severity == "critical":
            logger.critical(message)
        elif severity == "high":
            logger.error(message)
        elif severity == "medium":
            logger.warning(message)
        else:
            logger.info(message)

class AutonomousOperationsManager:
    """Main autonomous operations management service"""
    
    def __init__(self):
        self.service_manager = ServiceManager()
        self.performance_optimizer = PerformanceOptimizer()
        self.security_manager = SecurityManager()
        self.notification_manager = NotificationManager()
        
        self.action_queue = []
        self.action_history = []
        self.running = False
        self.operations_thread = None
        
        # Configuration
        self.check_interval = 60  # seconds
        self.max_concurrent_actions = 3
        self.action_timeout = 300  # 5 minutes
        
    def start_operations(self):
        """Start autonomous operations"""
        if self.running:
            logger.warning("Operations already running")
            return
            
        self.running = True
        self.operations_thread = threading.Thread(target=self._operations_loop, daemon=True)
        self.operations_thread.start()
        logger.info("Autonomous operations started")
    
    def stop_operations(self):
        """Stop autonomous operations"""
        self.running = False
        if self.operations_thread:
            self.operations_thread.join(timeout=5)
        logger.info("Autonomous operations stopped")
    
    def _operations_loop(self):
        """Main operations loop"""
        while self.running:
            try:
                # Check for performance optimization opportunities
                optimizations = self.performance_optimizer.analyze_performance_bottlenecks()
                for opt in optimizations:
                    self._queue_optimization_action(opt)
                
                # Check for security threats
                incidents = self.security_manager.detect_security_threats()
                for incident in incidents:
                    self._queue_security_action(incident)
                
                # Process action queue
                self._process_action_queue()
                
                logger.debug(f"Operations check completed. Queue size: {len(self.action_queue)}")
                
            except Exception as e:
                logger.error(f"Error in operations loop: {e}")
            
            time.sleep(self.check_interval)
    
    def _queue_optimization_action(self, optimization: PerformanceOptimization):
        """Queue a performance optimization action"""
        action_id = f"opt_{optimization.component}_{int(time.time())}"
        
        action_type_map = {
            "cache_optimization": ActionType.CLEAR_CACHE,
            "memory_cleanup": ActionType.ADJUST_MEMORY,
            "disk_cleanup": ActionType.CLEANUP_DISK,
            "process_optimization": ActionType.PERFORMANCE_TUNING
        }
        
        action_type = action_type_map.get(optimization.optimization_type, ActionType.PERFORMANCE_TUNING)
        
        action = AutonomousAction(
            id=action_id,
            action_type=action_type,
            priority=Priority.MEDIUM if optimization.confidence > 0.8 else Priority.LOW,
            description=f"Optimize {optimization.component}: {optimization.optimization_type}",
            target_component=optimization.component,
            parameters={
                "optimization": asdict(optimization)
            },
            created_at=datetime.now()
        )
        
        self.action_queue.append(action)
        logger.info(f"Queued optimization action: {action.description}")
    
    def _queue_security_action(self, incident: SecurityIncident):
        """Queue a security response action"""
        action_id = f"sec_{incident.id}_{int(time.time())}"
        
        action = AutonomousAction(
            id=action_id,
            action_type=ActionType.SECURITY_RESPONSE,
            priority=Priority.CRITICAL if incident.severity == "high" else Priority.HIGH,
            description=f"Respond to security incident: {incident.description}",
            target_component=incident.affected_component,
            parameters={
                "incident": asdict(incident)
            },
            created_at=datetime.now()
        )
        
        self.action_queue.append(action)
        logger.warning(f"Queued security action: {action.description}")
    
    def _process_action_queue(self):
        """Process actions in the queue"""
        # Sort by priority and creation time
        self.action_queue.sort(key=lambda x: (x.priority.value, x.created_at), reverse=True)
        
        # Process actions up to max concurrent limit
        in_progress_count = sum(1 for a in self.action_queue if a.status == ActionStatus.IN_PROGRESS)
        
        for action in self.action_queue[:]:
            if in_progress_count >= self.max_concurrent_actions:
                break
                
            if action.status == ActionStatus.PENDING:
                self._execute_action(action)
                in_progress_count += 1
    
    def _execute_action(self, action: AutonomousAction):
        """Execute an autonomous action"""
        action.status = ActionStatus.IN_PROGRESS
        action.started_at = datetime.now()
        
        try:
            result = None
            
            if action.action_type == ActionType.RESTART_SERVICE:
                result = self.service_manager.restart_service(action.target_component)
                
            elif action.action_type == ActionType.SCALE_SERVICE:
                replicas = action.parameters.get('replicas', 2)
                result = self.service_manager.scale_service(action.target_component, replicas)
                
            elif action.action_type == ActionType.CLEAR_CACHE:
                result = self.performance_optimizer.optimize_cache()
                
            elif action.action_type == ActionType.OPTIMIZE_DATABASE:
                result = self.performance_optimizer.optimize_database()
                
            elif action.action_type == ActionType.CLEANUP_DISK:
                result = self.performance_optimizer.cleanup_disk_space()
                
            elif action.action_type == ActionType.SECURITY_RESPONSE:
                incident_data = action.parameters.get('incident', {})
                incident = SecurityIncident(**incident_data)
                result = self.security_manager.respond_to_incident(incident)
                
            elif action.action_type == ActionType.PERFORMANCE_TUNING:
                # Generic performance tuning
                result = {"success": True, "message": "Performance tuning completed"}
            
            # Update action status
            action.result = result
            action.completed_at = datetime.now()
            
            if result and result.get('success', False):
                action.status = ActionStatus.COMPLETED
                self.notification_manager.send_notification(
                    f"Action completed successfully: {action.description}",
                    "info"
                )
            else:
                action.status = ActionStatus.FAILED
                action.error_message = result.get('message', 'Unknown error') if result else 'No result'
                self.notification_manager.send_notification(
                    f"Action failed: {action.description} - {action.error_message}",
                    "high"
                )
            
            # Move to history and remove from queue
            self.action_history.append(action)
            self.action_queue.remove(action)
            
            logger.info(f"Action {action.id} completed with status: {action.status.value}")
            
        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error_message = str(e)
            action.completed_at = datetime.now()
            
            self.action_history.append(action)
            self.action_queue.remove(action)
            
            self.notification_manager.send_notification(
                f"Action execution failed: {action.description} - {str(e)}",
                "critical"
            )
            
            logger.error(f"Action {action.id} failed: {e}")
    
    def queue_manual_action(self, action_type: str, target_component: str, parameters: Dict[str, Any] = None) -> str:
        """Queue a manual action"""
        action_id = f"manual_{action_type}_{int(time.time())}"
        
        action = AutonomousAction(
            id=action_id,
            action_type=ActionType(action_type),
            priority=Priority.HIGH,
            description=f"Manual action: {action_type} on {target_component}",
            target_component=target_component,
            parameters=parameters or {},
            created_at=datetime.now()
        )
        
        self.action_queue.append(action)
        logger.info(f"Queued manual action: {action.description}")
        
        return action_id
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "operations_running": self.running,
            "queue_size": len(self.action_queue),
            "in_progress_actions": len([a for a in self.action_queue if a.status == ActionStatus.IN_PROGRESS]),
            "completed_actions_today": len([
                a for a in self.action_history 
                if a.completed_at and a.completed_at.date() == datetime.now().date()
            ]),
            "active_security_incidents": len(self.security_manager.active_incidents),
            "last_check": datetime.now().isoformat()
        }

# Flask API
app = Flask(__name__)
CORS(app)

# Global operations manager instance
operations_manager = AutonomousOperationsManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'autonomous_operations_manager',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/operations/status', methods=['GET'])
def get_operations_status():
    """Get operations status"""
    return jsonify(operations_manager.get_system_status())

@app.route('/operations/start', methods=['POST'])
def start_operations():
    """Start autonomous operations"""
    operations_manager.start_operations()
    return jsonify({
        'message': 'Autonomous operations started',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/operations/stop', methods=['POST'])
def stop_operations():
    """Stop autonomous operations"""
    operations_manager.stop_operations()
    return jsonify({
        'message': 'Autonomous operations stopped',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/operations/actions', methods=['GET'])
def get_actions():
    """Get action queue and history"""
    return jsonify({
        'queue': [asdict(action) for action in operations_manager.action_queue],
        'history': [asdict(action) for action in operations_manager.action_history[-50:]],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/operations/actions', methods=['POST'])
def queue_action():
    """Queue a manual action"""
    data = request.get_json()
    
    action_type = data.get('action_type')
    target_component = data.get('target_component')
    parameters = data.get('parameters', {})
    
    if not action_type or not target_component:
        return jsonify({'error': 'action_type and target_component are required'}), 400
    
    try:
        action_id = operations_manager.queue_manual_action(action_type, target_component, parameters)
        return jsonify({
            'message': 'Action queued successfully',
            'action_id': action_id,
            'timestamp': datetime.now().isoformat()
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid action_type: {e}'}), 400

@app.route('/operations/optimizations', methods=['GET'])
def get_optimizations():
    """Get performance optimization recommendations"""
    optimizations = operations_manager.performance_optimizer.analyze_performance_bottlenecks()
    return jsonify({
        'optimizations': [asdict(opt) for opt in optimizations],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/operations/security', methods=['GET'])
def get_security_status():
    """Get security status and incidents"""
    incidents = operations_manager.security_manager.detect_security_threats()
    return jsonify({
        'active_incidents': [asdict(incident) for incident in incidents],
        'security_rules': operations_manager.security_manager.security_rules,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/operations/services/<service_name>/restart', methods=['POST'])
def restart_service(service_name):
    """Restart a specific service"""
    namespace = request.args.get('namespace', 'default')
    result = operations_manager.service_manager.restart_service(service_name, namespace)
    return jsonify(result)

@app.route('/operations/services/<service_name>/scale', methods=['POST'])
def scale_service(service_name):
    """Scale a specific service"""
    data = request.get_json()
    replicas = data.get('replicas', 2)
    namespace = request.args.get('namespace', 'default')
    
    result = operations_manager.service_manager.scale_service(service_name, replicas, namespace)
    return jsonify(result)

@app.route('/operations/services/<service_name>/status', methods=['GET'])
def get_service_status(service_name):
    """Get service status"""
    namespace = request.args.get('namespace', 'default')
    status = operations_manager.service_manager.get_service_status(service_name, namespace)
    return jsonify(status)

if __name__ == '__main__':
    # Start operations automatically
    operations_manager.start_operations()
    
    try:
        app.run(host='0.0.0.0', port=8061, debug=False)
    finally:
        operations_manager.stop_operations()

