#!/usr/bin/env python3
"""
Nexus Architect - WS4 Phase 5: System Health Monitor
Advanced self-monitoring with predictive analytics and behavioral analysis
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import psutil
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class HealthMetric:
    """Individual health metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    tags: Dict[str, str] = None

@dataclass
class SystemAlert:
    """System alert data structure"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    component: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class PredictionResult:
    """Prediction analysis result"""
    metric_name: str
    current_value: float
    predicted_value: float
    confidence: float
    time_horizon: int  # minutes
    trend: str  # "increasing", "decreasing", "stable"
    risk_level: HealthStatus

class HealthMetricsCollector:
    """Collects comprehensive system health metrics"""
    
    def __init__(self):
        self.metrics_history = {}
        self.collection_interval = 30  # seconds
        self.history_retention = 24 * 60 * 60  # 24 hours in seconds
        
    def collect_system_metrics(self) -> List[HealthMetric]:
        """Collect system-level performance metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # CPU Metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        
        metrics.extend([
            HealthMetric(
                name="cpu_usage_percent",
                value=cpu_percent,
                unit="percent",
                timestamp=timestamp,
                status=self._get_status(cpu_percent, 70, 90),
                threshold_warning=70,
                threshold_critical=90,
                tags={"component": "cpu"}
            ),
            HealthMetric(
                name="cpu_load_average",
                value=load_avg,
                unit="load",
                timestamp=timestamp,
                status=self._get_status(load_avg, cpu_count * 0.7, cpu_count * 0.9),
                threshold_warning=cpu_count * 0.7,
                threshold_critical=cpu_count * 0.9,
                tags={"component": "cpu"}
            )
        ])
        
        # Memory Metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics.extend([
            HealthMetric(
                name="memory_usage_percent",
                value=memory.percent,
                unit="percent",
                timestamp=timestamp,
                status=self._get_status(memory.percent, 80, 95),
                threshold_warning=80,
                threshold_critical=95,
                tags={"component": "memory"}
            ),
            HealthMetric(
                name="memory_available_gb",
                value=memory.available / (1024**3),
                unit="GB",
                timestamp=timestamp,
                status=self._get_status(memory.available / (1024**3), 2, 0.5, reverse=True),
                threshold_warning=2,
                threshold_critical=0.5,
                tags={"component": "memory"}
            ),
            HealthMetric(
                name="swap_usage_percent",
                value=swap.percent,
                unit="percent",
                timestamp=timestamp,
                status=self._get_status(swap.percent, 50, 80),
                threshold_warning=50,
                threshold_critical=80,
                tags={"component": "memory"}
            )
        ])
        
        # Disk Metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics.extend([
            HealthMetric(
                name="disk_usage_percent",
                value=(disk.used / disk.total) * 100,
                unit="percent",
                timestamp=timestamp,
                status=self._get_status((disk.used / disk.total) * 100, 80, 95),
                threshold_warning=80,
                threshold_critical=95,
                tags={"component": "disk", "mount": "/"}
            ),
            HealthMetric(
                name="disk_free_gb",
                value=disk.free / (1024**3),
                unit="GB",
                timestamp=timestamp,
                status=self._get_status(disk.free / (1024**3), 10, 2, reverse=True),
                threshold_warning=10,
                threshold_critical=2,
                tags={"component": "disk", "mount": "/"}
            )
        ])
        
        if disk_io:
            metrics.extend([
                HealthMetric(
                    name="disk_read_mb_per_sec",
                    value=disk_io.read_bytes / (1024**2),
                    unit="MB/s",
                    timestamp=timestamp,
                    status=HealthStatus.HEALTHY,
                    threshold_warning=100,
                    threshold_critical=500,
                    tags={"component": "disk", "type": "io"}
                ),
                HealthMetric(
                    name="disk_write_mb_per_sec",
                    value=disk_io.write_bytes / (1024**2),
                    unit="MB/s",
                    timestamp=timestamp,
                    status=HealthStatus.HEALTHY,
                    threshold_warning=100,
                    threshold_critical=500,
                    tags={"component": "disk", "type": "io"}
                )
            ])
        
        # Network Metrics
        network = psutil.net_io_counters()
        if network:
            metrics.extend([
                HealthMetric(
                    name="network_bytes_sent_mb",
                    value=network.bytes_sent / (1024**2),
                    unit="MB",
                    timestamp=timestamp,
                    status=HealthStatus.HEALTHY,
                    threshold_warning=1000,
                    threshold_critical=5000,
                    tags={"component": "network", "direction": "sent"}
                ),
                HealthMetric(
                    name="network_bytes_recv_mb",
                    value=network.bytes_recv / (1024**2),
                    unit="MB",
                    timestamp=timestamp,
                    status=HealthStatus.HEALTHY,
                    threshold_warning=1000,
                    threshold_critical=5000,
                    tags={"component": "network", "direction": "received"}
                )
            ])
        
        return metrics
    
    def collect_application_metrics(self) -> List[HealthMetric]:
        """Collect application-specific health metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # Process Metrics
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower() or 'nexus' in proc.info['name'].lower():
                    processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if processes:
            total_cpu = sum(p['cpu_percent'] or 0 for p in processes)
            total_memory = sum(p['memory_percent'] or 0 for p in processes)
            
            metrics.extend([
                HealthMetric(
                    name="application_cpu_usage_percent",
                    value=total_cpu,
                    unit="percent",
                    timestamp=timestamp,
                    status=self._get_status(total_cpu, 50, 80),
                    threshold_warning=50,
                    threshold_critical=80,
                    tags={"component": "application"}
                ),
                HealthMetric(
                    name="application_memory_usage_percent",
                    value=total_memory,
                    unit="percent",
                    timestamp=timestamp,
                    status=self._get_status(total_memory, 60, 85),
                    threshold_warning=60,
                    threshold_critical=85,
                    tags={"component": "application"}
                ),
                HealthMetric(
                    name="application_process_count",
                    value=len(processes),
                    unit="count",
                    timestamp=timestamp,
                    status=self._get_status(len(processes), 20, 50),
                    threshold_warning=20,
                    threshold_critical=50,
                    tags={"component": "application"}
                )
            ])
        
        return metrics
    
    def collect_service_metrics(self) -> List[HealthMetric]:
        """Collect service health metrics from other components"""
        metrics = []
        timestamp = datetime.now()
        
        # Check Redis connectivity
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_info = redis_client.info()
            redis_client.ping()
            
            metrics.extend([
                HealthMetric(
                    name="redis_connected_clients",
                    value=redis_info.get('connected_clients', 0),
                    unit="count",
                    timestamp=timestamp,
                    status=self._get_status(redis_info.get('connected_clients', 0), 100, 500),
                    threshold_warning=100,
                    threshold_critical=500,
                    tags={"component": "redis", "service": "cache"}
                ),
                HealthMetric(
                    name="redis_memory_usage_mb",
                    value=redis_info.get('used_memory', 0) / (1024**2),
                    unit="MB",
                    timestamp=timestamp,
                    status=self._get_status(redis_info.get('used_memory', 0) / (1024**2), 500, 1000),
                    threshold_warning=500,
                    threshold_critical=1000,
                    tags={"component": "redis", "service": "cache"}
                )
            ])
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            metrics.append(
                HealthMetric(
                    name="redis_connectivity",
                    value=0,
                    unit="boolean",
                    timestamp=timestamp,
                    status=HealthStatus.CRITICAL,
                    threshold_warning=1,
                    threshold_critical=1,
                    tags={"component": "redis", "service": "cache", "error": str(e)}
                )
            )
        
        # Check PostgreSQL connectivity
        try:
            conn = psycopg2.connect(
                host="localhost",
                database="nexus_architect",
                user="nexus_user",
                password="nexus_password"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pg_stat_activity;")
            active_connections = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            metrics.append(
                HealthMetric(
                    name="postgresql_active_connections",
                    value=active_connections,
                    unit="count",
                    timestamp=timestamp,
                    status=self._get_status(active_connections, 50, 100),
                    threshold_warning=50,
                    threshold_critical=100,
                    tags={"component": "postgresql", "service": "database"}
                )
            )
        except Exception as e:
            logger.warning(f"PostgreSQL health check failed: {e}")
            metrics.append(
                HealthMetric(
                    name="postgresql_connectivity",
                    value=0,
                    unit="boolean",
                    timestamp=timestamp,
                    status=HealthStatus.CRITICAL,
                    threshold_warning=1,
                    threshold_critical=1,
                    tags={"component": "postgresql", "service": "database", "error": str(e)}
                )
            )
        
        return metrics
    
    def _get_status(self, value: float, warning_threshold: float, critical_threshold: float, reverse: bool = False) -> HealthStatus:
        """Determine health status based on thresholds"""
        if reverse:
            if value <= critical_threshold:
                return HealthStatus.CRITICAL
            elif value <= warning_threshold:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
        else:
            if value >= critical_threshold:
                return HealthStatus.CRITICAL
            elif value >= warning_threshold:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY

class PredictiveAnalytics:
    """Predictive analytics for system health trends"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.prediction_horizon = 60  # minutes
        self.min_data_points = 10
        
    def analyze_trends(self, metrics_history: Dict[str, List[HealthMetric]]) -> List[PredictionResult]:
        """Analyze trends and predict future values"""
        predictions = []
        
        for metric_name, history in metrics_history.items():
            if len(history) < self.min_data_points:
                continue
                
            try:
                prediction = self._predict_metric_trend(metric_name, history)
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Prediction failed for {metric_name}: {e}")
                
        return predictions
    
    def _predict_metric_trend(self, metric_name: str, history: List[HealthMetric]) -> Optional[PredictionResult]:
        """Predict trend for a specific metric"""
        if len(history) < self.min_data_points:
            return None
            
        # Extract values and timestamps
        values = [m.value for m in history[-50:]]  # Use last 50 data points
        timestamps = [m.timestamp for m in history[-50:]]
        
        # Convert timestamps to minutes from start
        start_time = timestamps[0]
        time_points = [(ts - start_time).total_seconds() / 60 for ts in timestamps]
        
        # Simple linear regression for trend analysis
        if len(values) >= 3:
            # Calculate trend using linear regression
            x = np.array(time_points).reshape(-1, 1)
            y = np.array(values)
            
            # Simple linear regression
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x.flatten() * y)
            sum_x2 = np.sum(x**2)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict future value
            future_time = time_points[-1] + self.prediction_horizon
            predicted_value = slope * future_time + intercept
            
            # Calculate confidence based on R-squared
            y_pred = slope * x.flatten() + intercept
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            confidence = max(0, min(1, r_squared))
            
            # Determine trend direction
            if abs(slope) < 0.01:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
            
            # Assess risk level based on prediction and current thresholds
            current_value = values[-1]
            latest_metric = history[-1]
            
            if predicted_value >= latest_metric.threshold_critical:
                risk_level = HealthStatus.CRITICAL
            elif predicted_value >= latest_metric.threshold_warning:
                risk_level = HealthStatus.WARNING
            else:
                risk_level = HealthStatus.HEALTHY
            
            return PredictionResult(
                metric_name=metric_name,
                current_value=current_value,
                predicted_value=predicted_value,
                confidence=confidence,
                time_horizon=self.prediction_horizon,
                trend=trend,
                risk_level=risk_level
            )
        
        return None

class AnomalyDetector:
    """Behavioral analysis and anomaly detection"""
    
    def __init__(self):
        self.models = {}
        self.contamination = 0.1  # Expected proportion of anomalies
        self.min_training_samples = 20
        
    def detect_anomalies(self, metrics_history: Dict[str, List[HealthMetric]]) -> List[Dict[str, Any]]:
        """Detect anomalies in system behavior"""
        anomalies = []
        
        for metric_name, history in metrics_history.items():
            if len(history) < self.min_training_samples:
                continue
                
            try:
                metric_anomalies = self._detect_metric_anomalies(metric_name, history)
                anomalies.extend(metric_anomalies)
            except Exception as e:
                logger.warning(f"Anomaly detection failed for {metric_name}: {e}")
                
        return anomalies
    
    def _detect_metric_anomalies(self, metric_name: str, history: List[HealthMetric]) -> List[Dict[str, Any]]:
        """Detect anomalies for a specific metric"""
        anomalies = []
        
        # Prepare data
        values = np.array([m.value for m in history]).reshape(-1, 1)
        timestamps = [m.timestamp for m in history]
        
        # Train or update model
        if metric_name not in self.models:
            self.models[metric_name] = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            
        model = self.models[metric_name]
        
        # Fit model on historical data (excluding recent points for anomaly detection)
        training_data = values[:-5] if len(values) > 10 else values[:-1]
        if len(training_data) >= self.min_training_samples:
            model.fit(training_data)
            
            # Detect anomalies in recent data
            recent_data = values[-5:]
            recent_timestamps = timestamps[-5:]
            
            predictions = model.predict(recent_data)
            scores = model.decision_function(recent_data)
            
            for i, (pred, score, timestamp, value) in enumerate(zip(predictions, scores, recent_timestamps, recent_data.flatten())):
                if pred == -1:  # Anomaly detected
                    anomalies.append({
                        'metric_name': metric_name,
                        'timestamp': timestamp,
                        'value': value,
                        'anomaly_score': score,
                        'severity': 'high' if score < -0.5 else 'medium',
                        'description': f"Anomalous {metric_name} value: {value:.2f}"
                    })
        
        return anomalies

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = []
        self.notification_channels = []
        
    def process_metrics(self, metrics: List[HealthMetric]) -> List[SystemAlert]:
        """Process metrics and generate alerts"""
        new_alerts = []
        
        for metric in metrics:
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert_id = f"{metric.name}_{metric.status.value}"
                
                if alert_id not in self.active_alerts:
                    alert = SystemAlert(
                        id=alert_id,
                        title=f"{metric.name.replace('_', ' ').title()} {metric.status.value.title()}",
                        description=f"{metric.name} is {metric.value:.2f} {metric.unit}, exceeding {metric.status.value} threshold",
                        severity=AlertSeverity.HIGH if metric.status == HealthStatus.CRITICAL else AlertSeverity.MEDIUM,
                        component=metric.tags.get('component', 'unknown') if metric.tags else 'unknown',
                        timestamp=metric.timestamp,
                        metadata={
                            'metric_name': metric.name,
                            'current_value': metric.value,
                            'unit': metric.unit,
                            'threshold_warning': metric.threshold_warning,
                            'threshold_critical': metric.threshold_critical,
                            'tags': metric.tags
                        }
                    )
                    
                    self.active_alerts[alert_id] = alert
                    new_alerts.append(alert)
                    self.alert_history.append(alert)
                    
                    logger.warning(f"Alert generated: {alert.title}")
            else:
                # Check if we need to resolve any alerts
                alert_id = f"{metric.name}_warning"
                if alert_id in self.active_alerts:
                    self._resolve_alert(alert_id)
                    
                alert_id = f"{metric.name}_critical"
                if alert_id in self.active_alerts:
                    self._resolve_alert(alert_id)
        
        return new_alerts
    
    def _resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert.title}")

class SystemHealthMonitor:
    """Main system health monitoring service"""
    
    def __init__(self):
        self.metrics_collector = HealthMetricsCollector()
        self.predictive_analytics = PredictiveAnalytics()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        
        self.metrics_history = {}
        self.running = False
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()
        
        # Configuration
        self.collection_interval = 30  # seconds
        self.history_retention_hours = 24
        self.max_history_points = 2880  # 24 hours at 30-second intervals
        
    def start_monitoring(self):
        """Start the health monitoring service"""
        if self.running:
            logger.warning("Monitoring already running")
            return
            
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring service"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect all metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                app_metrics = self.metrics_collector.collect_application_metrics()
                service_metrics = self.metrics_collector.collect_service_metrics()
                
                all_metrics = system_metrics + app_metrics + service_metrics
                
                # Update metrics history
                self._update_metrics_history(all_metrics)
                
                # Process alerts
                new_alerts = self.alert_manager.process_metrics(all_metrics)
                
                # Store metrics in queue for API access
                self.metrics_queue.put({
                    'timestamp': datetime.now(),
                    'metrics': all_metrics,
                    'alerts': new_alerts
                })
                
                # Clean up old queue items
                while self.metrics_queue.qsize() > 100:
                    try:
                        self.metrics_queue.get_nowait()
                    except queue.Empty:
                        break
                
                logger.debug(f"Collected {len(all_metrics)} metrics, generated {len(new_alerts)} alerts")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.collection_interval)
    
    def _update_metrics_history(self, metrics: List[HealthMetric]):
        """Update metrics history with retention policy"""
        cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
        
        for metric in metrics:
            metric_name = metric.name
            
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            
            # Add new metric
            self.metrics_history[metric_name].append(metric)
            
            # Remove old metrics
            self.metrics_history[metric_name] = [
                m for m in self.metrics_history[metric_name]
                if m.timestamp > cutoff_time
            ]
            
            # Limit history size
            if len(self.metrics_history[metric_name]) > self.max_history_points:
                self.metrics_history[metric_name] = self.metrics_history[metric_name][-self.max_history_points:]
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current overall system health status"""
        if not self.metrics_queue.empty():
            latest_data = None
            while not self.metrics_queue.empty():
                try:
                    latest_data = self.metrics_queue.get_nowait()
                except queue.Empty:
                    break
            
            if latest_data:
                metrics = latest_data['metrics']
                alerts = latest_data['alerts']
                
                # Calculate overall health status
                status_counts = {status: 0 for status in HealthStatus}
                for metric in metrics:
                    status_counts[metric.status] += 1
                
                # Determine overall status
                if status_counts[HealthStatus.CRITICAL] > 0:
                    overall_status = HealthStatus.CRITICAL
                elif status_counts[HealthStatus.WARNING] > 0:
                    overall_status = HealthStatus.WARNING
                else:
                    overall_status = HealthStatus.HEALTHY
                
                return {
                    'overall_status': overall_status.value,
                    'timestamp': latest_data['timestamp'].isoformat(),
                    'metrics_count': len(metrics),
                    'active_alerts': len(self.alert_manager.active_alerts),
                    'status_breakdown': {status.value: count for status, count in status_counts.items()},
                    'recent_metrics': [asdict(m) for m in metrics],
                    'recent_alerts': [asdict(a) for a in alerts]
                }
        
        return {
            'overall_status': HealthStatus.UNKNOWN.value,
            'timestamp': datetime.now().isoformat(),
            'message': 'No recent metrics available'
        }
    
    def get_predictions(self) -> List[Dict[str, Any]]:
        """Get predictive analytics results"""
        predictions = self.predictive_analytics.analyze_trends(self.metrics_history)
        return [asdict(p) for p in predictions]
    
    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get anomaly detection results"""
        return self.anomaly_detector.detect_anomalies(self.metrics_history)
    
    def get_metrics_history(self, metric_name: str = None, hours: int = 1) -> Dict[str, Any]:
        """Get historical metrics data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if metric_name:
            if metric_name in self.metrics_history:
                filtered_metrics = [
                    m for m in self.metrics_history[metric_name]
                    if m.timestamp > cutoff_time
                ]
                return {
                    metric_name: [asdict(m) for m in filtered_metrics]
                }
            else:
                return {metric_name: []}
        else:
            result = {}
            for name, history in self.metrics_history.items():
                filtered_metrics = [
                    m for m in history
                    if m.timestamp > cutoff_time
                ]
                result[name] = [asdict(m) for m in filtered_metrics]
            return result

# Flask API
app = Flask(__name__)
CORS(app)

# Global monitor instance
monitor = SystemHealthMonitor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'system_health_monitor',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/health/status', methods=['GET'])
def get_health_status():
    """Get current system health status"""
    return jsonify(monitor.get_current_health_status())

@app.route('/health/predictions', methods=['GET'])
def get_predictions():
    """Get predictive analytics results"""
    return jsonify({
        'predictions': monitor.get_predictions(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health/anomalies', methods=['GET'])
def get_anomalies():
    """Get anomaly detection results"""
    return jsonify({
        'anomalies': monitor.get_anomalies(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health/metrics', methods=['GET'])
def get_metrics():
    """Get metrics history"""
    metric_name = request.args.get('metric')
    hours = int(request.args.get('hours', 1))
    
    return jsonify({
        'metrics': monitor.get_metrics_history(metric_name, hours),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health/alerts', methods=['GET'])
def get_alerts():
    """Get active alerts"""
    return jsonify({
        'active_alerts': [asdict(alert) for alert in monitor.alert_manager.active_alerts.values()],
        'alert_history': [asdict(alert) for alert in monitor.alert_manager.alert_history[-50:]],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health/start', methods=['POST'])
def start_monitoring():
    """Start health monitoring"""
    monitor.start_monitoring()
    return jsonify({
        'message': 'Health monitoring started',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health/stop', methods=['POST'])
def stop_monitoring():
    """Stop health monitoring"""
    monitor.stop_monitoring()
    return jsonify({
        'message': 'Health monitoring stopped',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Start monitoring automatically
    monitor.start_monitoring()
    
    try:
        app.run(host='0.0.0.0', port=8060, debug=False)
    finally:
        monitor.stop_monitoring()

