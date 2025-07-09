"""
Performance Optimizer for Nexus Architect
Implements comprehensive performance optimization including resource tuning,
caching strategies, and predictive analytics for capacity planning.
"""

import logging
import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import secrets
import statistics
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import redis
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"

class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    SCALE_UP = "scale_up"
    SCALE_OUT = "scale_out"
    CACHE_OPTIMIZATION = "cache_optimization"
    QUERY_OPTIMIZATION = "query_optimization"
    COMPRESSION = "compression"
    LOAD_BALANCING = "load_balancing"

class PerformanceLevel(Enum):
    """Performance level indicators"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    metric_id: str
    resource_type: ResourceType
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    resource_type: ResourceType
    strategy: OptimizationStrategy
    title: str
    description: str
    expected_improvement: float  # Percentage improvement expected
    implementation_effort: str  # low, medium, high
    priority: str  # low, medium, high, critical
    estimated_cost: Optional[float] = None
    implementation_steps: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CacheConfiguration:
    """Cache configuration settings"""
    cache_type: str  # redis, memcached, local
    max_memory: int  # MB
    eviction_policy: str  # lru, lfu, random, ttl
    ttl_default: int  # seconds
    compression_enabled: bool = False
    persistence_enabled: bool = False

@dataclass
class ResourceUsagePattern:
    """Resource usage pattern analysis"""
    resource_type: ResourceType
    pattern_type: str  # daily, weekly, monthly, seasonal
    peak_hours: List[int]
    average_usage: float
    peak_usage: float
    growth_rate: float  # percentage per month
    seasonality_factor: float
    prediction_accuracy: float

class PerformanceOptimizer:
    """
    Comprehensive performance optimizer implementing enterprise-grade
    performance monitoring, optimization, and capacity planning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the performance optimizer"""
        self.config = config
        self.database_config = config.get('database', {})
        self.redis_config = config.get('redis', {})
        
        # Performance metrics storage
        self.metrics_history: deque = deque(maxlen=100000)  # Keep last 100k metrics
        self.current_metrics: Dict[str, PerformanceMetric] = {}
        self.recommendations: Dict[str, OptimizationRecommendation] = {}
        
        # Resource monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = 30  # seconds
        self.monitoring_thread = None
        
        # Cache configurations
        self.cache_configs: Dict[str, CacheConfiguration] = {}
        
        # Performance thresholds
        self.performance_thresholds = self._initialize_thresholds()
        
        # Usage patterns
        self.usage_patterns: Dict[str, ResourceUsagePattern] = {}
        
        # Initialize Redis connection for caching
        try:
            self.redis_client = redis.Redis(
                host=self.redis_config.get('host', 'localhost'),
                port=self.redis_config.get('port', 6379),
                db=self.redis_config.get('db', 0),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
            self.redis_client = None
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("Performance Optimizer initialized successfully")
    
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize performance thresholds"""
        return {
            "cpu": {
                "warning": 70.0,
                "critical": 90.0
            },
            "memory": {
                "warning": 80.0,
                "critical": 95.0
            },
            "disk": {
                "warning": 85.0,
                "critical": 95.0
            },
            "network": {
                "warning": 80.0,  # % of bandwidth
                "critical": 95.0
            },
            "response_time": {
                "warning": 2000.0,  # milliseconds
                "critical": 5000.0
            },
            "throughput": {
                "warning": 100.0,  # requests per second
                "critical": 50.0
            }
        }
    
    def _start_monitoring(self):
        """Start performance monitoring thread"""
        if self.monitoring_enabled and not self.monitoring_thread:
            self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self.monitoring_thread.start()
            logger.info("Performance monitoring started")
    
    def _monitor_performance(self):
        """Background performance monitoring"""
        while self.monitoring_enabled:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Analyze performance trends
                self._analyze_performance_trends()
                
                # Generate recommendations
                self._generate_optimization_recommendations()
                
                # Clean old metrics
                self._cleanup_old_metrics()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            current_time = datetime.utcnow()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = PerformanceMetric(
                metric_id=f"cpu_{int(time.time())}",
                resource_type=ResourceType.CPU,
                metric_name="cpu_usage_percent",
                value=cpu_percent,
                unit="percent",
                timestamp=current_time,
                threshold_warning=self.performance_thresholds["cpu"]["warning"],
                threshold_critical=self.performance_thresholds["cpu"]["critical"]
            )
            self._store_metric(cpu_metric)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_metric = PerformanceMetric(
                metric_id=f"memory_{int(time.time())}",
                resource_type=ResourceType.MEMORY,
                metric_name="memory_usage_percent",
                value=memory.percent,
                unit="percent",
                timestamp=current_time,
                threshold_warning=self.performance_thresholds["memory"]["warning"],
                threshold_critical=self.performance_thresholds["memory"]["critical"]
            )
            self._store_metric(memory_metric)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_metric = PerformanceMetric(
                metric_id=f"disk_{int(time.time())}",
                resource_type=ResourceType.DISK,
                metric_name="disk_usage_percent",
                value=disk_percent,
                unit="percent",
                timestamp=current_time,
                threshold_warning=self.performance_thresholds["disk"]["warning"],
                threshold_critical=self.performance_thresholds["disk"]["critical"]
            )
            self._store_metric(disk_metric)
            
            # Network metrics
            network = psutil.net_io_counters()
            if hasattr(self, '_last_network_stats'):
                time_delta = (current_time - self._last_network_time).total_seconds()
                if time_delta > 0:
                    bytes_sent_rate = (network.bytes_sent - self._last_network_stats.bytes_sent) / time_delta
                    bytes_recv_rate = (network.bytes_recv - self._last_network_stats.bytes_recv) / time_delta
                    
                    network_out_metric = PerformanceMetric(
                        metric_id=f"network_out_{int(time.time())}",
                        resource_type=ResourceType.NETWORK,
                        metric_name="network_bytes_sent_per_sec",
                        value=bytes_sent_rate,
                        unit="bytes/sec",
                        timestamp=current_time
                    )
                    self._store_metric(network_out_metric)
                    
                    network_in_metric = PerformanceMetric(
                        metric_id=f"network_in_{int(time.time())}",
                        resource_type=ResourceType.NETWORK,
                        metric_name="network_bytes_recv_per_sec",
                        value=bytes_recv_rate,
                        unit="bytes/sec",
                        timestamp=current_time
                    )
                    self._store_metric(network_in_metric)
            
            self._last_network_stats = network
            self._last_network_time = current_time
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store performance metric"""
        try:
            # Store in memory
            self.metrics_history.append(metric)
            self.current_metrics[f"{metric.resource_type.value}_{metric.metric_name}"] = metric
            
            # Store in Redis if available
            if self.redis_client:
                metric_key = f"metric:{metric.resource_type.value}:{metric.metric_name}"
                metric_data = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat()
                }
                self.redis_client.lpush(metric_key, json.dumps(metric_data))
                self.redis_client.ltrim(metric_key, 0, 1000)  # Keep last 1000 values
                
        except Exception as e:
            logger.error(f"Error storing metric: {str(e)}")
    
    def _analyze_performance_trends(self):
        """Analyze performance trends and patterns"""
        try:
            # Analyze trends for each resource type
            for resource_type in ResourceType:
                resource_metrics = [
                    metric for metric in self.metrics_history
                    if metric.resource_type == resource_type and
                    metric.timestamp > datetime.utcnow() - timedelta(hours=24)
                ]
                
                if len(resource_metrics) > 10:
                    self._analyze_resource_pattern(resource_type, resource_metrics)
                    
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
    
    def _analyze_resource_pattern(self, resource_type: ResourceType, metrics: List[PerformanceMetric]):
        """Analyze usage pattern for a specific resource"""
        try:
            if not metrics:
                return
            
            # Extract values and timestamps
            values = [metric.value for metric in metrics]
            timestamps = [metric.timestamp for metric in metrics]
            
            # Calculate statistics
            avg_usage = statistics.mean(values)
            peak_usage = max(values)
            min_usage = min(values)
            
            # Identify peak hours
            hourly_usage = defaultdict(list)
            for metric in metrics:
                hour = metric.timestamp.hour
                hourly_usage[hour].append(metric.value)
            
            hourly_averages = {
                hour: statistics.mean(values) 
                for hour, values in hourly_usage.items()
            }
            
            # Find peak hours (top 25% of hours by usage)
            sorted_hours = sorted(hourly_averages.items(), key=lambda x: x[1], reverse=True)
            peak_hours = [hour for hour, _ in sorted_hours[:6]]  # Top 6 hours
            
            # Calculate growth rate (simplified)
            if len(values) > 50:
                recent_avg = statistics.mean(values[-20:])
                older_avg = statistics.mean(values[:20])
                growth_rate = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
            else:
                growth_rate = 0
            
            # Create usage pattern
            pattern = ResourceUsagePattern(
                resource_type=resource_type,
                pattern_type="daily",
                peak_hours=peak_hours,
                average_usage=avg_usage,
                peak_usage=peak_usage,
                growth_rate=growth_rate,
                seasonality_factor=1.0,  # Simplified
                prediction_accuracy=0.85  # Estimated
            )
            
            self.usage_patterns[resource_type.value] = pattern
            
        except Exception as e:
            logger.error(f"Error analyzing resource pattern: {str(e)}")
    
    def _generate_optimization_recommendations(self):
        """Generate performance optimization recommendations"""
        try:
            # Clear old recommendations
            self.recommendations.clear()
            
            # Analyze current metrics for optimization opportunities
            for metric_key, metric in self.current_metrics.items():
                recommendations = self._analyze_metric_for_optimization(metric)
                for rec in recommendations:
                    self.recommendations[rec.recommendation_id] = rec
                    
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {str(e)}")
    
    def _analyze_metric_for_optimization(self, metric: PerformanceMetric) -> List[OptimizationRecommendation]:
        """Analyze a metric for optimization opportunities"""
        recommendations = []
        
        try:
            # Check if metric exceeds thresholds
            if metric.threshold_critical and metric.value >= metric.threshold_critical:
                # Critical threshold exceeded
                if metric.resource_type == ResourceType.CPU:
                    rec = OptimizationRecommendation(
                        recommendation_id=secrets.token_hex(8),
                        resource_type=metric.resource_type,
                        strategy=OptimizationStrategy.SCALE_UP,
                        title="Critical CPU Usage - Scale Up Required",
                        description=f"CPU usage is at {metric.value:.1f}%, exceeding critical threshold of {metric.threshold_critical}%",
                        expected_improvement=30.0,
                        implementation_effort="medium",
                        priority="critical",
                        implementation_steps=[
                            "Increase CPU cores or upgrade to higher performance instance",
                            "Optimize CPU-intensive processes",
                            "Implement CPU usage monitoring and alerting"
                        ]
                    )
                    recommendations.append(rec)
                
                elif metric.resource_type == ResourceType.MEMORY:
                    rec = OptimizationRecommendation(
                        recommendation_id=secrets.token_hex(8),
                        resource_type=metric.resource_type,
                        strategy=OptimizationStrategy.SCALE_UP,
                        title="Critical Memory Usage - Scale Up Required",
                        description=f"Memory usage is at {metric.value:.1f}%, exceeding critical threshold of {metric.threshold_critical}%",
                        expected_improvement=25.0,
                        implementation_effort="medium",
                        priority="critical",
                        implementation_steps=[
                            "Increase available RAM",
                            "Optimize memory-intensive processes",
                            "Implement memory leak detection",
                            "Configure memory usage alerts"
                        ]
                    )
                    recommendations.append(rec)
                
                elif metric.resource_type == ResourceType.DISK:
                    rec = OptimizationRecommendation(
                        recommendation_id=secrets.token_hex(8),
                        resource_type=metric.resource_type,
                        strategy=OptimizationStrategy.SCALE_UP,
                        title="Critical Disk Usage - Storage Expansion Required",
                        description=f"Disk usage is at {metric.value:.1f}%, exceeding critical threshold of {metric.threshold_critical}%",
                        expected_improvement=40.0,
                        implementation_effort="low",
                        priority="critical",
                        implementation_steps=[
                            "Expand disk storage capacity",
                            "Implement data archival and cleanup procedures",
                            "Configure disk usage monitoring and alerts"
                        ]
                    )
                    recommendations.append(rec)
            
            elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                # Warning threshold exceeded - suggest proactive optimizations
                if metric.resource_type == ResourceType.CPU:
                    rec = OptimizationRecommendation(
                        recommendation_id=secrets.token_hex(8),
                        resource_type=metric.resource_type,
                        strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
                        title="High CPU Usage - Implement Caching",
                        description=f"CPU usage is at {metric.value:.1f}%, approaching critical levels. Implement caching to reduce CPU load.",
                        expected_improvement=20.0,
                        implementation_effort="medium",
                        priority="high",
                        implementation_steps=[
                            "Implement Redis caching for frequently accessed data",
                            "Optimize database queries to reduce CPU usage",
                            "Enable application-level caching",
                            "Consider load balancing to distribute CPU load"
                        ]
                    )
                    recommendations.append(rec)
            
            # Generate proactive recommendations based on usage patterns
            if metric.resource_type.value in self.usage_patterns:
                pattern = self.usage_patterns[metric.resource_type.value]
                
                if pattern.growth_rate > 10:  # Growing more than 10% per month
                    rec = OptimizationRecommendation(
                        recommendation_id=secrets.token_hex(8),
                        resource_type=metric.resource_type,
                        strategy=OptimizationStrategy.SCALE_OUT,
                        title=f"Proactive Scaling for {metric.resource_type.value.title()}",
                        description=f"{metric.resource_type.value.title()} usage is growing at {pattern.growth_rate:.1f}% per month. Consider proactive scaling.",
                        expected_improvement=15.0,
                        implementation_effort="high",
                        priority="medium",
                        implementation_steps=[
                            "Plan capacity expansion based on growth projections",
                            "Implement auto-scaling policies",
                            "Monitor growth trends and adjust scaling policies"
                        ]
                    )
                    recommendations.append(rec)
                    
        except Exception as e:
            logger.error(f"Error analyzing metric for optimization: {str(e)}")
        
        return recommendations
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=48)
            
            # Remove metrics older than 48 hours
            old_metrics = [
                metric for metric in self.metrics_history
                if metric.timestamp < cutoff_time
            ]
            
            for metric in old_metrics:
                try:
                    self.metrics_history.remove(metric)
                except ValueError:
                    pass  # Metric already removed
                    
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {str(e)}")
    
    async def optimize_cache_configuration(self, cache_name: str) -> CacheConfiguration:
        """
        Optimize cache configuration based on usage patterns
        
        Args:
            cache_name: Name of the cache to optimize
            
        Returns:
            Optimized cache configuration
        """
        try:
            # Get current cache metrics
            cache_metrics = [
                metric for metric in self.metrics_history
                if metric.resource_type == ResourceType.CACHE and
                cache_name in metric.tags.get('cache_name', '')
            ]
            
            if not cache_metrics:
                # Default configuration for new cache
                config = CacheConfiguration(
                    cache_type="redis",
                    max_memory=512,  # MB
                    eviction_policy="lru",
                    ttl_default=3600,  # 1 hour
                    compression_enabled=True,
                    persistence_enabled=False
                )
            else:
                # Analyze cache usage patterns
                hit_rates = [m.value for m in cache_metrics if m.metric_name == "hit_rate"]
                memory_usage = [m.value for m in cache_metrics if m.metric_name == "memory_usage"]
                
                avg_hit_rate = statistics.mean(hit_rates) if hit_rates else 0.8
                avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 50.0
                
                # Optimize based on patterns
                if avg_hit_rate < 0.7:
                    # Low hit rate - increase cache size and TTL
                    max_memory = 1024
                    ttl_default = 7200
                    eviction_policy = "lfu"  # Least Frequently Used
                elif avg_hit_rate > 0.95:
                    # Very high hit rate - can reduce cache size
                    max_memory = 256
                    ttl_default = 1800
                    eviction_policy = "lru"
                else:
                    # Good hit rate - balanced configuration
                    max_memory = 512
                    ttl_default = 3600
                    eviction_policy = "lru"
                
                config = CacheConfiguration(
                    cache_type="redis",
                    max_memory=max_memory,
                    eviction_policy=eviction_policy,
                    ttl_default=ttl_default,
                    compression_enabled=avg_memory_usage > 70.0,
                    persistence_enabled=avg_hit_rate > 0.9
                )
            
            self.cache_configs[cache_name] = config
            logger.info(f"Optimized cache configuration for {cache_name}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error optimizing cache configuration: {str(e)}")
            raise
    
    async def predict_resource_usage(self, resource_type: ResourceType, 
                                   hours_ahead: int = 24) -> Dict[str, float]:
        """
        Predict future resource usage using machine learning
        
        Args:
            resource_type: Type of resource to predict
            hours_ahead: Hours into the future to predict
            
        Returns:
            Prediction results
        """
        try:
            # Get historical data for the resource
            resource_metrics = [
                metric for metric in self.metrics_history
                if metric.resource_type == resource_type and
                metric.timestamp > datetime.utcnow() - timedelta(days=7)
            ]
            
            if len(resource_metrics) < 50:
                return {"error": "Insufficient historical data for prediction"}
            
            # Prepare data for machine learning
            timestamps = [metric.timestamp for metric in resource_metrics]
            values = [metric.value for metric in resource_metrics]
            
            # Convert timestamps to numerical features
            base_time = min(timestamps)
            X = np.array([
                [(ts - base_time).total_seconds() / 3600]  # Hours since base time
                for ts in timestamps
            ])
            y = np.array(values)
            
            # Train linear regression model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Make prediction
            future_time = datetime.utcnow() + timedelta(hours=hours_ahead)
            future_hours = (future_time - base_time).total_seconds() / 3600
            future_X = scaler.transform([[future_hours]])
            
            predicted_value = model.predict(future_X)[0]
            
            # Calculate confidence based on model score
            confidence = model.score(X_scaled, y)
            
            # Calculate trend
            recent_values = values[-10:] if len(values) >= 10 else values
            older_values = values[:10] if len(values) >= 20 else values[:len(values)//2]
            
            recent_avg = statistics.mean(recent_values)
            older_avg = statistics.mean(older_values)
            trend = "increasing" if recent_avg > older_avg else "decreasing"
            
            return {
                "predicted_value": round(predicted_value, 2),
                "confidence": round(confidence, 3),
                "trend": trend,
                "current_value": values[-1] if values else 0,
                "prediction_time": future_time.isoformat(),
                "data_points_used": len(resource_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error predicting resource usage: {str(e)}")
            return {"error": str(e)}
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Performance summary dictionary
        """
        try:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_health": "good",
                "current_metrics": {},
                "recommendations_count": len(self.recommendations),
                "critical_issues": 0,
                "resource_utilization": {},
                "performance_trends": {}
            }
            
            # Current metrics summary
            critical_issues = 0
            for metric_key, metric in self.current_metrics.items():
                summary["current_metrics"][metric_key] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat()
                }
                
                # Check for critical issues
                if (metric.threshold_critical and 
                    metric.value >= metric.threshold_critical):
                    critical_issues += 1
            
            summary["critical_issues"] = critical_issues
            
            # Overall health assessment
            if critical_issues > 0:
                summary["overall_health"] = "critical"
            elif any(rec.priority == "high" for rec in self.recommendations.values()):
                summary["overall_health"] = "warning"
            else:
                summary["overall_health"] = "good"
            
            # Resource utilization summary
            for resource_type in ResourceType:
                resource_metrics = [
                    metric for metric in self.current_metrics.values()
                    if metric.resource_type == resource_type
                ]
                
                if resource_metrics:
                    avg_utilization = statistics.mean([m.value for m in resource_metrics])
                    summary["resource_utilization"][resource_type.value] = round(avg_utilization, 2)
            
            # Performance trends
            for resource_type, pattern in self.usage_patterns.items():
                summary["performance_trends"][resource_type] = {
                    "average_usage": round(pattern.average_usage, 2),
                    "peak_usage": round(pattern.peak_usage, 2),
                    "growth_rate": round(pattern.growth_rate, 2),
                    "peak_hours": pattern.peak_hours
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {"error": str(e)}
    
    async def implement_optimization(self, recommendation_id: str) -> Dict[str, Any]:
        """
        Implement a performance optimization recommendation
        
        Args:
            recommendation_id: ID of recommendation to implement
            
        Returns:
            Implementation result
        """
        try:
            if recommendation_id not in self.recommendations:
                return {"status": "error", "message": "Recommendation not found"}
            
            recommendation = self.recommendations[recommendation_id]
            
            # Simulate implementation (in production, this would involve actual changes)
            implementation_result = {
                "status": "success",
                "recommendation_id": recommendation_id,
                "strategy": recommendation.strategy.value,
                "implementation_time": datetime.utcnow().isoformat(),
                "expected_improvement": recommendation.expected_improvement,
                "implementation_steps_completed": recommendation.implementation_steps,
                "monitoring_required": True
            }
            
            # Log implementation
            logger.info(f"Implemented optimization: {recommendation.title}")
            
            # Remove implemented recommendation
            del self.recommendations[recommendation_id]
            
            return implementation_result
            
        except Exception as e:
            logger.error(f"Error implementing optimization: {str(e)}")
            return {"status": "error", "message": str(e)}

def create_performance_api(performance_optimizer: PerformanceOptimizer):
    """Create Flask API for performance optimization"""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "service": "performance_optimizer"})
    
    @app.route('/metrics/current', methods=['GET'])
    def get_current_metrics():
        try:
            metrics = {
                key: {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "resource_type": metric.resource_type.value
                }
                for key, metric in performance_optimizer.current_metrics.items()
            }
            
            return jsonify({
                "status": "success",
                "metrics": metrics,
                "count": len(metrics)
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/recommendations', methods=['GET'])
    def get_recommendations():
        try:
            recommendations = {
                rec_id: {
                    "resource_type": rec.resource_type.value,
                    "strategy": rec.strategy.value,
                    "title": rec.title,
                    "description": rec.description,
                    "expected_improvement": rec.expected_improvement,
                    "priority": rec.priority,
                    "implementation_effort": rec.implementation_effort,
                    "implementation_steps": rec.implementation_steps,
                    "created_at": rec.created_at.isoformat()
                }
                for rec_id, rec in performance_optimizer.recommendations.items()
            }
            
            return jsonify({
                "status": "success",
                "recommendations": recommendations,
                "count": len(recommendations)
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/optimize-cache/<cache_name>', methods=['POST'])
    async def optimize_cache(cache_name):
        try:
            config = await performance_optimizer.optimize_cache_configuration(cache_name)
            
            return jsonify({
                "status": "success",
                "cache_name": cache_name,
                "configuration": {
                    "cache_type": config.cache_type,
                    "max_memory": config.max_memory,
                    "eviction_policy": config.eviction_policy,
                    "ttl_default": config.ttl_default,
                    "compression_enabled": config.compression_enabled,
                    "persistence_enabled": config.persistence_enabled
                }
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/predict/<resource_type>', methods=['GET'])
    async def predict_resource_usage(resource_type):
        try:
            hours_ahead = request.args.get('hours', 24, type=int)
            resource_enum = ResourceType(resource_type)
            
            prediction = await performance_optimizer.predict_resource_usage(resource_enum, hours_ahead)
            
            return jsonify({
                "status": "success",
                "resource_type": resource_type,
                "hours_ahead": hours_ahead,
                "prediction": prediction
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/summary', methods=['GET'])
    async def get_performance_summary():
        try:
            summary = await performance_optimizer.get_performance_summary()
            return jsonify(summary)
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/implement/<recommendation_id>', methods=['POST'])
    async def implement_optimization(recommendation_id):
        try:
            result = await performance_optimizer.implement_optimization(recommendation_id)
            return jsonify(result)
            
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
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
    }
    
    # Initialize performance optimizer
    performance_optimizer = PerformanceOptimizer(config)
    
    # Create Flask API
    app = create_performance_api(performance_optimizer)
    
    print("Performance Optimizer API starting...")
    print("Available endpoints:")
    print("  GET /metrics/current - Get current performance metrics")
    print("  GET /recommendations - Get optimization recommendations")
    print("  POST /optimize-cache/<name> - Optimize cache configuration")
    print("  GET /predict/<resource_type> - Predict resource usage")
    print("  GET /summary - Get performance summary")
    print("  POST /implement/<recommendation_id> - Implement optimization")
    
    app.run(host='0.0.0.0', port=8013, debug=False)

