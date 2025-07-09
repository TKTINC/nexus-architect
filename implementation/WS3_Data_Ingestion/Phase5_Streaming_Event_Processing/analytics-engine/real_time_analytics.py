"""
Real-Time Analytics Engine for Nexus Architect
Provides comprehensive stream analytics, complex event processing, and real-time monitoring
"""

import json
import logging
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics
from collections import defaultdict, deque, Counter as CollectionsCounter
import math

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import from our stream processing components
from stream_processor import ProcessedEvent, EventCorrelationEngine, StreamTransformer
from kafka_streaming_manager import StreamEvent, EventType, StreamingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsWindow(Enum):
    """Time window types for analytics"""
    REAL_TIME = "real_time"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    HOUR_1 = "1hour"
    DAY_1 = "1day"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class AnalyticsMetric:
    """Analytics metric definition"""
    metric_id: str
    name: str
    description: str
    metric_type: str  # counter, gauge, histogram, rate
    aggregation: str  # sum, avg, min, max, count
    window: AnalyticsWindow
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    tags: Dict[str, str] = None

@dataclass
class RealTimeAlert:
    """Real-time alert definition"""
    alert_id: str
    metric_id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    value: float
    threshold: float
    tags: Dict[str, str]
    correlation_events: List[str] = None

@dataclass
class StreamPattern:
    """Complex event pattern for stream processing"""
    pattern_id: str
    name: str
    description: str
    event_sequence: List[EventType]
    time_window_minutes: int
    conditions: Dict[str, Any]
    action: str
    confidence_threshold: float

class WindowedAggregator:
    """Handles windowed aggregations for real-time analytics"""
    
    def __init__(self, window_size_minutes: int, slide_interval_seconds: int = 60):
        self.window_size_minutes = window_size_minutes
        self.slide_interval_seconds = slide_interval_seconds
        self.windows: Dict[str, deque] = defaultdict(lambda: deque())
        self.aggregated_values: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Metrics
        self.aggregations_computed = Counter('windowed_aggregations_total', 'Total aggregations computed', ['metric_type'])
        self.aggregation_duration = Histogram('windowed_aggregation_duration_seconds', 'Aggregation computation time')
        
        # Start background aggregation
        self._start_aggregation_thread()
    
    def add_value(self, metric_key: str, value: float, timestamp: datetime):
        """Add a value to the windowed aggregator"""
        self.windows[metric_key].append((timestamp, value))
        
        # Clean old values outside window
        self._clean_window(metric_key, timestamp)
    
    def _clean_window(self, metric_key: str, current_time: datetime):
        """Remove values outside the time window"""
        window_start = current_time - timedelta(minutes=self.window_size_minutes)
        
        while (self.windows[metric_key] and 
               self.windows[metric_key][0][0] < window_start):
            self.windows[metric_key].popleft()
    
    def get_aggregation(self, metric_key: str, aggregation_type: str) -> Optional[float]:
        """Get aggregated value for a metric"""
        if metric_key not in self.windows or not self.windows[metric_key]:
            return None
        
        values = [value for _, value in self.windows[metric_key]]
        
        try:
            if aggregation_type == 'sum':
                return sum(values)
            elif aggregation_type == 'avg':
                return statistics.mean(values)
            elif aggregation_type == 'min':
                return min(values)
            elif aggregation_type == 'max':
                return max(values)
            elif aggregation_type == 'count':
                return len(values)
            elif aggregation_type == 'median':
                return statistics.median(values)
            elif aggregation_type == 'std':
                return statistics.stdev(values) if len(values) > 1 else 0.0
            elif aggregation_type == 'rate':
                # Calculate rate per minute
                if len(values) < 2:
                    return 0.0
                time_span = (self.windows[metric_key][-1][0] - self.windows[metric_key][0][0]).total_seconds() / 60
                return len(values) / time_span if time_span > 0 else 0.0
            else:
                return None
        except Exception as e:
            logger.error(f"Aggregation error for {metric_key}: {e}")
            return None
    
    def _start_aggregation_thread(self):
        """Start background thread for periodic aggregation"""
        def aggregate_periodically():
            while True:
                try:
                    current_time = datetime.now(timezone.utc)
                    
                    with self.aggregation_duration.time():
                        for metric_key in list(self.windows.keys()):
                            # Clean window
                            self._clean_window(metric_key, current_time)
                            
                            # Compute aggregations
                            aggregations = {}
                            for agg_type in ['sum', 'avg', 'min', 'max', 'count', 'rate']:
                                value = self.get_aggregation(metric_key, agg_type)
                                if value is not None:
                                    aggregations[agg_type] = value
                            
                            self.aggregated_values[metric_key] = aggregations
                            self.aggregations_computed.labels(metric_type=metric_key).inc()
                    
                    time.sleep(self.slide_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Aggregation thread error: {e}")
                    time.sleep(self.slide_interval_seconds)
        
        thread = threading.Thread(target=aggregate_periodically, daemon=True)
        thread.start()

class AnomalyDetector:
    """Detects anomalies in streaming data using multiple algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.models: Dict[str, Any] = {}
        
        # Anomaly detection metrics
        self.anomalies_detected = Counter('anomalies_detected_total', 'Total anomalies detected', ['metric_type', 'algorithm'])
        self.detection_duration = Histogram('anomaly_detection_duration_seconds', 'Anomaly detection time')
        self.false_positive_rate = Gauge('anomaly_false_positive_rate', 'False positive rate', ['algorithm'])
        
        # Detection parameters
        self.z_score_threshold = config.get('z_score_threshold', 3.0)
        self.iqr_multiplier = config.get('iqr_multiplier', 1.5)
        self.isolation_contamination = config.get('isolation_contamination', 0.1)
    
    def add_data_point(self, metric_key: str, value: float, timestamp: datetime):
        """Add a data point for anomaly detection"""
        self.historical_data[metric_key].append((timestamp, value))
    
    def detect_anomalies(self, metric_key: str, current_value: float) -> Dict[str, Any]:
        """Detect anomalies using multiple algorithms"""
        anomaly_results = {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'algorithms': {},
            'confidence': 0.0
        }
        
        if metric_key not in self.historical_data or len(self.historical_data[metric_key]) < 10:
            return anomaly_results
        
        try:
            with self.detection_duration.time():
                historical_values = [value for _, value in self.historical_data[metric_key]]
                
                # Z-Score based detection
                z_score_result = self._detect_z_score_anomaly(historical_values, current_value)
                anomaly_results['algorithms']['z_score'] = z_score_result
                
                # IQR based detection
                iqr_result = self._detect_iqr_anomaly(historical_values, current_value)
                anomaly_results['algorithms']['iqr'] = iqr_result
                
                # Isolation Forest (if enough data)
                if len(historical_values) >= 50:
                    isolation_result = self._detect_isolation_anomaly(historical_values, current_value)
                    anomaly_results['algorithms']['isolation_forest'] = isolation_result
                
                # Statistical change point detection
                changepoint_result = self._detect_changepoint_anomaly(historical_values, current_value)
                anomaly_results['algorithms']['changepoint'] = changepoint_result
                
                # Combine results
                anomaly_scores = [result['anomaly_score'] for result in anomaly_results['algorithms'].values()]
                anomaly_flags = [result['is_anomaly'] for result in anomaly_results['algorithms'].values()]
                
                # Overall anomaly decision (majority vote with confidence weighting)
                anomaly_results['anomaly_score'] = np.mean(anomaly_scores)
                anomaly_results['is_anomaly'] = sum(anomaly_flags) >= len(anomaly_flags) / 2
                anomaly_results['confidence'] = np.std(anomaly_scores)  # Lower std = higher confidence
                
                # Update metrics
                if anomaly_results['is_anomaly']:
                    for algorithm, result in anomaly_results['algorithms'].items():
                        if result['is_anomaly']:
                            self.anomalies_detected.labels(
                                metric_type=metric_key,
                                algorithm=algorithm
                            ).inc()
        
        except Exception as e:
            logger.error(f"Anomaly detection failed for {metric_key}: {e}")
        
        return anomaly_results
    
    def _detect_z_score_anomaly(self, historical_values: List[float], current_value: float) -> Dict[str, Any]:
        """Detect anomaly using Z-score"""
        if len(historical_values) < 2:
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'z_score': 0.0}
        
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values)
        
        if std_val == 0:
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'z_score': 0.0}
        
        z_score = abs((current_value - mean_val) / std_val)
        is_anomaly = z_score > self.z_score_threshold
        anomaly_score = min(z_score / self.z_score_threshold, 1.0)
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'z_score': z_score
        }
    
    def _detect_iqr_anomaly(self, historical_values: List[float], current_value: float) -> Dict[str, Any]:
        """Detect anomaly using Interquartile Range"""
        if len(historical_values) < 4:
            return {'is_anomaly': False, 'anomaly_score': 0.0}
        
        q1 = np.percentile(historical_values, 25)
        q3 = np.percentile(historical_values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - (self.iqr_multiplier * iqr)
        upper_bound = q3 + (self.iqr_multiplier * iqr)
        
        is_anomaly = current_value < lower_bound or current_value > upper_bound
        
        # Calculate anomaly score based on distance from bounds
        if current_value < lower_bound:
            anomaly_score = min((lower_bound - current_value) / (iqr * self.iqr_multiplier), 1.0)
        elif current_value > upper_bound:
            anomaly_score = min((current_value - upper_bound) / (iqr * self.iqr_multiplier), 1.0)
        else:
            anomaly_score = 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    def _detect_isolation_anomaly(self, historical_values: List[float], current_value: float) -> Dict[str, Any]:
        """Detect anomaly using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Prepare data
            X = np.array(historical_values + [current_value]).reshape(-1, 1)
            
            # Train isolation forest
            iso_forest = IsolationForest(contamination=self.isolation_contamination, random_state=42)
            predictions = iso_forest.fit_predict(X)
            scores = iso_forest.decision_function(X)
            
            # Check if current value is anomaly
            current_prediction = predictions[-1]
            current_score = scores[-1]
            
            is_anomaly = current_prediction == -1
            anomaly_score = max(0, -current_score)  # Convert to positive score
            
            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'isolation_score': current_score
            }
            
        except Exception as e:
            logger.error(f"Isolation forest detection failed: {e}")
            return {'is_anomaly': False, 'anomaly_score': 0.0}
    
    def _detect_changepoint_anomaly(self, historical_values: List[float], current_value: float) -> Dict[str, Any]:
        """Detect anomaly using change point detection"""
        if len(historical_values) < 20:
            return {'is_anomaly': False, 'anomaly_score': 0.0}
        
        try:
            # Simple change point detection using moving averages
            window_size = min(10, len(historical_values) // 2)
            
            recent_mean = np.mean(historical_values[-window_size:])
            older_mean = np.mean(historical_values[:-window_size])
            
            # Calculate change magnitude
            change_magnitude = abs(recent_mean - older_mean)
            historical_std = np.std(historical_values)
            
            if historical_std == 0:
                return {'is_anomaly': False, 'anomaly_score': 0.0}
            
            # Normalize change by historical standard deviation
            normalized_change = change_magnitude / historical_std
            
            # Check if current value continues the change pattern
            current_deviation = abs(current_value - recent_mean) / historical_std
            
            is_anomaly = normalized_change > 2.0 and current_deviation > 2.0
            anomaly_score = min((normalized_change + current_deviation) / 4.0, 1.0)
            
            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'change_magnitude': normalized_change
            }
            
        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            return {'is_anomaly': False, 'anomaly_score': 0.0}

class ComplexEventProcessor:
    """Processes complex event patterns and sequences"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_buffer: deque = deque(maxlen=10000)
        self.pattern_states: Dict[str, Dict] = defaultdict(dict)
        
        # CEP metrics
        self.patterns_matched = Counter('cep_patterns_matched_total', 'Total patterns matched', ['pattern_id'])
        self.pattern_processing_duration = Histogram('cep_pattern_processing_duration_seconds', 'Pattern processing time')
        
        # Load patterns
        self._load_patterns()
    
    def _load_patterns(self):
        """Load complex event patterns"""
        self.patterns = [
            StreamPattern(
                pattern_id='deployment_sequence',
                name='Code Deployment Sequence',
                description='Code change followed by system metrics change',
                event_sequence=[EventType.CODE_CHANGE, EventType.SYSTEM_METRIC],
                time_window_minutes=30,
                conditions={
                    'code_change': {'branch': ['main', 'master', 'production']},
                    'system_metric': {'metric_type': ['cpu', 'memory', 'response_time']}
                },
                action='trigger_deployment_alert',
                confidence_threshold=0.8
            ),
            StreamPattern(
                pattern_id='quality_degradation',
                name='Quality Degradation Pattern',
                description='Multiple quality alerts in sequence',
                event_sequence=[EventType.QUALITY_ALERT, EventType.QUALITY_ALERT],
                time_window_minutes=15,
                conditions={
                    'quality_alert': {'severity': ['high', 'critical']}
                },
                action='escalate_quality_issue',
                confidence_threshold=0.9
            ),
            StreamPattern(
                pattern_id='collaboration_burst',
                name='Collaboration Burst',
                description='High communication activity around project updates',
                event_sequence=[EventType.PROJECT_UPDATE, EventType.COMMUNICATION_MESSAGE, EventType.COMMUNICATION_MESSAGE],
                time_window_minutes=10,
                conditions={
                    'project_update': {'status_change': True},
                    'communication_message': {'urgency_level': ['high', 'urgent']}
                },
                action='highlight_active_collaboration',
                confidence_threshold=0.7
            ),
            StreamPattern(
                pattern_id='knowledge_discovery',
                name='Knowledge Discovery Sequence',
                description='Documentation update followed by knowledge extraction',
                event_sequence=[EventType.DOCUMENTATION_UPDATE, EventType.KNOWLEDGE_EXTRACTION],
                time_window_minutes=60,
                conditions={
                    'documentation_update': {'content_length': {'min': 500}},
                    'knowledge_extraction': {'entities_extracted': {'min': 5}}
                },
                action='promote_knowledge_content',
                confidence_threshold=0.6
            )
        ]
    
    def process_event(self, processed_event: ProcessedEvent):
        """Process event for complex pattern matching"""
        self.event_buffer.append(processed_event)
        
        try:
            with self.pattern_processing_duration.time():
                # Check all patterns
                for pattern in self.patterns:
                    matches = self._check_pattern(pattern, processed_event)
                    
                    if matches:
                        self.patterns_matched.labels(pattern_id=pattern.pattern_id).inc()
                        self._execute_pattern_action(pattern, matches)
        
        except Exception as e:
            logger.error(f"Complex event processing failed: {e}")
    
    def _check_pattern(self, pattern: StreamPattern, current_event: ProcessedEvent) -> Optional[List[ProcessedEvent]]:
        """Check if a pattern matches current event sequence"""
        if not pattern.event_sequence:
            return None
        
        # Get events within time window
        window_start = current_event.processed_timestamp - timedelta(minutes=pattern.time_window_minutes)
        window_events = [
            event for event in self.event_buffer
            if event.processed_timestamp >= window_start
        ]
        
        # Try to match the pattern sequence
        return self._match_sequence(pattern, window_events)
    
    def _match_sequence(self, pattern: StreamPattern, events: List[ProcessedEvent]) -> Optional[List[ProcessedEvent]]:
        """Match event sequence against pattern"""
        sequence = pattern.event_sequence
        matched_events = []
        
        # Simple sequential matching
        sequence_index = 0
        
        for event in events:
            if sequence_index >= len(sequence):
                break
            
            expected_type = sequence[sequence_index]
            
            if event.original_event.event_type == expected_type:
                # Check conditions
                if self._check_conditions(pattern, event, expected_type):
                    matched_events.append(event)
                    sequence_index += 1
        
        # Check if we matched the complete sequence
        if sequence_index == len(sequence):
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(pattern, matched_events)
            
            if confidence >= pattern.confidence_threshold:
                return matched_events
        
        return None
    
    def _check_conditions(self, pattern: StreamPattern, event: ProcessedEvent, event_type: EventType) -> bool:
        """Check if event meets pattern conditions"""
        conditions = pattern.conditions.get(event_type.value, {})
        
        for field, condition in conditions.items():
            event_value = event.original_event.data.get(field)
            
            if isinstance(condition, list):
                # Value must be in list
                if event_value not in condition:
                    return False
            elif isinstance(condition, dict):
                # Range or comparison conditions
                if 'min' in condition and (event_value is None or event_value < condition['min']):
                    return False
                if 'max' in condition and (event_value is None or event_value > condition['max']):
                    return False
            elif event_value != condition:
                # Exact match
                return False
        
        return True
    
    def _calculate_pattern_confidence(self, pattern: StreamPattern, matched_events: List[ProcessedEvent]) -> float:
        """Calculate confidence score for pattern match"""
        if not matched_events:
            return 0.0
        
        # Base confidence from event confidence scores
        event_confidences = [event.confidence_score for event in matched_events]
        base_confidence = np.mean(event_confidences)
        
        # Time proximity bonus (events closer in time = higher confidence)
        if len(matched_events) > 1:
            time_spans = []
            for i in range(1, len(matched_events)):
                time_diff = (matched_events[i].processed_timestamp - matched_events[i-1].processed_timestamp).total_seconds()
                time_spans.append(time_diff)
            
            avg_time_span = np.mean(time_spans)
            max_time_span = pattern.time_window_minutes * 60
            
            time_proximity = 1.0 - (avg_time_span / max_time_span)
            time_proximity = max(0.0, time_proximity)
        else:
            time_proximity = 1.0
        
        # Correlation bonus
        correlation_bonus = 0.0
        for event in matched_events:
            if event.correlations:
                correlation_bonus += 0.1
        
        correlation_bonus = min(correlation_bonus, 0.3)
        
        # Combined confidence
        final_confidence = (base_confidence * 0.6) + (time_proximity * 0.3) + correlation_bonus
        
        return min(final_confidence, 1.0)
    
    def _execute_pattern_action(self, pattern: StreamPattern, matched_events: List[ProcessedEvent]):
        """Execute action for matched pattern"""
        logger.info(f"Pattern matched: {pattern.name} with {len(matched_events)} events")
        
        # Create pattern match event
        pattern_event = {
            'pattern_id': pattern.pattern_id,
            'pattern_name': pattern.name,
            'matched_events': [event.original_event.event_id for event in matched_events],
            'confidence': self._calculate_pattern_confidence(pattern, matched_events),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': pattern.action
        }
        
        # Store pattern match (could trigger alerts, notifications, etc.)
        self._store_pattern_match(pattern_event)
    
    def _store_pattern_match(self, pattern_event: Dict[str, Any]):
        """Store pattern match for further processing"""
        # This could trigger alerts, update dashboards, etc.
        logger.info(f"Storing pattern match: {pattern_event['pattern_id']}")

class RealTimeAnalyticsEngine:
    """Main real-time analytics engine coordinating all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config['database']
        self.redis_config = config['redis']
        
        # Initialize components
        self.aggregator = WindowedAggregator(window_size_minutes=15)
        self.anomaly_detector = AnomalyDetector(config.get('anomaly_detection', {}))
        self.cep_processor = ComplexEventProcessor(config.get('complex_event_processing', {}))
        
        # Redis client for real-time data
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Analytics metrics
        self.events_processed = Counter('analytics_events_processed_total', 'Total events processed')
        self.alerts_generated = Counter('analytics_alerts_generated_total', 'Total alerts generated', ['severity'])
        self.processing_latency = Histogram('analytics_processing_latency_seconds', 'Event processing latency')
        
        # Real-time metrics storage
        self.current_metrics: Dict[str, float] = {}
        self.alerts: List[RealTimeAlert] = []
        
        # Initialize database schema
        self._init_analytics_schema()
        
        # Load analytics metrics configuration
        self._load_metrics_config()
    
    def _init_analytics_schema(self):
        """Initialize analytics database schema"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Analytics metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_metrics (
                    metric_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    metric_type VARCHAR(100) NOT NULL,
                    aggregation VARCHAR(100) NOT NULL,
                    window_type VARCHAR(100) NOT NULL,
                    threshold_warning FLOAT,
                    threshold_critical FLOAT,
                    tags JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS analytics_values (
                    value_id VARCHAR(255) PRIMARY KEY,
                    metric_id VARCHAR(255) REFERENCES analytics_metrics(metric_id),
                    value FLOAT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    tags JSONB,
                    window_type VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS analytics_alerts (
                    alert_id VARCHAR(255) PRIMARY KEY,
                    metric_id VARCHAR(255) REFERENCES analytics_metrics(metric_id),
                    severity VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    value FLOAT,
                    threshold_value FLOAT,
                    tags JSONB,
                    correlation_events JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS pattern_matches (
                    match_id VARCHAR(255) PRIMARY KEY,
                    pattern_id VARCHAR(255) NOT NULL,
                    pattern_name VARCHAR(255) NOT NULL,
                    matched_events JSONB NOT NULL,
                    confidence FLOAT NOT NULL,
                    action VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_analytics_values_metric_timestamp ON analytics_values(metric_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_analytics_alerts_severity ON analytics_alerts(severity);
                CREATE INDEX IF NOT EXISTS idx_pattern_matches_pattern ON pattern_matches(pattern_id);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Analytics database schema initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics schema: {e}")
    
    def _load_metrics_config(self):
        """Load analytics metrics configuration"""
        self.metrics_config = [
            AnalyticsMetric(
                metric_id='event_rate',
                name='Event Processing Rate',
                description='Number of events processed per minute',
                metric_type='rate',
                aggregation='count',
                window=AnalyticsWindow.MINUTE_1,
                threshold_warning=1000.0,
                threshold_critical=2000.0,
                tags={'category': 'performance'}
            ),
            AnalyticsMetric(
                metric_id='processing_latency',
                name='Processing Latency',
                description='Average event processing latency',
                metric_type='gauge',
                aggregation='avg',
                window=AnalyticsWindow.MINUTE_5,
                threshold_warning=100.0,
                threshold_critical=500.0,
                tags={'category': 'performance'}
            ),
            AnalyticsMetric(
                metric_id='error_rate',
                name='Error Rate',
                description='Percentage of events with processing errors',
                metric_type='rate',
                aggregation='avg',
                window=AnalyticsWindow.MINUTE_5,
                threshold_warning=5.0,
                threshold_critical=10.0,
                tags={'category': 'reliability'}
            ),
            AnalyticsMetric(
                metric_id='correlation_accuracy',
                name='Correlation Accuracy',
                description='Accuracy of event correlations',
                metric_type='gauge',
                aggregation='avg',
                window=AnalyticsWindow.MINUTE_15,
                threshold_warning=80.0,
                threshold_critical=70.0,
                tags={'category': 'quality'}
            ),
            AnalyticsMetric(
                metric_id='anomaly_detection_rate',
                name='Anomaly Detection Rate',
                description='Rate of anomalies detected per hour',
                metric_type='rate',
                aggregation='count',
                window=AnalyticsWindow.HOUR_1,
                threshold_warning=10.0,
                threshold_critical=50.0,
                tags={'category': 'security'}
            )
        ]
    
    def process_event(self, processed_event: ProcessedEvent):
        """Process event through analytics pipeline"""
        start_time = time.time()
        
        try:
            # Update event processing metrics
            self.events_processed.inc()
            
            # Extract metrics from event
            self._extract_event_metrics(processed_event)
            
            # Anomaly detection
            self._detect_anomalies(processed_event)
            
            # Complex event processing
            self.cep_processor.process_event(processed_event)
            
            # Update processing latency
            processing_time = time.time() - start_time
            self.processing_latency.observe(processing_time)
            
            # Store real-time metrics
            self._update_real_time_metrics(processed_event, processing_time)
            
        except Exception as e:
            logger.error(f"Analytics processing failed for event {processed_event.original_event.event_id}: {e}")
    
    def _extract_event_metrics(self, processed_event: ProcessedEvent):
        """Extract metrics from processed event"""
        event = processed_event.original_event
        
        # Event rate by type
        metric_key = f"event_rate_{event.event_type.value}"
        self.aggregator.add_value(metric_key, 1.0, event.timestamp)
        
        # Processing duration
        if processed_event.processing_duration_ms:
            self.aggregator.add_value('processing_latency', processed_event.processing_duration_ms, event.timestamp)
        
        # Correlation count
        correlation_count = len(processed_event.correlations)
        self.aggregator.add_value('correlation_count', correlation_count, event.timestamp)
        
        # Event size
        event_size = len(json.dumps(event.data))
        self.aggregator.add_value('event_size', event_size, event.timestamp)
        
        # Source system metrics
        source_metric_key = f"events_by_source_{event.source_system}"
        self.aggregator.add_value(source_metric_key, 1.0, event.timestamp)
    
    def _detect_anomalies(self, processed_event: ProcessedEvent):
        """Detect anomalies in event metrics"""
        event = processed_event.original_event
        
        # Check processing duration anomaly
        if processed_event.processing_duration_ms:
            self.anomaly_detector.add_data_point(
                'processing_duration',
                processed_event.processing_duration_ms,
                event.timestamp
            )
            
            anomaly_result = self.anomaly_detector.detect_anomalies(
                'processing_duration',
                processed_event.processing_duration_ms
            )
            
            if anomaly_result['is_anomaly']:
                self._generate_alert(
                    metric_id='processing_latency',
                    severity=AlertSeverity.WARNING,
                    message=f"Processing latency anomaly detected: {processed_event.processing_duration_ms}ms",
                    value=processed_event.processing_duration_ms,
                    threshold=0.0,
                    tags={'event_type': event.event_type.value, 'source': event.source_system},
                    correlation_events=[event.event_id]
                )
        
        # Check event size anomaly
        event_size = len(json.dumps(event.data))
        self.anomaly_detector.add_data_point('event_size', event_size, event.timestamp)
        
        size_anomaly = self.anomaly_detector.detect_anomalies('event_size', event_size)
        
        if size_anomaly['is_anomaly']:
            self._generate_alert(
                metric_id='event_size',
                severity=AlertSeverity.INFO,
                message=f"Event size anomaly detected: {event_size} bytes",
                value=event_size,
                threshold=0.0,
                tags={'event_type': event.event_type.value, 'source': event.source_system},
                correlation_events=[event.event_id]
            )
    
    def _generate_alert(self, metric_id: str, severity: AlertSeverity, message: str, 
                       value: float, threshold: float, tags: Dict[str, str], 
                       correlation_events: List[str] = None):
        """Generate real-time alert"""
        alert = RealTimeAlert(
            alert_id=str(uuid.uuid4()),
            metric_id=metric_id,
            severity=severity,
            message=message,
            timestamp=datetime.now(timezone.utc),
            value=value,
            threshold=threshold,
            tags=tags,
            correlation_events=correlation_events or []
        )
        
        # Store alert
        self.alerts.append(alert)
        
        # Update metrics
        self.alerts_generated.labels(severity=severity.value).inc()
        
        # Store in database
        self._store_alert(alert)
        
        # Store in Redis for real-time access
        self._cache_alert(alert)
        
        logger.warning(f"Alert generated: {alert.message}")
    
    def _store_alert(self, alert: RealTimeAlert):
        """Store alert in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analytics_alerts 
                (alert_id, metric_id, severity, message, value, threshold_value, tags, correlation_events)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                alert.alert_id,
                alert.metric_id,
                alert.severity.value,
                alert.message,
                alert.value,
                alert.threshold,
                json.dumps(alert.tags),
                json.dumps(alert.correlation_events)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def _cache_alert(self, alert: RealTimeAlert):
        """Cache alert in Redis for real-time access"""
        try:
            alert_data = {
                'alert_id': alert.alert_id,
                'metric_id': alert.metric_id,
                'severity': alert.severity.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'value': alert.value,
                'threshold': alert.threshold,
                'tags': alert.tags,
                'correlation_events': alert.correlation_events
            }
            
            # Store in Redis with expiration
            self.redis_client.setex(
                f"alert:{alert.alert_id}",
                3600,  # 1 hour expiration
                json.dumps(alert_data)
            )
            
            # Add to recent alerts list
            self.redis_client.lpush('recent_alerts', alert.alert_id)
            self.redis_client.ltrim('recent_alerts', 0, 99)  # Keep last 100 alerts
            
        except Exception as e:
            logger.error(f"Failed to cache alert: {e}")
    
    def _update_real_time_metrics(self, processed_event: ProcessedEvent, processing_time: float):
        """Update real-time metrics"""
        current_time = datetime.now(timezone.utc)
        
        # Update current metrics
        self.current_metrics.update({
            'events_processed_total': self.events_processed._value._value,
            'current_processing_latency': processing_time * 1000,  # Convert to ms
            'active_correlations': len(processed_event.correlations),
            'timestamp': current_time.isoformat()
        })
        
        # Cache in Redis
        try:
            self.redis_client.setex(
                'real_time_metrics',
                60,  # 1 minute expiration
                json.dumps(self.current_metrics)
            )
        except Exception as e:
            logger.error(f"Failed to cache real-time metrics: {e}")
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get data for real-time dashboard"""
        try:
            # Get current metrics
            dashboard_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': {},
                'alerts': [],
                'aggregations': {},
                'health': self._get_system_health()
            }
            
            # Get aggregated metrics
            for metric_config in self.metrics_config:
                metric_key = metric_config.metric_id
                aggregation = self.aggregator.get_aggregation(metric_key, metric_config.aggregation)
                
                if aggregation is not None:
                    dashboard_data['metrics'][metric_key] = {
                        'value': aggregation,
                        'name': metric_config.name,
                        'description': metric_config.description,
                        'threshold_warning': metric_config.threshold_warning,
                        'threshold_critical': metric_config.threshold_critical,
                        'tags': metric_config.tags
                    }
            
            # Get recent alerts
            recent_alert_ids = self.redis_client.lrange('recent_alerts', 0, 9)  # Last 10 alerts
            
            for alert_id in recent_alert_ids:
                alert_data = self.redis_client.get(f"alert:{alert_id.decode()}")
                if alert_data:
                    dashboard_data['alerts'].append(json.loads(alert_data))
            
            # Get windowed aggregations
            for metric_key, aggregations in self.aggregator.aggregated_values.items():
                dashboard_data['aggregations'][metric_key] = aggregations
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.now(timezone.utc).isoformat()}
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            # Check component health
            health = {
                'overall': 'healthy',
                'components': {
                    'aggregator': 'healthy',
                    'anomaly_detector': 'healthy',
                    'cep_processor': 'healthy',
                    'database': 'healthy',
                    'redis': 'healthy'
                },
                'metrics': {
                    'events_processed': self.events_processed._value._value,
                    'alerts_generated': sum(self.alerts_generated._value.values()),
                    'avg_processing_latency': self.current_metrics.get('current_processing_latency', 0)
                }
            }
            
            # Check database connectivity
            try:
                conn = psycopg2.connect(**self.db_config)
                conn.close()
            except Exception:
                health['components']['database'] = 'unhealthy'
                health['overall'] = 'degraded'
            
            # Check Redis connectivity
            try:
                self.redis_client.ping()
            except Exception:
                health['components']['redis'] = 'unhealthy'
                health['overall'] = 'degraded'
            
            return health
            
        except Exception as e:
            return {
                'overall': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

# Flask API for real-time dashboard
def create_analytics_api(analytics_engine: RealTimeAnalyticsEngine) -> Flask:
    """Create Flask API for analytics dashboard"""
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()})
    
    @app.route('/api/dashboard', methods=['GET'])
    def get_dashboard_data():
        """Get real-time dashboard data"""
        try:
            data = analytics_engine.get_real_time_dashboard_data()
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/metrics', methods=['GET'])
    def get_metrics():
        """Get current metrics"""
        try:
            return jsonify(analytics_engine.current_metrics)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/alerts', methods=['GET'])
    def get_alerts():
        """Get recent alerts"""
        try:
            severity_filter = request.args.get('severity')
            limit = int(request.args.get('limit', 50))
            
            # Get alerts from Redis
            alert_ids = analytics_engine.redis_client.lrange('recent_alerts', 0, limit - 1)
            alerts = []
            
            for alert_id in alert_ids:
                alert_data = analytics_engine.redis_client.get(f"alert:{alert_id.decode()}")
                if alert_data:
                    alert = json.loads(alert_data)
                    if not severity_filter or alert['severity'] == severity_filter:
                        alerts.append(alert)
            
            return jsonify({'alerts': alerts, 'count': len(alerts)})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/patterns', methods=['GET'])
    def get_pattern_matches():
        """Get recent pattern matches"""
        try:
            conn = psycopg2.connect(**analytics_engine.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM pattern_matches 
                ORDER BY created_at DESC 
                LIMIT 50
            """)
            
            patterns = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Convert to JSON serializable format
            pattern_list = []
            for pattern in patterns:
                pattern_dict = dict(pattern)
                pattern_dict['created_at'] = pattern_dict['created_at'].isoformat()
                pattern_list.append(pattern_dict)
            
            return jsonify({'patterns': pattern_list, 'count': len(pattern_list)})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

# Example usage
if __name__ == "__main__":
    # Configuration
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
        },
        'anomaly_detection': {
            'z_score_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'isolation_contamination': 0.1
        }
    }
    
    # Start Prometheus metrics server
    start_http_server(8091)
    
    # Initialize analytics engine
    analytics_engine = RealTimeAnalyticsEngine(config)
    
    # Create and run Flask API
    app = create_analytics_api(analytics_engine)
    
    print("Starting Real-Time Analytics Engine...")
    print("Prometheus metrics: http://localhost:8091")
    print("Analytics API: http://localhost:8004")
    
    app.run(host='0.0.0.0', port=8004, debug=False)

