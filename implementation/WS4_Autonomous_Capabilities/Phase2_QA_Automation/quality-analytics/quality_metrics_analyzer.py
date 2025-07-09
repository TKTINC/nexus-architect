"""
Quality Metrics Analyzer for Nexus Architect QA Automation
Implements comprehensive quality analytics with predictive insights and reporting
"""

import asyncio
import json
import logging
import math
import statistics
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import uuid

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityMetricType(Enum):
    """Quality metric types"""
    DEFECT_DENSITY = "defect_density"
    TEST_COVERAGE = "test_coverage"
    CODE_COMPLEXITY = "code_complexity"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    TECHNICAL_DEBT = "technical_debt"

class TrendDirection(Enum):
    """Trend direction enumeration"""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"

@dataclass
class QualityMetric:
    """Quality metric data structure"""
    id: str
    name: str
    type: QualityMetricType
    value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    unit: str
    description: str
    calculation_method: str
    data_source: str
    timestamp: datetime
    tags: List[str]

@dataclass
class QualityTrend:
    """Quality trend analysis"""
    metric_type: QualityMetricType
    direction: TrendDirection
    change_rate: float
    confidence: float
    prediction: float
    time_to_target: Optional[float]
    historical_data: List[Tuple[datetime, float]]

@dataclass
class QualityInsight:
    """Quality insight and recommendation"""
    id: str
    title: str
    description: str
    severity: str
    category: str
    affected_metrics: List[str]
    recommendations: List[str]
    estimated_impact: float
    effort_estimate: str
    priority_score: float
    created_at: datetime

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    id: str
    title: str
    period_start: datetime
    period_end: datetime
    overall_score: float
    metrics: List[QualityMetric]
    trends: List[QualityTrend]
    insights: List[QualityInsight]
    summary: Dict[str, Any]
    charts: List[Dict[str, Any]]
    created_at: datetime

class MetricsCalculator:
    """Calculate various quality metrics from test and code data"""
    
    def __init__(self):
        self.metric_definitions = self._load_metric_definitions()
    
    def _load_metric_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load metric definitions and calculation methods"""
        return {
            'defect_density': {
                'name': 'Defect Density',
                'unit': 'defects/kloc',
                'target': 2.0,
                'warning_threshold': 5.0,
                'critical_threshold': 10.0,
                'description': 'Number of defects per thousand lines of code'
            },
            'test_coverage': {
                'name': 'Test Coverage',
                'unit': '%',
                'target': 90.0,
                'warning_threshold': 70.0,
                'critical_threshold': 50.0,
                'description': 'Percentage of code covered by tests'
            },
            'code_complexity': {
                'name': 'Cyclomatic Complexity',
                'unit': 'complexity',
                'target': 5.0,
                'warning_threshold': 10.0,
                'critical_threshold': 15.0,
                'description': 'Average cyclomatic complexity per function'
            },
            'maintainability_index': {
                'name': 'Maintainability Index',
                'unit': 'index',
                'target': 80.0,
                'warning_threshold': 60.0,
                'critical_threshold': 40.0,
                'description': 'Code maintainability score (0-100)'
            },
            'test_pass_rate': {
                'name': 'Test Pass Rate',
                'unit': '%',
                'target': 95.0,
                'warning_threshold': 85.0,
                'critical_threshold': 75.0,
                'description': 'Percentage of tests passing'
            },
            'build_success_rate': {
                'name': 'Build Success Rate',
                'unit': '%',
                'target': 98.0,
                'warning_threshold': 90.0,
                'critical_threshold': 80.0,
                'description': 'Percentage of successful builds'
            },
            'mean_time_to_fix': {
                'name': 'Mean Time to Fix',
                'unit': 'hours',
                'target': 4.0,
                'warning_threshold': 8.0,
                'critical_threshold': 24.0,
                'description': 'Average time to fix defects'
            },
            'technical_debt_ratio': {
                'name': 'Technical Debt Ratio',
                'unit': '%',
                'target': 5.0,
                'warning_threshold': 10.0,
                'critical_threshold': 20.0,
                'description': 'Ratio of technical debt to total development cost'
            }
        }
    
    def calculate_defect_density(self, defects: int, lines_of_code: int) -> QualityMetric:
        """Calculate defect density metric"""
        if lines_of_code == 0:
            value = 0.0
        else:
            value = (defects / lines_of_code) * 1000  # per KLOC
        
        definition = self.metric_definitions['defect_density']
        
        return QualityMetric(
            id=f"defect_density_{int(time.time())}",
            name=definition['name'],
            type=QualityMetricType.DEFECT_DENSITY,
            value=value,
            target_value=definition['target'],
            threshold_warning=definition['warning_threshold'],
            threshold_critical=definition['critical_threshold'],
            unit=definition['unit'],
            description=definition['description'],
            calculation_method=f"({defects} defects / {lines_of_code} LOC) * 1000",
            data_source="defect_tracking_system",
            timestamp=datetime.now(),
            tags=["quality", "defects"]
        )
    
    def calculate_test_coverage(self, covered_lines: int, total_lines: int) -> QualityMetric:
        """Calculate test coverage metric"""
        if total_lines == 0:
            value = 0.0
        else:
            value = (covered_lines / total_lines) * 100
        
        definition = self.metric_definitions['test_coverage']
        
        return QualityMetric(
            id=f"test_coverage_{int(time.time())}",
            name=definition['name'],
            type=QualityMetricType.TEST_COVERAGE,
            value=value,
            target_value=definition['target'],
            threshold_warning=definition['warning_threshold'],
            threshold_critical=definition['critical_threshold'],
            unit=definition['unit'],
            description=definition['description'],
            calculation_method=f"({covered_lines} / {total_lines}) * 100",
            data_source="coverage_analysis",
            timestamp=datetime.now(),
            tags=["quality", "testing", "coverage"]
        )
    
    def calculate_code_complexity(self, complexity_scores: List[float]) -> QualityMetric:
        """Calculate average code complexity metric"""
        if not complexity_scores:
            value = 0.0
        else:
            value = statistics.mean(complexity_scores)
        
        definition = self.metric_definitions['code_complexity']
        
        return QualityMetric(
            id=f"code_complexity_{int(time.time())}",
            name=definition['name'],
            type=QualityMetricType.CODE_COMPLEXITY,
            value=value,
            target_value=definition['target'],
            threshold_warning=definition['warning_threshold'],
            threshold_critical=definition['critical_threshold'],
            unit=definition['unit'],
            description=definition['description'],
            calculation_method=f"mean({len(complexity_scores)} complexity scores)",
            data_source="static_analysis",
            timestamp=datetime.now(),
            tags=["quality", "complexity", "maintainability"]
        )
    
    def calculate_maintainability_index(self, halstead_volume: float, 
                                      cyclomatic_complexity: float, 
                                      lines_of_code: int) -> QualityMetric:
        """Calculate maintainability index"""
        if lines_of_code == 0:
            value = 0.0
        else:
            # Simplified maintainability index calculation
            value = max(0, 171 - 5.2 * math.log(halstead_volume) - 
                         0.23 * cyclomatic_complexity - 16.2 * math.log(lines_of_code))
        
        definition = self.metric_definitions['maintainability_index']
        
        return QualityMetric(
            id=f"maintainability_{int(time.time())}",
            name=definition['name'],
            type=QualityMetricType.MAINTAINABILITY,
            value=value,
            target_value=definition['target'],
            threshold_warning=definition['warning_threshold'],
            threshold_critical=definition['critical_threshold'],
            unit=definition['unit'],
            description=definition['description'],
            calculation_method="171 - 5.2*ln(HV) - 0.23*CC - 16.2*ln(LOC)",
            data_source="static_analysis",
            timestamp=datetime.now(),
            tags=["quality", "maintainability"]
        )
    
    def calculate_test_pass_rate(self, passed_tests: int, total_tests: int) -> QualityMetric:
        """Calculate test pass rate metric"""
        if total_tests == 0:
            value = 0.0
        else:
            value = (passed_tests / total_tests) * 100
        
        definition = self.metric_definitions['test_pass_rate']
        
        return QualityMetric(
            id=f"test_pass_rate_{int(time.time())}",
            name=definition['name'],
            type=QualityMetricType.RELIABILITY,
            value=value,
            target_value=definition['target'],
            threshold_warning=definition['warning_threshold'],
            threshold_critical=definition['critical_threshold'],
            unit=definition['unit'],
            description=definition['description'],
            calculation_method=f"({passed_tests} / {total_tests}) * 100",
            data_source="test_execution",
            timestamp=datetime.now(),
            tags=["quality", "testing", "reliability"]
        )
    
    def calculate_technical_debt_ratio(self, debt_hours: float, 
                                     development_hours: float) -> QualityMetric:
        """Calculate technical debt ratio"""
        if development_hours == 0:
            value = 0.0
        else:
            value = (debt_hours / development_hours) * 100
        
        definition = self.metric_definitions['technical_debt_ratio']
        
        return QualityMetric(
            id=f"technical_debt_{int(time.time())}",
            name=definition['name'],
            type=QualityMetricType.TECHNICAL_DEBT,
            value=value,
            target_value=definition['target'],
            threshold_warning=definition['warning_threshold'],
            threshold_critical=definition['critical_threshold'],
            unit=definition['unit'],
            description=definition['description'],
            calculation_method=f"({debt_hours} / {development_hours}) * 100",
            data_source="static_analysis",
            timestamp=datetime.now(),
            tags=["quality", "technical_debt"]
        )

class TrendAnalyzer:
    """Analyze quality trends and predict future values"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def analyze_trend(self, historical_data: List[Tuple[datetime, float]], 
                     metric_type: QualityMetricType) -> QualityTrend:
        """Analyze trend for a specific metric"""
        if len(historical_data) < 3:
            return QualityTrend(
                metric_type=metric_type,
                direction=TrendDirection.STABLE,
                change_rate=0.0,
                confidence=0.0,
                prediction=historical_data[-1][1] if historical_data else 0.0,
                time_to_target=None,
                historical_data=historical_data
            )
        
        # Prepare data for analysis
        timestamps = [data[0] for data in historical_data]
        values = [data[1] for data in historical_data]
        
        # Convert timestamps to numeric values (days since first measurement)
        base_time = timestamps[0]
        x_values = [(ts - base_time).days for ts in timestamps]
        
        # Perform linear regression
        X = np.array(x_values).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate trend metrics
        slope = model.coef_[0]
        r2 = r2_score(y, model.predict(X))
        
        # Determine trend direction
        direction = self._determine_trend_direction(slope, values)
        
        # Calculate change rate (percentage change per day)
        if len(values) > 1 and values[0] != 0:
            change_rate = (slope / abs(values[0])) * 100
        else:
            change_rate = 0.0
        
        # Predict next value (30 days ahead)
        next_x = x_values[-1] + 30
        prediction = model.predict([[next_x]])[0]
        
        # Calculate time to target (if applicable)
        time_to_target = self._calculate_time_to_target(
            model, x_values[-1], values[-1], 
            self._get_target_value(metric_type)
        )
        
        return QualityTrend(
            metric_type=metric_type,
            direction=direction,
            change_rate=change_rate,
            confidence=r2,
            prediction=prediction,
            time_to_target=time_to_target,
            historical_data=historical_data
        )
    
    def _determine_trend_direction(self, slope: float, values: List[float]) -> TrendDirection:
        """Determine trend direction based on slope and volatility"""
        # Calculate coefficient of variation for volatility
        if len(values) > 1:
            cv = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) != 0 else 0
        else:
            cv = 0
        
        # High volatility threshold
        if cv > 0.3:
            return TrendDirection.VOLATILE
        
        # Trend direction based on slope
        if abs(slope) < 0.01:  # Very small change
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DECLINING
    
    def _get_target_value(self, metric_type: QualityMetricType) -> float:
        """Get target value for metric type"""
        targets = {
            QualityMetricType.DEFECT_DENSITY: 2.0,
            QualityMetricType.TEST_COVERAGE: 90.0,
            QualityMetricType.CODE_COMPLEXITY: 5.0,
            QualityMetricType.MAINTAINABILITY: 80.0,
            QualityMetricType.RELIABILITY: 95.0,
            QualityMetricType.TECHNICAL_DEBT: 5.0
        }
        return targets.get(metric_type, 0.0)
    
    def _calculate_time_to_target(self, model: LinearRegression, 
                                current_x: float, current_value: float, 
                                target_value: float) -> Optional[float]:
        """Calculate time to reach target value"""
        slope = model.coef_[0]
        
        if abs(slope) < 0.001:  # No significant trend
            return None
        
        # Calculate days to reach target
        days_to_target = (target_value - current_value) / slope
        
        # Return only if positive and reasonable (less than 2 years)
        if 0 < days_to_target < 730:
            return days_to_target
        
        return None

class AnomalyDetector:
    """Detect anomalies in quality metrics"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
    
    def train(self, historical_metrics: List[QualityMetric]):
        """Train anomaly detection model"""
        if len(historical_metrics) < 10:
            logger.warning("Insufficient data for anomaly detection training")
            return
        
        # Prepare feature matrix
        features = []
        for metric in historical_metrics:
            features.append([
                metric.value,
                (metric.value - metric.target_value) / metric.target_value,  # Normalized deviation
                metric.timestamp.hour,  # Time of day
                metric.timestamp.weekday()  # Day of week
            ])
        
        X = np.array(features)
        self.isolation_forest.fit(X)
        self.is_trained = True
        
        logger.info(f"Anomaly detector trained on {len(historical_metrics)} metrics")
    
    def detect_anomalies(self, metrics: List[QualityMetric]) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics"""
        if not self.is_trained:
            return []
        
        anomalies = []
        
        for metric in metrics:
            features = np.array([[
                metric.value,
                (metric.value - metric.target_value) / metric.target_value,
                metric.timestamp.hour,
                metric.timestamp.weekday()
            ]])
            
            anomaly_score = self.isolation_forest.decision_function(features)[0]
            is_anomaly = self.isolation_forest.predict(features)[0] == -1
            
            if is_anomaly:
                anomalies.append({
                    'metric_id': metric.id,
                    'metric_name': metric.name,
                    'value': metric.value,
                    'target_value': metric.target_value,
                    'anomaly_score': anomaly_score,
                    'severity': self._calculate_anomaly_severity(anomaly_score),
                    'timestamp': metric.timestamp
                })
        
        return anomalies
    
    def _calculate_anomaly_severity(self, anomaly_score: float) -> str:
        """Calculate anomaly severity based on score"""
        if anomaly_score < -0.5:
            return "critical"
        elif anomaly_score < -0.3:
            return "high"
        elif anomaly_score < -0.1:
            return "medium"
        else:
            return "low"

class InsightGenerator:
    """Generate quality insights and recommendations"""
    
    def __init__(self):
        self.insight_rules = self._load_insight_rules()
    
    def _load_insight_rules(self) -> List[Dict[str, Any]]:
        """Load insight generation rules"""
        return [
            {
                'id': 'low_test_coverage',
                'condition': lambda m: m.type == QualityMetricType.TEST_COVERAGE and m.value < 70,
                'title': 'Low Test Coverage Detected',
                'description': 'Test coverage is below recommended threshold',
                'severity': 'high',
                'category': 'testing',
                'recommendations': [
                    'Implement automated test generation for uncovered code',
                    'Add unit tests for critical business logic',
                    'Set up coverage gates in CI/CD pipeline'
                ]
            },
            {
                'id': 'high_complexity',
                'condition': lambda m: m.type == QualityMetricType.CODE_COMPLEXITY and m.value > 10,
                'title': 'High Code Complexity',
                'description': 'Code complexity exceeds maintainability threshold',
                'severity': 'medium',
                'category': 'maintainability',
                'recommendations': [
                    'Refactor complex functions into smaller units',
                    'Apply design patterns to reduce complexity',
                    'Add complexity monitoring to code review process'
                ]
            },
            {
                'id': 'increasing_defect_density',
                'condition': lambda m: m.type == QualityMetricType.DEFECT_DENSITY and m.value > 5,
                'title': 'High Defect Density',
                'description': 'Defect density indicates quality issues',
                'severity': 'high',
                'category': 'quality',
                'recommendations': [
                    'Implement stricter code review processes',
                    'Increase test coverage for defect-prone areas',
                    'Conduct root cause analysis of recent defects'
                ]
            },
            {
                'id': 'technical_debt_accumulation',
                'condition': lambda m: m.type == QualityMetricType.TECHNICAL_DEBT and m.value > 15,
                'title': 'Technical Debt Accumulation',
                'description': 'Technical debt is accumulating rapidly',
                'severity': 'medium',
                'category': 'maintainability',
                'recommendations': [
                    'Allocate dedicated time for debt reduction',
                    'Prioritize refactoring in sprint planning',
                    'Implement automated code quality gates'
                ]
            }
        ]
    
    def generate_insights(self, metrics: List[QualityMetric], 
                         trends: List[QualityTrend],
                         anomalies: List[Dict[str, Any]]) -> List[QualityInsight]:
        """Generate quality insights from metrics and trends"""
        insights = []
        
        # Generate insights from metric rules
        for metric in metrics:
            for rule in self.insight_rules:
                if rule['condition'](metric):
                    insight = QualityInsight(
                        id=f"{rule['id']}_{int(time.time())}",
                        title=rule['title'],
                        description=rule['description'],
                        severity=rule['severity'],
                        category=rule['category'],
                        affected_metrics=[metric.id],
                        recommendations=rule['recommendations'],
                        estimated_impact=self._calculate_impact(metric, rule),
                        effort_estimate=self._estimate_effort(rule),
                        priority_score=self._calculate_priority(metric, rule),
                        created_at=datetime.now()
                    )
                    insights.append(insight)
        
        # Generate insights from trends
        for trend in trends:
            if trend.direction == TrendDirection.DECLINING and trend.confidence > 0.7:
                insight = QualityInsight(
                    id=f"declining_trend_{trend.metric_type.value}_{int(time.time())}",
                    title=f"Declining {trend.metric_type.value.replace('_', ' ').title()} Trend",
                    description=f"Quality metric showing consistent decline with {trend.confidence:.1%} confidence",
                    severity="medium" if trend.change_rate > -5 else "high",
                    category="trend_analysis",
                    affected_metrics=[],
                    recommendations=[
                        "Investigate root causes of quality decline",
                        "Implement corrective measures immediately",
                        "Monitor trend closely for improvement"
                    ],
                    estimated_impact=abs(trend.change_rate) * 10,
                    effort_estimate="medium",
                    priority_score=trend.confidence * abs(trend.change_rate),
                    created_at=datetime.now()
                )
                insights.append(insight)
        
        # Generate insights from anomalies
        for anomaly in anomalies:
            if anomaly['severity'] in ['high', 'critical']:
                insight = QualityInsight(
                    id=f"anomaly_{anomaly['metric_id']}_{int(time.time())}",
                    title=f"Quality Anomaly: {anomaly['metric_name']}",
                    description=f"Unusual pattern detected in {anomaly['metric_name']}",
                    severity=anomaly['severity'],
                    category="anomaly",
                    affected_metrics=[anomaly['metric_id']],
                    recommendations=[
                        "Investigate data collection process",
                        "Verify measurement accuracy",
                        "Check for environmental factors"
                    ],
                    estimated_impact=50.0,
                    effort_estimate="low",
                    priority_score=abs(anomaly['anomaly_score']) * 100,
                    created_at=datetime.now()
                )
                insights.append(insight)
        
        # Sort insights by priority score
        insights.sort(key=lambda x: x.priority_score, reverse=True)
        
        return insights
    
    def _calculate_impact(self, metric: QualityMetric, rule: Dict[str, Any]) -> float:
        """Calculate estimated impact of the insight"""
        deviation = abs(metric.value - metric.target_value) / metric.target_value
        severity_multiplier = {'low': 1, 'medium': 2, 'high': 3, 'critical': 5}
        return deviation * severity_multiplier.get(rule['severity'], 1) * 20
    
    def _estimate_effort(self, rule: Dict[str, Any]) -> str:
        """Estimate effort required to address the insight"""
        effort_map = {
            'testing': 'medium',
            'maintainability': 'high',
            'quality': 'high',
            'performance': 'medium',
            'security': 'high'
        }
        return effort_map.get(rule['category'], 'medium')
    
    def _calculate_priority(self, metric: QualityMetric, rule: Dict[str, Any]) -> float:
        """Calculate priority score for the insight"""
        severity_score = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        deviation_score = abs(metric.value - metric.target_value) / metric.target_value
        return severity_score.get(rule['severity'], 1) * (1 + deviation_score) * 25

class QualityReportGenerator:
    """Generate comprehensive quality reports"""
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
    
    def generate_report(self, metrics: List[QualityMetric],
                       trends: List[QualityTrend],
                       insights: List[QualityInsight],
                       period_start: datetime,
                       period_end: datetime) -> QualityReport:
        """Generate comprehensive quality report"""
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(metrics)
        
        # Generate summary statistics
        summary = self._generate_summary(metrics, trends, insights)
        
        # Generate charts
        charts = self._generate_charts(metrics, trends)
        
        return QualityReport(
            id=f"quality_report_{int(time.time())}",
            title=f"Quality Report - {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
            period_start=period_start,
            period_end=period_end,
            overall_score=overall_score,
            metrics=metrics,
            trends=trends,
            insights=insights,
            summary=summary,
            charts=charts,
            created_at=datetime.now()
        )
    
    def _calculate_overall_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate overall quality score from metrics"""
        if not metrics:
            return 0.0
        
        scores = []
        weights = {
            QualityMetricType.TEST_COVERAGE: 0.25,
            QualityMetricType.DEFECT_DENSITY: 0.20,
            QualityMetricType.CODE_COMPLEXITY: 0.15,
            QualityMetricType.MAINTAINABILITY: 0.15,
            QualityMetricType.RELIABILITY: 0.15,
            QualityMetricType.TECHNICAL_DEBT: 0.10
        }
        
        for metric in metrics:
            # Normalize metric value to 0-100 scale
            if metric.type in [QualityMetricType.TEST_COVERAGE, QualityMetricType.RELIABILITY]:
                # Higher is better
                normalized_score = min(100, metric.value)
            elif metric.type in [QualityMetricType.DEFECT_DENSITY, QualityMetricType.CODE_COMPLEXITY, 
                               QualityMetricType.TECHNICAL_DEBT]:
                # Lower is better
                if metric.value <= metric.target_value:
                    normalized_score = 100
                else:
                    normalized_score = max(0, 100 - ((metric.value - metric.target_value) / metric.target_value) * 50)
            else:
                # Maintainability index (higher is better)
                normalized_score = min(100, metric.value)
            
            weight = weights.get(metric.type, 0.1)
            scores.append(normalized_score * weight)
        
        return sum(scores) / sum(weights.values()) if weights else 0.0
    
    def _generate_summary(self, metrics: List[QualityMetric],
                         trends: List[QualityTrend],
                         insights: List[QualityInsight]) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        # Metric statistics
        metric_stats = {
            'total_metrics': len(metrics),
            'metrics_at_target': sum(1 for m in metrics if m.value >= m.target_value),
            'metrics_below_warning': sum(1 for m in metrics if m.value < m.threshold_warning),
            'metrics_below_critical': sum(1 for m in metrics if m.value < m.threshold_critical)
        }
        
        # Trend statistics
        trend_stats = {
            'improving_trends': sum(1 for t in trends if t.direction == TrendDirection.IMPROVING),
            'declining_trends': sum(1 for t in trends if t.direction == TrendDirection.DECLINING),
            'stable_trends': sum(1 for t in trends if t.direction == TrendDirection.STABLE),
            'volatile_trends': sum(1 for t in trends if t.direction == TrendDirection.VOLATILE)
        }
        
        # Insight statistics
        insight_stats = {
            'total_insights': len(insights),
            'critical_insights': sum(1 for i in insights if i.severity == 'critical'),
            'high_insights': sum(1 for i in insights if i.severity == 'high'),
            'medium_insights': sum(1 for i in insights if i.severity == 'medium'),
            'low_insights': sum(1 for i in insights if i.severity == 'low')
        }
        
        return {
            'metrics': metric_stats,
            'trends': trend_stats,
            'insights': insight_stats,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_charts(self, metrics: List[QualityMetric],
                        trends: List[QualityTrend]) -> List[Dict[str, Any]]:
        """Generate charts for the report"""
        charts = []
        
        # Metrics overview chart
        charts.append(self.chart_generator.create_metrics_overview_chart(metrics))
        
        # Trend analysis chart
        charts.append(self.chart_generator.create_trend_analysis_chart(trends))
        
        # Quality score radar chart
        charts.append(self.chart_generator.create_quality_radar_chart(metrics))
        
        return charts

class ChartGenerator:
    """Generate charts for quality reports"""
    
    def create_metrics_overview_chart(self, metrics: List[QualityMetric]) -> Dict[str, Any]:
        """Create metrics overview bar chart"""
        metric_names = [m.name for m in metrics]
        metric_values = [m.value for m in metrics]
        target_values = [m.target_value for m in metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current Value',
            x=metric_names,
            y=metric_values,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Target Value',
            x=metric_names,
            y=target_values,
            marker_color='green',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Quality Metrics Overview',
            xaxis_title='Metrics',
            yaxis_title='Values',
            barmode='group'
        )
        
        return {
            'type': 'metrics_overview',
            'title': 'Quality Metrics Overview',
            'data': fig.to_json()
        }
    
    def create_trend_analysis_chart(self, trends: List[QualityTrend]) -> Dict[str, Any]:
        """Create trend analysis line chart"""
        fig = make_subplots(
            rows=len(trends),
            cols=1,
            subplot_titles=[f"{trend.metric_type.value.replace('_', ' ').title()}" for trend in trends]
        )
        
        for i, trend in enumerate(trends, 1):
            timestamps = [data[0] for data in trend.historical_data]
            values = [data[1] for data in trend.historical_data]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name=trend.metric_type.value,
                    line=dict(color='blue')
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title='Quality Trends Analysis',
            height=300 * len(trends)
        )
        
        return {
            'type': 'trend_analysis',
            'title': 'Quality Trends Analysis',
            'data': fig.to_json()
        }
    
    def create_quality_radar_chart(self, metrics: List[QualityMetric]) -> Dict[str, Any]:
        """Create quality radar chart"""
        categories = []
        values = []
        
        for metric in metrics:
            categories.append(metric.name)
            # Normalize to 0-100 scale
            if metric.type in [QualityMetricType.TEST_COVERAGE, QualityMetricType.RELIABILITY]:
                normalized_value = min(100, metric.value)
            else:
                normalized_value = max(0, 100 - (metric.value / metric.target_value) * 50)
            values.append(normalized_value)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Quality Score'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Quality Radar Chart'
        )
        
        return {
            'type': 'quality_radar',
            'title': 'Quality Radar Chart',
            'data': fig.to_json()
        }

class QualityMetricsAnalyzer:
    """Main quality metrics analyzer orchestrating all components"""
    
    def __init__(self):
        self.calculator = MetricsCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.insight_generator = InsightGenerator()
        self.report_generator = QualityReportGenerator()
        self.historical_data = defaultdict(list)
    
    async def analyze_quality(self, test_results: List[Dict[str, Any]],
                            code_metrics: Dict[str, Any],
                            defect_data: Dict[str, Any]) -> QualityReport:
        """Perform comprehensive quality analysis"""
        
        # Calculate current metrics
        current_metrics = await self._calculate_current_metrics(
            test_results, code_metrics, defect_data
        )
        
        # Store metrics in historical data
        for metric in current_metrics:
            self.historical_data[metric.type].append((metric.timestamp, metric.value))
        
        # Analyze trends
        trends = []
        for metric_type, data in self.historical_data.items():
            if len(data) >= 3:  # Need at least 3 data points for trend analysis
                trend = self.trend_analyzer.analyze_trend(data, metric_type)
                trends.append(trend)
        
        # Train anomaly detector if enough historical data
        all_historical_metrics = []
        for metric_list in self.historical_data.values():
            # Convert to QualityMetric objects for training
            for timestamp, value in metric_list:
                # Create dummy metric for training
                dummy_metric = QualityMetric(
                    id="training",
                    name="training",
                    type=list(self.historical_data.keys())[0],
                    value=value,
                    target_value=0,
                    threshold_warning=0,
                    threshold_critical=0,
                    unit="",
                    description="",
                    calculation_method="",
                    data_source="",
                    timestamp=timestamp,
                    tags=[]
                )
                all_historical_metrics.append(dummy_metric)
        
        if len(all_historical_metrics) >= 10:
            self.anomaly_detector.train(all_historical_metrics)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(current_metrics)
        
        # Generate insights
        insights = self.insight_generator.generate_insights(
            current_metrics, trends, anomalies
        )
        
        # Generate comprehensive report
        report = self.report_generator.generate_report(
            current_metrics,
            trends,
            insights,
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        return report
    
    async def _calculate_current_metrics(self, test_results: List[Dict[str, Any]],
                                       code_metrics: Dict[str, Any],
                                       defect_data: Dict[str, Any]) -> List[QualityMetric]:
        """Calculate current quality metrics from input data"""
        metrics = []
        
        # Test coverage metric
        if 'coverage' in test_results[0] if test_results else {}:
            coverage_data = test_results[0]['coverage']
            coverage_metric = self.calculator.calculate_test_coverage(
                coverage_data.get('covered_lines', 0),
                coverage_data.get('total_lines', 1)
            )
            metrics.append(coverage_metric)
        
        # Test pass rate metric
        if test_results:
            passed_tests = sum(1 for result in test_results if result.get('status') == 'passed')
            total_tests = len(test_results)
            pass_rate_metric = self.calculator.calculate_test_pass_rate(passed_tests, total_tests)
            metrics.append(pass_rate_metric)
        
        # Code complexity metric
        if 'complexity_scores' in code_metrics:
            complexity_metric = self.calculator.calculate_code_complexity(
                code_metrics['complexity_scores']
            )
            metrics.append(complexity_metric)
        
        # Defect density metric
        if 'defect_count' in defect_data and 'lines_of_code' in code_metrics:
            defect_metric = self.calculator.calculate_defect_density(
                defect_data['defect_count'],
                code_metrics['lines_of_code']
            )
            metrics.append(defect_metric)
        
        # Maintainability index
        if all(key in code_metrics for key in ['halstead_volume', 'cyclomatic_complexity', 'lines_of_code']):
            maintainability_metric = self.calculator.calculate_maintainability_index(
                code_metrics['halstead_volume'],
                code_metrics['cyclomatic_complexity'],
                code_metrics['lines_of_code']
            )
            metrics.append(maintainability_metric)
        
        # Technical debt ratio
        if 'technical_debt_hours' in defect_data and 'development_hours' in defect_data:
            debt_metric = self.calculator.calculate_technical_debt_ratio(
                defect_data['technical_debt_hours'],
                defect_data['development_hours']
            )
            metrics.append(debt_metric)
        
        return metrics
    
    def export_report(self, report: QualityReport, format_type: str = 'json') -> str:
        """Export quality report in specified format"""
        if format_type == 'json':
            return json.dumps(asdict(report), default=str, indent=2)
        elif format_type == 'html':
            return self._export_html_report(report)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_html_report(self, report: QualityReport) -> str:
        """Export report as HTML"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .insight {{ margin: 10px 0; padding: 15px; background-color: #fff3cd; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p>Overall Quality Score: {report.overall_score:.1f}/100</p>
                <p>Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}</p>
            </div>
            
            <h2>Quality Metrics</h2>
            {self._generate_metrics_html(report.metrics)}
            
            <h2>Key Insights</h2>
            {self._generate_insights_html(report.insights)}
            
            <h2>Summary</h2>
            <pre>{json.dumps(report.summary, indent=2)}</pre>
        </body>
        </html>
        """
        return html_template
    
    def _generate_metrics_html(self, metrics: List[QualityMetric]) -> str:
        """Generate HTML for metrics section"""
        html = ""
        for metric in metrics:
            status_color = "green" if metric.value >= metric.target_value else "orange" if metric.value >= metric.threshold_warning else "red"
            html += f"""
            <div class="metric" style="border-left: 5px solid {status_color};">
                <h3>{metric.name}</h3>
                <p><strong>Value:</strong> {metric.value:.2f} {metric.unit}</p>
                <p><strong>Target:</strong> {metric.target_value:.2f} {metric.unit}</p>
                <p><strong>Description:</strong> {metric.description}</p>
            </div>
            """
        return html
    
    def _generate_insights_html(self, insights: List[QualityInsight]) -> str:
        """Generate HTML for insights section"""
        html = ""
        for insight in insights:
            severity_color = {"low": "#d4edda", "medium": "#fff3cd", "high": "#f8d7da", "critical": "#f5c6cb"}
            html += f"""
            <div class="insight" style="background-color: {severity_color.get(insight.severity, '#fff3cd')};">
                <h3>{insight.title}</h3>
                <p><strong>Severity:</strong> {insight.severity.upper()}</p>
                <p>{insight.description}</p>
                <p><strong>Recommendations:</strong></p>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in insight.recommendations)}
                </ul>
            </div>
            """
        return html

# Example usage and testing
async def main():
    """Example usage of the quality metrics analyzer"""
    analyzer = QualityMetricsAnalyzer()
    
    # Example test results
    test_results = [
        {'status': 'passed', 'coverage': {'covered_lines': 850, 'total_lines': 1000}},
        {'status': 'passed'},
        {'status': 'failed'},
        {'status': 'passed'}
    ]
    
    # Example code metrics
    code_metrics = {
        'complexity_scores': [3, 5, 8, 2, 12, 4, 6],
        'lines_of_code': 1000,
        'halstead_volume': 2500,
        'cyclomatic_complexity': 6
    }
    
    # Example defect data
    defect_data = {
        'defect_count': 3,
        'technical_debt_hours': 40,
        'development_hours': 800
    }
    
    # Perform quality analysis
    report = await analyzer.analyze_quality(test_results, code_metrics, defect_data)
    
    print("Quality Analysis Report:")
    print(f"Overall Score: {report.overall_score:.1f}/100")
    print(f"Total Metrics: {len(report.metrics)}")
    print(f"Total Insights: {len(report.insights)}")
    
    # Export report
    json_report = analyzer.export_report(report, 'json')
    print("\nJSON Report (first 500 chars):")
    print(json_report[:500] + "..." if len(json_report) > 500 else json_report)

if __name__ == "__main__":
    asyncio.run(main())

