"""
Data Quality Monitor for Nexus Architect
Implements comprehensive data quality monitoring, validation, and alerting
with real-time dashboards and automated quality assessment.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid
import statistics

# Data processing and validation
import pandas as pd
import numpy as np
from scipy import stats
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.dataset import PandasDataset

# Database and caching
import psycopg2
from psycopg2.extras import RealDictCursor
import redis

# Web framework
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
quality_checks = Counter('data_quality_checks_total', 'Total quality checks', ['check_type', 'status'])
quality_score = Gauge('data_quality_score', 'Data quality score', ['source', 'dimension'])
quality_issues = Counter('data_quality_issues_total', 'Total quality issues', ['issue_type', 'severity'])
validation_duration = Histogram('data_validation_duration_seconds', 'Validation duration', ['validation_type'])

@dataclass
class QualityDimension:
    """Data quality dimension definition"""
    name: str
    description: str
    weight: float
    thresholds: Dict[str, float]  # warning, critical
    metrics: List[str]

@dataclass
class QualityMetric:
    """Individual quality metric"""
    metric_id: str
    metric_name: str
    metric_type: str  # completeness, accuracy, consistency, timeliness, validity
    value: float
    threshold_warning: float
    threshold_critical: float
    status: str  # pass, warning, critical
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class QualityIssue:
    """Data quality issue"""
    issue_id: str
    issue_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_records: int
    source_table: str
    source_column: Optional[str]
    detection_rule: str
    suggested_action: str
    created_at: datetime
    resolved_at: Optional[datetime]

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    report_id: str
    source_name: str
    overall_score: float
    dimension_scores: Dict[str, float]
    metrics: List[QualityMetric]
    issues: List[QualityIssue]
    recommendations: List[str]
    generated_at: datetime

class DataProfiler:
    """Data profiling and statistical analysis"""
    
    def __init__(self):
        self.profile_cache = {}
    
    def profile_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        try:
            profile = {
                'dataset_name': dataset_name,
                'row_count': len(df),
                'column_count': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'columns': {},
                'correlations': {},
                'duplicates': {},
                'missing_data': {},
                'data_types': {},
                'statistical_summary': {}
            }
            
            # Column-level profiling
            for column in df.columns:
                col_profile = self._profile_column(df[column], column)
                profile['columns'][column] = col_profile
            
            # Data type analysis
            profile['data_types'] = {
                'numeric': len(df.select_dtypes(include=[np.number]).columns),
                'categorical': len(df.select_dtypes(include=['object']).columns),
                'datetime': len(df.select_dtypes(include=['datetime64']).columns),
                'boolean': len(df.select_dtypes(include=['bool']).columns)
            }
            
            # Missing data analysis
            missing_counts = df.isnull().sum()
            profile['missing_data'] = {
                'total_missing': missing_counts.sum(),
                'missing_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100,
                'columns_with_missing': missing_counts[missing_counts > 0].to_dict()
            }
            
            # Duplicate analysis
            duplicate_rows = df.duplicated().sum()
            profile['duplicates'] = {
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0,
                'unique_rows': len(df) - duplicate_rows
            }
            
            # Correlation analysis for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                correlation_matrix = numeric_df.corr()
                profile['correlations'] = {
                    'high_correlations': self._find_high_correlations(correlation_matrix),
                    'correlation_matrix': correlation_matrix.to_dict()
                }
            
            # Statistical summary
            profile['statistical_summary'] = df.describe(include='all').to_dict()
            
            return profile
            
        except Exception as e:
            logger.error(f"Data profiling failed: {e}")
            return {}
    
    def _profile_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Profile individual column"""
        try:
            profile = {
                'name': column_name,
                'dtype': str(series.dtype),
                'non_null_count': series.count(),
                'null_count': series.isnull().sum(),
                'null_percentage': (series.isnull().sum() / len(series)) * 100,
                'unique_count': series.nunique(),
                'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0
            }
            
            # Type-specific profiling
            if pd.api.types.is_numeric_dtype(series):
                profile.update(self._profile_numeric_column(series))
            elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                profile.update(self._profile_text_column(series))
            elif pd.api.types.is_datetime64_any_dtype(series):
                profile.update(self._profile_datetime_column(series))
            
            return profile
            
        except Exception as e:
            logger.error(f"Column profiling failed for {column_name}: {e}")
            return {'name': column_name, 'error': str(e)}
    
    def _profile_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric column"""
        try:
            non_null_series = series.dropna()
            
            if len(non_null_series) == 0:
                return {'numeric_stats': 'no_data'}
            
            return {
                'min': non_null_series.min(),
                'max': non_null_series.max(),
                'mean': non_null_series.mean(),
                'median': non_null_series.median(),
                'std': non_null_series.std(),
                'variance': non_null_series.var(),
                'skewness': stats.skew(non_null_series),
                'kurtosis': stats.kurtosis(non_null_series),
                'quartiles': {
                    'q1': non_null_series.quantile(0.25),
                    'q2': non_null_series.quantile(0.5),
                    'q3': non_null_series.quantile(0.75)
                },
                'outliers': self._detect_outliers(non_null_series),
                'zeros': (non_null_series == 0).sum(),
                'negatives': (non_null_series < 0).sum()
            }
            
        except Exception as e:
            logger.error(f"Numeric profiling failed: {e}")
            return {'numeric_stats': 'error'}
    
    def _profile_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile text column"""
        try:
            non_null_series = series.dropna().astype(str)
            
            if len(non_null_series) == 0:
                return {'text_stats': 'no_data'}
            
            lengths = non_null_series.str.len()
            
            return {
                'min_length': lengths.min(),
                'max_length': lengths.max(),
                'avg_length': lengths.mean(),
                'median_length': lengths.median(),
                'empty_strings': (non_null_series == '').sum(),
                'whitespace_only': non_null_series.str.strip().eq('').sum(),
                'most_common': non_null_series.value_counts().head(10).to_dict(),
                'pattern_analysis': self._analyze_text_patterns(non_null_series)
            }
            
        except Exception as e:
            logger.error(f"Text profiling failed: {e}")
            return {'text_stats': 'error'}
    
    def _profile_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column"""
        try:
            non_null_series = series.dropna()
            
            if len(non_null_series) == 0:
                return {'datetime_stats': 'no_data'}
            
            return {
                'min_date': non_null_series.min(),
                'max_date': non_null_series.max(),
                'date_range_days': (non_null_series.max() - non_null_series.min()).days,
                'future_dates': (non_null_series > datetime.now()).sum(),
                'weekend_dates': non_null_series.dt.dayofweek.isin([5, 6]).sum(),
                'year_distribution': non_null_series.dt.year.value_counts().to_dict(),
                'month_distribution': non_null_series.dt.month.value_counts().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Datetime profiling failed: {e}")
            return {'datetime_stats': 'error'}
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        try:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            return {
                'count': len(outliers),
                'percentage': (len(outliers) / len(series)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': outliers.tolist()[:20]  # Limit to first 20
            }
            
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return {'count': 0, 'error': str(e)}
    
    def _analyze_text_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text patterns"""
        try:
            import re
            
            patterns = {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b',
                'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                'numeric': r'^\d+$',
                'alphanumeric': r'^[a-zA-Z0-9]+$',
                'uppercase': r'^[A-Z\s]+$',
                'lowercase': r'^[a-z\s]+$'
            }
            
            pattern_matches = {}
            for pattern_name, pattern in patterns.items():
                matches = series.str.contains(pattern, regex=True, na=False).sum()
                pattern_matches[pattern_name] = {
                    'count': matches,
                    'percentage': (matches / len(series)) * 100
                }
            
            return pattern_matches
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {}
    
    def _find_high_correlations(self, correlation_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find highly correlated column pairs"""
        try:
            high_correlations = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    correlation = correlation_matrix.iloc[i, j]
                    
                    if abs(correlation) >= threshold:
                        high_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': correlation,
                            'strength': 'strong' if abs(correlation) >= 0.9 else 'moderate'
                        })
            
            return high_correlations
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return []

class QualityValidator:
    """Data quality validation engine"""
    
    def __init__(self):
        self.quality_dimensions = {
            'completeness': QualityDimension(
                name='completeness',
                description='Percentage of non-null values',
                weight=0.25,
                thresholds={'warning': 0.95, 'critical': 0.90},
                metrics=['null_percentage', 'empty_percentage']
            ),
            'accuracy': QualityDimension(
                name='accuracy',
                description='Correctness of data values',
                weight=0.25,
                thresholds={'warning': 0.95, 'critical': 0.90},
                metrics=['format_compliance', 'range_compliance', 'pattern_compliance']
            ),
            'consistency': QualityDimension(
                name='consistency',
                description='Uniformity across data sources',
                weight=0.20,
                thresholds={'warning': 0.95, 'critical': 0.90},
                metrics=['duplicate_percentage', 'format_consistency']
            ),
            'timeliness': QualityDimension(
                name='timeliness',
                description='Currency and freshness of data',
                weight=0.15,
                thresholds={'warning': 0.95, 'critical': 0.90},
                metrics=['age_compliance', 'update_frequency']
            ),
            'validity': QualityDimension(
                name='validity',
                description='Conformance to business rules',
                weight=0.15,
                thresholds={'warning': 0.95, 'critical': 0.90},
                metrics=['business_rule_compliance', 'referential_integrity']
            )
        }
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str, 
                        validation_rules: Dict[str, Any] = None) -> QualityReport:
        """Validate dataset against quality dimensions"""
        try:
            with validation_duration.labels(validation_type='full').time():
                metrics = []
                issues = []
                dimension_scores = {}
                
                # Validate each quality dimension
                for dimension_name, dimension in self.quality_dimensions.items():
                    dimension_metrics, dimension_issues = self._validate_dimension(
                        df, dataset_name, dimension, validation_rules
                    )
                    
                    metrics.extend(dimension_metrics)
                    issues.extend(dimension_issues)
                    
                    # Calculate dimension score
                    if dimension_metrics:
                        dimension_score = sum(m.value for m in dimension_metrics) / len(dimension_metrics)
                        dimension_scores[dimension_name] = dimension_score
                        quality_score.labels(source=dataset_name, dimension=dimension_name).set(dimension_score)
                
                # Calculate overall score
                overall_score = sum(
                    score * self.quality_dimensions[dim].weight 
                    for dim, score in dimension_scores.items()
                )
                
                # Generate recommendations
                recommendations = self._generate_recommendations(metrics, issues)
                
                # Update metrics
                quality_checks.labels(check_type='full_validation', status='completed').inc()
                
                report = QualityReport(
                    report_id=str(uuid.uuid4()),
                    source_name=dataset_name,
                    overall_score=overall_score,
                    dimension_scores=dimension_scores,
                    metrics=metrics,
                    issues=issues,
                    recommendations=recommendations,
                    generated_at=datetime.utcnow()
                )
                
                return report
                
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            quality_checks.labels(check_type='full_validation', status='error').inc()
            return QualityReport(
                report_id=str(uuid.uuid4()),
                source_name=dataset_name,
                overall_score=0.0,
                dimension_scores={},
                metrics=[],
                issues=[QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    issue_type='validation_error',
                    severity='critical',
                    description=f"Validation failed: {str(e)}",
                    affected_records=len(df) if df is not None else 0,
                    source_table=dataset_name,
                    source_column=None,
                    detection_rule='system_validation',
                    suggested_action='Check data format and validation rules',
                    created_at=datetime.utcnow(),
                    resolved_at=None
                )],
                recommendations=[],
                generated_at=datetime.utcnow()
            )
    
    def _validate_dimension(self, df: pd.DataFrame, dataset_name: str, 
                          dimension: QualityDimension, validation_rules: Dict[str, Any]) -> Tuple[List[QualityMetric], List[QualityIssue]]:
        """Validate specific quality dimension"""
        metrics = []
        issues = []
        
        try:
            if dimension.name == 'completeness':
                metrics, issues = self._validate_completeness(df, dataset_name, dimension)
            elif dimension.name == 'accuracy':
                metrics, issues = self._validate_accuracy(df, dataset_name, dimension, validation_rules)
            elif dimension.name == 'consistency':
                metrics, issues = self._validate_consistency(df, dataset_name, dimension)
            elif dimension.name == 'timeliness':
                metrics, issues = self._validate_timeliness(df, dataset_name, dimension)
            elif dimension.name == 'validity':
                metrics, issues = self._validate_validity(df, dataset_name, dimension, validation_rules)
            
        except Exception as e:
            logger.error(f"Dimension validation failed for {dimension.name}: {e}")
            
        return metrics, issues
    
    def _validate_completeness(self, df: pd.DataFrame, dataset_name: str, 
                             dimension: QualityDimension) -> Tuple[List[QualityMetric], List[QualityIssue]]:
        """Validate data completeness"""
        metrics = []
        issues = []
        
        try:
            # Overall completeness
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            completeness_score = 1.0 - (null_cells / total_cells) if total_cells > 0 else 0.0
            
            status = 'pass'
            if completeness_score < dimension.thresholds['critical']:
                status = 'critical'
            elif completeness_score < dimension.thresholds['warning']:
                status = 'warning'
            
            metrics.append(QualityMetric(
                metric_id=str(uuid.uuid4()),
                metric_name='overall_completeness',
                metric_type='completeness',
                value=completeness_score,
                threshold_warning=dimension.thresholds['warning'],
                threshold_critical=dimension.thresholds['critical'],
                status=status,
                details={
                    'total_cells': total_cells,
                    'null_cells': null_cells,
                    'null_percentage': (null_cells / total_cells) * 100 if total_cells > 0 else 0
                },
                timestamp=datetime.utcnow()
            ))
            
            # Column-level completeness
            for column in df.columns:
                null_count = df[column].isnull().sum()
                column_completeness = 1.0 - (null_count / len(df)) if len(df) > 0 else 0.0
                
                if column_completeness < dimension.thresholds['critical']:
                    issues.append(QualityIssue(
                        issue_id=str(uuid.uuid4()),
                        issue_type='completeness',
                        severity='high' if column_completeness < 0.5 else 'medium',
                        description=f"Column '{column}' has {null_count} null values ({(null_count/len(df)*100):.1f}%)",
                        affected_records=null_count,
                        source_table=dataset_name,
                        source_column=column,
                        detection_rule='completeness_threshold',
                        suggested_action='Investigate data source and collection process',
                        created_at=datetime.utcnow(),
                        resolved_at=None
                    ))
                    quality_issues.labels(issue_type='completeness', severity='high').inc()
            
        except Exception as e:
            logger.error(f"Completeness validation failed: {e}")
        
        return metrics, issues
    
    def _validate_accuracy(self, df: pd.DataFrame, dataset_name: str, 
                         dimension: QualityDimension, validation_rules: Dict[str, Any]) -> Tuple[List[QualityMetric], List[QualityIssue]]:
        """Validate data accuracy"""
        metrics = []
        issues = []
        
        try:
            if not validation_rules:
                validation_rules = {}
            
            # Format compliance
            format_rules = validation_rules.get('format_rules', {})
            format_compliance_scores = []
            
            for column, rule in format_rules.items():
                if column in df.columns:
                    pattern = rule.get('pattern')
                    if pattern:
                        valid_count = df[column].astype(str).str.contains(pattern, regex=True, na=False).sum()
                        total_count = df[column].count()
                        compliance_score = valid_count / total_count if total_count > 0 else 0.0
                        format_compliance_scores.append(compliance_score)
                        
                        if compliance_score < dimension.thresholds['warning']:
                            issues.append(QualityIssue(
                                issue_id=str(uuid.uuid4()),
                                issue_type='format_compliance',
                                severity='medium',
                                description=f"Column '{column}' format compliance: {compliance_score:.2%}",
                                affected_records=total_count - valid_count,
                                source_table=dataset_name,
                                source_column=column,
                                detection_rule=f"format_pattern: {pattern}",
                                suggested_action='Review and standardize data format',
                                created_at=datetime.utcnow(),
                                resolved_at=None
                            ))
                            quality_issues.labels(issue_type='format_compliance', severity='medium').inc()
            
            # Overall format compliance
            if format_compliance_scores:
                avg_format_compliance = statistics.mean(format_compliance_scores)
                
                status = 'pass'
                if avg_format_compliance < dimension.thresholds['critical']:
                    status = 'critical'
                elif avg_format_compliance < dimension.thresholds['warning']:
                    status = 'warning'
                
                metrics.append(QualityMetric(
                    metric_id=str(uuid.uuid4()),
                    metric_name='format_compliance',
                    metric_type='accuracy',
                    value=avg_format_compliance,
                    threshold_warning=dimension.thresholds['warning'],
                    threshold_critical=dimension.thresholds['critical'],
                    status=status,
                    details={
                        'rules_checked': len(format_rules),
                        'compliance_scores': format_compliance_scores
                    },
                    timestamp=datetime.utcnow()
                ))
            
        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}")
        
        return metrics, issues
    
    def _validate_consistency(self, df: pd.DataFrame, dataset_name: str, 
                            dimension: QualityDimension) -> Tuple[List[QualityMetric], List[QualityIssue]]:
        """Validate data consistency"""
        metrics = []
        issues = []
        
        try:
            # Duplicate detection
            duplicate_count = df.duplicated().sum()
            duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
            consistency_score = 1.0 - (duplicate_count / len(df)) if len(df) > 0 else 1.0
            
            status = 'pass'
            if consistency_score < dimension.thresholds['critical']:
                status = 'critical'
            elif consistency_score < dimension.thresholds['warning']:
                status = 'warning'
            
            metrics.append(QualityMetric(
                metric_id=str(uuid.uuid4()),
                metric_name='duplicate_consistency',
                metric_type='consistency',
                value=consistency_score,
                threshold_warning=dimension.thresholds['warning'],
                threshold_critical=dimension.thresholds['critical'],
                status=status,
                details={
                    'duplicate_count': duplicate_count,
                    'duplicate_percentage': duplicate_percentage,
                    'unique_count': len(df) - duplicate_count
                },
                timestamp=datetime.utcnow()
            ))
            
            if duplicate_count > 0:
                issues.append(QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    issue_type='duplicates',
                    severity='medium' if duplicate_percentage < 5 else 'high',
                    description=f"Found {duplicate_count} duplicate records ({duplicate_percentage:.1f}%)",
                    affected_records=duplicate_count,
                    source_table=dataset_name,
                    source_column=None,
                    detection_rule='duplicate_detection',
                    suggested_action='Remove or merge duplicate records',
                    created_at=datetime.utcnow(),
                    resolved_at=None
                ))
                quality_issues.labels(issue_type='duplicates', severity='medium').inc()
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
        
        return metrics, issues
    
    def _validate_timeliness(self, df: pd.DataFrame, dataset_name: str, 
                           dimension: QualityDimension) -> Tuple[List[QualityMetric], List[QualityIssue]]:
        """Validate data timeliness"""
        metrics = []
        issues = []
        
        try:
            # Look for timestamp columns
            timestamp_columns = df.select_dtypes(include=['datetime64']).columns
            
            if len(timestamp_columns) > 0:
                for col in timestamp_columns:
                    # Calculate data freshness
                    latest_timestamp = df[col].max()
                    if pd.notna(latest_timestamp):
                        age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
                        
                        # Assume data should be updated within 24 hours
                        timeliness_score = max(0, 1.0 - (age_hours / 24))
                        
                        status = 'pass'
                        if timeliness_score < dimension.thresholds['critical']:
                            status = 'critical'
                        elif timeliness_score < dimension.thresholds['warning']:
                            status = 'warning'
                        
                        metrics.append(QualityMetric(
                            metric_id=str(uuid.uuid4()),
                            metric_name=f'timeliness_{col}',
                            metric_type='timeliness',
                            value=timeliness_score,
                            threshold_warning=dimension.thresholds['warning'],
                            threshold_critical=dimension.thresholds['critical'],
                            status=status,
                            details={
                                'latest_timestamp': latest_timestamp.isoformat(),
                                'age_hours': age_hours,
                                'column': col
                            },
                            timestamp=datetime.utcnow()
                        ))
                        
                        if age_hours > 48:  # Data older than 48 hours
                            issues.append(QualityIssue(
                                issue_id=str(uuid.uuid4()),
                                issue_type='timeliness',
                                severity='high' if age_hours > 168 else 'medium',  # 1 week threshold
                                description=f"Data in column '{col}' is {age_hours:.1f} hours old",
                                affected_records=len(df),
                                source_table=dataset_name,
                                source_column=col,
                                detection_rule='timeliness_threshold',
                                suggested_action='Update data source or refresh schedule',
                                created_at=datetime.utcnow(),
                                resolved_at=None
                            ))
                            quality_issues.labels(issue_type='timeliness', severity='high').inc()
            
        except Exception as e:
            logger.error(f"Timeliness validation failed: {e}")
        
        return metrics, issues
    
    def _validate_validity(self, df: pd.DataFrame, dataset_name: str, 
                         dimension: QualityDimension, validation_rules: Dict[str, Any]) -> Tuple[List[QualityMetric], List[QualityIssue]]:
        """Validate data validity against business rules"""
        metrics = []
        issues = []
        
        try:
            if not validation_rules:
                validation_rules = {}
            
            business_rules = validation_rules.get('business_rules', {})
            validity_scores = []
            
            for rule_name, rule_config in business_rules.items():
                rule_type = rule_config.get('type')
                column = rule_config.get('column')
                
                if column not in df.columns:
                    continue
                
                valid_count = 0
                total_count = df[column].count()
                
                if rule_type == 'range':
                    min_val = rule_config.get('min')
                    max_val = rule_config.get('max')
                    if min_val is not None and max_val is not None:
                        valid_count = ((df[column] >= min_val) & (df[column] <= max_val)).sum()
                
                elif rule_type == 'enum':
                    allowed_values = rule_config.get('values', [])
                    valid_count = df[column].isin(allowed_values).sum()
                
                elif rule_type == 'custom':
                    # Custom validation logic would go here
                    valid_count = total_count  # Placeholder
                
                if total_count > 0:
                    validity_score = valid_count / total_count
                    validity_scores.append(validity_score)
                    
                    if validity_score < dimension.thresholds['warning']:
                        issues.append(QualityIssue(
                            issue_id=str(uuid.uuid4()),
                            issue_type='business_rule_violation',
                            severity='high' if validity_score < 0.5 else 'medium',
                            description=f"Business rule '{rule_name}' violation in column '{column}': {validity_score:.2%} compliance",
                            affected_records=total_count - valid_count,
                            source_table=dataset_name,
                            source_column=column,
                            detection_rule=rule_name,
                            suggested_action='Review business rules and data collection process',
                            created_at=datetime.utcnow(),
                            resolved_at=None
                        ))
                        quality_issues.labels(issue_type='business_rule_violation', severity='high').inc()
            
            # Overall validity score
            if validity_scores:
                avg_validity = statistics.mean(validity_scores)
                
                status = 'pass'
                if avg_validity < dimension.thresholds['critical']:
                    status = 'critical'
                elif avg_validity < dimension.thresholds['warning']:
                    status = 'warning'
                
                metrics.append(QualityMetric(
                    metric_id=str(uuid.uuid4()),
                    metric_name='business_rule_compliance',
                    metric_type='validity',
                    value=avg_validity,
                    threshold_warning=dimension.thresholds['warning'],
                    threshold_critical=dimension.thresholds['critical'],
                    status=status,
                    details={
                        'rules_checked': len(business_rules),
                        'validity_scores': validity_scores
                    },
                    timestamp=datetime.utcnow()
                ))
            
        except Exception as e:
            logger.error(f"Validity validation failed: {e}")
        
        return metrics, issues
    
    def _generate_recommendations(self, metrics: List[QualityMetric], issues: List[QualityIssue]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        try:
            # Analyze metrics for patterns
            critical_metrics = [m for m in metrics if m.status == 'critical']
            warning_metrics = [m for m in metrics if m.status == 'warning']
            
            if critical_metrics:
                recommendations.append("Address critical quality issues immediately to prevent data reliability problems")
            
            if warning_metrics:
                recommendations.append("Monitor warning-level metrics and implement preventive measures")
            
            # Issue-specific recommendations
            issue_types = {}
            for issue in issues:
                if issue.issue_type not in issue_types:
                    issue_types[issue.issue_type] = 0
                issue_types[issue.issue_type] += 1
            
            if 'completeness' in issue_types:
                recommendations.append("Implement data validation at source to improve completeness")
            
            if 'duplicates' in issue_types:
                recommendations.append("Establish deduplication processes and unique key constraints")
            
            if 'format_compliance' in issue_types:
                recommendations.append("Standardize data formats and implement input validation")
            
            if 'timeliness' in issue_types:
                recommendations.append("Review data refresh schedules and implement real-time updates where needed")
            
            if 'business_rule_violation' in issue_types:
                recommendations.append("Review and update business rules, provide training on data entry standards")
            
            # General recommendations
            if len(issues) > 10:
                recommendations.append("Consider implementing automated data quality monitoring and alerting")
            
            if not recommendations:
                recommendations.append("Data quality is good - maintain current processes and monitoring")
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Unable to generate specific recommendations - review quality report manually")
        
        return recommendations

class DataQualityMonitor:
    """Main data quality monitoring service"""
    
    def __init__(self):
        self.profiler = DataProfiler()
        self.validator = QualityValidator()
        
        # Database connections
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'nexus_architect'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 2)),
            decode_responses=True
        )
        
        # Processing executor
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Quality reports storage
        self.quality_reports = {}
    
    def get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None
    
    def monitor_dataset_quality(self, dataset_name: str, validation_rules: Dict[str, Any] = None) -> QualityReport:
        """Monitor quality of a specific dataset"""
        try:
            # Load dataset
            df = self._load_dataset(dataset_name)
            if df is None:
                raise Exception(f"Failed to load dataset: {dataset_name}")
            
            # Generate profile
            profile = self.profiler.profile_dataset(df, dataset_name)
            
            # Validate quality
            quality_report = self.validator.validate_dataset(df, dataset_name, validation_rules)
            
            # Store report
            self.quality_reports[quality_report.report_id] = quality_report
            self._store_quality_report(quality_report)
            
            # Cache results
            cache_key = f"quality_report:{dataset_name}"
            self.redis_client.setex(cache_key, 3600, json.dumps(asdict(quality_report), default=str))
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Quality monitoring failed for {dataset_name}: {e}")
            raise
    
    def _load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load dataset from database"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return None
            
            # Query based on dataset name
            query = f"SELECT * FROM {dataset_name} LIMIT 10000"  # Limit for performance
            df = pd.read_sql(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            return None
    
    def _store_quality_report(self, report: QualityReport):
        """Store quality report in database"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return
            
            with conn.cursor() as cursor:
                # Store main report
                cursor.execute("""
                    INSERT INTO quality_reports 
                    (report_id, source_name, overall_score, dimension_scores, 
                     recommendations, generated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (report_id) DO UPDATE SET
                    overall_score = EXCLUDED.overall_score,
                    dimension_scores = EXCLUDED.dimension_scores
                """, (
                    report.report_id, report.source_name, report.overall_score,
                    json.dumps(report.dimension_scores), json.dumps(report.recommendations),
                    report.generated_at
                ))
                
                # Store metrics
                for metric in report.metrics:
                    cursor.execute("""
                        INSERT INTO quality_metrics 
                        (metric_id, report_id, metric_name, metric_type, value,
                         threshold_warning, threshold_critical, status, details, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (metric_id) DO UPDATE SET
                        value = EXCLUDED.value,
                        status = EXCLUDED.status
                    """, (
                        metric.metric_id, report.report_id, metric.metric_name,
                        metric.metric_type, metric.value, metric.threshold_warning,
                        metric.threshold_critical, metric.status,
                        json.dumps(metric.details), metric.timestamp
                    ))
                
                # Store issues
                for issue in report.issues:
                    cursor.execute("""
                        INSERT INTO quality_issues 
                        (issue_id, report_id, issue_type, severity, description,
                         affected_records, source_table, source_column, detection_rule,
                         suggested_action, created_at, resolved_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (issue_id) DO UPDATE SET
                        severity = EXCLUDED.severity,
                        description = EXCLUDED.description
                    """, (
                        issue.issue_id, report.report_id, issue.issue_type,
                        issue.severity, issue.description, issue.affected_records,
                        issue.source_table, issue.source_column, issue.detection_rule,
                        issue.suggested_action, issue.created_at, issue.resolved_at
                    ))
                
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store quality report: {e}")
    
    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get data for quality dashboard"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return {}
            
            dashboard_data = {}
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Overall quality scores
                cursor.execute("""
                    SELECT source_name, overall_score, generated_at
                    FROM quality_reports 
                    ORDER BY generated_at DESC 
                    LIMIT 10
                """)
                dashboard_data['recent_scores'] = [dict(row) for row in cursor.fetchall()]
                
                # Quality trends
                cursor.execute("""
                    SELECT source_name, 
                           AVG(overall_score) as avg_score,
                           COUNT(*) as report_count
                    FROM quality_reports 
                    WHERE generated_at >= NOW() - INTERVAL '7 days'
                    GROUP BY source_name
                """)
                dashboard_data['quality_trends'] = [dict(row) for row in cursor.fetchall()]
                
                # Active issues
                cursor.execute("""
                    SELECT issue_type, severity, COUNT(*) as count
                    FROM quality_issues 
                    WHERE resolved_at IS NULL
                    GROUP BY issue_type, severity
                    ORDER BY count DESC
                """)
                dashboard_data['active_issues'] = [dict(row) for row in cursor.fetchall()]
                
                # Dimension performance
                cursor.execute("""
                    SELECT metric_type, AVG(value) as avg_value, COUNT(*) as count
                    FROM quality_metrics 
                    WHERE timestamp >= NOW() - INTERVAL '24 hours'
                    GROUP BY metric_type
                """)
                dashboard_data['dimension_performance'] = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}

# Flask application
app = Flask(__name__)
CORS(app)

# Initialize monitor
monitor = DataQualityMonitor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_conn = monitor.get_db_connection()
        db_status = "connected" if db_conn else "disconnected"
        if db_conn:
            db_conn.close()
        
        # Check Redis connection
        try:
            monitor.redis_client.ping()
            redis_status = "connected"
        except:
            redis_status = "disconnected"
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": db_status,
                "redis": redis_status
            }
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/v1/quality/monitor/<dataset_name>', methods=['POST'])
def monitor_quality(dataset_name):
    """Monitor dataset quality"""
    try:
        data = request.get_json() or {}
        validation_rules = data.get('validation_rules', {})
        
        report = monitor.monitor_dataset_quality(dataset_name, validation_rules)
        
        return jsonify(asdict(report))
        
    except Exception as e:
        logger.error(f"Quality monitoring failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/quality/reports/<report_id>', methods=['GET'])
def get_quality_report(report_id):
    """Get quality report by ID"""
    try:
        if report_id in monitor.quality_reports:
            report = monitor.quality_reports[report_id]
            return jsonify(asdict(report))
        else:
            return jsonify({"error": "Report not found"}), 404
    except Exception as e:
        logger.error(f"Failed to get quality report: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/quality/dashboard', methods=['GET'])
def get_dashboard():
    """Get quality dashboard data"""
    try:
        dashboard_data = monitor.get_quality_dashboard_data()
        return jsonify(dashboard_data)
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8003))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Data Quality Monitor on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

