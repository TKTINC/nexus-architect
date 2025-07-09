"""
Nexus Architect Bias Detection System

This module implements comprehensive bias detection and fairness assessment
for AI systems, including demographic parity, equalized odds, and bias mitigation.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime
import re

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import fairlearn.metrics as fl_metrics
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import aif360
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiasType(Enum):
    """Types of bias that can be detected"""
    DEMOGRAPHIC = "demographic"
    REPRESENTATION = "representation"
    MEASUREMENT = "measurement"
    AGGREGATION = "aggregation"
    EVALUATION = "evaluation"
    HISTORICAL = "historical"
    CONFIRMATION = "confirmation"
    SELECTION = "selection"

class ProtectedAttribute(Enum):
    """Protected attributes for bias assessment"""
    GENDER = "gender"
    RACE = "race"
    AGE = "age"
    RELIGION = "religion"
    NATIONALITY = "nationality"
    SEXUAL_ORIENTATION = "sexual_orientation"
    DISABILITY = "disability"
    SOCIOECONOMIC_STATUS = "socioeconomic_status"

class FairnessMetric(Enum):
    """Fairness metrics for bias assessment"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"

@dataclass
class BiasAssessment:
    """Bias assessment result"""
    bias_detected: bool
    bias_types: List[BiasType]
    affected_groups: List[str]
    fairness_scores: Dict[str, float]
    severity_score: float
    confidence: float
    mitigation_recommendations: List[str]
    explanation: str
    timestamp: datetime

@dataclass
class FairnessReport:
    """Comprehensive fairness report"""
    overall_fairness_score: float
    metric_scores: Dict[FairnessMetric, float]
    group_metrics: Dict[str, Dict[str, float]]
    bias_assessments: List[BiasAssessment]
    recommendations: List[str]
    timestamp: datetime

class BiasDetector:
    """Advanced bias detection using multiple approaches"""
    
    def __init__(self):
        self.bias_classifier = pipeline(
            "text-classification",
            model="d4data/bias-detection-model",
            device=0 if torch.cuda.is_available() else -1
        )
        self.gender_classifier = pipeline(
            "text-classification", 
            model="martin-ha/toxic-comment-model",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Bias keywords and patterns
        self.bias_patterns = {
            BiasType.DEMOGRAPHIC: {
                "gender": [
                    r"\b(he|she|his|her|him|man|woman|male|female|boy|girl)\b",
                    r"\b(masculine|feminine|manly|womanly)\b"
                ],
                "race": [
                    r"\b(black|white|asian|hispanic|latino|african|european|american)\b",
                    r"\b(race|racial|ethnicity|ethnic)\b"
                ],
                "age": [
                    r"\b(young|old|elderly|senior|teenager|millennial|boomer)\b",
                    r"\b(age|aged|aging)\b"
                ],
                "religion": [
                    r"\b(christian|muslim|jewish|hindu|buddhist|atheist|religious)\b",
                    r"\b(church|mosque|temple|synagogue|faith|belief)\b"
                ]
            },
            BiasType.REPRESENTATION: [
                r"\b(underrepresented|minority|majority|dominant|privileged)\b",
                r"\b(stereotype|stereotypical|typical|atypical)\b"
            ],
            BiasType.HISTORICAL: [
                r"\b(traditional|conventional|historical|legacy|established)\b",
                r"\b(always|never|typically|usually|generally)\b"
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for bias_type, patterns in self.bias_patterns.items():
            if isinstance(patterns, dict):
                self.compiled_patterns[bias_type] = {}
                for category, pattern_list in patterns.items():
                    self.compiled_patterns[bias_type][category] = [
                        re.compile(pattern, re.IGNORECASE) for pattern in pattern_list
                    ]
            else:
                self.compiled_patterns[bias_type] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in patterns
                ]
    
    async def detect_bias(self, text: str, context: Dict[str, Any] = None) -> BiasAssessment:
        """
        Detect bias in text using multiple approaches
        
        Args:
            text: Input text to analyze for bias
            context: Additional context information
            
        Returns:
            BiasAssessment with detailed bias analysis
        """
        if context is None:
            context = {}
        
        bias_types = []
        affected_groups = []
        confidence_scores = []
        
        # ML-based bias detection
        try:
            bias_result = self.bias_classifier(text)
            if bias_result[0]['label'] == 'BIASED' and bias_result[0]['score'] > 0.7:
                bias_types.append(BiasType.DEMOGRAPHIC)
                confidence_scores.append(bias_result[0]['score'])
        except Exception as e:
            logger.warning(f"ML bias detection failed: {e}")
        
        # Pattern-based bias detection
        pattern_results = self._detect_bias_patterns(text)
        bias_types.extend(pattern_results['bias_types'])
        affected_groups.extend(pattern_results['affected_groups'])
        confidence_scores.extend(pattern_results['confidence_scores'])
        
        # Statistical bias detection (if data provided)
        if 'predictions' in context and 'protected_attributes' in context:
            statistical_results = self._detect_statistical_bias(
                context['predictions'], 
                context['protected_attributes'],
                context.get('true_labels')
            )
            bias_types.extend(statistical_results['bias_types'])
            affected_groups.extend(statistical_results['affected_groups'])
            confidence_scores.extend(statistical_results['confidence_scores'])
        
        # Determine overall bias detection
        bias_detected = len(bias_types) > 0
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        severity_score = self._calculate_severity(bias_types, confidence_scores)
        
        # Generate fairness scores
        fairness_scores = self._calculate_fairness_scores(text, context)
        
        # Generate mitigation recommendations
        mitigation_recommendations = self._generate_mitigation_recommendations(bias_types, affected_groups)
        
        # Generate explanation
        explanation = self._generate_bias_explanation(bias_detected, bias_types, affected_groups, severity_score)
        
        return BiasAssessment(
            bias_detected=bias_detected,
            bias_types=list(set(bias_types)),
            affected_groups=list(set(affected_groups)),
            fairness_scores=fairness_scores,
            severity_score=severity_score,
            confidence=overall_confidence,
            mitigation_recommendations=mitigation_recommendations,
            explanation=explanation,
            timestamp=datetime.utcnow()
        )
    
    def _detect_bias_patterns(self, text: str) -> Dict[str, List]:
        """Detect bias using pattern matching"""
        bias_types = []
        affected_groups = []
        confidence_scores = []
        
        text_lower = text.lower()
        
        # Check demographic bias patterns
        if BiasType.DEMOGRAPHIC in self.compiled_patterns:
            for group, patterns in self.compiled_patterns[BiasType.DEMOGRAPHIC].items():
                matches = sum(1 for pattern in patterns if pattern.search(text))
                if matches > 0:
                    bias_types.append(BiasType.DEMOGRAPHIC)
                    affected_groups.append(group)
                    confidence_scores.append(min(matches / len(patterns), 1.0))
        
        # Check representation bias patterns
        if BiasType.REPRESENTATION in self.compiled_patterns:
            patterns = self.compiled_patterns[BiasType.REPRESENTATION]
            matches = sum(1 for pattern in patterns if pattern.search(text))
            if matches > 0:
                bias_types.append(BiasType.REPRESENTATION)
                confidence_scores.append(min(matches / len(patterns), 1.0))
        
        # Check historical bias patterns
        if BiasType.HISTORICAL in self.compiled_patterns:
            patterns = self.compiled_patterns[BiasType.HISTORICAL]
            matches = sum(1 for pattern in patterns if pattern.search(text))
            if matches > 0:
                bias_types.append(BiasType.HISTORICAL)
                confidence_scores.append(min(matches / len(patterns), 1.0))
        
        return {
            'bias_types': bias_types,
            'affected_groups': affected_groups,
            'confidence_scores': confidence_scores
        }
    
    def _detect_statistical_bias(self, predictions: np.ndarray, 
                               protected_attributes: np.ndarray,
                               true_labels: Optional[np.ndarray] = None) -> Dict[str, List]:
        """Detect statistical bias in predictions"""
        bias_types = []
        affected_groups = []
        confidence_scores = []
        
        try:
            # Convert to pandas for easier analysis
            df = pd.DataFrame({
                'prediction': predictions,
                'protected_attr': protected_attributes
            })
            
            if true_labels is not None:
                df['true_label'] = true_labels
            
            # Check demographic parity
            group_rates = df.groupby('protected_attr')['prediction'].mean()
            max_rate = group_rates.max()
            min_rate = group_rates.min()
            
            if max_rate - min_rate > 0.1:  # 10% threshold
                bias_types.append(BiasType.DEMOGRAPHIC)
                affected_groups.extend(group_rates.index.tolist())
                confidence_scores.append((max_rate - min_rate) / max_rate)
            
            # Check equalized odds (if true labels available)
            if true_labels is not None:
                for group in df['protected_attr'].unique():
                    group_data = df[df['protected_attr'] == group]
                    if len(group_data) > 10:  # Minimum sample size
                        group_accuracy = accuracy_score(group_data['true_label'], group_data['prediction'])
                        overall_accuracy = accuracy_score(df['true_label'], df['prediction'])
                        
                        if abs(group_accuracy - overall_accuracy) > 0.05:  # 5% threshold
                            bias_types.append(BiasType.EVALUATION)
                            affected_groups.append(str(group))
                            confidence_scores.append(abs(group_accuracy - overall_accuracy))
        
        except Exception as e:
            logger.warning(f"Statistical bias detection failed: {e}")
        
        return {
            'bias_types': bias_types,
            'affected_groups': affected_groups,
            'confidence_scores': confidence_scores
        }
    
    def _calculate_severity(self, bias_types: List[BiasType], confidence_scores: List[float]) -> float:
        """Calculate bias severity score"""
        if not bias_types:
            return 0.0
        
        # Weight different bias types
        type_weights = {
            BiasType.DEMOGRAPHIC: 1.0,
            BiasType.REPRESENTATION: 0.8,
            BiasType.MEASUREMENT: 0.9,
            BiasType.AGGREGATION: 0.7,
            BiasType.EVALUATION: 0.9,
            BiasType.HISTORICAL: 0.6,
            BiasType.CONFIRMATION: 0.7,
            BiasType.SELECTION: 0.8
        }
        
        weighted_scores = []
        for bias_type, confidence in zip(bias_types, confidence_scores):
            weight = type_weights.get(bias_type, 0.5)
            weighted_scores.append(confidence * weight)
        
        return np.mean(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_fairness_scores(self, text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various fairness scores"""
        fairness_scores = {}
        
        # Basic fairness score based on bias detection
        bias_indicators = sum(1 for pattern_list in self.compiled_patterns.values() 
                            for pattern in (pattern_list if isinstance(pattern_list, list) 
                                          else [p for sublist in pattern_list.values() for p in sublist])
                            if pattern.search(text))
        
        fairness_scores['overall'] = max(0.0, 1.0 - (bias_indicators * 0.1))
        
        # Demographic parity score
        if 'predictions' in context and 'protected_attributes' in context:
            try:
                df = pd.DataFrame({
                    'prediction': context['predictions'],
                    'protected_attr': context['protected_attributes']
                })
                group_rates = df.groupby('protected_attr')['prediction'].mean()
                parity_score = 1.0 - (group_rates.max() - group_rates.min())
                fairness_scores['demographic_parity'] = max(0.0, parity_score)
            except Exception:
                fairness_scores['demographic_parity'] = 0.5
        else:
            fairness_scores['demographic_parity'] = 0.7  # Default when no data
        
        # Representation score
        representation_keywords = ['diverse', 'inclusive', 'representative', 'balanced']
        representation_count = sum(1 for keyword in representation_keywords if keyword in text.lower())
        fairness_scores['representation'] = min(1.0, representation_count * 0.25 + 0.5)
        
        return fairness_scores
    
    def _generate_mitigation_recommendations(self, bias_types: List[BiasType], 
                                           affected_groups: List[str]) -> List[str]:
        """Generate bias mitigation recommendations"""
        recommendations = []
        
        if BiasType.DEMOGRAPHIC in bias_types:
            recommendations.append("Implement demographic parity constraints in model training")
            recommendations.append("Use fairness-aware preprocessing techniques")
            recommendations.append("Apply post-processing calibration for affected demographic groups")
        
        if BiasType.REPRESENTATION in bias_types:
            recommendations.append("Increase representation of underrepresented groups in training data")
            recommendations.append("Use data augmentation techniques to balance group representation")
            recommendations.append("Implement stratified sampling for balanced datasets")
        
        if BiasType.HISTORICAL in bias_types:
            recommendations.append("Remove or reduce reliance on historical biased features")
            recommendations.append("Use temporal validation to assess bias evolution")
            recommendations.append("Implement bias-aware feature selection")
        
        if BiasType.EVALUATION in bias_types:
            recommendations.append("Use group-specific evaluation metrics")
            recommendations.append("Implement fairness-constrained optimization")
            recommendations.append("Apply threshold optimization for equalized odds")
        
        # Group-specific recommendations
        if 'gender' in affected_groups:
            recommendations.append("Implement gender-neutral language processing")
            recommendations.append("Use gender-balanced training datasets")
        
        if 'race' in affected_groups:
            recommendations.append("Audit for racial bias in feature engineering")
            recommendations.append("Implement race-aware fairness constraints")
        
        if 'age' in affected_groups:
            recommendations.append("Consider age-specific model variants")
            recommendations.append("Implement age-balanced evaluation metrics")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_bias_explanation(self, bias_detected: bool, bias_types: List[BiasType],
                                 affected_groups: List[str], severity_score: float) -> str:
        """Generate human-readable bias explanation"""
        if not bias_detected:
            return "No significant bias detected in the analyzed content."
        
        explanation_parts = []
        
        if severity_score > 0.8:
            explanation_parts.append("High severity bias detected")
        elif severity_score > 0.5:
            explanation_parts.append("Moderate bias detected")
        else:
            explanation_parts.append("Low-level bias detected")
        
        if bias_types:
            bias_type_names = [bt.value.replace('_', ' ') for bt in bias_types]
            explanation_parts.append(f"Types: {', '.join(bias_type_names)}")
        
        if affected_groups:
            explanation_parts.append(f"Affected groups: {', '.join(affected_groups)}")
        
        explanation_parts.append(f"Severity score: {severity_score:.2f}")
        
        return ". ".join(explanation_parts) + "."

class FairnessEvaluator:
    """Comprehensive fairness evaluation using multiple metrics"""
    
    def __init__(self):
        self.fairness_metrics = {
            FairnessMetric.DEMOGRAPHIC_PARITY: self._demographic_parity,
            FairnessMetric.EQUALIZED_ODDS: self._equalized_odds,
            FairnessMetric.EQUAL_OPPORTUNITY: self._equal_opportunity,
            FairnessMetric.CALIBRATION: self._calibration,
            FairnessMetric.INDIVIDUAL_FAIRNESS: self._individual_fairness
        }
    
    def evaluate_fairness(self, predictions: np.ndarray, true_labels: np.ndarray,
                         protected_attributes: np.ndarray, 
                         prediction_probabilities: Optional[np.ndarray] = None) -> FairnessReport:
        """
        Comprehensive fairness evaluation
        
        Args:
            predictions: Model predictions
            true_labels: Ground truth labels
            protected_attributes: Protected attribute values
            prediction_probabilities: Prediction probabilities (optional)
            
        Returns:
            FairnessReport with detailed fairness analysis
        """
        metric_scores = {}
        group_metrics = {}
        
        # Calculate fairness metrics
        for metric, func in self.fairness_metrics.items():
            try:
                if metric == FairnessMetric.CALIBRATION and prediction_probabilities is not None:
                    score = func(true_labels, prediction_probabilities, protected_attributes)
                elif metric == FairnessMetric.INDIVIDUAL_FAIRNESS:
                    # Individual fairness requires additional similarity data
                    score = 0.7  # Placeholder
                else:
                    score = func(predictions, true_labels, protected_attributes)
                metric_scores[metric] = score
            except Exception as e:
                logger.warning(f"Failed to calculate {metric.value}: {e}")
                metric_scores[metric] = 0.5  # Default score
        
        # Calculate group-specific metrics
        unique_groups = np.unique(protected_attributes)
        for group in unique_groups:
            group_mask = protected_attributes == group
            group_preds = predictions[group_mask]
            group_labels = true_labels[group_mask]
            
            if len(group_preds) > 0:
                group_metrics[str(group)] = {
                    'accuracy': accuracy_score(group_labels, group_preds),
                    'precision': precision_score(group_labels, group_preds, average='weighted', zero_division=0),
                    'recall': recall_score(group_labels, group_preds, average='weighted', zero_division=0),
                    'size': len(group_preds),
                    'positive_rate': np.mean(group_preds)
                }
        
        # Calculate overall fairness score
        overall_fairness_score = np.mean(list(metric_scores.values()))
        
        # Generate recommendations
        recommendations = self._generate_fairness_recommendations(metric_scores, group_metrics)
        
        return FairnessReport(
            overall_fairness_score=overall_fairness_score,
            metric_scores=metric_scores,
            group_metrics=group_metrics,
            bias_assessments=[],  # Would be populated by BiasDetector
            recommendations=recommendations,
            timestamp=datetime.utcnow()
        )
    
    def _demographic_parity(self, predictions: np.ndarray, true_labels: np.ndarray,
                          protected_attributes: np.ndarray) -> float:
        """Calculate demographic parity score"""
        try:
            return fl_metrics.demographic_parity_difference(
                true_labels, predictions, sensitive_features=protected_attributes
            )
        except Exception:
            # Fallback calculation
            df = pd.DataFrame({
                'prediction': predictions,
                'protected_attr': protected_attributes
            })
            group_rates = df.groupby('protected_attr')['prediction'].mean()
            return 1.0 - abs(group_rates.max() - group_rates.min())
    
    def _equalized_odds(self, predictions: np.ndarray, true_labels: np.ndarray,
                       protected_attributes: np.ndarray) -> float:
        """Calculate equalized odds score"""
        try:
            return fl_metrics.equalized_odds_difference(
                true_labels, predictions, sensitive_features=protected_attributes
            )
        except Exception:
            # Fallback calculation
            scores = []
            for group in np.unique(protected_attributes):
                group_mask = protected_attributes == group
                if np.sum(group_mask) > 0:
                    group_accuracy = accuracy_score(true_labels[group_mask], predictions[group_mask])
                    scores.append(group_accuracy)
            return 1.0 - (max(scores) - min(scores)) if scores else 0.5
    
    def _equal_opportunity(self, predictions: np.ndarray, true_labels: np.ndarray,
                          protected_attributes: np.ndarray) -> float:
        """Calculate equal opportunity score"""
        try:
            return fl_metrics.true_positive_rate_difference(
                true_labels, predictions, sensitive_features=protected_attributes
            )
        except Exception:
            # Fallback calculation
            tpr_scores = []
            for group in np.unique(protected_attributes):
                group_mask = protected_attributes == group
                group_labels = true_labels[group_mask]
                group_preds = predictions[group_mask]
                
                if len(group_labels) > 0 and np.sum(group_labels) > 0:
                    tpr = np.sum((group_labels == 1) & (group_preds == 1)) / np.sum(group_labels == 1)
                    tpr_scores.append(tpr)
            
            return 1.0 - (max(tpr_scores) - min(tpr_scores)) if tpr_scores else 0.5
    
    def _calibration(self, true_labels: np.ndarray, prediction_probabilities: np.ndarray,
                    protected_attributes: np.ndarray) -> float:
        """Calculate calibration score"""
        calibration_scores = []
        
        for group in np.unique(protected_attributes):
            group_mask = protected_attributes == group
            group_labels = true_labels[group_mask]
            group_probs = prediction_probabilities[group_mask]
            
            if len(group_labels) > 10:  # Minimum sample size
                # Calculate calibration error
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                calibration_error = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (group_probs > bin_lower) & (group_probs <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = group_labels[in_bin].mean()
                        avg_confidence_in_bin = group_probs[in_bin].mean()
                        calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                calibration_scores.append(1.0 - calibration_error)
        
        return np.mean(calibration_scores) if calibration_scores else 0.5
    
    def _individual_fairness(self, predictions: np.ndarray, true_labels: np.ndarray,
                           protected_attributes: np.ndarray) -> float:
        """Calculate individual fairness score (placeholder)"""
        # Individual fairness requires similarity metrics between individuals
        # This is a simplified placeholder implementation
        return 0.7
    
    def _generate_fairness_recommendations(self, metric_scores: Dict[FairnessMetric, float],
                                         group_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate fairness improvement recommendations"""
        recommendations = []
        
        # Check demographic parity
        if metric_scores.get(FairnessMetric.DEMOGRAPHIC_PARITY, 1.0) < 0.8:
            recommendations.append("Improve demographic parity through data rebalancing or algorithmic constraints")
        
        # Check equalized odds
        if metric_scores.get(FairnessMetric.EQUALIZED_ODDS, 1.0) < 0.8:
            recommendations.append("Address equalized odds violations through post-processing or in-processing techniques")
        
        # Check group performance disparities
        if group_metrics:
            accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
            if max(accuracies) - min(accuracies) > 0.1:
                recommendations.append("Address performance disparities across demographic groups")
        
        # Check calibration
        if metric_scores.get(FairnessMetric.CALIBRATION, 1.0) < 0.8:
            recommendations.append("Improve model calibration across different demographic groups")
        
        # General recommendations
        recommendations.extend([
            "Regularly audit model performance across demographic groups",
            "Implement fairness-aware model selection and hyperparameter tuning",
            "Consider using fairness-constrained optimization techniques",
            "Establish ongoing monitoring for fairness metrics in production"
        ])
        
        return recommendations

class BiasDetectionSystem:
    """Main bias detection and fairness evaluation system"""
    
    def __init__(self):
        self.bias_detector = BiasDetector()
        self.fairness_evaluator = FairnessEvaluator()
        self.mitigation_strategies = {
            'preprocessing': [
                'data_rebalancing',
                'feature_selection',
                'disparate_impact_removal'
            ],
            'inprocessing': [
                'fairness_constraints',
                'adversarial_debiasing',
                'multi_task_learning'
            ],
            'postprocessing': [
                'threshold_optimization',
                'calibration',
                'output_modification'
            ]
        }
    
    async def comprehensive_bias_assessment(self, text: str = None, 
                                          predictions: np.ndarray = None,
                                          true_labels: np.ndarray = None,
                                          protected_attributes: np.ndarray = None,
                                          prediction_probabilities: np.ndarray = None,
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive bias assessment combining text and statistical analysis
        
        Args:
            text: Text to analyze for bias (optional)
            predictions: Model predictions (optional)
            true_labels: Ground truth labels (optional)
            protected_attributes: Protected attribute values (optional)
            prediction_probabilities: Prediction probabilities (optional)
            context: Additional context information
            
        Returns:
            Comprehensive bias assessment report
        """
        if context is None:
            context = {}
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'assessment_type': 'comprehensive'
        }
        
        # Text-based bias detection
        if text:
            text_bias_assessment = await self.bias_detector.detect_bias(text, context)
            results['text_bias_assessment'] = asdict(text_bias_assessment)
        
        # Statistical fairness evaluation
        if predictions is not None and true_labels is not None and protected_attributes is not None:
            fairness_report = self.fairness_evaluator.evaluate_fairness(
                predictions, true_labels, protected_attributes, prediction_probabilities
            )
            results['fairness_report'] = asdict(fairness_report)
        
        # Combined assessment
        overall_bias_detected = False
        overall_severity = 0.0
        combined_recommendations = []
        
        if 'text_bias_assessment' in results:
            text_assessment = results['text_bias_assessment']
            overall_bias_detected = overall_bias_detected or text_assessment['bias_detected']
            overall_severity = max(overall_severity, text_assessment['severity_score'])
            combined_recommendations.extend(text_assessment['mitigation_recommendations'])
        
        if 'fairness_report' in results:
            fairness_report = results['fairness_report']
            overall_bias_detected = overall_bias_detected or fairness_report['overall_fairness_score'] < 0.8
            overall_severity = max(overall_severity, 1.0 - fairness_report['overall_fairness_score'])
            combined_recommendations.extend(fairness_report['recommendations'])
        
        # Generate overall assessment
        results['overall_assessment'] = {
            'bias_detected': overall_bias_detected,
            'severity_score': overall_severity,
            'risk_level': self._determine_risk_level(overall_severity),
            'recommendations': list(set(combined_recommendations)),
            'mitigation_strategies': self._select_mitigation_strategies(overall_severity, overall_bias_detected)
        }
        
        return results
    
    def _determine_risk_level(self, severity_score: float) -> str:
        """Determine risk level based on severity score"""
        if severity_score > 0.8:
            return "HIGH"
        elif severity_score > 0.5:
            return "MEDIUM"
        elif severity_score > 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _select_mitigation_strategies(self, severity_score: float, bias_detected: bool) -> List[str]:
        """Select appropriate mitigation strategies"""
        if not bias_detected:
            return ["Continue monitoring for bias", "Maintain current fairness practices"]
        
        strategies = []
        
        if severity_score > 0.7:
            # High severity - use all approaches
            strategies.extend(self.mitigation_strategies['preprocessing'])
            strategies.extend(self.mitigation_strategies['inprocessing'])
            strategies.extend(self.mitigation_strategies['postprocessing'])
        elif severity_score > 0.4:
            # Medium severity - focus on preprocessing and postprocessing
            strategies.extend(self.mitigation_strategies['preprocessing'])
            strategies.extend(self.mitigation_strategies['postprocessing'])
        else:
            # Low severity - focus on postprocessing
            strategies.extend(self.mitigation_strategies['postprocessing'])
        
        return strategies

if __name__ == "__main__":
    # Example usage
    async def main():
        bias_system = BiasDetectionSystem()
        
        # Example text analysis
        test_text = "The software engineer, who was a young man, designed the system while the nurse, an older woman, provided feedback."
        
        result = await bias_system.comprehensive_bias_assessment(text=test_text)
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(main())

