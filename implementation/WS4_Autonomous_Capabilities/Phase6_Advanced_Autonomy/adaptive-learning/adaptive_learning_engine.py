#!/usr/bin/env python3
"""
Nexus Architect - WS4 Phase 6: Adaptive Learning Engine
Continuous learning from operational experience, adaptation to changing requirements, and complex decision making
"""

import asyncio
import json
import logging
import time
import threading
import uuid
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import redis
import psycopg2
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningType(Enum):
    """Types of learning algorithms"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    ONLINE = "online"
    TRANSFER = "transfer"

class DecisionType(Enum):
    """Types of decisions the system can make"""
    RESOURCE_ALLOCATION = "resource_allocation"
    TASK_PRIORITIZATION = "task_prioritization"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_RESPONSE = "security_response"
    CAPACITY_PLANNING = "capacity_planning"
    INCIDENT_RESPONSE = "incident_response"
    STRATEGIC_PLANNING = "strategic_planning"

class ObjectiveType(Enum):
    """Types of objectives for multi-objective optimization"""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_RELIABILITY = "maximize_reliability"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    MINIMIZE_DOWNTIME = "minimize_downtime"

@dataclass
class LearningExperience:
    """Represents a learning experience from system operations"""
    id: str
    timestamp: datetime
    context: Dict[str, Any]
    action_taken: Dict[str, Any]
    outcome: Dict[str, Any]
    success: bool
    performance_metrics: Dict[str, float]
    feedback_score: float
    learning_type: LearningType
    tags: List[str] = field(default_factory=list)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert experience to feature vector for ML"""
        features = []
        
        # Context features
        for key, value in self.context.items():
            if isinstance(value, (int, float)):
                features.append(value)
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
        
        # Performance metrics
        features.extend(list(self.performance_metrics.values()))
        
        # Success indicator
        features.append(1.0 if self.success else 0.0)
        
        # Feedback score
        features.append(self.feedback_score)
        
        return np.array(features)

@dataclass
class DecisionScenario:
    """Represents a complex decision scenario"""
    id: str
    scenario_type: DecisionType
    description: str
    context: Dict[str, Any]
    objectives: List[ObjectiveType]
    constraints: Dict[str, Any]
    available_actions: List[Dict[str, Any]]
    urgency: float  # 0.0 to 1.0
    complexity: float  # 0.0 to 1.0
    risk_tolerance: float  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
@dataclass
class DecisionResult:
    """Result of a decision-making process"""
    scenario_id: str
    selected_action: Dict[str, Any]
    confidence: float
    reasoning: str
    expected_outcomes: Dict[str, float]
    risk_assessment: Dict[str, float]
    alternative_actions: List[Dict[str, Any]]
    decision_time: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AdaptationRule:
    """Rule for adapting system behavior"""
    id: str
    condition: str
    action: str
    priority: int
    confidence: float
    success_rate: float
    usage_count: int
    created_at: datetime
    last_used: Optional[datetime] = None

class ExperienceBuffer:
    """Manages learning experiences with efficient storage and retrieval"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.index_by_type = defaultdict(list)
        self.index_by_success = defaultdict(list)
        self.index_by_timestamp = []
        
    def add_experience(self, experience: LearningExperience):
        """Add a new learning experience"""
        self.experiences.append(experience)
        
        # Update indices
        self.index_by_type[experience.learning_type].append(len(self.experiences) - 1)
        self.index_by_success[experience.success].append(len(self.experiences) - 1)
        self.index_by_timestamp.append((experience.timestamp, len(self.experiences) - 1))
        
        # Keep timestamp index sorted and limited
        self.index_by_timestamp.sort(key=lambda x: x[0])
        if len(self.index_by_timestamp) > self.max_size:
            self.index_by_timestamp = self.index_by_timestamp[-self.max_size:]
        
        logger.debug(f"Added experience {experience.id} to buffer")
    
    def get_experiences_by_type(self, learning_type: LearningType) -> List[LearningExperience]:
        """Get experiences by learning type"""
        indices = self.index_by_type.get(learning_type, [])
        return [self.experiences[i] for i in indices if i < len(self.experiences)]
    
    def get_recent_experiences(self, hours: int = 24) -> List[LearningExperience]:
        """Get experiences from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_indices = [idx for timestamp, idx in self.index_by_timestamp 
                         if timestamp >= cutoff_time]
        return [self.experiences[i] for i in recent_indices if i < len(self.experiences)]
    
    def get_successful_experiences(self) -> List[LearningExperience]:
        """Get only successful experiences"""
        indices = self.index_by_success.get(True, [])
        return [self.experiences[i] for i in indices if i < len(self.experiences)]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert experiences to pandas DataFrame for analysis"""
        data = []
        for exp in self.experiences:
            row = {
                'id': exp.id,
                'timestamp': exp.timestamp,
                'success': exp.success,
                'feedback_score': exp.feedback_score,
                'learning_type': exp.learning_type.value
            }
            
            # Add context features
            for key, value in exp.context.items():
                row[f'context_{key}'] = value
            
            # Add performance metrics
            for key, value in exp.performance_metrics.items():
                row[f'metric_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)

class PatternRecognizer:
    """Recognizes patterns in operational data and experiences"""
    
    def __init__(self):
        self.clustering_model = None
        self.pattern_cache = {}
        self.scaler = StandardScaler()
        
    def identify_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Identify patterns in experiences using clustering"""
        if len(experiences) < 10:
            return {'patterns': [], 'message': 'Insufficient data for pattern recognition'}
        
        # Convert experiences to feature vectors
        feature_vectors = []
        for exp in experiences:
            try:
                features = exp.to_feature_vector()
                if len(features) > 0:
                    feature_vectors.append(features)
            except Exception as e:
                logger.warning(f"Failed to convert experience {exp.id} to features: {e}")
        
        if len(feature_vectors) < 10:
            return {'patterns': [], 'message': 'Insufficient valid feature vectors'}
        
        # Normalize features
        feature_matrix = np.array(feature_vectors)
        
        # Handle variable feature lengths by padding
        max_length = max(len(fv) for fv in feature_vectors)
        padded_features = []
        for fv in feature_vectors:
            padded = np.pad(fv, (0, max_length - len(fv)), mode='constant')
            padded_features.append(padded)
        
        feature_matrix = np.array(padded_features)
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Perform clustering
        optimal_clusters = self._find_optimal_clusters(feature_matrix_scaled)
        self.clustering_model = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = self.clustering_model.fit_predict(feature_matrix_scaled)
        
        # Analyze clusters
        patterns = []
        for cluster_id in range(optimal_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_experiences = [experiences[i] for i in cluster_indices]
            
            pattern = self._analyze_cluster(cluster_id, cluster_experiences)
            patterns.append(pattern)
        
        return {
            'patterns': patterns,
            'total_clusters': optimal_clusters,
            'silhouette_score': silhouette_score(feature_matrix_scaled, cluster_labels)
        }
    
    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method"""
        max_clusters = min(10, len(data) // 3)
        if max_clusters < 2:
            return 2
        
        inertias = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_idx = np.argmax(second_diffs) + 2
            return cluster_range[elbow_idx]
        
        return 3  # Default
    
    def _analyze_cluster(self, cluster_id: int, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze a cluster of experiences to identify patterns"""
        if not experiences:
            return {'cluster_id': cluster_id, 'size': 0, 'characteristics': {}}
        
        # Calculate cluster characteristics
        success_rate = sum(1 for exp in experiences if exp.success) / len(experiences)
        avg_feedback = np.mean([exp.feedback_score for exp in experiences])
        
        # Common context patterns
        context_patterns = defaultdict(list)
        for exp in experiences:
            for key, value in exp.context.items():
                context_patterns[key].append(value)
        
        # Performance patterns
        performance_patterns = defaultdict(list)
        for exp in experiences:
            for key, value in exp.performance_metrics.items():
                performance_patterns[key].append(value)
        
        # Learning type distribution
        learning_types = defaultdict(int)
        for exp in experiences:
            learning_types[exp.learning_type.value] += 1
        
        return {
            'cluster_id': cluster_id,
            'size': len(experiences),
            'success_rate': success_rate,
            'avg_feedback_score': avg_feedback,
            'dominant_learning_type': max(learning_types.items(), key=lambda x: x[1])[0],
            'context_patterns': {k: {'mean': np.mean(v), 'std': np.std(v)} 
                               for k, v in context_patterns.items() 
                               if all(isinstance(x, (int, float)) for x in v)},
            'performance_patterns': {k: {'mean': np.mean(v), 'std': np.std(v)} 
                                   for k, v in performance_patterns.items()},
            'characteristics': {
                'high_success': success_rate > 0.8,
                'high_feedback': avg_feedback > 0.8,
                'consistent_performance': all(np.std(v) < 0.1 for v in performance_patterns.values())
            }
        }

class PerformancePredictor:
    """Predicts performance outcomes based on historical data"""
    
    def __init__(self):
        self.regression_models = {}
        self.classification_models = {}
        self.feature_columns = []
        
    def train_models(self, experiences: List[LearningExperience]):
        """Train prediction models on historical experiences"""
        if len(experiences) < 50:
            logger.warning("Insufficient data for model training")
            return
        
        # Prepare training data
        df = self._experiences_to_dataframe(experiences)
        
        if df.empty:
            logger.warning("No valid training data available")
            return
        
        # Separate features and targets
        feature_cols = [col for col in df.columns 
                       if col.startswith(('context_', 'metric_')) and 
                       df[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) < 3:
            logger.warning("Insufficient features for model training")
            return
        
        self.feature_columns = feature_cols
        X = df[feature_cols].fillna(0)
        
        # Train regression models for continuous metrics
        continuous_targets = ['feedback_score']
        for target in continuous_targets:
            if target in df.columns:
                y = df[target].fillna(0)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                self.regression_models[target] = {
                    'model': model,
                    'mse': mse,
                    'feature_importance': dict(zip(feature_cols, model.feature_importances_))
                }
                
                logger.info(f"Trained regression model for {target} with MSE: {mse:.4f}")
        
        # Train classification model for success prediction
        if 'success' in df.columns:
            y = df['success'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.classification_models['success'] = {
                'model': model,
                'accuracy': accuracy,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            logger.info(f"Trained classification model for success with accuracy: {accuracy:.4f}")
    
    def _experiences_to_dataframe(self, experiences: List[LearningExperience]) -> pd.DataFrame:
        """Convert experiences to DataFrame for model training"""
        data = []
        for exp in experiences:
            row = {
                'success': exp.success,
                'feedback_score': exp.feedback_score
            }
            
            # Add context features
            for key, value in exp.context.items():
                if isinstance(value, (int, float, bool)):
                    row[f'context_{key}'] = float(value)
            
            # Add performance metrics
            for key, value in exp.performance_metrics.items():
                if isinstance(value, (int, float)):
                    row[f'metric_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def predict_outcome(self, context: Dict[str, Any], 
                       performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Predict outcome for given context and metrics"""
        if not self.feature_columns:
            return {'error': 'Models not trained yet'}
        
        # Prepare feature vector
        features = {}
        for key, value in context.items():
            if isinstance(value, (int, float, bool)):
                features[f'context_{key}'] = float(value)
        
        for key, value in performance_metrics.items():
            if isinstance(value, (int, float)):
                features[f'metric_{key}'] = value
        
        # Create feature vector with same columns as training
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0.0))
        
        X = np.array(feature_vector).reshape(1, -1)
        
        predictions = {}
        
        # Regression predictions
        for target, model_info in self.regression_models.items():
            pred = model_info['model'].predict(X)[0]
            predictions[target] = {
                'predicted_value': pred,
                'model_mse': model_info['mse']
            }
        
        # Classification predictions
        for target, model_info in self.classification_models.items():
            pred_proba = model_info['model'].predict_proba(X)[0]
            pred_class = model_info['model'].predict(X)[0]
            
            predictions[target] = {
                'predicted_class': bool(pred_class),
                'probability': pred_proba[1] if len(pred_proba) > 1 else pred_proba[0],
                'model_accuracy': model_info['accuracy']
            }
        
        return predictions

class MultiObjectiveOptimizer:
    """Handles multi-objective optimization for complex decision scenarios"""
    
    def __init__(self):
        self.pareto_solutions = []
        self.objective_weights = {}
        
    def optimize(self, scenario: DecisionScenario) -> List[Dict[str, Any]]:
        """Perform multi-objective optimization for a decision scenario"""
        if not scenario.available_actions:
            return []
        
        # Evaluate each action against all objectives
        action_scores = []
        for action in scenario.available_actions:
            scores = self._evaluate_action(action, scenario)
            action_scores.append({
                'action': action,
                'scores': scores,
                'weighted_score': self._calculate_weighted_score(scores, scenario.objectives)
            })
        
        # Find Pareto optimal solutions
        pareto_optimal = self._find_pareto_optimal(action_scores, scenario.objectives)
        
        # Rank solutions
        ranked_solutions = self._rank_solutions(pareto_optimal, scenario)
        
        return ranked_solutions
    
    def _evaluate_action(self, action: Dict[str, Any], 
                        scenario: DecisionScenario) -> Dict[ObjectiveType, float]:
        """Evaluate an action against all objectives"""
        scores = {}
        
        for objective in scenario.objectives:
            score = self._calculate_objective_score(action, objective, scenario)
            scores[objective] = score
        
        return scores
    
    def _calculate_objective_score(self, action: Dict[str, Any], 
                                 objective: ObjectiveType, 
                                 scenario: DecisionScenario) -> float:
        """Calculate score for a specific objective"""
        # Simplified scoring based on action properties and scenario context
        base_score = 0.5  # Default neutral score
        
        if objective == ObjectiveType.MINIMIZE_COST:
            cost = action.get('estimated_cost', 50)
            base_score = max(0, 1.0 - (cost / 100.0))
        
        elif objective == ObjectiveType.MAXIMIZE_PERFORMANCE:
            performance_gain = action.get('performance_gain', 0.1)
            base_score = min(1.0, performance_gain)
        
        elif objective == ObjectiveType.MINIMIZE_LATENCY:
            latency = action.get('estimated_latency', 100)
            base_score = max(0, 1.0 - (latency / 1000.0))
        
        elif objective == ObjectiveType.MAXIMIZE_RELIABILITY:
            reliability = action.get('reliability_score', 0.8)
            base_score = reliability
        
        elif objective == ObjectiveType.MINIMIZE_RISK:
            risk = action.get('risk_score', 0.3)
            base_score = 1.0 - risk
        
        elif objective == ObjectiveType.MAXIMIZE_EFFICIENCY:
            efficiency = action.get('efficiency_score', 0.7)
            base_score = efficiency
        
        elif objective == ObjectiveType.MINIMIZE_DOWNTIME:
            downtime = action.get('estimated_downtime', 10)
            base_score = max(0, 1.0 - (downtime / 60.0))
        
        # Apply scenario-specific modifiers
        urgency_modifier = 1.0 + (scenario.urgency - 0.5) * 0.2
        complexity_modifier = 1.0 - (scenario.complexity * 0.1)
        
        return max(0, min(1.0, base_score * urgency_modifier * complexity_modifier))
    
    def _calculate_weighted_score(self, scores: Dict[ObjectiveType, float], 
                                objectives: List[ObjectiveType]) -> float:
        """Calculate weighted score across all objectives"""
        if not objectives:
            return 0.0
        
        # Equal weights if not specified
        weight_per_objective = 1.0 / len(objectives)
        
        total_score = 0.0
        for objective in objectives:
            weight = self.objective_weights.get(objective, weight_per_objective)
            total_score += scores.get(objective, 0.0) * weight
        
        return total_score
    
    def _find_pareto_optimal(self, action_scores: List[Dict[str, Any]], 
                           objectives: List[ObjectiveType]) -> List[Dict[str, Any]]:
        """Find Pareto optimal solutions"""
        pareto_optimal = []
        
        for i, candidate in enumerate(action_scores):
            is_dominated = False
            
            for j, other in enumerate(action_scores):
                if i == j:
                    continue
                
                # Check if 'other' dominates 'candidate'
                dominates = True
                for objective in objectives:
                    candidate_score = candidate['scores'].get(objective, 0.0)
                    other_score = other['scores'].get(objective, 0.0)
                    
                    # For minimization objectives, lower is better
                    if objective.value.startswith('minimize'):
                        if candidate_score < other_score:
                            dominates = False
                            break
                    else:  # Maximization objectives
                        if candidate_score > other_score:
                            dominates = False
                            break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(candidate)
        
        return pareto_optimal
    
    def _rank_solutions(self, solutions: List[Dict[str, Any]], 
                       scenario: DecisionScenario) -> List[Dict[str, Any]]:
        """Rank Pareto optimal solutions based on scenario preferences"""
        # Sort by weighted score
        ranked = sorted(solutions, key=lambda x: x['weighted_score'], reverse=True)
        
        # Add ranking information
        for i, solution in enumerate(ranked):
            solution['rank'] = i + 1
            solution['pareto_optimal'] = True
        
        return ranked

class StrategicPlanner:
    """Handles long-term strategic planning and decision making"""
    
    def __init__(self):
        self.strategic_goals = []
        self.planning_horizon = timedelta(days=90)  # 3 months
        self.scenario_cache = {}
        
    def create_strategic_plan(self, goals: List[Dict[str, Any]], 
                            constraints: Dict[str, Any],
                            time_horizon: timedelta = None) -> Dict[str, Any]:
        """Create a strategic plan to achieve multiple goals"""
        if time_horizon:
            self.planning_horizon = time_horizon
        
        plan = {
            'id': str(uuid.uuid4()),
            'goals': goals,
            'constraints': constraints,
            'time_horizon': self.planning_horizon,
            'phases': [],
            'milestones': [],
            'risk_assessment': {},
            'resource_requirements': {},
            'created_at': datetime.now()
        }
        
        # Break down goals into phases
        phases = self._decompose_goals_into_phases(goals, constraints)
        plan['phases'] = phases
        
        # Identify milestones
        milestones = self._identify_milestones(phases)
        plan['milestones'] = milestones
        
        # Assess risks
        risk_assessment = self._assess_strategic_risks(plan)
        plan['risk_assessment'] = risk_assessment
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(phases)
        plan['resource_requirements'] = resource_requirements
        
        return plan
    
    def _decompose_goals_into_phases(self, goals: List[Dict[str, Any]], 
                                   constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose strategic goals into executable phases"""
        phases = []
        
        # Group goals by priority and dependencies
        high_priority_goals = [g for g in goals if g.get('priority', 'medium') == 'high']
        medium_priority_goals = [g for g in goals if g.get('priority', 'medium') == 'medium']
        low_priority_goals = [g for g in goals if g.get('priority', 'medium') == 'low']
        
        # Phase 1: Foundation and high-priority goals
        if high_priority_goals:
            phases.append({
                'id': 'phase_1_foundation',
                'name': 'Foundation and Critical Goals',
                'goals': high_priority_goals,
                'duration_weeks': 4,
                'dependencies': [],
                'success_criteria': [g.get('success_criteria', []) for g in high_priority_goals]
            })
        
        # Phase 2: Core implementation
        if medium_priority_goals:
            phases.append({
                'id': 'phase_2_core',
                'name': 'Core Implementation',
                'goals': medium_priority_goals,
                'duration_weeks': 6,
                'dependencies': ['phase_1_foundation'] if phases else [],
                'success_criteria': [g.get('success_criteria', []) for g in medium_priority_goals]
            })
        
        # Phase 3: Optimization and enhancement
        if low_priority_goals:
            phases.append({
                'id': 'phase_3_optimization',
                'name': 'Optimization and Enhancement',
                'goals': low_priority_goals,
                'duration_weeks': 4,
                'dependencies': ['phase_2_core'] if len(phases) > 1 else (['phase_1_foundation'] if phases else []),
                'success_criteria': [g.get('success_criteria', []) for g in low_priority_goals]
            })
        
        return phases
    
    def _identify_milestones(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key milestones in the strategic plan"""
        milestones = []
        
        for i, phase in enumerate(phases):
            # Phase completion milestone
            milestones.append({
                'id': f"milestone_{phase['id']}_complete",
                'name': f"{phase['name']} Completion",
                'description': f"All goals in {phase['name']} achieved",
                'target_date': datetime.now() + timedelta(weeks=sum(p['duration_weeks'] for p in phases[:i+1])),
                'success_criteria': phase['success_criteria'],
                'phase_id': phase['id']
            })
            
            # Mid-phase checkpoint if phase is long
            if phase['duration_weeks'] > 4:
                milestones.append({
                    'id': f"milestone_{phase['id']}_checkpoint",
                    'name': f"{phase['name']} Checkpoint",
                    'description': f"Mid-phase review for {phase['name']}",
                    'target_date': datetime.now() + timedelta(weeks=sum(p['duration_weeks'] for p in phases[:i]) + phase['duration_weeks'] // 2),
                    'success_criteria': ['50% of phase goals achieved', 'No critical blockers'],
                    'phase_id': phase['id']
                })
        
        return sorted(milestones, key=lambda x: x['target_date'])
    
    def _assess_strategic_risks(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with the strategic plan"""
        risks = {
            'technical_risks': [],
            'resource_risks': [],
            'timeline_risks': [],
            'external_risks': [],
            'overall_risk_score': 0.0
        }
        
        # Technical risks
        complex_goals = [g for g in plan['goals'] if g.get('complexity', 'medium') == 'high']
        if complex_goals:
            risks['technical_risks'].append({
                'risk': 'High complexity goals may face implementation challenges',
                'probability': 0.6,
                'impact': 0.8,
                'mitigation': 'Break down complex goals into smaller tasks, allocate expert resources'
            })
        
        # Resource risks
        total_duration = sum(p['duration_weeks'] for p in plan['phases'])
        if total_duration > 16:  # More than 4 months
            risks['resource_risks'].append({
                'risk': 'Extended timeline may strain resource availability',
                'probability': 0.5,
                'impact': 0.7,
                'mitigation': 'Secure long-term resource commitments, plan for resource rotation'
            })
        
        # Timeline risks
        dependent_phases = [p for p in plan['phases'] if p['dependencies']]
        if len(dependent_phases) > 1:
            risks['timeline_risks'].append({
                'risk': 'Phase dependencies may cause cascading delays',
                'probability': 0.4,
                'impact': 0.6,
                'mitigation': 'Build buffer time, identify parallel execution opportunities'
            })
        
        # Calculate overall risk score
        all_risks = (risks['technical_risks'] + risks['resource_risks'] + 
                    risks['timeline_risks'] + risks['external_risks'])
        
        if all_risks:
            risk_scores = [r['probability'] * r['impact'] for r in all_risks]
            risks['overall_risk_score'] = np.mean(risk_scores)
        
        return risks
    
    def _calculate_resource_requirements(self, phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource requirements for the strategic plan"""
        requirements = {
            'human_resources': {},
            'infrastructure': {},
            'budget': {},
            'timeline': {}
        }
        
        # Human resources
        total_weeks = sum(p['duration_weeks'] for p in phases)
        requirements['human_resources'] = {
            'engineers': max(2, total_weeks // 4),
            'project_managers': 1,
            'specialists': len([g for g in sum([p['goals'] for p in phases], []) 
                              if g.get('requires_specialist', False)])
        }
        
        # Infrastructure
        requirements['infrastructure'] = {
            'compute_resources': 'Standard development environment',
            'storage_requirements': '100GB for development, 1TB for production',
            'network_requirements': 'High-speed internet, VPN access'
        }
        
        # Budget (simplified estimation)
        engineer_weeks = requirements['human_resources']['engineers'] * total_weeks
        requirements['budget'] = {
            'personnel_cost': engineer_weeks * 2000,  # $2000 per engineer-week
            'infrastructure_cost': total_weeks * 500,  # $500 per week
            'contingency': 0.2  # 20% contingency
        }
        
        # Timeline
        requirements['timeline'] = {
            'total_duration_weeks': total_weeks,
            'critical_path': [p['id'] for p in phases],
            'parallel_opportunities': []
        }
        
        return requirements

class AdaptiveLearningEngine:
    """Main adaptive learning engine that coordinates all learning components"""
    
    def __init__(self):
        self.experience_buffer = ExperienceBuffer()
        self.pattern_recognizer = PatternRecognizer()
        self.performance_predictor = PerformancePredictor()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        self.strategic_planner = StrategicPlanner()
        
        # Adaptation rules
        self.adaptation_rules = {}
        self.rule_performance = defaultdict(list)
        
        # Configuration
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.8
        self.retraining_interval = timedelta(hours=24)
        self.last_training = None
        
        # State management
        self.running = False
        self.learning_thread = None
        
        # Database connections
        self.redis_client = None
        self.postgres_conn = None
        self._init_connections()
        
    def _init_connections(self):
        """Initialize database connections"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            
        try:
            self.postgres_conn = psycopg2.connect(
                host="localhost",
                database="nexus_architect",
                user="nexus_user",
                password="nexus_password"
            )
            logger.info("PostgreSQL connection established")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
    
    def add_experience(self, experience: LearningExperience):
        """Add a new learning experience"""
        self.experience_buffer.add_experience(experience)
        
        # Store in database
        self._store_experience(experience)
        
        # Trigger adaptation if needed
        if self._should_adapt():
            self._trigger_adaptation()
    
    def _store_experience(self, experience: LearningExperience):
        """Store experience in database"""
        if not self.postgres_conn:
            return
        
        try:
            cursor = self.postgres_conn.cursor()
            cursor.execute("""
                INSERT INTO learning_experiences 
                (id, timestamp, context, action_taken, outcome, success, 
                 performance_metrics, feedback_score, learning_type, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                experience.id, experience.timestamp,
                json.dumps(experience.context), json.dumps(experience.action_taken),
                json.dumps(experience.outcome), experience.success,
                json.dumps(experience.performance_metrics), experience.feedback_score,
                experience.learning_type.value, json.dumps(experience.tags)
            ))
            self.postgres_conn.commit()
            cursor.close()
        except Exception as e:
            logger.warning(f"Failed to store experience in database: {e}")
    
    def _should_adapt(self) -> bool:
        """Determine if adaptation should be triggered"""
        # Check if enough new experiences have been collected
        recent_experiences = self.experience_buffer.get_recent_experiences(hours=1)
        
        if len(recent_experiences) < 10:
            return False
        
        # Check if performance has degraded
        recent_success_rate = sum(1 for exp in recent_experiences if exp.success) / len(recent_experiences)
        
        if recent_success_rate < self.adaptation_threshold:
            return True
        
        # Check if it's time for scheduled retraining
        if (self.last_training is None or 
            datetime.now() - self.last_training > self.retraining_interval):
            return True
        
        return False
    
    def _trigger_adaptation(self):
        """Trigger adaptation process"""
        logger.info("Triggering adaptation process")
        
        # Get recent experiences
        experiences = self.experience_buffer.get_recent_experiences(hours=24)
        
        if len(experiences) < 20:
            logger.warning("Insufficient experiences for adaptation")
            return
        
        # Identify patterns
        patterns = self.pattern_recognizer.identify_patterns(experiences)
        
        # Retrain prediction models
        self.performance_predictor.train_models(experiences)
        
        # Update adaptation rules
        self._update_adaptation_rules(patterns, experiences)
        
        self.last_training = datetime.now()
        logger.info("Adaptation process completed")
    
    def _update_adaptation_rules(self, patterns: Dict[str, Any], 
                               experiences: List[LearningExperience]):
        """Update adaptation rules based on patterns and experiences"""
        for pattern in patterns.get('patterns', []):
            if pattern['success_rate'] > 0.8:  # High success pattern
                rule_id = f"pattern_{pattern['cluster_id']}_rule"
                
                # Create adaptation rule
                rule = AdaptationRule(
                    id=rule_id,
                    condition=f"Context matches pattern {pattern['cluster_id']}",
                    action=f"Apply high-success pattern {pattern['cluster_id']}",
                    priority=int(pattern['success_rate'] * 10),
                    confidence=pattern['success_rate'],
                    success_rate=pattern['success_rate'],
                    usage_count=pattern['size'],
                    created_at=datetime.now()
                )
                
                self.adaptation_rules[rule_id] = rule
                logger.info(f"Created adaptation rule {rule_id} with {rule.success_rate:.2f} success rate")
    
    def make_complex_decision(self, scenario: DecisionScenario) -> DecisionResult:
        """Make a complex decision using multi-objective optimization"""
        start_time = time.time()
        
        # Get optimal solutions
        optimal_solutions = self.multi_objective_optimizer.optimize(scenario)
        
        if not optimal_solutions:
            return DecisionResult(
                scenario_id=scenario.id,
                selected_action={},
                confidence=0.0,
                reasoning="No viable solutions found",
                expected_outcomes={},
                risk_assessment={},
                alternative_actions=[],
                decision_time=time.time() - start_time
            )
        
        # Select best solution
        best_solution = optimal_solutions[0]
        
        # Predict outcomes
        predicted_outcomes = self.performance_predictor.predict_outcome(
            scenario.context,
            {}  # No current metrics for prediction
        )
        
        # Assess risks
        risk_assessment = self._assess_decision_risks(best_solution, scenario)
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(best_solution, scenario, predicted_outcomes)
        
        # Generate reasoning
        reasoning = self._generate_decision_reasoning(best_solution, scenario, optimal_solutions)
        
        decision_time = time.time() - start_time
        
        result = DecisionResult(
            scenario_id=scenario.id,
            selected_action=best_solution['action'],
            confidence=confidence,
            reasoning=reasoning,
            expected_outcomes=predicted_outcomes,
            risk_assessment=risk_assessment,
            alternative_actions=[sol['action'] for sol in optimal_solutions[1:5]],  # Top 5 alternatives
            decision_time=decision_time
        )
        
        # Store decision for learning
        self._store_decision(scenario, result)
        
        return result
    
    def _assess_decision_risks(self, solution: Dict[str, Any], 
                             scenario: DecisionScenario) -> Dict[str, float]:
        """Assess risks associated with a decision"""
        risks = {}
        
        # Implementation risk
        complexity = scenario.complexity
        risks['implementation_risk'] = complexity * 0.8
        
        # Performance risk
        performance_uncertainty = 1.0 - solution.get('weighted_score', 0.5)
        risks['performance_risk'] = performance_uncertainty * 0.6
        
        # Timeline risk
        urgency = scenario.urgency
        risks['timeline_risk'] = urgency * 0.7
        
        # Resource risk
        estimated_cost = solution['action'].get('estimated_cost', 50)
        risks['resource_risk'] = min(1.0, estimated_cost / 100.0)
        
        return risks
    
    def _calculate_decision_confidence(self, solution: Dict[str, Any], 
                                     scenario: DecisionScenario,
                                     predicted_outcomes: Dict[str, Any]) -> float:
        """Calculate confidence in the decision"""
        factors = []
        
        # Solution quality
        factors.append(solution.get('weighted_score', 0.5))
        
        # Prediction confidence
        if 'success' in predicted_outcomes:
            factors.append(predicted_outcomes['success'].get('probability', 0.5))
        
        # Scenario clarity
        clarity = 1.0 - scenario.complexity
        factors.append(clarity)
        
        # Historical performance
        similar_experiences = self._find_similar_experiences(scenario)
        if similar_experiences:
            success_rate = sum(1 for exp in similar_experiences if exp.success) / len(similar_experiences)
            factors.append(success_rate)
        
        return np.mean(factors) if factors else 0.5
    
    def _generate_decision_reasoning(self, solution: Dict[str, Any], 
                                   scenario: DecisionScenario,
                                   all_solutions: List[Dict[str, Any]]) -> str:
        """Generate human-readable reasoning for the decision"""
        reasoning_parts = []
        
        # Solution selection rationale
        reasoning_parts.append(f"Selected action with weighted score of {solution.get('weighted_score', 0):.2f}")
        
        # Objective performance
        objectives_text = ", ".join([obj.value.replace('_', ' ') for obj in scenario.objectives])
        reasoning_parts.append(f"Optimized for objectives: {objectives_text}")
        
        # Comparison with alternatives
        if len(all_solutions) > 1:
            second_best_score = all_solutions[1].get('weighted_score', 0)
            margin = solution.get('weighted_score', 0) - second_best_score
            reasoning_parts.append(f"Outperformed next best alternative by {margin:.2f}")
        
        # Risk considerations
        if scenario.risk_tolerance < 0.5:
            reasoning_parts.append("Conservative approach chosen due to low risk tolerance")
        elif scenario.risk_tolerance > 0.8:
            reasoning_parts.append("Aggressive approach chosen due to high risk tolerance")
        
        # Urgency considerations
        if scenario.urgency > 0.8:
            reasoning_parts.append("Fast execution prioritized due to high urgency")
        
        return ". ".join(reasoning_parts) + "."
    
    def _find_similar_experiences(self, scenario: DecisionScenario) -> List[LearningExperience]:
        """Find experiences similar to the current scenario"""
        # Simple similarity based on scenario type and context
        all_experiences = list(self.experience_buffer.experiences)
        
        similar = []
        for exp in all_experiences:
            # Check if context has similar keys
            context_similarity = len(set(exp.context.keys()) & set(scenario.context.keys()))
            if context_similarity > len(scenario.context) * 0.5:
                similar.append(exp)
        
        return similar[:10]  # Return top 10 similar experiences
    
    def _store_decision(self, scenario: DecisionScenario, result: DecisionResult):
        """Store decision for future learning"""
        if not self.postgres_conn:
            return
        
        try:
            cursor = self.postgres_conn.cursor()
            cursor.execute("""
                INSERT INTO decision_history 
                (scenario_id, scenario_type, selected_action, confidence, 
                 reasoning, expected_outcomes, risk_assessment, decision_time, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (scenario_id) DO NOTHING
            """, (
                scenario.id, scenario.scenario_type.value,
                json.dumps(result.selected_action), result.confidence,
                result.reasoning, json.dumps(result.expected_outcomes),
                json.dumps(result.risk_assessment), result.decision_time,
                result.timestamp
            ))
            self.postgres_conn.commit()
            cursor.close()
        except Exception as e:
            logger.warning(f"Failed to store decision in database: {e}")
    
    def create_strategic_plan(self, goals: List[Dict[str, Any]], 
                            constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Create a strategic plan using the strategic planner"""
        return self.strategic_planner.create_strategic_plan(goals, constraints)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status"""
        return {
            'total_experiences': len(self.experience_buffer.experiences),
            'recent_experiences_24h': len(self.experience_buffer.get_recent_experiences(24)),
            'successful_experiences': len(self.experience_buffer.get_successful_experiences()),
            'adaptation_rules': len(self.adaptation_rules),
            'models_trained': {
                'regression_models': len(self.performance_predictor.regression_models),
                'classification_models': len(self.performance_predictor.classification_models)
            },
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'learning_rate': self.learning_rate,
            'adaptation_threshold': self.adaptation_threshold,
            'timestamp': datetime.now().isoformat()
        }

# Flask API
app = Flask(__name__)
CORS(app)

# Global learning engine instance
learning_engine = AdaptiveLearningEngine()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'adaptive_learning_engine',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/learning/status', methods=['GET'])
def get_learning_status():
    """Get learning system status"""
    return jsonify(learning_engine.get_learning_status())

@app.route('/learning/experiences', methods=['POST'])
def add_experience():
    """Add a new learning experience"""
    data = request.get_json()
    
    try:
        experience = LearningExperience(
            id=data.get('id', str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            context=data['context'],
            action_taken=data['action_taken'],
            outcome=data['outcome'],
            success=data['success'],
            performance_metrics=data.get('performance_metrics', {}),
            feedback_score=data.get('feedback_score', 0.5),
            learning_type=LearningType(data.get('learning_type', 'supervised')),
            tags=data.get('tags', [])
        )
        
        learning_engine.add_experience(experience)
        
        return jsonify({
            'message': f'Experience {experience.id} added successfully',
            'experience_id': experience.id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Invalid experience data: {str(e)}'}), 400

@app.route('/learning/patterns', methods=['GET'])
def get_patterns():
    """Get identified patterns from experiences"""
    hours = request.args.get('hours', 24, type=int)
    experiences = learning_engine.experience_buffer.get_recent_experiences(hours)
    
    patterns = learning_engine.pattern_recognizer.identify_patterns(experiences)
    
    return jsonify({
        'patterns': patterns,
        'analysis_period_hours': hours,
        'experiences_analyzed': len(experiences),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/learning/predict', methods=['POST'])
def predict_outcome():
    """Predict outcome for given context and metrics"""
    data = request.get_json()
    
    context = data.get('context', {})
    performance_metrics = data.get('performance_metrics', {})
    
    predictions = learning_engine.performance_predictor.predict_outcome(context, performance_metrics)
    
    return jsonify({
        'predictions': predictions,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/decisions', methods=['POST'])
def make_decision():
    """Make a complex decision"""
    data = request.get_json()
    
    try:
        scenario = DecisionScenario(
            id=data.get('id', str(uuid.uuid4())),
            scenario_type=DecisionType(data['scenario_type']),
            description=data['description'],
            context=data['context'],
            objectives=[ObjectiveType(obj) for obj in data['objectives']],
            constraints=data.get('constraints', {}),
            available_actions=data['available_actions'],
            urgency=data.get('urgency', 0.5),
            complexity=data.get('complexity', 0.5),
            risk_tolerance=data.get('risk_tolerance', 0.5)
        )
        
        result = learning_engine.make_complex_decision(scenario)
        
        return jsonify({
            'decision_result': asdict(result),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Invalid decision scenario: {str(e)}'}), 400

@app.route('/strategic-plans', methods=['POST'])
def create_strategic_plan():
    """Create a strategic plan"""
    data = request.get_json()
    
    goals = data.get('goals', [])
    constraints = data.get('constraints', {})
    
    plan = learning_engine.create_strategic_plan(goals, constraints)
    
    return jsonify({
        'strategic_plan': plan,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/learning/adaptation-rules', methods=['GET'])
def get_adaptation_rules():
    """Get current adaptation rules"""
    rules_data = {}
    for rule_id, rule in learning_engine.adaptation_rules.items():
        rules_data[rule_id] = asdict(rule)
    
    return jsonify({
        'adaptation_rules': rules_data,
        'count': len(rules_data),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/learning/retrain', methods=['POST'])
def trigger_retraining():
    """Manually trigger model retraining"""
    learning_engine._trigger_adaptation()
    
    return jsonify({
        'message': 'Model retraining triggered successfully',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8071, debug=False)

