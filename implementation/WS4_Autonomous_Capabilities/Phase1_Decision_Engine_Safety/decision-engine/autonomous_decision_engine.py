"""
Autonomous Decision Engine for Nexus Architect
Implements multi-criteria decision analysis with weighted scoring, AHP, and TOPSIS methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
from datetime import datetime, timedelta
import asyncio
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions the engine can make"""
    CODE_CHANGE = "code_change"
    INFRASTRUCTURE_CHANGE = "infrastructure_change"
    PROCESS_IMPROVEMENT = "process_improvement"
    SECURITY_ACTION = "security_action"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"

class RiskLevel(Enum):
    """Risk levels for decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConfidenceLevel(Enum):
    """Confidence levels for decisions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class DecisionCriteria:
    """Criteria for decision evaluation"""
    name: str
    weight: float
    maximize: bool = True  # True if higher values are better
    description: str = ""

@dataclass
class DecisionAlternative:
    """Alternative solution for a decision"""
    id: str
    name: str
    description: str
    attributes: Dict[str, float]
    estimated_impact: Dict[str, Any]
    implementation_cost: float
    implementation_time: int  # in hours
    rollback_complexity: float  # 0-1 scale

@dataclass
class DecisionContext:
    """Context information for decision making"""
    decision_id: str
    decision_type: DecisionType
    description: str
    requester: str
    urgency: str  # low, medium, high, critical
    business_context: Dict[str, Any]
    technical_context: Dict[str, Any]
    constraints: List[str]
    deadline: Optional[datetime] = None

@dataclass
class DecisionResult:
    """Result of a decision analysis"""
    decision_id: str
    selected_alternative: str
    confidence_score: float
    risk_level: RiskLevel
    reasoning: str
    scores: Dict[str, float]
    requires_approval: bool
    estimated_impact: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    timestamp: datetime

class WeightedScoringModel:
    """Weighted scoring model for decision analysis"""
    
    def __init__(self, criteria: List[DecisionCriteria]):
        self.criteria = criteria
        self.scaler = MinMaxScaler()
        
    def evaluate_alternatives(self, alternatives: List[DecisionAlternative]) -> Dict[str, float]:
        """Evaluate alternatives using weighted scoring"""
        if not alternatives:
            return {}
            
        # Extract attribute matrix
        attribute_names = list(alternatives[0].attributes.keys())
        attribute_matrix = np.array([
            [alt.attributes.get(attr, 0) for attr in attribute_names]
            for alt in alternatives
        ])
        
        # Normalize attributes
        normalized_matrix = self.scaler.fit_transform(attribute_matrix)
        
        # Calculate weighted scores
        scores = {}
        for i, alternative in enumerate(alternatives):
            score = 0
            for j, criterion in enumerate(self.criteria):
                if criterion.name in attribute_names:
                    attr_index = attribute_names.index(criterion.name)
                    attr_value = normalized_matrix[i][attr_index]
                    
                    # Invert score if we want to minimize this criterion
                    if not criterion.maximize:
                        attr_value = 1 - attr_value
                        
                    score += criterion.weight * attr_value
                    
            scores[alternative.id] = score
            
        return scores

class AHPAnalyzer:
    """Analytic Hierarchy Process for complex decision making"""
    
    def __init__(self):
        self.consistency_threshold = 0.1
        
    def create_pairwise_matrix(self, criteria: List[DecisionCriteria]) -> np.ndarray:
        """Create pairwise comparison matrix from criteria weights"""
        n = len(criteria)
        matrix = np.ones((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate pairwise comparison based on weights
                    ratio = criteria[i].weight / criteria[j].weight
                    matrix[i][j] = ratio
                    matrix[j][i] = 1 / ratio
                    
        return matrix
    
    def calculate_consistency_ratio(self, matrix: np.ndarray) -> float:
        """Calculate consistency ratio for pairwise comparison matrix"""
        n = matrix.shape[0]
        eigenvalues = np.linalg.eigvals(matrix)
        lambda_max = np.real(eigenvalues[0])
        
        # Random consistency index values
        ri_values = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_values.get(n, 1.49)
        
        ci = (lambda_max - n) / (n - 1)
        cr = ci / ri if ri > 0 else 0
        
        return cr
    
    def calculate_priority_vector(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate priority vector from pairwise comparison matrix"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_eigenvalue_index = np.argmax(eigenvalues)
        priority_vector = np.real(eigenvectors[:, max_eigenvalue_index])
        
        # Normalize to sum to 1
        priority_vector = priority_vector / np.sum(priority_vector)
        
        return priority_vector

class TOPSISAnalyzer:
    """TOPSIS method for multi-attribute decision making"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def analyze(self, alternatives: List[DecisionAlternative], 
                criteria: List[DecisionCriteria]) -> Dict[str, float]:
        """Perform TOPSIS analysis on alternatives"""
        if not alternatives:
            return {}
            
        # Create decision matrix
        attribute_names = [c.name for c in criteria]
        decision_matrix = np.array([
            [alt.attributes.get(attr, 0) for attr in attribute_names]
            for alt in alternatives
        ])
        
        # Normalize decision matrix
        normalized_matrix = self.scaler.fit_transform(decision_matrix)
        
        # Apply weights
        weights = np.array([c.weight for c in criteria])
        weighted_matrix = normalized_matrix * weights
        
        # Determine ideal and negative-ideal solutions
        ideal_solution = np.zeros(len(criteria))
        negative_ideal_solution = np.zeros(len(criteria))
        
        for i, criterion in enumerate(criteria):
            if criterion.maximize:
                ideal_solution[i] = np.max(weighted_matrix[:, i])
                negative_ideal_solution[i] = np.min(weighted_matrix[:, i])
            else:
                ideal_solution[i] = np.min(weighted_matrix[:, i])
                negative_ideal_solution[i] = np.max(weighted_matrix[:, i])
        
        # Calculate distances
        distances_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
        distances_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))
        
        # Calculate TOPSIS scores
        scores = {}
        for i, alternative in enumerate(alternatives):
            score = distances_to_negative_ideal[i] / (distances_to_ideal[i] + distances_to_negative_ideal[i])
            scores[alternative.id] = score
            
        return scores

class FuzzyLogicProcessor:
    """Fuzzy logic system for handling uncertainty and imprecision"""
    
    def __init__(self):
        self.membership_functions = {}
        
    def triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    def trapezoidal_membership(self, x: float, a: float, b: float, c: float, d: float) -> float:
        """Trapezoidal membership function"""
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        else:
            return (d - x) / (d - c)
    
    def fuzzify_confidence(self, confidence: float) -> Dict[str, float]:
        """Fuzzify confidence score into linguistic terms"""
        return {
            'low': self.trapezoidal_membership(confidence, 0, 0, 0.3, 0.5),
            'medium': self.triangular_membership(confidence, 0.3, 0.5, 0.7),
            'high': self.triangular_membership(confidence, 0.5, 0.7, 0.9),
            'very_high': self.trapezoidal_membership(confidence, 0.7, 0.9, 1.0, 1.0)
        }
    
    def fuzzify_risk(self, risk_score: float) -> Dict[str, float]:
        """Fuzzify risk score into linguistic terms"""
        return {
            'low': self.trapezoidal_membership(risk_score, 0, 0, 0.25, 0.4),
            'medium': self.triangular_membership(risk_score, 0.25, 0.5, 0.75),
            'high': self.triangular_membership(risk_score, 0.6, 0.8, 1.0),
            'critical': self.trapezoidal_membership(risk_score, 0.8, 1.0, 1.0, 1.0)
        }
    
    def defuzzify(self, fuzzy_set: Dict[str, float]) -> float:
        """Defuzzify using centroid method"""
        numerator = sum(value * i for i, (key, value) in enumerate(fuzzy_set.items()))
        denominator = sum(fuzzy_set.values())
        
        return numerator / denominator if denominator > 0 else 0

class AutonomousDecisionEngine:
    """Main autonomous decision engine"""
    
    def __init__(self):
        self.weighted_scoring = None
        self.ahp_analyzer = AHPAnalyzer()
        self.topsis_analyzer = TOPSISAnalyzer()
        self.fuzzy_processor = FuzzyLogicProcessor()
        self.decision_history = []
        self.ml_model = None
        self.confidence_threshold = 0.7
        self.risk_threshold = 0.6
        
    def load_ml_model(self, model_path: str):
        """Load pre-trained ML model for decision support"""
        try:
            self.ml_model = joblib.load(model_path)
            logger.info(f"ML model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")
    
    def train_ml_model(self, training_data: List[Dict]):
        """Train ML model on historical decision data"""
        if not training_data:
            return
            
        # Prepare training data
        features = []
        labels = []
        
        for data in training_data:
            feature_vector = [
                data.get('urgency_score', 0),
                data.get('complexity_score', 0),
                data.get('impact_score', 0),
                data.get('cost_score', 0),
                data.get('time_score', 0)
            ]
            features.append(feature_vector)
            labels.append(data.get('success', 0))
        
        # Train model
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(features, labels)
        
        logger.info("ML model trained on historical decision data")
    
    def predict_success_probability(self, alternative: DecisionAlternative) -> float:
        """Predict success probability using ML model"""
        if not self.ml_model:
            return 0.5  # Default probability
            
        feature_vector = [[
            alternative.attributes.get('urgency_score', 0),
            alternative.attributes.get('complexity_score', 0),
            alternative.attributes.get('impact_score', 0),
            alternative.implementation_cost / 10000,  # Normalize cost
            alternative.implementation_time / 100     # Normalize time
        ]]
        
        try:
            probability = self.ml_model.predict_proba(feature_vector)[0][1]
            return probability
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return 0.5
    
    def evaluate_decision(self, context: DecisionContext, 
                         alternatives: List[DecisionAlternative],
                         criteria: List[DecisionCriteria]) -> DecisionResult:
        """Evaluate decision using multiple methods"""
        
        # Initialize weighted scoring model
        self.weighted_scoring = WeightedScoringModel(criteria)
        
        # Method 1: Weighted Scoring
        weighted_scores = self.weighted_scoring.evaluate_alternatives(alternatives)
        
        # Method 2: TOPSIS Analysis
        topsis_scores = self.topsis_analyzer.analyze(alternatives, criteria)
        
        # Method 3: AHP Analysis (if applicable)
        ahp_scores = {}
        if len(criteria) >= 3:
            pairwise_matrix = self.ahp_analyzer.create_pairwise_matrix(criteria)
            consistency_ratio = self.ahp_analyzer.calculate_consistency_ratio(pairwise_matrix)
            
            if consistency_ratio <= self.ahp_analyzer.consistency_threshold:
                priority_vector = self.ahp_analyzer.calculate_priority_vector(pairwise_matrix)
                # Use priority vector to weight TOPSIS scores
                for alt_id, score in topsis_scores.items():
                    ahp_scores[alt_id] = score * np.mean(priority_vector)
            else:
                logger.warning(f"AHP consistency ratio too high: {consistency_ratio}")
                ahp_scores = topsis_scores
        else:
            ahp_scores = topsis_scores
        
        # Combine scores using ensemble method
        combined_scores = {}
        for alt_id in weighted_scores.keys():
            ws_score = weighted_scores.get(alt_id, 0)
            topsis_score = topsis_scores.get(alt_id, 0)
            ahp_score = ahp_scores.get(alt_id, 0)
            
            # Weighted combination
            combined_score = 0.4 * ws_score + 0.4 * topsis_score + 0.2 * ahp_score
            combined_scores[alt_id] = combined_score
        
        # Select best alternative
        best_alternative_id = max(combined_scores.keys(), key=lambda k: combined_scores[k])
        best_alternative = next(alt for alt in alternatives if alt.id == best_alternative_id)
        
        # Calculate confidence using fuzzy logic
        max_score = max(combined_scores.values())
        score_variance = np.var(list(combined_scores.values()))
        confidence_score = max_score * (1 - score_variance)
        
        # ML-based success prediction
        ml_confidence = self.predict_success_probability(best_alternative)
        
        # Combine confidences
        final_confidence = 0.7 * confidence_score + 0.3 * ml_confidence
        
        # Fuzzify confidence
        fuzzy_confidence = self.fuzzy_processor.fuzzify_confidence(final_confidence)
        confidence_level = max(fuzzy_confidence.keys(), key=lambda k: fuzzy_confidence[k])
        
        # Calculate risk level
        risk_factors = [
            best_alternative.implementation_cost / 50000,  # Normalize cost risk
            best_alternative.rollback_complexity,
            1 - final_confidence,
            len(context.constraints) / 10  # Constraint complexity
        ]
        risk_score = np.mean(risk_factors)
        
        # Fuzzify risk
        fuzzy_risk = self.fuzzy_processor.fuzzify_risk(risk_score)
        risk_level_str = max(fuzzy_risk.keys(), key=lambda k: fuzzy_risk[k])
        risk_level = RiskLevel(risk_level_str)
        
        # Determine if approval is required
        requires_approval = (
            final_confidence < self.confidence_threshold or
            risk_score > self.risk_threshold or
            risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
            context.urgency == "critical"
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            best_alternative, combined_scores, final_confidence, 
            risk_level, context
        )
        
        # Create rollback plan
        rollback_plan = self._create_rollback_plan(best_alternative, context)
        
        # Create decision result
        result = DecisionResult(
            decision_id=context.decision_id,
            selected_alternative=best_alternative_id,
            confidence_score=final_confidence,
            risk_level=risk_level,
            reasoning=reasoning,
            scores=combined_scores,
            requires_approval=requires_approval,
            estimated_impact=best_alternative.estimated_impact,
            rollback_plan=rollback_plan,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.decision_history.append(result)
        
        return result
    
    def _generate_reasoning(self, alternative: DecisionAlternative, 
                          scores: Dict[str, float], confidence: float,
                          risk_level: RiskLevel, context: DecisionContext) -> str:
        """Generate human-readable reasoning for the decision"""
        
        reasoning_parts = [
            f"Selected alternative '{alternative.name}' based on multi-criteria analysis.",
            f"Decision confidence: {confidence:.2f} ({confidence*100:.1f}%)",
            f"Risk level: {risk_level.value}",
            f"Alternative scored {scores[alternative.id]:.3f} using combined weighted scoring, TOPSIS, and AHP methods."
        ]
        
        if alternative.implementation_cost > 10000:
            reasoning_parts.append(f"High implementation cost (${alternative.implementation_cost:,.2f}) considered in risk assessment.")
        
        if alternative.implementation_time > 40:
            reasoning_parts.append(f"Extended implementation time ({alternative.implementation_time} hours) factored into decision.")
        
        if context.urgency == "critical":
            reasoning_parts.append("Critical urgency level requires immediate attention and approval.")
        
        if len(context.constraints) > 3:
            reasoning_parts.append(f"Multiple constraints ({len(context.constraints)}) increase implementation complexity.")
        
        return " ".join(reasoning_parts)
    
    def _create_rollback_plan(self, alternative: DecisionAlternative, 
                            context: DecisionContext) -> Dict[str, Any]:
        """Create rollback plan for the selected alternative"""
        
        rollback_plan = {
            "rollback_complexity": alternative.rollback_complexity,
            "estimated_rollback_time": alternative.implementation_time * 0.3,  # 30% of implementation time
            "rollback_triggers": [
                "Performance degradation > 20%",
                "Error rate increase > 10%",
                "User satisfaction drop > 15%",
                "Security vulnerability detected"
            ],
            "rollback_steps": [],
            "validation_checks": [],
            "emergency_contacts": []
        }
        
        # Add decision-type specific rollback steps
        if context.decision_type == DecisionType.CODE_CHANGE:
            rollback_plan["rollback_steps"] = [
                "Revert code changes using version control",
                "Redeploy previous stable version",
                "Validate system functionality",
                "Monitor for 30 minutes post-rollback"
            ]
            rollback_plan["validation_checks"] = [
                "Unit tests pass",
                "Integration tests pass",
                "Performance benchmarks met",
                "No new errors in logs"
            ]
        
        elif context.decision_type == DecisionType.INFRASTRUCTURE_CHANGE:
            rollback_plan["rollback_steps"] = [
                "Restore previous infrastructure configuration",
                "Restart affected services",
                "Validate connectivity and performance",
                "Monitor system stability"
            ]
            rollback_plan["validation_checks"] = [
                "All services responding",
                "Network connectivity restored",
                "Performance metrics normal",
                "No infrastructure alerts"
            ]
        
        elif context.decision_type == DecisionType.PERFORMANCE_OPTIMIZATION:
            rollback_plan["rollback_steps"] = [
                "Disable performance optimizations",
                "Restore previous configuration",
                "Clear caches if necessary",
                "Monitor performance metrics"
            ]
            rollback_plan["validation_checks"] = [
                "Response times within acceptable range",
                "Resource utilization normal",
                "No performance degradation",
                "User experience maintained"
            ]
        
        return rollback_plan
    
    def get_decision_history(self, limit: int = 100) -> List[DecisionResult]:
        """Get recent decision history"""
        return self.decision_history[-limit:]
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about decision making performance"""
        if not self.decision_history:
            return {}
        
        total_decisions = len(self.decision_history)
        approved_decisions = sum(1 for d in self.decision_history if not d.requires_approval)
        high_confidence_decisions = sum(1 for d in self.decision_history if d.confidence_score > 0.8)
        
        risk_distribution = {}
        for risk_level in RiskLevel:
            count = sum(1 for d in self.decision_history if d.risk_level == risk_level)
            risk_distribution[risk_level.value] = count
        
        avg_confidence = np.mean([d.confidence_score for d in self.decision_history])
        
        return {
            "total_decisions": total_decisions,
            "auto_approved_rate": approved_decisions / total_decisions,
            "high_confidence_rate": high_confidence_decisions / total_decisions,
            "average_confidence": avg_confidence,
            "risk_distribution": risk_distribution,
            "last_updated": datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize decision engine
    engine = AutonomousDecisionEngine()
    
    # Define criteria for code change decisions
    criteria = [
        DecisionCriteria("impact", 0.3, True, "Business impact of the change"),
        DecisionCriteria("complexity", 0.2, False, "Implementation complexity"),
        DecisionCriteria("risk", 0.25, False, "Risk of introducing bugs"),
        DecisionCriteria("urgency", 0.15, True, "Urgency of the change"),
        DecisionCriteria("maintainability", 0.1, True, "Long-term maintainability")
    ]
    
    # Define alternatives for a bug fix decision
    alternatives = [
        DecisionAlternative(
            id="quick_fix",
            name="Quick Patch",
            description="Apply minimal fix to resolve immediate issue",
            attributes={
                "impact": 0.6,
                "complexity": 0.3,
                "risk": 0.4,
                "urgency": 0.9,
                "maintainability": 0.4
            },
            estimated_impact={"users_affected": 1000, "downtime_minutes": 5},
            implementation_cost=2000,
            implementation_time=4,
            rollback_complexity=0.2
        ),
        DecisionAlternative(
            id="comprehensive_fix",
            name="Comprehensive Solution",
            description="Implement thorough fix addressing root cause",
            attributes={
                "impact": 0.9,
                "complexity": 0.8,
                "risk": 0.3,
                "urgency": 0.6,
                "maintainability": 0.9
            },
            estimated_impact={"users_affected": 1000, "downtime_minutes": 30},
            implementation_cost=8000,
            implementation_time=16,
            rollback_complexity=0.6
        ),
        DecisionAlternative(
            id="workaround",
            name="Temporary Workaround",
            description="Implement temporary solution while planning proper fix",
            attributes={
                "impact": 0.4,
                "complexity": 0.2,
                "risk": 0.5,
                "urgency": 0.8,
                "maintainability": 0.3
            },
            estimated_impact={"users_affected": 1000, "downtime_minutes": 2},
            implementation_cost=1000,
            implementation_time=2,
            rollback_complexity=0.1
        )
    ]
    
    # Create decision context
    context = DecisionContext(
        decision_id="BUG-2025-001",
        decision_type=DecisionType.CODE_CHANGE,
        description="Critical bug in user authentication system",
        requester="security_team",
        urgency="high",
        business_context={"affected_users": 1000, "revenue_impact": 5000},
        technical_context={"component": "auth_service", "severity": "critical"},
        constraints=["must_maintain_uptime", "security_compliance", "data_integrity"],
        deadline=datetime.now() + timedelta(hours=8)
    )
    
    # Evaluate decision
    result = engine.evaluate_decision(context, alternatives, criteria)
    
    # Print results
    print(f"Decision ID: {result.decision_id}")
    print(f"Selected Alternative: {result.selected_alternative}")
    print(f"Confidence Score: {result.confidence_score:.3f}")
    print(f"Risk Level: {result.risk_level.value}")
    print(f"Requires Approval: {result.requires_approval}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Scores: {result.scores}")
    print(f"Rollback Plan: {json.dumps(result.rollback_plan, indent=2)}")
    
    # Get statistics
    stats = engine.get_decision_statistics()
    print(f"Decision Statistics: {json.dumps(stats, indent=2)}")

