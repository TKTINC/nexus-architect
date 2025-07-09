"""
Risk Assessment and Management System for Nexus Architect
Implements comprehensive risk analysis, impact assessment, and mitigation strategies
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
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    """Categories of risk"""
    TECHNICAL = "technical"
    SECURITY = "security"
    OPERATIONAL = "operational"
    BUSINESS = "business"
    COMPLIANCE = "compliance"
    FINANCIAL = "financial"

class ImpactLevel(Enum):
    """Impact levels for risk assessment"""
    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"

class Probability(Enum):
    """Probability levels for risk occurrence"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class RiskFactor:
    """Individual risk factor"""
    factor_id: str
    name: str
    category: RiskCategory
    description: str
    probability: Probability
    impact: ImpactLevel
    weight: float
    mitigation_strategies: List[str]
    detection_methods: List[str]

@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    assessment_id: str
    decision_id: str
    overall_risk_score: float
    risk_level: str
    risk_factors: List[RiskFactor]
    impact_analysis: Dict[str, Any]
    mitigation_plan: Dict[str, Any]
    monitoring_requirements: List[str]
    contingency_plans: List[str]
    timestamp: datetime

@dataclass
class ImpactMetrics:
    """Impact metrics for different dimensions"""
    performance_impact: float  # 0-1 scale
    availability_impact: float
    security_impact: float
    user_experience_impact: float
    financial_impact: float
    compliance_impact: float

class RiskCalculator:
    """Calculates risk scores using multiple methodologies"""
    
    def __init__(self):
        self.probability_scores = {
            Probability.VERY_LOW: 0.1,
            Probability.LOW: 0.3,
            Probability.MEDIUM: 0.5,
            Probability.HIGH: 0.7,
            Probability.VERY_HIGH: 0.9
        }
        
        self.impact_scores = {
            ImpactLevel.NEGLIGIBLE: 0.1,
            ImpactLevel.MINOR: 0.3,
            ImpactLevel.MODERATE: 0.5,
            ImpactLevel.MAJOR: 0.7,
            ImpactLevel.SEVERE: 0.9
        }
    
    def calculate_risk_score(self, probability: Probability, impact: ImpactLevel, weight: float = 1.0) -> float:
        """Calculate risk score using probability × impact × weight"""
        prob_score = self.probability_scores[probability]
        impact_score = self.impact_scores[impact]
        return prob_score * impact_score * weight
    
    def calculate_composite_risk(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate composite risk score from multiple factors"""
        if not risk_factors:
            return 0.0
        
        # Calculate weighted average of individual risk scores
        total_weighted_risk = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            risk_score = self.calculate_risk_score(factor.probability, factor.impact, factor.weight)
            total_weighted_risk += risk_score
            total_weight += factor.weight
        
        return total_weighted_risk / total_weight if total_weight > 0 else 0.0
    
    def calculate_category_risks(self, risk_factors: List[RiskFactor]) -> Dict[str, float]:
        """Calculate risk scores by category"""
        category_risks = {}
        
        for category in RiskCategory:
            category_factors = [f for f in risk_factors if f.category == category]
            if category_factors:
                category_risks[category.value] = self.calculate_composite_risk(category_factors)
            else:
                category_risks[category.value] = 0.0
        
        return category_risks

class ImpactAnalyzer:
    """Analyzes potential impact of decisions across multiple dimensions"""
    
    def __init__(self):
        self.impact_models = {}
        self.historical_data = []
    
    def analyze_impact(self, decision_context: Dict[str, Any], 
                      proposed_changes: Dict[str, Any]) -> ImpactMetrics:
        """Analyze potential impact across multiple dimensions"""
        
        # Extract relevant features
        features = self._extract_impact_features(decision_context, proposed_changes)
        
        # Calculate impact scores for each dimension
        performance_impact = self._calculate_performance_impact(features)
        availability_impact = self._calculate_availability_impact(features)
        security_impact = self._calculate_security_impact(features)
        user_experience_impact = self._calculate_user_experience_impact(features)
        financial_impact = self._calculate_financial_impact(features)
        compliance_impact = self._calculate_compliance_impact(features)
        
        return ImpactMetrics(
            performance_impact=performance_impact,
            availability_impact=availability_impact,
            security_impact=security_impact,
            user_experience_impact=user_experience_impact,
            financial_impact=financial_impact,
            compliance_impact=compliance_impact
        )
    
    def _extract_impact_features(self, decision_context: Dict[str, Any], 
                                proposed_changes: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for impact analysis"""
        features = {}
        
        # Decision context features
        features['urgency_score'] = self._map_urgency_to_score(decision_context.get('urgency', 'medium'))
        features['complexity_score'] = len(decision_context.get('constraints', [])) / 10.0
        features['scope_score'] = len(proposed_changes.get('affected_components', [])) / 20.0
        
        # Change characteristics
        features['code_change_size'] = len(proposed_changes.get('code', '')) / 10000.0
        features['infrastructure_changes'] = len(proposed_changes.get('infrastructure_changes', [])) / 10.0
        features['database_changes'] = 1.0 if proposed_changes.get('database_changes') else 0.0
        features['api_changes'] = 1.0 if proposed_changes.get('api_changes') else 0.0
        
        # Risk indicators
        features['security_sensitive'] = 1.0 if proposed_changes.get('security_sensitive') else 0.0
        features['user_facing'] = 1.0 if proposed_changes.get('user_facing') else 0.0
        features['critical_path'] = 1.0 if proposed_changes.get('critical_path') else 0.0
        
        return features
    
    def _map_urgency_to_score(self, urgency: str) -> float:
        """Map urgency level to numerical score"""
        urgency_mapping = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 1.0
        }
        return urgency_mapping.get(urgency.lower(), 0.5)
    
    def _calculate_performance_impact(self, features: Dict[str, float]) -> float:
        """Calculate potential performance impact"""
        # Weighted combination of relevant features
        impact = (
            features.get('code_change_size', 0) * 0.3 +
            features.get('infrastructure_changes', 0) * 0.4 +
            features.get('database_changes', 0) * 0.2 +
            features.get('critical_path', 0) * 0.1
        )
        return min(impact, 1.0)
    
    def _calculate_availability_impact(self, features: Dict[str, float]) -> float:
        """Calculate potential availability impact"""
        impact = (
            features.get('infrastructure_changes', 0) * 0.4 +
            features.get('database_changes', 0) * 0.3 +
            features.get('critical_path', 0) * 0.2 +
            features.get('urgency_score', 0) * 0.1
        )
        return min(impact, 1.0)
    
    def _calculate_security_impact(self, features: Dict[str, float]) -> float:
        """Calculate potential security impact"""
        impact = (
            features.get('security_sensitive', 0) * 0.5 +
            features.get('api_changes', 0) * 0.2 +
            features.get('user_facing', 0) * 0.2 +
            features.get('code_change_size', 0) * 0.1
        )
        return min(impact, 1.0)
    
    def _calculate_user_experience_impact(self, features: Dict[str, float]) -> float:
        """Calculate potential user experience impact"""
        impact = (
            features.get('user_facing', 0) * 0.4 +
            features.get('api_changes', 0) * 0.3 +
            features.get('performance_impact', 0) * 0.2 +
            features.get('availability_impact', 0) * 0.1
        )
        return min(impact, 1.0)
    
    def _calculate_financial_impact(self, features: Dict[str, float]) -> float:
        """Calculate potential financial impact"""
        impact = (
            features.get('scope_score', 0) * 0.3 +
            features.get('urgency_score', 0) * 0.2 +
            features.get('user_facing', 0) * 0.3 +
            features.get('critical_path', 0) * 0.2
        )
        return min(impact, 1.0)
    
    def _calculate_compliance_impact(self, features: Dict[str, float]) -> float:
        """Calculate potential compliance impact"""
        impact = (
            features.get('security_sensitive', 0) * 0.4 +
            features.get('database_changes', 0) * 0.3 +
            features.get('user_facing', 0) * 0.2 +
            features.get('api_changes', 0) * 0.1
        )
        return min(impact, 1.0)

class MitigationPlanner:
    """Plans risk mitigation strategies"""
    
    def __init__(self):
        self.mitigation_strategies = self._load_mitigation_strategies()
    
    def _load_mitigation_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined mitigation strategies"""
        return {
            "technical": {
                "code_review": {
                    "description": "Comprehensive code review by senior developers",
                    "effectiveness": 0.8,
                    "cost": "low",
                    "time_required": 2
                },
                "automated_testing": {
                    "description": "Extensive automated testing including unit, integration, and e2e tests",
                    "effectiveness": 0.7,
                    "cost": "medium",
                    "time_required": 4
                },
                "gradual_rollout": {
                    "description": "Gradual rollout with feature flags and monitoring",
                    "effectiveness": 0.9,
                    "cost": "medium",
                    "time_required": 8
                },
                "rollback_plan": {
                    "description": "Detailed rollback plan with automated rollback triggers",
                    "effectiveness": 0.6,
                    "cost": "low",
                    "time_required": 1
                }
            },
            "security": {
                "security_audit": {
                    "description": "Comprehensive security audit by security team",
                    "effectiveness": 0.9,
                    "cost": "high",
                    "time_required": 8
                },
                "penetration_testing": {
                    "description": "Penetration testing of affected systems",
                    "effectiveness": 0.8,
                    "cost": "high",
                    "time_required": 16
                },
                "access_controls": {
                    "description": "Enhanced access controls and monitoring",
                    "effectiveness": 0.7,
                    "cost": "medium",
                    "time_required": 4
                }
            },
            "operational": {
                "monitoring_enhancement": {
                    "description": "Enhanced monitoring and alerting",
                    "effectiveness": 0.8,
                    "cost": "medium",
                    "time_required": 4
                },
                "backup_procedures": {
                    "description": "Enhanced backup and recovery procedures",
                    "effectiveness": 0.7,
                    "cost": "medium",
                    "time_required": 6
                },
                "staff_training": {
                    "description": "Additional staff training on new procedures",
                    "effectiveness": 0.6,
                    "cost": "low",
                    "time_required": 8
                }
            }
        }
    
    def create_mitigation_plan(self, risk_factors: List[RiskFactor], 
                              impact_metrics: ImpactMetrics,
                              constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive mitigation plan"""
        
        mitigation_plan = {
            "strategies": [],
            "timeline": {},
            "resource_requirements": {},
            "effectiveness_score": 0.0,
            "total_cost": "medium",
            "implementation_time": 0
        }
        
        # Group risk factors by category
        category_factors = {}
        for factor in risk_factors:
            category = factor.category.value
            if category not in category_factors:
                category_factors[category] = []
            category_factors[category].append(factor)
        
        # Select mitigation strategies for each category
        total_effectiveness = 0.0
        total_time = 0
        cost_levels = []
        
        for category, factors in category_factors.items():
            if category in self.mitigation_strategies:
                category_strategies = self.mitigation_strategies[category]
                
                # Select best strategies based on risk level and constraints
                selected_strategies = self._select_strategies(
                    category_strategies, factors, constraints
                )
                
                for strategy_name, strategy in selected_strategies.items():
                    mitigation_plan["strategies"].append({
                        "category": category,
                        "name": strategy_name,
                        "description": strategy["description"],
                        "effectiveness": strategy["effectiveness"],
                        "cost": strategy["cost"],
                        "time_required": strategy["time_required"]
                    })
                    
                    total_effectiveness += strategy["effectiveness"]
                    total_time += strategy["time_required"]
                    cost_levels.append(strategy["cost"])
        
        # Calculate overall metrics
        mitigation_plan["effectiveness_score"] = total_effectiveness / len(mitigation_plan["strategies"]) if mitigation_plan["strategies"] else 0.0
        mitigation_plan["implementation_time"] = total_time
        mitigation_plan["total_cost"] = self._aggregate_cost_levels(cost_levels)
        
        # Create timeline
        mitigation_plan["timeline"] = self._create_implementation_timeline(mitigation_plan["strategies"])
        
        # Calculate resource requirements
        mitigation_plan["resource_requirements"] = self._calculate_resource_requirements(mitigation_plan["strategies"])
        
        return mitigation_plan
    
    def _select_strategies(self, available_strategies: Dict[str, Any], 
                          risk_factors: List[RiskFactor],
                          constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate mitigation strategies"""
        selected = {}
        
        # Calculate average risk level for the category
        avg_risk = np.mean([
            self._calculate_factor_risk_score(factor) for factor in risk_factors
        ])
        
        # Select strategies based on risk level and constraints
        max_cost = constraints.get("max_cost", "high")
        max_time = constraints.get("max_time", 24)
        
        for strategy_name, strategy in available_strategies.items():
            # Check constraints
            if self._cost_level_to_number(strategy["cost"]) > self._cost_level_to_number(max_cost):
                continue
            if strategy["time_required"] > max_time:
                continue
            
            # Select high-effectiveness strategies for high-risk factors
            if avg_risk > 0.7 and strategy["effectiveness"] > 0.7:
                selected[strategy_name] = strategy
            elif avg_risk > 0.4 and strategy["effectiveness"] > 0.5:
                selected[strategy_name] = strategy
            elif avg_risk <= 0.4 and strategy["effectiveness"] > 0.3:
                selected[strategy_name] = strategy
        
        return selected
    
    def _calculate_factor_risk_score(self, factor: RiskFactor) -> float:
        """Calculate risk score for a single factor"""
        prob_scores = {
            Probability.VERY_LOW: 0.1, Probability.LOW: 0.3, Probability.MEDIUM: 0.5,
            Probability.HIGH: 0.7, Probability.VERY_HIGH: 0.9
        }
        impact_scores = {
            ImpactLevel.NEGLIGIBLE: 0.1, ImpactLevel.MINOR: 0.3, ImpactLevel.MODERATE: 0.5,
            ImpactLevel.MAJOR: 0.7, ImpactLevel.SEVERE: 0.9
        }
        
        return prob_scores[factor.probability] * impact_scores[factor.impact] * factor.weight
    
    def _cost_level_to_number(self, cost_level: str) -> int:
        """Convert cost level to number for comparison"""
        mapping = {"low": 1, "medium": 2, "high": 3}
        return mapping.get(cost_level, 2)
    
    def _aggregate_cost_levels(self, cost_levels: List[str]) -> str:
        """Aggregate multiple cost levels"""
        if not cost_levels:
            return "low"
        
        cost_numbers = [self._cost_level_to_number(level) for level in cost_levels]
        avg_cost = np.mean(cost_numbers)
        
        if avg_cost <= 1.5:
            return "low"
        elif avg_cost <= 2.5:
            return "medium"
        else:
            return "high"
    
    def _create_implementation_timeline(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create implementation timeline for strategies"""
        timeline = {
            "phases": [],
            "total_duration": 0,
            "critical_path": []
        }
        
        # Sort strategies by priority (effectiveness and urgency)
        sorted_strategies = sorted(
            strategies, 
            key=lambda s: s["effectiveness"], 
            reverse=True
        )
        
        current_time = 0
        for i, strategy in enumerate(sorted_strategies):
            phase = {
                "phase": i + 1,
                "strategy": strategy["name"],
                "start_time": current_time,
                "duration": strategy["time_required"],
                "end_time": current_time + strategy["time_required"]
            }
            timeline["phases"].append(phase)
            current_time += strategy["time_required"]
        
        timeline["total_duration"] = current_time
        timeline["critical_path"] = [phase["strategy"] for phase in timeline["phases"]]
        
        return timeline
    
    def _calculate_resource_requirements(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource requirements for strategies"""
        requirements = {
            "personnel": {
                "developers": 0,
                "security_experts": 0,
                "operations_staff": 0,
                "qa_engineers": 0
            },
            "tools": [],
            "infrastructure": []
        }
        
        # Map strategies to resource requirements
        strategy_resources = {
            "code_review": {"developers": 2, "tools": ["code_review_tool"]},
            "automated_testing": {"developers": 1, "qa_engineers": 2, "tools": ["testing_framework"]},
            "security_audit": {"security_experts": 2, "tools": ["security_scanner"]},
            "monitoring_enhancement": {"operations_staff": 1, "infrastructure": ["monitoring_system"]}
        }
        
        for strategy in strategies:
            strategy_name = strategy["name"]
            if strategy_name in strategy_resources:
                resources = strategy_resources[strategy_name]
                
                # Aggregate personnel requirements
                for role, count in resources.get("personnel", {}).items():
                    requirements["personnel"][role] = max(
                        requirements["personnel"].get(role, 0), count
                    )
                
                # Aggregate tool requirements
                requirements["tools"].extend(resources.get("tools", []))
                requirements["infrastructure"].extend(resources.get("infrastructure", []))
        
        # Remove duplicates
        requirements["tools"] = list(set(requirements["tools"]))
        requirements["infrastructure"] = list(set(requirements["infrastructure"]))
        
        return requirements

class RiskManager:
    """Main risk management coordinator"""
    
    def __init__(self):
        self.risk_calculator = RiskCalculator()
        self.impact_analyzer = ImpactAnalyzer()
        self.mitigation_planner = MitigationPlanner()
        self.risk_history = []
        
    def assess_risk(self, decision_context: Dict[str, Any], 
                   proposed_changes: Dict[str, Any]) -> RiskAssessment:
        """Perform comprehensive risk assessment"""
        
        # Generate risk factors
        risk_factors = self._generate_risk_factors(decision_context, proposed_changes)
        
        # Calculate overall risk score
        overall_risk_score = self.risk_calculator.calculate_composite_risk(risk_factors)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_risk_score)
        
        # Analyze impact
        impact_metrics = self.impact_analyzer.analyze_impact(decision_context, proposed_changes)
        
        # Create mitigation plan
        constraints = decision_context.get("constraints", {})
        mitigation_plan = self.mitigation_planner.create_mitigation_plan(
            risk_factors, impact_metrics, constraints
        )
        
        # Generate monitoring requirements
        monitoring_requirements = self._generate_monitoring_requirements(risk_factors, impact_metrics)
        
        # Generate contingency plans
        contingency_plans = self._generate_contingency_plans(risk_factors, impact_metrics)
        
        # Create assessment
        assessment = RiskAssessment(
            assessment_id=f"RISK-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            decision_id=decision_context.get("decision_id", "unknown"),
            overall_risk_score=overall_risk_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            impact_analysis=asdict(impact_metrics),
            mitigation_plan=mitigation_plan,
            monitoring_requirements=monitoring_requirements,
            contingency_plans=contingency_plans,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.risk_history.append(assessment)
        
        return assessment
    
    def _generate_risk_factors(self, decision_context: Dict[str, Any], 
                              proposed_changes: Dict[str, Any]) -> List[RiskFactor]:
        """Generate risk factors based on context and changes"""
        risk_factors = []
        
        # Technical risks
        if proposed_changes.get("code_changes"):
            risk_factors.append(RiskFactor(
                factor_id="tech_001",
                name="Code Change Risk",
                category=RiskCategory.TECHNICAL,
                description="Risk associated with code modifications",
                probability=self._assess_code_change_probability(proposed_changes),
                impact=self._assess_code_change_impact(proposed_changes),
                weight=0.8,
                mitigation_strategies=["code_review", "automated_testing", "gradual_rollout"],
                detection_methods=["unit_tests", "integration_tests", "monitoring"]
            ))
        
        # Security risks
        if proposed_changes.get("security_sensitive"):
            risk_factors.append(RiskFactor(
                factor_id="sec_001",
                name="Security Risk",
                category=RiskCategory.SECURITY,
                description="Risk of introducing security vulnerabilities",
                probability=Probability.MEDIUM,
                impact=ImpactLevel.MAJOR,
                weight=1.0,
                mitigation_strategies=["security_audit", "penetration_testing"],
                detection_methods=["security_scanning", "vulnerability_assessment"]
            ))
        
        # Operational risks
        if proposed_changes.get("infrastructure_changes"):
            risk_factors.append(RiskFactor(
                factor_id="ops_001",
                name="Infrastructure Risk",
                category=RiskCategory.OPERATIONAL,
                description="Risk of infrastructure changes affecting operations",
                probability=Probability.MEDIUM,
                impact=ImpactLevel.MODERATE,
                weight=0.7,
                mitigation_strategies=["monitoring_enhancement", "backup_procedures"],
                detection_methods=["infrastructure_monitoring", "health_checks"]
            ))
        
        # Business risks
        if decision_context.get("urgency") == "critical":
            risk_factors.append(RiskFactor(
                factor_id="bus_001",
                name="Urgency Risk",
                category=RiskCategory.BUSINESS,
                description="Risk of rushed implementation due to urgency",
                probability=Probability.HIGH,
                impact=ImpactLevel.MODERATE,
                weight=0.6,
                mitigation_strategies=["additional_review", "parallel_development"],
                detection_methods=["quality_metrics", "user_feedback"]
            ))
        
        return risk_factors
    
    def _assess_code_change_probability(self, proposed_changes: Dict[str, Any]) -> Probability:
        """Assess probability of issues from code changes"""
        code_size = len(proposed_changes.get("code", ""))
        complexity = len(proposed_changes.get("affected_components", []))
        
        if code_size > 5000 or complexity > 10:
            return Probability.HIGH
        elif code_size > 1000 or complexity > 5:
            return Probability.MEDIUM
        else:
            return Probability.LOW
    
    def _assess_code_change_impact(self, proposed_changes: Dict[str, Any]) -> ImpactLevel:
        """Assess impact of code changes"""
        if proposed_changes.get("critical_path"):
            return ImpactLevel.MAJOR
        elif proposed_changes.get("user_facing"):
            return ImpactLevel.MODERATE
        else:
            return ImpactLevel.MINOR
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _generate_monitoring_requirements(self, risk_factors: List[RiskFactor], 
                                        impact_metrics: ImpactMetrics) -> List[str]:
        """Generate monitoring requirements based on risks"""
        requirements = []
        
        # Add monitoring based on risk categories
        categories = set(factor.category for factor in risk_factors)
        
        if RiskCategory.TECHNICAL in categories:
            requirements.extend([
                "Application performance monitoring",
                "Error rate monitoring",
                "Response time tracking"
            ])
        
        if RiskCategory.SECURITY in categories:
            requirements.extend([
                "Security event monitoring",
                "Access pattern analysis",
                "Vulnerability scanning"
            ])
        
        if RiskCategory.OPERATIONAL in categories:
            requirements.extend([
                "Infrastructure health monitoring",
                "Resource utilization tracking",
                "Service availability monitoring"
            ])
        
        # Add monitoring based on impact levels
        if impact_metrics.user_experience_impact > 0.5:
            requirements.append("User experience metrics monitoring")
        
        if impact_metrics.financial_impact > 0.5:
            requirements.append("Business metrics monitoring")
        
        return list(set(requirements))  # Remove duplicates
    
    def _generate_contingency_plans(self, risk_factors: List[RiskFactor], 
                                  impact_metrics: ImpactMetrics) -> List[str]:
        """Generate contingency plans for high-risk scenarios"""
        plans = []
        
        # High-risk scenarios
        high_risk_factors = [f for f in risk_factors if 
                           self.risk_calculator.calculate_risk_score(f.probability, f.impact, f.weight) > 0.6]
        
        if high_risk_factors:
            plans.extend([
                "Immediate rollback procedure",
                "Emergency communication plan",
                "Escalation to senior management"
            ])
        
        # Impact-specific contingency plans
        if impact_metrics.availability_impact > 0.7:
            plans.append("Service failover to backup systems")
        
        if impact_metrics.security_impact > 0.7:
            plans.append("Security incident response activation")
        
        if impact_metrics.financial_impact > 0.7:
            plans.append("Business continuity plan activation")
        
        return plans
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get risk assessment statistics"""
        if not self.risk_history:
            return {}
        
        risk_scores = [assessment.overall_risk_score for assessment in self.risk_history]
        risk_levels = [assessment.risk_level for assessment in self.risk_history]
        
        return {
            "total_assessments": len(self.risk_history),
            "average_risk_score": np.mean(risk_scores),
            "risk_level_distribution": {
                level: risk_levels.count(level) for level in set(risk_levels)
            },
            "high_risk_assessments": len([r for r in risk_scores if r > 0.6]),
            "last_updated": datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager()
    
    # Example decision context
    decision_context = {
        "decision_id": "RISK-TEST-001",
        "urgency": "high",
        "constraints": {
            "max_cost": "medium",
            "max_time": 12
        }
    }
    
    # Example proposed changes
    proposed_changes = {
        "code_changes": True,
        "code": "def new_function():\n    # New implementation\n    pass\n" * 100,
        "security_sensitive": True,
        "infrastructure_changes": ["database_migration", "api_changes"],
        "affected_components": ["auth_service", "user_service", "payment_service"],
        "user_facing": True,
        "critical_path": True
    }
    
    # Perform risk assessment
    assessment = risk_manager.assess_risk(decision_context, proposed_changes)
    
    # Print results
    print(f"Assessment ID: {assessment.assessment_id}")
    print(f"Decision ID: {assessment.decision_id}")
    print(f"Overall Risk Score: {assessment.overall_risk_score:.3f}")
    print(f"Risk Level: {assessment.risk_level}")
    print(f"Number of Risk Factors: {len(assessment.risk_factors)}")
    
    print("\nRisk Factors:")
    for factor in assessment.risk_factors:
        print(f"  - {factor.name} ({factor.category.value}): {factor.probability.value} probability, {factor.impact.value} impact")
    
    print(f"\nImpact Analysis:")
    for key, value in assessment.impact_analysis.items():
        print(f"  - {key}: {value:.3f}")
    
    print(f"\nMitigation Plan:")
    print(f"  - Effectiveness Score: {assessment.mitigation_plan['effectiveness_score']:.3f}")
    print(f"  - Implementation Time: {assessment.mitigation_plan['implementation_time']} hours")
    print(f"  - Total Cost: {assessment.mitigation_plan['total_cost']}")
    print(f"  - Number of Strategies: {len(assessment.mitigation_plan['strategies'])}")
    
    print(f"\nMonitoring Requirements:")
    for req in assessment.monitoring_requirements:
        print(f"  - {req}")
    
    print(f"\nContingency Plans:")
    for plan in assessment.contingency_plans:
        print(f"  - {plan}")
    
    # Get statistics
    stats = risk_manager.get_risk_statistics()
    print(f"\nRisk Statistics: {json.dumps(stats, indent=2)}")

