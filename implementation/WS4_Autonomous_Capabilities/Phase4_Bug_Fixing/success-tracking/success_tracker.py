"""
Success Tracker for Nexus Architect
Comprehensive tracking and analysis of autonomous bug fixing success rates
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import numpy as np
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixOutcome(Enum):
    """Fix outcome enumeration"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    REGRESSION = "regression"
    PENDING = "pending"

class MetricType(Enum):
    """Metric type enumeration"""
    SUCCESS_RATE = "success_rate"
    RESOLUTION_TIME = "resolution_time"
    QUALITY_SCORE = "quality_score"
    USER_SATISFACTION = "user_satisfaction"
    REGRESSION_RATE = "regression_rate"

@dataclass
class FixResult:
    """Fix result tracking"""
    fix_id: str
    ticket_id: str
    bug_type: str
    complexity: str
    fix_strategy: str
    outcome: FixOutcome
    resolution_time: float  # minutes
    quality_score: float
    user_satisfaction: Optional[float]
    regression_detected: bool
    feedback: str
    created_at: datetime
    verified_at: Optional[datetime]

@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    period_start: datetime
    period_end: datetime
    total_fixes: int
    successful_fixes: int
    failed_fixes: int
    success_rate: float
    average_resolution_time: float
    average_quality_score: float
    regression_rate: float
    user_satisfaction_avg: float
    improvement_suggestions: List[str]

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_type: MetricType
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float  # 0-1
    data_points: List[float]
    prediction: Optional[float]
    confidence: float

class SuccessTracker:
    """
    Comprehensive success tracking and analysis system
    """
    
    def __init__(self):
        self.fix_results = []
        self.performance_history = []
        self.learning_data = defaultdict(list)
        self.improvement_actions = []
    
    async def track_fix_result(self, fix_result: FixResult):
        """
        Track the result of a fix attempt
        
        Args:
            fix_result: Fix result data
        """
        try:
            # Store fix result
            self.fix_results.append(fix_result)
            
            # Update learning data
            self._update_learning_data(fix_result)
            
            # Analyze patterns
            await self._analyze_fix_patterns()
            
            # Generate improvement suggestions
            await self._generate_improvement_suggestions(fix_result)
            
            logger.info(f"Tracked fix result for {fix_result.fix_id}: {fix_result.outcome.value}")
            
        except Exception as e:
            logger.error(f"Error tracking fix result: {str(e)}")
    
    def _update_learning_data(self, fix_result: FixResult):
        """Update learning data with new fix result"""
        try:
            # Group by bug type
            self.learning_data[f"bug_type_{fix_result.bug_type}"].append({
                "outcome": fix_result.outcome.value,
                "strategy": fix_result.fix_strategy,
                "quality_score": fix_result.quality_score,
                "resolution_time": fix_result.resolution_time
            })
            
            # Group by complexity
            self.learning_data[f"complexity_{fix_result.complexity}"].append({
                "outcome": fix_result.outcome.value,
                "strategy": fix_result.fix_strategy,
                "quality_score": fix_result.quality_score,
                "resolution_time": fix_result.resolution_time
            })
            
            # Group by strategy
            self.learning_data[f"strategy_{fix_result.fix_strategy}"].append({
                "outcome": fix_result.outcome.value,
                "bug_type": fix_result.bug_type,
                "quality_score": fix_result.quality_score,
                "resolution_time": fix_result.resolution_time
            })
            
        except Exception as e:
            logger.error(f"Error updating learning data: {str(e)}")
    
    async def _analyze_fix_patterns(self):
        """Analyze patterns in fix results"""
        try:
            if len(self.fix_results) < 10:  # Need minimum data
                return
            
            # Analyze success patterns by bug type
            bug_type_success = defaultdict(list)
            for result in self.fix_results[-50:]:  # Last 50 results
                bug_type_success[result.bug_type].append(
                    1 if result.outcome == FixOutcome.SUCCESS else 0
                )
            
            # Analyze strategy effectiveness
            strategy_success = defaultdict(list)
            for result in self.fix_results[-50:]:
                strategy_success[result.fix_strategy].append(
                    1 if result.outcome == FixOutcome.SUCCESS else 0
                )
            
            # Log insights
            for bug_type, outcomes in bug_type_success.items():
                if len(outcomes) >= 5:
                    success_rate = sum(outcomes) / len(outcomes)
                    logger.info(f"Bug type {bug_type} success rate: {success_rate:.2%}")
            
            for strategy, outcomes in strategy_success.items():
                if len(outcomes) >= 5:
                    success_rate = sum(outcomes) / len(outcomes)
                    logger.info(f"Strategy {strategy} success rate: {success_rate:.2%}")
            
        except Exception as e:
            logger.error(f"Error analyzing fix patterns: {str(e)}")
    
    async def _generate_improvement_suggestions(self, fix_result: FixResult):
        """Generate improvement suggestions based on fix result"""
        try:
            suggestions = []
            
            # Analyze failure patterns
            if fix_result.outcome == FixOutcome.FAILURE:
                # Check if this bug type has low success rate
                recent_results = [r for r in self.fix_results[-20:] 
                                if r.bug_type == fix_result.bug_type]
                
                if len(recent_results) >= 3:
                    success_rate = sum(1 for r in recent_results 
                                     if r.outcome == FixOutcome.SUCCESS) / len(recent_results)
                    
                    if success_rate < 0.5:
                        suggestions.append(f"Low success rate for {fix_result.bug_type} bugs - consider improving detection patterns")
                
                # Check if strategy is ineffective
                strategy_results = [r for r in self.fix_results[-20:] 
                                  if r.fix_strategy == fix_result.fix_strategy]
                
                if len(strategy_results) >= 3:
                    strategy_success = sum(1 for r in strategy_results 
                                         if r.outcome == FixOutcome.SUCCESS) / len(strategy_results)
                    
                    if strategy_success < 0.4:
                        suggestions.append(f"Strategy {fix_result.fix_strategy} showing poor performance - review implementation")
            
            # Analyze quality issues
            if fix_result.quality_score < 0.7:
                suggestions.append("Low quality score - review fix generation algorithms")
            
            # Analyze resolution time
            if fix_result.resolution_time > 60:  # More than 1 hour
                suggestions.append("High resolution time - optimize processing pipeline")
            
            # Store suggestions
            for suggestion in suggestions:
                self.improvement_actions.append({
                    "suggestion": suggestion,
                    "fix_id": fix_result.fix_id,
                    "created_at": datetime.now(),
                    "priority": "medium"
                })
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {str(e)}")
    
    async def calculate_performance_metrics(self, 
                                          start_date: datetime, 
                                          end_date: datetime) -> PerformanceMetrics:
        """
        Calculate performance metrics for a given period
        
        Args:
            start_date: Period start date
            end_date: Period end date
            
        Returns:
            Performance metrics
        """
        try:
            # Filter results by date range
            period_results = [
                r for r in self.fix_results 
                if start_date <= r.created_at <= end_date
            ]
            
            if not period_results:
                return PerformanceMetrics(
                    period_start=start_date,
                    period_end=end_date,
                    total_fixes=0,
                    successful_fixes=0,
                    failed_fixes=0,
                    success_rate=0.0,
                    average_resolution_time=0.0,
                    average_quality_score=0.0,
                    regression_rate=0.0,
                    user_satisfaction_avg=0.0,
                    improvement_suggestions=[]
                )
            
            # Calculate basic metrics
            total_fixes = len(period_results)
            successful_fixes = sum(1 for r in period_results if r.outcome == FixOutcome.SUCCESS)
            failed_fixes = sum(1 for r in period_results if r.outcome == FixOutcome.FAILURE)
            
            success_rate = successful_fixes / total_fixes if total_fixes > 0 else 0
            
            # Calculate average resolution time
            resolution_times = [r.resolution_time for r in period_results]
            average_resolution_time = statistics.mean(resolution_times) if resolution_times else 0
            
            # Calculate average quality score
            quality_scores = [r.quality_score for r in period_results]
            average_quality_score = statistics.mean(quality_scores) if quality_scores else 0
            
            # Calculate regression rate
            regressions = sum(1 for r in period_results if r.regression_detected)
            regression_rate = regressions / total_fixes if total_fixes > 0 else 0
            
            # Calculate user satisfaction
            satisfaction_scores = [r.user_satisfaction for r in period_results 
                                 if r.user_satisfaction is not None]
            user_satisfaction_avg = statistics.mean(satisfaction_scores) if satisfaction_scores else 0
            
            # Generate improvement suggestions
            improvement_suggestions = await self._generate_period_suggestions(period_results)
            
            metrics = PerformanceMetrics(
                period_start=start_date,
                period_end=end_date,
                total_fixes=total_fixes,
                successful_fixes=successful_fixes,
                failed_fixes=failed_fixes,
                success_rate=success_rate,
                average_resolution_time=average_resolution_time,
                average_quality_score=average_quality_score,
                regression_rate=regression_rate,
                user_satisfaction_avg=user_satisfaction_avg,
                improvement_suggestions=improvement_suggestions
            )
            
            # Store in history
            self.performance_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    async def _generate_period_suggestions(self, period_results: List[FixResult]) -> List[str]:
        """Generate improvement suggestions for a period"""
        try:
            suggestions = []
            
            if not period_results:
                return suggestions
            
            # Analyze success rate
            success_rate = sum(1 for r in period_results if r.outcome == FixOutcome.SUCCESS) / len(period_results)
            
            if success_rate < 0.6:
                suggestions.append("Success rate below 60% - review fix generation algorithms")
            elif success_rate < 0.8:
                suggestions.append("Success rate below 80% - optimize fix validation process")
            
            # Analyze resolution time
            resolution_times = [r.resolution_time for r in period_results]
            avg_resolution_time = statistics.mean(resolution_times)
            
            if avg_resolution_time > 45:
                suggestions.append("Average resolution time exceeds 45 minutes - optimize processing pipeline")
            
            # Analyze quality scores
            quality_scores = [r.quality_score for r in period_results]
            avg_quality = statistics.mean(quality_scores)
            
            if avg_quality < 0.75:
                suggestions.append("Average quality score below 75% - improve fix generation quality")
            
            # Analyze regression rate
            regression_rate = sum(1 for r in period_results if r.regression_detected) / len(period_results)
            
            if regression_rate > 0.1:
                suggestions.append("Regression rate above 10% - strengthen validation and testing")
            
            # Analyze strategy performance
            strategy_performance = defaultdict(list)
            for result in period_results:
                strategy_performance[result.fix_strategy].append(
                    1 if result.outcome == FixOutcome.SUCCESS else 0
                )
            
            for strategy, outcomes in strategy_performance.items():
                if len(outcomes) >= 3:
                    strategy_success = sum(outcomes) / len(outcomes)
                    if strategy_success < 0.5:
                        suggestions.append(f"Strategy '{strategy}' underperforming - review implementation")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating period suggestions: {str(e)}")
            return []
    
    async def analyze_trends(self, metric_type: MetricType, days: int = 30) -> TrendAnalysis:
        """
        Analyze trends for a specific metric
        
        Args:
            metric_type: Type of metric to analyze
            days: Number of days to analyze
            
        Returns:
            Trend analysis results
        """
        try:
            # Get data points for the specified period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Calculate daily metrics
            daily_metrics = []
            current_date = start_date
            
            while current_date <= end_date:
                next_date = current_date + timedelta(days=1)
                
                day_results = [
                    r for r in self.fix_results 
                    if current_date <= r.created_at < next_date
                ]
                
                if day_results:
                    if metric_type == MetricType.SUCCESS_RATE:
                        value = sum(1 for r in day_results if r.outcome == FixOutcome.SUCCESS) / len(day_results)
                    elif metric_type == MetricType.RESOLUTION_TIME:
                        value = statistics.mean([r.resolution_time for r in day_results])
                    elif metric_type == MetricType.QUALITY_SCORE:
                        value = statistics.mean([r.quality_score for r in day_results])
                    elif metric_type == MetricType.USER_SATISFACTION:
                        satisfaction_scores = [r.user_satisfaction for r in day_results 
                                             if r.user_satisfaction is not None]
                        value = statistics.mean(satisfaction_scores) if satisfaction_scores else 0
                    elif metric_type == MetricType.REGRESSION_RATE:
                        value = sum(1 for r in day_results if r.regression_detected) / len(day_results)
                    else:
                        value = 0
                    
                    daily_metrics.append(value)
                
                current_date = next_date
            
            if len(daily_metrics) < 3:
                return TrendAnalysis(
                    metric_type=metric_type,
                    trend_direction="insufficient_data",
                    trend_strength=0.0,
                    data_points=daily_metrics,
                    prediction=None,
                    confidence=0.0
                )
            
            # Calculate trend
            x = np.arange(len(daily_metrics))
            y = np.array(daily_metrics)
            
            # Linear regression for trend
            slope, intercept = np.polyfit(x, y, 1)
            
            # Determine trend direction
            if abs(slope) < 0.001:  # Very small slope
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "improving"
                trend_strength = min(abs(slope) * 100, 1.0)
            else:
                trend_direction = "declining"
                trend_strength = min(abs(slope) * 100, 1.0)
            
            # Predict next value
            next_x = len(daily_metrics)
            prediction = slope * next_x + intercept
            
            # Calculate confidence based on R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            confidence = max(0, r_squared)
            
            return TrendAnalysis(
                metric_type=metric_type,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                data_points=daily_metrics,
                prediction=prediction,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return TrendAnalysis(
                metric_type=metric_type,
                trend_direction="error",
                trend_strength=0.0,
                data_points=[],
                prediction=None,
                confidence=0.0
            )
    
    def get_best_performing_strategies(self, min_samples: int = 5) -> Dict[str, float]:
        """Get best performing fix strategies"""
        try:
            strategy_performance = defaultdict(list)
            
            for result in self.fix_results:
                strategy_performance[result.fix_strategy].append(
                    1 if result.outcome == FixOutcome.SUCCESS else 0
                )
            
            # Calculate success rates for strategies with enough samples
            best_strategies = {}
            for strategy, outcomes in strategy_performance.items():
                if len(outcomes) >= min_samples:
                    success_rate = sum(outcomes) / len(outcomes)
                    best_strategies[strategy] = success_rate
            
            # Sort by success rate
            return dict(sorted(best_strategies.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting best strategies: {str(e)}")
            return {}
    
    def get_problematic_bug_types(self, min_samples: int = 3) -> Dict[str, Dict[str, Any]]:
        """Get bug types with low success rates"""
        try:
            bug_type_stats = defaultdict(lambda: {"total": 0, "successful": 0, "avg_resolution_time": 0})
            
            for result in self.fix_results:
                stats = bug_type_stats[result.bug_type]
                stats["total"] += 1
                if result.outcome == FixOutcome.SUCCESS:
                    stats["successful"] += 1
                stats["avg_resolution_time"] += result.resolution_time
            
            # Calculate final stats
            problematic_types = {}
            for bug_type, stats in bug_type_stats.items():
                if stats["total"] >= min_samples:
                    success_rate = stats["successful"] / stats["total"]
                    avg_time = stats["avg_resolution_time"] / stats["total"]
                    
                    if success_rate < 0.7:  # Less than 70% success rate
                        problematic_types[bug_type] = {
                            "success_rate": success_rate,
                            "total_attempts": stats["total"],
                            "avg_resolution_time": avg_time
                        }
            
            return problematic_types
            
        except Exception as e:
            logger.error(f"Error getting problematic bug types: {str(e)}")
            return {}
    
    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Get prioritized improvement recommendations"""
        try:
            # Analyze recent performance
            recent_results = self.fix_results[-50:] if len(self.fix_results) >= 50 else self.fix_results
            
            recommendations = []
            
            if not recent_results:
                return recommendations
            
            # Success rate analysis
            success_rate = sum(1 for r in recent_results if r.outcome == FixOutcome.SUCCESS) / len(recent_results)
            
            if success_rate < 0.6:
                recommendations.append({
                    "priority": "high",
                    "category": "success_rate",
                    "issue": f"Low success rate: {success_rate:.1%}",
                    "recommendation": "Review and improve fix generation algorithms",
                    "impact": "high"
                })
            
            # Resolution time analysis
            avg_resolution_time = statistics.mean([r.resolution_time for r in recent_results])
            
            if avg_resolution_time > 60:
                recommendations.append({
                    "priority": "medium",
                    "category": "performance",
                    "issue": f"High average resolution time: {avg_resolution_time:.1f} minutes",
                    "recommendation": "Optimize processing pipeline and parallel execution",
                    "impact": "medium"
                })
            
            # Quality analysis
            avg_quality = statistics.mean([r.quality_score for r in recent_results])
            
            if avg_quality < 0.75:
                recommendations.append({
                    "priority": "medium",
                    "category": "quality",
                    "issue": f"Low average quality score: {avg_quality:.2f}",
                    "recommendation": "Enhance fix validation and testing procedures",
                    "impact": "medium"
                })
            
            # Regression analysis
            regression_rate = sum(1 for r in recent_results if r.regression_detected) / len(recent_results)
            
            if regression_rate > 0.1:
                recommendations.append({
                    "priority": "high",
                    "category": "regression",
                    "issue": f"High regression rate: {regression_rate:.1%}",
                    "recommendation": "Strengthen pre-deployment testing and validation",
                    "impact": "high"
                })
            
            # Strategy-specific recommendations
            problematic_strategies = []
            strategy_performance = defaultdict(list)
            
            for result in recent_results:
                strategy_performance[result.fix_strategy].append(
                    1 if result.outcome == FixOutcome.SUCCESS else 0
                )
            
            for strategy, outcomes in strategy_performance.items():
                if len(outcomes) >= 3:
                    strategy_success = sum(outcomes) / len(outcomes)
                    if strategy_success < 0.5:
                        problematic_strategies.append(strategy)
            
            if problematic_strategies:
                recommendations.append({
                    "priority": "medium",
                    "category": "strategy",
                    "issue": f"Underperforming strategies: {', '.join(problematic_strategies)}",
                    "recommendation": "Review and improve specific fix generation strategies",
                    "impact": "medium"
                })
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting improvement recommendations: {str(e)}")
            return []
    
    def get_success_statistics(self) -> Dict[str, Any]:
        """Get comprehensive success statistics"""
        try:
            if not self.fix_results:
                return {"error": "No fix results available"}
            
            total_fixes = len(self.fix_results)
            successful_fixes = sum(1 for r in self.fix_results if r.outcome == FixOutcome.SUCCESS)
            failed_fixes = sum(1 for r in self.fix_results if r.outcome == FixOutcome.FAILURE)
            
            # Calculate various metrics
            overall_success_rate = successful_fixes / total_fixes
            
            resolution_times = [r.resolution_time for r in self.fix_results]
            avg_resolution_time = statistics.mean(resolution_times)
            median_resolution_time = statistics.median(resolution_times)
            
            quality_scores = [r.quality_score for r in self.fix_results]
            avg_quality_score = statistics.mean(quality_scores)
            
            regression_count = sum(1 for r in self.fix_results if r.regression_detected)
            regression_rate = regression_count / total_fixes
            
            # User satisfaction
            satisfaction_scores = [r.user_satisfaction for r in self.fix_results 
                                 if r.user_satisfaction is not None]
            avg_user_satisfaction = statistics.mean(satisfaction_scores) if satisfaction_scores else 0
            
            # Recent performance (last 30 days)
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_results = [r for r in self.fix_results if r.created_at >= recent_cutoff]
            
            recent_success_rate = 0
            if recent_results:
                recent_successful = sum(1 for r in recent_results if r.outcome == FixOutcome.SUCCESS)
                recent_success_rate = recent_successful / len(recent_results)
            
            return {
                "total_fixes": total_fixes,
                "successful_fixes": successful_fixes,
                "failed_fixes": failed_fixes,
                "overall_success_rate": overall_success_rate,
                "recent_success_rate": recent_success_rate,
                "average_resolution_time": avg_resolution_time,
                "median_resolution_time": median_resolution_time,
                "average_quality_score": avg_quality_score,
                "regression_rate": regression_rate,
                "average_user_satisfaction": avg_user_satisfaction,
                "best_strategies": self.get_best_performing_strategies(),
                "problematic_bug_types": self.get_problematic_bug_types(),
                "improvement_recommendations": self.get_improvement_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Error getting success statistics: {str(e)}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    async def test_success_tracker():
        """Test the success tracker"""
        tracker = SuccessTracker()
        
        # Generate sample fix results
        sample_results = [
            FixResult(
                fix_id=f"FIX-{i:03d}",
                ticket_id=f"BUG-{i:03d}",
                bug_type="null_pointer_exception",
                complexity="low",
                fix_strategy="pattern_based",
                outcome=FixOutcome.SUCCESS if i % 3 != 0 else FixOutcome.FAILURE,
                resolution_time=30 + (i % 20) * 2,
                quality_score=0.8 + (i % 10) * 0.02,
                user_satisfaction=0.7 + (i % 15) * 0.02,
                regression_detected=i % 10 == 0,
                feedback="Automated fix applied successfully",
                created_at=datetime.now() - timedelta(days=i),
                verified_at=datetime.now() - timedelta(days=i-1) if i % 3 != 0 else None
            )
            for i in range(1, 51)  # 50 sample results
        ]
        
        # Track all results
        for result in sample_results:
            await tracker.track_fix_result(result)
        
        print("Success Tracker Test Results:")
        print("=" * 50)
        
        # Get overall statistics
        stats = tracker.get_success_statistics()
        print(f"Total Fixes: {stats['total_fixes']}")
        print(f"Overall Success Rate: {stats['overall_success_rate']:.1%}")
        print(f"Recent Success Rate: {stats['recent_success_rate']:.1%}")
        print(f"Average Resolution Time: {stats['average_resolution_time']:.1f} minutes")
        print(f"Average Quality Score: {stats['average_quality_score']:.2f}")
        print(f"Regression Rate: {stats['regression_rate']:.1%}")
        
        # Get performance metrics for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        metrics = await tracker.calculate_performance_metrics(start_date, end_date)
        
        print(f"\nLast 30 Days Performance:")
        print(f"Success Rate: {metrics.success_rate:.1%}")
        print(f"Average Resolution Time: {metrics.average_resolution_time:.1f} minutes")
        print(f"Average Quality Score: {metrics.average_quality_score:.2f}")
        
        # Analyze trends
        success_trend = await tracker.analyze_trends(MetricType.SUCCESS_RATE, days=30)
        print(f"\nSuccess Rate Trend: {success_trend.trend_direction}")
        print(f"Trend Strength: {success_trend.trend_strength:.2f}")
        print(f"Confidence: {success_trend.confidence:.2f}")
        
        # Get recommendations
        recommendations = tracker.get_improvement_recommendations()
        print(f"\nImprovement Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"{i}. [{rec['priority'].upper()}] {rec['issue']}")
            print(f"   Recommendation: {rec['recommendation']}")
    
    # Run the test
    asyncio.run(test_success_tracker())

