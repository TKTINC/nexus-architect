"""
Nexus Architect Conversation Quality Monitoring System

This module provides comprehensive quality monitoring for conversational AI,
including response relevance scoring, conversation coherence analysis,
and continuous improvement mechanisms.
"""

import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import statistics

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class QualityMetric(str, Enum):
    """Quality metrics for conversation assessment"""
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    HELPFULNESS = "helpfulness"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    ENGAGEMENT = "engagement"
    SATISFACTION = "satisfaction"

class FeedbackType(str, Enum):
    """Types of feedback"""
    EXPLICIT_RATING = "explicit_rating"
    IMPLICIT_BEHAVIOR = "implicit_behavior"
    EXPERT_REVIEW = "expert_review"
    AUTOMATED_ANALYSIS = "automated_analysis"

class ImprovementAction(str, Enum):
    """Types of improvement actions"""
    RETRAIN_MODEL = "retrain_model"
    UPDATE_TEMPLATES = "update_templates"
    ADJUST_ROUTING = "adjust_routing"
    ENHANCE_CONTEXT = "enhance_context"
    IMPROVE_NLU = "improve_nlu"

@dataclass
class QualityScore:
    """Quality score for a specific metric"""
    metric: QualityMetric
    score: float  # 0.0 to 1.0
    confidence: float
    explanation: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class ConversationQuality:
    """Overall quality assessment for a conversation"""
    conversation_id: str
    turn_id: str
    user_message: str
    ai_response: str
    quality_scores: List[QualityScore]
    overall_score: float
    feedback_received: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    timestamp: datetime

@dataclass
class UserFeedback:
    """User feedback on conversation quality"""
    feedback_id: str
    conversation_id: str
    turn_id: str
    feedback_type: FeedbackType
    rating: Optional[float]  # 1-5 scale
    comments: Optional[str]
    specific_issues: List[str]
    timestamp: datetime
    user_id: str

class RelevanceAnalyzer:
    """Analyzes response relevance to user queries"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.semantic_model = None
        
    async def initialize(self):
        """Initialize semantic similarity model"""
        try:
            self.semantic_model = pipeline(
                "feature-extraction",
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Relevance analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize semantic model: {e}")
            
    async def analyze_relevance(self, user_message: str, ai_response: str) -> QualityScore:
        """Analyze relevance of AI response to user message"""
        try:
            # TF-IDF based similarity
            documents = [user_message, ai_response]
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Semantic similarity (if available)
            semantic_similarity = 0.0
            if self.semantic_model:
                user_embedding = self.semantic_model(user_message)[0]
                response_embedding = self.semantic_model(ai_response)[0]
                
                user_vec = np.mean(user_embedding, axis=0)
                response_vec = np.mean(response_embedding, axis=0)
                
                semantic_similarity = np.dot(user_vec, response_vec) / (
                    np.linalg.norm(user_vec) * np.linalg.norm(response_vec)
                )
            
            # Combined relevance score
            relevance_score = (tfidf_similarity * 0.4 + semantic_similarity * 0.6)
            
            # Generate explanation
            explanation = self._generate_relevance_explanation(
                relevance_score, tfidf_similarity, semantic_similarity
            )
            
            return QualityScore(
                metric=QualityMetric.RELEVANCE,
                score=float(relevance_score),
                confidence=0.8,
                explanation=explanation,
                timestamp=datetime.utcnow(),
                metadata={
                    "tfidf_similarity": float(tfidf_similarity),
                    "semantic_similarity": float(semantic_similarity)
                }
            )
            
        except Exception as e:
            logger.error(f"Relevance analysis failed: {e}")
            return QualityScore(
                metric=QualityMetric.RELEVANCE,
                score=0.5,
                confidence=0.1,
                explanation="Analysis failed",
                timestamp=datetime.utcnow()
            )
            
    def _generate_relevance_explanation(
        self, 
        overall_score: float, 
        tfidf_score: float, 
        semantic_score: float
    ) -> str:
        """Generate explanation for relevance score"""
        if overall_score >= 0.8:
            return "Response is highly relevant to the user's query"
        elif overall_score >= 0.6:
            return "Response is moderately relevant with some off-topic content"
        elif overall_score >= 0.4:
            return "Response has limited relevance to the user's query"
        else:
            return "Response appears to be off-topic or irrelevant"

class CoherenceAnalyzer:
    """Analyzes conversation coherence and flow"""
    
    def __init__(self):
        self.conversation_history = {}
        
    async def analyze_coherence(
        self, 
        conversation_id: str, 
        current_turn: Dict[str, str],
        conversation_history: List[Dict[str, str]]
    ) -> QualityScore:
        """Analyze coherence of current turn with conversation history"""
        
        if len(conversation_history) < 2:
            return QualityScore(
                metric=QualityMetric.COHERENCE,
                score=1.0,
                confidence=0.5,
                explanation="Insufficient history for coherence analysis",
                timestamp=datetime.utcnow()
            )
            
        # Analyze topic consistency
        topic_consistency = self._analyze_topic_consistency(
            current_turn, conversation_history
        )
        
        # Analyze context awareness
        context_awareness = self._analyze_context_awareness(
            current_turn, conversation_history
        )
        
        # Analyze response appropriateness
        response_appropriateness = self._analyze_response_appropriateness(
            current_turn, conversation_history
        )
        
        # Combined coherence score
        coherence_score = (
            topic_consistency * 0.4 +
            context_awareness * 0.4 +
            response_appropriateness * 0.2
        )
        
        explanation = self._generate_coherence_explanation(
            coherence_score, topic_consistency, context_awareness, response_appropriateness
        )
        
        return QualityScore(
            metric=QualityMetric.COHERENCE,
            score=coherence_score,
            confidence=0.7,
            explanation=explanation,
            timestamp=datetime.utcnow(),
            metadata={
                "topic_consistency": topic_consistency,
                "context_awareness": context_awareness,
                "response_appropriateness": response_appropriateness
            }
        )
        
    def _analyze_topic_consistency(
        self, 
        current_turn: Dict[str, str], 
        history: List[Dict[str, str]]
    ) -> float:
        """Analyze if current turn maintains topic consistency"""
        # Simple keyword overlap analysis
        current_words = set(current_turn["ai_response"].lower().split())
        
        topic_scores = []
        for turn in history[-3:]:  # Look at last 3 turns
            turn_words = set(turn["ai_response"].lower().split())
            overlap = len(current_words.intersection(turn_words))
            total_words = len(current_words.union(turn_words))
            if total_words > 0:
                topic_scores.append(overlap / total_words)
                
        return statistics.mean(topic_scores) if topic_scores else 0.5
        
    def _analyze_context_awareness(
        self, 
        current_turn: Dict[str, str], 
        history: List[Dict[str, str]]
    ) -> float:
        """Analyze if response shows awareness of conversation context"""
        response = current_turn["ai_response"].lower()
        
        # Check for context indicators
        context_indicators = [
            "as mentioned", "previously", "earlier", "building on",
            "following up", "continuing", "based on our discussion"
        ]
        
        context_score = 0.0
        for indicator in context_indicators:
            if indicator in response:
                context_score += 0.2
                
        return min(context_score, 1.0)
        
    def _analyze_response_appropriateness(
        self, 
        current_turn: Dict[str, str], 
        history: List[Dict[str, str]]
    ) -> float:
        """Analyze if response is appropriate given the conversation flow"""
        # Check if response addresses the user's question
        user_message = current_turn["user_message"].lower()
        ai_response = current_turn["ai_response"].lower()
        
        # Simple heuristics for appropriateness
        if "?" in user_message and len(ai_response) < 20:
            return 0.3  # Question deserves more detailed response
        elif "help" in user_message and "sorry" in ai_response:
            return 0.4  # Help request met with apology
        elif len(ai_response) > 1000:
            return 0.7  # Very long responses might be overwhelming
        else:
            return 0.8  # Default appropriate score
            
    def _generate_coherence_explanation(
        self, 
        overall_score: float, 
        topic_consistency: float, 
        context_awareness: float, 
        response_appropriateness: float
    ) -> str:
        """Generate explanation for coherence score"""
        issues = []
        
        if topic_consistency < 0.5:
            issues.append("topic drift detected")
        if context_awareness < 0.3:
            issues.append("limited context awareness")
        if response_appropriateness < 0.5:
            issues.append("response may be inappropriate")
            
        if not issues:
            return "Conversation maintains good coherence and flow"
        else:
            return f"Coherence issues: {', '.join(issues)}"

class HelpfulnessAnalyzer:
    """Analyzes how helpful responses are to users"""
    
    def __init__(self):
        self.helpfulness_indicators = self._load_helpfulness_indicators()
        
    def _load_helpfulness_indicators(self) -> Dict[str, List[str]]:
        """Load indicators of helpful vs unhelpful responses"""
        return {
            "helpful_patterns": [
                r"\bhere's how\b", r"\bstep by step\b", r"\bfor example\b",
                r"\bspecifically\b", r"\byou can\b", r"\bi recommend\b",
                r"\btry this\b", r"\bhere's what\b"
            ],
            "unhelpful_patterns": [
                r"\bi don't know\b", r"\bcan't help\b", r"\bnot sure\b",
                r"\bmaybe\b", r"\bpossibly\b", r"\bmight work\b"
            ],
            "action_words": [
                "implement", "configure", "setup", "install", "deploy",
                "create", "build", "design", "optimize", "fix"
            ]
        }
        
    async def analyze_helpfulness(
        self, 
        user_message: str, 
        ai_response: str,
        user_intent: str = None
    ) -> QualityScore:
        """Analyze helpfulness of AI response"""
        
        # Check for helpful patterns
        helpful_score = self._count_patterns(
            ai_response, self.helpfulness_indicators["helpful_patterns"]
        )
        
        # Check for unhelpful patterns
        unhelpful_score = self._count_patterns(
            ai_response, self.helpfulness_indicators["unhelpful_patterns"]
        )
        
        # Check for actionable content
        action_score = self._count_action_words(ai_response)
        
        # Check response length appropriateness
        length_score = self._analyze_response_length(user_message, ai_response)
        
        # Combined helpfulness score
        helpfulness_score = (
            helpful_score * 0.3 +
            (1.0 - unhelpful_score) * 0.3 +
            action_score * 0.2 +
            length_score * 0.2
        )
        
        explanation = self._generate_helpfulness_explanation(
            helpfulness_score, helpful_score, unhelpful_score, action_score
        )
        
        return QualityScore(
            metric=QualityMetric.HELPFULNESS,
            score=helpfulness_score,
            confidence=0.6,
            explanation=explanation,
            timestamp=datetime.utcnow(),
            metadata={
                "helpful_patterns": helpful_score,
                "unhelpful_patterns": unhelpful_score,
                "action_content": action_score,
                "length_appropriateness": length_score
            }
        )
        
    def _count_patterns(self, text: str, patterns: List[str]) -> float:
        """Count pattern matches in text"""
        import re
        text_lower = text.lower()
        matches = 0
        for pattern in patterns:
            matches += len(re.findall(pattern, text_lower))
        return min(matches / len(patterns), 1.0)
        
    def _count_action_words(self, text: str) -> float:
        """Count actionable words in response"""
        text_lower = text.lower()
        action_count = sum(1 for word in self.helpfulness_indicators["action_words"] 
                          if word in text_lower)
        return min(action_count / 3.0, 1.0)  # Normalize to 0-1
        
    def _analyze_response_length(self, user_message: str, ai_response: str) -> float:
        """Analyze if response length is appropriate"""
        user_length = len(user_message.split())
        response_length = len(ai_response.split())
        
        if user_length < 10 and response_length > 200:
            return 0.5  # Too verbose for simple question
        elif user_length > 30 and response_length < 50:
            return 0.6  # Too brief for complex question
        else:
            return 0.8  # Appropriate length
            
    def _generate_helpfulness_explanation(
        self, 
        overall_score: float, 
        helpful_score: float, 
        unhelpful_score: float, 
        action_score: float
    ) -> str:
        """Generate explanation for helpfulness score"""
        if overall_score >= 0.8:
            return "Response provides helpful, actionable guidance"
        elif overall_score >= 0.6:
            return "Response is moderately helpful with some actionable content"
        elif overall_score >= 0.4:
            return "Response has limited helpfulness"
        else:
            return "Response may not be helpful to the user"

class ConversationQualityMonitor:
    """Main quality monitoring system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = None
        self.redis_url = redis_url
        self.relevance_analyzer = RelevanceAnalyzer()
        self.coherence_analyzer = CoherenceAnalyzer()
        self.helpfulness_analyzer = HelpfulnessAnalyzer()
        self.quality_thresholds = self._load_quality_thresholds()
        
    def _load_quality_thresholds(self) -> Dict[QualityMetric, float]:
        """Load quality thresholds for alerts"""
        return {
            QualityMetric.RELEVANCE: 0.6,
            QualityMetric.COHERENCE: 0.7,
            QualityMetric.HELPFULNESS: 0.6,
            QualityMetric.ACCURACY: 0.8,
            QualityMetric.SATISFACTION: 0.7
        }
        
    async def initialize(self):
        """Initialize quality monitoring system"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        
        await self.relevance_analyzer.initialize()
        
        logger.info("Conversation quality monitor initialized")
        
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def analyze_conversation_quality(
        self,
        conversation_id: str,
        turn_id: str,
        user_message: str,
        ai_response: str,
        conversation_history: List[Dict[str, str]] = None,
        user_intent: str = None
    ) -> ConversationQuality:
        """Analyze quality of a conversation turn"""
        
        conversation_history = conversation_history or []
        
        # Run quality analyses in parallel
        relevance_task = self.relevance_analyzer.analyze_relevance(user_message, ai_response)
        coherence_task = self.coherence_analyzer.analyze_coherence(
            conversation_id, 
            {"user_message": user_message, "ai_response": ai_response},
            conversation_history
        )
        helpfulness_task = self.helpfulness_analyzer.analyze_helpfulness(
            user_message, ai_response, user_intent
        )
        
        quality_scores = await asyncio.gather(
            relevance_task, coherence_task, helpfulness_task
        )
        
        # Calculate overall quality score
        overall_score = statistics.mean([score.score for score in quality_scores])
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(quality_scores)
        
        # Create quality assessment
        quality_assessment = ConversationQuality(
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_message=user_message,
            ai_response=ai_response,
            quality_scores=quality_scores,
            overall_score=overall_score,
            feedback_received=[],
            improvement_suggestions=improvement_suggestions,
            timestamp=datetime.utcnow()
        )
        
        # Store quality assessment
        await self._store_quality_assessment(quality_assessment)
        
        # Check for quality alerts
        await self._check_quality_alerts(quality_assessment)
        
        return quality_assessment
        
    async def _store_quality_assessment(self, assessment: ConversationQuality):
        """Store quality assessment in Redis"""
        assessment_data = {
            "conversation_id": assessment.conversation_id,
            "turn_id": assessment.turn_id,
            "overall_score": assessment.overall_score,
            "quality_scores": [asdict(score) for score in assessment.quality_scores],
            "improvement_suggestions": assessment.improvement_suggestions,
            "timestamp": assessment.timestamp.isoformat()
        }
        
        # Convert datetime objects to strings
        for score_data in assessment_data["quality_scores"]:
            score_data["timestamp"] = score_data["timestamp"].isoformat()
        
        await self.redis_client.setex(
            f"quality:{assessment.conversation_id}:{assessment.turn_id}",
            86400,  # 24 hours
            json.dumps(assessment_data)
        )
        
        # Update conversation quality metrics
        await self._update_quality_metrics(assessment)
        
    async def _update_quality_metrics(self, assessment: ConversationQuality):
        """Update aggregated quality metrics"""
        # Update daily metrics
        today = datetime.utcnow().date().isoformat()
        
        for score in assessment.quality_scores:
            metric_key = f"metrics:{today}:{score.metric.value}"
            await self.redis_client.lpush(metric_key, score.score)
            await self.redis_client.expire(metric_key, 86400 * 7)  # Keep for 7 days
            
        # Update overall score
        overall_key = f"metrics:{today}:overall"
        await self.redis_client.lpush(overall_key, assessment.overall_score)
        await self.redis_client.expire(overall_key, 86400 * 7)
        
    async def _check_quality_alerts(self, assessment: ConversationQuality):
        """Check if quality scores trigger alerts"""
        alerts = []
        
        for score in assessment.quality_scores:
            threshold = self.quality_thresholds.get(score.metric, 0.5)
            if score.score < threshold:
                alerts.append({
                    "metric": score.metric.value,
                    "score": score.score,
                    "threshold": threshold,
                    "explanation": score.explanation,
                    "conversation_id": assessment.conversation_id,
                    "turn_id": assessment.turn_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
        if alerts:
            await self._send_quality_alerts(alerts)
            
    async def _send_quality_alerts(self, alerts: List[Dict[str, Any]]):
        """Send quality alerts to monitoring system"""
        for alert in alerts:
            alert_key = f"alerts:quality:{datetime.utcnow().isoformat()}"
            await self.redis_client.setex(alert_key, 86400, json.dumps(alert))
            logger.warning(f"Quality alert: {alert['metric']} score {alert['score']} below threshold {alert['threshold']}")
            
    def _generate_improvement_suggestions(self, quality_scores: List[QualityScore]) -> List[str]:
        """Generate improvement suggestions based on quality scores"""
        suggestions = []
        
        for score in quality_scores:
            if score.score < 0.6:
                if score.metric == QualityMetric.RELEVANCE:
                    suggestions.append("Improve response relevance by better understanding user intent")
                elif score.metric == QualityMetric.COHERENCE:
                    suggestions.append("Enhance context awareness and conversation flow")
                elif score.metric == QualityMetric.HELPFULNESS:
                    suggestions.append("Provide more actionable and specific guidance")
                    
        return suggestions
        
    async def record_user_feedback(
        self,
        conversation_id: str,
        turn_id: str,
        feedback_type: FeedbackType,
        rating: Optional[float] = None,
        comments: Optional[str] = None,
        specific_issues: List[str] = None,
        user_id: str = None
    ) -> str:
        """Record user feedback on conversation quality"""
        
        feedback = UserFeedback(
            feedback_id=f"feedback_{int(time.time())}_{conversation_id}",
            conversation_id=conversation_id,
            turn_id=turn_id,
            feedback_type=feedback_type,
            rating=rating,
            comments=comments,
            specific_issues=specific_issues or [],
            timestamp=datetime.utcnow(),
            user_id=user_id or "anonymous"
        )
        
        # Store feedback
        feedback_data = asdict(feedback)
        feedback_data["timestamp"] = feedback.timestamp.isoformat()
        
        await self.redis_client.setex(
            f"feedback:{feedback.feedback_id}",
            86400 * 30,  # Keep for 30 days
            json.dumps(feedback_data)
        )
        
        # Update quality assessment with feedback
        await self._update_quality_with_feedback(feedback)
        
        return feedback.feedback_id
        
    async def _update_quality_with_feedback(self, feedback: UserFeedback):
        """Update quality assessment with user feedback"""
        quality_key = f"quality:{feedback.conversation_id}:{feedback.turn_id}"
        quality_data = await self.redis_client.get(quality_key)
        
        if quality_data:
            quality_dict = json.loads(quality_data)
            if "feedback_received" not in quality_dict:
                quality_dict["feedback_received"] = []
                
            quality_dict["feedback_received"].append({
                "feedback_id": feedback.feedback_id,
                "rating": feedback.rating,
                "comments": feedback.comments,
                "timestamp": feedback.timestamp.isoformat()
            })
            
            await self.redis_client.setex(quality_key, 86400, json.dumps(quality_dict))
            
    async def get_quality_metrics(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get aggregated quality metrics for a date range"""
        
        metrics = {}
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            date_str = current_date.isoformat()
            
            # Get metrics for each quality dimension
            for metric in QualityMetric:
                metric_key = f"metrics:{date_str}:{metric.value}"
                scores = await self.redis_client.lrange(metric_key, 0, -1)
                
                if scores:
                    float_scores = [float(score) for score in scores]
                    metrics[f"{date_str}_{metric.value}"] = {
                        "average": statistics.mean(float_scores),
                        "median": statistics.median(float_scores),
                        "min": min(float_scores),
                        "max": max(float_scores),
                        "count": len(float_scores)
                    }
                    
            current_date += timedelta(days=1)
            
        return metrics
        
    async def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for improving conversation quality"""
        
        # Analyze recent quality trends
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        metrics = await self.get_quality_metrics(start_date, end_date)
        
        recommendations = []
        
        # Analyze each metric
        for metric in QualityMetric:
            metric_scores = []
            for key, value in metrics.items():
                if metric.value in key:
                    metric_scores.append(value["average"])
                    
            if metric_scores:
                avg_score = statistics.mean(metric_scores)
                threshold = self.quality_thresholds.get(metric, 0.5)
                
                if avg_score < threshold:
                    recommendations.append({
                        "metric": metric.value,
                        "current_score": avg_score,
                        "target_score": threshold,
                        "improvement_needed": threshold - avg_score,
                        "action": self._get_improvement_action(metric),
                        "priority": "high" if avg_score < threshold * 0.8 else "medium"
                    })
                    
        return recommendations
        
    def _get_improvement_action(self, metric: QualityMetric) -> ImprovementAction:
        """Get recommended improvement action for a metric"""
        action_mapping = {
            QualityMetric.RELEVANCE: ImprovementAction.IMPROVE_NLU,
            QualityMetric.COHERENCE: ImprovementAction.ENHANCE_CONTEXT,
            QualityMetric.HELPFULNESS: ImprovementAction.UPDATE_TEMPLATES,
            QualityMetric.ACCURACY: ImprovementAction.RETRAIN_MODEL,
            QualityMetric.SATISFACTION: ImprovementAction.ADJUST_ROUTING
        }
        return action_mapping.get(metric, ImprovementAction.RETRAIN_MODEL)

# Example usage and testing
async def main():
    """Example usage of conversation quality monitoring"""
    monitor = ConversationQualityMonitor()
    await monitor.initialize()
    
    try:
        # Analyze conversation quality
        quality_assessment = await monitor.analyze_conversation_quality(
            conversation_id="conv_123",
            turn_id="turn_456",
            user_message="How do I implement OAuth authentication in my microservices?",
            ai_response="To implement OAuth authentication in microservices, you'll need to set up an authorization server, configure your services as resource servers, and implement token validation. Here's a step-by-step approach...",
            conversation_history=[
                {"user_message": "I'm building a microservices architecture", "ai_response": "That's great! Microservices offer many benefits..."},
                {"user_message": "What about security?", "ai_response": "Security is crucial in microservices. You'll need to consider authentication, authorization, and secure communication..."}
            ]
        )
        
        print(f"Overall Quality Score: {quality_assessment.overall_score:.2f}")
        print("\nQuality Scores:")
        for score in quality_assessment.quality_scores:
            print(f"- {score.metric.value}: {score.score:.2f} ({score.explanation})")
            
        print(f"\nImprovement Suggestions:")
        for suggestion in quality_assessment.improvement_suggestions:
            print(f"- {suggestion}")
            
        # Record user feedback
        feedback_id = await monitor.record_user_feedback(
            conversation_id="conv_123",
            turn_id="turn_456",
            feedback_type=FeedbackType.EXPLICIT_RATING,
            rating=4.5,
            comments="Very helpful response with clear steps",
            user_id="user_789"
        )
        print(f"\nFeedback recorded: {feedback_id}")
        
        # Get improvement recommendations
        recommendations = await monitor.get_improvement_recommendations()
        print(f"\nImprovement Recommendations:")
        for rec in recommendations:
            print(f"- {rec['metric']}: {rec['action'].value} (priority: {rec['priority']})")
            
    finally:
        await monitor.close()

if __name__ == "__main__":
    asyncio.run(main())

