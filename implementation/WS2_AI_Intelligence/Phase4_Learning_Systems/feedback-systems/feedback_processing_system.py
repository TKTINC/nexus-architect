"""
Nexus Architect Feedback Processing System

This module implements comprehensive feedback collection and processing for continuous
improvement of AI models and conversational systems.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path

import redis.asyncio as redis
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class FeedbackType(str, Enum):
    """Types of feedback"""
    EXPLICIT_RATING = "explicit_rating"
    EXPLICIT_TEXT = "explicit_text"
    IMPLICIT_BEHAVIOR = "implicit_behavior"
    EXPERT_REVIEW = "expert_review"
    SYSTEM_METRIC = "system_metric"
    USER_CORRECTION = "user_correction"

class FeedbackSource(str, Enum):
    """Sources of feedback"""
    USER_INTERFACE = "user_interface"
    API_CLIENT = "api_client"
    EXPERT_PANEL = "expert_panel"
    AUTOMATED_SYSTEM = "automated_system"
    CONVERSATION_ANALYSIS = "conversation_analysis"
    PERFORMANCE_MONITORING = "performance_monitoring"

class FeedbackSentiment(str, Enum):
    """Sentiment of feedback"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class FeedbackItem:
    """Individual feedback item"""
    feedback_id: str
    conversation_id: str
    user_id: str
    feedback_type: FeedbackType
    feedback_source: FeedbackSource
    content: str
    rating: Optional[float] = None
    sentiment: Optional[FeedbackSentiment] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    processed: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ProcessedFeedback:
    """Processed feedback with insights"""
    feedback_id: str
    original_feedback: FeedbackItem
    sentiment_score: float
    emotion_scores: Dict[str, float]
    topics: List[str]
    keywords: List[str]
    improvement_suggestions: List[str]
    priority_score: float
    category: str
    actionable: bool
    processed_at: datetime

@dataclass
class FeedbackInsights:
    """Aggregated feedback insights"""
    time_period: str
    total_feedback_count: int
    average_rating: float
    sentiment_distribution: Dict[str, float]
    top_topics: List[Tuple[str, int]]
    improvement_areas: List[str]
    user_satisfaction_trend: List[Tuple[datetime, float]]
    model_performance_correlation: Dict[str, float]

# Database models
Base = declarative_base()

class FeedbackDB(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True)
    feedback_id = Column(String(255), unique=True, nullable=False)
    conversation_id = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=False)
    feedback_type = Column(String(50), nullable=False)
    feedback_source = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    rating = Column(Float)
    sentiment = Column(String(20))
    metadata = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    processed = Column(Boolean, default=False)

class ProcessedFeedbackDB(Base):
    __tablename__ = "processed_feedback"
    
    id = Column(Integer, primary_key=True)
    feedback_id = Column(String(255), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    emotion_scores = Column(Text, nullable=False)
    topics = Column(Text, nullable=False)
    keywords = Column(Text, nullable=False)
    improvement_suggestions = Column(Text, nullable=False)
    priority_score = Column(Float, nullable=False)
    category = Column(String(100), nullable=False)
    actionable = Column(Boolean, nullable=False)
    processed_at = Column(DateTime, nullable=False)

class FeedbackInsightsDB(Base):
    __tablename__ = "feedback_insights"
    
    id = Column(Integer, primary_key=True)
    time_period = Column(String(50), nullable=False)
    total_feedback_count = Column(Integer, nullable=False)
    average_rating = Column(Float, nullable=False)
    sentiment_distribution = Column(Text, nullable=False)
    top_topics = Column(Text, nullable=False)
    improvement_areas = Column(Text, nullable=False)
    user_satisfaction_trend = Column(Text, nullable=False)
    model_performance_correlation = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)

class FeedbackProcessor:
    """Advanced feedback processing with NLP and ML"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.topic_model = None
        self.emotion_keywords = self._load_emotion_keywords()
        
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        """Load emotion keyword mappings"""
        return {
            "joy": ["happy", "pleased", "satisfied", "delighted", "excellent", "great", "amazing"],
            "anger": ["angry", "frustrated", "annoyed", "terrible", "awful", "hate", "worst"],
            "sadness": ["sad", "disappointed", "upset", "unhappy", "poor", "bad"],
            "fear": ["worried", "concerned", "anxious", "scared", "uncertain"],
            "surprise": ["surprised", "unexpected", "shocked", "amazed"],
            "trust": ["reliable", "trustworthy", "confident", "secure", "dependable"],
            "anticipation": ["excited", "eager", "looking forward", "hopeful"]
        }
        
    def analyze_sentiment(self, text: str) -> Tuple[FeedbackSentiment, float]:
        """Analyze sentiment of feedback text"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = FeedbackSentiment.POSITIVE
        elif polarity < -0.1:
            sentiment = FeedbackSentiment.NEGATIVE
        else:
            sentiment = FeedbackSentiment.NEUTRAL
            
        return sentiment, polarity
        
    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotional content of feedback"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score / len(keywords)  # Normalize
            
        return emotion_scores
        
    def extract_topics(self, texts: List[str], n_topics: int = 5) -> List[str]:
        """Extract topics from feedback texts using clustering"""
        if len(texts) < n_topics:
            return ["general_feedback"] * len(texts)
            
        # Vectorize texts
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Extract representative terms for each cluster
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for i in range(n_topics):
            # Get top terms for this cluster
            center = kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-5:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            topic_name = "_".join(top_terms[:2])
            topics.append(topic_name)
            
        return [topics[cluster] for cluster in clusters]
        
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from feedback text"""
        blob = TextBlob(text)
        
        # Extract noun phrases and important words
        noun_phrases = list(blob.noun_phrases)
        words = [word.lower() for word in blob.words if len(word) > 3]
        
        # Combine and deduplicate
        keywords = list(set(noun_phrases + words))
        
        # Simple frequency-based ranking (in practice, use TF-IDF)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Sort keywords by frequency
        keywords.sort(key=lambda x: word_freq.get(x, 0), reverse=True)
        
        return keywords[:top_k]
        
    def generate_improvement_suggestions(self, feedback: FeedbackItem, 
                                       processed_data: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on feedback"""
        suggestions = []
        
        # Sentiment-based suggestions
        if processed_data['sentiment'] == FeedbackSentiment.NEGATIVE:
            if processed_data['sentiment_score'] < -0.5:
                suggestions.append("Critical issue requiring immediate attention")
            else:
                suggestions.append("Moderate improvement needed")
                
        # Emotion-based suggestions
        emotions = processed_data['emotion_scores']
        if emotions.get('anger', 0) > 0.3:
            suggestions.append("Address user frustration points")
        if emotions.get('fear', 0) > 0.3:
            suggestions.append("Improve system reliability and user confidence")
        if emotions.get('sadness', 0) > 0.3:
            suggestions.append("Enhance user experience and satisfaction")
            
        # Content-based suggestions
        content_lower = feedback.content.lower()
        if any(word in content_lower for word in ['slow', 'timeout', 'delay']):
            suggestions.append("Optimize response time and performance")
        if any(word in content_lower for word in ['confusing', 'unclear', 'understand']):
            suggestions.append("Improve response clarity and explanation")
        if any(word in content_lower for word in ['wrong', 'incorrect', 'error']):
            suggestions.append("Enhance accuracy and correctness")
        if any(word in content_lower for word in ['feature', 'functionality', 'capability']):
            suggestions.append("Consider feature enhancement or addition")
            
        return suggestions if suggestions else ["General improvement opportunity"]
        
    def calculate_priority_score(self, feedback: FeedbackItem, 
                               processed_data: Dict[str, Any]) -> float:
        """Calculate priority score for feedback"""
        score = 0.0
        
        # Base score from rating
        if feedback.rating is not None:
            score += (5.0 - feedback.rating) * 0.2  # Lower ratings = higher priority
            
        # Sentiment impact
        sentiment_score = processed_data['sentiment_score']
        if sentiment_score < -0.5:
            score += 0.4  # Very negative
        elif sentiment_score < 0:
            score += 0.2  # Negative
            
        # Emotion intensity
        emotions = processed_data['emotion_scores']
        max_emotion = max(emotions.values()) if emotions else 0
        score += max_emotion * 0.3
        
        # Feedback type weight
        type_weights = {
            FeedbackType.EXPERT_REVIEW: 0.3,
            FeedbackType.USER_CORRECTION: 0.25,
            FeedbackType.EXPLICIT_TEXT: 0.2,
            FeedbackType.EXPLICIT_RATING: 0.15,
            FeedbackType.IMPLICIT_BEHAVIOR: 0.1,
            FeedbackType.SYSTEM_METRIC: 0.05
        }
        score += type_weights.get(feedback.feedback_type, 0.1)
        
        return min(score, 1.0)  # Cap at 1.0
        
    def categorize_feedback(self, feedback: FeedbackItem, 
                          processed_data: Dict[str, Any]) -> str:
        """Categorize feedback into predefined categories"""
        content_lower = feedback.content.lower()
        
        # Performance-related
        if any(word in content_lower for word in ['slow', 'fast', 'performance', 'speed', 'timeout']):
            return "performance"
            
        # Accuracy-related
        if any(word in content_lower for word in ['wrong', 'correct', 'accurate', 'mistake', 'error']):
            return "accuracy"
            
        # Usability-related
        if any(word in content_lower for word in ['easy', 'difficult', 'confusing', 'clear', 'interface']):
            return "usability"
            
        # Feature-related
        if any(word in content_lower for word in ['feature', 'functionality', 'capability', 'option']):
            return "features"
            
        # Content quality
        if any(word in content_lower for word in ['helpful', 'useful', 'relevant', 'quality', 'content']):
            return "content_quality"
            
        # General satisfaction
        if any(word in content_lower for word in ['satisfied', 'happy', 'pleased', 'disappointed']):
            return "satisfaction"
            
        return "general"

class FeedbackProcessingSystem:
    """Main feedback processing system"""
    
    def __init__(self, 
                 database_url: str,
                 redis_url: str = "redis://localhost:6379"):
        
        self.database_url = database_url
        self.redis_url = redis_url
        
        # Initialize database
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize Redis
        self.redis_client = None
        
        # Initialize processor
        self.processor = FeedbackProcessor()
        
        # Processing configuration
        self.processing_config = {
            "batch_size": 100,
            "processing_interval": 300,  # 5 minutes
            "insight_generation_interval": 3600,  # 1 hour
            "retention_days": 365
        }
        
    async def initialize(self):
        """Initialize the feedback processing system"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        
        # Start background processing
        asyncio.create_task(self._background_processor())
        asyncio.create_task(self._insight_generator())
        
        logger.info("Feedback processing system initialized")
        
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def collect_feedback(self, 
                             conversation_id: str,
                             user_id: str,
                             feedback_type: FeedbackType,
                             feedback_source: FeedbackSource,
                             content: str,
                             rating: Optional[float] = None,
                             metadata: Dict[str, Any] = None) -> str:
        """Collect new feedback"""
        
        feedback_id = f"fb_{int(time.time())}_{user_id}"
        
        feedback = FeedbackItem(
            feedback_id=feedback_id,
            conversation_id=conversation_id,
            user_id=user_id,
            feedback_type=feedback_type,
            feedback_source=feedback_source,
            content=content,
            rating=rating,
            metadata=metadata or {}
        )
        
        # Store in database
        session = self.SessionLocal()
        try:
            feedback_db = FeedbackDB(
                feedback_id=feedback.feedback_id,
                conversation_id=feedback.conversation_id,
                user_id=feedback.user_id,
                feedback_type=feedback.feedback_type.value,
                feedback_source=feedback.feedback_source.value,
                content=feedback.content,
                rating=feedback.rating,
                metadata=json.dumps(feedback.metadata),
                timestamp=feedback.timestamp
            )
            session.add(feedback_db)
            session.commit()
            
            # Queue for processing
            await self.redis_client.lpush("feedback_queue", feedback_id)
            
            logger.info(f"Collected feedback: {feedback_id}")
            return feedback_id
            
        finally:
            session.close()
            
    async def process_feedback(self, feedback_id: str) -> Optional[ProcessedFeedback]:
        """Process individual feedback item"""
        
        # Load feedback
        session = self.SessionLocal()
        try:
            feedback_db = session.query(FeedbackDB).filter(
                FeedbackDB.feedback_id == feedback_id
            ).first()
            
            if not feedback_db:
                logger.error(f"Feedback not found: {feedback_id}")
                return None
                
            feedback = FeedbackItem(
                feedback_id=feedback_db.feedback_id,
                conversation_id=feedback_db.conversation_id,
                user_id=feedback_db.user_id,
                feedback_type=FeedbackType(feedback_db.feedback_type),
                feedback_source=FeedbackSource(feedback_db.feedback_source),
                content=feedback_db.content,
                rating=feedback_db.rating,
                sentiment=FeedbackSentiment(feedback_db.sentiment) if feedback_db.sentiment else None,
                metadata=json.loads(feedback_db.metadata) if feedback_db.metadata else {},
                timestamp=feedback_db.timestamp,
                processed=feedback_db.processed
            )
            
        finally:
            session.close()
            
        # Process feedback
        sentiment, sentiment_score = self.processor.analyze_sentiment(feedback.content)
        emotion_scores = self.processor.analyze_emotions(feedback.content)
        keywords = self.processor.extract_keywords(feedback.content)
        
        processed_data = {
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'emotion_scores': emotion_scores
        }
        
        improvement_suggestions = self.processor.generate_improvement_suggestions(
            feedback, processed_data
        )
        priority_score = self.processor.calculate_priority_score(feedback, processed_data)
        category = self.processor.categorize_feedback(feedback, processed_data)
        
        # Determine if actionable
        actionable = (
            priority_score > 0.5 or 
            sentiment == FeedbackSentiment.NEGATIVE or
            feedback.feedback_type in [FeedbackType.EXPERT_REVIEW, FeedbackType.USER_CORRECTION]
        )
        
        processed_feedback = ProcessedFeedback(
            feedback_id=feedback_id,
            original_feedback=feedback,
            sentiment_score=sentiment_score,
            emotion_scores=emotion_scores,
            topics=[category],  # Single topic for now
            keywords=keywords,
            improvement_suggestions=improvement_suggestions,
            priority_score=priority_score,
            category=category,
            actionable=actionable,
            processed_at=datetime.utcnow()
        )
        
        # Store processed feedback
        await self._store_processed_feedback(processed_feedback)
        
        # Mark original as processed
        await self._mark_feedback_processed(feedback_id, sentiment)
        
        logger.info(f"Processed feedback: {feedback_id}")
        return processed_feedback
        
    async def _store_processed_feedback(self, processed_feedback: ProcessedFeedback):
        """Store processed feedback in database"""
        session = self.SessionLocal()
        try:
            processed_db = ProcessedFeedbackDB(
                feedback_id=processed_feedback.feedback_id,
                sentiment_score=processed_feedback.sentiment_score,
                emotion_scores=json.dumps(processed_feedback.emotion_scores),
                topics=json.dumps(processed_feedback.topics),
                keywords=json.dumps(processed_feedback.keywords),
                improvement_suggestions=json.dumps(processed_feedback.improvement_suggestions),
                priority_score=processed_feedback.priority_score,
                category=processed_feedback.category,
                actionable=processed_feedback.actionable,
                processed_at=processed_feedback.processed_at
            )
            session.add(processed_db)
            session.commit()
        finally:
            session.close()
            
    async def _mark_feedback_processed(self, feedback_id: str, sentiment: FeedbackSentiment):
        """Mark feedback as processed"""
        session = self.SessionLocal()
        try:
            feedback_db = session.query(FeedbackDB).filter(
                FeedbackDB.feedback_id == feedback_id
            ).first()
            if feedback_db:
                feedback_db.processed = True
                feedback_db.sentiment = sentiment.value
                session.commit()
        finally:
            session.close()
            
    async def _background_processor(self):
        """Background task to process feedback queue"""
        while True:
            try:
                # Process batch of feedback
                feedback_ids = []
                for _ in range(self.processing_config["batch_size"]):
                    feedback_id = await self.redis_client.rpop("feedback_queue")
                    if feedback_id:
                        feedback_ids.append(feedback_id)
                    else:
                        break
                        
                if feedback_ids:
                    logger.info(f"Processing {len(feedback_ids)} feedback items")
                    
                    for feedback_id in feedback_ids:
                        try:
                            await self.process_feedback(feedback_id)
                        except Exception as e:
                            logger.error(f"Error processing feedback {feedback_id}: {e}")
                            
                await asyncio.sleep(self.processing_config["processing_interval"])
                
            except Exception as e:
                logger.error(f"Background processor error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
                
    async def _insight_generator(self):
        """Background task to generate insights"""
        while True:
            try:
                await self.generate_insights()
                await asyncio.sleep(self.processing_config["insight_generation_interval"])
            except Exception as e:
                logger.error(f"Insight generator error: {e}")
                await asyncio.sleep(300)  # Wait before retrying
                
    async def generate_insights(self, time_period: str = "daily") -> FeedbackInsights:
        """Generate aggregated feedback insights"""
        
        # Calculate time range
        now = datetime.utcnow()
        if time_period == "daily":
            start_time = now - timedelta(days=1)
        elif time_period == "weekly":
            start_time = now - timedelta(weeks=1)
        elif time_period == "monthly":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)
            
        session = self.SessionLocal()
        try:
            # Get feedback in time period
            feedback_items = session.query(FeedbackDB).filter(
                FeedbackDB.timestamp >= start_time,
                FeedbackDB.processed == True
            ).all()
            
            if not feedback_items:
                return FeedbackInsights(
                    time_period=time_period,
                    total_feedback_count=0,
                    average_rating=0.0,
                    sentiment_distribution={},
                    top_topics=[],
                    improvement_areas=[],
                    user_satisfaction_trend=[],
                    model_performance_correlation={}
                )
                
            # Calculate metrics
            total_count = len(feedback_items)
            ratings = [f.rating for f in feedback_items if f.rating is not None]
            average_rating = sum(ratings) / len(ratings) if ratings else 0.0
            
            # Sentiment distribution
            sentiments = [f.sentiment for f in feedback_items if f.sentiment]
            sentiment_counts = {}
            for sentiment in sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            sentiment_distribution = {
                k: v / len(sentiments) for k, v in sentiment_counts.items()
            } if sentiments else {}
            
            # Get processed feedback for more insights
            processed_items = session.query(ProcessedFeedbackDB).join(
                FeedbackDB, ProcessedFeedbackDB.feedback_id == FeedbackDB.feedback_id
            ).filter(
                FeedbackDB.timestamp >= start_time
            ).all()
            
            # Top topics
            categories = [p.category for p in processed_items]
            category_counts = {}
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            top_topics = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Improvement areas (from high-priority actionable feedback)
            improvement_areas = []
            for processed in processed_items:
                if processed.actionable and processed.priority_score > 0.7:
                    suggestions = json.loads(processed.improvement_suggestions)
                    improvement_areas.extend(suggestions)
                    
            # Deduplicate and get top improvement areas
            improvement_counts = {}
            for area in improvement_areas:
                improvement_counts[area] = improvement_counts.get(area, 0) + 1
            improvement_areas = sorted(improvement_counts.keys(), 
                                     key=lambda x: improvement_counts[x], 
                                     reverse=True)[:5]
            
            # User satisfaction trend (simplified)
            satisfaction_trend = [(now, average_rating)]
            
            # Model performance correlation (placeholder)
            model_performance_correlation = {
                "response_time": 0.7,
                "accuracy": 0.8,
                "relevance": 0.75
            }
            
            insights = FeedbackInsights(
                time_period=time_period,
                total_feedback_count=total_count,
                average_rating=average_rating,
                sentiment_distribution=sentiment_distribution,
                top_topics=top_topics,
                improvement_areas=improvement_areas,
                user_satisfaction_trend=satisfaction_trend,
                model_performance_correlation=model_performance_correlation
            )
            
            # Store insights
            await self._store_insights(insights)
            
            logger.info(f"Generated insights for {time_period}: {total_count} feedback items")
            return insights
            
        finally:
            session.close()
            
    async def _store_insights(self, insights: FeedbackInsights):
        """Store insights in database"""
        session = self.SessionLocal()
        try:
            insights_db = FeedbackInsightsDB(
                time_period=insights.time_period,
                total_feedback_count=insights.total_feedback_count,
                average_rating=insights.average_rating,
                sentiment_distribution=json.dumps(insights.sentiment_distribution),
                top_topics=json.dumps(insights.top_topics),
                improvement_areas=json.dumps(insights.improvement_areas),
                user_satisfaction_trend=json.dumps([(t.isoformat(), v) for t, v in insights.user_satisfaction_trend]),
                model_performance_correlation=json.dumps(insights.model_performance_correlation),
                created_at=datetime.utcnow()
            )
            session.add(insights_db)
            session.commit()
        finally:
            session.close()
            
    async def get_feedback_insights(self, time_period: str = "daily") -> Optional[FeedbackInsights]:
        """Get latest feedback insights"""
        session = self.SessionLocal()
        try:
            insights_db = session.query(FeedbackInsightsDB).filter(
                FeedbackInsightsDB.time_period == time_period
            ).order_by(FeedbackInsightsDB.created_at.desc()).first()
            
            if not insights_db:
                return None
                
            # Parse satisfaction trend
            trend_data = json.loads(insights_db.user_satisfaction_trend)
            satisfaction_trend = [(datetime.fromisoformat(t), v) for t, v in trend_data]
            
            return FeedbackInsights(
                time_period=insights_db.time_period,
                total_feedback_count=insights_db.total_feedback_count,
                average_rating=insights_db.average_rating,
                sentiment_distribution=json.loads(insights_db.sentiment_distribution),
                top_topics=json.loads(insights_db.top_topics),
                improvement_areas=json.loads(insights_db.improvement_areas),
                user_satisfaction_trend=satisfaction_trend,
                model_performance_correlation=json.loads(insights_db.model_performance_correlation)
            )
            
        finally:
            session.close()
            
    async def get_actionable_feedback(self, limit: int = 50) -> List[ProcessedFeedback]:
        """Get actionable feedback items for review"""
        session = self.SessionLocal()
        try:
            processed_items = session.query(ProcessedFeedbackDB).filter(
                ProcessedFeedbackDB.actionable == True
            ).order_by(ProcessedFeedbackDB.priority_score.desc()).limit(limit).all()
            
            results = []
            for item in processed_items:
                # Get original feedback
                feedback_db = session.query(FeedbackDB).filter(
                    FeedbackDB.feedback_id == item.feedback_id
                ).first()
                
                if feedback_db:
                    original_feedback = FeedbackItem(
                        feedback_id=feedback_db.feedback_id,
                        conversation_id=feedback_db.conversation_id,
                        user_id=feedback_db.user_id,
                        feedback_type=FeedbackType(feedback_db.feedback_type),
                        feedback_source=FeedbackSource(feedback_db.feedback_source),
                        content=feedback_db.content,
                        rating=feedback_db.rating,
                        sentiment=FeedbackSentiment(feedback_db.sentiment) if feedback_db.sentiment else None,
                        metadata=json.loads(feedback_db.metadata) if feedback_db.metadata else {},
                        timestamp=feedback_db.timestamp,
                        processed=feedback_db.processed
                    )
                    
                    processed_feedback = ProcessedFeedback(
                        feedback_id=item.feedback_id,
                        original_feedback=original_feedback,
                        sentiment_score=item.sentiment_score,
                        emotion_scores=json.loads(item.emotion_scores),
                        topics=json.loads(item.topics),
                        keywords=json.loads(item.keywords),
                        improvement_suggestions=json.loads(item.improvement_suggestions),
                        priority_score=item.priority_score,
                        category=item.category,
                        actionable=item.actionable,
                        processed_at=item.processed_at
                    )
                    
                    results.append(processed_feedback)
                    
            return results
            
        finally:
            session.close()

# Example usage and testing
async def main():
    """Example usage of feedback processing system"""
    system = FeedbackProcessingSystem(
        database_url="sqlite:///feedback.db",
        redis_url="redis://localhost:6379"
    )
    
    await system.initialize()
    
    try:
        # Collect some sample feedback
        feedback_id1 = await system.collect_feedback(
            conversation_id="conv_123",
            user_id="user_456",
            feedback_type=FeedbackType.EXPLICIT_TEXT,
            feedback_source=FeedbackSource.USER_INTERFACE,
            content="The response was very helpful and accurate. Great job!",
            rating=5.0
        )
        
        feedback_id2 = await system.collect_feedback(
            conversation_id="conv_124",
            user_id="user_457",
            feedback_type=FeedbackType.EXPLICIT_TEXT,
            feedback_source=FeedbackSource.USER_INTERFACE,
            content="The system was too slow and gave me wrong information. Very frustrated.",
            rating=1.0
        )
        
        print(f"Collected feedback: {feedback_id1}, {feedback_id2}")
        
        # Wait for processing
        await asyncio.sleep(5)
        
        # Get insights
        insights = await system.get_feedback_insights("daily")
        if insights:
            print(f"Total feedback: {insights.total_feedback_count}")
            print(f"Average rating: {insights.average_rating}")
            print(f"Sentiment distribution: {insights.sentiment_distribution}")
            print(f"Top topics: {insights.top_topics}")
            print(f"Improvement areas: {insights.improvement_areas}")
            
        # Get actionable feedback
        actionable = await system.get_actionable_feedback(10)
        print(f"Actionable feedback items: {len(actionable)}")
        
        for item in actionable:
            print(f"Priority: {item.priority_score:.2f}, Category: {item.category}")
            print(f"Suggestions: {item.improvement_suggestions}")
            
    finally:
        await system.close()

if __name__ == "__main__":
    asyncio.run(main())

