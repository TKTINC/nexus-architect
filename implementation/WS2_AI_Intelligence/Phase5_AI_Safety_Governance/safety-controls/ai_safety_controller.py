"""
Nexus Architect AI Safety Controller

This module implements comprehensive AI safety controls including content filtering,
response validation, confidence scoring, and fallback mechanisms for safe AI operations.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety levels for AI responses"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    BLOCKED = "blocked"

class ContentCategory(Enum):
    """Content categories for filtering"""
    HARMFUL = "harmful"
    INAPPROPRIATE = "inappropriate"
    BIASED = "biased"
    MISINFORMATION = "misinformation"
    PERSONAL_INFO = "personal_info"
    SAFE = "safe"

@dataclass
class SafetyAssessment:
    """Safety assessment result"""
    safety_level: SafetyLevel
    confidence_score: float
    content_categories: List[ContentCategory]
    risk_factors: List[str]
    mitigation_actions: List[str]
    explanation: str
    timestamp: datetime

@dataclass
class ResponseValidation:
    """Response validation result"""
    is_valid: bool
    quality_score: float
    safety_score: float
    confidence_score: float
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime

class SafetyConfig:
    """Configuration for AI safety controls"""
    
    def __init__(self):
        self.content_filters = {
            "harmful_content": {
                "enabled": True,
                "threshold": 0.8,
                "keywords": [
                    "violence", "harm", "illegal", "dangerous", "weapon",
                    "suicide", "self-harm", "abuse", "harassment", "threat"
                ],
                "patterns": [
                    r"\b(kill|murder|attack|bomb|terrorist)\b",
                    r"\b(hack|crack|exploit|vulnerability)\b",
                    r"\b(drug|cocaine|heroin|meth)\b"
                ]
            },
            "inappropriate_content": {
                "enabled": True,
                "threshold": 0.7,
                "keywords": [
                    "sexual", "explicit", "adult", "pornographic", "nsfw"
                ],
                "patterns": [
                    r"\b(sex|porn|nude|naked)\b",
                    r"\b(xxx|adult|erotic)\b"
                ]
            },
            "personal_info": {
                "enabled": True,
                "threshold": 0.9,
                "patterns": [
                    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                    r"\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b"  # Phone
                ]
            },
            "misinformation": {
                "enabled": True,
                "threshold": 0.75,
                "keywords": [
                    "conspiracy", "hoax", "fake news", "misinformation",
                    "disinformation", "propaganda"
                ]
            }
        }
        
        self.confidence_thresholds = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5
        }
        
        self.fallback_responses = {
            "low_confidence": "I'm not entirely confident in my response. Let me connect you with a human expert who can provide more accurate information.",
            "safety_concern": "I've detected potential safety concerns with this request. Please rephrase your question or contact our support team for assistance.",
            "inappropriate_content": "I cannot provide information on this topic as it may be inappropriate or harmful. Please ask about something else.",
            "personal_info": "I've detected what appears to be personal information in your message. For your privacy and security, please avoid sharing sensitive details."
        }

class ContentFilter:
    """Advanced content filtering system"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1
        )
        self.bias_detector = pipeline(
            "text-classification",
            model="d4data/bias-detection-model",
            device=0 if torch.cuda.is_available() else -1
        )
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.harmful_patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for content filtering"""
        patterns = []
        for category, settings in self.config.content_filters.items():
            if settings.get("enabled", True) and "patterns" in settings:
                for pattern in settings["patterns"]:
                    patterns.append(re.compile(pattern, re.IGNORECASE))
        return patterns
    
    async def filter_content(self, text: str) -> Tuple[bool, List[ContentCategory], float]:
        """
        Filter content for safety and appropriateness
        
        Args:
            text: Input text to filter
            
        Returns:
            Tuple of (is_safe, detected_categories, confidence_score)
        """
        detected_categories = []
        max_risk_score = 0.0
        
        # Check for harmful patterns
        for pattern in self.harmful_patterns:
            if pattern.search(text):
                detected_categories.append(ContentCategory.HARMFUL)
                max_risk_score = max(max_risk_score, 0.9)
        
        # Check for personal information
        if self._contains_personal_info(text):
            detected_categories.append(ContentCategory.PERSONAL_INFO)
            max_risk_score = max(max_risk_score, 0.8)
        
        # Use ML models for toxicity detection
        try:
            toxicity_result = self.toxicity_classifier(text)
            if toxicity_result[0]['label'] == 'TOXIC' and toxicity_result[0]['score'] > 0.7:
                detected_categories.append(ContentCategory.HARMFUL)
                max_risk_score = max(max_risk_score, toxicity_result[0]['score'])
        except Exception as e:
            logger.warning(f"Toxicity classification failed: {e}")
        
        # Use ML models for bias detection
        try:
            bias_result = self.bias_detector(text)
            if bias_result[0]['label'] == 'BIASED' and bias_result[0]['score'] > 0.7:
                detected_categories.append(ContentCategory.BIASED)
                max_risk_score = max(max_risk_score, bias_result[0]['score'])
        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")
        
        # Check keyword-based filters
        text_lower = text.lower()
        for category, settings in self.config.content_filters.items():
            if not settings.get("enabled", True):
                continue
                
            keywords = settings.get("keywords", [])
            threshold = settings.get("threshold", 0.7)
            
            keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_matches > 0:
                risk_score = min(keyword_matches / len(keywords), 1.0)
                if risk_score >= threshold:
                    if category == "harmful_content":
                        detected_categories.append(ContentCategory.HARMFUL)
                    elif category == "inappropriate_content":
                        detected_categories.append(ContentCategory.INAPPROPRIATE)
                    elif category == "misinformation":
                        detected_categories.append(ContentCategory.MISINFORMATION)
                    max_risk_score = max(max_risk_score, risk_score)
        
        # Determine if content is safe
        is_safe = len(detected_categories) == 0 or max_risk_score < 0.5
        if is_safe and len(detected_categories) == 0:
            detected_categories.append(ContentCategory.SAFE)
        
        return is_safe, detected_categories, 1.0 - max_risk_score
    
    def _contains_personal_info(self, text: str) -> bool:
        """Check if text contains personal information"""
        patterns = self.config.content_filters["personal_info"]["patterns"]
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

class ConfidenceScorer:
    """Confidence scoring for AI responses"""
    
    def __init__(self):
        self.uncertainty_indicators = [
            "i think", "maybe", "possibly", "might be", "could be",
            "not sure", "uncertain", "unclear", "ambiguous", "probably"
        ]
        self.confidence_indicators = [
            "definitely", "certainly", "absolutely", "clearly", "obviously",
            "without doubt", "confirmed", "verified", "established"
        ]
    
    def calculate_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """
        Calculate confidence score for AI response
        
        Args:
            response: AI response text
            context: Additional context information
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.7  # Base confidence level
        
        # Adjust based on uncertainty indicators
        uncertainty_count = sum(1 for indicator in self.uncertainty_indicators 
                               if indicator in response.lower())
        confidence_count = sum(1 for indicator in self.confidence_indicators 
                              if indicator in response.lower())
        
        # Adjust confidence based on indicators
        confidence_adjustment = (confidence_count - uncertainty_count) * 0.1
        adjusted_confidence = base_confidence + confidence_adjustment
        
        # Adjust based on response length and complexity
        response_length = len(response.split())
        if response_length < 10:
            adjusted_confidence -= 0.1  # Very short responses are less confident
        elif response_length > 200:
            adjusted_confidence += 0.05  # Detailed responses are more confident
        
        # Adjust based on context quality
        if context.get("knowledge_graph_matches", 0) > 5:
            adjusted_confidence += 0.1
        if context.get("training_data_similarity", 0) > 0.8:
            adjusted_confidence += 0.1
        if context.get("persona_expertise_match", 0) > 0.9:
            adjusted_confidence += 0.15
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, adjusted_confidence))

class ResponseValidator:
    """Response validation and quality assurance"""
    
    def __init__(self, content_filter: ContentFilter, confidence_scorer: ConfidenceScorer):
        self.content_filter = content_filter
        self.confidence_scorer = confidence_scorer
        self.quality_metrics = {
            "coherence": 0.0,
            "relevance": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0
        }
    
    async def validate_response(self, response: str, query: str, context: Dict[str, Any]) -> ResponseValidation:
        """
        Validate AI response for quality and safety
        
        Args:
            response: AI response to validate
            query: Original user query
            context: Additional context information
            
        Returns:
            ResponseValidation object with validation results
        """
        issues = []
        recommendations = []
        
        # Safety validation
        is_safe, categories, safety_score = await self.content_filter.filter_content(response)
        if not is_safe:
            issues.append(f"Safety concerns detected: {', '.join([cat.value for cat in categories])}")
            recommendations.append("Review and modify response to address safety concerns")
        
        # Confidence scoring
        confidence_score = self.confidence_scorer.calculate_confidence(response, context)
        if confidence_score < 0.5:
            issues.append("Low confidence in response accuracy")
            recommendations.append("Consider providing alternative sources or human expert consultation")
        
        # Quality assessment
        quality_score = self._assess_quality(response, query, context)
        if quality_score < 0.6:
            issues.append("Response quality below acceptable threshold")
            recommendations.append("Improve response clarity, relevance, and completeness")
        
        # Length validation
        if len(response.strip()) < 20:
            issues.append("Response too short")
            recommendations.append("Provide more detailed and comprehensive response")
        elif len(response) > 5000:
            issues.append("Response too long")
            recommendations.append("Summarize key points for better readability")
        
        # Relevance check
        relevance_score = self._calculate_relevance(response, query)
        if relevance_score < 0.7:
            issues.append("Response not sufficiently relevant to query")
            recommendations.append("Focus response more directly on the user's question")
        
        is_valid = len(issues) == 0 and safety_score > 0.7 and quality_score > 0.6
        
        return ResponseValidation(
            is_valid=is_valid,
            quality_score=quality_score,
            safety_score=safety_score,
            confidence_score=confidence_score,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.utcnow()
        )
    
    def _assess_quality(self, response: str, query: str, context: Dict[str, Any]) -> float:
        """Assess overall response quality"""
        scores = []
        
        # Coherence: Check if response is well-structured
        coherence_score = self._assess_coherence(response)
        scores.append(coherence_score)
        
        # Relevance: Check if response addresses the query
        relevance_score = self._calculate_relevance(response, query)
        scores.append(relevance_score)
        
        # Completeness: Check if response is comprehensive
        completeness_score = self._assess_completeness(response, query)
        scores.append(completeness_score)
        
        # Technical accuracy (if applicable)
        if context.get("technical_query", False):
            accuracy_score = self._assess_technical_accuracy(response, context)
            scores.append(accuracy_score)
        
        return np.mean(scores)
    
    def _assess_coherence(self, response: str) -> float:
        """Assess response coherence and structure"""
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.6  # Single sentence responses are less coherent
        
        # Check for logical flow indicators
        flow_indicators = ["first", "second", "then", "next", "finally", "however", "therefore", "because"]
        flow_count = sum(1 for indicator in flow_indicators if indicator in response.lower())
        
        # Check for proper sentence structure
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        coherence_score = 0.7  # Base score
        if flow_count > 0:
            coherence_score += min(flow_count * 0.05, 0.2)
        if 10 <= avg_sentence_length <= 25:
            coherence_score += 0.1
        
        return min(1.0, coherence_score)
    
    def _calculate_relevance(self, response: str, query: str) -> float:
        """Calculate relevance between response and query"""
        try:
            # Use TF-IDF similarity
            documents = [query, response]
            tfidf_matrix = self.content_filter.vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            # Fallback to keyword overlap
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words.intersection(response_words))
            return min(overlap / len(query_words), 1.0) if query_words else 0.0
    
    def _assess_completeness(self, response: str, query: str) -> float:
        """Assess if response completely addresses the query"""
        # Check for question words in query
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        query_lower = query.lower()
        
        completeness_score = 0.7  # Base score
        
        # If query asks "how", check for step-by-step explanation
        if "how" in query_lower and any(word in response.lower() for word in ["step", "first", "then", "next"]):
            completeness_score += 0.2
        
        # If query asks "why", check for explanations
        if "why" in query_lower and any(word in response.lower() for word in ["because", "due to", "reason", "cause"]):
            completeness_score += 0.2
        
        # Check response length relative to query complexity
        query_complexity = len(query.split())
        response_length = len(response.split())
        
        if query_complexity > 10 and response_length > 50:
            completeness_score += 0.1
        
        return min(1.0, completeness_score)
    
    def _assess_technical_accuracy(self, response: str, context: Dict[str, Any]) -> float:
        """Assess technical accuracy of response"""
        # This would integrate with knowledge base validation
        # For now, return a score based on context confidence
        knowledge_confidence = context.get("knowledge_graph_confidence", 0.7)
        training_confidence = context.get("training_data_confidence", 0.7)
        
        return (knowledge_confidence + training_confidence) / 2

class FallbackManager:
    """Manage fallback responses for various scenarios"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.fallback_strategies = {
            "low_confidence": self._handle_low_confidence,
            "safety_concern": self._handle_safety_concern,
            "inappropriate_content": self._handle_inappropriate_content,
            "personal_info": self._handle_personal_info,
            "technical_error": self._handle_technical_error,
            "knowledge_gap": self._handle_knowledge_gap
        }
    
    def get_fallback_response(self, scenario: str, context: Dict[str, Any]) -> str:
        """Get appropriate fallback response for scenario"""
        if scenario in self.fallback_strategies:
            return self.fallback_strategies[scenario](context)
        else:
            return self.config.fallback_responses.get(scenario, 
                "I apologize, but I'm unable to provide a response to your request. Please try rephrasing your question or contact our support team.")
    
    def _handle_low_confidence(self, context: Dict[str, Any]) -> str:
        """Handle low confidence scenarios"""
        confidence = context.get("confidence_score", 0.0)
        base_response = self.config.fallback_responses["low_confidence"]
        
        if confidence < 0.3:
            return f"{base_response} My confidence in this response is very low ({confidence:.1%})."
        else:
            return f"{base_response} My confidence in this response is moderate ({confidence:.1%})."
    
    def _handle_safety_concern(self, context: Dict[str, Any]) -> str:
        """Handle safety concern scenarios"""
        categories = context.get("safety_categories", [])
        base_response = self.config.fallback_responses["safety_concern"]
        
        if ContentCategory.HARMFUL in categories:
            return f"{base_response} The request appears to involve potentially harmful content."
        elif ContentCategory.INAPPROPRIATE in categories:
            return f"{base_response} The request contains inappropriate content."
        else:
            return base_response
    
    def _handle_inappropriate_content(self, context: Dict[str, Any]) -> str:
        """Handle inappropriate content scenarios"""
        return self.config.fallback_responses["inappropriate_content"]
    
    def _handle_personal_info(self, context: Dict[str, Any]) -> str:
        """Handle personal information scenarios"""
        return self.config.fallback_responses["personal_info"]
    
    def _handle_technical_error(self, context: Dict[str, Any]) -> str:
        """Handle technical error scenarios"""
        error_type = context.get("error_type", "unknown")
        return f"I'm experiencing technical difficulties ({error_type}). Please try again in a moment or contact our support team if the issue persists."
    
    def _handle_knowledge_gap(self, context: Dict[str, Any]) -> str:
        """Handle knowledge gap scenarios"""
        topic = context.get("topic", "this topic")
        return f"I don't have sufficient information about {topic} to provide a reliable response. Let me connect you with a subject matter expert who can help."

class AISafetyController:
    """Main AI Safety Controller orchestrating all safety mechanisms"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 db_url: str = "postgresql://localhost/nexus_safety"):
        self.config = SafetyConfig()
        self.content_filter = ContentFilter(self.config)
        self.confidence_scorer = ConfidenceScorer()
        self.response_validator = ResponseValidator(self.content_filter, self.confidence_scorer)
        self.fallback_manager = FallbackManager(self.config)
        
        # Initialize connections
        self.redis_client = redis.from_url(redis_url)
        self.db_url = db_url
        
        # Metrics tracking
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "fallback_responses": 0,
            "safety_violations": 0,
            "confidence_scores": []
        }
    
    async def assess_safety(self, text: str, context: Dict[str, Any] = None) -> SafetyAssessment:
        """
        Comprehensive safety assessment of input text
        
        Args:
            text: Input text to assess
            context: Additional context information
            
        Returns:
            SafetyAssessment with detailed safety analysis
        """
        if context is None:
            context = {}
        
        # Content filtering
        is_safe, categories, safety_score = await self.content_filter.filter_content(text)
        
        # Determine safety level
        if safety_score > 0.9:
            safety_level = SafetyLevel.SAFE
        elif safety_score > 0.7:
            safety_level = SafetyLevel.CAUTION
        elif safety_score > 0.5:
            safety_level = SafetyLevel.WARNING
        else:
            safety_level = SafetyLevel.BLOCKED
        
        # Identify risk factors
        risk_factors = []
        mitigation_actions = []
        
        for category in categories:
            if category == ContentCategory.HARMFUL:
                risk_factors.append("Potentially harmful content detected")
                mitigation_actions.append("Block response and provide safety resources")
            elif category == ContentCategory.INAPPROPRIATE:
                risk_factors.append("Inappropriate content detected")
                mitigation_actions.append("Filter content and provide appropriate alternative")
            elif category == ContentCategory.BIASED:
                risk_factors.append("Potential bias detected")
                mitigation_actions.append("Review for bias and provide balanced perspective")
            elif category == ContentCategory.PERSONAL_INFO:
                risk_factors.append("Personal information detected")
                mitigation_actions.append("Redact personal information and warn user")
            elif category == ContentCategory.MISINFORMATION:
                risk_factors.append("Potential misinformation detected")
                mitigation_actions.append("Fact-check and provide verified sources")
        
        # Generate explanation
        explanation = self._generate_safety_explanation(safety_level, categories, safety_score)
        
        # Update metrics
        self.metrics["total_requests"] += 1
        if safety_level == SafetyLevel.BLOCKED:
            self.metrics["blocked_requests"] += 1
            self.metrics["safety_violations"] += 1
        
        # Log assessment
        await self._log_safety_assessment(text, safety_level, categories, safety_score)
        
        return SafetyAssessment(
            safety_level=safety_level,
            confidence_score=safety_score,
            content_categories=categories,
            risk_factors=risk_factors,
            mitigation_actions=mitigation_actions,
            explanation=explanation,
            timestamp=datetime.utcnow()
        )
    
    async def validate_and_process_response(self, response: str, query: str, 
                                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate AI response and apply safety controls
        
        Args:
            response: AI response to validate
            query: Original user query
            context: Additional context information
            
        Returns:
            Dictionary with processed response and safety information
        """
        if context is None:
            context = {}
        
        # Validate response
        validation = await self.response_validator.validate_response(response, query, context)
        
        # Assess safety of response
        safety_assessment = await self.assess_safety(response, context)
        
        # Determine if response should be blocked or modified
        if safety_assessment.safety_level == SafetyLevel.BLOCKED:
            # Use fallback response
            fallback_response = self.fallback_manager.get_fallback_response(
                "safety_concern", 
                {"safety_categories": safety_assessment.content_categories}
            )
            self.metrics["fallback_responses"] += 1
            
            return {
                "response": fallback_response,
                "original_response": response,
                "safety_assessment": asdict(safety_assessment),
                "validation": asdict(validation),
                "action_taken": "blocked_and_replaced",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif not validation.is_valid or validation.confidence_score < 0.5:
            # Use fallback for low confidence or invalid responses
            fallback_scenario = "low_confidence" if validation.confidence_score < 0.5 else "technical_error"
            fallback_response = self.fallback_manager.get_fallback_response(
                fallback_scenario,
                {"confidence_score": validation.confidence_score}
            )
            self.metrics["fallback_responses"] += 1
            
            return {
                "response": fallback_response,
                "original_response": response,
                "safety_assessment": asdict(safety_assessment),
                "validation": asdict(validation),
                "action_taken": "replaced_due_to_quality",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif safety_assessment.safety_level == SafetyLevel.WARNING:
            # Add safety disclaimer to response
            disclaimer = "\n\n⚠️ Please note: This response has been flagged for potential safety concerns. Use this information carefully and consider consulting additional sources."
            modified_response = response + disclaimer
            
            return {
                "response": modified_response,
                "original_response": response,
                "safety_assessment": asdict(safety_assessment),
                "validation": asdict(validation),
                "action_taken": "modified_with_warning",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        else:
            # Response is safe and valid
            return {
                "response": response,
                "safety_assessment": asdict(safety_assessment),
                "validation": asdict(validation),
                "action_taken": "approved",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _generate_safety_explanation(self, safety_level: SafetyLevel, 
                                   categories: List[ContentCategory], 
                                   safety_score: float) -> str:
        """Generate human-readable safety explanation"""
        if safety_level == SafetyLevel.SAFE:
            return f"Content appears safe with high confidence (score: {safety_score:.2f})"
        elif safety_level == SafetyLevel.CAUTION:
            return f"Content requires caution due to: {', '.join([cat.value for cat in categories])} (score: {safety_score:.2f})"
        elif safety_level == SafetyLevel.WARNING:
            return f"Content has safety warnings for: {', '.join([cat.value for cat in categories])} (score: {safety_score:.2f})"
        else:
            return f"Content blocked due to: {', '.join([cat.value for cat in categories])} (score: {safety_score:.2f})"
    
    async def _log_safety_assessment(self, text: str, safety_level: SafetyLevel, 
                                   categories: List[ContentCategory], safety_score: float):
        """Log safety assessment for monitoring and analysis"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "text_hash": hashlib.sha256(text.encode()).hexdigest(),
            "safety_level": safety_level.value,
            "categories": [cat.value for cat in categories],
            "safety_score": safety_score,
            "text_length": len(text)
        }
        
        # Store in Redis for real-time monitoring
        await self._store_in_redis("safety_log", log_entry)
        
        # Store in database for long-term analysis
        await self._store_in_database("safety_assessments", log_entry)
    
    async def _store_in_redis(self, key: str, data: Dict[str, Any]):
        """Store data in Redis"""
        try:
            self.redis_client.lpush(key, json.dumps(data))
            self.redis_client.ltrim(key, 0, 9999)  # Keep last 10k entries
        except Exception as e:
            logger.error(f"Failed to store in Redis: {e}")
    
    async def _store_in_database(self, table: str, data: Dict[str, Any]):
        """Store data in PostgreSQL database"""
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            columns = list(data.keys())
            values = list(data.values())
            placeholders = ', '.join(['%s'] * len(values))
            
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            cursor.execute(query, values)
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store in database: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get safety controller metrics"""
        avg_confidence = np.mean(self.metrics["confidence_scores"]) if self.metrics["confidence_scores"] else 0.0
        
        return {
            "total_requests": self.metrics["total_requests"],
            "blocked_requests": self.metrics["blocked_requests"],
            "fallback_responses": self.metrics["fallback_responses"],
            "safety_violations": self.metrics["safety_violations"],
            "block_rate": self.metrics["blocked_requests"] / max(self.metrics["total_requests"], 1),
            "fallback_rate": self.metrics["fallback_responses"] / max(self.metrics["total_requests"], 1),
            "average_confidence": avg_confidence,
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }

# FastAPI application
app = FastAPI(title="Nexus Architect AI Safety Controller", version="1.0.0")
security = HTTPBearer()

# Global safety controller instance
safety_controller = None

@app.on_event("startup")
async def startup_event():
    """Initialize safety controller on startup"""
    global safety_controller
    safety_controller = AISafetyController()
    safety_controller.start_time = time.time()
    logger.info("AI Safety Controller started successfully")

# Pydantic models for API
class SafetyAssessmentRequest(BaseModel):
    text: str = Field(..., description="Text to assess for safety")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class ResponseValidationRequest(BaseModel):
    response: str = Field(..., description="AI response to validate")
    query: str = Field(..., description="Original user query")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

@app.post("/safety/assess")
async def assess_safety(request: SafetyAssessmentRequest, 
                       credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Assess safety of input text"""
    try:
        assessment = await safety_controller.assess_safety(request.text, request.context)
        return asdict(assessment)
    except Exception as e:
        logger.error(f"Safety assessment failed: {e}")
        raise HTTPException(status_code=500, detail="Safety assessment failed")

@app.post("/safety/validate-response")
async def validate_response(request: ResponseValidationRequest,
                          credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate and process AI response"""
    try:
        result = await safety_controller.validate_and_process_response(
            request.response, request.query, request.context
        )
        return result
    except Exception as e:
        logger.error(f"Response validation failed: {e}")
        raise HTTPException(status_code=500, detail="Response validation failed")

@app.get("/safety/metrics")
async def get_metrics(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get safety controller metrics"""
    try:
        return safety_controller.get_metrics()
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@app.get("/safety/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

