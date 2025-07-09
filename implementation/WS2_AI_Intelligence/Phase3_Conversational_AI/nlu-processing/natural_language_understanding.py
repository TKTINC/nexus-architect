"""
Nexus Architect Natural Language Understanding System

This module provides comprehensive NLU capabilities including intent classification,
entity extraction, sentiment analysis, and query complexity assessment.
"""

import re
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

import spacy
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

class IntentCategory(str, Enum):
    """Categories of user intents"""
    QUESTION = "question"
    REQUEST = "request"
    COMMAND = "command"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    HELP = "help"
    TECHNICAL_SUPPORT = "technical_support"
    ARCHITECTURE_REVIEW = "architecture_review"
    CODE_REVIEW = "code_review"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    SECURITY_ASSESSMENT = "security_assessment"
    COMPLIANCE_CHECK = "compliance_check"

class QueryComplexity(str, Enum):
    """Complexity levels for queries"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class SentimentPolarity(str, Enum):
    """Sentiment polarity levels"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    metadata: Dict[str, Any] = None

@dataclass
class Intent:
    """Represents a classified intent"""
    category: IntentCategory
    confidence: float
    subcategory: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class Sentiment:
    """Represents sentiment analysis results"""
    polarity: SentimentPolarity
    score: float
    confidence: float
    emotions: Dict[str, float] = None

@dataclass
class NLUResult:
    """Complete NLU analysis result"""
    text: str
    intent: Intent
    entities: List[Entity]
    sentiment: Sentiment
    complexity: QueryComplexity
    language: str
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any] = None

class IntentClassifier:
    """Intent classification using transformer models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.intent_patterns = self._load_intent_patterns()
        
    async def initialize(self):
        """Initialize the intent classification model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info(f"Intent classifier initialized with {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize intent classifier: {e}")
            # Fallback to pattern-based classification
            
    def _load_intent_patterns(self) -> Dict[IntentCategory, List[str]]:
        """Load intent classification patterns"""
        return {
            IntentCategory.QUESTION: [
                r"\b(what|how|why|when|where|which|who)\b",
                r"\?$",
                r"\b(explain|describe|tell me)\b"
            ],
            IntentCategory.REQUEST: [
                r"\b(please|can you|could you|would you)\b",
                r"\b(help me|assist me|show me)\b",
                r"\b(I need|I want|I would like)\b"
            ],
            IntentCategory.COMMAND: [
                r"\b(create|build|implement|deploy|configure)\b",
                r"\b(run|execute|start|stop|restart)\b",
                r"\b(update|modify|change|fix)\b"
            ],
            IntentCategory.TECHNICAL_SUPPORT: [
                r"\b(error|bug|issue|problem|broken)\b",
                r"\b(not working|failing|crash)\b",
                r"\b(troubleshoot|debug|diagnose)\b"
            ],
            IntentCategory.ARCHITECTURE_REVIEW: [
                r"\b(architecture|design|pattern|structure)\b",
                r"\b(review|evaluate|assess|analyze)\b",
                r"\b(best practice|recommendation)\b"
            ],
            IntentCategory.SECURITY_ASSESSMENT: [
                r"\b(security|vulnerability|threat|risk)\b",
                r"\b(authentication|authorization|encryption)\b",
                r"\b(secure|protect|safety)\b"
            ],
            IntentCategory.PERFORMANCE_ANALYSIS: [
                r"\b(performance|speed|latency|throughput)\b",
                r"\b(optimize|improve|faster|slower)\b",
                r"\b(bottleneck|scalability|load)\b"
            ]
        }
        
    async def classify_intent(self, text: str) -> Intent:
        """Classify the intent of the input text"""
        text_lower = text.lower()
        
        # Pattern-based classification
        intent_scores = {}
        for intent_category, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            if score > 0:
                intent_scores[intent_category] = score / len(patterns)
                
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[best_intent], 1.0)
            
            return Intent(
                category=best_intent,
                confidence=confidence,
                metadata={"pattern_scores": intent_scores}
            )
        
        # Default to question if no clear intent
        return Intent(
            category=IntentCategory.QUESTION,
            confidence=0.5,
            metadata={"fallback": True}
        )

class EntityExtractor:
    """Named entity recognition and extraction"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self.custom_entities = self._load_custom_entities()
        
    async def initialize(self):
        """Initialize the entity extraction model"""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Entity extractor initialized with {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            # Fallback to pattern-based extraction
            
    def _load_custom_entities(self) -> Dict[str, List[str]]:
        """Load custom entity patterns for technical domains"""
        return {
            "TECHNOLOGY": [
                "kubernetes", "docker", "redis", "postgresql", "mongodb",
                "react", "angular", "vue", "nodejs", "python", "java",
                "aws", "azure", "gcp", "terraform", "ansible"
            ],
            "ARCHITECTURE_PATTERN": [
                "microservices", "monolith", "serverless", "event-driven",
                "mvc", "mvp", "mvvm", "clean architecture", "hexagonal"
            ],
            "SECURITY_CONCEPT": [
                "oauth", "jwt", "ssl", "tls", "encryption", "authentication",
                "authorization", "rbac", "saml", "ldap"
            ],
            "PERFORMANCE_METRIC": [
                "latency", "throughput", "rps", "cpu", "memory", "disk",
                "bandwidth", "response time", "load time"
            ]
        }
        
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        entities = []
        
        if self.nlp:
            # Use spaCy for standard entity extraction
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9,  # spaCy doesn't provide confidence scores
                    metadata={"source": "spacy"}
                ))
        
        # Custom entity extraction for technical terms
        text_lower = text.lower()
        for entity_type, terms in self.custom_entities.items():
            for term in terms:
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    entities.append(Entity(
                        text=text[match.start():match.end()],
                        label=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                        metadata={"source": "custom_patterns"}
                    ))
        
        # Remove duplicates and overlapping entities
        entities = self._remove_overlapping_entities(entities)
        return entities
        
    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping the one with higher confidence"""
        entities.sort(key=lambda x: (x.start, -x.confidence))
        filtered = []
        
        for entity in entities:
            overlap = False
            for existing in filtered:
                if (entity.start < existing.end and entity.end > existing.start):
                    overlap = True
                    break
            if not overlap:
                filtered.append(entity)
                
        return filtered

class SentimentAnalyzer:
    """Sentiment analysis and emotion detection"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        
    async def initialize(self):
        """Initialize sentiment analysis models"""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            logger.info("Sentiment analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            
    async def analyze_sentiment(self, text: str) -> Sentiment:
        """Analyze sentiment and emotions in text"""
        if not self.sentiment_pipeline:
            return Sentiment(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0
            )
            
        try:
            # Sentiment analysis
            sentiment_result = self.sentiment_pipeline(text)[0]
            
            # Map labels to our enum
            label_mapping = {
                "POSITIVE": SentimentPolarity.POSITIVE,
                "NEGATIVE": SentimentPolarity.NEGATIVE,
                "NEUTRAL": SentimentPolarity.NEUTRAL
            }
            
            polarity = label_mapping.get(sentiment_result["label"], SentimentPolarity.NEUTRAL)
            
            # Emotion analysis
            emotions = {}
            if self.emotion_pipeline:
                emotion_results = self.emotion_pipeline(text)
                for result in emotion_results:
                    emotions[result["label"]] = result["score"]
            
            return Sentiment(
                polarity=polarity,
                score=sentiment_result["score"],
                confidence=sentiment_result["score"],
                emotions=emotions
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return Sentiment(
                polarity=SentimentPolarity.NEUTRAL,
                score=0.0,
                confidence=0.0
            )

class ComplexityAssessor:
    """Assess query complexity for appropriate routing"""
    
    def __init__(self):
        self.technical_terms = self._load_technical_terms()
        self.complexity_indicators = self._load_complexity_indicators()
        
    def _load_technical_terms(self) -> List[str]:
        """Load technical terms that indicate complexity"""
        return [
            "algorithm", "optimization", "scalability", "architecture",
            "distributed", "microservices", "kubernetes", "docker",
            "machine learning", "artificial intelligence", "blockchain",
            "cryptography", "security", "performance", "latency",
            "throughput", "concurrency", "parallelism", "async"
        ]
        
    def _load_complexity_indicators(self) -> Dict[QueryComplexity, Dict[str, Any]]:
        """Load indicators for different complexity levels"""
        return {
            QueryComplexity.SIMPLE: {
                "max_words": 10,
                "max_technical_terms": 1,
                "patterns": [r"\b(what is|how to|can you)\b"]
            },
            QueryComplexity.MODERATE: {
                "max_words": 25,
                "max_technical_terms": 3,
                "patterns": [r"\b(implement|configure|setup)\b"]
            },
            QueryComplexity.COMPLEX: {
                "max_words": 50,
                "max_technical_terms": 5,
                "patterns": [r"\b(optimize|integrate|architect)\b"]
            },
            QueryComplexity.EXPERT: {
                "max_words": float('inf'),
                "max_technical_terms": float('inf'),
                "patterns": [r"\b(distributed|scalable|enterprise)\b"]
            }
        }
        
    async def assess_complexity(self, text: str, entities: List[Entity]) -> QueryComplexity:
        """Assess the complexity of a query"""
        word_count = len(text.split())
        technical_term_count = sum(1 for entity in entities 
                                 if entity.label in ["TECHNOLOGY", "ARCHITECTURE_PATTERN"])
        
        text_lower = text.lower()
        
        # Check for expert-level indicators
        expert_patterns = self.complexity_indicators[QueryComplexity.EXPERT]["patterns"]
        if any(re.search(pattern, text_lower) for pattern in expert_patterns):
            return QueryComplexity.EXPERT
            
        # Check for complex indicators
        complex_patterns = self.complexity_indicators[QueryComplexity.COMPLEX]["patterns"]
        if (word_count > 25 or technical_term_count > 3 or
            any(re.search(pattern, text_lower) for pattern in complex_patterns)):
            return QueryComplexity.COMPLEX
            
        # Check for moderate indicators
        moderate_patterns = self.complexity_indicators[QueryComplexity.MODERATE]["patterns"]
        if (word_count > 10 or technical_term_count > 1 or
            any(re.search(pattern, text_lower) for pattern in moderate_patterns)):
            return QueryComplexity.MODERATE
            
        return QueryComplexity.SIMPLE

class NaturalLanguageUnderstanding:
    """Main NLU system combining all components"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.complexity_assessor = ComplexityAssessor()
        self.language_detector = None
        
    async def initialize(self):
        """Initialize all NLU components"""
        await self.intent_classifier.initialize()
        await self.entity_extractor.initialize()
        await self.sentiment_analyzer.initialize()
        
        # Initialize language detection
        try:
            from langdetect import detect
            self.language_detector = detect
            logger.info("Language detection initialized")
        except ImportError:
            logger.warning("langdetect not available, defaulting to English")
            
        logger.info("NLU system fully initialized")
        
    async def analyze(self, text: str) -> NLUResult:
        """Perform complete NLU analysis on input text"""
        start_time = asyncio.get_event_loop().time()
        
        # Detect language
        language = "en"
        if self.language_detector:
            try:
                language = self.language_detector(text)
            except:
                language = "en"
                
        # Parallel processing of NLU components
        intent_task = self.intent_classifier.classify_intent(text)
        entities_task = self.entity_extractor.extract_entities(text)
        sentiment_task = self.sentiment_analyzer.analyze_sentiment(text)
        
        intent, entities, sentiment = await asyncio.gather(
            intent_task, entities_task, sentiment_task
        )
        
        # Assess complexity
        complexity = await self.complexity_assessor.assess_complexity(text, entities)
        
        # Calculate overall confidence score
        confidence_score = (
            intent.confidence * 0.4 +
            sentiment.confidence * 0.3 +
            (1.0 if entities else 0.5) * 0.3
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return NLUResult(
            text=text,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            complexity=complexity,
            language=language,
            confidence_score=confidence_score,
            processing_time=processing_time,
            metadata={
                "entity_count": len(entities),
                "word_count": len(text.split()),
                "character_count": len(text)
            }
        )
        
    async def batch_analyze(self, texts: List[str]) -> List[NLUResult]:
        """Analyze multiple texts in batch"""
        tasks = [self.analyze(text) for text in texts]
        return await asyncio.gather(*tasks)

# Query routing based on NLU results
class QueryRouter:
    """Route queries to appropriate personas based on NLU analysis"""
    
    def __init__(self):
        self.routing_rules = self._load_routing_rules()
        
    def _load_routing_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load routing rules for different personas"""
        return {
            "security_architect": {
                "intents": [IntentCategory.SECURITY_ASSESSMENT],
                "entities": ["SECURITY_CONCEPT"],
                "keywords": ["security", "vulnerability", "threat", "authentication"],
                "complexity": [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]
            },
            "performance_engineer": {
                "intents": [IntentCategory.PERFORMANCE_ANALYSIS],
                "entities": ["PERFORMANCE_METRIC"],
                "keywords": ["performance", "optimization", "latency", "throughput"],
                "complexity": [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
            },
            "application_architect": {
                "intents": [IntentCategory.ARCHITECTURE_REVIEW],
                "entities": ["ARCHITECTURE_PATTERN", "TECHNOLOGY"],
                "keywords": ["architecture", "design", "pattern", "structure"],
                "complexity": [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]
            },
            "devops_specialist": {
                "intents": [IntentCategory.COMMAND, IntentCategory.TECHNICAL_SUPPORT],
                "entities": ["TECHNOLOGY"],
                "keywords": ["deploy", "ci/cd", "infrastructure", "automation"],
                "complexity": [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
            },
            "compliance_auditor": {
                "intents": [IntentCategory.COMPLIANCE_CHECK],
                "entities": [],
                "keywords": ["compliance", "audit", "regulation", "policy"],
                "complexity": [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]
            }
        }
        
    async def route_query(self, nlu_result: NLUResult) -> List[Tuple[str, float]]:
        """Route query to appropriate personas with confidence scores"""
        persona_scores = {}
        
        for persona, rules in self.routing_rules.items():
            score = 0.0
            
            # Intent matching
            if nlu_result.intent.category in rules["intents"]:
                score += 0.4 * nlu_result.intent.confidence
                
            # Entity matching
            entity_labels = [entity.label for entity in nlu_result.entities]
            entity_matches = sum(1 for label in entity_labels if label in rules["entities"])
            if entity_matches > 0:
                score += 0.3 * (entity_matches / len(rules["entities"]) if rules["entities"] else 0)
                
            # Keyword matching
            text_lower = nlu_result.text.lower()
            keyword_matches = sum(1 for keyword in rules["keywords"] if keyword in text_lower)
            if keyword_matches > 0:
                score += 0.2 * (keyword_matches / len(rules["keywords"]))
                
            # Complexity matching
            if nlu_result.complexity in rules["complexity"]:
                score += 0.1
                
            if score > 0:
                persona_scores[persona] = score
                
        # Sort by score and return top candidates
        sorted_personas = sorted(persona_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_personas[:3]  # Return top 3 candidates

# Example usage and testing
async def main():
    """Example usage of the NLU system"""
    nlu = NaturalLanguageUnderstanding()
    await nlu.initialize()
    
    router = QueryRouter()
    
    test_queries = [
        "How do I implement OAuth authentication in my microservices architecture?",
        "The system is running slowly, can you help optimize the database performance?",
        "What are the best practices for designing a scalable event-driven architecture?",
        "I need to set up a CI/CD pipeline for our Kubernetes deployment",
        "Can you review our system for GDPR compliance requirements?"
    ]
    
    for query in test_queries:
        print(f"\nAnalyzing: {query}")
        result = await nlu.analyze(query)
        
        print(f"Intent: {result.intent.category} (confidence: {result.intent.confidence:.2f})")
        print(f"Entities: {[(e.text, e.label) for e in result.entities]}")
        print(f"Sentiment: {result.sentiment.polarity} (score: {result.sentiment.score:.2f})")
        print(f"Complexity: {result.complexity}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        # Route to personas
        persona_routes = await router.route_query(result)
        print(f"Recommended personas: {persona_routes}")

if __name__ == "__main__":
    asyncio.run(main())

