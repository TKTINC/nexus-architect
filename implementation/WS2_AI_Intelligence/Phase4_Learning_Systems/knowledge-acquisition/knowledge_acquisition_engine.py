"""
Nexus Architect Knowledge Acquisition Engine

This module implements automated knowledge acquisition from conversations,
pattern recognition in user queries, and best practice identification.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import re

import redis.asyncio as redis
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import spacy
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict, Counter
import pandas as pd

logger = logging.getLogger(__name__)

class KnowledgeType(str, Enum):
    """Types of knowledge"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    METACOGNITIVE = "metacognitive"
    EXPERIENTIAL = "experiential"

class KnowledgeSource(str, Enum):
    """Sources of knowledge"""
    CONVERSATION = "conversation"
    DOCUMENTATION = "documentation"
    CODE_ANALYSIS = "code_analysis"
    USER_BEHAVIOR = "user_behavior"
    EXPERT_INPUT = "expert_input"
    SYSTEM_LOGS = "system_logs"

class PatternType(str, Enum):
    """Types of patterns"""
    QUERY_PATTERN = "query_pattern"
    RESPONSE_PATTERN = "response_pattern"
    INTERACTION_PATTERN = "interaction_pattern"
    TEMPORAL_PATTERN = "temporal_pattern"
    BEHAVIORAL_PATTERN = "behavioral_pattern"

class KnowledgeConfidence(str, Enum):
    """Confidence levels for knowledge"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

@dataclass
class KnowledgeItem:
    """Individual knowledge item"""
    knowledge_id: str
    knowledge_type: KnowledgeType
    source: KnowledgeSource
    content: str
    context: str
    confidence: KnowledgeConfidence
    evidence_count: int
    related_concepts: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    validated: bool = False

@dataclass
class Pattern:
    """Identified pattern"""
    pattern_id: str
    pattern_type: PatternType
    description: str
    frequency: int
    confidence_score: float
    examples: List[str]
    related_patterns: List[str]
    metadata: Dict[str, Any]
    discovered_at: datetime

@dataclass
class BestPractice:
    """Best practice identified from interactions"""
    practice_id: str
    title: str
    description: str
    domain: str
    success_rate: float
    evidence_examples: List[str]
    related_knowledge: List[str]
    applicability_conditions: List[str]
    metadata: Dict[str, Any]
    identified_at: datetime

@dataclass
class ConversationAnalysis:
    """Analysis of a conversation"""
    conversation_id: str
    user_intent: str
    topics: List[str]
    entities: List[Dict[str, Any]]
    knowledge_gaps: List[str]
    successful_responses: List[str]
    improvement_opportunities: List[str]
    patterns_identified: List[str]
    metadata: Dict[str, Any]
    analyzed_at: datetime

# Database models
Base = declarative_base()

class KnowledgeItemDB(Base):
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True)
    knowledge_id = Column(String(255), unique=True, nullable=False)
    knowledge_type = Column(String(50), nullable=False)
    source = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    context = Column(Text, nullable=False)
    confidence = Column(String(20), nullable=False)
    evidence_count = Column(Integer, default=1)
    related_concepts = Column(Text, nullable=False)
    metadata = Column(Text)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    validated = Column(Boolean, default=False)

class PatternDB(Base):
    __tablename__ = "patterns"
    
    id = Column(Integer, primary_key=True)
    pattern_id = Column(String(255), unique=True, nullable=False)
    pattern_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    frequency = Column(Integer, nullable=False)
    confidence_score = Column(Float, nullable=False)
    examples = Column(Text, nullable=False)
    related_patterns = Column(Text, nullable=False)
    metadata = Column(Text)
    discovered_at = Column(DateTime, nullable=False)

class BestPracticeDB(Base):
    __tablename__ = "best_practices"
    
    id = Column(Integer, primary_key=True)
    practice_id = Column(String(255), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    domain = Column(String(100), nullable=False)
    success_rate = Column(Float, nullable=False)
    evidence_examples = Column(Text, nullable=False)
    related_knowledge = Column(Text, nullable=False)
    applicability_conditions = Column(Text, nullable=False)
    metadata = Column(Text)
    identified_at = Column(DateTime, nullable=False)

class ConversationAnalysisDB(Base):
    __tablename__ = "conversation_analyses"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(255), unique=True, nullable=False)
    user_intent = Column(String(500), nullable=False)
    topics = Column(Text, nullable=False)
    entities = Column(Text, nullable=False)
    knowledge_gaps = Column(Text, nullable=False)
    successful_responses = Column(Text, nullable=False)
    improvement_opportunities = Column(Text, nullable=False)
    patterns_identified = Column(Text, nullable=False)
    metadata = Column(Text)
    analyzed_at = Column(DateTime, nullable=False)

class NLPProcessor:
    """Advanced NLP processing for knowledge extraction"""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic processing")
            self.nlp = None
            
        # Initialize transformers
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Initialize pipelines
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Knowledge extraction patterns
        self.fact_patterns = [
            r"(.+) is (.+)",
            r"(.+) means (.+)",
            r"(.+) refers to (.+)",
            r"(.+) can be defined as (.+)",
            r"(.+) represents (.+)"
        ]
        
        self.procedure_patterns = [
            r"to (.+), you (.+)",
            r"in order to (.+), (.+)",
            r"the steps to (.+) are (.+)",
            r"first (.+), then (.+)",
            r"(.+) by (.+)"
        ]
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0  # spaCy doesn't provide confidence scores
                })
        else:
            # Basic entity extraction using regex
            # This is a simplified fallback
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            
            for match in re.finditer(email_pattern, text):
                entities.append({
                    "text": match.group(),
                    "label": "EMAIL",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
                
            for match in re.finditer(url_pattern, text):
                entities.append({
                    "text": match.group(),
                    "label": "URL",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
                
        return entities
        
    def extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract factual knowledge from text"""
        facts = []
        
        for pattern in self.fact_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    facts.append({
                        "subject": groups[0].strip(),
                        "predicate": "is/means/refers_to",
                        "object": groups[1].strip(),
                        "confidence": 0.7,
                        "source_text": match.group()
                    })
                    
        return facts
        
    def extract_procedures(self, text: str) -> List[Dict[str, Any]]:
        """Extract procedural knowledge from text"""
        procedures = []
        
        for pattern in self.procedure_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    procedures.append({
                        "goal": groups[0].strip(),
                        "method": groups[1].strip(),
                        "confidence": 0.6,
                        "source_text": match.group()
                    })
                    
        return procedures
        
    def classify_intent(self, text: str, candidate_labels: List[str]) -> Dict[str, float]:
        """Classify user intent"""
        try:
            result = self.classifier(text, candidate_labels)
            return dict(zip(result['labels'], result['scores']))
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {label: 1.0/len(candidate_labels) for label in candidate_labels}
            
    def extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """Extract main topics from text"""
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract noun phrases and important terms
            topics = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to short phrases
                    topics.append(chunk.text.lower())
                    
            # Extract important single words
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN'] and 
                    not token.is_stop and 
                    len(token.text) > 3):
                    topics.append(token.text.lower())
                    
            # Count frequency and return top topics
            topic_counts = Counter(topics)
            return [topic for topic, count in topic_counts.most_common(max_topics)]
        else:
            # Simple word-based topic extraction
            words = text.lower().split()
            word_counts = Counter(word for word in words if len(word) > 3)
            return [word for word, count in word_counts.most_common(max_topics)]
            
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text content"""
        try:
            if len(text) > 1024:  # Model input limit
                text = text[:1024]
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            # Fallback to first few sentences
            sentences = text.split('.')
            return '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else text

class PatternRecognizer:
    """Pattern recognition in conversations and interactions"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.query_patterns = defaultdict(list)
        self.response_patterns = defaultdict(list)
        self.interaction_sequences = []
        
    def analyze_query_patterns(self, queries: List[str]) -> List[Pattern]:
        """Analyze patterns in user queries"""
        if len(queries) < 5:
            return []
            
        # Vectorize queries
        try:
            tfidf_matrix = self.vectorizer.fit_transform(queries)
            
            # Cluster similar queries
            clustering = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
            clusters = clustering.fit_predict(tfidf_matrix)
            
            patterns = []
            feature_names = self.vectorizer.get_feature_names_out()
            
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Noise cluster
                    continue
                    
                cluster_queries = [queries[i] for i, c in enumerate(clusters) if c == cluster_id]
                
                if len(cluster_queries) >= 3:
                    # Extract representative terms
                    cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                    cluster_tfidf = tfidf_matrix[cluster_indices]
                    mean_tfidf = np.mean(cluster_tfidf.toarray(), axis=0)
                    
                    top_indices = mean_tfidf.argsort()[-5:][::-1]
                    top_terms = [feature_names[idx] for idx in top_indices]
                    
                    pattern = Pattern(
                        pattern_id=f"query_pattern_{cluster_id}_{int(time.time())}",
                        pattern_type=PatternType.QUERY_PATTERN,
                        description=f"Query pattern: {' '.join(top_terms)}",
                        frequency=len(cluster_queries),
                        confidence_score=len(cluster_queries) / len(queries),
                        examples=cluster_queries[:3],
                        related_patterns=[],
                        metadata={"top_terms": top_terms, "cluster_size": len(cluster_queries)},
                        discovered_at=datetime.utcnow()
                    )
                    patterns.append(pattern)
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Query pattern analysis error: {e}")
            return []
            
    def analyze_interaction_patterns(self, interactions: List[Dict[str, Any]]) -> List[Pattern]:
        """Analyze patterns in user interactions"""
        patterns = []
        
        # Temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(interactions)
        patterns.extend(temporal_patterns)
        
        # Behavioral patterns
        behavioral_patterns = self._analyze_behavioral_patterns(interactions)
        patterns.extend(behavioral_patterns)
        
        return patterns
        
    def _analyze_temporal_patterns(self, interactions: List[Dict[str, Any]]) -> List[Pattern]:
        """Analyze temporal patterns in interactions"""
        patterns = []
        
        # Group interactions by hour of day
        hour_counts = defaultdict(int)
        for interaction in interactions:
            timestamp = interaction.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                hour_counts[timestamp.hour] += 1
                
        # Find peak hours
        if hour_counts:
            max_count = max(hour_counts.values())
            peak_hours = [hour for hour, count in hour_counts.items() if count > max_count * 0.8]
            
            if peak_hours:
                pattern = Pattern(
                    pattern_id=f"temporal_peak_{int(time.time())}",
                    pattern_type=PatternType.TEMPORAL_PATTERN,
                    description=f"Peak usage hours: {peak_hours}",
                    frequency=sum(hour_counts[hour] for hour in peak_hours),
                    confidence_score=0.8,
                    examples=[f"Hour {hour}: {hour_counts[hour]} interactions" for hour in peak_hours[:3]],
                    related_patterns=[],
                    metadata={"peak_hours": peak_hours, "hour_distribution": dict(hour_counts)},
                    discovered_at=datetime.utcnow()
                )
                patterns.append(pattern)
                
        return patterns
        
    def _analyze_behavioral_patterns(self, interactions: List[Dict[str, Any]]) -> List[Pattern]:
        """Analyze behavioral patterns in interactions"""
        patterns = []
        
        # Session length patterns
        session_lengths = []
        for interaction in interactions:
            session_length = interaction.get('session_length', 0)
            if session_length > 0:
                session_lengths.append(session_length)
                
        if session_lengths:
            avg_length = np.mean(session_lengths)
            std_length = np.std(session_lengths)
            
            # Identify long sessions
            long_sessions = [length for length in session_lengths if length > avg_length + std_length]
            
            if long_sessions:
                pattern = Pattern(
                    pattern_id=f"behavioral_long_session_{int(time.time())}",
                    pattern_type=PatternType.BEHAVIORAL_PATTERN,
                    description=f"Long session pattern: avg {avg_length:.1f}s, {len(long_sessions)} long sessions",
                    frequency=len(long_sessions),
                    confidence_score=len(long_sessions) / len(session_lengths),
                    examples=[f"Session length: {length:.1f}s" for length in long_sessions[:3]],
                    related_patterns=[],
                    metadata={
                        "avg_session_length": avg_length,
                        "long_session_threshold": avg_length + std_length,
                        "long_session_count": len(long_sessions)
                    },
                    discovered_at=datetime.utcnow()
                )
                patterns.append(pattern)
                
        return patterns

class KnowledgeAcquisitionEngine:
    """Main knowledge acquisition engine"""
    
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
        
        # Initialize processors
        self.nlp_processor = NLPProcessor()
        self.pattern_recognizer = PatternRecognizer()
        
        # Knowledge graph for relationships
        self.knowledge_graph = nx.DiGraph()
        
        # Configuration
        self.config = {
            "min_evidence_count": 3,
            "confidence_threshold": 0.7,
            "pattern_frequency_threshold": 5,
            "best_practice_success_threshold": 0.8,
            "analysis_batch_size": 100,
            "processing_interval": 600  # 10 minutes
        }
        
    async def initialize(self):
        """Initialize the knowledge acquisition engine"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        
        # Load existing knowledge into graph
        await self._load_knowledge_graph()
        
        # Start background processing
        asyncio.create_task(self._background_processor())
        
        logger.info("Knowledge acquisition engine initialized")
        
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def _load_knowledge_graph(self):
        """Load existing knowledge into graph structure"""
        session = self.SessionLocal()
        try:
            knowledge_items = session.query(KnowledgeItemDB).filter(
                KnowledgeItemDB.validated == True
            ).all()
            
            for item in knowledge_items:
                self.knowledge_graph.add_node(
                    item.knowledge_id,
                    content=item.content,
                    knowledge_type=item.knowledge_type,
                    confidence=item.confidence
                )
                
                # Add relationships
                related_concepts = json.loads(item.related_concepts)
                for concept in related_concepts:
                    if concept in self.knowledge_graph:
                        self.knowledge_graph.add_edge(item.knowledge_id, concept, relation="related_to")
                        
            logger.info(f"Loaded {len(self.knowledge_graph.nodes)} knowledge items into graph")
            
        finally:
            session.close()
            
    async def analyze_conversation(self, 
                                 conversation_id: str,
                                 messages: List[Dict[str, Any]],
                                 metadata: Dict[str, Any] = None) -> ConversationAnalysis:
        """Analyze a conversation for knowledge extraction"""
        
        # Combine all messages into text
        full_text = " ".join([msg.get('content', '') for msg in messages])
        user_messages = [msg.get('content', '') for msg in messages if msg.get('role') == 'user']
        ai_messages = [msg.get('content', '') for msg in messages if msg.get('role') == 'assistant']
        
        # Extract user intent
        intent_candidates = [
            "information_seeking", "problem_solving", "task_completion",
            "learning", "troubleshooting", "planning", "analysis"
        ]
        intent_scores = self.nlp_processor.classify_intent(full_text, intent_candidates)
        user_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Extract topics
        topics = self.nlp_processor.extract_topics(full_text)
        
        # Extract entities
        entities = self.nlp_processor.extract_entities(full_text)
        
        # Identify knowledge gaps
        knowledge_gaps = await self._identify_knowledge_gaps(user_messages, ai_messages)
        
        # Identify successful responses
        successful_responses = await self._identify_successful_responses(messages)
        
        # Identify improvement opportunities
        improvement_opportunities = await self._identify_improvement_opportunities(messages)
        
        # Identify patterns
        patterns_identified = await self._identify_conversation_patterns(messages)
        
        analysis = ConversationAnalysis(
            conversation_id=conversation_id,
            user_intent=user_intent,
            topics=topics,
            entities=entities,
            knowledge_gaps=knowledge_gaps,
            successful_responses=successful_responses,
            improvement_opportunities=improvement_opportunities,
            patterns_identified=patterns_identified,
            metadata=metadata or {},
            analyzed_at=datetime.utcnow()
        )
        
        # Store analysis
        await self._store_conversation_analysis(analysis)
        
        # Extract knowledge from conversation
        await self._extract_knowledge_from_conversation(analysis, messages)
        
        return analysis
        
    async def _identify_knowledge_gaps(self, user_messages: List[str], ai_messages: List[str]) -> List[str]:
        """Identify knowledge gaps from conversation"""
        gaps = []
        
        # Look for uncertainty indicators in AI responses
        uncertainty_indicators = [
            "i'm not sure", "i don't know", "unclear", "uncertain",
            "might be", "could be", "possibly", "perhaps"
        ]
        
        for message in ai_messages:
            message_lower = message.lower()
            for indicator in uncertainty_indicators:
                if indicator in message_lower:
                    # Extract the topic of uncertainty
                    sentences = message.split('.')
                    for sentence in sentences:
                        if indicator in sentence.lower():
                            topics = self.nlp_processor.extract_topics(sentence, max_topics=2)
                            if topics:
                                gaps.extend(topics)
                                
        # Look for unanswered questions
        for message in user_messages:
            if '?' in message:
                topics = self.nlp_processor.extract_topics(message, max_topics=2)
                gaps.extend(topics)
                
        return list(set(gaps))  # Remove duplicates
        
    async def _identify_successful_responses(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Identify successful AI responses"""
        successful = []
        
        for i, message in enumerate(messages):
            if message.get('role') == 'assistant':
                # Look for positive feedback in next user message
                if i + 1 < len(messages):
                    next_message = messages[i + 1]
                    if next_message.get('role') == 'user':
                        content = next_message.get('content', '').lower()
                        positive_indicators = [
                            "thank", "great", "perfect", "exactly", "helpful",
                            "good", "excellent", "right", "correct"
                        ]
                        if any(indicator in content for indicator in positive_indicators):
                            successful.append(message.get('content', ''))
                            
        return successful
        
    async def _identify_improvement_opportunities(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Identify improvement opportunities"""
        opportunities = []
        
        for i, message in enumerate(messages):
            if message.get('role') == 'assistant':
                # Look for negative feedback or follow-up questions
                if i + 1 < len(messages):
                    next_message = messages[i + 1]
                    if next_message.get('role') == 'user':
                        content = next_message.get('content', '').lower()
                        
                        # Negative feedback
                        negative_indicators = [
                            "wrong", "incorrect", "not right", "doesn't work",
                            "confused", "unclear", "not helpful"
                        ]
                        if any(indicator in content for indicator in negative_indicators):
                            opportunities.append(f"Improve accuracy: {message.get('content', '')[:100]}...")
                            
                        # Follow-up questions indicating incomplete answer
                        if '?' in content and any(word in content for word in ['but', 'however', 'what about']):
                            opportunities.append(f"Provide more complete answer: {message.get('content', '')[:100]}...")
                            
        return opportunities
        
    async def _identify_conversation_patterns(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns in conversation"""
        patterns = []
        
        # Conversation length pattern
        if len(messages) > 10:
            patterns.append("long_conversation")
        elif len(messages) < 3:
            patterns.append("short_conversation")
            
        # Question-answer pattern
        question_count = sum(1 for msg in messages if '?' in msg.get('content', ''))
        if question_count > len(messages) * 0.5:
            patterns.append("question_heavy")
            
        # Code-related pattern
        code_indicators = ['```', 'function', 'class', 'import', 'def ', 'var ', 'const ']
        code_messages = sum(1 for msg in messages 
                          if any(indicator in msg.get('content', '') for indicator in code_indicators))
        if code_messages > 0:
            patterns.append("code_discussion")
            
        return patterns
        
    async def _extract_knowledge_from_conversation(self, 
                                                 analysis: ConversationAnalysis, 
                                                 messages: List[Dict[str, Any]]):
        """Extract knowledge items from conversation"""
        
        # Extract from AI responses (validated knowledge)
        for message in messages:
            if message.get('role') == 'assistant':
                content = message.get('content', '')
                
                # Extract facts
                facts = self.nlp_processor.extract_facts(content)
                for fact in facts:
                    await self._create_knowledge_item(
                        knowledge_type=KnowledgeType.FACTUAL,
                        source=KnowledgeSource.CONVERSATION,
                        content=f"{fact['subject']} {fact['predicate']} {fact['object']}",
                        context=analysis.conversation_id,
                        confidence=KnowledgeConfidence.MEDIUM,
                        related_concepts=analysis.topics,
                        metadata={
                            "source_text": fact['source_text'],
                            "conversation_id": analysis.conversation_id,
                            "user_intent": analysis.user_intent
                        }
                    )
                    
                # Extract procedures
                procedures = self.nlp_processor.extract_procedures(content)
                for procedure in procedures:
                    await self._create_knowledge_item(
                        knowledge_type=KnowledgeType.PROCEDURAL,
                        source=KnowledgeSource.CONVERSATION,
                        content=f"To {procedure['goal']}: {procedure['method']}",
                        context=analysis.conversation_id,
                        confidence=KnowledgeConfidence.MEDIUM,
                        related_concepts=analysis.topics,
                        metadata={
                            "source_text": procedure['source_text'],
                            "conversation_id": analysis.conversation_id,
                            "goal": procedure['goal'],
                            "method": procedure['method']
                        }
                    )
                    
    async def _create_knowledge_item(self,
                                   knowledge_type: KnowledgeType,
                                   source: KnowledgeSource,
                                   content: str,
                                   context: str,
                                   confidence: KnowledgeConfidence,
                                   related_concepts: List[str],
                                   metadata: Dict[str, Any]) -> str:
        """Create a new knowledge item"""
        
        knowledge_id = f"knowledge_{int(time.time())}_{hash(content) % 10000}"
        
        # Check for existing similar knowledge
        existing = await self._find_similar_knowledge(content)
        if existing:
            # Update evidence count
            await self._update_knowledge_evidence(existing, metadata)
            return existing
            
        knowledge_item = KnowledgeItem(
            knowledge_id=knowledge_id,
            knowledge_type=knowledge_type,
            source=source,
            content=content,
            context=context,
            confidence=confidence,
            evidence_count=1,
            related_concepts=related_concepts,
            metadata=metadata,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store in database
        session = self.SessionLocal()
        try:
            knowledge_db = KnowledgeItemDB(
                knowledge_id=knowledge_item.knowledge_id,
                knowledge_type=knowledge_item.knowledge_type.value,
                source=knowledge_item.source.value,
                content=knowledge_item.content,
                context=knowledge_item.context,
                confidence=knowledge_item.confidence.value,
                evidence_count=knowledge_item.evidence_count,
                related_concepts=json.dumps(knowledge_item.related_concepts),
                metadata=json.dumps(knowledge_item.metadata),
                created_at=knowledge_item.created_at,
                updated_at=knowledge_item.updated_at,
                validated=False
            )
            session.add(knowledge_db)
            session.commit()
            
            # Add to knowledge graph
            self.knowledge_graph.add_node(
                knowledge_id,
                content=content,
                knowledge_type=knowledge_type.value,
                confidence=confidence.value
            )
            
            # Add relationships
            for concept in related_concepts:
                if concept in self.knowledge_graph:
                    self.knowledge_graph.add_edge(knowledge_id, concept, relation="related_to")
                    
            logger.info(f"Created knowledge item: {knowledge_id}")
            return knowledge_id
            
        finally:
            session.close()
            
    async def _find_similar_knowledge(self, content: str, threshold: float = 0.8) -> Optional[str]:
        """Find similar existing knowledge"""
        session = self.SessionLocal()
        try:
            existing_items = session.query(KnowledgeItemDB).all()
            
            if not existing_items:
                return None
                
            # Simple similarity check (in practice, use embeddings)
            content_words = set(content.lower().split())
            
            for item in existing_items:
                item_words = set(item.content.lower().split())
                similarity = len(content_words & item_words) / len(content_words | item_words)
                
                if similarity > threshold:
                    return item.knowledge_id
                    
            return None
            
        finally:
            session.close()
            
    async def _update_knowledge_evidence(self, knowledge_id: str, new_metadata: Dict[str, Any]):
        """Update evidence count for existing knowledge"""
        session = self.SessionLocal()
        try:
            knowledge_db = session.query(KnowledgeItemDB).filter(
                KnowledgeItemDB.knowledge_id == knowledge_id
            ).first()
            
            if knowledge_db:
                knowledge_db.evidence_count += 1
                knowledge_db.updated_at = datetime.utcnow()
                
                # Merge metadata
                existing_metadata = json.loads(knowledge_db.metadata) if knowledge_db.metadata else {}
                existing_metadata.update(new_metadata)
                knowledge_db.metadata = json.dumps(existing_metadata)
                
                session.commit()
                
        finally:
            session.close()
            
    async def _store_conversation_analysis(self, analysis: ConversationAnalysis):
        """Store conversation analysis in database"""
        session = self.SessionLocal()
        try:
            analysis_db = ConversationAnalysisDB(
                conversation_id=analysis.conversation_id,
                user_intent=analysis.user_intent,
                topics=json.dumps(analysis.topics),
                entities=json.dumps(analysis.entities),
                knowledge_gaps=json.dumps(analysis.knowledge_gaps),
                successful_responses=json.dumps(analysis.successful_responses),
                improvement_opportunities=json.dumps(analysis.improvement_opportunities),
                patterns_identified=json.dumps(analysis.patterns_identified),
                metadata=json.dumps(analysis.metadata),
                analyzed_at=analysis.analyzed_at
            )
            session.add(analysis_db)
            session.commit()
        finally:
            session.close()
            
    async def _background_processor(self):
        """Background task for continuous knowledge acquisition"""
        while True:
            try:
                # Process pending conversations
                await self._process_pending_conversations()
                
                # Analyze patterns
                await self._analyze_patterns()
                
                # Identify best practices
                await self._identify_best_practices()
                
                # Validate knowledge
                await self._validate_knowledge()
                
                await asyncio.sleep(self.config["processing_interval"])
                
            except Exception as e:
                logger.error(f"Background processor error: {e}")
                await asyncio.sleep(60)
                
    async def _process_pending_conversations(self):
        """Process conversations queued for analysis"""
        # This would process conversations from a queue
        # For now, just log
        logger.info("Processing pending conversations")
        
    async def _analyze_patterns(self):
        """Analyze patterns in recent data"""
        # Get recent conversations
        session = self.SessionLocal()
        try:
            recent_analyses = session.query(ConversationAnalysisDB).filter(
                ConversationAnalysisDB.analyzed_at >= datetime.utcnow() - timedelta(hours=24)
            ).all()
            
            if len(recent_analyses) >= self.config["pattern_frequency_threshold"]:
                # Analyze query patterns
                user_intents = [analysis.user_intent for analysis in recent_analyses]
                intent_counts = Counter(user_intents)
                
                for intent, count in intent_counts.items():
                    if count >= self.config["pattern_frequency_threshold"]:
                        pattern = Pattern(
                            pattern_id=f"intent_pattern_{intent}_{int(time.time())}",
                            pattern_type=PatternType.BEHAVIORAL_PATTERN,
                            description=f"Frequent user intent: {intent}",
                            frequency=count,
                            confidence_score=count / len(recent_analyses),
                            examples=[intent],
                            related_patterns=[],
                            metadata={"intent": intent, "count": count},
                            discovered_at=datetime.utcnow()
                        )
                        await self._store_pattern(pattern)
                        
        finally:
            session.close()
            
    async def _identify_best_practices(self):
        """Identify best practices from successful interactions"""
        session = self.SessionLocal()
        try:
            # Get conversations with successful responses
            analyses = session.query(ConversationAnalysisDB).filter(
                ConversationAnalysisDB.analyzed_at >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            success_patterns = defaultdict(list)
            
            for analysis in analyses:
                successful_responses = json.loads(analysis.successful_responses)
                if successful_responses:
                    topics = json.loads(analysis.topics)
                    for topic in topics:
                        success_patterns[topic].extend(successful_responses)
                        
            # Identify best practices
            for topic, responses in success_patterns.items():
                if len(responses) >= 3:  # Minimum evidence
                    success_rate = len(responses) / len([a for a in analyses if topic in json.loads(a.topics)])
                    
                    if success_rate >= self.config["best_practice_success_threshold"]:
                        practice = BestPractice(
                            practice_id=f"practice_{topic}_{int(time.time())}",
                            title=f"Best practice for {topic}",
                            description=f"Successful approach for {topic} queries",
                            domain=topic,
                            success_rate=success_rate,
                            evidence_examples=responses[:3],
                            related_knowledge=[],
                            applicability_conditions=[f"User asking about {topic}"],
                            metadata={"topic": topic, "response_count": len(responses)},
                            identified_at=datetime.utcnow()
                        )
                        await self._store_best_practice(practice)
                        
        finally:
            session.close()
            
    async def _validate_knowledge(self):
        """Validate knowledge items based on evidence"""
        session = self.SessionLocal()
        try:
            unvalidated_items = session.query(KnowledgeItemDB).filter(
                KnowledgeItemDB.validated == False,
                KnowledgeItemDB.evidence_count >= self.config["min_evidence_count"]
            ).all()
            
            for item in unvalidated_items:
                # Simple validation based on evidence count and confidence
                if (item.evidence_count >= self.config["min_evidence_count"] and
                    item.confidence in [KnowledgeConfidence.HIGH.value, KnowledgeConfidence.MEDIUM.value]):
                    
                    item.validated = True
                    session.commit()
                    
                    logger.info(f"Validated knowledge item: {item.knowledge_id}")
                    
        finally:
            session.close()
            
    async def _store_pattern(self, pattern: Pattern):
        """Store identified pattern"""
        session = self.SessionLocal()
        try:
            pattern_db = PatternDB(
                pattern_id=pattern.pattern_id,
                pattern_type=pattern.pattern_type.value,
                description=pattern.description,
                frequency=pattern.frequency,
                confidence_score=pattern.confidence_score,
                examples=json.dumps(pattern.examples),
                related_patterns=json.dumps(pattern.related_patterns),
                metadata=json.dumps(pattern.metadata),
                discovered_at=pattern.discovered_at
            )
            session.add(pattern_db)
            session.commit()
        finally:
            session.close()
            
    async def _store_best_practice(self, practice: BestPractice):
        """Store identified best practice"""
        session = self.SessionLocal()
        try:
            practice_db = BestPracticeDB(
                practice_id=practice.practice_id,
                title=practice.title,
                description=practice.description,
                domain=practice.domain,
                success_rate=practice.success_rate,
                evidence_examples=json.dumps(practice.evidence_examples),
                related_knowledge=json.dumps(practice.related_knowledge),
                applicability_conditions=json.dumps(practice.applicability_conditions),
                metadata=json.dumps(practice.metadata),
                identified_at=practice.identified_at
            )
            session.add(practice_db)
            session.commit()
        finally:
            session.close()
            
    async def get_knowledge_items(self, 
                                knowledge_type: Optional[KnowledgeType] = None,
                                validated_only: bool = True,
                                limit: int = 100) -> List[KnowledgeItem]:
        """Get knowledge items"""
        session = self.SessionLocal()
        try:
            query = session.query(KnowledgeItemDB)
            
            if knowledge_type:
                query = query.filter(KnowledgeItemDB.knowledge_type == knowledge_type.value)
            if validated_only:
                query = query.filter(KnowledgeItemDB.validated == True)
                
            items_db = query.order_by(KnowledgeItemDB.evidence_count.desc()).limit(limit).all()
            
            items = []
            for item_db in items_db:
                item = KnowledgeItem(
                    knowledge_id=item_db.knowledge_id,
                    knowledge_type=KnowledgeType(item_db.knowledge_type),
                    source=KnowledgeSource(item_db.source),
                    content=item_db.content,
                    context=item_db.context,
                    confidence=KnowledgeConfidence(item_db.confidence),
                    evidence_count=item_db.evidence_count,
                    related_concepts=json.loads(item_db.related_concepts),
                    metadata=json.loads(item_db.metadata) if item_db.metadata else {},
                    created_at=item_db.created_at,
                    updated_at=item_db.updated_at,
                    validated=item_db.validated
                )
                items.append(item)
                
            return items
            
        finally:
            session.close()

# Example usage and testing
async def main():
    """Example usage of knowledge acquisition engine"""
    engine = KnowledgeAcquisitionEngine(
        database_url="sqlite:///knowledge.db",
        redis_url="redis://localhost:6379"
    )
    
    await engine.initialize()
    
    try:
        # Analyze a sample conversation
        messages = [
            {"role": "user", "content": "How do I implement OAuth 2.0 authentication?"},
            {"role": "assistant", "content": "OAuth 2.0 is an authorization framework that enables applications to obtain limited access to user accounts. To implement it, you need to register your application with the OAuth provider, configure redirect URIs, and implement the authorization code flow."},
            {"role": "user", "content": "That's helpful! Can you show me the specific steps?"},
            {"role": "assistant", "content": "Sure! First, register your app to get client credentials. Then redirect users to the authorization server, handle the callback with the authorization code, exchange the code for an access token, and use the token to access protected resources."},
            {"role": "user", "content": "Perfect, thank you!"}
        ]
        
        analysis = await engine.analyze_conversation(
            conversation_id="conv_oauth_example",
            messages=messages,
            metadata={"domain": "authentication", "complexity": "medium"}
        )
        
        print(f"Conversation analysis:")
        print(f"User intent: {analysis.user_intent}")
        print(f"Topics: {analysis.topics}")
        print(f"Knowledge gaps: {analysis.knowledge_gaps}")
        print(f"Successful responses: {len(analysis.successful_responses)}")
        print(f"Patterns: {analysis.patterns_identified}")
        
        # Wait for background processing
        await asyncio.sleep(5)
        
        # Get knowledge items
        knowledge_items = await engine.get_knowledge_items(limit=10)
        print(f"\nExtracted knowledge items: {len(knowledge_items)}")
        
        for item in knowledge_items:
            print(f"- {item.knowledge_type.value}: {item.content[:100]}...")
            print(f"  Evidence: {item.evidence_count}, Confidence: {item.confidence.value}")
            
    finally:
        await engine.close()

if __name__ == "__main__":
    asyncio.run(main())

