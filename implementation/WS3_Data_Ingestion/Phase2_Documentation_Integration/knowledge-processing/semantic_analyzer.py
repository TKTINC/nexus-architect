"""
Semantic Analyzer for Nexus Architect
Advanced semantic processing and knowledge graph integration for documentation
"""

import os
import re
import json
import logging
import time
import asyncio
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge

# NLP and ML libraries
try:
    import spacy
    from spacy import displacy
    from spacy.matcher import Matcher, PhraseMatcher
except ImportError:
    spacy = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
except ImportError:
    nltk = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    KMeans = None
    LatentDirichletAllocation = None
    cosine_similarity = None

try:
    import networkx as nx
except ImportError:
    nx = None

# Metrics
SEMANTIC_ANALYSIS_REQUESTS = PrometheusCounter('semantic_analysis_requests_total', 'Total semantic analysis requests', ['operation', 'status'])
SEMANTIC_ANALYSIS_LATENCY = Histogram('semantic_analysis_latency_seconds', 'Semantic analysis latency', ['operation'])
ENTITY_EXTRACTION_ACCURACY = Gauge('entity_extraction_accuracy', 'Entity extraction accuracy score')
TOPIC_MODELING_COHERENCE = Gauge('topic_modeling_coherence_score', 'Topic modeling coherence score')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Types of extracted entities"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    PROCESS = "process"
    PRODUCT = "product"
    DATE = "date"
    MONEY = "money"
    PERCENTAGE = "percentage"
    CUSTOM = "custom"

class RelationType(Enum):
    """Types of relationships between entities"""
    MENTIONS = "mentions"
    CONTAINS = "contains"
    RELATES_TO = "relates_to"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    USES = "uses"
    CREATED_BY = "created_by"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    REFERENCES = "references"

@dataclass
class Entity:
    """Extracted entity with metadata"""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    context: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    canonical_form: Optional[str] = None
    aliases: List[str] = field(default_factory=list)

@dataclass
class Relationship:
    """Relationship between entities"""
    source_entity: Entity
    target_entity: Entity
    relation_type: RelationType
    confidence: float
    context: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Topic:
    """Discovered topic with keywords and documents"""
    id: str
    name: str
    keywords: List[Tuple[str, float]]
    documents: List[str]
    coherence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticAnalysisResult:
    """Result of semantic analysis"""
    document_id: str
    entities: List[Entity]
    relationships: List[Relationship]
    topics: List[Topic]
    key_phrases: List[Tuple[str, float]]
    summary: str
    sentiment: Dict[str, float]
    language: str
    processing_time: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class EntityExtractor:
    """Advanced entity extraction with custom patterns"""
    
    def __init__(self):
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        self.custom_patterns = {}
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            if spacy:
                # Try to load the English model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.matcher = Matcher(self.nlp.vocab)
                    self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
                    self._setup_custom_patterns()
                    logger.info("spaCy NLP model loaded successfully")
                except OSError:
                    logger.warning("spaCy English model not found")
        except Exception as e:
            logger.error(f"Failed to initialize NLP: {e}")
    
    def _setup_custom_patterns(self):
        """Setup custom entity patterns"""
        if not self.matcher:
            return
        
        # Technology patterns
        tech_patterns = [
            [{"LOWER": {"IN": ["api", "rest", "graphql", "grpc"]}},
             {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["kubernetes", "docker", "aws", "azure", "gcp"]}},
             {"IS_ALPHA": True, "OP": "*"}],
            [{"LOWER": {"IN": ["python", "javascript", "java", "go", "rust"]}},
             {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["database", "sql", "nosql", "mongodb", "postgresql"]}},
             {"IS_ALPHA": True, "OP": "?"}]
        ]
        
        for i, pattern in enumerate(tech_patterns):
            self.matcher.add(f"TECHNOLOGY_{i}", [pattern])
        
        # Process patterns
        process_patterns = [
            [{"LOWER": {"IN": ["deployment", "testing", "integration", "monitoring"]}},
             {"IS_ALPHA": True, "OP": "*"}],
            [{"LOWER": {"IN": ["ci", "cd", "devops", "mlops"]}},
             {"IS_ALPHA": True, "OP": "?"}],
            [{"LOWER": {"IN": ["authentication", "authorization", "security"]}},
             {"IS_ALPHA": True, "OP": "*"}]
        ]
        
        for i, pattern in enumerate(process_patterns):
            self.matcher.add(f"PROCESS_{i}", [pattern])
        
        # Concept patterns
        concept_patterns = [
            [{"LOWER": {"IN": ["architecture", "design", "pattern", "framework"]}},
             {"IS_ALPHA": True, "OP": "*"}],
            [{"LOWER": {"IN": ["microservices", "serverless", "event-driven"]}},
             {"IS_ALPHA": True, "OP": "*"}],
            [{"LOWER": {"IN": ["scalability", "performance", "reliability"]}},
             {"IS_ALPHA": True, "OP": "*"}]
        ]
        
        for i, pattern in enumerate(concept_patterns):
            self.matcher.add(f"CONCEPT_{i}", [pattern])
    
    async def extract_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities from text"""
        start_time = time.time()
        entities = []
        
        try:
            if self.nlp:
                entities.extend(await self._extract_with_spacy(text))
            
            if nltk:
                entities.extend(await self._extract_with_nltk(text))
            
            # Deduplicate and merge entities
            entities = self._deduplicate_entities(entities)
            
            SEMANTIC_ANALYSIS_REQUESTS.labels(
                operation='entity_extraction',
                status='success'
            ).inc()
            
            # Calculate accuracy based on entity confidence
            if entities:
                avg_confidence = sum(e.confidence for e in entities) / len(entities)
                ENTITY_EXTRACTION_ACCURACY.set(avg_confidence)
        
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            SEMANTIC_ANALYSIS_REQUESTS.labels(
                operation='entity_extraction',
                status='error'
            ).inc()
        
        finally:
            latency = time.time() - start_time
            SEMANTIC_ANALYSIS_LATENCY.labels(operation='entity_extraction').observe(latency)
        
        return entities
    
    async def _extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy"""
        entities = []
        
        try:
            doc = self.nlp(text[:1000000])  # Limit to 1M characters
            
            # Extract named entities
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                
                entity = Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.9,  # spaCy doesn't provide confidence
                    context=self._get_context(text, ent.start_char, ent.end_char),
                    metadata={
                        "spacy_label": ent.label_,
                        "spacy_description": spacy.explain(ent.label_) if spacy else None
                    }
                )
                entities.append(entity)
            
            # Extract custom patterns
            if self.matcher:
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    span = doc[start:end]
                    match_label = self.nlp.vocab.strings[match_id]
                    
                    entity_type = self._map_custom_label(match_label)
                    
                    entity = Entity(
                        text=span.text,
                        entity_type=entity_type,
                        start_pos=span.start_char,
                        end_pos=span.end_char,
                        confidence=0.8,
                        context=self._get_context(text, span.start_char, span.end_char),
                        metadata={
                            "pattern_match": match_label,
                            "custom_extraction": True
                        }
                    )
                    entities.append(entity)
        
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    async def _extract_with_nltk(self, text: str) -> List[Entity]:
        """Extract entities using NLTK"""
        entities = []
        
        try:
            # Tokenize and tag
            sentences = sent_tokenize(text)
            
            for sentence in sentences[:100]:  # Limit to 100 sentences
                words = word_tokenize(sentence)
                pos_tags = pos_tag(words)
                
                # Named entity recognition
                tree = ne_chunk(pos_tags)
                
                current_entity = []
                current_label = None
                start_pos = 0
                
                for item in tree:
                    if hasattr(item, 'label'):
                        # Start of named entity
                        if current_entity and current_label:
                            # Finish previous entity
                            entity_text = ' '.join(current_entity)
                            entity_type = self._map_nltk_label(current_label)
                            
                            entity = Entity(
                                text=entity_text,
                                entity_type=entity_type,
                                start_pos=text.find(entity_text, start_pos),
                                end_pos=text.find(entity_text, start_pos) + len(entity_text),
                                confidence=0.7,
                                context=sentence,
                                metadata={
                                    "nltk_label": current_label,
                                    "pos_tags": pos_tags
                                }
                            )
                            entities.append(entity)
                        
                        # Start new entity
                        current_entity = [child[0] for child in item]
                        current_label = item.label()
                    else:
                        # Finish current entity if exists
                        if current_entity and current_label:
                            entity_text = ' '.join(current_entity)
                            entity_type = self._map_nltk_label(current_label)
                            
                            entity = Entity(
                                text=entity_text,
                                entity_type=entity_type,
                                start_pos=text.find(entity_text, start_pos),
                                end_pos=text.find(entity_text, start_pos) + len(entity_text),
                                confidence=0.7,
                                context=sentence,
                                metadata={
                                    "nltk_label": current_label,
                                    "pos_tags": pos_tags
                                }
                            )
                            entities.append(entity)
                            
                            current_entity = []
                            current_label = None
        
        except Exception as e:
            logger.error(f"NLTK entity extraction failed: {e}")
        
        return entities
    
    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.CONCEPT,
            "WORK_OF_ART": EntityType.CONCEPT,
            "LAW": EntityType.CONCEPT,
            "LANGUAGE": EntityType.TECHNOLOGY,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "PERCENT": EntityType.PERCENTAGE,
            "MONEY": EntityType.MONEY,
            "QUANTITY": EntityType.CONCEPT,
            "ORDINAL": EntityType.CONCEPT,
            "CARDINAL": EntityType.CONCEPT
        }
        
        return mapping.get(label, EntityType.CUSTOM)
    
    def _map_nltk_label(self, label: str) -> EntityType:
        """Map NLTK entity labels to our entity types"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOCATION": EntityType.LOCATION,
            "FACILITY": EntityType.LOCATION,
            "GSP": EntityType.LOCATION
        }
        
        return mapping.get(label, EntityType.CUSTOM)
    
    def _map_custom_label(self, label: str) -> EntityType:
        """Map custom pattern labels to entity types"""
        if "TECHNOLOGY" in label:
            return EntityType.TECHNOLOGY
        elif "PROCESS" in label:
            return EntityType.PROCESS
        elif "CONCEPT" in label:
            return EntityType.CONCEPT
        else:
            return EntityType.CUSTOM
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities and merge similar ones"""
        if not entities:
            return entities
        
        # Group by text and type
        entity_groups = defaultdict(list)
        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)
            entity_groups[key].append(entity)
        
        # Merge entities in each group
        merged_entities = []
        for group in entity_groups.values():
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # Merge multiple entities
                best_entity = max(group, key=lambda e: e.confidence)
                
                # Combine metadata
                combined_metadata = {}
                for entity in group:
                    combined_metadata.update(entity.metadata)
                
                best_entity.metadata = combined_metadata
                best_entity.confidence = sum(e.confidence for e in group) / len(group)
                
                merged_entities.append(best_entity)
        
        return merged_entities

class RelationshipExtractor:
    """Extract relationships between entities"""
    
    def __init__(self):
        self.nlp = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            if spacy:
                self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"Failed to initialize NLP for relationships: {e}")
    
    async def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities"""
        start_time = time.time()
        relationships = []
        
        try:
            if self.nlp:
                relationships.extend(await self._extract_with_dependency_parsing(text, entities))
            
            relationships.extend(await self._extract_with_patterns(text, entities))
            relationships.extend(await self._extract_with_proximity(text, entities))
            
            SEMANTIC_ANALYSIS_REQUESTS.labels(
                operation='relationship_extraction',
                status='success'
            ).inc()
        
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            SEMANTIC_ANALYSIS_REQUESTS.labels(
                operation='relationship_extraction',
                status='error'
            ).inc()
        
        finally:
            latency = time.time() - start_time
            SEMANTIC_ANALYSIS_LATENCY.labels(operation='relationship_extraction').observe(latency)
        
        return relationships
    
    async def _extract_with_dependency_parsing(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships using dependency parsing"""
        relationships = []
        
        try:
            doc = self.nlp(text[:1000000])
            
            # Create entity spans
            entity_spans = {}
            for entity in entities:
                for token in doc:
                    if token.idx >= entity.start_pos and token.idx < entity.end_pos:
                        entity_spans[token.i] = entity
            
            # Analyze dependencies
            for token in doc:
                if token.i in entity_spans:
                    source_entity = entity_spans[token.i]
                    
                    # Check for direct dependencies
                    for child in token.children:
                        if child.i in entity_spans:
                            target_entity = entity_spans[child.i]
                            relation_type = self._map_dependency_to_relation(token.dep_, child.dep_)
                            
                            if relation_type:
                                relationship = Relationship(
                                    source_entity=source_entity,
                                    target_entity=target_entity,
                                    relation_type=relation_type,
                                    confidence=0.8,
                                    context=token.sent.text,
                                    metadata={
                                        "dependency": child.dep_,
                                        "head_dependency": token.dep_
                                    }
                                )
                                relationships.append(relationship)
        
        except Exception as e:
            logger.error(f"Dependency parsing relationship extraction failed: {e}")
        
        return relationships
    
    async def _extract_with_patterns(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships using pattern matching"""
        relationships = []
        
        # Define relationship patterns
        patterns = [
            (r'\b(\w+)\s+uses?\s+(\w+)', RelationType.USES),
            (r'\b(\w+)\s+implements?\s+(\w+)', RelationType.IMPLEMENTS),
            (r'\b(\w+)\s+depends?\s+on\s+(\w+)', RelationType.DEPENDS_ON),
            (r'\b(\w+)\s+contains?\s+(\w+)', RelationType.CONTAINS),
            (r'\b(\w+)\s+created?\s+by\s+(\w+)', RelationType.CREATED_BY),
            (r'\b(\w+)\s+part\s+of\s+(\w+)', RelationType.PART_OF),
            (r'\b(\w+)\s+references?\s+(\w+)', RelationType.REFERENCES),
            (r'\b(\w+)\s+similar\s+to\s+(\w+)', RelationType.SIMILAR_TO)
        ]
        
        # Create entity lookup
        entity_lookup = {entity.text.lower(): entity for entity in entities}
        
        for pattern, relation_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                source_text = match.group(1).lower()
                target_text = match.group(2).lower()
                
                source_entity = entity_lookup.get(source_text)
                target_entity = entity_lookup.get(target_text)
                
                if source_entity and target_entity:
                    relationship = Relationship(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        relation_type=relation_type,
                        confidence=0.7,
                        context=self._get_sentence_context(text, match.start()),
                        metadata={
                            "pattern_match": pattern,
                            "match_text": match.group(0)
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def _extract_with_proximity(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships based on entity proximity"""
        relationships = []
        
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda e: e.start_pos)
        
        # Find entities within proximity threshold
        proximity_threshold = 100  # characters
        
        for i, entity1 in enumerate(sorted_entities):
            for j, entity2 in enumerate(sorted_entities[i+1:], i+1):
                distance = entity2.start_pos - entity1.end_pos
                
                if distance > proximity_threshold:
                    break
                
                # Create proximity-based relationship
                relationship = Relationship(
                    source_entity=entity1,
                    target_entity=entity2,
                    relation_type=RelationType.RELATES_TO,
                    confidence=max(0.3, 1.0 - (distance / proximity_threshold)),
                    context=text[entity1.start_pos:entity2.end_pos],
                    metadata={
                        "proximity_distance": distance,
                        "proximity_based": True
                    }
                )
                relationships.append(relationship)
        
        return relationships
    
    def _map_dependency_to_relation(self, head_dep: str, child_dep: str) -> Optional[RelationType]:
        """Map dependency relations to our relation types"""
        mapping = {
            ("nsubj", "dobj"): RelationType.USES,
            ("nsubj", "pobj"): RelationType.RELATES_TO,
            ("compound", "compound"): RelationType.PART_OF,
            ("amod", "noun"): RelationType.RELATES_TO
        }
        
        return mapping.get((head_dep, child_dep))
    
    def _get_sentence_context(self, text: str, position: int) -> str:
        """Get sentence containing the position"""
        # Find sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        current_pos = 0
        
        for sentence in sentences:
            if current_pos <= position <= current_pos + len(sentence):
                return sentence.strip()
            current_pos += len(sentence) + 1
        
        return ""

class TopicModeler:
    """Topic modeling and document clustering"""
    
    def __init__(self):
        self.vectorizer = None
        self.lda_model = None
        self.kmeans_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            if TfidfVectorizer:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8
                )
            
            if LatentDirichletAllocation:
                self.lda_model = LatentDirichletAllocation(
                    n_components=10,
                    random_state=42,
                    max_iter=100
                )
            
            if KMeans:
                self.kmeans_model = KMeans(
                    n_clusters=5,
                    random_state=42,
                    n_init=10
                )
        
        except Exception as e:
            logger.warning(f"Failed to initialize topic modeling: {e}")
    
    async def extract_topics(self, documents: List[str], document_ids: List[str]) -> List[Topic]:
        """Extract topics from documents"""
        start_time = time.time()
        topics = []
        
        try:
            if not self.vectorizer or not documents:
                return topics
            
            # Vectorize documents
            doc_vectors = self.vectorizer.fit_transform(documents)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # LDA topic modeling
            if self.lda_model:
                lda_topics = await self._extract_lda_topics(
                    doc_vectors, feature_names, documents, document_ids
                )
                topics.extend(lda_topics)
            
            # K-means clustering
            if self.kmeans_model:
                cluster_topics = await self._extract_cluster_topics(
                    doc_vectors, feature_names, documents, document_ids
                )
                topics.extend(cluster_topics)
            
            SEMANTIC_ANALYSIS_REQUESTS.labels(
                operation='topic_modeling',
                status='success'
            ).inc()
            
            # Calculate average coherence
            if topics:
                avg_coherence = sum(t.coherence_score for t in topics) / len(topics)
                TOPIC_MODELING_COHERENCE.set(avg_coherence)
        
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            SEMANTIC_ANALYSIS_REQUESTS.labels(
                operation='topic_modeling',
                status='error'
            ).inc()
        
        finally:
            latency = time.time() - start_time
            SEMANTIC_ANALYSIS_LATENCY.labels(operation='topic_modeling').observe(latency)
        
        return topics
    
    async def _extract_lda_topics(self, doc_vectors, feature_names, documents, document_ids) -> List[Topic]:
        """Extract topics using LDA"""
        topics = []
        
        try:
            # Fit LDA model
            lda_result = self.lda_model.fit_transform(doc_vectors)
            
            # Extract topics
            for topic_idx, topic in enumerate(self.lda_model.components_):
                # Get top keywords
                top_indices = topic.argsort()[-10:][::-1]
                keywords = [(feature_names[i], topic[i]) for i in top_indices]
                
                # Get documents for this topic
                topic_docs = []
                for doc_idx, doc_topics in enumerate(lda_result):
                    if doc_topics[topic_idx] > 0.3:  # Threshold for topic assignment
                        topic_docs.append(document_ids[doc_idx])
                
                # Calculate coherence (simplified)
                coherence_score = self._calculate_topic_coherence(keywords, documents)
                
                topic = Topic(
                    id=f"lda_topic_{topic_idx}",
                    name=f"Topic {topic_idx + 1}: {', '.join([kw[0] for kw in keywords[:3]])}",
                    keywords=keywords,
                    documents=topic_docs,
                    coherence_score=coherence_score,
                    metadata={
                        "method": "lda",
                        "topic_index": topic_idx,
                        "document_count": len(topic_docs)
                    }
                )
                topics.append(topic)
        
        except Exception as e:
            logger.error(f"LDA topic extraction failed: {e}")
        
        return topics
    
    async def _extract_cluster_topics(self, doc_vectors, feature_names, documents, document_ids) -> List[Topic]:
        """Extract topics using K-means clustering"""
        topics = []
        
        try:
            # Fit K-means model
            cluster_labels = self.kmeans_model.fit_predict(doc_vectors)
            
            # Extract topics from clusters
            for cluster_idx in range(self.kmeans_model.n_clusters):
                # Get documents in this cluster
                cluster_docs = [document_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_idx]
                
                if not cluster_docs:
                    continue
                
                # Get cluster center
                cluster_center = self.kmeans_model.cluster_centers_[cluster_idx]
                
                # Get top keywords
                top_indices = cluster_center.argsort()[-10:][::-1]
                keywords = [(feature_names[i], cluster_center[i]) for i in top_indices]
                
                # Calculate coherence
                cluster_documents = [documents[i] for i, label in enumerate(cluster_labels) if label == cluster_idx]
                coherence_score = self._calculate_topic_coherence(keywords, cluster_documents)
                
                topic = Topic(
                    id=f"cluster_topic_{cluster_idx}",
                    name=f"Cluster {cluster_idx + 1}: {', '.join([kw[0] for kw in keywords[:3]])}",
                    keywords=keywords,
                    documents=cluster_docs,
                    coherence_score=coherence_score,
                    metadata={
                        "method": "kmeans",
                        "cluster_index": cluster_idx,
                        "document_count": len(cluster_docs)
                    }
                )
                topics.append(topic)
        
        except Exception as e:
            logger.error(f"K-means topic extraction failed: {e}")
        
        return topics
    
    def _calculate_topic_coherence(self, keywords: List[Tuple[str, float]], documents: List[str]) -> float:
        """Calculate topic coherence score (simplified)"""
        try:
            if not keywords or not documents:
                return 0.0
            
            # Simple coherence based on keyword co-occurrence
            keyword_texts = [kw[0] for kw in keywords[:5]]
            coherence_scores = []
            
            for doc in documents[:10]:  # Sample documents
                doc_lower = doc.lower()
                keyword_count = sum(1 for kw in keyword_texts if kw.lower() in doc_lower)
                coherence_scores.append(keyword_count / len(keyword_texts))
            
            return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        except Exception:
            return 0.5  # Default coherence

class SemanticAnalyzer:
    """Main semantic analyzer coordinating all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.topic_modeler = TopicModeler()
        
        # Knowledge graph
        self.knowledge_graph = nx.DiGraph() if nx else None
    
    async def analyze_document(self, text: str, document_id: str, metadata: Dict[str, Any] = None) -> SemanticAnalysisResult:
        """Perform comprehensive semantic analysis"""
        start_time = time.time()
        
        try:
            # Extract entities
            entities = await self.entity_extractor.extract_entities(text, document_id)
            
            # Extract relationships
            relationships = await self.relationship_extractor.extract_relationships(text, entities)
            
            # Extract key phrases
            key_phrases = await self._extract_key_phrases(text)
            
            # Generate summary
            summary = await self._generate_summary(text)
            
            # Analyze sentiment
            sentiment = await self._analyze_sentiment(text)
            
            # Detect language
            language = await self._detect_language(text)
            
            # Calculate confidence
            confidence_score = self._calculate_overall_confidence(entities, relationships)
            
            processing_time = time.time() - start_time
            
            result = SemanticAnalysisResult(
                document_id=document_id,
                entities=entities,
                relationships=relationships,
                topics=[],  # Will be filled by batch processing
                key_phrases=key_phrases,
                summary=summary,
                sentiment=sentiment,
                language=language,
                processing_time=processing_time,
                confidence_score=confidence_score,
                metadata=metadata or {}
            )
            
            # Update knowledge graph
            if self.knowledge_graph:
                await self._update_knowledge_graph(entities, relationships)
            
            return result
        
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            
            return SemanticAnalysisResult(
                document_id=document_id,
                entities=[],
                relationships=[],
                topics=[],
                key_phrases=[],
                summary="",
                sentiment={},
                language="unknown",
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
    
    async def analyze_document_batch(self, documents: List[Tuple[str, str, Dict[str, Any]]]) -> List[SemanticAnalysisResult]:
        """Analyze multiple documents and extract cross-document topics"""
        results = []
        
        # Analyze individual documents
        for text, document_id, metadata in documents:
            result = await self.analyze_document(text, document_id, metadata)
            results.append(result)
        
        # Extract topics across all documents
        if len(documents) > 1:
            texts = [doc[0] for doc in documents]
            doc_ids = [doc[1] for doc in documents]
            topics = await self.topic_modeler.extract_topics(texts, doc_ids)
            
            # Assign topics to results
            for result in results:
                result.topics = [t for t in topics if result.document_id in t.documents]
        
        return results
    
    async def _extract_key_phrases(self, text: str) -> List[Tuple[str, float]]:
        """Extract key phrases from text"""
        try:
            if not TfidfVectorizer:
                return []
            
            # Simple key phrase extraction using TF-IDF
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                stop_words='english',
                max_features=20
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases
            phrase_scores = list(zip(feature_names, scores))
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            return phrase_scores[:10]
        
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {e}")
            return []
    
    async def _generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate text summary"""
        try:
            if not text.strip():
                return ""
            
            # Simple extractive summarization
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= max_sentences:
                return '. '.join(sentences)
            
            # Score sentences by length and position
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                # Prefer longer sentences and earlier positions
                score = len(sentence.split()) * (1.0 - i / len(sentences))
                sentence_scores.append((sentence, score))
            
            # Get top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
            
            return '. '.join(top_sentences)
        
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ""
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze text sentiment"""
        try:
            # Simple rule-based sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'success', 'effective']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'failure', 'problem', 'issue', 'error']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            
            if total_words == 0:
                return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
            positive_score = positive_count / total_words
            negative_score = negative_count / total_words
            neutral_score = 1.0 - positive_score - negative_score
            
            return {
                "positive": positive_score,
                "negative": negative_score,
                "neutral": max(0.0, neutral_score)
            }
        
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    
    async def _detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            # Simple heuristic language detection
            # In a real implementation, you'd use a proper language detection library
            
            # Check for common English words
            english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            text_lower = text.lower()
            english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')
            
            if english_count > 5:
                return "en"
            else:
                return "unknown"
        
        except Exception:
            return "unknown"
    
    def _calculate_overall_confidence(self, entities: List[Entity], relationships: List[Relationship]) -> float:
        """Calculate overall confidence score"""
        try:
            if not entities and not relationships:
                return 0.0
            
            entity_confidence = sum(e.confidence for e in entities) / len(entities) if entities else 0.0
            relationship_confidence = sum(r.confidence for r in relationships) / len(relationships) if relationships else 0.0
            
            # Weighted average
            if entities and relationships:
                return (entity_confidence * 0.6 + relationship_confidence * 0.4)
            elif entities:
                return entity_confidence * 0.8
            else:
                return relationship_confidence * 0.8
        
        except Exception:
            return 0.5
    
    async def _update_knowledge_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """Update knowledge graph with new entities and relationships"""
        try:
            if not self.knowledge_graph:
                return
            
            # Add entities as nodes
            for entity in entities:
                self.knowledge_graph.add_node(
                    entity.text,
                    entity_type=entity.entity_type.value,
                    confidence=entity.confidence,
                    canonical_form=entity.canonical_form,
                    aliases=entity.aliases
                )
            
            # Add relationships as edges
            for relationship in relationships:
                self.knowledge_graph.add_edge(
                    relationship.source_entity.text,
                    relationship.target_entity.text,
                    relation_type=relationship.relation_type.value,
                    confidence=relationship.confidence,
                    context=relationship.context
                )
        
        except Exception as e:
            logger.error(f"Knowledge graph update failed: {e}")
    
    async def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        if not self.knowledge_graph:
            return {}
        
        try:
            return {
                "nodes": self.knowledge_graph.number_of_nodes(),
                "edges": self.knowledge_graph.number_of_edges(),
                "density": nx.density(self.knowledge_graph),
                "connected_components": nx.number_weakly_connected_components(self.knowledge_graph),
                "average_clustering": nx.average_clustering(self.knowledge_graph.to_undirected())
            }
        except Exception as e:
            logger.error(f"Knowledge graph stats failed: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'entity_extractor': 'available' if self.entity_extractor.nlp else 'limited',
                'relationship_extractor': 'available' if self.relationship_extractor.nlp else 'limited',
                'topic_modeler': 'available' if self.topic_modeler.vectorizer else 'unavailable',
                'knowledge_graph': 'available' if self.knowledge_graph else 'unavailable'
            },
            'knowledge_graph_stats': await self.get_knowledge_graph_stats()
        }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        config = {}
        analyzer = SemanticAnalyzer(config)
        
        # Example usage
        sample_text = """
        The Nexus Architect system uses Kubernetes for container orchestration.
        It implements microservices architecture with FastAPI and PostgreSQL.
        The authentication service depends on Keycloak for identity management.
        Docker containers are deployed using CI/CD pipelines.
        """
        
        result = await analyzer.analyze_document(sample_text, "test_doc_1")
        
        print(f"Entities found: {len(result.entities)}")
        for entity in result.entities:
            print(f"  - {entity.text} ({entity.entity_type.value})")
        
        print(f"Relationships found: {len(result.relationships)}")
        for rel in result.relationships:
            print(f"  - {rel.source_entity.text} {rel.relation_type.value} {rel.target_entity.text}")
        
        print(f"Key phrases: {result.key_phrases}")
        print(f"Summary: {result.summary}")
        print(f"Sentiment: {result.sentiment}")
        
        # Health check
        health = await analyzer.health_check()
        print(f"Health: {health}")
    
    asyncio.run(main())

