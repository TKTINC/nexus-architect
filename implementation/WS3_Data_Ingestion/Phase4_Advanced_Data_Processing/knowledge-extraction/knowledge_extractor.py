"""
Knowledge Extractor for Nexus Architect
Implements intelligent knowledge extraction including named entity recognition,
relationship extraction, event extraction, and concept identification.
"""

import os
import json
import logging
import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

# NLP and ML imports
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Database and caching
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import pandas as pd

# Web framework
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    logger.warning("Failed to download NLTK data")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found, using basic NLP")
    nlp = None

# Metrics
extraction_requests = Counter('knowledge_extraction_requests_total', 'Total extraction requests', ['extraction_type', 'status'])
extraction_duration = Histogram('knowledge_extraction_duration_seconds', 'Extraction duration', ['extraction_type'])
extracted_entities = Counter('extracted_entities_total', 'Total extracted entities', ['entity_type'])
extracted_relationships = Counter('extracted_relationships_total', 'Total extracted relationships', ['relationship_type'])

@dataclass
class Entity:
    """Extracted entity representation"""
    entity_id: str
    entity_type: str
    entity_text: str
    confidence: float
    start_pos: int
    end_pos: int
    properties: Dict[str, Any]
    source_document: str
    extraction_method: str

@dataclass
class Relationship:
    """Extracted relationship representation"""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    evidence_text: str
    properties: Dict[str, Any]
    source_document: str
    extraction_method: str

@dataclass
class Event:
    """Extracted event representation"""
    event_id: str
    event_type: str
    event_text: str
    timestamp: Optional[datetime]
    participants: List[str]
    location: Optional[str]
    confidence: float
    properties: Dict[str, Any]
    source_document: str

@dataclass
class Concept:
    """Extracted concept representation"""
    concept_id: str
    concept_text: str
    concept_type: str
    definition: Optional[str]
    related_terms: List[str]
    confidence: float
    properties: Dict[str, Any]
    source_document: str

class NamedEntityRecognizer:
    """Advanced named entity recognition"""
    
    def __init__(self):
        self.entity_patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b[A-Z]\. [A-Z][a-z]+\b',      # F. Last
                r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b'  # First M. Last
            ],
            'ORGANIZATION': [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                r'\b[A-Z][A-Z]+ [A-Z][a-z]+\b',  # IBM Corp
                r'\b[A-Z][a-z]+ & [A-Z][a-z]+\b'  # Smith & Jones
            ],
            'TECHNOLOGY': [
                r'\b(?:Python|Java|JavaScript|TypeScript|React|Angular|Vue|Docker|Kubernetes|AWS|Azure|GCP)\b',
                r'\b[A-Z][a-z]+(?:JS|SQL|API|SDK|CLI)\b',
                r'\b(?:REST|GraphQL|gRPC|HTTP|HTTPS|TCP|UDP)\b'
            ],
            'PROJECT': [
                r'\b[A-Z][a-z]+ Project\b',
                r'\bProject [A-Z][a-z]+\b',
                r'\b[A-Z][a-z]+-[A-Z][a-z]+\b'  # Multi-word projects
            ]
        }
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) if stopwords else set()
    
    def extract_entities_spacy(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using spaCy"""
        entities = []
        
        if not nlp:
            return entities
        
        try:
            doc = nlp(text)
            
            for ent in doc.ents:
                entity = Entity(
                    entity_id=str(uuid.uuid4()),
                    entity_type=ent.label_,
                    entity_text=ent.text,
                    confidence=0.8,  # spaCy doesn't provide confidence scores
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    properties={
                        'lemma': ent.lemma_,
                        'pos': ent.root.pos_,
                        'dependency': ent.root.dep_
                    },
                    source_document=document_id,
                    extraction_method='spacy'
                )
                entities.append(entity)
                extracted_entities.labels(entity_type=ent.label_).inc()
            
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    def extract_entities_nltk(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using NLTK"""
        entities = []
        
        try:
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join([token for token, pos in chunk])
                        start_pos = text.find(entity_text)
                        
                        if start_pos != -1:
                            entity = Entity(
                                entity_id=str(uuid.uuid4()),
                                entity_type=chunk.label(),
                                entity_text=entity_text,
                                confidence=0.7,
                                start_pos=start_pos,
                                end_pos=start_pos + len(entity_text),
                                properties={
                                    'pos_tags': [pos for token, pos in chunk]
                                },
                                source_document=document_id,
                                extraction_method='nltk'
                            )
                            entities.append(entity)
                            extracted_entities.labels(entity_type=chunk.label()).inc()
            
        except Exception as e:
            logger.error(f"NLTK entity extraction failed: {e}")
        
        return entities
    
    def extract_entities_patterns(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        
        try:
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text)
                    
                    for match in matches:
                        entity = Entity(
                            entity_id=str(uuid.uuid4()),
                            entity_type=entity_type,
                            entity_text=match.group(),
                            confidence=0.6,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            properties={
                                'pattern': pattern
                            },
                            source_document=document_id,
                            extraction_method='pattern'
                        )
                        entities.append(entity)
                        extracted_entities.labels(entity_type=entity_type).inc()
            
        except Exception as e:
            logger.error(f"Pattern entity extraction failed: {e}")
        
        return entities
    
    def extract_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using all available methods"""
        all_entities = []
        
        # Extract using different methods
        spacy_entities = self.extract_entities_spacy(text, document_id)
        nltk_entities = self.extract_entities_nltk(text, document_id)
        pattern_entities = self.extract_entities_patterns(text, document_id)
        
        all_entities.extend(spacy_entities)
        all_entities.extend(nltk_entities)
        all_entities.extend(pattern_entities)
        
        # Deduplicate entities
        deduplicated_entities = self._deduplicate_entities(all_entities)
        
        return deduplicated_entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on text and position"""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.entity_text.lower(), entity.start_pos, entity.end_pos)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated

class RelationshipExtractor:
    """Extract relationships between entities"""
    
    def __init__(self):
        self.relationship_patterns = {
            'WORKS_FOR': [
                r'(\w+(?:\s+\w+)*)\s+(?:works for|employed by|at)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+is\s+(?:a|an)\s+\w+\s+at\s+(\w+(?:\s+\w+)*)'
            ],
            'DEVELOPS': [
                r'(\w+(?:\s+\w+)*)\s+(?:develops|created|built|maintains)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+is\s+(?:developing|creating|building)\s+(\w+(?:\s+\w+)*)'
            ],
            'USES': [
                r'(\w+(?:\s+\w+)*)\s+(?:uses|utilizes|implements)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+is\s+(?:using|utilizing|implementing)\s+(\w+(?:\s+\w+)*)'
            ],
            'COLLABORATES_WITH': [
                r'(\w+(?:\s+\w+)*)\s+(?:collaborates with|works with|partners with)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)\s+(?:collaborate|work together)'
            ]
        }
    
    def extract_relationships_patterns(self, text: str, entities: List[Entity], document_id: str) -> List[Relationship]:
        """Extract relationships using pattern matching"""
        relationships = []
        
        try:
            for rel_type, patterns in self.relationship_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        source_text = match.group(1).strip()
                        target_text = match.group(2).strip()
                        
                        # Find matching entities
                        source_entity = self._find_entity_by_text(source_text, entities)
                        target_entity = self._find_entity_by_text(target_text, entities)
                        
                        if source_entity and target_entity:
                            relationship = Relationship(
                                relationship_id=str(uuid.uuid4()),
                                source_entity_id=source_entity.entity_id,
                                target_entity_id=target_entity.entity_id,
                                relationship_type=rel_type,
                                confidence=0.7,
                                evidence_text=match.group(),
                                properties={
                                    'pattern': pattern,
                                    'source_text': source_text,
                                    'target_text': target_text
                                },
                                source_document=document_id,
                                extraction_method='pattern'
                            )
                            relationships.append(relationship)
                            extracted_relationships.labels(relationship_type=rel_type).inc()
            
        except Exception as e:
            logger.error(f"Pattern relationship extraction failed: {e}")
        
        return relationships
    
    def extract_relationships_dependency(self, text: str, entities: List[Entity], document_id: str) -> List[Relationship]:
        """Extract relationships using dependency parsing"""
        relationships = []
        
        if not nlp:
            return relationships
        
        try:
            doc = nlp(text)
            
            for sent in doc.sents:
                # Find entities in this sentence
                sent_entities = [e for e in entities if e.start_pos >= sent.start_char and e.end_pos <= sent.end_char]
                
                if len(sent_entities) < 2:
                    continue
                
                # Analyze dependencies
                for token in sent:
                    if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                        # Find related entities
                        head_entities = [e for e in sent_entities if self._token_overlaps_entity(token.head, e)]
                        child_entities = [e for e in sent_entities if self._token_overlaps_entity(token, e)]
                        
                        for head_entity in head_entities:
                            for child_entity in child_entities:
                                if head_entity.entity_id != child_entity.entity_id:
                                    rel_type = self._infer_relationship_type(token.head, token)
                                    
                                    relationship = Relationship(
                                        relationship_id=str(uuid.uuid4()),
                                        source_entity_id=head_entity.entity_id,
                                        target_entity_id=child_entity.entity_id,
                                        relationship_type=rel_type,
                                        confidence=0.6,
                                        evidence_text=sent.text,
                                        properties={
                                            'dependency': token.dep_,
                                            'head_pos': token.head.pos_,
                                            'child_pos': token.pos_
                                        },
                                        source_document=document_id,
                                        extraction_method='dependency'
                                    )
                                    relationships.append(relationship)
                                    extracted_relationships.labels(relationship_type=rel_type).inc()
            
        except Exception as e:
            logger.error(f"Dependency relationship extraction failed: {e}")
        
        return relationships
    
    def _find_entity_by_text(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """Find entity by text match"""
        text_lower = text.lower()
        for entity in entities:
            if entity.entity_text.lower() == text_lower:
                return entity
        return None
    
    def _token_overlaps_entity(self, token, entity: Entity) -> bool:
        """Check if token overlaps with entity"""
        return (token.idx >= entity.start_pos and 
                token.idx + len(token.text) <= entity.end_pos)
    
    def _infer_relationship_type(self, head_token, child_token) -> str:
        """Infer relationship type from dependency structure"""
        if head_token.pos_ == 'VERB':
            if child_token.dep_ == 'nsubj':
                return 'PERFORMS'
            elif child_token.dep_ == 'dobj':
                return 'ACTS_ON'
        elif head_token.pos_ == 'NOUN':
            return 'RELATED_TO'
        
        return 'UNKNOWN'

class EventExtractor:
    """Extract events and temporal information"""
    
    def __init__(self):
        self.event_patterns = {
            'MEETING': [
                r'(?:meeting|call|conference|discussion)\s+(?:with|about|on)\s+(\w+(?:\s+\w+)*)',
                r'(?:scheduled|planned|held)\s+(?:a\s+)?(?:meeting|call|conference)\s+(?:with|about|on)\s+(\w+(?:\s+\w+)*)'
            ],
            'RELEASE': [
                r'(?:released|launched|deployed)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:was|is)\s+(?:released|launched|deployed)'
            ],
            'DECISION': [
                r'(?:decided|agreed|determined)\s+(?:to|that)\s+(\w+(?:\s+\w+)*)',
                r'(?:decision|agreement|determination)\s+(?:to|about|on)\s+(\w+(?:\s+\w+)*)'
            ],
            'ISSUE': [
                r'(?:issue|problem|bug|error)\s+(?:with|in|on)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:has|had)\s+(?:an\s+)?(?:issue|problem|bug|error)'
            ]
        }
        
        self.time_patterns = [
            r'\b(?:yesterday|today|tomorrow)\b',
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}:\d{2}(?:\s*(?:AM|PM))?\b'
        ]
    
    def extract_events(self, text: str, entities: List[Entity], document_id: str) -> List[Event]:
        """Extract events from text"""
        events = []
        
        try:
            for event_type, patterns in self.event_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        # Extract participants
                        participants = self._extract_participants(match.group(), entities)
                        
                        # Extract timestamp
                        timestamp = self._extract_timestamp(text, match.start(), match.end())
                        
                        # Extract location
                        location = self._extract_location(text, match.start(), match.end())
                        
                        event = Event(
                            event_id=str(uuid.uuid4()),
                            event_type=event_type,
                            event_text=match.group(),
                            timestamp=timestamp,
                            participants=participants,
                            location=location,
                            confidence=0.7,
                            properties={
                                'pattern': pattern,
                                'context': text[max(0, match.start()-50):match.end()+50]
                            },
                            source_document=document_id
                        )
                        events.append(event)
            
        except Exception as e:
            logger.error(f"Event extraction failed: {e}")
        
        return events
    
    def _extract_participants(self, event_text: str, entities: List[Entity]) -> List[str]:
        """Extract event participants from entities"""
        participants = []
        
        for entity in entities:
            if entity.entity_type in ['PERSON', 'ORGANIZATION'] and entity.entity_text.lower() in event_text.lower():
                participants.append(entity.entity_id)
        
        return participants
    
    def _extract_timestamp(self, text: str, start_pos: int, end_pos: int) -> Optional[datetime]:
        """Extract timestamp from surrounding context"""
        try:
            # Look for time patterns in surrounding text
            context = text[max(0, start_pos-100):end_pos+100]
            
            for pattern in self.time_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    time_text = match.group()
                    # Simple time parsing - in production, use dateutil.parser
                    if 'today' in time_text.lower():
                        return datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
                    elif 'yesterday' in time_text.lower():
                        return datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
                    elif 'tomorrow' in time_text.lower():
                        return datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)
            
            return None
            
        except Exception as e:
            logger.error(f"Timestamp extraction failed: {e}")
            return None
    
    def _extract_location(self, text: str, start_pos: int, end_pos: int) -> Optional[str]:
        """Extract location from surrounding context"""
        try:
            # Look for location indicators
            context = text[max(0, start_pos-50):end_pos+50]
            location_patterns = [
                r'(?:at|in|on)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:room|office|building)\s+([A-Z0-9]+)',
                r'(?:via|through)\s+(Zoom|Teams|Slack|Google Meet)'
            ]
            
            for pattern in location_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            return None
            
        except Exception as e:
            logger.error(f"Location extraction failed: {e}")
            return None

class ConceptExtractor:
    """Extract domain concepts and definitions"""
    
    def __init__(self):
        self.concept_indicators = [
            'is defined as',
            'refers to',
            'means',
            'is a type of',
            'is an example of',
            'also known as',
            'abbreviated as'
        ]
        
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.concept_clusters = None
    
    def extract_concepts(self, text: str, document_id: str) -> List[Concept]:
        """Extract concepts from text"""
        concepts = []
        
        try:
            # Extract definition-based concepts
            definition_concepts = self._extract_definition_concepts(text, document_id)
            concepts.extend(definition_concepts)
            
            # Extract technical concepts
            technical_concepts = self._extract_technical_concepts(text, document_id)
            concepts.extend(technical_concepts)
            
            # Extract domain-specific concepts
            domain_concepts = self._extract_domain_concepts(text, document_id)
            concepts.extend(domain_concepts)
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
        
        return concepts
    
    def _extract_definition_concepts(self, text: str, document_id: str) -> List[Concept]:
        """Extract concepts with explicit definitions"""
        concepts = []
        
        try:
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                for indicator in self.concept_indicators:
                    if indicator in sentence.lower():
                        # Extract concept and definition
                        parts = sentence.lower().split(indicator)
                        if len(parts) >= 2:
                            concept_text = parts[0].strip()
                            definition = parts[1].strip()
                            
                            # Clean up concept text
                            concept_text = re.sub(r'^.*\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b.*$', r'\1', concept_text)
                            
                            if len(concept_text) > 2 and len(definition) > 10:
                                concept = Concept(
                                    concept_id=str(uuid.uuid4()),
                                    concept_text=concept_text,
                                    concept_type='DEFINITION',
                                    definition=definition,
                                    related_terms=[],
                                    confidence=0.8,
                                    properties={
                                        'indicator': indicator,
                                        'sentence': sentence
                                    },
                                    source_document=document_id
                                )
                                concepts.append(concept)
            
        except Exception as e:
            logger.error(f"Definition concept extraction failed: {e}")
        
        return concepts
    
    def _extract_technical_concepts(self, text: str, document_id: str) -> List[Concept]:
        """Extract technical concepts"""
        concepts = []
        
        try:
            technical_patterns = [
                r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b',  # Acronyms
                r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
                r'\b\w+(?:\.\w+)+\b',  # Dotted notation
                r'\b\w+(?:-\w+)+\b'    # Hyphenated terms
            ]
            
            for pattern in technical_patterns:
                matches = re.finditer(pattern, text)
                
                for match in matches:
                    concept_text = match.group()
                    
                    # Filter out common words
                    if len(concept_text) > 2 and concept_text.lower() not in self._get_common_words():
                        concept = Concept(
                            concept_id=str(uuid.uuid4()),
                            concept_text=concept_text,
                            concept_type='TECHNICAL',
                            definition=None,
                            related_terms=[],
                            confidence=0.6,
                            properties={
                                'pattern': pattern
                            },
                            source_document=document_id
                        )
                        concepts.append(concept)
            
        except Exception as e:
            logger.error(f"Technical concept extraction failed: {e}")
        
        return concepts
    
    def _extract_domain_concepts(self, text: str, document_id: str) -> List[Concept]:
        """Extract domain-specific concepts using clustering"""
        concepts = []
        
        try:
            # Extract noun phrases as potential concepts
            if nlp:
                doc = nlp(text)
                noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 3]
            else:
                # Fallback to simple noun extraction
                words = word_tokenize(text)
                pos_tags = pos_tag(words)
                noun_phrases = [word for word, pos in pos_tags if pos.startswith('NN') and len(word) > 3]
            
            if len(noun_phrases) > 5:
                # Vectorize and cluster
                try:
                    vectors = self.vectorizer.fit_transform(noun_phrases)
                    n_clusters = min(10, len(noun_phrases) // 3)
                    
                    if n_clusters > 1:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(vectors)
                        
                        # Extract representative concepts from each cluster
                        for cluster_id in range(n_clusters):
                            cluster_phrases = [noun_phrases[i] for i in range(len(noun_phrases)) if clusters[i] == cluster_id]
                            
                            if cluster_phrases:
                                # Use most frequent phrase as concept
                                concept_text = max(cluster_phrases, key=cluster_phrases.count)
                                
                                concept = Concept(
                                    concept_id=str(uuid.uuid4()),
                                    concept_text=concept_text,
                                    concept_type='DOMAIN',
                                    definition=None,
                                    related_terms=cluster_phrases[:5],
                                    confidence=0.5,
                                    properties={
                                        'cluster_id': cluster_id,
                                        'cluster_size': len(cluster_phrases)
                                    },
                                    source_document=document_id
                                )
                                concepts.append(concept)
                
                except Exception as e:
                    logger.error(f"Clustering failed: {e}")
            
        except Exception as e:
            logger.error(f"Domain concept extraction failed: {e}")
        
        return concepts
    
    def _get_common_words(self) -> Set[str]:
        """Get set of common words to filter out"""
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        return common_words

class KnowledgeExtractor:
    """Main knowledge extraction orchestrator"""
    
    def __init__(self):
        self.ner = NamedEntityRecognizer()
        self.relationship_extractor = RelationshipExtractor()
        self.event_extractor = EventExtractor()
        self.concept_extractor = ConceptExtractor()
        
        # Database connections
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'nexus_architect'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 1)),
            decode_responses=True
        )
        
        # Processing executor
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None
    
    def extract_knowledge(self, text: str, document_id: str, extraction_types: List[str] = None) -> Dict[str, Any]:
        """Extract all types of knowledge from text"""
        if extraction_types is None:
            extraction_types = ['entities', 'relationships', 'events', 'concepts']
        
        results = {}
        
        try:
            with extraction_duration.labels(extraction_type='full').time():
                # Extract entities first (needed for other extractions)
                entities = []
                if 'entities' in extraction_types:
                    with extraction_duration.labels(extraction_type='entities').time():
                        entities = self.ner.extract_entities(text, document_id)
                        results['entities'] = [asdict(e) for e in entities]
                        extraction_requests.labels(extraction_type='entities', status='success').inc()
                
                # Extract relationships
                relationships = []
                if 'relationships' in extraction_types and entities:
                    with extraction_duration.labels(extraction_type='relationships').time():
                        pattern_rels = self.relationship_extractor.extract_relationships_patterns(text, entities, document_id)
                        dependency_rels = self.relationship_extractor.extract_relationships_dependency(text, entities, document_id)
                        relationships = pattern_rels + dependency_rels
                        results['relationships'] = [asdict(r) for r in relationships]
                        extraction_requests.labels(extraction_type='relationships', status='success').inc()
                
                # Extract events
                events = []
                if 'events' in extraction_types:
                    with extraction_duration.labels(extraction_type='events').time():
                        events = self.event_extractor.extract_events(text, entities, document_id)
                        results['events'] = [asdict(e) for e in events]
                        extraction_requests.labels(extraction_type='events', status='success').inc()
                
                # Extract concepts
                concepts = []
                if 'concepts' in extraction_types:
                    with extraction_duration.labels(extraction_type='concepts').time():
                        concepts = self.concept_extractor.extract_concepts(text, document_id)
                        results['concepts'] = [asdict(c) for c in concepts]
                        extraction_requests.labels(extraction_type='concepts', status='success').inc()
                
                # Store results in database
                self._store_extraction_results(document_id, entities, relationships, events, concepts)
                
                # Cache results
                cache_key = f"extraction:{document_id}"
                self.redis_client.setex(cache_key, 3600, json.dumps(results))
                
                return {
                    'document_id': document_id,
                    'extraction_types': extraction_types,
                    'results': results,
                    'summary': {
                        'entity_count': len(entities),
                        'relationship_count': len(relationships),
                        'event_count': len(events),
                        'concept_count': len(concepts)
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            extraction_requests.labels(extraction_type='full', status='error').inc()
            return {
                'document_id': document_id,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _store_extraction_results(self, document_id: str, entities: List[Entity], 
                                relationships: List[Relationship], events: List[Event], 
                                concepts: List[Concept]):
        """Store extraction results in database"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return
            
            with conn.cursor() as cursor:
                # Store entities
                for entity in entities:
                    cursor.execute("""
                        INSERT INTO extracted_entities 
                        (entity_id, entity_type, entity_text, confidence, start_pos, end_pos, 
                         properties, source_document, extraction_method, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (entity_id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        properties = EXCLUDED.properties
                    """, (
                        entity.entity_id, entity.entity_type, entity.entity_text,
                        entity.confidence, entity.start_pos, entity.end_pos,
                        json.dumps(entity.properties), entity.source_document,
                        entity.extraction_method, datetime.utcnow()
                    ))
                
                # Store relationships
                for relationship in relationships:
                    cursor.execute("""
                        INSERT INTO extracted_relationships 
                        (relationship_id, source_entity_id, target_entity_id, relationship_type,
                         confidence, evidence_text, properties, source_document, extraction_method, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (relationship_id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        properties = EXCLUDED.properties
                    """, (
                        relationship.relationship_id, relationship.source_entity_id,
                        relationship.target_entity_id, relationship.relationship_type,
                        relationship.confidence, relationship.evidence_text,
                        json.dumps(relationship.properties), relationship.source_document,
                        relationship.extraction_method, datetime.utcnow()
                    ))
                
                # Store events
                for event in events:
                    cursor.execute("""
                        INSERT INTO extracted_events 
                        (event_id, event_type, event_text, timestamp, participants, location,
                         confidence, properties, source_document, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (event_id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        properties = EXCLUDED.properties
                    """, (
                        event.event_id, event.event_type, event.event_text,
                        event.timestamp, json.dumps(event.participants), event.location,
                        event.confidence, json.dumps(event.properties),
                        event.source_document, datetime.utcnow()
                    ))
                
                # Store concepts
                for concept in concepts:
                    cursor.execute("""
                        INSERT INTO extracted_concepts 
                        (concept_id, concept_text, concept_type, definition, related_terms,
                         confidence, properties, source_document, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (concept_id) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        properties = EXCLUDED.properties
                    """, (
                        concept.concept_id, concept.concept_text, concept.concept_type,
                        concept.definition, json.dumps(concept.related_terms),
                        concept.confidence, json.dumps(concept.properties),
                        concept.source_document, datetime.utcnow()
                    ))
                
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store extraction results: {e}")
    
    def get_extraction_results(self, document_id: str) -> Dict[str, Any]:
        """Get cached extraction results"""
        try:
            cache_key = f"extraction:{document_id}"
            cached_results = self.redis_client.get(cache_key)
            
            if cached_results:
                return json.loads(cached_results)
            
            # Fallback to database
            return self._get_extraction_results_from_db(document_id)
            
        except Exception as e:
            logger.error(f"Failed to get extraction results: {e}")
            return {}
    
    def _get_extraction_results_from_db(self, document_id: str) -> Dict[str, Any]:
        """Get extraction results from database"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return {}
            
            results = {}
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get entities
                cursor.execute("""
                    SELECT * FROM extracted_entities 
                    WHERE source_document = %s
                """, (document_id,))
                results['entities'] = [dict(row) for row in cursor.fetchall()]
                
                # Get relationships
                cursor.execute("""
                    SELECT * FROM extracted_relationships 
                    WHERE source_document = %s
                """, (document_id,))
                results['relationships'] = [dict(row) for row in cursor.fetchall()]
                
                # Get events
                cursor.execute("""
                    SELECT * FROM extracted_events 
                    WHERE source_document = %s
                """, (document_id,))
                results['events'] = [dict(row) for row in cursor.fetchall()]
                
                # Get concepts
                cursor.execute("""
                    SELECT * FROM extracted_concepts 
                    WHERE source_document = %s
                """, (document_id,))
                results['concepts'] = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Failed to get extraction results from database: {e}")
            return {}

# Flask application
app = Flask(__name__)
CORS(app)

# Initialize extractor
extractor = KnowledgeExtractor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_conn = extractor.get_db_connection()
        db_status = "connected" if db_conn else "disconnected"
        if db_conn:
            db_conn.close()
        
        # Check Redis connection
        try:
            extractor.redis_client.ping()
            redis_status = "connected"
        except:
            redis_status = "disconnected"
        
        # Check NLP models
        nlp_status = "available" if nlp else "unavailable"
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": db_status,
                "redis": redis_status,
                "nlp": nlp_status
            }
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/v1/extraction/extract', methods=['POST'])
def extract_knowledge():
    """Extract knowledge from text"""
    try:
        data = request.get_json()
        
        text = data.get('text', '')
        document_id = data.get('document_id', str(uuid.uuid4()))
        extraction_types = data.get('extraction_types', ['entities', 'relationships', 'events', 'concepts'])
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        results = extractor.extract_knowledge(text, document_id, extraction_types)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Knowledge extraction failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/extraction/results/<document_id>', methods=['GET'])
def get_extraction_results(document_id):
    """Get extraction results for document"""
    try:
        results = extractor.get_extraction_results(document_id)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Failed to get extraction results: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/extraction/entities', methods=['GET'])
def list_entities():
    """List extracted entities with filtering"""
    try:
        entity_type = request.args.get('type')
        source_document = request.args.get('document')
        limit = int(request.args.get('limit', 100))
        
        conn = extractor.get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        query = "SELECT * FROM extracted_entities WHERE 1=1"
        params = []
        
        if entity_type:
            query += " AND entity_type = %s"
            params.append(entity_type)
        
        if source_document:
            query += " AND source_document = %s"
            params.append(source_document)
        
        query += " ORDER BY confidence DESC LIMIT %s"
        params.append(limit)
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            entities = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({"entities": entities})
        
    except Exception as e:
        logger.error(f"Failed to list entities: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8002))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Knowledge Extractor on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

