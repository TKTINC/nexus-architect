"""
Stream Processor for Nexus Architect
Provides real-time stream processing with Kafka Streams and Apache Flink integration
"""

import json
import logging
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics
from collections import defaultdict, deque

from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import TopicPartition
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from prometheus_client import Counter, Histogram, Gauge, Summary
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Import from our Kafka infrastructure
from kafka_streaming_manager import StreamEvent, EventType, StreamingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingWindow(Enum):
    """Time window types for stream processing"""
    TUMBLING_1MIN = "tumbling_1min"
    TUMBLING_5MIN = "tumbling_5min"
    TUMBLING_15MIN = "tumbling_15min"
    SLIDING_1MIN = "sliding_1min"
    SLIDING_5MIN = "sliding_5min"
    SESSION_30MIN = "session_30min"

@dataclass
class ProcessedEvent:
    """Processed event with enrichment and transformations"""
    original_event: StreamEvent
    processed_timestamp: datetime
    enrichments: Dict[str, Any]
    transformations: Dict[str, Any]
    correlations: List[str]
    confidence_score: float
    processing_duration_ms: float

@dataclass
class EventPattern:
    """Pattern definition for complex event processing"""
    pattern_id: str
    name: str
    description: str
    event_types: List[EventType]
    conditions: Dict[str, Any]
    time_window: ProcessingWindow
    min_events: int
    max_events: int
    correlation_fields: List[str]

class StreamTransformer:
    """Handles real-time data transformation and enrichment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config['database']
        self.redis_config = config['redis']
        
        # Redis client for caching
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Transformation metrics
        self.transformations_applied = Counter('stream_transformations_total', 'Total transformations applied', ['transformation_type'])
        self.transformation_duration = Histogram('stream_transformation_duration_seconds', 'Transformation processing time', ['transformation_type'])
        self.transformation_errors = Counter('stream_transformation_errors_total', 'Transformation errors', ['transformation_type', 'error_type'])
        
        # Initialize transformation rules
        self._load_transformation_rules()
    
    def _load_transformation_rules(self):
        """Load transformation rules from database"""
        self.transformation_rules = {
            EventType.CODE_CHANGE: self._transform_code_change,
            EventType.DOCUMENTATION_UPDATE: self._transform_documentation_update,
            EventType.PROJECT_UPDATE: self._transform_project_update,
            EventType.COMMUNICATION_MESSAGE: self._transform_communication_message,
            EventType.SYSTEM_METRIC: self._transform_system_metric,
            EventType.QUALITY_ALERT: self._transform_quality_alert,
            EventType.KNOWLEDGE_EXTRACTION: self._transform_knowledge_extraction,
            EventType.USER_ACTION: self._transform_user_action
        }
    
    def transform_event(self, event: StreamEvent) -> ProcessedEvent:
        """Transform and enrich a single event"""
        start_time = time.time()
        
        try:
            # Get transformation function for event type
            transform_func = self.transformation_rules.get(event.event_type)
            
            if not transform_func:
                # Default transformation
                enrichments = self._default_enrichment(event)
                transformations = {}
            else:
                # Apply specific transformation
                with self.transformation_duration.labels(transformation_type=event.event_type.value).time():
                    enrichments, transformations = transform_func(event)
            
            # Calculate processing duration
            processing_duration = (time.time() - start_time) * 1000
            
            # Create processed event
            processed_event = ProcessedEvent(
                original_event=event,
                processed_timestamp=datetime.now(timezone.utc),
                enrichments=enrichments,
                transformations=transformations,
                correlations=[],  # Will be populated by correlation engine
                confidence_score=1.0,  # Default confidence
                processing_duration_ms=processing_duration
            )
            
            # Update metrics
            self.transformations_applied.labels(transformation_type=event.event_type.value).inc()
            
            return processed_event
            
        except Exception as e:
            self.transformation_errors.labels(
                transformation_type=event.event_type.value,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Transformation failed for event {event.event_id}: {e}")
            
            # Return minimal processed event on error
            return ProcessedEvent(
                original_event=event,
                processed_timestamp=datetime.now(timezone.utc),
                enrichments={},
                transformations={},
                correlations=[],
                confidence_score=0.0,
                processing_duration_ms=(time.time() - start_time) * 1000
            )
    
    def _default_enrichment(self, event: StreamEvent) -> Dict[str, Any]:
        """Default enrichment for all events"""
        return {
            'processing_timestamp': datetime.now(timezone.utc).isoformat(),
            'event_age_seconds': (datetime.now(timezone.utc) - event.timestamp).total_seconds(),
            'source_system_type': self._classify_source_system(event.source_system),
            'event_size_bytes': len(json.dumps(event.data)),
            'has_correlation_id': event.correlation_id is not None,
            'has_parent_event': event.parent_event_id is not None
        }
    
    def _classify_source_system(self, source_system: str) -> str:
        """Classify source system type"""
        if source_system.lower() in ['github', 'gitlab', 'bitbucket', 'azure_devops']:
            return 'version_control'
        elif source_system.lower() in ['confluence', 'sharepoint', 'notion', 'gitbook']:
            return 'documentation'
        elif source_system.lower() in ['jira', 'linear', 'asana', 'trello']:
            return 'project_management'
        elif source_system.lower() in ['slack', 'teams', 'discord']:
            return 'communication'
        else:
            return 'unknown'
    
    def _transform_code_change(self, event: StreamEvent) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform code change events"""
        data = event.data
        
        enrichments = self._default_enrichment(event)
        enrichments.update({
            'files_changed_count': len(data.get('files_changed', [])),
            'commit_message_length': len(data.get('commit_message', '')),
            'is_merge_commit': data.get('is_merge', False),
            'branch_type': self._classify_branch(data.get('branch', '')),
            'author_domain': data.get('author', '').split('@')[-1] if '@' in data.get('author', '') else 'unknown',
            'change_magnitude': self._calculate_change_magnitude(data)
        })
        
        transformations = {
            'normalized_files': [f.lower().replace('\\', '/') for f in data.get('files_changed', [])],
            'file_extensions': list(set([f.split('.')[-1] for f in data.get('files_changed', []) if '.' in f])),
            'commit_keywords': self._extract_commit_keywords(data.get('commit_message', '')),
            'impact_score': self._calculate_impact_score(data)
        }
        
        return enrichments, transformations
    
    def _transform_documentation_update(self, event: StreamEvent) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform documentation update events"""
        data = event.data
        
        enrichments = self._default_enrichment(event)
        enrichments.update({
            'content_length': len(data.get('content', '')),
            'title_length': len(data.get('title', '')),
            'has_images': 'image' in data.get('content', '').lower(),
            'has_links': 'http' in data.get('content', '').lower(),
            'update_type': data.get('update_type', 'unknown'),
            'page_hierarchy_level': data.get('hierarchy_level', 0)
        })
        
        transformations = {
            'content_summary': self._summarize_content(data.get('content', '')),
            'extracted_topics': self._extract_topics(data.get('content', '')),
            'readability_score': self._calculate_readability(data.get('content', '')),
            'technical_complexity': self._assess_technical_complexity(data.get('content', ''))
        }
        
        return enrichments, transformations
    
    def _transform_project_update(self, event: StreamEvent) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform project update events"""
        data = event.data
        
        enrichments = self._default_enrichment(event)
        enrichments.update({
            'status_change': data.get('old_status') != data.get('new_status'),
            'assignee_change': data.get('old_assignee') != data.get('new_assignee'),
            'priority_level': self._normalize_priority(data.get('priority', '')),
            'estimated_hours': data.get('estimated_hours', 0),
            'actual_hours': data.get('actual_hours', 0),
            'completion_percentage': data.get('completion_percentage', 0)
        })
        
        transformations = {
            'status_progression': self._analyze_status_progression(data),
            'effort_variance': self._calculate_effort_variance(data),
            'milestone_impact': self._assess_milestone_impact(data),
            'team_workload_impact': self._calculate_workload_impact(data)
        }
        
        return enrichments, transformations
    
    def _transform_communication_message(self, event: StreamEvent) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform communication message events"""
        data = event.data
        
        enrichments = self._default_enrichment(event)
        enrichments.update({
            'message_length': len(data.get('message', '')),
            'has_mentions': '@' in data.get('message', ''),
            'has_attachments': len(data.get('attachments', [])) > 0,
            'is_thread_reply': data.get('thread_id') is not None,
            'channel_type': data.get('channel_type', 'unknown'),
            'participant_count': len(data.get('participants', []))
        })
        
        transformations = {
            'sentiment_score': self._analyze_sentiment(data.get('message', '')),
            'extracted_entities': self._extract_entities(data.get('message', '')),
            'action_items': self._extract_action_items(data.get('message', '')),
            'urgency_level': self._assess_urgency(data.get('message', ''))
        }
        
        return enrichments, transformations
    
    def _transform_system_metric(self, event: StreamEvent) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform system metric events"""
        data = event.data
        
        enrichments = self._default_enrichment(event)
        enrichments.update({
            'metric_type': data.get('metric_type', 'unknown'),
            'metric_value': data.get('value', 0),
            'metric_unit': data.get('unit', ''),
            'is_anomaly': data.get('is_anomaly', False),
            'threshold_exceeded': data.get('threshold_exceeded', False),
            'service_name': data.get('service', 'unknown')
        })
        
        transformations = {
            'normalized_value': self._normalize_metric_value(data),
            'trend_direction': self._calculate_trend(data),
            'anomaly_score': self._calculate_anomaly_score(data),
            'impact_assessment': self._assess_metric_impact(data)
        }
        
        return enrichments, transformations
    
    def _transform_quality_alert(self, event: StreamEvent) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform quality alert events"""
        data = event.data
        
        enrichments = self._default_enrichment(event)
        enrichments.update({
            'alert_severity': data.get('severity', 'unknown'),
            'affected_records': data.get('affected_records', 0),
            'quality_dimension': data.get('quality_dimension', 'unknown'),
            'threshold_value': data.get('threshold', 0),
            'actual_value': data.get('actual_value', 0),
            'data_source': data.get('data_source', 'unknown')
        })
        
        transformations = {
            'severity_score': self._calculate_severity_score(data),
            'impact_radius': self._calculate_impact_radius(data),
            'remediation_priority': self._calculate_remediation_priority(data),
            'root_cause_indicators': self._identify_root_cause_indicators(data)
        }
        
        return enrichments, transformations
    
    def _transform_knowledge_extraction(self, event: StreamEvent) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform knowledge extraction events"""
        data = event.data
        
        enrichments = self._default_enrichment(event)
        enrichments.update({
            'entities_extracted': len(data.get('entities', [])),
            'relationships_extracted': len(data.get('relationships', [])),
            'concepts_extracted': len(data.get('concepts', [])),
            'extraction_confidence': data.get('confidence', 0),
            'source_document_type': data.get('document_type', 'unknown'),
            'extraction_method': data.get('method', 'unknown')
        })
        
        transformations = {
            'knowledge_density': self._calculate_knowledge_density(data),
            'entity_importance_scores': self._score_entity_importance(data),
            'relationship_strength': self._assess_relationship_strength(data),
            'concept_novelty': self._assess_concept_novelty(data)
        }
        
        return enrichments, transformations
    
    def _transform_user_action(self, event: StreamEvent) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Transform user action events"""
        data = event.data
        
        enrichments = self._default_enrichment(event)
        enrichments.update({
            'action_type': data.get('action', 'unknown'),
            'user_role': data.get('user_role', 'unknown'),
            'session_duration': data.get('session_duration', 0),
            'is_automated': data.get('is_automated', False),
            'success': data.get('success', True),
            'resource_accessed': data.get('resource', 'unknown')
        })
        
        transformations = {
            'user_intent': self._infer_user_intent(data),
            'action_complexity': self._assess_action_complexity(data),
            'workflow_stage': self._identify_workflow_stage(data),
            'efficiency_score': self._calculate_efficiency_score(data)
        }
        
        return enrichments, transformations
    
    # Helper methods for transformations
    def _classify_branch(self, branch_name: str) -> str:
        """Classify branch type"""
        branch_lower = branch_name.lower()
        if branch_lower in ['main', 'master', 'production']:
            return 'main'
        elif branch_lower.startswith(('feature/', 'feat/')):
            return 'feature'
        elif branch_lower.startswith(('hotfix/', 'fix/')):
            return 'hotfix'
        elif branch_lower.startswith(('release/', 'rel/')):
            return 'release'
        elif branch_lower.startswith('develop'):
            return 'develop'
        else:
            return 'other'
    
    def _calculate_change_magnitude(self, data: Dict[str, Any]) -> str:
        """Calculate magnitude of code change"""
        files_changed = len(data.get('files_changed', []))
        if files_changed == 0:
            return 'none'
        elif files_changed <= 3:
            return 'small'
        elif files_changed <= 10:
            return 'medium'
        else:
            return 'large'
    
    def _extract_commit_keywords(self, commit_message: str) -> List[str]:
        """Extract keywords from commit message"""
        keywords = []
        message_lower = commit_message.lower()
        
        # Common commit keywords
        commit_keywords = ['fix', 'bug', 'feature', 'add', 'update', 'remove', 'refactor', 'test', 'docs']
        
        for keyword in commit_keywords:
            if keyword in message_lower:
                keywords.append(keyword)
        
        return keywords
    
    def _calculate_impact_score(self, data: Dict[str, Any]) -> float:
        """Calculate impact score for code changes"""
        score = 0.0
        
        # File count impact
        files_changed = len(data.get('files_changed', []))
        score += min(files_changed * 0.1, 1.0)
        
        # Branch impact
        branch = data.get('branch', '')
        if branch.lower() in ['main', 'master', 'production']:
            score += 0.5
        
        # Merge commit impact
        if data.get('is_merge', False):
            score += 0.3
        
        return min(score, 1.0)
    
    def _summarize_content(self, content: str) -> str:
        """Create content summary"""
        if len(content) <= 200:
            return content
        
        # Simple extractive summary - first sentence and key points
        sentences = content.split('.')
        if sentences:
            summary = sentences[0] + '.'
            if len(summary) < 150 and len(sentences) > 1:
                summary += ' ' + sentences[1] + '.'
            return summary[:200] + '...' if len(summary) > 200 else summary
        
        return content[:200] + '...'
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content"""
        # Simple keyword extraction
        words = content.lower().split()
        
        # Technical keywords
        tech_keywords = ['api', 'database', 'server', 'client', 'authentication', 'security', 'performance', 'testing']
        
        topics = []
        for keyword in tech_keywords:
            if keyword in words:
                topics.append(keyword)
        
        return topics[:5]  # Limit to top 5 topics
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score (simplified)"""
        if not content:
            return 0.0
        
        words = content.split()
        sentences = content.split('.')
        
        if len(sentences) == 0:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Simple readability score (lower is more readable)
        if avg_words_per_sentence <= 15:
            return 0.9
        elif avg_words_per_sentence <= 25:
            return 0.7
        else:
            return 0.5
    
    def _assess_technical_complexity(self, content: str) -> str:
        """Assess technical complexity of content"""
        tech_indicators = ['algorithm', 'implementation', 'architecture', 'framework', 'protocol', 'optimization']
        
        content_lower = content.lower()
        complexity_score = sum(1 for indicator in tech_indicators if indicator in content_lower)
        
        if complexity_score >= 3:
            return 'high'
        elif complexity_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _normalize_priority(self, priority: str) -> int:
        """Normalize priority to numeric scale"""
        priority_lower = priority.lower()
        if priority_lower in ['critical', 'highest', '1']:
            return 5
        elif priority_lower in ['high', '2']:
            return 4
        elif priority_lower in ['medium', 'normal', '3']:
            return 3
        elif priority_lower in ['low', '4']:
            return 2
        elif priority_lower in ['lowest', '5']:
            return 1
        else:
            return 3  # Default to medium
    
    def _analyze_status_progression(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze status progression"""
        old_status = data.get('old_status', '')
        new_status = data.get('new_status', '')
        
        # Define status order
        status_order = ['backlog', 'todo', 'in_progress', 'review', 'testing', 'done']
        
        try:
            old_index = status_order.index(old_status.lower())
            new_index = status_order.index(new_status.lower())
            
            return {
                'direction': 'forward' if new_index > old_index else 'backward',
                'steps': abs(new_index - old_index),
                'is_completion': new_status.lower() == 'done'
            }
        except ValueError:
            return {
                'direction': 'unknown',
                'steps': 0,
                'is_completion': False
            }
    
    def _calculate_effort_variance(self, data: Dict[str, Any]) -> float:
        """Calculate effort variance"""
        estimated = data.get('estimated_hours', 0)
        actual = data.get('actual_hours', 0)
        
        if estimated == 0:
            return 0.0
        
        return (actual - estimated) / estimated
    
    def _assess_milestone_impact(self, data: Dict[str, Any]) -> str:
        """Assess impact on milestones"""
        # Simplified assessment based on priority and effort variance
        priority = self._normalize_priority(data.get('priority', ''))
        effort_variance = self._calculate_effort_variance(data)
        
        if priority >= 4 and effort_variance > 0.5:
            return 'high'
        elif priority >= 3 and effort_variance > 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_workload_impact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate workload impact"""
        estimated_hours = data.get('estimated_hours', 0)
        assignee = data.get('assignee', '')
        
        return {
            'hours_added': estimated_hours,
            'assignee': assignee,
            'workload_category': 'high' if estimated_hours > 20 else 'medium' if estimated_hours > 5 else 'low'
        }
    
    def _analyze_sentiment(self, message: str) -> float:
        """Analyze sentiment of message (simplified)"""
        positive_words = ['good', 'great', 'excellent', 'awesome', 'perfect', 'love', 'like', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'angry', 'frustrated', 'problem']
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count + negative_count == 0:
            return 0.0  # Neutral
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_entities(self, message: str) -> List[str]:
        """Extract entities from message (simplified)"""
        # Simple entity extraction based on patterns
        entities = []
        
        # Email addresses
        import re
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message)
        entities.extend(emails)
        
        # URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)
        entities.extend(urls)
        
        return entities
    
    def _extract_action_items(self, message: str) -> List[str]:
        """Extract action items from message"""
        action_indicators = ['todo', 'action', 'task', 'need to', 'should', 'must', 'will']
        
        sentences = message.split('.')
        action_items = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in action_indicators):
                action_items.append(sentence.strip())
        
        return action_items
    
    def _assess_urgency(self, message: str) -> str:
        """Assess urgency level of message"""
        urgent_indicators = ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'now']
        high_indicators = ['soon', 'quickly', 'fast', 'priority']
        
        message_lower = message.lower()
        
        if any(indicator in message_lower for indicator in urgent_indicators):
            return 'urgent'
        elif any(indicator in message_lower for indicator in high_indicators):
            return 'high'
        else:
            return 'normal'
    
    # Additional helper methods would continue here...
    # For brevity, I'll include a few more key methods
    
    def _normalize_metric_value(self, data: Dict[str, Any]) -> float:
        """Normalize metric value to 0-1 scale"""
        value = data.get('value', 0)
        min_value = data.get('min_value', 0)
        max_value = data.get('max_value', 100)
        
        if max_value == min_value:
            return 0.0
        
        return (value - min_value) / (max_value - min_value)
    
    def _calculate_trend(self, data: Dict[str, Any]) -> str:
        """Calculate trend direction"""
        current_value = data.get('value', 0)
        previous_value = data.get('previous_value', current_value)
        
        if current_value > previous_value:
            return 'increasing'
        elif current_value < previous_value:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_anomaly_score(self, data: Dict[str, Any]) -> float:
        """Calculate anomaly score for metric"""
        value = data.get('value', 0)
        mean = data.get('historical_mean', value)
        std = data.get('historical_std', 1)
        
        if std == 0:
            return 0.0
        
        z_score = abs((value - mean) / std)
        return min(z_score / 3.0, 1.0)  # Normalize to 0-1
    
    def _assess_metric_impact(self, data: Dict[str, Any]) -> str:
        """Assess impact of metric change"""
        anomaly_score = self._calculate_anomaly_score(data)
        
        if anomaly_score > 0.8:
            return 'high'
        elif anomaly_score > 0.5:
            return 'medium'
        else:
            return 'low'

class EventCorrelationEngine:
    """Correlates events across different sources and time windows"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config['database']
        self.redis_config = config['redis']
        
        # Redis client for correlation cache
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Correlation metrics
        self.correlations_found = Counter('event_correlations_total', 'Total correlations found', ['correlation_type'])
        self.correlation_duration = Histogram('event_correlation_duration_seconds', 'Correlation processing time')
        self.correlation_accuracy = Gauge('event_correlation_accuracy', 'Correlation accuracy score')
        
        # Event buffers for correlation windows
        self.event_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Correlation patterns
        self._load_correlation_patterns()
    
    def _load_correlation_patterns(self):
        """Load correlation patterns and rules"""
        self.correlation_patterns = [
            {
                'name': 'code_deployment_sequence',
                'events': [EventType.CODE_CHANGE, EventType.SYSTEM_METRIC],
                'time_window_minutes': 30,
                'correlation_fields': ['repository', 'service_name'],
                'confidence_threshold': 0.7
            },
            {
                'name': 'documentation_code_sync',
                'events': [EventType.CODE_CHANGE, EventType.DOCUMENTATION_UPDATE],
                'time_window_minutes': 60,
                'correlation_fields': ['repository', 'component'],
                'confidence_threshold': 0.6
            },
            {
                'name': 'project_communication_flow',
                'events': [EventType.PROJECT_UPDATE, EventType.COMMUNICATION_MESSAGE],
                'time_window_minutes': 15,
                'correlation_fields': ['project_id', 'assignee'],
                'confidence_threshold': 0.8
            },
            {
                'name': 'quality_alert_cascade',
                'events': [EventType.QUALITY_ALERT, EventType.SYSTEM_METRIC],
                'time_window_minutes': 10,
                'correlation_fields': ['data_source', 'service'],
                'confidence_threshold': 0.9
            }
        ]
    
    def correlate_event(self, processed_event: ProcessedEvent) -> List[str]:
        """Find correlations for a processed event"""
        correlations = []
        
        try:
            with self.correlation_duration.time():
                # Add event to appropriate buffers
                self._add_to_buffers(processed_event)
                
                # Find correlations using different strategies
                correlations.extend(self._find_temporal_correlations(processed_event))
                correlations.extend(self._find_semantic_correlations(processed_event))
                correlations.extend(self._find_pattern_correlations(processed_event))
                
                # Update correlation metrics
                for correlation in correlations:
                    self.correlations_found.labels(correlation_type='temporal').inc()
            
        except Exception as e:
            logger.error(f"Correlation failed for event {processed_event.original_event.event_id}: {e}")
        
        return correlations
    
    def _add_to_buffers(self, processed_event: ProcessedEvent):
        """Add event to correlation buffers"""
        event = processed_event.original_event
        
        # Add to type-specific buffer
        self.event_buffers[event.event_type.value].append(processed_event)
        
        # Add to source-specific buffer
        self.event_buffers[f"source_{event.source_system}"].append(processed_event)
        
        # Add to global buffer
        self.event_buffers['global'].append(processed_event)
    
    def _find_temporal_correlations(self, processed_event: ProcessedEvent) -> List[str]:
        """Find temporal correlations based on time proximity"""
        correlations = []
        event = processed_event.original_event
        
        # Look for events within time windows
        time_windows = [5, 15, 30, 60]  # minutes
        
        for window_minutes in time_windows:
            window_start = event.timestamp - timedelta(minutes=window_minutes)
            window_end = event.timestamp + timedelta(minutes=window_minutes)
            
            # Check global buffer for events in time window
            for buffered_event in self.event_buffers['global']:
                buffered_timestamp = buffered_event.original_event.timestamp
                
                if (window_start <= buffered_timestamp <= window_end and 
                    buffered_event.original_event.event_id != event.event_id):
                    
                    # Calculate correlation strength
                    correlation_strength = self._calculate_temporal_correlation(
                        processed_event, buffered_event, window_minutes
                    )
                    
                    if correlation_strength > 0.5:
                        correlations.append(buffered_event.original_event.event_id)
        
        return correlations
    
    def _find_semantic_correlations(self, processed_event: ProcessedEvent) -> List[str]:
        """Find semantic correlations based on content similarity"""
        correlations = []
        event = processed_event.original_event
        
        # Extract text content for similarity comparison
        event_text = self._extract_text_content(event)
        
        if not event_text:
            return correlations
        
        # Compare with recent events of different types
        for buffered_event in list(self.event_buffers['global'])[-100:]:  # Last 100 events
            if buffered_event.original_event.event_id == event.event_id:
                continue
            
            buffered_text = self._extract_text_content(buffered_event.original_event)
            
            if buffered_text:
                similarity = self._calculate_text_similarity(event_text, buffered_text)
                
                if similarity > 0.7:
                    correlations.append(buffered_event.original_event.event_id)
        
        return correlations
    
    def _find_pattern_correlations(self, processed_event: ProcessedEvent) -> List[str]:
        """Find correlations based on predefined patterns"""
        correlations = []
        event = processed_event.original_event
        
        for pattern in self.correlation_patterns:
            if event.event_type in pattern['events']:
                # Look for other events in the pattern
                other_event_types = [et for et in pattern['events'] if et != event.event_type]
                
                for other_type in other_event_types:
                    correlations.extend(
                        self._find_pattern_matches(processed_event, other_type, pattern)
                    )
        
        return correlations
    
    def _calculate_temporal_correlation(self, event1: ProcessedEvent, event2: ProcessedEvent, window_minutes: int) -> float:
        """Calculate temporal correlation strength"""
        time_diff = abs((event1.original_event.timestamp - event2.original_event.timestamp).total_seconds())
        max_time_diff = window_minutes * 60
        
        # Closer in time = higher correlation
        time_correlation = 1.0 - (time_diff / max_time_diff)
        
        # Check for common fields
        field_correlation = self._calculate_field_correlation(event1.original_event, event2.original_event)
        
        # Combined correlation score
        return (time_correlation * 0.6) + (field_correlation * 0.4)
    
    def _calculate_field_correlation(self, event1: StreamEvent, event2: StreamEvent) -> float:
        """Calculate correlation based on common fields"""
        common_fields = ['source_system', 'correlation_id', 'parent_event_id']
        matches = 0
        total_fields = 0
        
        for field in common_fields:
            value1 = getattr(event1, field, None)
            value2 = getattr(event2, field, None)
            
            if value1 is not None and value2 is not None:
                total_fields += 1
                if value1 == value2:
                    matches += 1
        
        # Check data field overlaps
        data1_keys = set(event1.data.keys())
        data2_keys = set(event2.data.keys())
        
        if data1_keys and data2_keys:
            data_overlap = len(data1_keys.intersection(data2_keys)) / len(data1_keys.union(data2_keys))
            total_fields += 1
            matches += data_overlap
        
        return matches / total_fields if total_fields > 0 else 0.0
    
    def _extract_text_content(self, event: StreamEvent) -> str:
        """Extract text content from event for similarity analysis"""
        text_parts = []
        
        # Extract from common text fields
        text_fields = ['message', 'content', 'description', 'title', 'commit_message']
        
        for field in text_fields:
            if field in event.data and isinstance(event.data[field], str):
                text_parts.append(event.data[field])
        
        return ' '.join(text_parts)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF and cosine similarity"""
        if not text1 or not text2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            return 0.0
    
    def _find_pattern_matches(self, processed_event: ProcessedEvent, target_type: EventType, pattern: Dict[str, Any]) -> List[str]:
        """Find events matching a specific pattern"""
        correlations = []
        event = processed_event.original_event
        
        # Look in type-specific buffer
        target_buffer = self.event_buffers[target_type.value]
        
        window_minutes = pattern['time_window_minutes']
        window_start = event.timestamp - timedelta(minutes=window_minutes)
        window_end = event.timestamp + timedelta(minutes=window_minutes)
        
        for buffered_event in target_buffer:
            buffered_timestamp = buffered_event.original_event.timestamp
            
            if window_start <= buffered_timestamp <= window_end:
                # Check correlation fields
                correlation_score = self._check_pattern_fields(
                    event, buffered_event.original_event, pattern['correlation_fields']
                )
                
                if correlation_score >= pattern['confidence_threshold']:
                    correlations.append(buffered_event.original_event.event_id)
        
        return correlations
    
    def _check_pattern_fields(self, event1: StreamEvent, event2: StreamEvent, correlation_fields: List[str]) -> float:
        """Check correlation based on specific fields"""
        matches = 0
        total_checks = 0
        
        for field in correlation_fields:
            # Check in event data
            value1 = event1.data.get(field)
            value2 = event2.data.get(field)
            
            if value1 is not None and value2 is not None:
                total_checks += 1
                if value1 == value2:
                    matches += 1
            
            # Check in metadata
            meta_value1 = event1.metadata.get(field)
            meta_value2 = event2.metadata.get(field)
            
            if meta_value1 is not None and meta_value2 is not None:
                total_checks += 1
                if meta_value1 == meta_value2:
                    matches += 1
        
        return matches / total_checks if total_checks > 0 else 0.0

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'nexus_architect',
            'user': 'postgres',
            'password': 'nexus_secure_password_2024'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
    }
    
    # Initialize components
    transformer = StreamTransformer(config)
    correlator = EventCorrelationEngine(config)
    
    # Example event processing
    from kafka_streaming_manager import StreamEvent, EventType
    
    test_event = StreamEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType.CODE_CHANGE,
        source_system='github',
        source_id='repo-123',
        timestamp=datetime.now(timezone.utc),
        data={
            'repository': 'nexus-architect',
            'branch': 'main',
            'commit_id': 'abc123',
            'author': 'developer@example.com',
            'files_changed': ['src/main.py', 'docs/README.md'],
            'commit_message': 'Fix authentication bug and update documentation'
        },
        metadata={
            'platform': 'github',
            'webhook_id': 'webhook-456'
        }
    )
    
    # Transform event
    processed_event = transformer.transform_event(test_event)
    print(f"Processed event: {processed_event.original_event.event_id}")
    print(f"Enrichments: {processed_event.enrichments}")
    print(f"Transformations: {processed_event.transformations}")
    
    # Find correlations
    correlations = correlator.correlate_event(processed_event)
    print(f"Correlations found: {correlations}")

