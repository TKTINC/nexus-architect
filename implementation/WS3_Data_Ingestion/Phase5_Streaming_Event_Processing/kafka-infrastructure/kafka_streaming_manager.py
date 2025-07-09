"""
Kafka Streaming Manager for Nexus Architect
Provides comprehensive Apache Kafka infrastructure for real-time event streaming
"""

import json
import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types for different data sources"""
    CODE_CHANGE = "code_change"
    DOCUMENTATION_UPDATE = "documentation_update"
    PROJECT_UPDATE = "project_update"
    COMMUNICATION_MESSAGE = "communication_message"
    SYSTEM_METRIC = "system_metric"
    QUALITY_ALERT = "quality_alert"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    USER_ACTION = "user_action"

@dataclass
class StreamEvent:
    """Standard event structure for all streaming events"""
    event_id: str
    event_type: EventType
    source_system: str
    source_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source_system': self.source_system,
            'source_id': self.source_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'parent_event_id': self.parent_event_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        """Create event from dictionary"""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            source_system=data['source_system'],
            source_id=data['source_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            metadata=data['metadata'],
            correlation_id=data.get('correlation_id'),
            parent_event_id=data.get('parent_event_id')
        )

class KafkaTopicManager:
    """Manages Kafka topics and configurations"""
    
    def __init__(self, bootstrap_servers: str):
        self.bootstrap_servers = bootstrap_servers
        self.admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='nexus_topic_manager'
        )
        
        # Topic configurations
        self.topic_configs = {
            'nexus-code-changes': {
                'num_partitions': 12,
                'replication_factor': 3,
                'config': {
                    'retention.ms': '604800000',  # 7 days
                    'compression.type': 'snappy',
                    'cleanup.policy': 'delete'
                }
            },
            'nexus-documentation-updates': {
                'num_partitions': 8,
                'replication_factor': 3,
                'config': {
                    'retention.ms': '2592000000',  # 30 days
                    'compression.type': 'snappy',
                    'cleanup.policy': 'delete'
                }
            },
            'nexus-project-updates': {
                'num_partitions': 6,
                'replication_factor': 3,
                'config': {
                    'retention.ms': '2592000000',  # 30 days
                    'compression.type': 'snappy',
                    'cleanup.policy': 'delete'
                }
            },
            'nexus-communication-messages': {
                'num_partitions': 10,
                'replication_factor': 3,
                'config': {
                    'retention.ms': '1209600000',  # 14 days
                    'compression.type': 'snappy',
                    'cleanup.policy': 'delete'
                }
            },
            'nexus-system-metrics': {
                'num_partitions': 4,
                'replication_factor': 3,
                'config': {
                    'retention.ms': '259200000',  # 3 days
                    'compression.type': 'snappy',
                    'cleanup.policy': 'delete'
                }
            },
            'nexus-quality-alerts': {
                'num_partitions': 3,
                'replication_factor': 3,
                'config': {
                    'retention.ms': '2592000000',  # 30 days
                    'compression.type': 'snappy',
                    'cleanup.policy': 'delete'
                }
            },
            'nexus-knowledge-extraction': {
                'num_partitions': 8,
                'replication_factor': 3,
                'config': {
                    'retention.ms': '5184000000',  # 60 days
                    'compression.type': 'snappy',
                    'cleanup.policy': 'delete'
                }
            },
            'nexus-user-actions': {
                'num_partitions': 6,
                'replication_factor': 3,
                'config': {
                    'retention.ms': '1209600000',  # 14 days
                    'compression.type': 'snappy',
                    'cleanup.policy': 'delete'
                }
            }
        }
    
    def create_topics(self) -> Dict[str, bool]:
        """Create all required Kafka topics"""
        topics_to_create = []
        results = {}
        
        for topic_name, config in self.topic_configs.items():
            topic = NewTopic(
                name=topic_name,
                num_partitions=config['num_partitions'],
                replication_factor=config['replication_factor'],
                topic_configs=config['config']
            )
            topics_to_create.append(topic)
        
        try:
            # Create topics
            future_map = self.admin_client.create_topics(topics_to_create)
            
            for topic_name, future in future_map.items():
                try:
                    future.result()  # Block until topic is created
                    results[topic_name] = True
                    logger.info(f"Successfully created topic: {topic_name}")
                except TopicAlreadyExistsError:
                    results[topic_name] = True
                    logger.info(f"Topic already exists: {topic_name}")
                except Exception as e:
                    results[topic_name] = False
                    logger.error(f"Failed to create topic {topic_name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to create topics: {e}")
            for topic_name in self.topic_configs.keys():
                results[topic_name] = False
        
        return results
    
    def get_topic_for_event_type(self, event_type: EventType) -> str:
        """Get appropriate topic for event type"""
        topic_mapping = {
            EventType.CODE_CHANGE: 'nexus-code-changes',
            EventType.DOCUMENTATION_UPDATE: 'nexus-documentation-updates',
            EventType.PROJECT_UPDATE: 'nexus-project-updates',
            EventType.COMMUNICATION_MESSAGE: 'nexus-communication-messages',
            EventType.SYSTEM_METRIC: 'nexus-system-metrics',
            EventType.QUALITY_ALERT: 'nexus-quality-alerts',
            EventType.KNOWLEDGE_EXTRACTION: 'nexus-knowledge-extraction',
            EventType.USER_ACTION: 'nexus-user-actions'
        }
        return topic_mapping.get(event_type, 'nexus-user-actions')

class KafkaEventProducer:
    """High-performance Kafka event producer"""
    
    def __init__(self, bootstrap_servers: str, schema_registry_url: Optional[str] = None):
        self.bootstrap_servers = bootstrap_servers
        self.schema_registry_url = schema_registry_url
        
        # Producer configuration for high throughput and reliability
        producer_config = {
            'bootstrap_servers': bootstrap_servers,
            'client_id': 'nexus_event_producer',
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
            'batch_size': 16384,
            'linger_ms': 10,  # Small delay for batching
            'buffer_memory': 33554432,  # 32MB buffer
            'compression_type': 'snappy',
            'max_in_flight_requests_per_connection': 5,
            'enable_idempotence': True,
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None
        }
        
        self.producer = KafkaProducer(**producer_config)
        self.topic_manager = KafkaTopicManager(bootstrap_servers)
        
        # Metrics
        self.events_produced = Counter('kafka_events_produced_total', 'Total events produced', ['topic', 'event_type'])
        self.produce_duration = Histogram('kafka_produce_duration_seconds', 'Time spent producing events', ['topic'])
        self.produce_errors = Counter('kafka_produce_errors_total', 'Total produce errors', ['topic', 'error_type'])
        
        # Schema registry client (if available)
        self.schema_registry_client = None
        if schema_registry_url:
            try:
                self.schema_registry_client = SchemaRegistryClient({'url': schema_registry_url})
            except Exception as e:
                logger.warning(f"Failed to connect to schema registry: {e}")
    
    def produce_event(self, event: StreamEvent, partition_key: Optional[str] = None) -> bool:
        """Produce a single event to Kafka"""
        topic = self.topic_manager.get_topic_for_event_type(event.event_type)
        
        # Use event_id as partition key if not provided
        if not partition_key:
            partition_key = event.source_id or event.event_id
        
        try:
            with self.produce_duration.labels(topic=topic).time():
                # Serialize event
                event_data = event.to_dict()
                
                # Send to Kafka
                future = self.producer.send(
                    topic=topic,
                    key=partition_key,
                    value=event_data,
                    timestamp_ms=int(event.timestamp.timestamp() * 1000)
                )
                
                # Wait for send to complete
                record_metadata = future.get(timeout=10)
                
                # Update metrics
                self.events_produced.labels(
                    topic=topic,
                    event_type=event.event_type.value
                ).inc()
                
                logger.debug(f"Event {event.event_id} sent to {topic}:{record_metadata.partition}:{record_metadata.offset}")
                return True
                
        except Exception as e:
            self.produce_errors.labels(
                topic=topic,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to produce event {event.event_id}: {e}")
            return False
    
    def produce_events_batch(self, events: List[StreamEvent]) -> Dict[str, int]:
        """Produce multiple events in batch"""
        results = {'success': 0, 'failed': 0}
        
        for event in events:
            if self.produce_event(event):
                results['success'] += 1
            else:
                results['failed'] += 1
        
        # Flush producer to ensure all messages are sent
        self.producer.flush()
        
        return results
    
    def close(self):
        """Close producer and clean up resources"""
        try:
            self.producer.flush()
            self.producer.close()
        except Exception as e:
            logger.error(f"Error closing producer: {e}")

class KafkaEventConsumer:
    """High-performance Kafka event consumer with processing callbacks"""
    
    def __init__(self, bootstrap_servers: str, group_id: str, topics: List[str]):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics
        self.running = False
        self.processing_callbacks: Dict[EventType, List[Callable]] = {}
        
        # Consumer configuration
        consumer_config = {
            'bootstrap_servers': bootstrap_servers,
            'group_id': group_id,
            'client_id': f'nexus_consumer_{group_id}',
            'auto_offset_reset': 'latest',
            'enable_auto_commit': False,  # Manual commit for reliability
            'max_poll_records': 500,
            'fetch_min_bytes': 1024,
            'fetch_max_wait_ms': 500,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda k: k.decode('utf-8') if k else None
        }
        
        self.consumer = KafkaConsumer(*topics, **consumer_config)
        
        # Metrics
        self.events_consumed = Counter('kafka_events_consumed_total', 'Total events consumed', ['topic', 'event_type'])
        self.processing_duration = Histogram('kafka_processing_duration_seconds', 'Time spent processing events', ['event_type'])
        self.processing_errors = Counter('kafka_processing_errors_total', 'Total processing errors', ['event_type', 'error_type'])
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def register_callback(self, event_type: EventType, callback: Callable[[StreamEvent], None]):
        """Register a callback function for specific event type"""
        if event_type not in self.processing_callbacks:
            self.processing_callbacks[event_type] = []
        self.processing_callbacks[event_type].append(callback)
    
    def process_event(self, event: StreamEvent):
        """Process a single event with registered callbacks"""
        try:
            with self.processing_duration.labels(event_type=event.event_type.value).time():
                # Get callbacks for this event type
                callbacks = self.processing_callbacks.get(event.event_type, [])
                
                # Execute all callbacks
                for callback in callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        self.processing_errors.labels(
                            event_type=event.event_type.value,
                            error_type=type(e).__name__
                        ).inc()
                        logger.error(f"Callback error for event {event.event_id}: {e}")
                
        except Exception as e:
            self.processing_errors.labels(
                event_type=event.event_type.value,
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Failed to process event {event.event_id}: {e}")
    
    def start_consuming(self):
        """Start consuming events from Kafka"""
        self.running = True
        logger.info(f"Starting consumer for topics: {self.topics}")
        
        try:
            while self.running:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    continue
                
                # Process messages
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Deserialize event
                            event = StreamEvent.from_dict(message.value)
                            
                            # Update metrics
                            self.events_consumed.labels(
                                topic=topic_partition.topic,
                                event_type=event.event_type.value
                            ).inc()
                            
                            # Submit for processing
                            self.executor.submit(self.process_event, event)
                            
                        except Exception as e:
                            logger.error(f"Failed to deserialize message: {e}")
                
                # Commit offsets after processing batch
                try:
                    self.consumer.commit()
                except Exception as e:
                    logger.error(f"Failed to commit offsets: {e}")
                    
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.close()
    
    def stop_consuming(self):
        """Stop consuming events"""
        self.running = False
    
    def close(self):
        """Close consumer and clean up resources"""
        try:
            self.consumer.close()
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error closing consumer: {e}")

class StreamingManager:
    """Main streaming manager coordinating producers and consumers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bootstrap_servers = config['kafka']['bootstrap_servers']
        self.schema_registry_url = config.get('schema_registry', {}).get('url')
        
        # Initialize components
        self.topic_manager = KafkaTopicManager(self.bootstrap_servers)
        self.producer = KafkaEventProducer(self.bootstrap_servers, self.schema_registry_url)
        self.consumers: Dict[str, KafkaEventConsumer] = {}
        
        # Database connection
        self.db_config = config['database']
        self.redis_config = config['redis']
        
        # Metrics
        self.stream_health = Gauge('kafka_stream_health', 'Stream health status', ['component'])
        
        # Initialize database schema
        self._init_database_schema()
        
        # Setup topics
        self._setup_topics()
    
    def _init_database_schema(self):
        """Initialize database schema for streaming events"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create streaming events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS streaming_events (
                    event_id VARCHAR(255) PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    source_system VARCHAR(255) NOT NULL,
                    source_id VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data JSONB NOT NULL,
                    metadata JSONB,
                    correlation_id VARCHAR(255),
                    parent_event_id VARCHAR(255),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status VARCHAR(50) DEFAULT 'pending'
                );
                
                CREATE INDEX IF NOT EXISTS idx_streaming_events_type ON streaming_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_streaming_events_source ON streaming_events(source_system);
                CREATE INDEX IF NOT EXISTS idx_streaming_events_timestamp ON streaming_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_streaming_events_correlation ON streaming_events(correlation_id);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
    
    def _setup_topics(self):
        """Setup Kafka topics"""
        try:
            results = self.topic_manager.create_topics()
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            logger.info(f"Topic setup completed: {success_count}/{total_count} topics ready")
            
            if success_count == total_count:
                self.stream_health.labels(component='topics').set(1)
            else:
                self.stream_health.labels(component='topics').set(0)
                
        except Exception as e:
            logger.error(f"Failed to setup topics: {e}")
            self.stream_health.labels(component='topics').set(0)
    
    def create_consumer(self, group_id: str, topics: List[str]) -> KafkaEventConsumer:
        """Create a new consumer for specified topics"""
        consumer = KafkaEventConsumer(self.bootstrap_servers, group_id, topics)
        self.consumers[group_id] = consumer
        return consumer
    
    def publish_event(self, event: StreamEvent) -> bool:
        """Publish an event to the streaming platform"""
        try:
            # Store event in database
            self._store_event(event)
            
            # Publish to Kafka
            success = self.producer.produce_event(event)
            
            if success:
                self.stream_health.labels(component='producer').set(1)
            else:
                self.stream_health.labels(component='producer').set(0)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            self.stream_health.labels(component='producer').set(0)
            return False
    
    def _store_event(self, event: StreamEvent):
        """Store event in database for persistence and querying"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO streaming_events 
                (event_id, event_type, source_system, source_id, timestamp, data, metadata, correlation_id, parent_event_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
            """, (
                event.event_id,
                event.event_type.value,
                event.source_system,
                event.source_id,
                event.timestamp,
                json.dumps(event.data),
                json.dumps(event.metadata),
                event.correlation_id,
                event.parent_event_id
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store event in database: {e}")
    
    def get_stream_health(self) -> Dict[str, Any]:
        """Get overall streaming system health"""
        try:
            # Check Kafka connectivity
            admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
            metadata = admin_client.describe_cluster()
            kafka_healthy = len(metadata.brokers) > 0
            
            # Check database connectivity
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            db_healthy = True
            
            # Check Redis connectivity
            redis_client = redis.Redis(**self.redis_config)
            redis_client.ping()
            redis_healthy = True
            
            return {
                'kafka': kafka_healthy,
                'database': db_healthy,
                'redis': redis_healthy,
                'overall': kafka_healthy and db_healthy and redis_healthy,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'kafka': False,
                'database': False,
                'redis': False,
                'overall': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def shutdown(self):
        """Shutdown streaming manager and all components"""
        logger.info("Shutting down streaming manager...")
        
        # Stop all consumers
        for consumer in self.consumers.values():
            consumer.stop_consuming()
        
        # Close producer
        self.producer.close()
        
        logger.info("Streaming manager shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'kafka': {
            'bootstrap_servers': 'localhost:9092'
        },
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
    
    # Start metrics server
    start_http_server(8090)
    
    # Initialize streaming manager
    streaming_manager = StreamingManager(config)
    
    # Example event
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
            'files_changed': ['src/main.py', 'docs/README.md']
        },
        metadata={
            'platform': 'github',
            'webhook_id': 'webhook-456'
        }
    )
    
    # Publish test event
    success = streaming_manager.publish_event(test_event)
    print(f"Event published: {success}")
    
    # Check health
    health = streaming_manager.get_stream_health()
    print(f"Stream health: {health}")

