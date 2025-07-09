"""
Webhook Processor for Nexus Architect
Handles real-time webhook events from Git platforms for immediate processing
"""

import asyncio
import logging
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import aiohttp
from aiohttp import web, ClientSession
import aioredis
from prometheus_client import Counter, Histogram, Gauge
import jwt
from cryptography.hazmat.primitives import serialization

# Metrics
WEBHOOK_REQUESTS = Counter('webhook_requests_total', 'Total webhook requests', ['platform', 'event_type', 'status'])
WEBHOOK_LATENCY = Histogram('webhook_processing_latency_seconds', 'Webhook processing latency', ['platform', 'event_type'])
WEBHOOK_QUEUE_SIZE = Gauge('webhook_queue_size', 'Size of webhook processing queue', ['platform'])
WEBHOOK_ERRORS = Counter('webhook_errors_total', 'Total webhook processing errors', ['platform', 'error_type'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebhookEvent(Enum):
    """Types of webhook events"""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    MERGE_REQUEST = "merge_request"
    ISSUE = "issue"
    REPOSITORY = "repository"
    BRANCH = "branch"
    TAG = "tag"
    RELEASE = "release"
    COMMIT_COMMENT = "commit_comment"
    PULL_REQUEST_REVIEW = "pull_request_review"

class WebhookPlatform(Enum):
    """Webhook source platforms"""
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    AZURE_DEVOPS = "azure_devops"

@dataclass
class WebhookPayload:
    """Standardized webhook payload"""
    platform: WebhookPlatform
    event_type: WebhookEvent
    repository_id: str
    repository_name: str
    repository_url: str
    timestamp: datetime
    raw_payload: Dict[str, Any]
    
    # Event-specific data
    commits: List[Dict[str, Any]] = field(default_factory=list)
    pull_request: Optional[Dict[str, Any]] = None
    issue: Optional[Dict[str, Any]] = None
    branch: Optional[str] = None
    tag: Optional[str] = None
    release: Optional[Dict[str, Any]] = None
    
    # Metadata
    sender: Optional[Dict[str, Any]] = None
    organization: Optional[str] = None
    action: Optional[str] = None

class GitHubWebhookProcessor:
    """GitHub webhook processor"""
    
    def __init__(self, secret: Optional[str] = None):
        self.secret = secret
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify GitHub webhook signature"""
        if not self.secret:
            return True  # Skip verification if no secret configured
        
        expected_signature = hmac.new(
            self.secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(f"sha256={expected_signature}", signature)
    
    def parse_payload(self, headers: Dict[str, str], payload: Dict[str, Any]) -> Optional[WebhookPayload]:
        """Parse GitHub webhook payload"""
        event_type = headers.get('X-GitHub-Event', '').lower()
        
        # Map GitHub events to our event types
        event_mapping = {
            'push': WebhookEvent.PUSH,
            'pull_request': WebhookEvent.PULL_REQUEST,
            'issues': WebhookEvent.ISSUE,
            'repository': WebhookEvent.REPOSITORY,
            'create': WebhookEvent.BRANCH,  # Could be branch or tag
            'delete': WebhookEvent.BRANCH,  # Could be branch or tag
            'release': WebhookEvent.RELEASE,
            'commit_comment': WebhookEvent.COMMIT_COMMENT,
            'pull_request_review': WebhookEvent.PULL_REQUEST_REVIEW
        }
        
        if event_type not in event_mapping:
            logger.warning(f"Unsupported GitHub event type: {event_type}")
            return None
        
        repository = payload.get('repository', {})
        
        webhook_payload = WebhookPayload(
            platform=WebhookPlatform.GITHUB,
            event_type=event_mapping[event_type],
            repository_id=str(repository.get('id', '')),
            repository_name=repository.get('full_name', ''),
            repository_url=repository.get('html_url', ''),
            timestamp=datetime.utcnow(),
            raw_payload=payload,
            sender=payload.get('sender'),
            organization=payload.get('organization', {}).get('login'),
            action=payload.get('action')
        )
        
        # Parse event-specific data
        if event_type == 'push':
            webhook_payload.commits = payload.get('commits', [])
            webhook_payload.branch = payload.get('ref', '').replace('refs/heads/', '')
        
        elif event_type == 'pull_request':
            webhook_payload.pull_request = payload.get('pull_request')
        
        elif event_type == 'issues':
            webhook_payload.issue = payload.get('issue')
        
        elif event_type in ['create', 'delete']:
            ref_type = payload.get('ref_type')
            if ref_type == 'branch':
                webhook_payload.event_type = WebhookEvent.BRANCH
                webhook_payload.branch = payload.get('ref')
            elif ref_type == 'tag':
                webhook_payload.event_type = WebhookEvent.TAG
                webhook_payload.tag = payload.get('ref')
        
        elif event_type == 'release':
            webhook_payload.release = payload.get('release')
        
        return webhook_payload

class GitLabWebhookProcessor:
    """GitLab webhook processor"""
    
    def __init__(self, secret: Optional[str] = None):
        self.secret = secret
    
    def verify_signature(self, payload: bytes, token: str) -> bool:
        """Verify GitLab webhook token"""
        if not self.secret:
            return True
        
        return hmac.compare_digest(self.secret, token)
    
    def parse_payload(self, headers: Dict[str, str], payload: Dict[str, Any]) -> Optional[WebhookPayload]:
        """Parse GitLab webhook payload"""
        event_type = headers.get('X-Gitlab-Event', '').lower().replace(' ', '_')
        
        # Map GitLab events to our event types
        event_mapping = {
            'push_hook': WebhookEvent.PUSH,
            'merge_request_hook': WebhookEvent.MERGE_REQUEST,
            'issue_hook': WebhookEvent.ISSUE,
            'tag_push_hook': WebhookEvent.TAG,
            'repository_update_hook': WebhookEvent.REPOSITORY,
            'release_hook': WebhookEvent.RELEASE
        }
        
        if event_type not in event_mapping:
            logger.warning(f"Unsupported GitLab event type: {event_type}")
            return None
        
        project = payload.get('project', {})
        
        webhook_payload = WebhookPayload(
            platform=WebhookPlatform.GITLAB,
            event_type=event_mapping[event_type],
            repository_id=str(project.get('id', '')),
            repository_name=project.get('path_with_namespace', ''),
            repository_url=project.get('web_url', ''),
            timestamp=datetime.utcnow(),
            raw_payload=payload,
            sender=payload.get('user'),
            action=payload.get('object_attributes', {}).get('action')
        )
        
        # Parse event-specific data
        if event_type == 'push_hook':
            webhook_payload.commits = payload.get('commits', [])
            webhook_payload.branch = payload.get('ref', '').replace('refs/heads/', '')
        
        elif event_type == 'merge_request_hook':
            webhook_payload.pull_request = payload.get('object_attributes')
        
        elif event_type == 'issue_hook':
            webhook_payload.issue = payload.get('object_attributes')
        
        elif event_type == 'tag_push_hook':
            webhook_payload.tag = payload.get('ref', '').replace('refs/tags/', '')
        
        elif event_type == 'release_hook':
            webhook_payload.release = payload.get('object_attributes')
        
        return webhook_payload

class BitbucketWebhookProcessor:
    """Bitbucket webhook processor"""
    
    def __init__(self, secret: Optional[str] = None):
        self.secret = secret
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify Bitbucket webhook signature"""
        if not self.secret:
            return True
        
        expected_signature = hmac.new(
            self.secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected_signature, signature)
    
    def parse_payload(self, headers: Dict[str, str], payload: Dict[str, Any]) -> Optional[WebhookPayload]:
        """Parse Bitbucket webhook payload"""
        event_type = headers.get('X-Event-Key', '').lower()
        
        # Map Bitbucket events to our event types
        event_mapping = {
            'repo:push': WebhookEvent.PUSH,
            'pullrequest:created': WebhookEvent.PULL_REQUEST,
            'pullrequest:updated': WebhookEvent.PULL_REQUEST,
            'pullrequest:approved': WebhookEvent.PULL_REQUEST,
            'pullrequest:merged': WebhookEvent.PULL_REQUEST,
            'issue:created': WebhookEvent.ISSUE,
            'issue:updated': WebhookEvent.ISSUE
        }
        
        if event_type not in event_mapping:
            logger.warning(f"Unsupported Bitbucket event type: {event_type}")
            return None
        
        repository = payload.get('repository', {})
        
        webhook_payload = WebhookPayload(
            platform=WebhookPlatform.BITBUCKET,
            event_type=event_mapping[event_type],
            repository_id=repository.get('uuid', ''),
            repository_name=repository.get('full_name', ''),
            repository_url=repository.get('links', {}).get('html', {}).get('href', ''),
            timestamp=datetime.utcnow(),
            raw_payload=payload,
            sender=payload.get('actor')
        )
        
        # Parse event-specific data
        if event_type == 'repo:push':
            webhook_payload.commits = []
            for change in payload.get('push', {}).get('changes', []):
                webhook_payload.commits.extend(change.get('commits', []))
                if change.get('new', {}).get('name'):
                    webhook_payload.branch = change['new']['name']
        
        elif 'pullrequest' in event_type:
            webhook_payload.pull_request = payload.get('pullrequest')
        
        elif 'issue' in event_type:
            webhook_payload.issue = payload.get('issue')
        
        return webhook_payload

class WebhookProcessor:
    """Main webhook processor that handles all platforms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.processors = {}
        self.event_handlers: Dict[WebhookEvent, List[Callable]] = {}
        self.processing_queue = asyncio.Queue()
        self.worker_tasks = []
        
        # Initialize platform processors
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize platform-specific processors"""
        platforms_config = self.config.get('platforms', {})
        
        if 'github' in platforms_config:
            github_config = platforms_config['github']
            self.processors[WebhookPlatform.GITHUB] = GitHubWebhookProcessor(
                secret=github_config.get('webhook_secret')
            )
        
        if 'gitlab' in platforms_config:
            gitlab_config = platforms_config['gitlab']
            self.processors[WebhookPlatform.GITLAB] = GitLabWebhookProcessor(
                secret=gitlab_config.get('webhook_secret')
            )
        
        if 'bitbucket' in platforms_config:
            bitbucket_config = platforms_config['bitbucket']
            self.processors[WebhookPlatform.BITBUCKET] = BitbucketWebhookProcessor(
                secret=bitbucket_config.get('webhook_secret')
            )
    
    async def initialize(self):
        """Initialize webhook processor"""
        # Initialize Redis for queue management
        redis_config = self.config.get('redis', {})
        self.redis_client = aioredis.from_url(
            f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}"
        )
        
        # Start worker tasks
        num_workers = self.config.get('num_workers', 4)
        for i in range(num_workers):
            task = asyncio.create_task(self._worker())
            self.worker_tasks.append(task)
        
        logger.info(f"Webhook processor initialized with {num_workers} workers")
    
    def register_handler(self, event_type: WebhookEvent, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value} events")
    
    async def process_webhook(self, platform: str, headers: Dict[str, str], 
                            payload_data: bytes) -> Dict[str, Any]:
        """Process incoming webhook"""
        start_time = time.time()
        
        try:
            # Parse platform
            platform_enum = WebhookPlatform(platform.lower())
            processor = self.processors.get(platform_enum)
            
            if not processor:
                raise ValueError(f"No processor configured for platform: {platform}")
            
            # Verify signature if applicable
            signature = headers.get('X-Hub-Signature-256') or headers.get('X-Gitlab-Token') or headers.get('X-Hub-Signature')
            if signature and hasattr(processor, 'verify_signature'):
                if not processor.verify_signature(payload_data, signature):
                    WEBHOOK_ERRORS.labels(platform=platform, error_type='signature_verification').inc()
                    raise ValueError("Webhook signature verification failed")
            
            # Parse JSON payload
            try:
                payload = json.loads(payload_data.decode('utf-8'))
            except json.JSONDecodeError as e:
                WEBHOOK_ERRORS.labels(platform=platform, error_type='json_parsing').inc()
                raise ValueError(f"Invalid JSON payload: {e}")
            
            # Parse webhook payload
            webhook_payload = processor.parse_payload(headers, payload)
            
            if not webhook_payload:
                WEBHOOK_ERRORS.labels(platform=platform, error_type='unsupported_event').inc()
                return {'status': 'ignored', 'reason': 'Unsupported event type'}
            
            # Add to processing queue
            await self.processing_queue.put(webhook_payload)
            WEBHOOK_QUEUE_SIZE.labels(platform=platform).inc()
            
            # Cache webhook for debugging
            await self._cache_webhook(webhook_payload)
            
            WEBHOOK_REQUESTS.labels(
                platform=platform,
                event_type=webhook_payload.event_type.value,
                status='success'
            ).inc()
            
            return {
                'status': 'accepted',
                'event_type': webhook_payload.event_type.value,
                'repository': webhook_payload.repository_name,
                'timestamp': webhook_payload.timestamp.isoformat()
            }
        
        except Exception as e:
            logger.error(f"Webhook processing failed for {platform}: {e}")
            WEBHOOK_REQUESTS.labels(
                platform=platform,
                event_type='unknown',
                status='error'
            ).inc()
            
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
        
        finally:
            latency = time.time() - start_time
            WEBHOOK_LATENCY.labels(platform=platform, event_type='all').observe(latency)
    
    async def _worker(self):
        """Worker task to process webhooks from queue"""
        while True:
            try:
                # Get webhook from queue
                webhook_payload = await self.processing_queue.get()
                WEBHOOK_QUEUE_SIZE.labels(platform=webhook_payload.platform.value).dec()
                
                start_time = time.time()
                
                # Process webhook
                await self._handle_webhook(webhook_payload)
                
                # Mark task as done
                self.processing_queue.task_done()
                
                # Record processing time
                latency = time.time() - start_time
                WEBHOOK_LATENCY.labels(
                    platform=webhook_payload.platform.value,
                    event_type=webhook_payload.event_type.value
                ).observe(latency)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                WEBHOOK_ERRORS.labels(platform='unknown', error_type='worker_error').inc()
    
    async def _handle_webhook(self, webhook_payload: WebhookPayload):
        """Handle processed webhook payload"""
        try:
            # Call registered handlers
            handlers = self.event_handlers.get(webhook_payload.event_type, [])
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(webhook_payload)
                    else:
                        handler(webhook_payload)
                except Exception as e:
                    logger.error(f"Handler error for {webhook_payload.event_type.value}: {e}")
            
            # Store webhook event for analytics
            await self._store_webhook_event(webhook_payload)
            
            logger.info(f"Processed {webhook_payload.event_type.value} event for {webhook_payload.repository_name}")
            
        except Exception as e:
            logger.error(f"Webhook handling failed: {e}")
            WEBHOOK_ERRORS.labels(
                platform=webhook_payload.platform.value,
                error_type='handling_error'
            ).inc()
    
    async def _cache_webhook(self, webhook_payload: WebhookPayload):
        """Cache webhook for debugging and replay"""
        try:
            cache_key = f"webhook:{webhook_payload.platform.value}:{webhook_payload.repository_id}:{int(webhook_payload.timestamp.timestamp())}"
            
            cache_data = {
                'platform': webhook_payload.platform.value,
                'event_type': webhook_payload.event_type.value,
                'repository_name': webhook_payload.repository_name,
                'timestamp': webhook_payload.timestamp.isoformat(),
                'payload': webhook_payload.raw_payload
            }
            
            # Cache for 24 hours
            await self.redis_client.setex(
                cache_key,
                86400,
                json.dumps(cache_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache webhook: {e}")
    
    async def _store_webhook_event(self, webhook_payload: WebhookPayload):
        """Store webhook event for analytics"""
        try:
            event_key = f"webhook_events:{webhook_payload.platform.value}:{webhook_payload.event_type.value}"
            
            event_data = {
                'repository_id': webhook_payload.repository_id,
                'repository_name': webhook_payload.repository_name,
                'timestamp': webhook_payload.timestamp.isoformat(),
                'action': webhook_payload.action,
                'sender': webhook_payload.sender.get('login') if webhook_payload.sender else None
            }
            
            # Add to time-series data
            await self.redis_client.zadd(
                event_key,
                {json.dumps(event_data, default=str): webhook_payload.timestamp.timestamp()}
            )
            
            # Keep only last 1000 events per type
            await self.redis_client.zremrangebyrank(event_key, 0, -1001)
            
        except Exception as e:
            logger.error(f"Failed to store webhook event: {e}")
    
    async def get_webhook_stats(self, platform: Optional[str] = None, 
                              hours: int = 24) -> Dict[str, Any]:
        """Get webhook processing statistics"""
        try:
            stats = {
                'timestamp': datetime.utcnow().isoformat(),
                'period_hours': hours,
                'platforms': {}
            }
            
            # Calculate time range
            end_time = time.time()
            start_time = end_time - (hours * 3600)
            
            platforms = [platform] if platform else [p.value for p in WebhookPlatform]
            
            for platform_name in platforms:
                platform_stats = {
                    'total_events': 0,
                    'event_types': {},
                    'repositories': set()
                }
                
                # Get events for each event type
                for event_type in WebhookEvent:
                    event_key = f"webhook_events:{platform_name}:{event_type.value}"
                    
                    # Get events in time range
                    events = await self.redis_client.zrangebyscore(
                        event_key, start_time, end_time, withscores=True
                    )
                    
                    if events:
                        event_count = len(events)
                        platform_stats['total_events'] += event_count
                        platform_stats['event_types'][event_type.value] = event_count
                        
                        # Extract repository names
                        for event_data, _ in events:
                            try:
                                event_obj = json.loads(event_data)
                                platform_stats['repositories'].add(event_obj['repository_name'])
                            except:
                                pass
                
                # Convert set to count
                platform_stats['unique_repositories'] = len(platform_stats['repositories'])
                del platform_stats['repositories']
                
                stats['platforms'][platform_name] = platform_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get webhook stats: {e}")
            return {'error': str(e)}
    
    async def replay_webhook(self, cache_key: str) -> Dict[str, Any]:
        """Replay cached webhook for debugging"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if not cached_data:
                return {'error': 'Webhook not found in cache'}
            
            webhook_data = json.loads(cached_data)
            
            # Reconstruct webhook payload
            platform = WebhookPlatform(webhook_data['platform'])
            event_type = WebhookEvent(webhook_data['event_type'])
            
            webhook_payload = WebhookPayload(
                platform=platform,
                event_type=event_type,
                repository_id='',
                repository_name=webhook_data['repository_name'],
                repository_url='',
                timestamp=datetime.fromisoformat(webhook_data['timestamp']),
                raw_payload=webhook_data['payload']
            )
            
            # Process webhook
            await self._handle_webhook(webhook_payload)
            
            return {
                'status': 'replayed',
                'webhook': webhook_data
            }
            
        except Exception as e:
            logger.error(f"Failed to replay webhook: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'processors': {},
            'queue_size': self.processing_queue.qsize(),
            'workers': len(self.worker_tasks)
        }
        
        # Check platform processors
        for platform, processor in self.processors.items():
            health_status['processors'][platform.value] = 'configured'
        
        # Check Redis
        try:
            await self.redis_client.ping()
            health_status['redis'] = 'healthy'
        except Exception:
            health_status['redis'] = 'unhealthy'
            health_status['status'] = 'degraded'
        
        # Check worker tasks
        active_workers = sum(1 for task in self.worker_tasks if not task.done())
        if active_workers < len(self.worker_tasks):
            health_status['status'] = 'degraded'
            health_status['workers_active'] = active_workers
        
        return health_status
    
    async def close(self):
        """Close webhook processor"""
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

# Web server for webhook endpoints
class WebhookServer:
    """HTTP server for receiving webhooks"""
    
    def __init__(self, webhook_processor: WebhookProcessor, port: int = 8005):
        self.webhook_processor = webhook_processor
        self.port = port
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_post('/webhook/{platform}', self.handle_webhook)
        self.app.router.add_get('/webhook/stats', self.get_stats)
        self.app.router.add_post('/webhook/replay/{cache_key}', self.replay_webhook)
        self.app.router.add_get('/health', self.health_check)
    
    async def handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming webhook"""
        platform = request.match_info['platform']
        headers = dict(request.headers)
        payload_data = await request.read()
        
        result = await self.webhook_processor.process_webhook(platform, headers, payload_data)
        
        if result['status'] == 'error':
            return web.json_response(result, status=400)
        else:
            return web.json_response(result, status=200)
    
    async def get_stats(self, request: web.Request) -> web.Response:
        """Get webhook statistics"""
        platform = request.query.get('platform')
        hours = int(request.query.get('hours', 24))
        
        stats = await self.webhook_processor.get_webhook_stats(platform, hours)
        return web.json_response(stats)
    
    async def replay_webhook(self, request: web.Request) -> web.Response:
        """Replay cached webhook"""
        cache_key = request.match_info['cache_key']
        
        result = await self.webhook_processor.replay_webhook(cache_key)
        return web.json_response(result)
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        health = await self.webhook_processor.health_check()
        status_code = 200 if health['status'] == 'healthy' else 503
        return web.json_response(health, status=status_code)
    
    async def start(self):
        """Start webhook server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"Webhook server started on port {self.port}")
        return runner

# Configuration
DEFAULT_CONFIG = {
    'platforms': {
        'github': {
            'webhook_secret': 'your-github-webhook-secret'
        },
        'gitlab': {
            'webhook_secret': 'your-gitlab-webhook-secret'
        },
        'bitbucket': {
            'webhook_secret': 'your-bitbucket-webhook-secret'
        }
    },
    'redis': {
        'host': 'localhost',
        'port': 6379
    },
    'num_workers': 4
}

if __name__ == "__main__":
    import asyncio
    
    async def example_handler(webhook_payload: WebhookPayload):
        """Example webhook handler"""
        print(f"Received {webhook_payload.event_type.value} event for {webhook_payload.repository_name}")
        
        if webhook_payload.event_type == WebhookEvent.PUSH:
            print(f"  - {len(webhook_payload.commits)} commits pushed to {webhook_payload.branch}")
        elif webhook_payload.event_type == WebhookEvent.PULL_REQUEST:
            pr = webhook_payload.pull_request
            if pr:
                print(f"  - Pull request #{pr.get('number')}: {pr.get('title')}")
    
    async def main():
        # Initialize webhook processor
        processor = WebhookProcessor(DEFAULT_CONFIG)
        await processor.initialize()
        
        # Register event handlers
        processor.register_handler(WebhookEvent.PUSH, example_handler)
        processor.register_handler(WebhookEvent.PULL_REQUEST, example_handler)
        
        # Start webhook server
        server = WebhookServer(processor)
        runner = await server.start()
        
        try:
            # Keep server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down webhook server")
        finally:
            await runner.cleanup()
            await processor.close()
    
    asyncio.run(main())

