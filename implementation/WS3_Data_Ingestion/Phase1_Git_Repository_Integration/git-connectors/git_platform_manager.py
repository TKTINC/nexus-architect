"""
Git Platform Manager for Nexus Architect
Provides unified interface for multiple Git platforms (GitHub, GitLab, Bitbucket, Azure DevOps)
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
import base64
from urllib.parse import urljoin, quote
import jwt
from cryptography.hazmat.primitives import serialization
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis
import aioredis

# Metrics
GIT_REQUESTS = Counter('git_requests_total', 'Total Git API requests', ['platform', 'endpoint', 'status'])
GIT_LATENCY = Histogram('git_request_latency_seconds', 'Git API request latency', ['platform', 'endpoint'])
RATE_LIMIT_REMAINING = Gauge('git_rate_limit_remaining', 'Remaining API rate limit', ['platform'])
REPOSITORIES_PROCESSED = Counter('repositories_processed_total', 'Total repositories processed', ['platform'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitPlatform(Enum):
    """Supported Git platforms"""
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    AZURE_DEVOPS = "azure_devops"

class AuthType(Enum):
    """Authentication types"""
    TOKEN = "token"
    OAUTH = "oauth"
    APP = "app"
    SSH_KEY = "ssh_key"

@dataclass
class GitCredentials:
    """Git platform credentials"""
    platform: GitPlatform
    auth_type: AuthType
    token: Optional[str] = None
    app_id: Optional[str] = None
    private_key: Optional[str] = None
    installation_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

@dataclass
class Repository:
    """Repository information"""
    id: str
    name: str
    full_name: str
    platform: GitPlatform
    url: str
    clone_url: str
    ssh_url: str
    description: Optional[str] = None
    language: Optional[str] = None
    languages: Dict[str, int] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    size: int = 0
    default_branch: str = "main"
    is_private: bool = False
    is_fork: bool = False
    is_archived: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    pushed_at: Optional[datetime] = None
    owner: Dict[str, Any] = field(default_factory=dict)
    license: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Commit:
    """Commit information"""
    sha: str
    message: str
    author: Dict[str, Any]
    committer: Dict[str, Any]
    timestamp: datetime
    url: str
    added_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    removed_files: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)
    parents: List[str] = field(default_factory=list)

@dataclass
class Branch:
    """Branch information"""
    name: str
    sha: str
    protected: bool = False
    default: bool = False
    url: str = ""

@dataclass
class PullRequest:
    """Pull request information"""
    id: str
    number: int
    title: str
    description: str
    state: str
    author: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    merged_at: Optional[datetime] = None
    source_branch: str = ""
    target_branch: str = ""
    url: str = ""
    commits: int = 0
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0

class GitHubConnector:
    """GitHub API connector"""
    
    def __init__(self, credentials: GitCredentials):
        self.credentials = credentials
        self.base_url = "https://api.github.com"
        self.session = None
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = time.time() + 3600
    
    async def initialize(self):
        """Initialize the connector"""
        self.session = aiohttp.ClientSession()
        await self._setup_authentication()
    
    async def _setup_authentication(self):
        """Setup authentication headers"""
        if self.credentials.auth_type == AuthType.TOKEN:
            self.session.headers.update({
                'Authorization': f'token {self.credentials.token}',
                'Accept': 'application/vnd.github.v3+json'
            })
        elif self.credentials.auth_type == AuthType.APP:
            # GitHub App authentication
            jwt_token = self._generate_jwt()
            self.session.headers.update({
                'Authorization': f'Bearer {jwt_token}',
                'Accept': 'application/vnd.github.v3+json'
            })
    
    def _generate_jwt(self) -> str:
        """Generate JWT for GitHub App authentication"""
        now = int(time.time())
        payload = {
            'iat': now,
            'exp': now + 600,  # 10 minutes
            'iss': self.credentials.app_id
        }
        
        private_key = serialization.load_pem_private_key(
            self.credentials.private_key.encode(),
            password=None
        )
        
        return jwt.encode(payload, private_key, algorithm='RS256')
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request"""
        url = urljoin(self.base_url, endpoint)
        
        start_time = time.time()
        
        try:
            # Check rate limit
            await self._check_rate_limit()
            
            async with self.session.request(method, url, **kwargs) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', time.time() + 3600))
                
                RATE_LIMIT_REMAINING.labels(platform='github').set(self.rate_limit_remaining)
                
                if response.status == 200:
                    data = await response.json()
                    GIT_REQUESTS.labels(platform='github', endpoint=endpoint, status='success').inc()
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"GitHub API error {response.status}: {error_text}")
                    GIT_REQUESTS.labels(platform='github', endpoint=endpoint, status='error').inc()
                    raise Exception(f"GitHub API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"GitHub API request failed: {e}")
            GIT_REQUESTS.labels(platform='github', endpoint=endpoint, status='error').inc()
            raise
        
        finally:
            latency = time.time() - start_time
            GIT_LATENCY.labels(platform='github', endpoint=endpoint).observe(latency)
    
    async def _check_rate_limit(self):
        """Check and handle rate limiting"""
        if self.rate_limit_remaining < 100:
            wait_time = max(0, self.rate_limit_reset - time.time())
            if wait_time > 0:
                logger.warning(f"GitHub rate limit low, waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
    
    async def get_repositories(self, org: Optional[str] = None, user: Optional[str] = None) -> List[Repository]:
        """Get repositories for organization or user"""
        repositories = []
        
        if org:
            endpoint = f"/orgs/{org}/repos"
        elif user:
            endpoint = f"/users/{user}/repos"
        else:
            endpoint = "/user/repos"
        
        page = 1
        while True:
            params = {'page': page, 'per_page': 100}
            data = await self._make_request('GET', endpoint, params=params)
            
            if not data:
                break
            
            for repo_data in data:
                repo = self._parse_repository(repo_data)
                repositories.append(repo)
                REPOSITORIES_PROCESSED.labels(platform='github').inc()
            
            if len(data) < 100:
                break
            
            page += 1
        
        return repositories
    
    def _parse_repository(self, data: Dict[str, Any]) -> Repository:
        """Parse repository data from GitHub API"""
        return Repository(
            id=str(data['id']),
            name=data['name'],
            full_name=data['full_name'],
            platform=GitPlatform.GITHUB,
            url=data['html_url'],
            clone_url=data['clone_url'],
            ssh_url=data['ssh_url'],
            description=data.get('description'),
            language=data.get('language'),
            topics=data.get('topics', []),
            stars=data.get('stargazers_count', 0),
            forks=data.get('forks_count', 0),
            watchers=data.get('watchers_count', 0),
            size=data.get('size', 0),
            default_branch=data.get('default_branch', 'main'),
            is_private=data.get('private', False),
            is_fork=data.get('fork', False),
            is_archived=data.get('archived', False),
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00')),
            pushed_at=datetime.fromisoformat(data['pushed_at'].replace('Z', '+00:00')) if data.get('pushed_at') else None,
            owner={
                'login': data['owner']['login'],
                'id': data['owner']['id'],
                'type': data['owner']['type']
            },
            license=data.get('license'),
            metadata=data
        )
    
    async def get_repository_languages(self, repo: Repository) -> Dict[str, int]:
        """Get programming languages used in repository"""
        endpoint = f"/repos/{repo.full_name}/languages"
        return await self._make_request('GET', endpoint)
    
    async def get_commits(self, repo: Repository, since: Optional[datetime] = None, 
                         until: Optional[datetime] = None, branch: Optional[str] = None) -> List[Commit]:
        """Get commits for repository"""
        endpoint = f"/repos/{repo.full_name}/commits"
        params = {'per_page': 100}
        
        if since:
            params['since'] = since.isoformat()
        if until:
            params['until'] = until.isoformat()
        if branch:
            params['sha'] = branch
        
        commits = []
        page = 1
        
        while True:
            params['page'] = page
            data = await self._make_request('GET', endpoint, params=params)
            
            if not data:
                break
            
            for commit_data in data:
                commit = self._parse_commit(commit_data)
                commits.append(commit)
            
            if len(data) < 100:
                break
            
            page += 1
        
        return commits
    
    def _parse_commit(self, data: Dict[str, Any]) -> Commit:
        """Parse commit data from GitHub API"""
        return Commit(
            sha=data['sha'],
            message=data['commit']['message'],
            author={
                'name': data['commit']['author']['name'],
                'email': data['commit']['author']['email'],
                'login': data['author']['login'] if data.get('author') else None
            },
            committer={
                'name': data['commit']['committer']['name'],
                'email': data['commit']['committer']['email'],
                'login': data['committer']['login'] if data.get('committer') else None
            },
            timestamp=datetime.fromisoformat(data['commit']['author']['date'].replace('Z', '+00:00')),
            url=data['html_url'],
            parents=[parent['sha'] for parent in data.get('parents', [])]
        )
    
    async def get_branches(self, repo: Repository) -> List[Branch]:
        """Get branches for repository"""
        endpoint = f"/repos/{repo.full_name}/branches"
        data = await self._make_request('GET', endpoint)
        
        branches = []
        for branch_data in data:
            branch = Branch(
                name=branch_data['name'],
                sha=branch_data['commit']['sha'],
                protected=branch_data.get('protected', False),
                default=(branch_data['name'] == repo.default_branch)
            )
            branches.append(branch)
        
        return branches
    
    async def get_pull_requests(self, repo: Repository, state: str = 'all') -> List[PullRequest]:
        """Get pull requests for repository"""
        endpoint = f"/repos/{repo.full_name}/pulls"
        params = {'state': state, 'per_page': 100}
        
        pull_requests = []
        page = 1
        
        while True:
            params['page'] = page
            data = await self._make_request('GET', endpoint, params=params)
            
            if not data:
                break
            
            for pr_data in data:
                pr = self._parse_pull_request(pr_data)
                pull_requests.append(pr)
            
            if len(data) < 100:
                break
            
            page += 1
        
        return pull_requests
    
    def _parse_pull_request(self, data: Dict[str, Any]) -> PullRequest:
        """Parse pull request data from GitHub API"""
        return PullRequest(
            id=str(data['id']),
            number=data['number'],
            title=data['title'],
            description=data.get('body', ''),
            state=data['state'],
            author={
                'login': data['user']['login'],
                'id': data['user']['id']
            },
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00')),
            merged_at=datetime.fromisoformat(data['merged_at'].replace('Z', '+00:00')) if data.get('merged_at') else None,
            source_branch=data['head']['ref'],
            target_branch=data['base']['ref'],
            url=data['html_url'],
            commits=data.get('commits', 0),
            additions=data.get('additions', 0),
            deletions=data.get('deletions', 0),
            changed_files=data.get('changed_files', 0)
        )
    
    async def get_file_content(self, repo: Repository, path: str, ref: Optional[str] = None) -> Optional[str]:
        """Get file content from repository"""
        endpoint = f"/repos/{repo.full_name}/contents/{quote(path)}"
        params = {}
        
        if ref:
            params['ref'] = ref
        
        try:
            data = await self._make_request('GET', endpoint, params=params)
            
            if data.get('encoding') == 'base64':
                content = base64.b64decode(data['content']).decode('utf-8')
                return content
            else:
                return data.get('content')
        
        except Exception as e:
            logger.error(f"Failed to get file content for {path}: {e}")
            return None
    
    async def close(self):
        """Close the connector"""
        if self.session:
            await self.session.close()

class GitLabConnector:
    """GitLab API connector"""
    
    def __init__(self, credentials: GitCredentials, base_url: str = "https://gitlab.com"):
        self.credentials = credentials
        self.base_url = f"{base_url}/api/v4"
        self.session = None
        self.rate_limit_remaining = 2000
    
    async def initialize(self):
        """Initialize the connector"""
        self.session = aiohttp.ClientSession()
        await self._setup_authentication()
    
    async def _setup_authentication(self):
        """Setup authentication headers"""
        if self.credentials.auth_type == AuthType.TOKEN:
            self.session.headers.update({
                'Authorization': f'Bearer {self.credentials.token}',
                'Content-Type': 'application/json'
            })
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request"""
        url = urljoin(self.base_url, endpoint)
        
        start_time = time.time()
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('RateLimit-Remaining', 2000))
                RATE_LIMIT_REMAINING.labels(platform='gitlab').set(self.rate_limit_remaining)
                
                if response.status == 200:
                    data = await response.json()
                    GIT_REQUESTS.labels(platform='gitlab', endpoint=endpoint, status='success').inc()
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"GitLab API error {response.status}: {error_text}")
                    GIT_REQUESTS.labels(platform='gitlab', endpoint=endpoint, status='error').inc()
                    raise Exception(f"GitLab API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"GitLab API request failed: {e}")
            GIT_REQUESTS.labels(platform='gitlab', endpoint=endpoint, status='error').inc()
            raise
        
        finally:
            latency = time.time() - start_time
            GIT_LATENCY.labels(platform='gitlab', endpoint=endpoint).observe(latency)
    
    async def get_repositories(self, group: Optional[str] = None) -> List[Repository]:
        """Get repositories for group or user"""
        repositories = []
        
        if group:
            endpoint = f"/groups/{group}/projects"
        else:
            endpoint = "/projects"
        
        params = {'per_page': 100, 'membership': True}
        page = 1
        
        while True:
            params['page'] = page
            data = await self._make_request('GET', endpoint, params=params)
            
            if not data:
                break
            
            for repo_data in data:
                repo = self._parse_repository(repo_data)
                repositories.append(repo)
                REPOSITORIES_PROCESSED.labels(platform='gitlab').inc()
            
            if len(data) < 100:
                break
            
            page += 1
        
        return repositories
    
    def _parse_repository(self, data: Dict[str, Any]) -> Repository:
        """Parse repository data from GitLab API"""
        return Repository(
            id=str(data['id']),
            name=data['name'],
            full_name=data['path_with_namespace'],
            platform=GitPlatform.GITLAB,
            url=data['web_url'],
            clone_url=data['http_url_to_repo'],
            ssh_url=data['ssh_url_to_repo'],
            description=data.get('description'),
            topics=data.get('tag_list', []),
            stars=data.get('star_count', 0),
            forks=data.get('forks_count', 0),
            default_branch=data.get('default_branch', 'main'),
            is_private=data.get('visibility') == 'private',
            is_archived=data.get('archived', False),
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(data['last_activity_at'].replace('Z', '+00:00')),
            owner={
                'login': data['namespace']['path'],
                'id': data['namespace']['id'],
                'type': data['namespace']['kind']
            },
            metadata=data
        )
    
    async def close(self):
        """Close the connector"""
        if self.session:
            await self.session.close()

class BitbucketConnector:
    """Bitbucket API connector"""
    
    def __init__(self, credentials: GitCredentials):
        self.credentials = credentials
        self.base_url = "https://api.bitbucket.org/2.0"
        self.session = None
    
    async def initialize(self):
        """Initialize the connector"""
        self.session = aiohttp.ClientSession()
        await self._setup_authentication()
    
    async def _setup_authentication(self):
        """Setup authentication headers"""
        if self.credentials.auth_type == AuthType.TOKEN:
            self.session.headers.update({
                'Authorization': f'Bearer {self.credentials.token}',
                'Accept': 'application/json'
            })
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request"""
        url = urljoin(self.base_url, endpoint)
        
        start_time = time.time()
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    data = await response.json()
                    GIT_REQUESTS.labels(platform='bitbucket', endpoint=endpoint, status='success').inc()
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Bitbucket API error {response.status}: {error_text}")
                    GIT_REQUESTS.labels(platform='bitbucket', endpoint=endpoint, status='error').inc()
                    raise Exception(f"Bitbucket API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Bitbucket API request failed: {e}")
            GIT_REQUESTS.labels(platform='bitbucket', endpoint=endpoint, status='error').inc()
            raise
        
        finally:
            latency = time.time() - start_time
            GIT_LATENCY.labels(platform='bitbucket', endpoint=endpoint).observe(latency)
    
    async def close(self):
        """Close the connector"""
        if self.session:
            await self.session.close()

class AzureDevOpsConnector:
    """Azure DevOps API connector"""
    
    def __init__(self, credentials: GitCredentials, organization: str):
        self.credentials = credentials
        self.organization = organization
        self.base_url = f"https://dev.azure.com/{organization}/_apis"
        self.session = None
    
    async def initialize(self):
        """Initialize the connector"""
        self.session = aiohttp.ClientSession()
        await self._setup_authentication()
    
    async def _setup_authentication(self):
        """Setup authentication headers"""
        if self.credentials.auth_type == AuthType.TOKEN:
            auth_string = base64.b64encode(f":{self.credentials.token}".encode()).decode()
            self.session.headers.update({
                'Authorization': f'Basic {auth_string}',
                'Accept': 'application/json'
            })
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request"""
        url = urljoin(self.base_url, endpoint)
        
        start_time = time.time()
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    data = await response.json()
                    GIT_REQUESTS.labels(platform='azure_devops', endpoint=endpoint, status='success').inc()
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Azure DevOps API error {response.status}: {error_text}")
                    GIT_REQUESTS.labels(platform='azure_devops', endpoint=endpoint, status='error').inc()
                    raise Exception(f"Azure DevOps API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Azure DevOps API request failed: {e}")
            GIT_REQUESTS.labels(platform='azure_devops', endpoint=endpoint, status='error').inc()
            raise
        
        finally:
            latency = time.time() - start_time
            GIT_LATENCY.labels(platform='azure_devops', endpoint=endpoint).observe(latency)
    
    async def close(self):
        """Close the connector"""
        if self.session:
            await self.session.close()

class GitPlatformManager:
    """Unified manager for all Git platforms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connectors: Dict[GitPlatform, Any] = {}
        self.redis_client = None
        self.cache_ttl = config.get('cache_ttl', 3600)
        
        # Start metrics server
        start_http_server(8003)
    
    async def initialize(self):
        """Initialize all configured connectors"""
        # Initialize Redis for caching
        redis_config = self.config.get('redis', {})
        self.redis_client = aioredis.from_url(
            f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}"
        )
        
        # Initialize platform connectors
        for platform_config in self.config.get('platforms', []):
            platform = GitPlatform(platform_config['platform'])
            credentials = GitCredentials(**platform_config['credentials'])
            
            if platform == GitPlatform.GITHUB:
                connector = GitHubConnector(credentials)
            elif platform == GitPlatform.GITLAB:
                connector = GitLabConnector(credentials, platform_config.get('base_url', 'https://gitlab.com'))
            elif platform == GitPlatform.BITBUCKET:
                connector = BitbucketConnector(credentials)
            elif platform == GitPlatform.AZURE_DEVOPS:
                connector = AzureDevOpsConnector(credentials, platform_config['organization'])
            else:
                continue
            
            await connector.initialize()
            self.connectors[platform] = connector
        
        logger.info(f"Initialized {len(self.connectors)} Git platform connectors")
    
    async def get_all_repositories(self) -> List[Repository]:
        """Get repositories from all configured platforms"""
        all_repositories = []
        
        for platform, connector in self.connectors.items():
            try:
                logger.info(f"Fetching repositories from {platform.value}")
                repositories = await connector.get_repositories()
                all_repositories.extend(repositories)
                logger.info(f"Fetched {len(repositories)} repositories from {platform.value}")
            except Exception as e:
                logger.error(f"Failed to fetch repositories from {platform.value}: {e}")
        
        return all_repositories
    
    async def get_repository_details(self, repo: Repository) -> Dict[str, Any]:
        """Get detailed information for a repository"""
        connector = self.connectors.get(repo.platform)
        if not connector:
            raise ValueError(f"No connector available for platform {repo.platform}")
        
        # Check cache first
        cache_key = f"repo_details:{repo.platform.value}:{repo.id}"
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        details = {
            'repository': repo,
            'languages': {},
            'branches': [],
            'recent_commits': [],
            'pull_requests': []
        }
        
        try:
            # Get programming languages
            if hasattr(connector, 'get_repository_languages'):
                details['languages'] = await connector.get_repository_languages(repo)
            
            # Get branches
            if hasattr(connector, 'get_branches'):
                details['branches'] = await connector.get_branches(repo)
            
            # Get recent commits (last 30 days)
            if hasattr(connector, 'get_commits'):
                since = datetime.utcnow() - timedelta(days=30)
                details['recent_commits'] = await connector.get_commits(repo, since=since)
            
            # Get pull requests
            if hasattr(connector, 'get_pull_requests'):
                details['pull_requests'] = await connector.get_pull_requests(repo)
            
            # Cache the results
            await self._cache_data(cache_key, details)
            
        except Exception as e:
            logger.error(f"Failed to get details for repository {repo.full_name}: {e}")
        
        return details
    
    async def get_file_content(self, repo: Repository, path: str, ref: Optional[str] = None) -> Optional[str]:
        """Get file content from repository"""
        connector = self.connectors.get(repo.platform)
        if not connector:
            raise ValueError(f"No connector available for platform {repo.platform}")
        
        if hasattr(connector, 'get_file_content'):
            return await connector.get_file_content(repo, path, ref)
        
        return None
    
    async def _get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        try:
            cached_data = await self.redis_client.get(f"git:{key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None
    
    async def _cache_data(self, key: str, data: Dict[str, Any]):
        """Cache data with TTL"""
        try:
            await self.redis_client.setex(
                f"git:{key}",
                self.cache_ttl,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all connectors"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'platforms': {}
        }
        
        for platform, connector in self.connectors.items():
            try:
                # Simple health check - try to make a basic API call
                if platform == GitPlatform.GITHUB and hasattr(connector, '_make_request'):
                    await connector._make_request('GET', '/user')
                
                health_status['platforms'][platform.value] = 'healthy'
            except Exception as e:
                health_status['platforms'][platform.value] = f'unhealthy: {str(e)}'
                health_status['status'] = 'degraded'
        
        # Check Redis
        try:
            await self.redis_client.ping()
            health_status['redis'] = 'healthy'
        except Exception:
            health_status['redis'] = 'unhealthy'
            health_status['status'] = 'degraded'
        
        return health_status
    
    async def close(self):
        """Close all connectors"""
        for connector in self.connectors.values():
            await connector.close()
        
        if self.redis_client:
            await self.redis_client.close()

# Configuration
DEFAULT_CONFIG = {
    'platforms': [
        {
            'platform': 'github',
            'credentials': {
                'platform': 'github',
                'auth_type': 'token',
                'token': 'your-github-token'
            }
        }
    ],
    'redis': {
        'host': 'localhost',
        'port': 6379
    },
    'cache_ttl': 3600
}

if __name__ == "__main__":
    import asyncio
    
    async def main():
        manager = GitPlatformManager(DEFAULT_CONFIG)
        await manager.initialize()
        
        # Example usage
        repositories = await manager.get_all_repositories()
        print(f"Found {len(repositories)} repositories")
        
        if repositories:
            repo = repositories[0]
            details = await manager.get_repository_details(repo)
            print(f"Repository details: {details}")
        
        # Health check
        health = await manager.health_check()
        print(f"Health: {health}")
        
        await manager.close()
    
    asyncio.run(main())

