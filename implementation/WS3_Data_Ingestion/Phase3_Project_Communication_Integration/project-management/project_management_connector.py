"""
Project Management Platform Connector for Nexus Architect
Comprehensive integration with major project management tools
"""

import os
import re
import json
import logging
import time
import asyncio
import aiohttp
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import base64
from urllib.parse import urlencode, quote
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
PM_REQUESTS = Counter('pm_requests_total', 'Total PM platform requests', ['platform', 'operation', 'status'])
PM_LATENCY = Histogram('pm_latency_seconds', 'PM platform request latency', ['platform', 'operation'])
PM_ISSUES_PROCESSED = Gauge('pm_issues_processed', 'Total issues processed', ['platform'])
PM_PROJECTS_TRACKED = Gauge('pm_projects_tracked', 'Total projects tracked', ['platform'])

class PlatformType(Enum):
    """Supported project management platforms"""
    JIRA = "jira"
    LINEAR = "linear"
    ASANA = "asana"
    TRELLO = "trello"
    AZURE_DEVOPS = "azure_devops"
    GITHUB = "github"
    GITLAB = "gitlab"

class IssueStatus(Enum):
    """Standard issue status mapping"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class IssuePriority(Enum):
    """Standard issue priority mapping"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

@dataclass
class ProjectUser:
    """Project user information"""
    user_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: Optional[str] = None
    is_active: bool = True
    platform_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectIssue:
    """Standardized project issue representation"""
    issue_id: str
    project_id: str
    platform: PlatformType
    title: str
    description: Optional[str] = None
    status: IssueStatus = IssueStatus.OPEN
    priority: IssuePriority = IssuePriority.MEDIUM
    assignee: Optional[ProjectUser] = None
    reporter: Optional[ProjectUser] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    labels: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    epic_id: Optional[str] = None
    sprint_id: Optional[str] = None
    story_points: Optional[int] = None
    time_estimate: Optional[int] = None  # seconds
    time_spent: Optional[int] = None  # seconds
    comments: List[Dict[str, Any]] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    platform_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Project:
    """Project information"""
    project_id: str
    platform: PlatformType
    name: str
    key: Optional[str] = None
    description: Optional[str] = None
    lead: Optional[ProjectUser] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True
    project_type: Optional[str] = None
    category: Optional[str] = None
    url: Optional[str] = None
    members: List[ProjectUser] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    platform_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Sprint:
    """Sprint/iteration information"""
    sprint_id: str
    project_id: str
    platform: PlatformType
    name: str
    state: str  # active, closed, future
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    goal: Optional[str] = None
    issues: List[str] = field(default_factory=list)  # issue IDs
    platform_specific: Dict[str, Any] = field(default_factory=dict)

class RateLimiter:
    """Rate limiting for API requests"""
    
    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.requests = []
        self.burst_count = 0
    
    async def acquire(self):
        """Acquire rate limit permission"""
        now = time.time()
        
        # Clean old requests
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        # Check burst limit
        if self.burst_count >= self.burst_limit:
            await asyncio.sleep(1)
            self.burst_count = 0
        
        # Check rate limit
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)
        self.burst_count += 1

class BaseProjectConnector:
    """Base class for project management platform connectors"""
    
    def __init__(self, platform: PlatformType, config: Dict[str, Any]):
        self.platform = platform
        self.config = config
        self.base_url = config.get('base_url')
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.get('rate_limit', {}).get('requests_per_minute', 60),
            burst_limit=config.get('rate_limit', {}).get('burst_limit', 10)
        )
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=await self._get_headers()
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        raise NotImplementedError
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request with rate limiting"""
        await self.rate_limiter.acquire()
        
        start_time = time.time()
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    await asyncio.sleep(retry_after)
                    return await self._make_request(method, url, **kwargs)
                
                response.raise_for_status()
                data = await response.json()
                
                PM_REQUESTS.labels(
                    platform=self.platform.value,
                    operation=method.lower(),
                    status='success'
                ).inc()
                
                return data
        
        except Exception as e:
            PM_REQUESTS.labels(
                platform=self.platform.value,
                operation=method.lower(),
                status='error'
            ).inc()
            logger.error(f"Request failed for {self.platform.value}: {e}")
            raise
        
        finally:
            latency = time.time() - start_time
            PM_LATENCY.labels(
                platform=self.platform.value,
                operation=method.lower()
            ).observe(latency)
    
    async def get_projects(self) -> List[Project]:
        """Get all accessible projects"""
        raise NotImplementedError
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get specific project"""
        raise NotImplementedError
    
    async def get_issues(self, project_id: str, **filters) -> List[ProjectIssue]:
        """Get issues for a project"""
        raise NotImplementedError
    
    async def get_issue(self, issue_id: str) -> Optional[ProjectIssue]:
        """Get specific issue"""
        raise NotImplementedError
    
    async def get_sprints(self, project_id: str) -> List[Sprint]:
        """Get sprints for a project"""
        raise NotImplementedError
    
    async def search_issues(self, query: str, **filters) -> List[ProjectIssue]:
        """Search issues across projects"""
        raise NotImplementedError

class JiraConnector(BaseProjectConnector):
    """Jira platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(PlatformType.JIRA, config)
        self.username = config.get('username')
        self.api_token = config.get('api_token')
        self.cloud = config.get('cloud', True)
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get Jira authentication headers"""
        if self.cloud:
            # Jira Cloud uses basic auth with email and API token
            credentials = base64.b64encode(f"{self.username}:{self.api_token}".encode()).decode()
            return {
                'Authorization': f'Basic {credentials}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        else:
            # Jira Server might use different auth
            return {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }
    
    async def get_projects(self) -> List[Project]:
        """Get all Jira projects"""
        url = f"{self.base_url}/rest/api/3/project"
        data = await self._make_request('GET', url)
        
        projects = []
        for project_data in data:
            project = Project(
                project_id=project_data['id'],
                platform=self.platform,
                name=project_data['name'],
                key=project_data['key'],
                description=project_data.get('description'),
                project_type=project_data.get('projectTypeKey'),
                url=project_data.get('self'),
                platform_specific=project_data
            )
            
            # Get project lead
            if 'lead' in project_data:
                lead_data = project_data['lead']
                project.lead = ProjectUser(
                    user_id=lead_data['accountId'],
                    email=lead_data.get('emailAddress'),
                    display_name=lead_data.get('displayName'),
                    avatar_url=lead_data.get('avatarUrls', {}).get('48x48'),
                    is_active=lead_data.get('active', True),
                    platform_specific=lead_data
                )
            
            projects.append(project)
        
        PM_PROJECTS_TRACKED.labels(platform=self.platform.value).set(len(projects))
        return projects
    
    async def get_issues(self, project_id: str, **filters) -> List[ProjectIssue]:
        """Get Jira issues for a project"""
        jql = f"project = {project_id}"
        
        # Add filters to JQL
        if 'status' in filters:
            jql += f" AND status = '{filters['status']}'"
        if 'assignee' in filters:
            jql += f" AND assignee = '{filters['assignee']}'"
        if 'updated_since' in filters:
            jql += f" AND updated >= '{filters['updated_since']}'"
        
        url = f"{self.base_url}/rest/api/3/search"
        params = {
            'jql': jql,
            'expand': 'changelog,comments,attachments',
            'maxResults': filters.get('max_results', 100),
            'startAt': filters.get('start_at', 0)
        }
        
        data = await self._make_request('GET', url, params=params)
        
        issues = []
        for issue_data in data.get('issues', []):
            issue = await self._parse_jira_issue(issue_data)
            issues.append(issue)
        
        PM_ISSUES_PROCESSED.labels(platform=self.platform.value).set(len(issues))
        return issues
    
    async def _parse_jira_issue(self, issue_data: Dict[str, Any]) -> ProjectIssue:
        """Parse Jira issue data into standardized format"""
        fields = issue_data['fields']
        
        # Map Jira status to standard status
        status_name = fields.get('status', {}).get('name', '').lower()
        if 'done' in status_name or 'closed' in status_name or 'resolved' in status_name:
            status = IssueStatus.DONE
        elif 'progress' in status_name or 'development' in status_name:
            status = IssueStatus.IN_PROGRESS
        elif 'review' in status_name:
            status = IssueStatus.IN_REVIEW
        else:
            status = IssueStatus.OPEN
        
        # Map Jira priority to standard priority
        priority_name = fields.get('priority', {}).get('name', '').lower()
        if 'critical' in priority_name or 'highest' in priority_name:
            priority = IssuePriority.CRITICAL
        elif 'high' in priority_name:
            priority = IssuePriority.HIGH
        elif 'medium' in priority_name:
            priority = IssuePriority.MEDIUM
        elif 'low' in priority_name or 'lowest' in priority_name:
            priority = IssuePriority.LOW
        else:
            priority = IssuePriority.NONE
        
        # Parse assignee
        assignee = None
        if fields.get('assignee'):
            assignee_data = fields['assignee']
            assignee = ProjectUser(
                user_id=assignee_data['accountId'],
                email=assignee_data.get('emailAddress'),
                display_name=assignee_data.get('displayName'),
                avatar_url=assignee_data.get('avatarUrls', {}).get('48x48'),
                is_active=assignee_data.get('active', True),
                platform_specific=assignee_data
            )
        
        # Parse reporter
        reporter = None
        if fields.get('reporter'):
            reporter_data = fields['reporter']
            reporter = ProjectUser(
                user_id=reporter_data['accountId'],
                email=reporter_data.get('emailAddress'),
                display_name=reporter_data.get('displayName'),
                avatar_url=reporter_data.get('avatarUrls', {}).get('48x48'),
                is_active=reporter_data.get('active', True),
                platform_specific=reporter_data
            )
        
        # Parse dates
        created_at = None
        if fields.get('created'):
            created_at = datetime.fromisoformat(fields['created'].replace('Z', '+00:00'))
        
        updated_at = None
        if fields.get('updated'):
            updated_at = datetime.fromisoformat(fields['updated'].replace('Z', '+00:00'))
        
        due_date = None
        if fields.get('duedate'):
            due_date = datetime.fromisoformat(fields['duedate'] + 'T00:00:00+00:00')
        
        # Parse labels and components
        labels = [label for label in fields.get('labels', [])]
        components = [comp['name'] for comp in fields.get('components', [])]
        
        # Parse comments
        comments = []
        if 'comment' in fields and 'comments' in fields['comment']:
            for comment_data in fields['comment']['comments']:
                comments.append({
                    'id': comment_data['id'],
                    'author': comment_data.get('author', {}).get('displayName'),
                    'body': comment_data.get('body'),
                    'created': comment_data.get('created'),
                    'updated': comment_data.get('updated')
                })
        
        # Parse attachments
        attachments = []
        if 'attachment' in fields:
            for attachment_data in fields['attachment']:
                attachments.append({
                    'id': attachment_data['id'],
                    'filename': attachment_data.get('filename'),
                    'size': attachment_data.get('size'),
                    'mimeType': attachment_data.get('mimeType'),
                    'content': attachment_data.get('content'),
                    'author': attachment_data.get('author', {}).get('displayName'),
                    'created': attachment_data.get('created')
                })
        
        # Parse custom fields
        custom_fields = {}
        for key, value in fields.items():
            if key.startswith('customfield_'):
                custom_fields[key] = value
        
        return ProjectIssue(
            issue_id=issue_data['id'],
            project_id=fields['project']['id'],
            platform=self.platform,
            title=fields.get('summary', ''),
            description=fields.get('description'),
            status=status,
            priority=priority,
            assignee=assignee,
            reporter=reporter,
            created_at=created_at,
            updated_at=updated_at,
            due_date=due_date,
            labels=labels,
            components=components,
            epic_id=fields.get('epic', {}).get('id') if fields.get('epic') else None,
            story_points=fields.get('customfield_10016'),  # Common story points field
            time_estimate=fields.get('timeoriginalestimate'),
            time_spent=fields.get('timespent'),
            comments=comments,
            attachments=attachments,
            custom_fields=custom_fields,
            platform_specific=issue_data
        )

class LinearConnector(BaseProjectConnector):
    """Linear platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(PlatformType.LINEAR, config)
        self.api_token = config.get('api_token')
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get Linear authentication headers"""
        return {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
    
    async def _make_graphql_request(self, query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make GraphQL request to Linear API"""
        url = f"{self.base_url}/graphql"
        payload = {
            'query': query,
            'variables': variables or {}
        }
        
        return await self._make_request('POST', url, json=payload)
    
    async def get_projects(self) -> List[Project]:
        """Get all Linear teams (projects)"""
        query = """
        query {
            teams {
                nodes {
                    id
                    name
                    description
                    key
                    createdAt
                    updatedAt
                    private
                    members {
                        nodes {
                            id
                            name
                            email
                            displayName
                            avatarUrl
                            active
                        }
                    }
                }
            }
        }
        """
        
        data = await self._make_graphql_request(query)
        teams = data.get('data', {}).get('teams', {}).get('nodes', [])
        
        projects = []
        for team_data in teams:
            # Parse team members
            members = []
            for member_data in team_data.get('members', {}).get('nodes', []):
                member = ProjectUser(
                    user_id=member_data['id'],
                    email=member_data.get('email'),
                    display_name=member_data.get('displayName') or member_data.get('name'),
                    avatar_url=member_data.get('avatarUrl'),
                    is_active=member_data.get('active', True),
                    platform_specific=member_data
                )
                members.append(member)
            
            project = Project(
                project_id=team_data['id'],
                platform=self.platform,
                name=team_data['name'],
                key=team_data.get('key'),
                description=team_data.get('description'),
                created_at=datetime.fromisoformat(team_data['createdAt'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(team_data['updatedAt'].replace('Z', '+00:00')),
                is_active=not team_data.get('private', False),
                members=members,
                platform_specific=team_data
            )
            projects.append(project)
        
        PM_PROJECTS_TRACKED.labels(platform=self.platform.value).set(len(projects))
        return projects
    
    async def get_issues(self, project_id: str, **filters) -> List[ProjectIssue]:
        """Get Linear issues for a team"""
        query = """
        query($teamId: String!, $first: Int, $after: String) {
            team(id: $teamId) {
                issues(first: $first, after: $after) {
                    nodes {
                        id
                        identifier
                        title
                        description
                        priority
                        estimate
                        createdAt
                        updatedAt
                        dueDate
                        state {
                            id
                            name
                            type
                        }
                        assignee {
                            id
                            name
                            email
                            displayName
                            avatarUrl
                            active
                        }
                        creator {
                            id
                            name
                            email
                            displayName
                            avatarUrl
                            active
                        }
                        labels {
                            nodes {
                                id
                                name
                                color
                            }
                        }
                        comments {
                            nodes {
                                id
                                body
                                createdAt
                                updatedAt
                                user {
                                    id
                                    name
                                    displayName
                                }
                            }
                        }
                        attachments {
                            nodes {
                                id
                                title
                                url
                                subtitle
                                metadata
                            }
                        }
                    }
                    pageInfo {
                        hasNextPage
                        endCursor
                    }
                }
            }
        }
        """
        
        variables = {
            'teamId': project_id,
            'first': filters.get('max_results', 100),
            'after': filters.get('after')
        }
        
        data = await self._make_graphql_request(query, variables)
        issue_nodes = data.get('data', {}).get('team', {}).get('issues', {}).get('nodes', [])
        
        issues = []
        for issue_data in issue_nodes:
            issue = await self._parse_linear_issue(issue_data, project_id)
            issues.append(issue)
        
        PM_ISSUES_PROCESSED.labels(platform=self.platform.value).set(len(issues))
        return issues
    
    async def _parse_linear_issue(self, issue_data: Dict[str, Any], project_id: str) -> ProjectIssue:
        """Parse Linear issue data into standardized format"""
        # Map Linear state to standard status
        state_type = issue_data.get('state', {}).get('type', '').lower()
        if state_type == 'completed':
            status = IssueStatus.DONE
        elif state_type == 'started':
            status = IssueStatus.IN_PROGRESS
        elif state_type == 'unstarted':
            status = IssueStatus.OPEN
        else:
            status = IssueStatus.OPEN
        
        # Map Linear priority to standard priority
        priority_value = issue_data.get('priority', 0)
        if priority_value == 1:
            priority = IssuePriority.CRITICAL
        elif priority_value == 2:
            priority = IssuePriority.HIGH
        elif priority_value == 3:
            priority = IssuePriority.MEDIUM
        elif priority_value == 4:
            priority = IssuePriority.LOW
        else:
            priority = IssuePriority.NONE
        
        # Parse assignee
        assignee = None
        if issue_data.get('assignee'):
            assignee_data = issue_data['assignee']
            assignee = ProjectUser(
                user_id=assignee_data['id'],
                email=assignee_data.get('email'),
                display_name=assignee_data.get('displayName') or assignee_data.get('name'),
                avatar_url=assignee_data.get('avatarUrl'),
                is_active=assignee_data.get('active', True),
                platform_specific=assignee_data
            )
        
        # Parse creator (reporter)
        reporter = None
        if issue_data.get('creator'):
            creator_data = issue_data['creator']
            reporter = ProjectUser(
                user_id=creator_data['id'],
                email=creator_data.get('email'),
                display_name=creator_data.get('displayName') or creator_data.get('name'),
                avatar_url=creator_data.get('avatarUrl'),
                is_active=creator_data.get('active', True),
                platform_specific=creator_data
            )
        
        # Parse dates
        created_at = datetime.fromisoformat(issue_data['createdAt'].replace('Z', '+00:00'))
        updated_at = datetime.fromisoformat(issue_data['updatedAt'].replace('Z', '+00:00'))
        
        due_date = None
        if issue_data.get('dueDate'):
            due_date = datetime.fromisoformat(issue_data['dueDate'].replace('Z', '+00:00'))
        
        # Parse labels
        labels = [label['name'] for label in issue_data.get('labels', {}).get('nodes', [])]
        
        # Parse comments
        comments = []
        for comment_data in issue_data.get('comments', {}).get('nodes', []):
            comments.append({
                'id': comment_data['id'],
                'author': comment_data.get('user', {}).get('displayName') or comment_data.get('user', {}).get('name'),
                'body': comment_data.get('body'),
                'created': comment_data.get('createdAt'),
                'updated': comment_data.get('updatedAt')
            })
        
        # Parse attachments
        attachments = []
        for attachment_data in issue_data.get('attachments', {}).get('nodes', []):
            attachments.append({
                'id': attachment_data['id'],
                'title': attachment_data.get('title'),
                'url': attachment_data.get('url'),
                'subtitle': attachment_data.get('subtitle'),
                'metadata': attachment_data.get('metadata')
            })
        
        return ProjectIssue(
            issue_id=issue_data['id'],
            project_id=project_id,
            platform=self.platform,
            title=issue_data.get('title', ''),
            description=issue_data.get('description'),
            status=status,
            priority=priority,
            assignee=assignee,
            reporter=reporter,
            created_at=created_at,
            updated_at=updated_at,
            due_date=due_date,
            labels=labels,
            story_points=issue_data.get('estimate'),
            comments=comments,
            attachments=attachments,
            platform_specific=issue_data
        )

class AsanaConnector(BaseProjectConnector):
    """Asana platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(PlatformType.ASANA, config)
        self.access_token = config.get('access_token')
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get Asana authentication headers"""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    async def get_projects(self) -> List[Project]:
        """Get all Asana projects"""
        url = f"{self.base_url}/projects"
        params = {
            'opt_fields': 'name,notes,created_at,modified_at,owner,team,members,archived,public'
        }
        
        data = await self._make_request('GET', url, params=params)
        
        projects = []
        for project_data in data.get('data', []):
            # Parse project owner
            lead = None
            if project_data.get('owner'):
                owner_data = project_data['owner']
                lead = ProjectUser(
                    user_id=owner_data['gid'],
                    display_name=owner_data.get('name'),
                    platform_specific=owner_data
                )
            
            # Parse project members
            members = []
            for member_data in project_data.get('members', []):
                member = ProjectUser(
                    user_id=member_data['gid'],
                    display_name=member_data.get('name'),
                    platform_specific=member_data
                )
                members.append(member)
            
            project = Project(
                project_id=project_data['gid'],
                platform=self.platform,
                name=project_data['name'],
                description=project_data.get('notes'),
                lead=lead,
                created_at=datetime.fromisoformat(project_data['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(project_data['modified_at'].replace('Z', '+00:00')),
                is_active=not project_data.get('archived', False),
                members=members,
                platform_specific=project_data
            )
            projects.append(project)
        
        PM_PROJECTS_TRACKED.labels(platform=self.platform.value).set(len(projects))
        return projects

class ProjectManagementConnectorManager:
    """Manager for all project management platform connectors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connectors: Dict[PlatformType, BaseProjectConnector] = {}
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize platform connectors based on configuration"""
        platforms_config = self.config.get('platforms', {})
        
        for platform_name, platform_config in platforms_config.items():
            if not platform_config.get('enabled', False):
                continue
            
            try:
                platform_type = PlatformType(platform_name)
                
                if platform_type == PlatformType.JIRA:
                    connector_class = JiraConnector
                elif platform_type == PlatformType.LINEAR:
                    connector_class = LinearConnector
                elif platform_type == PlatformType.ASANA:
                    connector_class = AsanaConnector
                else:
                    logger.warning(f"Connector not implemented for platform: {platform_name}")
                    continue
                
                self.connectors[platform_type] = connector_class(platform_config)
                logger.info(f"Initialized connector for {platform_name}")
            
            except ValueError:
                logger.error(f"Unknown platform: {platform_name}")
            except Exception as e:
                logger.error(f"Failed to initialize connector for {platform_name}: {e}")
    
    async def get_all_projects(self) -> List[Project]:
        """Get projects from all connected platforms"""
        all_projects = []
        
        for platform_type, connector in self.connectors.items():
            try:
                async with connector:
                    projects = await connector.get_projects()
                    all_projects.extend(projects)
                    logger.info(f"Retrieved {len(projects)} projects from {platform_type.value}")
            except Exception as e:
                logger.error(f"Failed to get projects from {platform_type.value}: {e}")
        
        return all_projects
    
    async def get_all_issues(self, project_filters: Dict[str, Any] = None) -> List[ProjectIssue]:
        """Get issues from all connected platforms"""
        all_issues = []
        
        for platform_type, connector in self.connectors.items():
            try:
                async with connector:
                    # Get projects first
                    projects = await connector.get_projects()
                    
                    for project in projects:
                        # Apply project filters if specified
                        if project_filters:
                            if 'platform' in project_filters and project.platform != PlatformType(project_filters['platform']):
                                continue
                            if 'project_name' in project_filters and project_filters['project_name'].lower() not in project.name.lower():
                                continue
                        
                        # Get issues for this project
                        issues = await connector.get_issues(project.project_id)
                        all_issues.extend(issues)
                        logger.info(f"Retrieved {len(issues)} issues from {project.name} ({platform_type.value})")
            
            except Exception as e:
                logger.error(f"Failed to get issues from {platform_type.value}: {e}")
        
        return all_issues
    
    async def search_issues(self, query: str, platforms: List[PlatformType] = None) -> List[ProjectIssue]:
        """Search issues across platforms"""
        all_issues = []
        target_platforms = platforms or list(self.connectors.keys())
        
        for platform_type in target_platforms:
            if platform_type not in self.connectors:
                continue
            
            connector = self.connectors[platform_type]
            try:
                async with connector:
                    if hasattr(connector, 'search_issues'):
                        issues = await connector.search_issues(query)
                        all_issues.extend(issues)
                        logger.info(f"Found {len(issues)} issues matching '{query}' in {platform_type.value}")
            except Exception as e:
                logger.error(f"Failed to search issues in {platform_type.value}: {e}")
        
        return all_issues
    
    async def get_platform_statistics(self) -> Dict[str, Any]:
        """Get statistics from all connected platforms"""
        stats = {
            'platforms_connected': len(self.connectors),
            'platforms': {}
        }
        
        for platform_type, connector in self.connectors.items():
            try:
                async with connector:
                    projects = await connector.get_projects()
                    total_issues = 0
                    
                    for project in projects[:5]:  # Sample first 5 projects
                        issues = await connector.get_issues(project.project_id, max_results=10)
                        total_issues += len(issues)
                    
                    stats['platforms'][platform_type.value] = {
                        'projects': len(projects),
                        'sample_issues': total_issues,
                        'status': 'connected'
                    }
            except Exception as e:
                stats['platforms'][platform_type.value] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all connectors"""
        health_status = {
            'status': 'healthy',
            'platforms': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        for platform_type, connector in self.connectors.items():
            try:
                async with connector:
                    # Try to get a small amount of data
                    projects = await connector.get_projects()
                    
                    health_status['platforms'][platform_type.value] = {
                        'status': 'healthy',
                        'projects_accessible': len(projects),
                        'response_time': 'normal'
                    }
            except Exception as e:
                health_status['platforms'][platform_type.value] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['status'] = 'degraded'
        
        return health_status

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example configuration
        config = {
            'platforms': {
                'jira': {
                    'enabled': True,
                    'base_url': 'https://your-domain.atlassian.net',
                    'username': 'your-email@company.com',
                    'api_token': 'your-api-token',
                    'cloud': True,
                    'rate_limit': {
                        'requests_per_minute': 100,
                        'burst_limit': 20
                    }
                },
                'linear': {
                    'enabled': True,
                    'base_url': 'https://api.linear.app',
                    'api_token': 'your-linear-api-token',
                    'rate_limit': {
                        'requests_per_minute': 60,
                        'burst_limit': 10
                    }
                }
            }
        }
        
        manager = ProjectManagementConnectorManager(config)
        
        # Get all projects
        projects = await manager.get_all_projects()
        print(f"Found {len(projects)} projects across all platforms")
        
        # Get all issues
        issues = await manager.get_all_issues()
        print(f"Found {len(issues)} issues across all platforms")
        
        # Get statistics
        stats = await manager.get_platform_statistics()
        print(f"Platform statistics: {stats}")
        
        # Health check
        health = await manager.health_check()
        print(f"Health status: {health['status']}")
    
    asyncio.run(main())

