"""
Communication Platform Connector for Nexus Architect
Comprehensive integration with Slack, Microsoft Teams, and other communication platforms
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
from urllib.parse import urlencode, quote, urlparse
from prometheus_client import Counter, Histogram, Gauge
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLTK components
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    sentiment_analyzer = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"NLTK initialization failed: {e}")
    sentiment_analyzer = None
    lemmatizer = None
    stop_words = set()

# Metrics
COMM_REQUESTS = Counter('comm_requests_total', 'Total communication platform requests', ['platform', 'operation', 'status'])
COMM_LATENCY = Histogram('comm_latency_seconds', 'Communication platform request latency', ['platform', 'operation'])
COMM_MESSAGES_PROCESSED = Gauge('comm_messages_processed', 'Total messages processed', ['platform'])
COMM_CHANNELS_TRACKED = Gauge('comm_channels_tracked', 'Total channels tracked', ['platform'])

class CommunicationPlatform(Enum):
    """Supported communication platforms"""
    SLACK = "slack"
    MICROSOFT_TEAMS = "microsoft_teams"
    DISCORD = "discord"
    MATTERMOST = "mattermost"

class MessageType(Enum):
    """Types of messages"""
    TEXT = "text"
    FILE = "file"
    IMAGE = "image"
    THREAD_REPLY = "thread_reply"
    SYSTEM = "system"
    BOT = "bot"
    REACTION = "reaction"

class ChannelType(Enum):
    """Types of channels"""
    PUBLIC = "public"
    PRIVATE = "private"
    DIRECT_MESSAGE = "direct_message"
    GROUP_MESSAGE = "group_message"

@dataclass
class CommunicationUser:
    """Communication platform user"""
    user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    is_bot: bool = False
    is_active: bool = True
    timezone: Optional[str] = None
    status: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    platform_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Channel:
    """Communication channel"""
    channel_id: str
    platform: CommunicationPlatform
    name: str
    channel_type: ChannelType
    description: Optional[str] = None
    topic: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_archived: bool = False
    member_count: int = 0
    creator: Optional[CommunicationUser] = None
    members: List[CommunicationUser] = field(default_factory=list)
    platform_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MessageAttachment:
    """Message attachment"""
    attachment_id: str
    filename: str
    file_type: str
    size_bytes: Optional[int] = None
    url: Optional[str] = None
    download_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    mime_type: Optional[str] = None
    platform_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MessageReaction:
    """Message reaction"""
    emoji: str
    count: int
    users: List[CommunicationUser] = field(default_factory=list)

@dataclass
class Message:
    """Communication message"""
    message_id: str
    channel_id: str
    platform: CommunicationPlatform
    message_type: MessageType
    text: Optional[str] = None
    author: Optional[CommunicationUser] = None
    timestamp: Optional[datetime] = None
    edited_at: Optional[datetime] = None
    thread_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    reply_count: int = 0
    attachments: List[MessageAttachment] = field(default_factory=list)
    reactions: List[MessageReaction] = field(default_factory=list)
    mentions: List[CommunicationUser] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    platform_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conversation:
    """Conversation thread"""
    conversation_id: str
    channel_id: str
    platform: CommunicationPlatform
    title: Optional[str] = None
    participants: List[CommunicationUser] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    message_count: int = 0
    is_active: bool = True
    summary: Optional[str] = None
    key_topics: List[str] = field(default_factory=list)
    sentiment_trend: List[float] = field(default_factory=list)
    platform_specific: Dict[str, Any] = field(default_factory=dict)

class MessageAnalyzer:
    """Analyze message content for insights"""
    
    def __init__(self):
        self.sentiment_analyzer = sentiment_analyzer
        self.lemmatizer = lemmatizer
        self.stop_words = stop_words
    
    def analyze_message(self, message: Message) -> Message:
        """Analyze message content and extract insights"""
        if not message.text:
            return message
        
        try:
            # Sentiment analysis
            if self.sentiment_analyzer:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(message.text)
                message.sentiment_score = sentiment_scores['compound']
                
                if sentiment_scores['compound'] >= 0.05:
                    message.sentiment_label = 'positive'
                elif sentiment_scores['compound'] <= -0.05:
                    message.sentiment_label = 'negative'
                else:
                    message.sentiment_label = 'neutral'
            
            # Extract mentions
            message.mentions = self._extract_mentions(message.text)
            
            # Extract links
            message.links = self._extract_links(message.text)
            
            # Extract hashtags
            message.hashtags = self._extract_hashtags(message.text)
            
            # Extract topics
            message.topics = self._extract_topics(message.text)
            
            # Extract action items
            message.action_items = self._extract_action_items(message.text)
            
            # Extract decisions
            message.decisions = self._extract_decisions(message.text)
        
        except Exception as e:
            logger.warning(f"Message analysis failed: {e}")
        
        return message
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract user mentions from text"""
        # Common mention patterns: @username, @user.name, <@userid>
        mention_patterns = [
            r'@([a-zA-Z0-9._-]+)',
            r'<@([A-Z0-9]+)>',
            r'@here',
            r'@channel',
            r'@everyone'
        ]
        
        mentions = []
        for pattern in mention_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.extend(matches)
        
        return list(set(mentions))
    
    def _extract_links(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        links = re.findall(url_pattern, text)
        return links
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        hashtag_pattern = r'#([a-zA-Z0-9_]+)'
        hashtags = re.findall(hashtag_pattern, text)
        return hashtags
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text using simple keyword extraction"""
        if not self.lemmatizer:
            return []
        
        try:
            # Tokenize and clean text
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in self.stop_words]
            
            # Lemmatize words
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            
            # Simple frequency-based topic extraction
            word_freq = {}
            for word in lemmatized_words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top topics
            topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            return [topic[0] for topic in topics]
        
        except Exception:
            return []
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from text"""
        action_patterns = [
            r'(?:TODO|todo|To do|TO DO)[:]\s*(.+?)(?:\n|$)',
            r'(?:Action|ACTION)[:]\s*(.+?)(?:\n|$)',
            r'(?:Need to|needs to|should|must|will)\s+(.+?)(?:\.|!|\?|\n|$)',
            r'(?:Let\'s|let\'s|Let me|let me)\s+(.+?)(?:\.|!|\?|\n|$)',
            r'(?:I\'ll|I will|We\'ll|We will)\s+(.+?)(?:\.|!|\?|\n|$)'
        ]
        
        action_items = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            action_items.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
        return action_items[:10]  # Limit to 10 action items
    
    def _extract_decisions(self, text: str) -> List[str]:
        """Extract decisions from text"""
        decision_patterns = [
            r'(?:We decided|decided|We\'ve decided|We have decided)\s+(.+?)(?:\.|!|\?|\n|$)',
            r'(?:Decision|DECISION)[:]\s*(.+?)(?:\n|$)',
            r'(?:We agreed|agreed|We\'ve agreed)\s+(.+?)(?:\.|!|\?|\n|$)',
            r'(?:Final decision|final decision)\s*[:]\s*(.+?)(?:\n|$)',
            r'(?:We\'re going with|going with|We will go with)\s+(.+?)(?:\.|!|\?|\n|$)'
        ]
        
        decisions = []
        for pattern in decision_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            decisions.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
        return decisions[:5]  # Limit to 5 decisions

class BaseCommunicationConnector:
    """Base class for communication platform connectors"""
    
    def __init__(self, platform: CommunicationPlatform, config: Dict[str, Any]):
        self.platform = platform
        self.config = config
        self.base_url = config.get('base_url')
        self.session = None
        self.analyzer = MessageAnalyzer()
    
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
        """Make authenticated API request"""
        start_time = time.time()
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    await asyncio.sleep(retry_after)
                    return await self._make_request(method, url, **kwargs)
                
                response.raise_for_status()
                
                # Handle different content types
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    data = await response.json()
                else:
                    text_data = await response.text()
                    data = {'text': text_data}
                
                COMM_REQUESTS.labels(
                    platform=self.platform.value,
                    operation=method.lower(),
                    status='success'
                ).inc()
                
                return data
        
        except Exception as e:
            COMM_REQUESTS.labels(
                platform=self.platform.value,
                operation=method.lower(),
                status='error'
            ).inc()
            logger.error(f"Request failed for {self.platform.value}: {e}")
            raise
        
        finally:
            latency = time.time() - start_time
            COMM_LATENCY.labels(
                platform=self.platform.value,
                operation=method.lower()
            ).observe(latency)
    
    async def get_channels(self) -> List[Channel]:
        """Get all accessible channels"""
        raise NotImplementedError
    
    async def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get specific channel"""
        raise NotImplementedError
    
    async def get_messages(self, channel_id: str, **filters) -> List[Message]:
        """Get messages from a channel"""
        raise NotImplementedError
    
    async def get_conversations(self, channel_id: str, **filters) -> List[Conversation]:
        """Get conversation threads from a channel"""
        raise NotImplementedError
    
    async def search_messages(self, query: str, **filters) -> List[Message]:
        """Search messages across channels"""
        raise NotImplementedError

class SlackConnector(BaseCommunicationConnector):
    """Slack platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(CommunicationPlatform.SLACK, config)
        self.bot_token = config.get('bot_token')
        self.user_token = config.get('user_token')  # For additional permissions
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get Slack authentication headers"""
        return {
            'Authorization': f'Bearer {self.bot_token}',
            'Content-Type': 'application/json'
        }
    
    async def get_channels(self) -> List[Channel]:
        """Get all Slack channels"""
        channels = []
        
        # Get public channels
        url = f"{self.base_url}/conversations.list"
        params = {
            'types': 'public_channel,private_channel',
            'exclude_archived': 'false',
            'limit': 1000
        }
        
        data = await self._make_request('GET', url, params=params)
        
        for channel_data in data.get('channels', []):
            channel = await self._parse_slack_channel(channel_data)
            channels.append(channel)
        
        # Get direct messages
        dm_url = f"{self.base_url}/conversations.list"
        dm_params = {
            'types': 'im,mpim',
            'limit': 1000
        }
        
        dm_data = await self._make_request('GET', dm_url, params=dm_params)
        
        for dm_data in dm_data.get('channels', []):
            channel = await self._parse_slack_channel(dm_data)
            channels.append(channel)
        
        COMM_CHANNELS_TRACKED.labels(platform=self.platform.value).set(len(channels))
        return channels
    
    async def _parse_slack_channel(self, channel_data: Dict[str, Any]) -> Channel:
        """Parse Slack channel data"""
        # Determine channel type
        if channel_data.get('is_im'):
            channel_type = ChannelType.DIRECT_MESSAGE
        elif channel_data.get('is_mpim'):
            channel_type = ChannelType.GROUP_MESSAGE
        elif channel_data.get('is_private'):
            channel_type = ChannelType.PRIVATE
        else:
            channel_type = ChannelType.PUBLIC
        
        # Parse creation timestamp
        created_at = None
        if 'created' in channel_data:
            created_at = datetime.fromtimestamp(channel_data['created'], tz=timezone.utc)
        
        # Get channel members
        members = []
        if 'members' in channel_data:
            for member_id in channel_data['members']:
                # Get user info
                user_info = await self._get_user_info(member_id)
                if user_info:
                    members.append(user_info)
        
        return Channel(
            channel_id=channel_data['id'],
            platform=self.platform,
            name=channel_data.get('name', channel_data.get('id')),
            channel_type=channel_type,
            description=channel_data.get('purpose', {}).get('value'),
            topic=channel_data.get('topic', {}).get('value'),
            created_at=created_at,
            is_archived=channel_data.get('is_archived', False),
            member_count=channel_data.get('num_members', 0),
            members=members,
            platform_specific=channel_data
        )
    
    async def _get_user_info(self, user_id: str) -> Optional[CommunicationUser]:
        """Get Slack user information"""
        try:
            url = f"{self.base_url}/users.info"
            params = {'user': user_id}
            
            data = await self._make_request('GET', url, params=params)
            user_data = data.get('user', {})
            
            profile = user_data.get('profile', {})
            
            return CommunicationUser(
                user_id=user_data['id'],
                username=user_data.get('name'),
                display_name=profile.get('display_name') or profile.get('real_name'),
                email=profile.get('email'),
                avatar_url=profile.get('image_512'),
                is_bot=user_data.get('is_bot', False),
                is_active=not user_data.get('deleted', False),
                timezone=user_data.get('tz'),
                status=profile.get('status_text'),
                title=profile.get('title'),
                platform_specific=user_data
            )
        
        except Exception as e:
            logger.warning(f"Failed to get user info for {user_id}: {e}")
            return None
    
    async def get_messages(self, channel_id: str, **filters) -> List[Message]:
        """Get messages from a Slack channel"""
        url = f"{self.base_url}/conversations.history"
        params = {
            'channel': channel_id,
            'limit': filters.get('limit', 100),
            'oldest': filters.get('oldest'),
            'latest': filters.get('latest'),
            'inclusive': 'true'
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        data = await self._make_request('GET', url, params=params)
        
        messages = []
        for message_data in data.get('messages', []):
            message = await self._parse_slack_message(message_data, channel_id)
            if message:
                # Analyze message content
                message = self.analyzer.analyze_message(message)
                messages.append(message)
        
        COMM_MESSAGES_PROCESSED.labels(platform=self.platform.value).set(len(messages))
        return messages
    
    async def _parse_slack_message(self, message_data: Dict[str, Any], channel_id: str) -> Optional[Message]:
        """Parse Slack message data"""
        try:
            # Skip system messages and some bot messages
            if message_data.get('subtype') in ['channel_join', 'channel_leave', 'channel_topic', 'channel_purpose']:
                return None
            
            # Determine message type
            if message_data.get('files'):
                message_type = MessageType.FILE
            elif message_data.get('thread_ts') and message_data.get('thread_ts') != message_data.get('ts'):
                message_type = MessageType.THREAD_REPLY
            elif message_data.get('bot_id'):
                message_type = MessageType.BOT
            else:
                message_type = MessageType.TEXT
            
            # Parse timestamp
            timestamp = None
            if 'ts' in message_data:
                timestamp = datetime.fromtimestamp(float(message_data['ts']), tz=timezone.utc)
            
            # Parse edited timestamp
            edited_at = None
            if 'edited' in message_data:
                edited_at = datetime.fromtimestamp(float(message_data['edited']['ts']), tz=timezone.utc)
            
            # Get author information
            author = None
            if 'user' in message_data:
                author = await self._get_user_info(message_data['user'])
            elif 'bot_id' in message_data:
                # Handle bot messages
                author = CommunicationUser(
                    user_id=message_data['bot_id'],
                    username=message_data.get('username', 'bot'),
                    display_name=message_data.get('username', 'Bot'),
                    is_bot=True,
                    platform_specific=message_data
                )
            
            # Parse attachments
            attachments = []
            if 'files' in message_data:
                for file_data in message_data['files']:
                    attachment = MessageAttachment(
                        attachment_id=file_data['id'],
                        filename=file_data.get('name', 'unknown'),
                        file_type=file_data.get('filetype', 'unknown'),
                        size_bytes=file_data.get('size'),
                        url=file_data.get('url_private'),
                        download_url=file_data.get('url_private_download'),
                        thumbnail_url=file_data.get('thumb_360'),
                        mime_type=file_data.get('mimetype'),
                        platform_specific=file_data
                    )
                    attachments.append(attachment)
            
            # Parse reactions
            reactions = []
            if 'reactions' in message_data:
                for reaction_data in message_data['reactions']:
                    # Get user info for reactions (simplified)
                    reaction_users = []
                    for user_id in reaction_data.get('users', []):
                        user = await self._get_user_info(user_id)
                        if user:
                            reaction_users.append(user)
                    
                    reaction = MessageReaction(
                        emoji=reaction_data['name'],
                        count=reaction_data['count'],
                        users=reaction_users
                    )
                    reactions.append(reaction)
            
            return Message(
                message_id=message_data.get('client_msg_id', message_data.get('ts')),
                channel_id=channel_id,
                platform=self.platform,
                message_type=message_type,
                text=message_data.get('text'),
                author=author,
                timestamp=timestamp,
                edited_at=edited_at,
                thread_id=message_data.get('thread_ts'),
                parent_message_id=message_data.get('parent_user_id'),
                reply_count=message_data.get('reply_count', 0),
                attachments=attachments,
                reactions=reactions,
                platform_specific=message_data
            )
        
        except Exception as e:
            logger.warning(f"Failed to parse Slack message: {e}")
            return None
    
    async def search_messages(self, query: str, **filters) -> List[Message]:
        """Search messages in Slack"""
        url = f"{self.base_url}/search.messages"
        params = {
            'query': query,
            'count': filters.get('count', 100),
            'page': filters.get('page', 1)
        }
        
        data = await self._make_request('GET', url, params=params)
        
        messages = []
        for match in data.get('messages', {}).get('matches', []):
            # Parse search result message
            message = await self._parse_slack_message(match, match.get('channel', {}).get('id'))
            if message:
                message = self.analyzer.analyze_message(message)
                messages.append(message)
        
        return messages

class MicrosoftTeamsConnector(BaseCommunicationConnector):
    """Microsoft Teams platform connector"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(CommunicationPlatform.MICROSOFT_TEAMS, config)
        self.access_token = config.get('access_token')
        self.tenant_id = config.get('tenant_id')
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get Microsoft Teams authentication headers"""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
    
    async def get_channels(self) -> List[Channel]:
        """Get all Teams channels"""
        channels = []
        
        # Get all teams first
        teams_url = f"{self.base_url}/teams"
        teams_data = await self._make_request('GET', teams_url)
        
        for team in teams_data.get('value', []):
            team_id = team['id']
            
            # Get channels for this team
            channels_url = f"{self.base_url}/teams/{team_id}/channels"
            channels_data = await self._make_request('GET', channels_url)
            
            for channel_data in channels_data.get('value', []):
                channel = await self._parse_teams_channel(channel_data, team_id)
                channels.append(channel)
        
        COMM_CHANNELS_TRACKED.labels(platform=self.platform.value).set(len(channels))
        return channels
    
    async def _parse_teams_channel(self, channel_data: Dict[str, Any], team_id: str) -> Channel:
        """Parse Teams channel data"""
        # Determine channel type
        membership_type = channel_data.get('membershipType', 'standard')
        if membership_type == 'private':
            channel_type = ChannelType.PRIVATE
        else:
            channel_type = ChannelType.PUBLIC
        
        # Parse creation timestamp
        created_at = None
        if 'createdDateTime' in channel_data:
            created_at = datetime.fromisoformat(channel_data['createdDateTime'].replace('Z', '+00:00'))
        
        return Channel(
            channel_id=channel_data['id'],
            platform=self.platform,
            name=channel_data.get('displayName', ''),
            channel_type=channel_type,
            description=channel_data.get('description'),
            created_at=created_at,
            platform_specific={**channel_data, 'team_id': team_id}
        )
    
    async def get_messages(self, channel_id: str, **filters) -> List[Message]:
        """Get messages from a Teams channel"""
        # Extract team_id from channel info
        channel = await self.get_channel(channel_id)
        if not channel:
            return []
        
        team_id = channel.platform_specific.get('team_id')
        if not team_id:
            return []
        
        url = f"{self.base_url}/teams/{team_id}/channels/{channel_id}/messages"
        params = {
            '$top': filters.get('limit', 50),
            '$orderby': 'createdDateTime desc'
        }
        
        data = await self._make_request('GET', url, params=params)
        
        messages = []
        for message_data in data.get('value', []):
            message = await self._parse_teams_message(message_data, channel_id)
            if message:
                message = self.analyzer.analyze_message(message)
                messages.append(message)
        
        COMM_MESSAGES_PROCESSED.labels(platform=self.platform.value).set(len(messages))
        return messages
    
    async def _parse_teams_message(self, message_data: Dict[str, Any], channel_id: str) -> Optional[Message]:
        """Parse Teams message data"""
        try:
            # Parse timestamp
            timestamp = None
            if 'createdDateTime' in message_data:
                timestamp = datetime.fromisoformat(message_data['createdDateTime'].replace('Z', '+00:00'))
            
            # Parse author
            author = None
            if 'from' in message_data:
                from_data = message_data['from']
                user_data = from_data.get('user', {})
                
                author = CommunicationUser(
                    user_id=user_data.get('id', ''),
                    username=user_data.get('userPrincipalName'),
                    display_name=user_data.get('displayName'),
                    platform_specific=from_data
                )
            
            # Extract text content
            text_content = None
            body = message_data.get('body', {})
            if body.get('content'):
                # Teams messages can be HTML, extract text
                text_content = self._extract_text_from_html(body['content'])
            
            # Parse attachments
            attachments = []
            if 'attachments' in message_data:
                for attachment_data in message_data['attachments']:
                    attachment = MessageAttachment(
                        attachment_id=attachment_data.get('id', ''),
                        filename=attachment_data.get('name', 'unknown'),
                        file_type=attachment_data.get('contentType', 'unknown'),
                        url=attachment_data.get('contentUrl'),
                        platform_specific=attachment_data
                    )
                    attachments.append(attachment)
            
            return Message(
                message_id=message_data['id'],
                channel_id=channel_id,
                platform=self.platform,
                message_type=MessageType.TEXT,
                text=text_content,
                author=author,
                timestamp=timestamp,
                attachments=attachments,
                platform_specific=message_data
            )
        
        except Exception as e:
            logger.warning(f"Failed to parse Teams message: {e}")
            return None
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract plain text from HTML content"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(strip=True)
        except ImportError:
            # Fallback: simple regex-based HTML tag removal
            import re
            clean_text = re.sub(r'<[^>]+>', '', html_content)
            return clean_text.strip()
        except Exception:
            return html_content

class CommunicationPlatformManager:
    """Manager for all communication platform connectors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connectors: Dict[CommunicationPlatform, BaseCommunicationConnector] = {}
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize platform connectors based on configuration"""
        platforms_config = self.config.get('platforms', {})
        
        for platform_name, platform_config in platforms_config.items():
            if not platform_config.get('enabled', False):
                continue
            
            try:
                platform_type = CommunicationPlatform(platform_name)
                
                if platform_type == CommunicationPlatform.SLACK:
                    connector_class = SlackConnector
                elif platform_type == CommunicationPlatform.MICROSOFT_TEAMS:
                    connector_class = MicrosoftTeamsConnector
                else:
                    logger.warning(f"Connector not implemented for platform: {platform_name}")
                    continue
                
                self.connectors[platform_type] = connector_class(platform_config)
                logger.info(f"Initialized connector for {platform_name}")
            
            except ValueError:
                logger.error(f"Unknown platform: {platform_name}")
            except Exception as e:
                logger.error(f"Failed to initialize connector for {platform_name}: {e}")
    
    async def get_all_channels(self) -> List[Channel]:
        """Get channels from all connected platforms"""
        all_channels = []
        
        for platform_type, connector in self.connectors.items():
            try:
                async with connector:
                    channels = await connector.get_channels()
                    all_channels.extend(channels)
                    logger.info(f"Retrieved {len(channels)} channels from {platform_type.value}")
            except Exception as e:
                logger.error(f"Failed to get channels from {platform_type.value}: {e}")
        
        return all_channels
    
    async def get_all_messages(self, channel_filters: Dict[str, Any] = None, **message_filters) -> List[Message]:
        """Get messages from all connected platforms"""
        all_messages = []
        
        for platform_type, connector in self.connectors.items():
            try:
                async with connector:
                    # Get channels first
                    channels = await connector.get_channels()
                    
                    for channel in channels:
                        # Apply channel filters if specified
                        if channel_filters:
                            if 'platform' in channel_filters and channel.platform != CommunicationPlatform(channel_filters['platform']):
                                continue
                            if 'channel_type' in channel_filters and channel.channel_type != ChannelType(channel_filters['channel_type']):
                                continue
                            if 'channel_name' in channel_filters and channel_filters['channel_name'].lower() not in channel.name.lower():
                                continue
                        
                        # Get messages for this channel
                        messages = await connector.get_messages(channel.channel_id, **message_filters)
                        all_messages.extend(messages)
                        logger.info(f"Retrieved {len(messages)} messages from {channel.name} ({platform_type.value})")
            
            except Exception as e:
                logger.error(f"Failed to get messages from {platform_type.value}: {e}")
        
        return all_messages
    
    async def search_messages(self, query: str, platforms: List[CommunicationPlatform] = None) -> List[Message]:
        """Search messages across platforms"""
        all_messages = []
        target_platforms = platforms or list(self.connectors.keys())
        
        for platform_type in target_platforms:
            if platform_type not in self.connectors:
                continue
            
            connector = self.connectors[platform_type]
            try:
                async with connector:
                    if hasattr(connector, 'search_messages'):
                        messages = await connector.search_messages(query)
                        all_messages.extend(messages)
                        logger.info(f"Found {len(messages)} messages matching '{query}' in {platform_type.value}")
            except Exception as e:
                logger.error(f"Failed to search messages in {platform_type.value}: {e}")
        
        return all_messages
    
    async def analyze_team_communication(self, team_filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze team communication patterns"""
        messages = await self.get_all_messages(channel_filters=team_filters)
        
        if not messages:
            return {'error': 'No messages found'}
        
        # Basic analytics
        total_messages = len(messages)
        platforms = list(set(msg.platform.value for msg in messages))
        
        # Sentiment analysis
        sentiments = [msg.sentiment_score for msg in messages if msg.sentiment_score is not None]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Most active users
        user_activity = {}
        for msg in messages:
            if msg.author and msg.author.display_name:
                user_activity[msg.author.display_name] = user_activity.get(msg.author.display_name, 0) + 1
        
        most_active_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Common topics
        all_topics = []
        for msg in messages:
            all_topics.extend(msg.topics)
        
        topic_frequency = {}
        for topic in all_topics:
            topic_frequency[topic] = topic_frequency.get(topic, 0) + 1
        
        common_topics = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Action items and decisions
        all_action_items = []
        all_decisions = []
        for msg in messages:
            all_action_items.extend(msg.action_items)
            all_decisions.extend(msg.decisions)
        
        return {
            'summary': {
                'total_messages': total_messages,
                'platforms': platforms,
                'average_sentiment': avg_sentiment,
                'analysis_period': {
                    'start': min(msg.timestamp for msg in messages if msg.timestamp).isoformat(),
                    'end': max(msg.timestamp for msg in messages if msg.timestamp).isoformat()
                }
            },
            'user_activity': {
                'most_active_users': most_active_users,
                'total_active_users': len(user_activity)
            },
            'content_analysis': {
                'common_topics': common_topics,
                'action_items_count': len(all_action_items),
                'decisions_count': len(all_decisions),
                'sample_action_items': all_action_items[:5],
                'sample_decisions': all_decisions[:5]
            },
            'sentiment_distribution': {
                'positive': len([s for s in sentiments if s > 0.05]),
                'neutral': len([s for s in sentiments if -0.05 <= s <= 0.05]),
                'negative': len([s for s in sentiments if s < -0.05])
            }
        }
    
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
                    channels = await connector.get_channels()
                    
                    health_status['platforms'][platform_type.value] = {
                        'status': 'healthy',
                        'channels_accessible': len(channels),
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
                'slack': {
                    'enabled': True,
                    'base_url': 'https://slack.com/api',
                    'bot_token': 'xoxb-your-bot-token',
                    'user_token': 'xoxp-your-user-token'
                },
                'microsoft_teams': {
                    'enabled': True,
                    'base_url': 'https://graph.microsoft.com/v1.0',
                    'access_token': 'your-access-token',
                    'tenant_id': 'your-tenant-id'
                }
            }
        }
        
        manager = CommunicationPlatformManager(config)
        
        # Get all channels
        channels = await manager.get_all_channels()
        print(f"Found {len(channels)} channels across all platforms")
        
        # Get recent messages
        messages = await manager.get_all_messages(limit=50)
        print(f"Found {len(messages)} recent messages")
        
        # Analyze team communication
        analysis = await manager.analyze_team_communication()
        print(f"Communication analysis: {analysis['summary']}")
        
        # Health check
        health = await manager.health_check()
        print(f"Health status: {health['status']}")
    
    asyncio.run(main())

