"""
Documentation Platform Manager for Nexus Architect
Comprehensive integration with major documentation platforms
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
from datetime import datetime, timezone
import tempfile
from pathlib import Path
import aiofiles
import aiohttp
from urllib.parse import urljoin, urlparse
import base64
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
PLATFORM_REQUESTS = Counter('documentation_platform_requests_total', 'Total platform API requests', ['platform', 'operation', 'status'])
PLATFORM_REQUEST_LATENCY = Histogram('documentation_platform_request_latency_seconds', 'Platform API request latency', ['platform', 'operation'])
DOCUMENTS_PROCESSED = Counter('documents_processed_total', 'Total documents processed', ['platform', 'format'])
CONTENT_EXTRACTION_LATENCY = Histogram('content_extraction_latency_seconds', 'Content extraction latency', ['format'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentFormat(Enum):
    """Supported document formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    TXT = "txt"
    RTF = "rtf"
    ODT = "odt"
    CONFLUENCE = "confluence"
    NOTION = "notion"

class DocumentPlatform(Enum):
    """Supported documentation platforms"""
    CONFLUENCE = "confluence"
    SHAREPOINT = "sharepoint"
    NOTION = "notion"
    GITBOOK = "gitbook"
    WIKI = "wiki"
    GITHUB_WIKI = "github_wiki"
    GITLAB_WIKI = "gitlab_wiki"
    CUSTOM = "custom"

@dataclass
class DocumentMetadata:
    """Document metadata information"""
    id: str
    title: str
    platform: DocumentPlatform
    format: DocumentFormat
    url: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: Optional[str] = None
    parent_id: Optional[str] = None
    space_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    permissions: Dict[str, Any] = field(default_factory=dict)
    size_bytes: Optional[int] = None
    language: Optional[str] = None
    content_type: Optional[str] = None

@dataclass
class DocumentContent:
    """Document content and extracted information"""
    metadata: DocumentMetadata
    raw_content: str
    extracted_text: str
    structured_content: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PlatformCredentials:
    """Platform authentication credentials"""
    platform: DocumentPlatform
    auth_type: str  # token, oauth, basic, api_key
    credentials: Dict[str, str]
    base_url: Optional[str] = None
    api_version: Optional[str] = None

class ConfluenceConnector:
    """Confluence platform connector"""
    
    def __init__(self, credentials: PlatformCredentials):
        self.credentials = credentials
        self.base_url = credentials.base_url
        self.session = None
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
    
    async def initialize(self):
        """Initialize the connector"""
        self.session = aiohttp.ClientSession(
            headers=self._get_auth_headers(),
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def close(self):
        """Close the connector"""
        if self.session:
            await self.session.close()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if self.credentials.auth_type == "token":
            return {
                "Authorization": f"Bearer {self.credentials.credentials['token']}",
                "Content-Type": "application/json"
            }
        elif self.credentials.auth_type == "basic":
            username = self.credentials.credentials['username']
            password = self.credentials.credentials['password']
            auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
            return {
                "Authorization": f"Basic {auth_string}",
                "Content-Type": "application/json"
            }
        else:
            raise ValueError(f"Unsupported auth type: {self.credentials.auth_type}")
    
    async def get_spaces(self) -> List[Dict[str, Any]]:
        """Get all accessible spaces"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter:
                url = f"{self.base_url}/rest/api/space"
                params = {
                    "limit": 500,
                    "expand": "description,homepage,metadata"
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        spaces = data.get('results', [])
                        
                        PLATFORM_REQUESTS.labels(
                            platform='confluence',
                            operation='get_spaces',
                            status='success'
                        ).inc()
                        
                        return spaces
                    else:
                        logger.error(f"Failed to get Confluence spaces: {response.status}")
                        PLATFORM_REQUESTS.labels(
                            platform='confluence',
                            operation='get_spaces',
                            status='error'
                        ).inc()
                        return []
        
        except Exception as e:
            logger.error(f"Error getting Confluence spaces: {e}")
            PLATFORM_REQUESTS.labels(
                platform='confluence',
                operation='get_spaces',
                status='error'
            ).inc()
            return []
        
        finally:
            latency = time.time() - start_time
            PLATFORM_REQUEST_LATENCY.labels(
                platform='confluence',
                operation='get_spaces'
            ).observe(latency)
    
    async def get_pages(self, space_key: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get pages from a space"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter:
                url = f"{self.base_url}/rest/api/content"
                params = {
                    "spaceKey": space_key,
                    "type": "page",
                    "limit": limit,
                    "expand": "version,space,body.storage,metadata.labels,children.page"
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pages = data.get('results', [])
                        
                        PLATFORM_REQUESTS.labels(
                            platform='confluence',
                            operation='get_pages',
                            status='success'
                        ).inc()
                        
                        return pages
                    else:
                        logger.error(f"Failed to get Confluence pages: {response.status}")
                        PLATFORM_REQUESTS.labels(
                            platform='confluence',
                            operation='get_pages',
                            status='error'
                        ).inc()
                        return []
        
        except Exception as e:
            logger.error(f"Error getting Confluence pages: {e}")
            PLATFORM_REQUESTS.labels(
                platform='confluence',
                operation='get_pages',
                status='error'
            ).inc()
            return []
        
        finally:
            latency = time.time() - start_time
            PLATFORM_REQUEST_LATENCY.labels(
                platform='confluence',
                operation='get_pages'
            ).observe(latency)
    
    async def get_page_content(self, page_id: str) -> Optional[DocumentContent]:
        """Get detailed page content"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter:
                url = f"{self.base_url}/rest/api/content/{page_id}"
                params = {
                    "expand": "version,space,body.storage,body.view,metadata.labels,children.attachment"
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract metadata
                        metadata = DocumentMetadata(
                            id=data['id'],
                            title=data['title'],
                            platform=DocumentPlatform.CONFLUENCE,
                            format=DocumentFormat.CONFLUENCE,
                            url=f"{self.base_url}/pages/viewpage.action?pageId={page_id}",
                            created_at=datetime.fromisoformat(data['version']['when'].replace('Z', '+00:00')),
                            updated_at=datetime.fromisoformat(data['version']['when'].replace('Z', '+00:00')),
                            version=str(data['version']['number']),
                            space_id=data['space']['key'],
                            tags=[label['name'] for label in data.get('metadata', {}).get('labels', {}).get('results', [])]
                        )
                        
                        # Extract content
                        raw_content = data.get('body', {}).get('storage', {}).get('value', '')
                        extracted_text = self._extract_text_from_confluence_storage(raw_content)
                        
                        # Extract structured content
                        structured_content = {
                            'headings': self._extract_headings(raw_content),
                            'lists': self._extract_lists(raw_content),
                            'code_blocks': self._extract_code_blocks(raw_content),
                            'macros': self._extract_macros(raw_content)
                        }
                        
                        # Extract links and attachments
                        links = self._extract_links(raw_content)
                        attachments = await self._get_page_attachments(page_id)
                        
                        content = DocumentContent(
                            metadata=metadata,
                            raw_content=raw_content,
                            extracted_text=extracted_text,
                            structured_content=structured_content,
                            attachments=attachments,
                            links=links
                        )
                        
                        PLATFORM_REQUESTS.labels(
                            platform='confluence',
                            operation='get_page_content',
                            status='success'
                        ).inc()
                        
                        DOCUMENTS_PROCESSED.labels(
                            platform='confluence',
                            format='confluence'
                        ).inc()
                        
                        return content
                    
                    else:
                        logger.error(f"Failed to get Confluence page content: {response.status}")
                        PLATFORM_REQUESTS.labels(
                            platform='confluence',
                            operation='get_page_content',
                            status='error'
                        ).inc()
                        return None
        
        except Exception as e:
            logger.error(f"Error getting Confluence page content: {e}")
            PLATFORM_REQUESTS.labels(
                platform='confluence',
                operation='get_page_content',
                status='error'
            ).inc()
            return None
        
        finally:
            latency = time.time() - start_time
            PLATFORM_REQUEST_LATENCY.labels(
                platform='confluence',
                operation='get_page_content'
            ).observe(latency)
    
    def _extract_text_from_confluence_storage(self, storage_content: str) -> str:
        """Extract plain text from Confluence storage format"""
        # Remove XML tags but preserve content
        text = re.sub(r'<[^>]+>', ' ', storage_content)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """Extract headings from Confluence content"""
        headings = []
        # Match Confluence heading tags
        heading_pattern = r'<h([1-6])[^>]*>(.*?)</h[1-6]>'
        matches = re.finditer(heading_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            level = int(match.group(1))
            text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            headings.append({
                'level': level,
                'text': text,
                'position': match.start()
            })
        
        return headings
    
    def _extract_lists(self, content: str) -> List[Dict[str, Any]]:
        """Extract lists from Confluence content"""
        lists = []
        
        # Extract unordered lists
        ul_pattern = r'<ul[^>]*>(.*?)</ul>'
        ul_matches = re.finditer(ul_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in ul_matches:
            list_content = match.group(1)
            items = re.findall(r'<li[^>]*>(.*?)</li>', list_content, re.IGNORECASE | re.DOTALL)
            items = [re.sub(r'<[^>]+>', '', item).strip() for item in items]
            
            lists.append({
                'type': 'unordered',
                'items': items,
                'position': match.start()
            })
        
        # Extract ordered lists
        ol_pattern = r'<ol[^>]*>(.*?)</ol>'
        ol_matches = re.finditer(ol_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in ol_matches:
            list_content = match.group(1)
            items = re.findall(r'<li[^>]*>(.*?)</li>', list_content, re.IGNORECASE | re.DOTALL)
            items = [re.sub(r'<[^>]+>', '', item).strip() for item in items]
            
            lists.append({
                'type': 'ordered',
                'items': items,
                'position': match.start()
            })
        
        return lists
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from Confluence content"""
        code_blocks = []
        
        # Extract code macro blocks
        code_pattern = r'<ac:structured-macro[^>]*ac:name="code"[^>]*>(.*?)</ac:structured-macro>'
        matches = re.finditer(code_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            macro_content = match.group(1)
            
            # Extract language parameter
            language_match = re.search(r'<ac:parameter[^>]*ac:name="language"[^>]*>(.*?)</ac:parameter>', macro_content)
            language = language_match.group(1) if language_match else None
            
            # Extract code content
            code_match = re.search(r'<ac:plain-text-body><!\[CDATA\[(.*?)\]\]></ac:plain-text-body>', macro_content, re.DOTALL)
            code = code_match.group(1) if code_match else ''
            
            code_blocks.append({
                'language': language,
                'code': code,
                'position': match.start()
            })
        
        return code_blocks
    
    def _extract_macros(self, content: str) -> List[Dict[str, Any]]:
        """Extract Confluence macros"""
        macros = []
        
        macro_pattern = r'<ac:structured-macro[^>]*ac:name="([^"]*)"[^>]*>(.*?)</ac:structured-macro>'
        matches = re.finditer(macro_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            macro_name = match.group(1)
            macro_content = match.group(2)
            
            # Extract parameters
            param_pattern = r'<ac:parameter[^>]*ac:name="([^"]*)"[^>]*>(.*?)</ac:parameter>'
            params = {}
            param_matches = re.finditer(param_pattern, macro_content)
            for param_match in param_matches:
                params[param_match.group(1)] = param_match.group(2)
            
            macros.append({
                'name': macro_name,
                'parameters': params,
                'position': match.start()
            })
        
        return macros
    
    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract links from Confluence content"""
        links = []
        
        # Extract regular links
        link_pattern = r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        matches = re.finditer(link_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            url = match.group(1)
            text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            
            links.append({
                'url': url,
                'text': text,
                'type': 'external' if url.startswith('http') else 'internal'
            })
        
        # Extract Confluence page links
        page_link_pattern = r'<ac:link[^>]*><ri:page[^>]*ri:content-title="([^"]*)"[^>]*\/><ac:plain-text-link-body><!\[CDATA\[(.*?)\]\]></ac:plain-text-link-body></ac:link>'
        page_matches = re.finditer(page_link_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in page_matches:
            page_title = match.group(1)
            link_text = match.group(2)
            
            links.append({
                'url': f"confluence://page/{page_title}",
                'text': link_text,
                'type': 'confluence_page'
            })
        
        return links
    
    async def _get_page_attachments(self, page_id: str) -> List[Dict[str, Any]]:
        """Get page attachments"""
        try:
            url = f"{self.base_url}/rest/api/content/{page_id}/child/attachment"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    attachments = []
                    
                    for attachment in data.get('results', []):
                        attachments.append({
                            'id': attachment['id'],
                            'title': attachment['title'],
                            'mediaType': attachment.get('metadata', {}).get('mediaType'),
                            'fileSize': attachment.get('metadata', {}).get('fileSize'),
                            'download_url': f"{self.base_url}{attachment['_links']['download']}"
                        })
                    
                    return attachments
                else:
                    return []
        
        except Exception as e:
            logger.error(f"Error getting page attachments: {e}")
            return []

class SharePointConnector:
    """SharePoint platform connector using Microsoft Graph API"""
    
    def __init__(self, credentials: PlatformCredentials):
        self.credentials = credentials
        self.base_url = "https://graph.microsoft.com/v1.0"
        self.session = None
        self.rate_limiter = asyncio.Semaphore(10)
    
    async def initialize(self):
        """Initialize the connector"""
        self.session = aiohttp.ClientSession(
            headers=self._get_auth_headers(),
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def close(self):
        """Close the connector"""
        if self.session:
            await self.session.close()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        return {
            "Authorization": f"Bearer {self.credentials.credentials['access_token']}",
            "Content-Type": "application/json"
        }
    
    async def get_sites(self) -> List[Dict[str, Any]]:
        """Get SharePoint sites"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter:
                url = f"{self.base_url}/sites"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        sites = data.get('value', [])
                        
                        PLATFORM_REQUESTS.labels(
                            platform='sharepoint',
                            operation='get_sites',
                            status='success'
                        ).inc()
                        
                        return sites
                    else:
                        logger.error(f"Failed to get SharePoint sites: {response.status}")
                        PLATFORM_REQUESTS.labels(
                            platform='sharepoint',
                            operation='get_sites',
                            status='error'
                        ).inc()
                        return []
        
        except Exception as e:
            logger.error(f"Error getting SharePoint sites: {e}")
            PLATFORM_REQUESTS.labels(
                platform='sharepoint',
                operation='get_sites',
                status='error'
            ).inc()
            return []
        
        finally:
            latency = time.time() - start_time
            PLATFORM_REQUEST_LATENCY.labels(
                platform='sharepoint',
                operation='get_sites'
            ).observe(latency)
    
    async def get_site_pages(self, site_id: str) -> List[Dict[str, Any]]:
        """Get pages from a SharePoint site"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter:
                url = f"{self.base_url}/sites/{site_id}/pages"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        pages = data.get('value', [])
                        
                        PLATFORM_REQUESTS.labels(
                            platform='sharepoint',
                            operation='get_pages',
                            status='success'
                        ).inc()
                        
                        return pages
                    else:
                        logger.error(f"Failed to get SharePoint pages: {response.status}")
                        PLATFORM_REQUESTS.labels(
                            platform='sharepoint',
                            operation='get_pages',
                            status='error'
                        ).inc()
                        return []
        
        except Exception as e:
            logger.error(f"Error getting SharePoint pages: {e}")
            PLATFORM_REQUESTS.labels(
                platform='sharepoint',
                operation='get_pages',
                status='error'
            ).inc()
            return []
        
        finally:
            latency = time.time() - start_time
            PLATFORM_REQUEST_LATENCY.labels(
                platform='sharepoint',
                operation='get_pages'
            ).observe(latency)
    
    async def get_document_libraries(self, site_id: str) -> List[Dict[str, Any]]:
        """Get document libraries from a site"""
        try:
            async with self.rate_limiter:
                url = f"{self.base_url}/sites/{site_id}/drives"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('value', [])
                    else:
                        return []
        
        except Exception as e:
            logger.error(f"Error getting document libraries: {e}")
            return []
    
    async def get_drive_items(self, drive_id: str, folder_path: str = "/") -> List[Dict[str, Any]]:
        """Get items from a document library"""
        try:
            async with self.rate_limiter:
                if folder_path == "/":
                    url = f"{self.base_url}/drives/{drive_id}/root/children"
                else:
                    url = f"{self.base_url}/drives/{drive_id}/root:/{folder_path}:/children"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('value', [])
                    else:
                        return []
        
        except Exception as e:
            logger.error(f"Error getting drive items: {e}")
            return []

class NotionConnector:
    """Notion platform connector"""
    
    def __init__(self, credentials: PlatformCredentials):
        self.credentials = credentials
        self.base_url = "https://api.notion.com/v1"
        self.session = None
        self.rate_limiter = asyncio.Semaphore(3)  # Notion has strict rate limits
    
    async def initialize(self):
        """Initialize the connector"""
        self.session = aiohttp.ClientSession(
            headers=self._get_auth_headers(),
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def close(self):
        """Close the connector"""
        if self.session:
            await self.session.close()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        return {
            "Authorization": f"Bearer {self.credentials.credentials['token']}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
    
    async def search_pages(self, query: str = "", page_size: int = 100) -> List[Dict[str, Any]]:
        """Search for pages in Notion"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter:
                url = f"{self.base_url}/search"
                payload = {
                    "query": query,
                    "page_size": page_size,
                    "filter": {
                        "property": "object",
                        "value": "page"
                    }
                }
                
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        pages = data.get('results', [])
                        
                        PLATFORM_REQUESTS.labels(
                            platform='notion',
                            operation='search_pages',
                            status='success'
                        ).inc()
                        
                        return pages
                    else:
                        logger.error(f"Failed to search Notion pages: {response.status}")
                        PLATFORM_REQUESTS.labels(
                            platform='notion',
                            operation='search_pages',
                            status='error'
                        ).inc()
                        return []
        
        except Exception as e:
            logger.error(f"Error searching Notion pages: {e}")
            PLATFORM_REQUESTS.labels(
                platform='notion',
                operation='search_pages',
                status='error'
            ).inc()
            return []
        
        finally:
            latency = time.time() - start_time
            PLATFORM_REQUEST_LATENCY.labels(
                platform='notion',
                operation='search_pages'
            ).observe(latency)
    
    async def get_page_content(self, page_id: str) -> Optional[DocumentContent]:
        """Get Notion page content"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter:
                # Get page metadata
                page_url = f"{self.base_url}/pages/{page_id}"
                async with self.session.get(page_url) as response:
                    if response.status != 200:
                        return None
                    
                    page_data = await response.json()
                
                # Get page blocks (content)
                blocks_url = f"{self.base_url}/blocks/{page_id}/children"
                async with self.session.get(blocks_url) as response:
                    if response.status != 200:
                        return None
                    
                    blocks_data = await response.json()
                
                # Extract metadata
                metadata = DocumentMetadata(
                    id=page_data['id'],
                    title=self._extract_notion_title(page_data),
                    platform=DocumentPlatform.NOTION,
                    format=DocumentFormat.NOTION,
                    url=page_data['url'],
                    created_at=datetime.fromisoformat(page_data['created_time'].replace('Z', '+00:00')),
                    updated_at=datetime.fromisoformat(page_data['last_edited_time'].replace('Z', '+00:00'))
                )
                
                # Extract content from blocks
                extracted_text, structured_content = self._process_notion_blocks(blocks_data.get('results', []))
                
                content = DocumentContent(
                    metadata=metadata,
                    raw_content=json.dumps(blocks_data, indent=2),
                    extracted_text=extracted_text,
                    structured_content=structured_content
                )
                
                PLATFORM_REQUESTS.labels(
                    platform='notion',
                    operation='get_page_content',
                    status='success'
                ).inc()
                
                DOCUMENTS_PROCESSED.labels(
                    platform='notion',
                    format='notion'
                ).inc()
                
                return content
        
        except Exception as e:
            logger.error(f"Error getting Notion page content: {e}")
            PLATFORM_REQUESTS.labels(
                platform='notion',
                operation='get_page_content',
                status='error'
            ).inc()
            return None
        
        finally:
            latency = time.time() - start_time
            PLATFORM_REQUEST_LATENCY.labels(
                platform='notion',
                operation='get_page_content'
            ).observe(latency)
    
    def _extract_notion_title(self, page_data: Dict[str, Any]) -> str:
        """Extract title from Notion page data"""
        properties = page_data.get('properties', {})
        
        # Look for title property
        for prop_name, prop_data in properties.items():
            if prop_data.get('type') == 'title':
                title_array = prop_data.get('title', [])
                if title_array:
                    return ''.join([item.get('plain_text', '') for item in title_array])
        
        return "Untitled"
    
    def _process_notion_blocks(self, blocks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """Process Notion blocks to extract text and structure"""
        extracted_text = []
        structured_content = {
            'headings': [],
            'paragraphs': [],
            'lists': [],
            'code_blocks': [],
            'tables': [],
            'images': []
        }
        
        for block in blocks:
            block_type = block.get('type')
            block_content = block.get(block_type, {})
            
            if block_type in ['paragraph', 'heading_1', 'heading_2', 'heading_3']:
                text = self._extract_rich_text(block_content.get('rich_text', []))
                extracted_text.append(text)
                
                if block_type.startswith('heading'):
                    level = int(block_type.split('_')[1])
                    structured_content['headings'].append({
                        'level': level,
                        'text': text
                    })
                else:
                    structured_content['paragraphs'].append(text)
            
            elif block_type in ['bulleted_list_item', 'numbered_list_item']:
                text = self._extract_rich_text(block_content.get('rich_text', []))
                extracted_text.append(f"• {text}")
                
                list_type = 'unordered' if block_type == 'bulleted_list_item' else 'ordered'
                structured_content['lists'].append({
                    'type': list_type,
                    'text': text
                })
            
            elif block_type == 'code':
                code_text = self._extract_rich_text(block_content.get('rich_text', []))
                language = block_content.get('language', 'plain')
                
                structured_content['code_blocks'].append({
                    'language': language,
                    'code': code_text
                })
                
                extracted_text.append(f"```{language}\n{code_text}\n```")
            
            elif block_type == 'image':
                image_data = block_content.get('file', {}) or block_content.get('external', {})
                if image_data:
                    structured_content['images'].append({
                        'url': image_data.get('url'),
                        'caption': self._extract_rich_text(block_content.get('caption', []))
                    })
        
        return '\n'.join(extracted_text), structured_content
    
    def _extract_rich_text(self, rich_text_array: List[Dict[str, Any]]) -> str:
        """Extract plain text from Notion rich text array"""
        return ''.join([item.get('plain_text', '') for item in rich_text_array])

class DocumentationPlatformManager:
    """Main manager for all documentation platforms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connectors: Dict[DocumentPlatform, Any] = {}
        self.initialized = False
        
        # Start metrics server
        start_http_server(8007)
    
    async def initialize(self):
        """Initialize all configured platform connectors"""
        logger.info("Initializing documentation platform connectors...")
        
        for platform_config in self.config.get('platforms', []):
            platform = DocumentPlatform(platform_config['platform'])
            credentials = PlatformCredentials(**platform_config)
            
            if platform == DocumentPlatform.CONFLUENCE:
                connector = ConfluenceConnector(credentials)
            elif platform == DocumentPlatform.SHAREPOINT:
                connector = SharePointConnector(credentials)
            elif platform == DocumentPlatform.NOTION:
                connector = NotionConnector(credentials)
            else:
                logger.warning(f"Unsupported platform: {platform}")
                continue
            
            await connector.initialize()
            self.connectors[platform] = connector
            logger.info(f"Initialized {platform.value} connector")
        
        self.initialized = True
        logger.info("All documentation platform connectors initialized")
    
    async def close(self):
        """Close all connectors"""
        for connector in self.connectors.values():
            await connector.close()
    
    async def discover_documents(self, platform: DocumentPlatform) -> List[DocumentMetadata]:
        """Discover all documents from a platform"""
        if not self.initialized:
            await self.initialize()
        
        connector = self.connectors.get(platform)
        if not connector:
            logger.error(f"No connector available for platform: {platform}")
            return []
        
        documents = []
        
        try:
            if platform == DocumentPlatform.CONFLUENCE:
                spaces = await connector.get_spaces()
                for space in spaces:
                    pages = await connector.get_pages(space['key'])
                    for page in pages:
                        metadata = DocumentMetadata(
                            id=page['id'],
                            title=page['title'],
                            platform=platform,
                            format=DocumentFormat.CONFLUENCE,
                            url=f"{connector.base_url}/pages/viewpage.action?pageId={page['id']}",
                            space_id=space['key'],
                            created_at=datetime.fromisoformat(page['version']['when'].replace('Z', '+00:00')),
                            updated_at=datetime.fromisoformat(page['version']['when'].replace('Z', '+00:00')),
                            version=str(page['version']['number'])
                        )
                        documents.append(metadata)
            
            elif platform == DocumentPlatform.SHAREPOINT:
                sites = await connector.get_sites()
                for site in sites:
                    pages = await connector.get_site_pages(site['id'])
                    for page in pages:
                        metadata = DocumentMetadata(
                            id=page['id'],
                            title=page['title'],
                            platform=platform,
                            format=DocumentFormat.HTML,
                            url=page['webUrl'],
                            created_at=datetime.fromisoformat(page['createdDateTime'].replace('Z', '+00:00')),
                            updated_at=datetime.fromisoformat(page['lastModifiedDateTime'].replace('Z', '+00:00'))
                        )
                        documents.append(metadata)
            
            elif platform == DocumentPlatform.NOTION:
                pages = await connector.search_pages()
                for page in pages:
                    metadata = DocumentMetadata(
                        id=page['id'],
                        title=connector._extract_notion_title(page),
                        platform=platform,
                        format=DocumentFormat.NOTION,
                        url=page['url'],
                        created_at=datetime.fromisoformat(page['created_time'].replace('Z', '+00:00')),
                        updated_at=datetime.fromisoformat(page['last_edited_time'].replace('Z', '+00:00'))
                    )
                    documents.append(metadata)
        
        except Exception as e:
            logger.error(f"Error discovering documents from {platform}: {e}")
        
        logger.info(f"Discovered {len(documents)} documents from {platform.value}")
        return documents
    
    async def get_document_content(self, platform: DocumentPlatform, document_id: str) -> Optional[DocumentContent]:
        """Get content for a specific document"""
        if not self.initialized:
            await self.initialize()
        
        connector = self.connectors.get(platform)
        if not connector:
            logger.error(f"No connector available for platform: {platform}")
            return None
        
        return await connector.get_page_content(document_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all connectors"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'platforms': {}
        }
        
        for platform, connector in self.connectors.items():
            try:
                # Simple connectivity test
                if hasattr(connector, 'session') and connector.session:
                    health_status['platforms'][platform.value] = 'healthy'
                else:
                    health_status['platforms'][platform.value] = 'disconnected'
            except Exception as e:
                health_status['platforms'][platform.value] = f'error: {str(e)}'
        
        # Overall status
        if any(status != 'healthy' for status in health_status['platforms'].values()):
            health_status['status'] = 'degraded'
        
        return health_status

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example configuration
        config = {
            'platforms': [
                {
                    'platform': 'confluence',
                    'auth_type': 'basic',
                    'credentials': {
                        'username': 'user@example.com',
                        'password': 'api_token'
                    },
                    'base_url': 'https://company.atlassian.net/wiki'
                },
                {
                    'platform': 'notion',
                    'auth_type': 'token',
                    'credentials': {
                        'token': 'notion_integration_token'
                    }
                }
            ]
        }
        
        manager = DocumentationPlatformManager(config)
        
        try:
            await manager.initialize()
            
            # Discover documents
            for platform in [DocumentPlatform.CONFLUENCE, DocumentPlatform.NOTION]:
                if platform in manager.connectors:
                    documents = await manager.discover_documents(platform)
                    print(f"Found {len(documents)} documents in {platform.value}")
                    
                    # Get content for first document
                    if documents:
                        content = await manager.get_document_content(platform, documents[0].id)
                        if content:
                            print(f"Retrieved content for: {content.metadata.title}")
            
            # Health check
            health = await manager.health_check()
            print(f"Health status: {health}")
        
        finally:
            await manager.close()
    
    asyncio.run(main())

