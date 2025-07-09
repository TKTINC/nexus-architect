"""
Nexus Architect Graph Construction Pipeline
Automated knowledge extraction and graph population from various data sources
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import re

import aiohttp
import asyncpg
from neo4j import GraphDatabase
import yaml
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import openai
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Failed to download NLTK data")

class DataSourceType(str, Enum):
    CONFLUENCE = "confluence"
    JIRA = "jira"
    GITHUB = "github"
    GITLAB = "gitlab"
    SLACK = "slack"
    EMAIL = "email"
    DATABASE = "database"
    API_DOCUMENTATION = "api_documentation"
    CODE_REPOSITORY = "code_repository"
    MONITORING_LOGS = "monitoring_logs"
    INCIDENT_REPORTS = "incident_reports"
    ARCHITECTURE_DOCS = "architecture_docs"

class ExtractionType(str, Enum):
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"

@dataclass
class DataSource:
    id: str
    name: str
    type: DataSourceType
    connection_config: Dict[str, Any]
    extraction_config: Dict[str, Any]
    last_processed: Optional[datetime] = None
    status: str = "active"

@dataclass
class ExtractedEntity:
    id: str
    type: str
    name: str
    properties: Dict[str, Any]
    confidence: float
    source: str
    extracted_at: datetime

@dataclass
class ExtractedRelationship:
    id: str
    type: str
    source_entity: str
    target_entity: str
    properties: Dict[str, Any]
    confidence: float
    source: str
    extracted_at: datetime

class GraphConstructionPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.neo4j_driver = GraphDatabase.driver(
            config["neo4j"]["uri"],
            auth=(config["neo4j"]["user"], config["neo4j"]["password"])
        )
        
        # Initialize NLP models
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_pipeline = pipeline("ner", 
                                   model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                   aggregation_strategy="simple")
        self.relation_pipeline = pipeline("text-classification",
                                         model="microsoft/DialoGPT-medium")
        
        # Initialize AI clients
        self.openai_client = openai.AsyncOpenAI(api_key=config.get("openai_api_key"))
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=config.get("anthropic_api_key"))
        
        # Data sources
        self.data_sources: Dict[str, DataSource] = {}
        
        # Processing statistics
        self.stats = {
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "documents_processed": 0,
            "errors": 0
        }
    
    async def register_data_source(self, data_source: DataSource):
        """Register a new data source for processing"""
        self.data_sources[data_source.id] = data_source
        logger.info(f"Registered data source: {data_source.name} ({data_source.type})")
    
    async def process_all_sources(self):
        """Process all registered data sources"""
        logger.info("Starting processing of all data sources...")
        
        tasks = []
        for source_id, source in self.data_sources.items():
            task = asyncio.create_task(self.process_data_source(source))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        logger.info(f"Data source processing completed. Success: {successful}, Failed: {failed}")
        return results
    
    async def process_data_source(self, source: DataSource) -> Dict[str, Any]:
        """Process a single data source"""
        logger.info(f"Processing data source: {source.name}")
        
        try:
            # Extract data based on source type
            if source.type == DataSourceType.CONFLUENCE:
                data = await self._extract_confluence_data(source)
            elif source.type == DataSourceType.JIRA:
                data = await self._extract_jira_data(source)
            elif source.type == DataSourceType.GITHUB:
                data = await self._extract_github_data(source)
            elif source.type == DataSourceType.CODE_REPOSITORY:
                data = await self._extract_code_repository_data(source)
            elif source.type == DataSourceType.API_DOCUMENTATION:
                data = await self._extract_api_documentation_data(source)
            elif source.type == DataSourceType.MONITORING_LOGS:
                data = await self._extract_monitoring_logs_data(source)
            else:
                logger.warning(f"Unsupported data source type: {source.type}")
                return {"status": "skipped", "reason": "unsupported_type"}
            
            # Process extracted data
            entities, relationships = await self._process_extracted_data(data, source)
            
            # Store in knowledge graph
            await self._store_in_graph(entities, relationships)
            
            # Update source status
            source.last_processed = datetime.utcnow()
            
            return {
                "status": "success",
                "entities_count": len(entities),
                "relationships_count": len(relationships)
            }
            
        except Exception as e:
            logger.error(f"Error processing data source {source.name}: {e}")
            self.stats["errors"] += 1
            return {"status": "error", "error": str(e)}
    
    async def _extract_confluence_data(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract data from Confluence"""
        config = source.connection_config
        base_url = config["base_url"]
        username = config["username"]
        api_token = config["api_token"]
        
        documents = []
        
        async with aiohttp.ClientSession() as session:
            # Get spaces
            spaces_url = f"{base_url}/rest/api/content"
            auth = aiohttp.BasicAuth(username, api_token)
            
            async with session.get(spaces_url, auth=auth) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for page in data.get("results", []):
                        # Extract page content
                        page_url = f"{base_url}/rest/api/content/{page['id']}?expand=body.storage,metadata.labels"
                        
                        async with session.get(page_url, auth=auth) as page_response:
                            if page_response.status == 200:
                                page_data = await page_response.json()
                                
                                documents.append({
                                    "id": page["id"],
                                    "title": page["title"],
                                    "content": page_data.get("body", {}).get("storage", {}).get("value", ""),
                                    "type": "confluence_page",
                                    "url": f"{base_url}/pages/viewpage.action?pageId={page['id']}",
                                    "labels": [label["name"] for label in page_data.get("metadata", {}).get("labels", {}).get("results", [])],
                                    "created_date": page.get("history", {}).get("createdDate"),
                                    "last_modified": page.get("version", {}).get("when")
                                })
        
        logger.info(f"Extracted {len(documents)} documents from Confluence")
        return documents
    
    async def _extract_jira_data(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract data from Jira"""
        config = source.connection_config
        base_url = config["base_url"]
        username = config["username"]
        api_token = config["api_token"]
        
        issues = []
        
        async with aiohttp.ClientSession() as session:
            # Search for issues
            search_url = f"{base_url}/rest/api/2/search"
            auth = aiohttp.BasicAuth(username, api_token)
            
            jql = config.get("jql", "project is not EMPTY ORDER BY created DESC")
            params = {
                "jql": jql,
                "maxResults": config.get("max_results", 1000),
                "fields": "summary,description,issuetype,status,priority,assignee,reporter,created,updated,components,labels"
            }
            
            async with session.get(search_url, auth=auth, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for issue in data.get("issues", []):
                        fields = issue["fields"]
                        
                        issues.append({
                            "id": issue["key"],
                            "title": fields["summary"],
                            "description": fields.get("description", ""),
                            "type": "jira_issue",
                            "issue_type": fields["issuetype"]["name"],
                            "status": fields["status"]["name"],
                            "priority": fields.get("priority", {}).get("name"),
                            "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
                            "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None,
                            "created_date": fields["created"],
                            "updated_date": fields["updated"],
                            "components": [comp["name"] for comp in fields.get("components", [])],
                            "labels": fields.get("labels", []),
                            "url": f"{base_url}/browse/{issue['key']}"
                        })
        
        logger.info(f"Extracted {len(issues)} issues from Jira")
        return issues
    
    async def _extract_github_data(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract data from GitHub repositories"""
        config = source.connection_config
        token = config["token"]
        repositories = config["repositories"]
        
        data = []
        
        headers = {"Authorization": f"token {token}"}
        
        async with aiohttp.ClientSession(headers=headers) as session:
            for repo in repositories:
                # Get repository information
                repo_url = f"https://api.github.com/repos/{repo}"
                
                async with session.get(repo_url) as response:
                    if response.status == 200:
                        repo_data = await response.json()
                        
                        data.append({
                            "id": str(repo_data["id"]),
                            "name": repo_data["name"],
                            "full_name": repo_data["full_name"],
                            "description": repo_data.get("description", ""),
                            "type": "github_repository",
                            "language": repo_data.get("language"),
                            "topics": repo_data.get("topics", []),
                            "created_date": repo_data["created_at"],
                            "updated_date": repo_data["updated_at"],
                            "url": repo_data["html_url"],
                            "clone_url": repo_data["clone_url"]
                        })
                
                # Get README content
                readme_url = f"https://api.github.com/repos/{repo}/readme"
                
                async with session.get(readme_url) as response:
                    if response.status == 200:
                        readme_data = await response.json()
                        
                        # Decode base64 content
                        import base64
                        content = base64.b64decode(readme_data["content"]).decode("utf-8")
                        
                        data.append({
                            "id": f"{repo}_readme",
                            "name": f"{repo} README",
                            "content": content,
                            "type": "github_readme",
                            "repository": repo,
                            "url": readme_data["html_url"]
                        })
        
        logger.info(f"Extracted {len(data)} items from GitHub")
        return data
    
    async def _extract_code_repository_data(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract data from code repositories (static analysis)"""
        config = source.connection_config
        repository_path = config["path"]
        
        data = []
        
        # Analyze code structure
        for root, dirs, files in os.walk(repository_path):
            for file in files:
                if file.endswith(('.py', '.js', '.java', '.cpp', '.cs', '.go', '.rs')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract basic information
                        relative_path = os.path.relpath(file_path, repository_path)
                        
                        data.append({
                            "id": f"file_{relative_path.replace('/', '_')}",
                            "name": file,
                            "path": relative_path,
                            "content": content[:10000],  # Limit content size
                            "type": "source_code_file",
                            "language": self._detect_language(file),
                            "size": len(content),
                            "lines": content.count('\n') + 1
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to read file {file_path}: {e}")
        
        logger.info(f"Extracted {len(data)} code files")
        return data
    
    async def _extract_api_documentation_data(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract data from API documentation"""
        config = source.connection_config
        documentation_urls = config["urls"]
        
        data = []
        
        async with aiohttp.ClientSession() as session:
            for url in documentation_urls:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            # Parse HTML content
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract API endpoints and documentation
                            data.append({
                                "id": f"api_doc_{hash(url)}",
                                "name": soup.title.string if soup.title else "API Documentation",
                                "content": soup.get_text(),
                                "type": "api_documentation",
                                "url": url,
                                "extracted_at": datetime.utcnow().isoformat()
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to extract API documentation from {url}: {e}")
        
        logger.info(f"Extracted {len(data)} API documentation pages")
        return data
    
    async def _extract_monitoring_logs_data(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract data from monitoring logs"""
        config = source.connection_config
        
        # This would typically connect to log aggregation systems
        # For now, we'll simulate with sample data
        data = []
        
        # Sample monitoring data structure
        sample_metrics = [
            {
                "id": "metric_cpu_usage",
                "name": "CPU Usage",
                "type": "monitoring_metric",
                "metric_type": "gauge",
                "unit": "percentage",
                "description": "CPU utilization percentage"
            },
            {
                "id": "metric_memory_usage",
                "name": "Memory Usage",
                "type": "monitoring_metric",
                "metric_type": "gauge",
                "unit": "bytes",
                "description": "Memory utilization in bytes"
            }
        ]
        
        data.extend(sample_metrics)
        
        logger.info(f"Extracted {len(data)} monitoring metrics")
        return data
    
    async def _process_extracted_data(self, data: List[Dict[str, Any]], source: DataSource) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Process extracted data to identify entities and relationships"""
        entities = []
        relationships = []
        
        for item in data:
            # Extract entities from each data item
            item_entities = await self._extract_entities_from_item(item, source)
            entities.extend(item_entities)
            
            # Extract relationships from each data item
            item_relationships = await self._extract_relationships_from_item(item, source)
            relationships.extend(item_relationships)
            
            self.stats["documents_processed"] += 1
        
        # Deduplicate entities and relationships
        entities = self._deduplicate_entities(entities)
        relationships = self._deduplicate_relationships(relationships)
        
        self.stats["entities_extracted"] += len(entities)
        self.stats["relationships_extracted"] += len(relationships)
        
        logger.info(f"Processed {len(data)} items, extracted {len(entities)} entities and {len(relationships)} relationships")
        return entities, relationships
    
    async def _extract_entities_from_item(self, item: Dict[str, Any], source: DataSource) -> List[ExtractedEntity]:
        """Extract entities from a single data item"""
        entities = []
        
        # Extract basic entity from the item itself
        entity_id = item.get("id", str(uuid.uuid4()))
        entity_type = item.get("type", "Document")
        entity_name = item.get("name") or item.get("title", "Unknown")
        
        # Create primary entity
        primary_entity = ExtractedEntity(
            id=entity_id,
            type=entity_type,
            name=entity_name,
            properties={
                "description": item.get("description", ""),
                "url": item.get("url", ""),
                "created_date": item.get("created_date"),
                "updated_date": item.get("updated_date"),
                "source_type": source.type.value
            },
            confidence=1.0,
            source=source.id,
            extracted_at=datetime.utcnow()
        )
        entities.append(primary_entity)
        
        # Extract entities from content using NLP
        content = item.get("content", "") or item.get("description", "")
        if content:
            nlp_entities = await self._extract_nlp_entities(content, source)
            entities.extend(nlp_entities)
        
        # Extract domain-specific entities
        if item.get("type") == "jira_issue":
            entities.extend(self._extract_jira_entities(item, source))
        elif item.get("type") == "github_repository":
            entities.extend(self._extract_github_entities(item, source))
        elif item.get("type") == "source_code_file":
            entities.extend(await self._extract_code_entities(item, source))
        
        return entities
    
    async def _extract_nlp_entities(self, content: str, source: DataSource) -> List[ExtractedEntity]:
        """Extract entities using NLP techniques"""
        entities = []
        
        # Use spaCy for named entity recognition
        doc = self.nlp(content[:10000])  # Limit content length
        
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE"]:
                entity_type = self._map_spacy_label_to_node_type(ent.label_)
                
                entity = ExtractedEntity(
                    id=f"nlp_{hash(ent.text)}_{entity_type}",
                    type=entity_type,
                    name=ent.text,
                    properties={
                        "spacy_label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "confidence": 0.8
                    },
                    confidence=0.8,
                    source=source.id,
                    extracted_at=datetime.utcnow()
                )
                entities.append(entity)
        
        # Use transformer-based NER
        try:
            ner_results = self.ner_pipeline(content[:512])  # Limit for transformer
            
            for result in ner_results:
                if result["score"] > 0.7:  # High confidence threshold
                    entity_type = self._map_transformer_label_to_node_type(result["entity_group"])
                    
                    entity = ExtractedEntity(
                        id=f"transformer_{hash(result['word'])}_{entity_type}",
                        type=entity_type,
                        name=result["word"],
                        properties={
                            "transformer_label": result["entity_group"],
                            "confidence": result["score"]
                        },
                        confidence=result["score"],
                        source=source.id,
                        extracted_at=datetime.utcnow()
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.warning(f"Transformer NER failed: {e}")
        
        return entities
    
    def _extract_jira_entities(self, item: Dict[str, Any], source: DataSource) -> List[ExtractedEntity]:
        """Extract Jira-specific entities"""
        entities = []
        
        # Extract assignee as Person entity
        if item.get("assignee"):
            person_entity = ExtractedEntity(
                id=f"person_{hash(item['assignee'])}",
                type="Person",
                name=item["assignee"],
                properties={"role": "assignee"},
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(person_entity)
        
        # Extract components as System entities
        for component in item.get("components", []):
            component_entity = ExtractedEntity(
                id=f"component_{hash(component)}",
                type="Component",
                name=component,
                properties={"source": "jira"},
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(component_entity)
        
        return entities
    
    def _extract_github_entities(self, item: Dict[str, Any], source: DataSource) -> List[ExtractedEntity]:
        """Extract GitHub-specific entities"""
        entities = []
        
        # Extract programming language as Technology entity
        if item.get("language"):
            tech_entity = ExtractedEntity(
                id=f"technology_{hash(item['language'])}",
                type="Technology",
                name=item["language"],
                properties={"category": "programming_language"},
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(tech_entity)
        
        # Extract topics as tags/categories
        for topic in item.get("topics", []):
            topic_entity = ExtractedEntity(
                id=f"topic_{hash(topic)}",
                type="Technology",
                name=topic,
                properties={"category": "topic"},
                confidence=0.8,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(topic_entity)
        
        return entities
    
    async def _extract_code_entities(self, item: Dict[str, Any], source: DataSource) -> List[ExtractedEntity]:
        """Extract entities from source code"""
        entities = []
        
        content = item.get("content", "")
        language = item.get("language", "")
        
        # Extract classes, functions, and imports using regex patterns
        if language == "python":
            entities.extend(self._extract_python_entities(content, item, source))
        elif language == "javascript":
            entities.extend(self._extract_javascript_entities(content, item, source))
        elif language == "java":
            entities.extend(self._extract_java_entities(content, item, source))
        
        return entities
    
    def _extract_python_entities(self, content: str, item: Dict[str, Any], source: DataSource) -> List[ExtractedEntity]:
        """Extract Python-specific entities"""
        entities = []
        
        # Extract class definitions
        class_pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            entity = ExtractedEntity(
                id=f"class_{hash(f'{item["path"]}_{class_name}')}",
                type="Component",
                name=class_name,
                properties={
                    "component_type": "class",
                    "language": "python",
                    "file_path": item["path"]
                },
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(entity)
        
        # Extract function definitions
        function_pattern = r'def\s+(\w+)\s*\('
        for match in re.finditer(function_pattern, content):
            function_name = match.group(1)
            entity = ExtractedEntity(
                id=f"function_{hash(f'{item["path"]}_{function_name}')}",
                type="Component",
                name=function_name,
                properties={
                    "component_type": "function",
                    "language": "python",
                    "file_path": item["path"]
                },
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(entity)
        
        # Extract imports
        import_pattern = r'(?:from\s+(\w+(?:\.\w+)*)\s+)?import\s+([^#\n]+)'
        for match in re.finditer(import_pattern, content):
            module = match.group(1) or match.group(2).split(',')[0].strip()
            entity = ExtractedEntity(
                id=f"module_{hash(module)}",
                type="Library",
                name=module,
                properties={
                    "category": "python_module",
                    "language": "python"
                },
                confidence=0.9,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(entity)
        
        return entities
    
    def _extract_javascript_entities(self, content: str, item: Dict[str, Any], source: DataSource) -> List[ExtractedEntity]:
        """Extract JavaScript-specific entities"""
        entities = []
        
        # Extract function definitions
        function_pattern = r'function\s+(\w+)\s*\('
        for match in re.finditer(function_pattern, content):
            function_name = match.group(1)
            entity = ExtractedEntity(
                id=f"function_{hash(f'{item["path"]}_{function_name}')}",
                type="Component",
                name=function_name,
                properties={
                    "component_type": "function",
                    "language": "javascript",
                    "file_path": item["path"]
                },
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(entity)
        
        # Extract class definitions
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            entity = ExtractedEntity(
                id=f"class_{hash(f'{item["path"]}_{class_name}')}",
                type="Component",
                name=class_name,
                properties={
                    "component_type": "class",
                    "language": "javascript",
                    "file_path": item["path"]
                },
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(entity)
        
        return entities
    
    def _extract_java_entities(self, content: str, item: Dict[str, Any], source: DataSource) -> List[ExtractedEntity]:
        """Extract Java-specific entities"""
        entities = []
        
        # Extract class definitions
        class_pattern = r'(?:public\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            entity = ExtractedEntity(
                id=f"class_{hash(f'{item["path"]}_{class_name}')}",
                type="Component",
                name=class_name,
                properties={
                    "component_type": "class",
                    "language": "java",
                    "file_path": item["path"]
                },
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            entities.append(entity)
        
        # Extract method definitions
        method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{'
        for match in re.finditer(method_pattern, content):
            method_name = match.group(1)
            if method_name not in ['if', 'for', 'while', 'switch']:  # Filter out keywords
                entity = ExtractedEntity(
                    id=f"method_{hash(f'{item["path"]}_{method_name}')}",
                    type="Component",
                    name=method_name,
                    properties={
                        "component_type": "method",
                        "language": "java",
                        "file_path": item["path"]
                    },
                    confidence=0.8,
                    source=source.id,
                    extracted_at=datetime.utcnow()
                )
                entities.append(entity)
        
        return entities
    
    async def _extract_relationships_from_item(self, item: Dict[str, Any], source: DataSource) -> List[ExtractedRelationship]:
        """Extract relationships from a single data item"""
        relationships = []
        
        # Extract relationships based on item type
        if item.get("type") == "jira_issue":
            relationships.extend(self._extract_jira_relationships(item, source))
        elif item.get("type") == "github_repository":
            relationships.extend(self._extract_github_relationships(item, source))
        elif item.get("type") == "source_code_file":
            relationships.extend(self._extract_code_relationships(item, source))
        
        return relationships
    
    def _extract_jira_relationships(self, item: Dict[str, Any], source: DataSource) -> List[ExtractedRelationship]:
        """Extract Jira-specific relationships"""
        relationships = []
        
        issue_id = item["id"]
        
        # Assignee relationship
        if item.get("assignee"):
            relationship = ExtractedRelationship(
                id=f"rel_{issue_id}_assignee",
                type="ASSIGNED_TO",
                source_entity=f"person_{hash(item['assignee'])}",
                target_entity=issue_id,
                properties={"relationship_type": "assignment"},
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            relationships.append(relationship)
        
        # Component relationships
        for component in item.get("components", []):
            relationship = ExtractedRelationship(
                id=f"rel_{issue_id}_{component}",
                type="RELATES_TO",
                source_entity=issue_id,
                target_entity=f"component_{hash(component)}",
                properties={"relationship_type": "component_issue"},
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            relationships.append(relationship)
        
        return relationships
    
    def _extract_github_relationships(self, item: Dict[str, Any], source: DataSource) -> List[ExtractedRelationship]:
        """Extract GitHub-specific relationships"""
        relationships = []
        
        repo_id = item["id"]
        
        # Language relationship
        if item.get("language"):
            relationship = ExtractedRelationship(
                id=f"rel_{repo_id}_language",
                type="USES",
                source_entity=repo_id,
                target_entity=f"technology_{hash(item['language'])}",
                properties={"relationship_type": "primary_language"},
                confidence=1.0,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            relationships.append(relationship)
        
        # Topic relationships
        for topic in item.get("topics", []):
            relationship = ExtractedRelationship(
                id=f"rel_{repo_id}_{topic}",
                type="TAGGED_WITH",
                source_entity=repo_id,
                target_entity=f"topic_{hash(topic)}",
                properties={"relationship_type": "topic"},
                confidence=0.8,
                source=source.id,
                extracted_at=datetime.utcnow()
            )
            relationships.append(relationship)
        
        return relationships
    
    def _extract_code_relationships(self, item: Dict[str, Any], source: DataSource) -> List[ExtractedRelationship]:
        """Extract relationships from source code"""
        relationships = []
        
        content = item.get("content", "")
        language = item.get("language", "")
        file_id = item["id"]
        
        # Extract import/dependency relationships
        if language == "python":
            import_pattern = r'(?:from\s+(\w+(?:\.\w+)*)\s+)?import\s+([^#\n]+)'
            for match in re.finditer(import_pattern, content):
                module = match.group(1) or match.group(2).split(',')[0].strip()
                relationship = ExtractedRelationship(
                    id=f"rel_{file_id}_{module}",
                    type="DEPENDS_ON",
                    source_entity=file_id,
                    target_entity=f"module_{hash(module)}",
                    properties={"dependency_type": "import"},
                    confidence=1.0,
                    source=source.id,
                    extracted_at=datetime.utcnow()
                )
                relationships.append(relationship)
        
        return relationships
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities based on ID"""
        seen_ids = set()
        unique_entities = []
        
        for entity in entities:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_relationships(self, relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Remove duplicate relationships based on ID"""
        seen_ids = set()
        unique_relationships = []
        
        for relationship in relationships:
            if relationship.id not in seen_ids:
                seen_ids.add(relationship.id)
                unique_relationships.append(relationship)
        
        return unique_relationships
    
    async def _store_in_graph(self, entities: List[ExtractedEntity], relationships: List[ExtractedRelationship]):
        """Store extracted entities and relationships in the knowledge graph"""
        logger.info(f"Storing {len(entities)} entities and {len(relationships)} relationships in graph")
        
        with self.neo4j_driver.session() as session:
            # Store entities
            for entity in entities:
                query = f"""
                MERGE (n:{entity.type} {{id: $id}})
                SET n.name = $name,
                    n.confidence = $confidence,
                    n.source = $source,
                    n.extracted_at = $extracted_at,
                    n += $properties
                """
                
                session.run(query, {
                    "id": entity.id,
                    "name": entity.name,
                    "confidence": entity.confidence,
                    "source": entity.source,
                    "extracted_at": entity.extracted_at.isoformat(),
                    "properties": entity.properties
                })
            
            # Store relationships
            for relationship in relationships:
                query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{relationship.type}]->(target)
                SET r.confidence = $confidence,
                    r.source = $source,
                    r.extracted_at = $extracted_at,
                    r += $properties
                """
                
                try:
                    session.run(query, {
                        "source_id": relationship.source_entity,
                        "target_id": relationship.target_entity,
                        "confidence": relationship.confidence,
                        "source": relationship.source,
                        "extracted_at": relationship.extracted_at.isoformat(),
                        "properties": relationship.properties
                    })
                except Exception as e:
                    logger.warning(f"Failed to create relationship {relationship.id}: {e}")
        
        logger.info("Graph storage completed")
    
    def _detect_language(self, filename: str) -> str:
        """Detect programming language from filename"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        ext = os.path.splitext(filename)[1].lower()
        return extension_map.get(ext, 'unknown')
    
    def _map_spacy_label_to_node_type(self, label: str) -> str:
        """Map spaCy entity labels to graph node types"""
        mapping = {
            "ORG": "Organization",
            "PERSON": "Person",
            "PRODUCT": "Technology",
            "GPE": "Location"
        }
        return mapping.get(label, "Entity")
    
    def _map_transformer_label_to_node_type(self, label: str) -> str:
        """Map transformer entity labels to graph node types"""
        mapping = {
            "ORG": "Organization",
            "PER": "Person",
            "MISC": "Technology",
            "LOC": "Location"
        }
        return mapping.get(label, "Entity")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()
    
    def close(self):
        """Close database connections"""
        self.neo4j_driver.close()

# Example usage
if __name__ == "__main__":
    config = {
        "neo4j": {
            "uri": "bolt://neo4j-lb.nexus-knowledge-graph:7687",
            "user": "neo4j",
            "password": "nexus-architect-graph-password"
        },
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY")
    }
    
    pipeline = GraphConstructionPipeline(config)
    
    # Example data source registration
    confluence_source = DataSource(
        id="confluence_main",
        name="Main Confluence Space",
        type=DataSourceType.CONFLUENCE,
        connection_config={
            "base_url": "https://company.atlassian.net/wiki",
            "username": "api_user",
            "api_token": "api_token"
        },
        extraction_config={}
    )
    
    async def main():
        await pipeline.register_data_source(confluence_source)
        results = await pipeline.process_all_sources()
        stats = pipeline.get_statistics()
        print(f"Processing completed: {stats}")
        pipeline.close()
    
    # asyncio.run(main())

