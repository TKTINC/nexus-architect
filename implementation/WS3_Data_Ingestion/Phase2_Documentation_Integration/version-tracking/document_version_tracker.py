"""
Document Version Tracker for Nexus Architect
Comprehensive document versioning, change detection, and real-time updates
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
import difflib
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
VERSION_TRACKING_REQUESTS = Counter('version_tracking_requests_total', 'Total version tracking requests', ['operation', 'status'])
VERSION_TRACKING_LATENCY = Histogram('version_tracking_latency_seconds', 'Version tracking latency', ['operation'])
CHANGE_DETECTION_ACCURACY = Gauge('change_detection_accuracy', 'Change detection accuracy score')
DOCUMENT_VERSIONS_COUNT = Gauge('document_versions_count', 'Total number of document versions tracked')

class ChangeType(Enum):
    """Types of document changes"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    RENAMED = "renamed"
    CONTENT_CHANGED = "content_changed"
    METADATA_CHANGED = "metadata_changed"
    STRUCTURE_CHANGED = "structure_changed"

class VersionStatus(Enum):
    """Version status"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    DRAFT = "draft"
    PUBLISHED = "published"

@dataclass
class DocumentChange:
    """Represents a change in a document"""
    change_id: str
    document_id: str
    change_type: ChangeType
    timestamp: datetime
    author: Optional[str] = None
    description: str = ""
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

@dataclass
class DocumentVersion:
    """Represents a version of a document"""
    version_id: str
    document_id: str
    version_number: str
    content_hash: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    author: Optional[str] = None
    status: VersionStatus = VersionStatus.ACTIVE
    parent_version_id: Optional[str] = None
    changes: List[DocumentChange] = field(default_factory=list)
    size_bytes: int = 0
    word_count: int = 0

@dataclass
class DocumentHistory:
    """Complete history of a document"""
    document_id: str
    versions: List[DocumentVersion] = field(default_factory=list)
    current_version_id: Optional[str] = None
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    total_changes: int = 0
    authors: Set[str] = field(default_factory=set)

class ContentHasher:
    """Content hashing for change detection"""
    
    @staticmethod
    def hash_content(content: str) -> str:
        """Generate hash for content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_metadata(metadata: Dict[str, Any]) -> str:
        """Generate hash for metadata"""
        # Sort keys for consistent hashing
        sorted_metadata = json.dumps(metadata, sort_keys=True, default=str)
        return hashlib.sha256(sorted_metadata.encode('utf-8')).hexdigest()
    
    @staticmethod
    def hash_structure(structured_content: List[Dict[str, Any]]) -> str:
        """Generate hash for document structure"""
        # Extract structure without content
        structure = []
        for item in structured_content:
            structure.append({
                'type': item.get('content_type'),
                'level': item.get('metadata', {}).get('level'),
                'position': item.get('position')
            })
        
        structure_json = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(structure_json.encode('utf-8')).hexdigest()

class ChangeDetector:
    """Detect changes between document versions"""
    
    def __init__(self):
        self.similarity_threshold = 0.8
    
    async def detect_changes(self, old_version: DocumentVersion, new_content: str, 
                           new_metadata: Dict[str, Any]) -> List[DocumentChange]:
        """Detect changes between versions"""
        start_time = time.time()
        changes = []
        
        try:
            # Content changes
            content_changes = await self._detect_content_changes(old_version, new_content)
            changes.extend(content_changes)
            
            # Metadata changes
            metadata_changes = await self._detect_metadata_changes(old_version, new_metadata)
            changes.extend(metadata_changes)
            
            # Structure changes
            structure_changes = await self._detect_structure_changes(old_version, new_content)
            changes.extend(structure_changes)
            
            VERSION_TRACKING_REQUESTS.labels(
                operation='change_detection',
                status='success'
            ).inc()
            
            # Calculate accuracy based on change confidence
            if changes:
                avg_confidence = sum(c.confidence for c in changes) / len(changes)
                CHANGE_DETECTION_ACCURACY.set(avg_confidence)
        
        except Exception as e:
            logger.error(f"Change detection failed: {e}")
            VERSION_TRACKING_REQUESTS.labels(
                operation='change_detection',
                status='error'
            ).inc()
        
        finally:
            latency = time.time() - start_time
            VERSION_TRACKING_LATENCY.labels(operation='change_detection').observe(latency)
        
        return changes
    
    async def _detect_content_changes(self, old_version: DocumentVersion, new_content: str) -> List[DocumentChange]:
        """Detect content changes"""
        changes = []
        
        if old_version.content == new_content:
            return changes
        
        # Calculate similarity
        similarity = self._calculate_similarity(old_version.content, new_content)
        
        if similarity < self.similarity_threshold:
            # Significant content change
            change = DocumentChange(
                change_id=self._generate_change_id(),
                document_id=old_version.document_id,
                change_type=ChangeType.CONTENT_CHANGED,
                timestamp=datetime.now(timezone.utc),
                description=f"Content changed (similarity: {similarity:.2f})",
                old_value=old_version.content[:500] + "..." if len(old_version.content) > 500 else old_version.content,
                new_value=new_content[:500] + "..." if len(new_content) > 500 else new_content,
                confidence=1.0 - similarity,
                metadata={
                    "similarity_score": similarity,
                    "old_word_count": len(old_version.content.split()),
                    "new_word_count": len(new_content.split()),
                    "size_change": len(new_content) - len(old_version.content)
                }
            )
            changes.append(change)
            
            # Detailed diff analysis
            diff_changes = await self._analyze_content_diff(old_version, new_content)
            changes.extend(diff_changes)
        
        return changes
    
    async def _detect_metadata_changes(self, old_version: DocumentVersion, new_metadata: Dict[str, Any]) -> List[DocumentChange]:
        """Detect metadata changes"""
        changes = []
        
        old_metadata = old_version.metadata
        
        # Check for added metadata
        for key, value in new_metadata.items():
            if key not in old_metadata:
                change = DocumentChange(
                    change_id=self._generate_change_id(),
                    document_id=old_version.document_id,
                    change_type=ChangeType.METADATA_CHANGED,
                    timestamp=datetime.now(timezone.utc),
                    description=f"Metadata added: {key}",
                    old_value=None,
                    new_value=str(value),
                    confidence=1.0,
                    metadata={"metadata_key": key, "operation": "added"}
                )
                changes.append(change)
        
        # Check for modified metadata
        for key, old_value in old_metadata.items():
            if key in new_metadata:
                new_value = new_metadata[key]
                if old_value != new_value:
                    change = DocumentChange(
                        change_id=self._generate_change_id(),
                        document_id=old_version.document_id,
                        change_type=ChangeType.METADATA_CHANGED,
                        timestamp=datetime.now(timezone.utc),
                        description=f"Metadata modified: {key}",
                        old_value=str(old_value),
                        new_value=str(new_value),
                        confidence=1.0,
                        metadata={"metadata_key": key, "operation": "modified"}
                    )
                    changes.append(change)
            else:
                # Metadata removed
                change = DocumentChange(
                    change_id=self._generate_change_id(),
                    document_id=old_version.document_id,
                    change_type=ChangeType.METADATA_CHANGED,
                    timestamp=datetime.now(timezone.utc),
                    description=f"Metadata removed: {key}",
                    old_value=str(old_value),
                    new_value=None,
                    confidence=1.0,
                    metadata={"metadata_key": key, "operation": "removed"}
                )
                changes.append(change)
        
        return changes
    
    async def _detect_structure_changes(self, old_version: DocumentVersion, new_content: str) -> List[DocumentChange]:
        """Detect structural changes"""
        changes = []
        
        try:
            # Extract structure from old and new content
            old_structure = self._extract_structure(old_version.content)
            new_structure = self._extract_structure(new_content)
            
            # Compare structures
            if old_structure != new_structure:
                change = DocumentChange(
                    change_id=self._generate_change_id(),
                    document_id=old_version.document_id,
                    change_type=ChangeType.STRUCTURE_CHANGED,
                    timestamp=datetime.now(timezone.utc),
                    description="Document structure changed",
                    confidence=0.9,
                    metadata={
                        "old_structure_hash": ContentHasher.hash_structure(old_structure),
                        "new_structure_hash": ContentHasher.hash_structure(new_structure),
                        "old_sections": len(old_structure),
                        "new_sections": len(new_structure)
                    }
                )
                changes.append(change)
        
        except Exception as e:
            logger.warning(f"Structure change detection failed: {e}")
        
        return changes
    
    async def _analyze_content_diff(self, old_version: DocumentVersion, new_content: str) -> List[DocumentChange]:
        """Analyze detailed content differences"""
        changes = []
        
        try:
            # Split content into lines for diff analysis
            old_lines = old_version.content.splitlines()
            new_lines = new_content.splitlines()
            
            # Generate diff
            diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=''))
            
            if diff:
                # Count additions and deletions
                additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
                deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
                
                if additions > 0:
                    change = DocumentChange(
                        change_id=self._generate_change_id(),
                        document_id=old_version.document_id,
                        change_type=ChangeType.CONTENT_CHANGED,
                        timestamp=datetime.now(timezone.utc),
                        description=f"Content added: {additions} lines",
                        confidence=0.9,
                        metadata={
                            "diff_type": "addition",
                            "line_count": additions,
                            "diff_sample": '\n'.join([line for line in diff if line.startswith('+')][:5])
                        }
                    )
                    changes.append(change)
                
                if deletions > 0:
                    change = DocumentChange(
                        change_id=self._generate_change_id(),
                        document_id=old_version.document_id,
                        change_type=ChangeType.CONTENT_CHANGED,
                        timestamp=datetime.now(timezone.utc),
                        description=f"Content deleted: {deletions} lines",
                        confidence=0.9,
                        metadata={
                            "diff_type": "deletion",
                            "line_count": deletions,
                            "diff_sample": '\n'.join([line for line in diff if line.startswith('-')][:5])
                        }
                    )
                    changes.append(change)
        
        except Exception as e:
            logger.warning(f"Content diff analysis failed: {e}")
        
        return changes
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            # Use difflib for similarity calculation
            similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
            return similarity
        except Exception:
            return 0.0
    
    def _extract_structure(self, content: str) -> List[Dict[str, Any]]:
        """Extract document structure"""
        structure = []
        
        # Extract headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2)
            structure.append({
                'type': 'heading',
                'level': level,
                'text': text,
                'position': match.start()
            })
        
        # Extract lists
        list_pattern = r'^[\s]*[-*+]\s+(.+)$'
        for match in re.finditer(list_pattern, content, re.MULTILINE):
            structure.append({
                'type': 'list_item',
                'text': match.group(1),
                'position': match.start()
            })
        
        # Extract code blocks
        code_pattern = r'```[\s\S]*?```'
        for match in re.finditer(code_pattern, content):
            structure.append({
                'type': 'code_block',
                'position': match.start()
            })
        
        return structure
    
    def _generate_change_id(self) -> str:
        """Generate unique change ID"""
        return hashlib.md5(f"{time.time()}_{os.urandom(8).hex()}".encode()).hexdigest()

class DocumentVersionTracker:
    """Main document version tracking system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.change_detector = ChangeDetector()
        self.document_histories: Dict[str, DocumentHistory] = {}
        self.max_versions_per_document = config.get('max_versions_per_document', 100)
        self.auto_cleanup_enabled = config.get('auto_cleanup_enabled', True)
    
    async def track_document(self, document_id: str, content: str, metadata: Dict[str, Any], 
                           author: Optional[str] = None) -> DocumentVersion:
        """Track a new version of a document"""
        start_time = time.time()
        
        try:
            # Get or create document history
            if document_id not in self.document_histories:
                self.document_histories[document_id] = DocumentHistory(
                    document_id=document_id,
                    created_at=datetime.now(timezone.utc)
                )
            
            history = self.document_histories[document_id]
            
            # Generate content hash
            content_hash = ContentHasher.hash_content(content)
            
            # Check if content has actually changed
            if history.current_version_id:
                current_version = self._get_version_by_id(history.current_version_id)
                if current_version and current_version.content_hash == content_hash:
                    # No changes, return current version
                    return current_version
            
            # Create new version
            version_number = self._generate_version_number(history)
            version_id = f"{document_id}_{version_number}_{int(time.time())}"
            
            new_version = DocumentVersion(
                version_id=version_id,
                document_id=document_id,
                version_number=version_number,
                content_hash=content_hash,
                content=content,
                metadata=metadata,
                timestamp=datetime.now(timezone.utc),
                author=author,
                parent_version_id=history.current_version_id,
                size_bytes=len(content.encode('utf-8')),
                word_count=len(content.split())
            )
            
            # Detect changes if there's a previous version
            if history.current_version_id:
                old_version = self._get_version_by_id(history.current_version_id)
                if old_version:
                    changes = await self.change_detector.detect_changes(old_version, content, metadata)
                    new_version.changes = changes
            else:
                # First version
                create_change = DocumentChange(
                    change_id=self.change_detector._generate_change_id(),
                    document_id=document_id,
                    change_type=ChangeType.CREATED,
                    timestamp=new_version.timestamp,
                    author=author,
                    description="Document created",
                    confidence=1.0
                )
                new_version.changes = [create_change]
            
            # Add version to history
            history.versions.append(new_version)
            history.current_version_id = version_id
            history.last_modified = new_version.timestamp
            history.total_changes += len(new_version.changes)
            
            if author:
                history.authors.add(author)
            
            # Cleanup old versions if needed
            if self.auto_cleanup_enabled:
                await self._cleanup_old_versions(history)
            
            # Update metrics
            DOCUMENT_VERSIONS_COUNT.set(sum(len(h.versions) for h in self.document_histories.values()))
            
            VERSION_TRACKING_REQUESTS.labels(
                operation='track_document',
                status='success'
            ).inc()
            
            logger.info(f"Tracked new version {version_number} for document {document_id}")
            
            return new_version
        
        except Exception as e:
            logger.error(f"Document tracking failed: {e}")
            VERSION_TRACKING_REQUESTS.labels(
                operation='track_document',
                status='error'
            ).inc()
            raise
        
        finally:
            latency = time.time() - start_time
            VERSION_TRACKING_LATENCY.labels(operation='track_document').observe(latency)
    
    async def get_document_history(self, document_id: str) -> Optional[DocumentHistory]:
        """Get complete history of a document"""
        return self.document_histories.get(document_id)
    
    async def get_document_version(self, document_id: str, version_id: Optional[str] = None) -> Optional[DocumentVersion]:
        """Get specific version of a document"""
        history = self.document_histories.get(document_id)
        if not history:
            return None
        
        if version_id:
            return self._get_version_by_id(version_id)
        else:
            # Return current version
            if history.current_version_id:
                return self._get_version_by_id(history.current_version_id)
        
        return None
    
    async def get_document_changes(self, document_id: str, since: Optional[datetime] = None) -> List[DocumentChange]:
        """Get changes for a document since a specific time"""
        history = self.document_histories.get(document_id)
        if not history:
            return []
        
        all_changes = []
        for version in history.versions:
            for change in version.changes:
                if not since or change.timestamp >= since:
                    all_changes.append(change)
        
        # Sort by timestamp
        all_changes.sort(key=lambda c: c.timestamp)
        return all_changes
    
    async def compare_versions(self, document_id: str, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Compare two versions of a document"""
        version1 = self._get_version_by_id(version1_id)
        version2 = self._get_version_by_id(version2_id)
        
        if not version1 or not version2:
            return {}
        
        # Generate diff
        diff = list(difflib.unified_diff(
            version1.content.splitlines(),
            version2.content.splitlines(),
            fromfile=f"Version {version1.version_number}",
            tofile=f"Version {version2.version_number}",
            lineterm=''
        ))
        
        # Calculate statistics
        similarity = self.change_detector._calculate_similarity(version1.content, version2.content)
        
        return {
            "version1": {
                "version_id": version1.version_id,
                "version_number": version1.version_number,
                "timestamp": version1.timestamp.isoformat(),
                "author": version1.author,
                "word_count": version1.word_count,
                "size_bytes": version1.size_bytes
            },
            "version2": {
                "version_id": version2.version_id,
                "version_number": version2.version_number,
                "timestamp": version2.timestamp.isoformat(),
                "author": version2.author,
                "word_count": version2.word_count,
                "size_bytes": version2.size_bytes
            },
            "comparison": {
                "similarity": similarity,
                "word_count_change": version2.word_count - version1.word_count,
                "size_change": version2.size_bytes - version1.size_bytes,
                "time_difference": (version2.timestamp - version1.timestamp).total_seconds(),
                "diff_lines": len(diff),
                "diff": diff[:100]  # Limit diff output
            }
        }
    
    async def restore_version(self, document_id: str, version_id: str, author: Optional[str] = None) -> DocumentVersion:
        """Restore a previous version as the current version"""
        version_to_restore = self._get_version_by_id(version_id)
        if not version_to_restore:
            raise ValueError(f"Version {version_id} not found")
        
        # Create new version with restored content
        restored_version = await self.track_document(
            document_id=document_id,
            content=version_to_restore.content,
            metadata=version_to_restore.metadata,
            author=author
        )
        
        # Add restore change
        restore_change = DocumentChange(
            change_id=self.change_detector._generate_change_id(),
            document_id=document_id,
            change_type=ChangeType.MODIFIED,
            timestamp=restored_version.timestamp,
            author=author,
            description=f"Restored to version {version_to_restore.version_number}",
            confidence=1.0,
            metadata={
                "restored_from_version": version_to_restore.version_id,
                "restored_from_version_number": version_to_restore.version_number
            }
        )
        restored_version.changes.append(restore_change)
        
        return restored_version
    
    async def delete_document(self, document_id: str, author: Optional[str] = None) -> bool:
        """Mark document as deleted"""
        history = self.document_histories.get(document_id)
        if not history:
            return False
        
        # Create deletion change
        delete_change = DocumentChange(
            change_id=self.change_detector._generate_change_id(),
            document_id=document_id,
            change_type=ChangeType.DELETED,
            timestamp=datetime.now(timezone.utc),
            author=author,
            description="Document deleted",
            confidence=1.0
        )
        
        # Mark current version as deleted
        if history.current_version_id:
            current_version = self._get_version_by_id(history.current_version_id)
            if current_version:
                current_version.status = VersionStatus.DELETED
                current_version.changes.append(delete_change)
        
        return True
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get version tracking statistics"""
        total_documents = len(self.document_histories)
        total_versions = sum(len(h.versions) for h in self.document_histories.values())
        total_changes = sum(h.total_changes for h in self.document_histories.values())
        
        # Calculate average versions per document
        avg_versions = total_versions / total_documents if total_documents > 0 else 0
        
        # Find most active documents
        most_active = sorted(
            self.document_histories.items(),
            key=lambda x: len(x[1].versions),
            reverse=True
        )[:5]
        
        # Calculate change types distribution
        change_types = {}
        for history in self.document_histories.values():
            for version in history.versions:
                for change in version.changes:
                    change_type = change.change_type.value
                    change_types[change_type] = change_types.get(change_type, 0) + 1
        
        return {
            "total_documents": total_documents,
            "total_versions": total_versions,
            "total_changes": total_changes,
            "average_versions_per_document": avg_versions,
            "most_active_documents": [
                {
                    "document_id": doc_id,
                    "versions": len(history.versions),
                    "changes": history.total_changes,
                    "authors": len(history.authors)
                }
                for doc_id, history in most_active
            ],
            "change_types_distribution": change_types,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_version_by_id(self, version_id: str) -> Optional[DocumentVersion]:
        """Get version by ID"""
        for history in self.document_histories.values():
            for version in history.versions:
                if version.version_id == version_id:
                    return version
        return None
    
    def _generate_version_number(self, history: DocumentHistory) -> str:
        """Generate version number"""
        version_count = len(history.versions)
        return f"v{version_count + 1}"
    
    async def _cleanup_old_versions(self, history: DocumentHistory):
        """Clean up old versions if limit exceeded"""
        if len(history.versions) > self.max_versions_per_document:
            # Keep the most recent versions and important milestones
            versions_to_keep = []
            
            # Always keep current version
            if history.current_version_id:
                current_version = self._get_version_by_id(history.current_version_id)
                if current_version:
                    versions_to_keep.append(current_version)
            
            # Keep recent versions
            recent_versions = sorted(history.versions, key=lambda v: v.timestamp, reverse=True)
            versions_to_keep.extend(recent_versions[:self.max_versions_per_document - 1])
            
            # Remove duplicates
            unique_versions = []
            seen_ids = set()
            for version in versions_to_keep:
                if version.version_id not in seen_ids:
                    unique_versions.append(version)
                    seen_ids.add(version.version_id)
            
            history.versions = unique_versions
            logger.info(f"Cleaned up old versions for document {history.document_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        stats = await self.get_statistics()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'statistics': stats,
            'configuration': {
                'max_versions_per_document': self.max_versions_per_document,
                'auto_cleanup_enabled': self.auto_cleanup_enabled
            }
        }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        config = {
            'max_versions_per_document': 50,
            'auto_cleanup_enabled': True
        }
        
        tracker = DocumentVersionTracker(config)
        
        # Example usage
        document_id = "test_doc_1"
        
        # Track initial version
        version1 = await tracker.track_document(
            document_id=document_id,
            content="# Test Document\n\nThis is the initial content.",
            metadata={"title": "Test Document", "author": "John Doe"},
            author="john.doe@example.com"
        )
        print(f"Created version: {version1.version_number}")
        
        # Track modified version
        version2 = await tracker.track_document(
            document_id=document_id,
            content="# Test Document\n\nThis is the modified content with more details.",
            metadata={"title": "Test Document", "author": "John Doe", "status": "updated"},
            author="john.doe@example.com"
        )
        print(f"Created version: {version2.version_number}")
        print(f"Changes detected: {len(version2.changes)}")
        
        # Get document history
        history = await tracker.get_document_history(document_id)
        if history:
            print(f"Document has {len(history.versions)} versions")
            print(f"Total changes: {history.total_changes}")
        
        # Compare versions
        comparison = await tracker.compare_versions(document_id, version1.version_id, version2.version_id)
        print(f"Similarity: {comparison['comparison']['similarity']:.2f}")
        
        # Get statistics
        stats = await tracker.get_statistics()
        print(f"Statistics: {stats}")
        
        # Health check
        health = await tracker.health_check()
        print(f"Health: {health['status']}")
    
    asyncio.run(main())

