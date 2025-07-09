"""
Nexus Architect Conversation Context Manager

This module provides comprehensive context management for conversational AI,
including long-term memory, context window management, and multi-session continuity.
"""

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import redis.asyncio as redis
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class ContextType(str, Enum):
    """Types of context information"""
    USER_PROFILE = "user_profile"
    CONVERSATION_HISTORY = "conversation_history"
    DOMAIN_CONTEXT = "domain_context"
    TASK_CONTEXT = "task_context"
    SYSTEM_STATE = "system_state"

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    turn_id: str
    user_message: str
    ai_response: str
    timestamp: datetime
    persona_used: str
    confidence_score: float
    context_used: List[str]
    metadata: Dict[str, Any]

class ConversationSession(BaseModel):
    """Represents a conversation session"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    turns: List[ConversationTurn] = Field(default_factory=list)
    context_summary: Dict[str, Any] = Field(default_factory=dict)
    active_personas: List[str] = Field(default_factory=list)
    session_metadata: Dict[str, Any] = Field(default_factory=dict)

class ContextWindow:
    """Manages context window for large conversations"""
    
    def __init__(self, max_tokens: int = 8000, overlap_tokens: int = 500):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
    def create_windows(self, conversation_text: str) -> List[str]:
        """Create overlapping context windows from conversation text"""
        tokens = conversation_text.split()
        windows = []
        
        if len(tokens) <= self.max_tokens:
            return [conversation_text]
        
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            window = " ".join(tokens[start:end])
            windows.append(window)
            
            if end >= len(tokens):
                break
                
            start = end - self.overlap_tokens
            
        return windows
    
    def get_relevant_context(self, query: str, windows: List[str]) -> str:
        """Get most relevant context window for a query"""
        # Simple relevance scoring based on keyword overlap
        best_window = ""
        best_score = 0
        
        query_words = set(query.lower().split())
        
        for window in windows:
            window_words = set(window.lower().split())
            overlap = len(query_words.intersection(window_words))
            score = overlap / len(query_words) if query_words else 0
            
            if score > best_score:
                best_score = score
                best_window = window
                
        return best_window or windows[-1]  # Return last window if no good match

class ConversationContextManager:
    """Main context management system for conversational AI"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = None
        self.redis_url = redis_url
        self.context_window = ContextWindow()
        self.session_timeout = timedelta(hours=24)
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        logger.info("Context manager initialized with Redis connection")
        
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def create_session(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new conversation session"""
        session = ConversationSession(
            user_id=user_id,
            session_metadata=metadata or {}
        )
        
        await self._store_session(session)
        logger.info(f"Created new session {session.session_id} for user {user_id}")
        return session.session_id
        
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a conversation session"""
        session_data = await self.redis_client.get(f"session:{session_id}")
        if not session_data:
            return None
            
        session_dict = json.loads(session_data)
        # Convert datetime strings back to datetime objects
        session_dict['start_time'] = datetime.fromisoformat(session_dict['start_time'])
        session_dict['last_activity'] = datetime.fromisoformat(session_dict['last_activity'])
        
        # Convert turns
        turns = []
        for turn_data in session_dict.get('turns', []):
            turn_data['timestamp'] = datetime.fromisoformat(turn_data['timestamp'])
            turns.append(ConversationTurn(**turn_data))
        session_dict['turns'] = turns
        
        return ConversationSession(**session_dict)
        
    async def _store_session(self, session: ConversationSession):
        """Store session in Redis"""
        session_dict = session.dict()
        # Convert datetime objects to strings for JSON serialization
        session_dict['start_time'] = session.start_time.isoformat()
        session_dict['last_activity'] = session.last_activity.isoformat()
        
        # Convert turns to dictionaries
        turns_data = []
        for turn in session.turns:
            turn_dict = asdict(turn)
            turn_dict['timestamp'] = turn.timestamp.isoformat()
            turns_data.append(turn_dict)
        session_dict['turns'] = turns_data
        
        await self.redis_client.setex(
            f"session:{session.session_id}",
            int(self.session_timeout.total_seconds()),
            json.dumps(session_dict)
        )
        
    async def add_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        persona_used: str,
        confidence_score: float,
        context_used: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add a new turn to the conversation"""
        session = await self.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return False
            
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            user_message=user_message,
            ai_response=ai_response,
            timestamp=datetime.utcnow(),
            persona_used=persona_used,
            confidence_score=confidence_score,
            context_used=context_used or [],
            metadata=metadata or {}
        )
        
        session.turns.append(turn)
        session.last_activity = datetime.utcnow()
        
        # Update active personas
        if persona_used not in session.active_personas:
            session.active_personas.append(persona_used)
            
        await self._store_session(session)
        logger.info(f"Added turn to session {session_id}")
        return True
        
    async def get_conversation_context(
        self,
        session_id: str,
        max_turns: int = 10,
        include_system_context: bool = True
    ) -> Dict[str, Any]:
        """Get conversation context for AI processing"""
        session = await self.get_session(session_id)
        if not session:
            return {}
            
        # Get recent turns
        recent_turns = session.turns[-max_turns:] if session.turns else []
        
        # Build conversation history
        conversation_history = []
        for turn in recent_turns:
            conversation_history.append({
                "user": turn.user_message,
                "assistant": turn.ai_response,
                "timestamp": turn.timestamp.isoformat(),
                "persona": turn.persona_used,
                "confidence": turn.confidence_score
            })
            
        context = {
            "session_id": session_id,
            "user_id": session.user_id,
            "conversation_history": conversation_history,
            "active_personas": session.active_personas,
            "session_metadata": session.session_metadata,
            "context_summary": session.context_summary,
            "turn_count": len(session.turns),
            "session_duration": (datetime.utcnow() - session.start_time).total_seconds()
        }
        
        if include_system_context:
            context["system_context"] = await self._get_system_context(session.user_id)
            
        return context
        
    async def _get_system_context(self, user_id: str) -> Dict[str, Any]:
        """Get system-level context for user"""
        # Get user profile and preferences
        user_profile = await self.redis_client.get(f"user_profile:{user_id}")
        if user_profile:
            user_profile = json.loads(user_profile)
        else:
            user_profile = {}
            
        # Get recent system interactions
        recent_sessions = await self.redis_client.lrange(f"user_sessions:{user_id}", 0, 4)
        
        return {
            "user_profile": user_profile,
            "recent_sessions": recent_sessions,
            "preferences": user_profile.get("preferences", {}),
            "role": user_profile.get("role", "user"),
            "expertise_level": user_profile.get("expertise_level", "intermediate")
        }
        
    async def update_context_summary(
        self,
        session_id: str,
        summary_updates: Dict[str, Any]
    ) -> bool:
        """Update context summary for session"""
        session = await self.get_session(session_id)
        if not session:
            return False
            
        session.context_summary.update(summary_updates)
        await self._store_session(session)
        return True
        
    async def get_context_windows(self, session_id: str) -> List[str]:
        """Get context windows for large conversations"""
        session = await self.get_session(session_id)
        if not session:
            return []
            
        # Build full conversation text
        conversation_text = ""
        for turn in session.turns:
            conversation_text += f"User: {turn.user_message}\n"
            conversation_text += f"Assistant: {turn.ai_response}\n\n"
            
        return self.context_window.create_windows(conversation_text)
        
    async def get_relevant_context(
        self,
        session_id: str,
        query: str
    ) -> str:
        """Get most relevant context for a query"""
        windows = await self.get_context_windows(session_id)
        if not windows:
            return ""
            
        return self.context_window.get_relevant_context(query, windows)
        
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        # Redis TTL handles automatic cleanup, but we can do additional cleanup here
        pattern = "session:*"
        async for key in self.redis_client.scan_iter(match=pattern):
            ttl = await self.redis_client.ttl(key)
            if ttl == -1:  # No expiration set
                await self.redis_client.expire(key, int(self.session_timeout.total_seconds()))
                
    async def get_user_sessions(self, user_id: str, limit: int = 10) -> List[str]:
        """Get recent sessions for a user"""
        return await self.redis_client.lrange(f"user_sessions:{user_id}", 0, limit - 1)
        
    async def link_session_to_user(self, session_id: str, user_id: str):
        """Link session to user for tracking"""
        await self.redis_client.lpush(f"user_sessions:{user_id}", session_id)
        await self.redis_client.ltrim(f"user_sessions:{user_id}", 0, 49)  # Keep last 50 sessions

# Context sharing utilities
class MultiSessionContextManager:
    """Manages context sharing across multiple sessions"""
    
    def __init__(self, context_manager: ConversationContextManager):
        self.context_manager = context_manager
        
    async def share_context_between_sessions(
        self,
        source_session_id: str,
        target_session_id: str,
        context_types: List[ContextType]
    ) -> bool:
        """Share specific context types between sessions"""
        source_session = await self.context_manager.get_session(source_session_id)
        target_session = await self.context_manager.get_session(target_session_id)
        
        if not source_session or not target_session:
            return False
            
        # Share context based on types
        shared_context = {}
        
        if ContextType.DOMAIN_CONTEXT in context_types:
            shared_context["domain_knowledge"] = source_session.context_summary.get("domain_knowledge", {})
            
        if ContextType.TASK_CONTEXT in context_types:
            shared_context["current_tasks"] = source_session.context_summary.get("current_tasks", [])
            
        if ContextType.USER_PROFILE in context_types:
            shared_context["user_preferences"] = source_session.context_summary.get("user_preferences", {})
            
        # Update target session
        await self.context_manager.update_context_summary(target_session_id, shared_context)
        return True
        
    async def create_shared_workspace(
        self,
        user_ids: List[str],
        workspace_name: str
    ) -> str:
        """Create a shared workspace for collaborative conversations"""
        workspace_id = str(uuid.uuid4())
        
        workspace_data = {
            "workspace_id": workspace_id,
            "name": workspace_name,
            "users": user_ids,
            "created_at": datetime.utcnow().isoformat(),
            "shared_context": {},
            "active_sessions": []
        }
        
        await self.context_manager.redis_client.setex(
            f"workspace:{workspace_id}",
            int(timedelta(days=7).total_seconds()),
            json.dumps(workspace_data)
        )
        
        return workspace_id

# Example usage and testing
async def main():
    """Example usage of the conversation context manager"""
    context_manager = ConversationContextManager()
    await context_manager.initialize()
    
    try:
        # Create a new session
        session_id = await context_manager.create_session(
            user_id="user123",
            metadata={"role": "developer", "project": "nexus-architect"}
        )
        
        # Add conversation turns
        await context_manager.add_conversation_turn(
            session_id=session_id,
            user_message="How do I implement authentication in the system?",
            ai_response="I'll help you implement authentication. Based on your role as a developer...",
            persona_used="security_architect",
            confidence_score=0.95,
            context_used=["security_patterns", "authentication_docs"]
        )
        
        # Get conversation context
        context = await context_manager.get_conversation_context(session_id)
        print(f"Context: {json.dumps(context, indent=2)}")
        
        # Get relevant context for a new query
        relevant_context = await context_manager.get_relevant_context(
            session_id,
            "What are the security best practices?"
        )
        print(f"Relevant context: {relevant_context}")
        
    finally:
        await context_manager.close()

if __name__ == "__main__":
    asyncio.run(main())

