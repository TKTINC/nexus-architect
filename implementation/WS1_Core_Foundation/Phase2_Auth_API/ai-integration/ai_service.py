"""
AI Integration Service for Nexus Architect
Provides unified interface for multiple AI providers with role-based responses
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json

import httpx
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

class AIProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FALLBACK = "fallback"

class UserRole(str, Enum):
    ADMIN = "admin"
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    PROJECT_MANAGER = "project_manager"
    EXECUTIVE = "executive"
    VIEWER = "viewer"

class AIMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AIRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_role: UserRole = Field(default=UserRole.VIEWER, description="User's role for context")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    conversation_history: Optional[List[AIMessage]] = Field(default=None, description="Previous messages")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum response tokens")
    temperature: Optional[float] = Field(default=0.7, description="Response creativity (0.0-1.0)")

class AIResponse(BaseModel):
    content: str = Field(..., description="AI response content")
    provider: AIProvider = Field(..., description="AI provider used")
    model: str = Field(..., description="Specific model used")
    tokens_used: Optional[int] = Field(default=None, description="Tokens consumed")
    response_time: float = Field(..., description="Response time in seconds")
    confidence: Optional[float] = Field(default=None, description="Response confidence score")

class AIServiceConfig:
    """Configuration for AI services"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODELS = {
        UserRole.ADMIN: "gpt-4-turbo-preview",
        UserRole.ARCHITECT: "gpt-4-turbo-preview", 
        UserRole.DEVELOPER: "gpt-4",
        UserRole.PROJECT_MANAGER: "gpt-3.5-turbo",
        UserRole.EXECUTIVE: "gpt-4",
        UserRole.VIEWER: "gpt-3.5-turbo"
    }
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    ANTHROPIC_MODELS = {
        UserRole.ADMIN: "claude-3-opus-20240229",
        UserRole.ARCHITECT: "claude-3-opus-20240229",
        UserRole.DEVELOPER: "claude-3-sonnet-20240229", 
        UserRole.PROJECT_MANAGER: "claude-3-sonnet-20240229",
        UserRole.EXECUTIVE: "claude-3-opus-20240229",
        UserRole.VIEWER: "claude-3-haiku-20240307"
    }
    
    # System prompts for different roles
    SYSTEM_PROMPTS = {
        UserRole.ADMIN: """You are Nexus Architect, an advanced AI assistant for system administrators and technical leaders. 
        You have comprehensive knowledge of software architecture, system administration, security, and business strategy.
        Provide detailed technical analysis, strategic recommendations, and actionable insights.
        Consider both technical excellence and business impact in your responses.""",
        
        UserRole.ARCHITECT: """You are Nexus Architect, a senior software architecture AI assistant.
        You specialize in system design, architectural patterns, scalability, performance optimization, and technical decision-making.
        Provide detailed architectural analysis, design recommendations, technology assessments, and best practices.
        Focus on long-term maintainability, scalability, and technical excellence.""",
        
        UserRole.DEVELOPER: """You are Nexus Architect, a helpful coding and development AI assistant.
        You specialize in software development, code review, debugging, testing, and implementation guidance.
        Provide practical code solutions, development best practices, and technical guidance.
        Focus on code quality, maintainability, and developer productivity.""",
        
        UserRole.PROJECT_MANAGER: """You are Nexus Architect, a project management AI assistant.
        You specialize in project planning, resource allocation, timeline management, and team coordination.
        Provide project insights, planning recommendations, risk assessments, and progress tracking guidance.
        Focus on delivery success, team efficiency, and stakeholder communication.""",
        
        UserRole.EXECUTIVE: """You are Nexus Architect, an executive AI assistant for business and technology leadership.
        You specialize in strategic planning, business impact analysis, ROI assessment, and high-level decision support.
        Provide strategic insights, business recommendations, and executive-level analysis.
        Focus on business value, competitive advantage, and organizational impact.""",
        
        UserRole.VIEWER: """You are Nexus Architect, a helpful AI assistant.
        You provide general information and guidance within appropriate access permissions.
        Offer helpful insights while respecting security and access boundaries.
        Focus on being informative and supportive within your scope."""
    }

class OpenAIService:
    """OpenAI integration service"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0
        )
    
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate response using OpenAI API"""
        start_time = datetime.utcnow()
        
        try:
            model = AIServiceConfig.OPENAI_MODELS.get(request.user_role, "gpt-3.5-turbo")
            system_prompt = AIServiceConfig.SYSTEM_PROMPTS.get(request.user_role, "")
            
            # Build messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history if provided
            if request.conversation_history:
                for msg in request.conversation_history[-10:]:  # Last 10 messages
                    messages.append({"role": msg.role, "content": msg.content})
            
            # Add current message
            messages.append({"role": "user", "content": request.message})
            
            # Make API request
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": False
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response
            content = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AIResponse(
                content=content,
                provider=AIProvider.OPENAI,
                model=model,
                tokens_used=tokens_used,
                response_time=response_time,
                confidence=0.9  # OpenAI doesn't provide confidence scores
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class AnthropicService:
    """Anthropic Claude integration service"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            timeout=60.0
        )
    
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate response using Anthropic API"""
        start_time = datetime.utcnow()
        
        try:
            model = AIServiceConfig.ANTHROPIC_MODELS.get(request.user_role, "claude-3-haiku-20240307")
            system_prompt = AIServiceConfig.SYSTEM_PROMPTS.get(request.user_role, "")
            
            # Build messages for Anthropic format
            messages = []
            
            # Add conversation history if provided
            if request.conversation_history:
                for msg in request.conversation_history[-10:]:  # Last 10 messages
                    if msg.role != "system":  # Anthropic handles system separately
                        messages.append({"role": msg.role, "content": msg.content})
            
            # Add current message
            messages.append({"role": "user", "content": request.message})
            
            # Make API request
            response = await self.client.post(
                "/v1/messages",
                json={
                    "model": model,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "system": system_prompt,
                    "messages": messages
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract response
            content = data["content"][0]["text"]
            tokens_used = data.get("usage", {}).get("output_tokens", 0)
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AIResponse(
                content=content,
                provider=AIProvider.ANTHROPIC,
                model=model,
                tokens_used=tokens_used,
                response_time=response_time,
                confidence=0.9  # Anthropic doesn't provide confidence scores
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

class FallbackService:
    """Fallback service when AI providers are unavailable"""
    
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate fallback response"""
        start_time = datetime.utcnow()
        
        # Role-specific fallback responses
        fallback_responses = {
            UserRole.ADMIN: "I understand you're looking for administrative guidance. While AI services are currently unavailable, I recommend checking the system documentation or contacting technical support for immediate assistance.",
            
            UserRole.ARCHITECT: "I see you're seeking architectural guidance. Although AI services are temporarily unavailable, consider reviewing our architectural patterns documentation or consulting with senior team members for immediate decisions.",
            
            UserRole.DEVELOPER: "I understand you need development assistance. While AI services are currently down, you might find help in our code documentation, team knowledge base, or by reaching out to senior developers.",
            
            UserRole.PROJECT_MANAGER: "I recognize you're looking for project management insights. With AI services temporarily unavailable, consider checking project templates, consulting team leads, or reviewing historical project data.",
            
            UserRole.EXECUTIVE: "I understand you need strategic insights. While AI services are currently unavailable, consider reviewing business metrics dashboards or consulting with department heads for immediate strategic decisions.",
            
            UserRole.VIEWER: "I appreciate your question. AI services are currently unavailable, but you may find helpful information in our documentation or by contacting support."
        }
        
        content = fallback_responses.get(
            request.user_role,
            "I apologize, but AI services are currently unavailable. Please try again later or contact support for assistance."
        )
        
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        return AIResponse(
            content=content,
            provider=AIProvider.FALLBACK,
            model="fallback",
            tokens_used=0,
            response_time=response_time,
            confidence=0.5
        )

class AIService:
    """Main AI service that coordinates multiple providers"""
    
    def __init__(self):
        self.openai_service = None
        self.anthropic_service = None
        self.fallback_service = FallbackService()
        
        # Initialize available services
        if AIServiceConfig.OPENAI_API_KEY:
            self.openai_service = OpenAIService(
                AIServiceConfig.OPENAI_API_KEY,
                AIServiceConfig.OPENAI_BASE_URL
            )
            logger.info("OpenAI service initialized")
        
        if AIServiceConfig.ANTHROPIC_API_KEY:
            self.anthropic_service = AnthropicService(
                AIServiceConfig.ANTHROPIC_API_KEY,
                AIServiceConfig.ANTHROPIC_BASE_URL
            )
            logger.info("Anthropic service initialized")
        
        if not self.openai_service and not self.anthropic_service:
            logger.warning("No AI services available - using fallback only")
    
    async def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate AI response using best available provider"""
        
        # Determine preferred provider based on user role and availability
        preferred_providers = self._get_preferred_providers(request.user_role)
        
        for provider in preferred_providers:
            try:
                if provider == AIProvider.OPENAI and self.openai_service:
                    return await self.openai_service.generate_response(request)
                elif provider == AIProvider.ANTHROPIC and self.anthropic_service:
                    return await self.anthropic_service.generate_response(request)
            except Exception as e:
                logger.error(f"Provider {provider} failed: {e}")
                continue
        
        # Fallback if all providers fail
        logger.warning("All AI providers failed, using fallback")
        return await self.fallback_service.generate_response(request)
    
    def _get_preferred_providers(self, user_role: UserRole) -> List[AIProvider]:
        """Get preferred provider order based on user role"""
        
        # High-privilege roles prefer more capable models
        if user_role in [UserRole.ADMIN, UserRole.ARCHITECT, UserRole.EXECUTIVE]:
            return [AIProvider.OPENAI, AIProvider.ANTHROPIC, AIProvider.FALLBACK]
        else:
            return [AIProvider.ANTHROPIC, AIProvider.OPENAI, AIProvider.FALLBACK]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of AI services"""
        health = {
            "openai": {"available": False, "status": "unavailable"},
            "anthropic": {"available": False, "status": "unavailable"},
            "fallback": {"available": True, "status": "available"}
        }
        
        # Test OpenAI
        if self.openai_service:
            try:
                test_request = AIRequest(
                    message="Health check",
                    user_role=UserRole.VIEWER,
                    max_tokens=10
                )
                await self.openai_service.generate_response(test_request)
                health["openai"] = {"available": True, "status": "healthy"}
            except Exception as e:
                health["openai"] = {"available": False, "status": f"error: {str(e)}"}
        
        # Test Anthropic
        if self.anthropic_service:
            try:
                test_request = AIRequest(
                    message="Health check",
                    user_role=UserRole.VIEWER,
                    max_tokens=10
                )
                await self.anthropic_service.generate_response(test_request)
                health["anthropic"] = {"available": True, "status": "healthy"}
            except Exception as e:
                health["anthropic"] = {"available": False, "status": f"error: {str(e)}"}
        
        return health

# Global AI service instance
ai_service = AIService()

