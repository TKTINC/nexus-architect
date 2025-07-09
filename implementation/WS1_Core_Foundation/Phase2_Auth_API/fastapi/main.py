"""
Nexus Architect FastAPI Application
Core API foundation with authentication, authorization, and basic AI integration
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import jwt
import httpx
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, generate_latest
import uvicorn

# Configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak.nexus-auth:8080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "nexus-architect")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "nexus-api")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "nexus-api-secret-2024")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://nexus:NexusDB2024@postgresql-primary.nexus-infrastructure:5432/nexus")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis-cluster.nexus-infrastructure:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('nexus_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('nexus_api_request_duration_seconds', 'Request duration')

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    roles = Column(Text)  # JSON string of roles
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    title = Column(String)
    messages = Column(Text)  # JSON string of messages
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Redis connection
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client
    redis_client = redis.from_url(REDIS_URL)
    logger.info("FastAPI application started")
    yield
    # Shutdown
    if redis_client:
        await redis_client.close()
    logger.info("FastAPI application stopped")

# FastAPI app
app = FastAPI(
    title="Nexus Architect API",
    description="Core API for Nexus Architect AI-powered development platform",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.nexus-architect.local",
        "http://localhost:3000",
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.nexus-architect.local", "localhost", "127.0.0.1"]
)

# Security
security = HTTPBearer()

# Pydantic models
class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
    roles: List[str] = []
    scopes: List[str] = []

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    roles: List[str]
    is_active: bool

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    response_time: float
    model_used: str

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication functions
async def get_keycloak_public_key():
    """Get Keycloak public key for JWT verification"""
    try:
        cache_key = f"keycloak_public_key_{KEYCLOAK_REALM}"
        cached_key = await redis_client.get(cache_key)
        
        if cached_key:
            return cached_key.decode()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid_connect/certs"
            )
            response.raise_for_status()
            keys = response.json()
            
            # Extract the public key (simplified - in production, handle multiple keys)
            public_key = keys["keys"][0]["x5c"][0]
            
            # Cache for 1 hour
            await redis_client.setex(cache_key, 3600, public_key)
            
            return public_key
    except Exception as e:
        logger.error(f"Failed to get Keycloak public key: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
    """Verify JWT token and extract user information"""
    try:
        token = credentials.credentials
        
        # Get public key for verification
        public_key = await get_keycloak_public_key()
        
        # Decode and verify token
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=KEYCLOAK_CLIENT_ID,
            issuer=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}"
        )
        
        # Extract user information
        user_id = payload.get("sub")
        username = payload.get("preferred_username")
        roles = payload.get("realm_access", {}).get("roles", [])
        scopes = payload.get("scope", "").split()
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID"
            )
        
        return TokenData(
            user_id=user_id,
            username=username,
            roles=roles,
            scopes=scopes
        )
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )

def require_role(required_role: str):
    """Dependency to require specific role"""
    def role_checker(token_data: TokenData = Depends(verify_token)):
        if required_role not in token_data.roles and "admin" not in token_data.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        return token_data
    return role_checker

# AI Integration
class AIService:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_API_KEY:
            import openai
            self.openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        if ANTHROPIC_API_KEY:
            import anthropic
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    
    async def generate_response(self, message: str, context: Dict[str, Any] = None, user_role: str = "user") -> str:
        """Generate AI response based on user message and context"""
        try:
            # Determine AI model based on user role and context
            if user_role in ["architect", "admin"]:
                model = "gpt-4" if self.openai_client else "claude-3-opus"
            else:
                model = "gpt-3.5-turbo" if self.openai_client else "claude-3-sonnet"
            
            # Build system prompt based on user role
            system_prompts = {
                "architect": "You are a senior software architect assistant. Provide detailed technical analysis, architectural recommendations, and best practices.",
                "developer": "You are a helpful coding assistant. Provide practical code solutions, debugging help, and development guidance.",
                "project_manager": "You are a project management assistant. Focus on project planning, resource allocation, and progress tracking.",
                "executive": "You are an executive assistant. Provide high-level insights, business impact analysis, and strategic recommendations.",
                "admin": "You are a system administrator assistant. Provide comprehensive technical and business guidance.",
                "viewer": "You are a helpful assistant. Provide general information and guidance within your access permissions."
            }
            
            system_prompt = system_prompts.get(user_role, system_prompts["viewer"])
            
            # Generate response using available AI service
            if self.openai_client and model.startswith("gpt"):
                response = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            elif self.anthropic_client and model.startswith("claude"):
                response = await self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=1000,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[{"role": "user", "content": message}]
                )
                return response.content[0].text
            
            else:
                # Fallback response when no AI service is available
                return f"I understand you're asking about: {message}. However, AI services are currently unavailable. Please try again later or contact support."
                
        except Exception as e:
            logger.error(f"AI service error: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."

ai_service = AIService()

# API Routes
@app.get("/")
async def root():
    return {"message": "Nexus Architect API", "version": "1.0.0", "status": "operational"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        # Check Redis connection
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "healthy",
                "redis": "healthy",
                "ai_services": "available" if (OPENAI_API_KEY or ANTHROPIC_API_KEY) else "unavailable"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/api/v1/user/profile", response_model=UserResponse)
async def get_user_profile(
    token_data: TokenData = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get current user profile"""
    user = db.query(User).filter(User.id == token_data.user_id).first()
    
    if not user:
        # Create user if doesn't exist
        user = User(
            id=token_data.user_id,
            username=token_data.username,
            email=f"{token_data.username}@nexus-architect.local",
            full_name=token_data.username,
            roles=",".join(token_data.roles),
            last_login=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        roles=user.roles.split(",") if user.roles else [],
        is_active=user.is_active
    )

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    token_data: TokenData = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Chat with AI assistant"""
    start_time = datetime.utcnow()
    
    try:
        # Determine user's primary role for AI context
        primary_role = "viewer"
        if "admin" in token_data.roles:
            primary_role = "admin"
        elif "architect" in token_data.roles:
            primary_role = "architect"
        elif "developer" in token_data.roles:
            primary_role = "developer"
        elif "project_manager" in token_data.roles:
            primary_role = "project_manager"
        elif "executive" in token_data.roles:
            primary_role = "executive"
        
        # Generate AI response
        ai_response = await ai_service.generate_response(
            request.message,
            request.context,
            primary_role
        )
        
        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Save conversation (simplified - in production, implement proper conversation management)
        conversation_id = request.conversation_id or f"conv_{token_data.user_id}_{int(datetime.utcnow().timestamp())}"
        
        return ChatResponse(
            message=ai_response,
            conversation_id=conversation_id,
            response_time=response_time,
            model_used="gpt-4" if OPENAI_API_KEY else "claude-3" if ANTHROPIC_API_KEY else "fallback"
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request"
        )

@app.get("/api/v1/admin/users")
async def list_users(
    token_data: TokenData = Depends(require_role("admin")),
    db: Session = Depends(get_db)
):
    """List all users (admin only)"""
    users = db.query(User).all()
    return [
        UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            roles=user.roles.split(",") if user.roles else [],
            is_active=user.is_active
        )
        for user in users
    ]

@app.get("/api/v1/architect/analyze")
async def architectural_analysis(
    token_data: TokenData = Depends(require_role("architect")),
):
    """Architectural analysis endpoint (architect role required)"""
    return {
        "message": "Architectural analysis feature",
        "user": token_data.username,
        "available_tools": [
            "Code Architecture Analysis",
            "Performance Assessment", 
            "Security Review",
            "Scalability Planning"
        ]
    }

@app.get("/api/v1/developer/tools")
async def developer_tools(
    token_data: TokenData = Depends(require_role("developer")),
):
    """Developer tools endpoint (developer role required)"""
    return {
        "message": "Developer tools and utilities",
        "user": token_data.username,
        "available_tools": [
            "Code Review Assistant",
            "Bug Analysis",
            "Test Generation",
            "Documentation Helper"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

