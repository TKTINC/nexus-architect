"""
Nexus Architect Multi-Persona AI Orchestrator
Intelligent routing and collaboration framework for specialized AI personas
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import yaml
import redis
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis.nexus-infrastructure:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TORCHSERVE_URL = os.getenv("TORCHSERVE_URL", "http://torchserve-multi-persona-service.nexus-ai-intelligence:8080")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate.nexus-infrastructure:8080")

# Initialize clients
redis_client = redis.from_url(REDIS_URL)
openai.api_key = OPENAI_API_KEY
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

class PersonaType(str, Enum):
    SECURITY_ARCHITECT = "security_architect"
    PERFORMANCE_ENGINEER = "performance_engineer"
    APPLICATION_ARCHITECT = "application_architect"
    DEVOPS_SPECIALIST = "devops_specialist"
    COMPLIANCE_AUDITOR = "compliance_auditor"

class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_DOMAIN = "multi_domain"

@dataclass
class PersonaConfig:
    id: str
    name: str
    description: str
    domain_expertise: List[str]
    primary_model: str
    fallback_model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    keywords: List[str]
    confidence_boost: float

@dataclass
class QueryAnalysis:
    query: str
    complexity: QueryComplexity
    domains: List[str]
    keywords: List[str]
    confidence_scores: Dict[str, float]
    recommended_personas: List[str]
    requires_collaboration: bool

@dataclass
class PersonaResponse:
    persona_id: str
    response: str
    confidence: float
    reasoning: str
    sources: List[str]
    execution_time: float
    model_used: str

@dataclass
class CollaborationResult:
    query: str
    participating_personas: List[str]
    individual_responses: List[PersonaResponse]
    consensus_response: str
    confidence: float
    collaboration_reasoning: str
    execution_time: float

class PersonaOrchestrator:
    def __init__(self):
        self.personas = {}
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.keyword_vectors = None
        self.load_persona_configurations()
        self.initialize_keyword_vectors()
    
    def load_persona_configurations(self):
        """Load persona configurations from YAML file"""
        try:
            with open('/app/config/persona_definitions.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            for persona_id, persona_data in config['data']['personas.yaml']['personas'].items():
                self.personas[persona_id] = PersonaConfig(
                    id=persona_id,
                    name=persona_data['name'],
                    description=persona_data['description'],
                    domain_expertise=persona_data['domain_expertise']['primary_domains'],
                    primary_model=persona_data['model_configuration']['primary_model'],
                    fallback_model=persona_data['model_configuration']['fallback_model'],
                    temperature=persona_data['model_configuration']['temperature'],
                    max_tokens=persona_data['model_configuration']['max_tokens'],
                    system_prompt=persona_data['model_configuration']['system_prompt'],
                    keywords=config['data']['personas.yaml']['orchestration']['persona_selection']['intent_mapping'].get(f"{persona_id}_queries", {}).get('keywords', []),
                    confidence_boost=config['data']['personas.yaml']['orchestration']['persona_selection']['intent_mapping'].get(f"{persona_id}_queries", {}).get('confidence_boost', 0.0)
                )
            
            logger.info(f"Loaded {len(self.personas)} persona configurations")
        except Exception as e:
            logger.error(f"Error loading persona configurations: {e}")
            raise
    
    def initialize_keyword_vectors(self):
        """Initialize keyword vectors for persona matching"""
        try:
            all_keywords = []
            persona_keywords = {}
            
            for persona_id, persona in self.personas.items():
                keywords_text = " ".join(persona.keywords + persona.domain_expertise)
                all_keywords.append(keywords_text)
                persona_keywords[persona_id] = keywords_text
            
            if all_keywords:
                self.keyword_vectors = self.vectorizer.fit_transform(all_keywords)
                logger.info("Initialized keyword vectors for persona matching")
        except Exception as e:
            logger.error(f"Error initializing keyword vectors: {e}")
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine complexity, domains, and recommended personas"""
        try:
            # Extract keywords and analyze complexity
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarity scores with each persona
            similarity_scores = cosine_similarity(query_vector, self.keyword_vectors)[0]
            confidence_scores = {}
            
            for i, (persona_id, persona) in enumerate(self.personas.items()):
                base_score = similarity_scores[i]
                # Apply confidence boost for keyword matches
                keyword_matches = sum(1 for keyword in persona.keywords if keyword.lower() in query.lower())
                boost = persona.confidence_boost * keyword_matches
                confidence_scores[persona_id] = min(base_score + boost, 1.0)
            
            # Determine query complexity
            complexity = self._determine_complexity(query, confidence_scores)
            
            # Identify domains
            domains = self._identify_domains(query, confidence_scores)
            
            # Recommend personas
            recommended_personas = self._recommend_personas(confidence_scores, complexity)
            
            # Determine if collaboration is needed
            requires_collaboration = self._requires_collaboration(confidence_scores, complexity)
            
            return QueryAnalysis(
                query=query,
                complexity=complexity,
                domains=domains,
                keywords=self._extract_keywords(query),
                confidence_scores=confidence_scores,
                recommended_personas=recommended_personas,
                requires_collaboration=requires_collaboration
            )
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            raise
    
    def _determine_complexity(self, query: str, confidence_scores: Dict[str, float]) -> QueryComplexity:
        """Determine query complexity based on various factors"""
        # Count high-confidence personas
        high_confidence_count = sum(1 for score in confidence_scores.values() if score > 0.7)
        
        # Check for complexity indicators
        complexity_indicators = [
            "architecture", "design", "implement", "integrate", "optimize",
            "security", "performance", "compliance", "deployment", "monitoring"
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query.lower())
        
        if high_confidence_count >= 3 or indicator_count >= 4:
            return QueryComplexity.MULTI_DOMAIN
        elif high_confidence_count >= 2 or indicator_count >= 3:
            return QueryComplexity.COMPLEX
        elif indicator_count >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _identify_domains(self, query: str, confidence_scores: Dict[str, float]) -> List[str]:
        """Identify relevant domains based on confidence scores"""
        domains = []
        for persona_id, score in confidence_scores.items():
            if score > 0.5:
                persona = self.personas[persona_id]
                domains.extend(persona.domain_expertise)
        return list(set(domains))
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query"""
        # Simple keyword extraction - can be enhanced with NLP
        words = query.lower().split()
        keywords = [word for word in words if len(word) > 3]
        return keywords[:10]  # Limit to top 10 keywords
    
    def _recommend_personas(self, confidence_scores: Dict[str, float], complexity: QueryComplexity) -> List[str]:
        """Recommend personas based on confidence scores and complexity"""
        sorted_personas = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        
        if complexity == QueryComplexity.MULTI_DOMAIN:
            return [persona_id for persona_id, score in sorted_personas if score > 0.4][:3]
        elif complexity == QueryComplexity.COMPLEX:
            return [persona_id for persona_id, score in sorted_personas if score > 0.5][:2]
        else:
            return [sorted_personas[0][0]] if sorted_personas[0][1] > 0.3 else []
    
    def _requires_collaboration(self, confidence_scores: Dict[str, float], complexity: QueryComplexity) -> bool:
        """Determine if multi-persona collaboration is needed"""
        high_confidence_count = sum(1 for score in confidence_scores.values() if score > 0.6)
        return complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_DOMAIN] or high_confidence_count >= 2
    
    async def get_persona_response(self, persona_id: str, query: str, context: Dict[str, Any] = None) -> PersonaResponse:
        """Get response from a specific persona"""
        start_time = datetime.now()
        
        try:
            persona = self.personas[persona_id]
            
            # Prepare context-aware prompt
            full_prompt = self._prepare_prompt(persona, query, context)
            
            # Try primary model first
            try:
                response = await self._call_model(persona.primary_model, full_prompt, persona)
                model_used = persona.primary_model
            except Exception as e:
                logger.warning(f"Primary model failed for {persona_id}, trying fallback: {e}")
                response = await self._call_model(persona.fallback_model, full_prompt, persona)
                model_used = persona.fallback_model
            
            # Calculate confidence based on response quality
            confidence = self._calculate_response_confidence(response, query)
            
            # Extract reasoning and sources
            reasoning = self._extract_reasoning(response)
            sources = self._extract_sources(response, persona)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PersonaResponse(
                persona_id=persona_id,
                response=response,
                confidence=confidence,
                reasoning=reasoning,
                sources=sources,
                execution_time=execution_time,
                model_used=model_used
            )
        except Exception as e:
            logger.error(f"Error getting response from persona {persona_id}: {e}")
            raise
    
    def _prepare_prompt(self, persona: PersonaConfig, query: str, context: Dict[str, Any] = None) -> str:
        """Prepare context-aware prompt for persona"""
        prompt_parts = [persona.system_prompt]
        
        if context:
            if context.get('conversation_history'):
                prompt_parts.append(f"Conversation History: {context['conversation_history']}")
            if context.get('project_context'):
                prompt_parts.append(f"Project Context: {context['project_context']}")
            if context.get('user_role'):
                prompt_parts.append(f"User Role: {context['user_role']}")
        
        prompt_parts.append(f"User Query: {query}")
        prompt_parts.append("Please provide a comprehensive response based on your expertise.")
        
        return "\n\n".join(prompt_parts)
    
    async def _call_model(self, model_name: str, prompt: str, persona: PersonaConfig) -> str:
        """Call the specified AI model"""
        if model_name.startswith("gpt-"):
            return await self._call_openai(model_name, prompt, persona)
        elif model_name.startswith("claude-"):
            return await self._call_anthropic(model_name, prompt, persona)
        else:
            return await self._call_local_model(model_name, prompt, persona)
    
    async def _call_openai(self, model_name: str, prompt: str, persona: PersonaConfig) -> str:
        """Call OpenAI API"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=persona.temperature,
                max_tokens=persona.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _call_anthropic(self, model_name: str, prompt: str, persona: PersonaConfig) -> str:
        """Call Anthropic API"""
        try:
            response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=persona.max_tokens,
                temperature=persona.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _call_local_model(self, model_name: str, prompt: str, persona: PersonaConfig) -> str:
        """Call local TorchServe model"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{TORCHSERVE_URL}/predictions/{model_name}",
                    json={
                        "prompt": prompt,
                        "temperature": persona.temperature,
                        "max_tokens": persona.max_tokens
                    },
                    timeout=120
                )
                response.raise_for_status()
                return response.json()["response"]
        except Exception as e:
            logger.error(f"Local model API error: {e}")
            raise
    
    def _calculate_response_confidence(self, response: str, query: str) -> float:
        """Calculate confidence score for response quality"""
        # Simple heuristic-based confidence calculation
        confidence = 0.5  # Base confidence
        
        # Length-based confidence
        if len(response) > 100:
            confidence += 0.1
        if len(response) > 500:
            confidence += 0.1
        
        # Keyword relevance
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        confidence += min(overlap * 0.05, 0.2)
        
        # Structure indicators
        if any(indicator in response.lower() for indicator in ["recommendation", "solution", "approach", "strategy"]):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response"""
        # Simple extraction - look for reasoning indicators
        reasoning_indicators = ["because", "due to", "reasoning", "rationale", "analysis"]
        sentences = response.split('.')
        
        reasoning_sentences = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in reasoning_indicators):
                reasoning_sentences.append(sentence.strip())
        
        return ". ".join(reasoning_sentences[:3]) if reasoning_sentences else "Standard domain expertise applied"
    
    def _extract_sources(self, response: str, persona: PersonaConfig) -> List[str]:
        """Extract or infer sources for the response"""
        # For now, return persona's knowledge sources
        return persona.domain_expertise[:3]
    
    async def collaborate_personas(self, query: str, persona_ids: List[str], context: Dict[str, Any] = None) -> CollaborationResult:
        """Orchestrate collaboration between multiple personas"""
        start_time = datetime.now()
        
        try:
            # Get individual responses
            tasks = [self.get_persona_response(persona_id, query, context) for persona_id in persona_ids]
            individual_responses = await asyncio.gather(*tasks)
            
            # Generate consensus response
            consensus_response = await self._generate_consensus(query, individual_responses)
            
            # Calculate overall confidence
            confidence = self._calculate_collaboration_confidence(individual_responses)
            
            # Generate collaboration reasoning
            collaboration_reasoning = self._generate_collaboration_reasoning(individual_responses)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return CollaborationResult(
                query=query,
                participating_personas=persona_ids,
                individual_responses=individual_responses,
                consensus_response=consensus_response,
                confidence=confidence,
                collaboration_reasoning=collaboration_reasoning,
                execution_time=execution_time
            )
        except Exception as e:
            logger.error(f"Error in persona collaboration: {e}")
            raise
    
    async def _generate_consensus(self, query: str, responses: List[PersonaResponse]) -> str:
        """Generate consensus response from multiple persona responses"""
        try:
            # Prepare consensus prompt
            consensus_prompt = f"""
            Query: {query}
            
            Multiple expert perspectives:
            """
            
            for i, response in enumerate(responses, 1):
                persona_name = self.personas[response.persona_id].name
                consensus_prompt += f"\n{i}. {persona_name}: {response.response}\n"
            
            consensus_prompt += """
            Please synthesize these expert perspectives into a comprehensive, balanced response that:
            1. Integrates the key insights from each expert
            2. Resolves any conflicts or contradictions
            3. Provides a unified recommendation
            4. Maintains the technical depth and accuracy
            """
            
            # Use GPT-4 for consensus generation
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": consensus_prompt}],
                temperature=0.3,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating consensus: {e}")
            # Fallback: combine responses
            return self._fallback_consensus(responses)
    
    def _fallback_consensus(self, responses: List[PersonaResponse]) -> str:
        """Fallback consensus generation by combining responses"""
        consensus_parts = ["Based on expert analysis from multiple perspectives:"]
        
        for response in responses:
            persona_name = self.personas[response.persona_id].name
            consensus_parts.append(f"\n{persona_name} perspective: {response.response[:200]}...")
        
        consensus_parts.append("\nIntegrated recommendation: The experts agree on implementing a comprehensive approach that addresses all identified concerns.")
        
        return "\n".join(consensus_parts)
    
    def _calculate_collaboration_confidence(self, responses: List[PersonaResponse]) -> float:
        """Calculate overall confidence for collaboration result"""
        if not responses:
            return 0.0
        
        # Average individual confidences
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        
        # Boost for multiple expert agreement
        agreement_boost = min(len(responses) * 0.1, 0.3)
        
        return min(avg_confidence + agreement_boost, 1.0)
    
    def _generate_collaboration_reasoning(self, responses: List[PersonaResponse]) -> str:
        """Generate reasoning for collaboration approach"""
        persona_names = [self.personas[r.persona_id].name for r in responses]
        return f"Collaborative analysis from {', '.join(persona_names)} provides comprehensive multi-domain expertise"

# FastAPI Application
app = FastAPI(
    title="Nexus Architect Multi-Persona AI Orchestrator",
    description="Intelligent routing and collaboration framework for specialized AI personas",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = PersonaOrchestrator()

# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query to be processed")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")
    preferred_personas: Optional[List[str]] = Field(None, description="Preferred personas for handling the query")
    require_collaboration: Optional[bool] = Field(None, description="Force multi-persona collaboration")

class QueryResponse(BaseModel):
    query_id: str
    analysis: Dict[str, Any]
    response: str
    personas_used: List[str]
    confidence: float
    execution_time: float
    collaboration_used: bool

class PersonaListResponse(BaseModel):
    personas: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    personas_loaded: int
    models_available: List[str]

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        personas_loaded=len(orchestrator.personas),
        models_available=list(set(p.primary_model for p in orchestrator.personas.values()))
    )

@app.get("/personas", response_model=PersonaListResponse)
async def list_personas():
    """List available personas and their capabilities"""
    personas_info = []
    for persona_id, persona in orchestrator.personas.items():
        personas_info.append({
            "id": persona.id,
            "name": persona.name,
            "description": persona.description,
            "domain_expertise": persona.domain_expertise,
            "primary_model": persona.primary_model
        })
    
    return PersonaListResponse(personas=personas_info)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process query with intelligent persona selection and collaboration"""
    query_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Analyze query
        analysis = await orchestrator.analyze_query(request.query)
        
        # Determine execution strategy
        if request.require_collaboration or (analysis.requires_collaboration and not request.preferred_personas):
            # Multi-persona collaboration
            personas_to_use = request.preferred_personas or analysis.recommended_personas
            result = await orchestrator.collaborate_personas(request.query, personas_to_use, request.context)
            
            response = result.consensus_response
            confidence = result.confidence
            collaboration_used = True
        else:
            # Single persona response
            persona_id = request.preferred_personas[0] if request.preferred_personas else analysis.recommended_personas[0]
            result = await orchestrator.get_persona_response(persona_id, request.query, request.context)
            
            response = result.response
            confidence = result.confidence
            personas_to_use = [persona_id]
            collaboration_used = False
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Cache result for future reference
        background_tasks.add_task(cache_query_result, query_id, request.query, response, analysis)
        
        return QueryResponse(
            query_id=query_id,
            analysis=asdict(analysis),
            response=response,
            personas_used=personas_to_use,
            confidence=confidence,
            execution_time=execution_time,
            collaboration_used=collaboration_used
        )
    except Exception as e:
        logger.error(f"Error processing query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/personas/{persona_id}/query")
async def query_specific_persona(persona_id: str, request: QueryRequest):
    """Query a specific persona directly"""
    try:
        if persona_id not in orchestrator.personas:
            raise HTTPException(status_code=404, detail=f"Persona {persona_id} not found")
        
        result = await orchestrator.get_persona_response(persona_id, request.query, request.context)
        
        return {
            "persona_id": persona_id,
            "response": result.response,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "sources": result.sources,
            "execution_time": result.execution_time,
            "model_used": result.model_used
        }
    except Exception as e:
        logger.error(f"Error querying persona {persona_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/personas")
async def get_persona_analytics():
    """Get analytics on persona usage and performance"""
    try:
        # Get usage statistics from Redis
        analytics = {}
        for persona_id in orchestrator.personas.keys():
            usage_key = f"persona_usage:{persona_id}"
            usage_count = redis_client.get(usage_key) or 0
            analytics[persona_id] = {
                "usage_count": int(usage_count),
                "name": orchestrator.personas[persona_id].name
            }
        
        return {"persona_analytics": analytics}
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def cache_query_result(query_id: str, query: str, response: str, analysis: QueryAnalysis):
    """Cache query result for analytics and future reference"""
    try:
        cache_data = {
            "query_id": query_id,
            "query": query,
            "response": response,
            "analysis": asdict(analysis),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache in Redis with 24-hour expiration
        redis_client.setex(f"query_cache:{query_id}", 86400, json.dumps(cache_data))
        
        # Update usage statistics
        for persona_id in analysis.recommended_personas:
            redis_client.incr(f"persona_usage:{persona_id}")
        
        logger.info(f"Cached query result {query_id}")
    except Exception as e:
        logger.error(f"Error caching query result: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

