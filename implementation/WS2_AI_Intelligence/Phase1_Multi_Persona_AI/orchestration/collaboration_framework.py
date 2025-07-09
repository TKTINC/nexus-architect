"""
Nexus Architect Multi-Persona Collaboration Framework
Advanced collaboration, consensus building, and conflict resolution for AI personas
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborationType(str, Enum):
    SEQUENTIAL = "sequential"  # Personas respond in sequence
    PARALLEL = "parallel"     # Personas respond simultaneously
    HIERARCHICAL = "hierarchical"  # Lead persona coordinates others
    CONSENSUS = "consensus"   # All personas work toward agreement

class ConflictType(str, Enum):
    TECHNICAL_DISAGREEMENT = "technical_disagreement"
    PRIORITY_CONFLICT = "priority_conflict"
    APPROACH_DIFFERENCE = "approach_difference"
    SCOPE_OVERLAP = "scope_overlap"
    RESOURCE_CONTENTION = "resource_contention"

@dataclass
class CollaborationContext:
    query: str
    user_role: str
    project_context: Dict[str, Any]
    constraints: List[str]
    priorities: List[str]
    timeline: Optional[str]
    budget_constraints: Optional[str]

@dataclass
class PersonaContribution:
    persona_id: str
    response: str
    confidence: float
    key_points: List[str]
    recommendations: List[str]
    concerns: List[str]
    dependencies: List[str]
    estimated_effort: Optional[str]
    risk_assessment: Optional[str]

@dataclass
class ConflictResolution:
    conflict_type: ConflictType
    conflicting_personas: List[str]
    conflict_description: str
    resolution_strategy: str
    resolved_recommendation: str
    confidence: float

@dataclass
class CollaborationResult:
    collaboration_id: str
    query: str
    collaboration_type: CollaborationType
    participating_personas: List[str]
    individual_contributions: List[PersonaContribution]
    conflicts_identified: List[ConflictResolution]
    consensus_response: str
    implementation_plan: Dict[str, Any]
    confidence: float
    collaboration_quality: float
    execution_time: float

class CollaborationFramework:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        self.collaboration_history = {}
        self.conflict_patterns = {}
        
    async def orchestrate_collaboration(
        self, 
        context: CollaborationContext, 
        persona_ids: List[str],
        collaboration_type: CollaborationType = CollaborationType.CONSENSUS
    ) -> CollaborationResult:
        """Orchestrate multi-persona collaboration with advanced coordination"""
        collaboration_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting collaboration {collaboration_id} with {len(persona_ids)} personas")
        
        try:
            # Phase 1: Individual Contributions
            individual_contributions = await self._gather_individual_contributions(
                context, persona_ids
            )
            
            # Phase 2: Conflict Detection and Analysis
            conflicts = await self._detect_and_analyze_conflicts(
                individual_contributions, context
            )
            
            # Phase 3: Conflict Resolution
            resolved_conflicts = await self._resolve_conflicts(
                conflicts, individual_contributions, context
            )
            
            # Phase 4: Consensus Building
            consensus_response = await self._build_consensus(
                individual_contributions, resolved_conflicts, context
            )
            
            # Phase 5: Implementation Planning
            implementation_plan = await self._create_implementation_plan(
                individual_contributions, consensus_response, context
            )
            
            # Calculate collaboration quality metrics
            collaboration_quality = self._calculate_collaboration_quality(
                individual_contributions, conflicts, resolved_conflicts
            )
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                individual_contributions, resolved_conflicts
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = CollaborationResult(
                collaboration_id=collaboration_id,
                query=context.query,
                collaboration_type=collaboration_type,
                participating_personas=persona_ids,
                individual_contributions=individual_contributions,
                conflicts_identified=resolved_conflicts,
                consensus_response=consensus_response,
                implementation_plan=implementation_plan,
                confidence=confidence,
                collaboration_quality=collaboration_quality,
                execution_time=execution_time
            )
            
            # Store collaboration for learning
            await self._store_collaboration_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in collaboration {collaboration_id}: {e}")
            raise
    
    async def _gather_individual_contributions(
        self, 
        context: CollaborationContext, 
        persona_ids: List[str]
    ) -> List[PersonaContribution]:
        """Gather individual contributions from each persona"""
        contributions = []
        
        for persona_id in persona_ids:
            try:
                # Prepare persona-specific context
                persona_context = await self._prepare_persona_context(context, persona_id)
                
                # Get persona response
                response = await self.orchestrator.get_persona_response(
                    persona_id, context.query, persona_context
                )
                
                # Extract structured information from response
                contribution = await self._extract_contribution_structure(
                    persona_id, response, context
                )
                
                contributions.append(contribution)
                
            except Exception as e:
                logger.error(f"Error getting contribution from {persona_id}: {e}")
                continue
        
        return contributions
    
    async def _prepare_persona_context(
        self, 
        context: CollaborationContext, 
        persona_id: str
    ) -> Dict[str, Any]:
        """Prepare persona-specific context for collaboration"""
        persona_context = {
            "collaboration_mode": True,
            "user_role": context.user_role,
            "project_context": context.project_context,
            "constraints": context.constraints,
            "priorities": context.priorities
        }
        
        # Add persona-specific instructions
        persona = self.orchestrator.personas[persona_id]
        persona_context["collaboration_instructions"] = f"""
        You are participating in a multi-expert collaboration as a {persona.name}.
        Other experts will also provide their perspectives on this query.
        
        Please provide:
        1. Your expert analysis and recommendations
        2. Key concerns or risks from your domain perspective
        3. Dependencies or prerequisites you identify
        4. Estimated effort or timeline if applicable
        5. Any potential conflicts with other domains
        
        Be specific and actionable in your recommendations.
        """
        
        return persona_context
    
    async def _extract_contribution_structure(
        self, 
        persona_id: str, 
        response, 
        context: CollaborationContext
    ) -> PersonaContribution:
        """Extract structured information from persona response"""
        # Use AI to extract structured information
        extraction_prompt = f"""
        Extract structured information from this expert response:
        
        Response: {response.response}
        
        Extract and format as JSON:
        {{
            "key_points": ["list of main points"],
            "recommendations": ["list of specific recommendations"],
            "concerns": ["list of concerns or risks"],
            "dependencies": ["list of dependencies or prerequisites"],
            "estimated_effort": "effort estimate if mentioned",
            "risk_assessment": "risk level if mentioned"
        }}
        
        Return only valid JSON.
        """
        
        try:
            import openai
            extraction_response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            
            extracted_data = json.loads(extraction_response.choices[0].message.content)
            
            return PersonaContribution(
                persona_id=persona_id,
                response=response.response,
                confidence=response.confidence,
                key_points=extracted_data.get("key_points", []),
                recommendations=extracted_data.get("recommendations", []),
                concerns=extracted_data.get("concerns", []),
                dependencies=extracted_data.get("dependencies", []),
                estimated_effort=extracted_data.get("estimated_effort"),
                risk_assessment=extracted_data.get("risk_assessment")
            )
            
        except Exception as e:
            logger.error(f"Error extracting contribution structure: {e}")
            # Fallback to basic extraction
            return PersonaContribution(
                persona_id=persona_id,
                response=response.response,
                confidence=response.confidence,
                key_points=[response.response[:200] + "..."],
                recommendations=["See full response for recommendations"],
                concerns=[],
                dependencies=[],
                estimated_effort=None,
                risk_assessment=None
            )
    
    async def _detect_and_analyze_conflicts(
        self, 
        contributions: List[PersonaContribution], 
        context: CollaborationContext
    ) -> List[Dict[str, Any]]:
        """Detect and analyze conflicts between persona contributions"""
        conflicts = []
        
        # Compare recommendations for conflicts
        for i, contrib1 in enumerate(contributions):
            for j, contrib2 in enumerate(contributions[i+1:], i+1):
                conflict = await self._analyze_contribution_conflict(
                    contrib1, contrib2, context
                )
                if conflict:
                    conflicts.append(conflict)
        
        # Detect priority conflicts
        priority_conflicts = self._detect_priority_conflicts(contributions, context)
        conflicts.extend(priority_conflicts)
        
        # Detect resource conflicts
        resource_conflicts = self._detect_resource_conflicts(contributions, context)
        conflicts.extend(resource_conflicts)
        
        return conflicts
    
    async def _analyze_contribution_conflict(
        self, 
        contrib1: PersonaContribution, 
        contrib2: PersonaContribution, 
        context: CollaborationContext
    ) -> Optional[Dict[str, Any]]:
        """Analyze potential conflict between two contributions"""
        # Use semantic similarity to detect conflicts
        try:
            # Combine recommendations for comparison
            text1 = " ".join(contrib1.recommendations)
            text2 = " ".join(contrib2.recommendations)
            
            if not text1 or not text2:
                return None
            
            # Calculate semantic similarity
            vectors = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Low similarity might indicate conflict
            if similarity < 0.3:
                # Use AI to analyze the potential conflict
                conflict_analysis_prompt = f"""
                Analyze these two expert recommendations for potential conflicts:
                
                Expert 1 ({contrib1.persona_id}): {text1}
                Expert 2 ({contrib2.persona_id}): {text2}
                
                Determine if there is a genuine conflict and classify it:
                - technical_disagreement: Different technical approaches
                - priority_conflict: Different priorities or focus areas
                - approach_difference: Different methodological approaches
                - scope_overlap: Overlapping responsibilities
                - resource_contention: Competing for same resources
                
                Return JSON:
                {{
                    "has_conflict": true/false,
                    "conflict_type": "type if conflict exists",
                    "description": "description of the conflict",
                    "severity": "low/medium/high"
                }}
                """
                
                import openai
                analysis_response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[{"role": "user", "content": conflict_analysis_prompt}],
                    temperature=0.1,
                    max_tokens=512
                )
                
                analysis = json.loads(analysis_response.choices[0].message.content)
                
                if analysis.get("has_conflict"):
                    return {
                        "type": analysis.get("conflict_type"),
                        "personas": [contrib1.persona_id, contrib2.persona_id],
                        "description": analysis.get("description"),
                        "severity": analysis.get("severity"),
                        "similarity_score": similarity
                    }
            
        except Exception as e:
            logger.error(f"Error analyzing conflict: {e}")
        
        return None
    
    def _detect_priority_conflicts(
        self, 
        contributions: List[PersonaContribution], 
        context: CollaborationContext
    ) -> List[Dict[str, Any]]:
        """Detect conflicts in priorities between personas"""
        conflicts = []
        
        # Extract priority keywords from each contribution
        priority_keywords = {
            "security": ["security", "secure", "protection", "vulnerability", "threat"],
            "performance": ["performance", "speed", "optimization", "latency", "throughput"],
            "compliance": ["compliance", "regulation", "audit", "policy", "governance"],
            "cost": ["cost", "budget", "expense", "resource", "efficiency"],
            "timeline": ["timeline", "schedule", "deadline", "urgent", "priority"]
        }
        
        persona_priorities = {}
        for contrib in contributions:
            priorities = []
            text = contrib.response.lower()
            
            for priority, keywords in priority_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text)
                if score > 0:
                    priorities.append((priority, score))
            
            persona_priorities[contrib.persona_id] = sorted(priorities, key=lambda x: x[1], reverse=True)
        
        # Check for conflicting top priorities
        top_priorities = {pid: priorities[0][0] if priorities else None 
                         for pid, priorities in persona_priorities.items()}
        
        unique_priorities = set(p for p in top_priorities.values() if p)
        if len(unique_priorities) > 2:
            conflicts.append({
                "type": "priority_conflict",
                "personas": list(top_priorities.keys()),
                "description": f"Multiple conflicting priorities: {', '.join(unique_priorities)}",
                "severity": "medium"
            })
        
        return conflicts
    
    def _detect_resource_conflicts(
        self, 
        contributions: List[PersonaContribution], 
        context: CollaborationContext
    ) -> List[Dict[str, Any]]:
        """Detect resource allocation conflicts"""
        conflicts = []
        
        # Simple heuristic: look for mentions of same resources
        resource_keywords = ["team", "developer", "budget", "time", "infrastructure", "server"]
        
        resource_mentions = {}
        for contrib in contributions:
            text = contrib.response.lower()
            for resource in resource_keywords:
                if resource in text:
                    if resource not in resource_mentions:
                        resource_mentions[resource] = []
                    resource_mentions[resource].append(contrib.persona_id)
        
        # Check for multiple personas mentioning same resources
        for resource, personas in resource_mentions.items():
            if len(personas) > 1:
                conflicts.append({
                    "type": "resource_contention",
                    "personas": personas,
                    "description": f"Multiple experts require {resource} resources",
                    "severity": "low"
                })
        
        return conflicts
    
    async def _resolve_conflicts(
        self, 
        conflicts: List[Dict[str, Any]], 
        contributions: List[PersonaContribution], 
        context: CollaborationContext
    ) -> List[ConflictResolution]:
        """Resolve identified conflicts through various strategies"""
        resolutions = []
        
        for conflict in conflicts:
            try:
                resolution = await self._resolve_single_conflict(
                    conflict, contributions, context
                )
                resolutions.append(resolution)
            except Exception as e:
                logger.error(f"Error resolving conflict: {e}")
                continue
        
        return resolutions
    
    async def _resolve_single_conflict(
        self, 
        conflict: Dict[str, Any], 
        contributions: List[PersonaContribution], 
        context: CollaborationContext
    ) -> ConflictResolution:
        """Resolve a single conflict using appropriate strategy"""
        conflict_type = ConflictType(conflict["type"])
        conflicting_personas = conflict["personas"]
        
        # Get contributions from conflicting personas
        conflicting_contributions = [
            c for c in contributions if c.persona_id in conflicting_personas
        ]
        
        # Choose resolution strategy based on conflict type
        if conflict_type == ConflictType.TECHNICAL_DISAGREEMENT:
            resolution = await self._resolve_technical_disagreement(
                conflicting_contributions, context
            )
        elif conflict_type == ConflictType.PRIORITY_CONFLICT:
            resolution = await self._resolve_priority_conflict(
                conflicting_contributions, context
            )
        elif conflict_type == ConflictType.APPROACH_DIFFERENCE:
            resolution = await self._resolve_approach_difference(
                conflicting_contributions, context
            )
        else:
            resolution = await self._resolve_generic_conflict(
                conflicting_contributions, context
            )
        
        return ConflictResolution(
            conflict_type=conflict_type,
            conflicting_personas=conflicting_personas,
            conflict_description=conflict["description"],
            resolution_strategy=resolution["strategy"],
            resolved_recommendation=resolution["recommendation"],
            confidence=resolution["confidence"]
        )
    
    async def _resolve_technical_disagreement(
        self, 
        contributions: List[PersonaContribution], 
        context: CollaborationContext
    ) -> Dict[str, Any]:
        """Resolve technical disagreements through expert arbitration"""
        resolution_prompt = f"""
        Resolve this technical disagreement between experts:
        
        Context: {context.query}
        User Role: {context.user_role}
        Constraints: {', '.join(context.constraints)}
        
        Expert Opinions:
        """
        
        for contrib in contributions:
            persona_name = self.orchestrator.personas[contrib.persona_id].name
            resolution_prompt += f"\n{persona_name}: {contrib.response[:300]}..."
        
        resolution_prompt += """
        
        As a senior technical leader, provide a balanced resolution that:
        1. Acknowledges the validity of each expert's concerns
        2. Proposes a hybrid approach or clear decision criteria
        3. Explains the reasoning for the recommended approach
        4. Addresses implementation considerations
        
        Return JSON:
        {
            "strategy": "description of resolution strategy",
            "recommendation": "specific recommendation",
            "confidence": 0.0-1.0
        }
        """
        
        import openai
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": resolution_prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def _resolve_priority_conflict(
        self, 
        contributions: List[PersonaContribution], 
        context: CollaborationContext
    ) -> Dict[str, Any]:
        """Resolve priority conflicts based on context and constraints"""
        # Use context to determine priority hierarchy
        priority_weights = {
            "security": 0.9 if "security" in context.constraints else 0.7,
            "performance": 0.8 if "performance" in context.constraints else 0.6,
            "compliance": 0.9 if "compliance" in context.constraints else 0.5,
            "cost": 0.7 if context.budget_constraints else 0.4,
            "timeline": 0.8 if context.timeline else 0.5
        }
        
        # Simple resolution based on weighted priorities
        return {
            "strategy": "Context-based priority weighting",
            "recommendation": "Implement phased approach addressing highest-weighted priorities first",
            "confidence": 0.7
        }
    
    async def _resolve_approach_difference(
        self, 
        contributions: List[PersonaContribution], 
        context: CollaborationContext
    ) -> Dict[str, Any]:
        """Resolve different methodological approaches"""
        return {
            "strategy": "Hybrid approach integration",
            "recommendation": "Combine complementary aspects of different approaches",
            "confidence": 0.6
        }
    
    async def _resolve_generic_conflict(
        self, 
        contributions: List[PersonaContribution], 
        context: CollaborationContext
    ) -> Dict[str, Any]:
        """Generic conflict resolution strategy"""
        return {
            "strategy": "Consensus building through compromise",
            "recommendation": "Find middle ground that addresses core concerns of all parties",
            "confidence": 0.5
        }
    
    async def _build_consensus(
        self, 
        contributions: List[PersonaContribution], 
        resolutions: List[ConflictResolution], 
        context: CollaborationContext
    ) -> str:
        """Build consensus response incorporating all contributions and resolutions"""
        consensus_prompt = f"""
        Build a comprehensive consensus response that integrates multiple expert perspectives:
        
        Query: {context.query}
        Context: {context.project_context}
        User Role: {context.user_role}
        
        Expert Contributions:
        """
        
        for contrib in contributions:
            persona_name = self.orchestrator.personas[contrib.persona_id].name
            consensus_prompt += f"\n{persona_name}:\n"
            consensus_prompt += f"  Key Points: {', '.join(contrib.key_points[:3])}\n"
            consensus_prompt += f"  Recommendations: {', '.join(contrib.recommendations[:3])}\n"
            if contrib.concerns:
                consensus_prompt += f"  Concerns: {', '.join(contrib.concerns[:2])}\n"
        
        if resolutions:
            consensus_prompt += "\nConflict Resolutions:\n"
            for resolution in resolutions:
                consensus_prompt += f"- {resolution.resolution_strategy}: {resolution.resolved_recommendation}\n"
        
        consensus_prompt += """
        
        Create a unified response that:
        1. Synthesizes the expert insights into a coherent recommendation
        2. Addresses the resolved conflicts appropriately
        3. Provides clear implementation guidance
        4. Maintains technical accuracy and depth
        5. Is actionable for the user's role and context
        
        Structure the response with clear sections and actionable recommendations.
        """
        
        import openai
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": consensus_prompt}],
            temperature=0.3,
            max_tokens=2048
        )
        
        return response.choices[0].message.content
    
    async def _create_implementation_plan(
        self, 
        contributions: List[PersonaContribution], 
        consensus_response: str, 
        context: CollaborationContext
    ) -> Dict[str, Any]:
        """Create detailed implementation plan from consensus"""
        # Extract implementation steps from consensus
        plan_prompt = f"""
        Create a detailed implementation plan from this consensus recommendation:
        
        {consensus_response}
        
        Context:
        - User Role: {context.user_role}
        - Timeline: {context.timeline or 'Not specified'}
        - Constraints: {', '.join(context.constraints)}
        
        Return JSON with:
        {{
            "phases": [
                {{
                    "name": "phase name",
                    "duration": "estimated duration",
                    "tasks": ["list of tasks"],
                    "dependencies": ["dependencies"],
                    "responsible_roles": ["roles"],
                    "deliverables": ["deliverables"]
                }}
            ],
            "risks": ["identified risks"],
            "success_criteria": ["success metrics"],
            "resource_requirements": ["required resources"]
        }}
        """
        
        try:
            import openai
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": plan_prompt}],
                temperature=0.2,
                max_tokens=1536
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error creating implementation plan: {e}")
            return {
                "phases": [{"name": "Implementation", "tasks": ["Execute consensus recommendation"]}],
                "risks": ["Implementation complexity"],
                "success_criteria": ["Successful deployment"],
                "resource_requirements": ["Development team"]
            }
    
    def _calculate_collaboration_quality(
        self, 
        contributions: List[PersonaContribution], 
        conflicts: List[Dict[str, Any]], 
        resolutions: List[ConflictResolution]
    ) -> float:
        """Calculate quality score for collaboration"""
        quality_score = 0.5  # Base score
        
        # Contribution quality
        avg_confidence = sum(c.confidence for c in contributions) / len(contributions)
        quality_score += avg_confidence * 0.3
        
        # Conflict resolution effectiveness
        if conflicts:
            resolution_rate = len(resolutions) / len(conflicts)
            avg_resolution_confidence = sum(r.confidence for r in resolutions) / len(resolutions)
            quality_score += (resolution_rate * avg_resolution_confidence) * 0.2
        else:
            quality_score += 0.2  # No conflicts is good
        
        return min(quality_score, 1.0)
    
    def _calculate_overall_confidence(
        self, 
        contributions: List[PersonaContribution], 
        resolutions: List[ConflictResolution]
    ) -> float:
        """Calculate overall confidence in collaboration result"""
        # Average contribution confidence
        contrib_confidence = sum(c.confidence for c in contributions) / len(contributions)
        
        # Resolution confidence impact
        if resolutions:
            resolution_confidence = sum(r.confidence for r in resolutions) / len(resolutions)
            # Conflicts reduce overall confidence
            confidence_penalty = len(resolutions) * 0.1
            overall_confidence = (contrib_confidence + resolution_confidence) / 2 - confidence_penalty
        else:
            overall_confidence = contrib_confidence
        
        return max(0.1, min(overall_confidence, 1.0))
    
    async def _store_collaboration_result(self, result: CollaborationResult):
        """Store collaboration result for learning and analytics"""
        try:
            # Store in collaboration history
            self.collaboration_history[result.collaboration_id] = result
            
            # Update conflict patterns for learning
            for conflict in result.conflicts_identified:
                pattern_key = f"{conflict.conflict_type}_{len(conflict.conflicting_personas)}"
                if pattern_key not in self.conflict_patterns:
                    self.conflict_patterns[pattern_key] = []
                
                self.conflict_patterns[pattern_key].append({
                    "resolution_strategy": conflict.resolution_strategy,
                    "confidence": conflict.confidence,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            logger.info(f"Stored collaboration result {result.collaboration_id}")
        except Exception as e:
            logger.error(f"Error storing collaboration result: {e}")

# Export main class
__all__ = ["CollaborationFramework", "CollaborationContext", "CollaborationType"]

