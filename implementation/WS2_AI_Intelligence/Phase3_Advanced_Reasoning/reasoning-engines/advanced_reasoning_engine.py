"""
Nexus Architect Advanced Reasoning Engine
Sophisticated AI reasoning with logical inference, complex problem solving, and strategic analysis
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncio
import aiohttp
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# AI and ML imports
import openai
import anthropic
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Knowledge graph and reasoning
from neo4j import GraphDatabase
import networkx as nx
from pyke import knowledge_engine, krb_traceback

# Logic and reasoning
import sympy
from sympy import symbols, And, Or, Not, Implies, satisfiable
from sympy.logic.boolalg import to_cnf
import z3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    PROBABILISTIC = "probabilistic"
    STRATEGIC = "strategic"

class ConfidenceLevel(Enum):
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

@dataclass
class ReasoningStep:
    step_id: str
    reasoning_type: ReasoningType
    premise: str
    conclusion: str
    confidence: float
    evidence: List[str]
    timestamp: datetime

@dataclass
class LogicalRule:
    rule_id: str
    name: str
    premise: str
    conclusion: str
    confidence: float
    domain: str
    created_at: datetime

@dataclass
class ReasoningChain:
    chain_id: str
    problem: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    reasoning_path: List[str]
    created_at: datetime

@dataclass
class StrategicInsight:
    insight_id: str
    category: str
    description: str
    implications: List[str]
    recommendations: List[str]
    confidence: float
    priority: str
    timeline: str

class AdvancedReasoningEngine:
    def __init__(self, 
                 neo4j_uri: str, 
                 neo4j_user: str, 
                 neo4j_password: str,
                 openai_api_key: str,
                 anthropic_api_key: str):
        
        # Database connections
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # AI model clients
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Load reasoning models
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.logical_reasoning_pipeline = pipeline("text-classification", 
                                                 model="microsoft/DialoGPT-medium")
        
        # Knowledge base and rules
        self.logical_rules: Dict[str, LogicalRule] = {}
        self.reasoning_chains: Dict[str, ReasoningChain] = {}
        self.knowledge_base = {}
        
        # Reasoning context
        self.current_context = {}
        self.reasoning_history = []
        
        # Initialize reasoning components
        self._initialize_logical_rules()
        self._initialize_knowledge_base()
    
    def _initialize_logical_rules(self):
        """Initialize basic logical reasoning rules"""
        
        # Architectural reasoning rules
        arch_rules = [
            LogicalRule(
                rule_id="arch_001",
                name="Dependency Transitivity",
                premise="IF A depends on B AND B depends on C",
                conclusion="THEN A transitively depends on C",
                confidence=0.95,
                domain="architecture",
                created_at=datetime.utcnow()
            ),
            LogicalRule(
                rule_id="arch_002", 
                name="Circular Dependency Detection",
                premise="IF A depends on B AND B depends on A",
                conclusion="THEN there is a circular dependency requiring resolution",
                confidence=0.98,
                domain="architecture",
                created_at=datetime.utcnow()
            ),
            LogicalRule(
                rule_id="arch_003",
                name="Single Point of Failure",
                premise="IF component C has no redundancy AND multiple systems depend on C",
                conclusion="THEN C is a single point of failure requiring mitigation",
                confidence=0.90,
                domain="architecture",
                created_at=datetime.utcnow()
            )
        ]
        
        # Security reasoning rules
        security_rules = [
            LogicalRule(
                rule_id="sec_001",
                name="Privilege Escalation Risk",
                premise="IF user has elevated privileges AND system has vulnerabilities",
                conclusion="THEN there is risk of privilege escalation",
                confidence=0.85,
                domain="security",
                created_at=datetime.utcnow()
            ),
            LogicalRule(
                rule_id="sec_002",
                name="Data Exposure Risk",
                premise="IF data is unencrypted AND network is untrusted",
                conclusion="THEN there is risk of data exposure",
                confidence=0.92,
                domain="security",
                created_at=datetime.utcnow()
            )
        ]
        
        # Performance reasoning rules
        performance_rules = [
            LogicalRule(
                rule_id="perf_001",
                name="Bottleneck Identification",
                premise="IF component utilization > 80% AND response time increasing",
                conclusion="THEN component is likely a performance bottleneck",
                confidence=0.88,
                domain="performance",
                created_at=datetime.utcnow()
            ),
            LogicalRule(
                rule_id="perf_002",
                name="Scaling Requirement",
                premise="IF load is increasing AND current capacity < projected demand",
                conclusion="THEN scaling is required to meet demand",
                confidence=0.85,
                domain="performance",
                created_at=datetime.utcnow()
            )
        ]
        
        # Store all rules
        all_rules = arch_rules + security_rules + performance_rules
        for rule in all_rules:
            self.logical_rules[rule.rule_id] = rule
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base with domain facts"""
        
        self.knowledge_base = {
            "architectural_patterns": {
                "microservices": {
                    "benefits": ["scalability", "independence", "technology_diversity"],
                    "challenges": ["complexity", "network_overhead", "data_consistency"],
                    "best_practices": ["service_boundaries", "api_design", "monitoring"]
                },
                "monolith": {
                    "benefits": ["simplicity", "performance", "easier_testing"],
                    "challenges": ["scalability", "technology_lock_in", "deployment_coupling"],
                    "best_practices": ["modular_design", "clear_interfaces", "gradual_decomposition"]
                }
            },
            "security_principles": {
                "zero_trust": {
                    "principles": ["verify_explicitly", "least_privilege", "assume_breach"],
                    "implementation": ["identity_verification", "access_controls", "monitoring"]
                },
                "defense_in_depth": {
                    "layers": ["network", "host", "application", "data"],
                    "controls": ["firewalls", "encryption", "authentication", "monitoring"]
                }
            },
            "performance_metrics": {
                "response_time": {"threshold": "< 2s", "measurement": "p95"},
                "throughput": {"threshold": "> 1000 rps", "measurement": "sustained"},
                "availability": {"threshold": "> 99.9%", "measurement": "monthly"}
            }
        }
    
    async def reason_about_problem(self, 
                                 problem_statement: str,
                                 context: Dict[str, Any] = None,
                                 reasoning_types: List[ReasoningType] = None) -> ReasoningChain:
        """Perform comprehensive reasoning about a complex problem"""
        
        logger.info(f"Starting reasoning about: {problem_statement}")
        
        if reasoning_types is None:
            reasoning_types = [ReasoningType.DEDUCTIVE, ReasoningType.CAUSAL, ReasoningType.STRATEGIC]
        
        if context is None:
            context = {}
        
        # Generate unique chain ID
        chain_id = str(uuid.uuid4())
        
        # Initialize reasoning chain
        reasoning_steps = []
        
        # Step 1: Problem decomposition
        decomposition_step = await self._decompose_problem(problem_statement, context)
        reasoning_steps.append(decomposition_step)
        
        # Step 2: Knowledge retrieval
        knowledge_step = await self._retrieve_relevant_knowledge(problem_statement, context)
        reasoning_steps.append(knowledge_step)
        
        # Step 3: Apply different reasoning types
        for reasoning_type in reasoning_types:
            reasoning_step = await self._apply_reasoning_type(
                problem_statement, context, reasoning_type, reasoning_steps
            )
            reasoning_steps.append(reasoning_step)
        
        # Step 4: Synthesize conclusions
        synthesis_step = await self._synthesize_conclusions(reasoning_steps, context)
        reasoning_steps.append(synthesis_step)
        
        # Step 5: Validate reasoning chain
        validation_step = await self._validate_reasoning_chain(reasoning_steps, context)
        reasoning_steps.append(validation_step)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_chain_confidence(reasoning_steps)
        
        # Create reasoning chain
        reasoning_chain = ReasoningChain(
            chain_id=chain_id,
            problem=problem_statement,
            steps=reasoning_steps,
            final_conclusion=synthesis_step.conclusion,
            overall_confidence=overall_confidence,
            reasoning_path=[step.step_id for step in reasoning_steps],
            created_at=datetime.utcnow()
        )
        
        # Store reasoning chain
        self.reasoning_chains[chain_id] = reasoning_chain
        
        # Store in Neo4j for persistence
        await self._store_reasoning_chain(reasoning_chain)
        
        logger.info(f"Reasoning completed with confidence: {overall_confidence:.2f}")
        return reasoning_chain
    
    async def _decompose_problem(self, problem: str, context: Dict[str, Any]) -> ReasoningStep:
        """Decompose complex problem into manageable components"""
        
        # Use AI to decompose the problem
        decomposition_prompt = f"""
        Analyze and decompose this complex problem into its key components:
        
        Problem: {problem}
        Context: {json.dumps(context, indent=2)}
        
        Identify:
        1. Core issues and sub-problems
        2. Dependencies and relationships
        3. Stakeholders and impacts
        4. Constraints and requirements
        
        Provide a structured decomposition.
        """
        
        try:
            response = await self._query_ai_model(decomposition_prompt, "gpt-4")
            
            return ReasoningStep(
                step_id=str(uuid.uuid4()),
                reasoning_type=ReasoningType.DEDUCTIVE,
                premise=f"Complex problem: {problem}",
                conclusion=f"Problem decomposition: {response}",
                confidence=0.8,
                evidence=[f"AI analysis of problem structure"],
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error in problem decomposition: {e}")
            return ReasoningStep(
                step_id=str(uuid.uuid4()),
                reasoning_type=ReasoningType.DEDUCTIVE,
                premise=f"Complex problem: {problem}",
                conclusion="Problem decomposition failed - proceeding with original problem",
                confidence=0.3,
                evidence=[f"Error: {str(e)}"],
                timestamp=datetime.utcnow()
            )
    
    async def _retrieve_relevant_knowledge(self, problem: str, context: Dict[str, Any]) -> ReasoningStep:
        """Retrieve relevant knowledge from knowledge base and graph"""
        
        # Query knowledge graph for relevant information
        relevant_entities = await self._query_knowledge_graph(problem, context)
        
        # Search knowledge base for relevant patterns
        relevant_patterns = self._search_knowledge_base(problem)
        
        # Combine knowledge sources
        knowledge_summary = {
            "graph_entities": relevant_entities,
            "knowledge_patterns": relevant_patterns,
            "applicable_rules": self._find_applicable_rules(problem, context)
        }
        
        return ReasoningStep(
            step_id=str(uuid.uuid4()),
            reasoning_type=ReasoningType.INDUCTIVE,
            premise=f"Knowledge retrieval for: {problem}",
            conclusion=f"Relevant knowledge: {json.dumps(knowledge_summary, indent=2)}",
            confidence=0.85,
            evidence=[f"Knowledge graph query", f"Knowledge base search", f"Rule matching"],
            timestamp=datetime.utcnow()
        )
    
    async def _apply_reasoning_type(self, 
                                  problem: str, 
                                  context: Dict[str, Any],
                                  reasoning_type: ReasoningType,
                                  previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Apply specific type of reasoning to the problem"""
        
        if reasoning_type == ReasoningType.DEDUCTIVE:
            return await self._deductive_reasoning(problem, context, previous_steps)
        elif reasoning_type == ReasoningType.INDUCTIVE:
            return await self._inductive_reasoning(problem, context, previous_steps)
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            return await self._abductive_reasoning(problem, context, previous_steps)
        elif reasoning_type == ReasoningType.CAUSAL:
            return await self._causal_reasoning(problem, context, previous_steps)
        elif reasoning_type == ReasoningType.TEMPORAL:
            return await self._temporal_reasoning(problem, context, previous_steps)
        elif reasoning_type == ReasoningType.STRATEGIC:
            return await self._strategic_reasoning(problem, context, previous_steps)
        else:
            return await self._analogical_reasoning(problem, context, previous_steps)
    
    async def _deductive_reasoning(self, 
                                 problem: str, 
                                 context: Dict[str, Any],
                                 previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Apply deductive reasoning using logical rules"""
        
        # Find applicable logical rules
        applicable_rules = []
        for rule in self.logical_rules.values():
            if self._rule_applies_to_problem(rule, problem, context):
                applicable_rules.append(rule)
        
        # Apply rules to derive conclusions
        conclusions = []
        for rule in applicable_rules:
            if self._evaluate_rule_premise(rule, context):
                conclusions.append({
                    "rule": rule.name,
                    "conclusion": rule.conclusion,
                    "confidence": rule.confidence
                })
        
        # Combine conclusions
        if conclusions:
            combined_conclusion = self._combine_deductive_conclusions(conclusions)
            confidence = np.mean([c["confidence"] for c in conclusions])
        else:
            combined_conclusion = "No applicable deductive rules found"
            confidence = 0.2
        
        return ReasoningStep(
            step_id=str(uuid.uuid4()),
            reasoning_type=ReasoningType.DEDUCTIVE,
            premise=f"Logical rules applied to: {problem}",
            conclusion=combined_conclusion,
            confidence=confidence,
            evidence=[f"Applied {len(applicable_rules)} logical rules"],
            timestamp=datetime.utcnow()
        )
    
    async def _causal_reasoning(self, 
                              problem: str, 
                              context: Dict[str, Any],
                              previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Apply causal reasoning to identify cause-effect relationships"""
        
        # Query causal relationships from knowledge graph
        causal_query = """
        MATCH (cause)-[r:CAUSES|INFLUENCES|LEADS_TO]->(effect)
        WHERE cause.name CONTAINS $problem_term OR effect.name CONTAINS $problem_term
        RETURN cause.name as cause, type(r) as relationship, effect.name as effect, r.confidence as confidence
        LIMIT 20
        """
        
        # Extract key terms from problem
        problem_terms = self._extract_key_terms(problem)
        
        causal_relationships = []
        with self.neo4j_driver.session() as session:
            for term in problem_terms:
                result = session.run(causal_query, problem_term=term)
                causal_relationships.extend([
                    {
                        "cause": record["cause"],
                        "relationship": record["relationship"],
                        "effect": record["effect"],
                        "confidence": record["confidence"]
                    }
                    for record in result
                ])
        
        # Analyze causal chains
        causal_analysis = self._analyze_causal_chains(causal_relationships, problem)
        
        return ReasoningStep(
            step_id=str(uuid.uuid4()),
            reasoning_type=ReasoningType.CAUSAL,
            premise=f"Causal analysis for: {problem}",
            conclusion=f"Causal insights: {json.dumps(causal_analysis, indent=2)}",
            confidence=0.75,
            evidence=[f"Found {len(causal_relationships)} causal relationships"],
            timestamp=datetime.utcnow()
        )
    
    async def _strategic_reasoning(self, 
                                 problem: str, 
                                 context: Dict[str, Any],
                                 previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Apply strategic reasoning for high-level decision making"""
        
        strategic_prompt = f"""
        Perform strategic analysis for this problem:
        
        Problem: {problem}
        Context: {json.dumps(context, indent=2)}
        Previous Analysis: {[step.conclusion for step in previous_steps[-3:]]}
        
        Provide strategic insights including:
        1. Strategic implications and impacts
        2. Risk assessment and mitigation strategies
        3. Opportunity identification
        4. Resource requirements and constraints
        5. Timeline and prioritization recommendations
        6. Success metrics and KPIs
        
        Focus on business value, competitive advantage, and long-term sustainability.
        """
        
        try:
            strategic_analysis = await self._query_ai_model(strategic_prompt, "claude-3-opus")
            
            return ReasoningStep(
                step_id=str(uuid.uuid4()),
                reasoning_type=ReasoningType.STRATEGIC,
                premise=f"Strategic analysis for: {problem}",
                conclusion=strategic_analysis,
                confidence=0.8,
                evidence=["AI strategic analysis", "Business context evaluation"],
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error in strategic reasoning: {e}")
            return ReasoningStep(
                step_id=str(uuid.uuid4()),
                reasoning_type=ReasoningType.STRATEGIC,
                premise=f"Strategic analysis for: {problem}",
                conclusion="Strategic analysis failed - manual review required",
                confidence=0.2,
                evidence=[f"Error: {str(e)}"],
                timestamp=datetime.utcnow()
            )
    
    async def _synthesize_conclusions(self, 
                                    reasoning_steps: List[ReasoningStep],
                                    context: Dict[str, Any]) -> ReasoningStep:
        """Synthesize conclusions from multiple reasoning steps"""
        
        # Collect all conclusions
        conclusions = [step.conclusion for step in reasoning_steps]
        confidences = [step.confidence for step in reasoning_steps]
        
        # Use AI to synthesize conclusions
        synthesis_prompt = f"""
        Synthesize these reasoning conclusions into a coherent final analysis:
        
        Individual Conclusions:
        {json.dumps(conclusions, indent=2)}
        
        Context: {json.dumps(context, indent=2)}
        
        Provide:
        1. Integrated analysis combining all perspectives
        2. Key insights and findings
        3. Recommendations and next steps
        4. Confidence assessment and limitations
        5. Areas requiring further investigation
        
        Ensure the synthesis is logical, coherent, and actionable.
        """
        
        try:
            synthesis = await self._query_ai_model(synthesis_prompt, "gpt-4")
            synthesis_confidence = np.mean(confidences) * 0.9  # Slight reduction for synthesis uncertainty
            
            return ReasoningStep(
                step_id=str(uuid.uuid4()),
                reasoning_type=ReasoningType.DEDUCTIVE,
                premise="Synthesis of multiple reasoning perspectives",
                conclusion=synthesis,
                confidence=synthesis_confidence,
                evidence=[f"Synthesized {len(reasoning_steps)} reasoning steps"],
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Error in conclusion synthesis: {e}")
            return ReasoningStep(
                step_id=str(uuid.uuid4()),
                reasoning_type=ReasoningType.DEDUCTIVE,
                premise="Synthesis of multiple reasoning perspectives",
                conclusion="Synthesis failed - review individual conclusions",
                confidence=0.3,
                evidence=[f"Error: {str(e)}"],
                timestamp=datetime.utcnow()
            )
    
    async def _validate_reasoning_chain(self, 
                                      reasoning_steps: List[ReasoningStep],
                                      context: Dict[str, Any]) -> ReasoningStep:
        """Validate the logical consistency of the reasoning chain"""
        
        # Check for logical consistency
        consistency_issues = []
        
        # Check for contradictions
        conclusions = [step.conclusion for step in reasoning_steps]
        contradictions = self._detect_contradictions(conclusions)
        if contradictions:
            consistency_issues.extend(contradictions)
        
        # Check confidence alignment
        confidence_issues = self._check_confidence_alignment(reasoning_steps)
        if confidence_issues:
            consistency_issues.extend(confidence_issues)
        
        # Check evidence quality
        evidence_issues = self._check_evidence_quality(reasoning_steps)
        if evidence_issues:
            consistency_issues.extend(evidence_issues)
        
        # Generate validation conclusion
        if not consistency_issues:
            validation_conclusion = "Reasoning chain is logically consistent and well-supported"
            validation_confidence = 0.9
        else:
            validation_conclusion = f"Reasoning chain has {len(consistency_issues)} issues: {consistency_issues}"
            validation_confidence = 0.5
        
        return ReasoningStep(
            step_id=str(uuid.uuid4()),
            reasoning_type=ReasoningType.DEDUCTIVE,
            premise="Validation of reasoning chain consistency",
            conclusion=validation_conclusion,
            confidence=validation_confidence,
            evidence=[f"Checked {len(reasoning_steps)} reasoning steps"],
            timestamp=datetime.utcnow()
        )
    
    async def generate_strategic_insights(self, 
                                        domain: str,
                                        time_horizon: str = "6 months") -> List[StrategicInsight]:
        """Generate strategic insights for a specific domain"""
        
        logger.info(f"Generating strategic insights for domain: {domain}")
        
        # Query relevant data from knowledge graph
        domain_data = await self._query_domain_data(domain)
        
        # Analyze trends and patterns
        trends = await self._analyze_domain_trends(domain_data, time_horizon)
        
        # Generate insights using AI
        insights_prompt = f"""
        Generate strategic insights for the {domain} domain:
        
        Domain Data: {json.dumps(domain_data, indent=2)}
        Trends Analysis: {json.dumps(trends, indent=2)}
        Time Horizon: {time_horizon}
        
        Generate 5-10 strategic insights covering:
        1. Emerging opportunities and threats
        2. Technology trends and disruptions
        3. Resource optimization opportunities
        4. Risk mitigation strategies
        5. Competitive positioning
        6. Innovation opportunities
        
        For each insight, provide:
        - Category and description
        - Business implications
        - Recommended actions
        - Priority level (High/Medium/Low)
        - Timeline for implementation
        """
        
        try:
            insights_response = await self._query_ai_model(insights_prompt, "claude-3-opus")
            
            # Parse insights from AI response
            insights = self._parse_strategic_insights(insights_response, domain)
            
            # Store insights in knowledge graph
            await self._store_strategic_insights(insights)
            
            logger.info(f"Generated {len(insights)} strategic insights for {domain}")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating strategic insights: {e}")
            return []
    
    async def solve_complex_problem(self, 
                                  problem_description: str,
                                  constraints: List[str] = None,
                                  objectives: List[str] = None) -> Dict[str, Any]:
        """Solve complex multi-faceted problems using advanced reasoning"""
        
        logger.info(f"Solving complex problem: {problem_description}")
        
        if constraints is None:
            constraints = []
        if objectives is None:
            objectives = ["optimize_solution", "minimize_risk", "maximize_value"]
        
        # Step 1: Problem analysis and decomposition
        problem_analysis = await self._analyze_complex_problem(
            problem_description, constraints, objectives
        )
        
        # Step 2: Generate multiple solution approaches
        solution_approaches = await self._generate_solution_approaches(
            problem_analysis, constraints, objectives
        )
        
        # Step 3: Evaluate and rank solutions
        solution_evaluation = await self._evaluate_solutions(
            solution_approaches, constraints, objectives
        )
        
        # Step 4: Optimize best solution
        optimized_solution = await self._optimize_solution(
            solution_evaluation["best_solution"], constraints, objectives
        )
        
        # Step 5: Risk assessment and mitigation
        risk_assessment = await self._assess_solution_risks(
            optimized_solution, constraints
        )
        
        # Step 6: Implementation planning
        implementation_plan = await self._create_implementation_plan(
            optimized_solution, risk_assessment
        )
        
        # Compile comprehensive solution
        solution = {
            "problem": problem_description,
            "analysis": problem_analysis,
            "solution_approaches": solution_approaches,
            "evaluation": solution_evaluation,
            "optimized_solution": optimized_solution,
            "risk_assessment": risk_assessment,
            "implementation_plan": implementation_plan,
            "confidence": solution_evaluation.get("confidence", 0.7),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Store solution in knowledge graph
        await self._store_complex_solution(solution)
        
        logger.info("Complex problem solving completed")
        return solution
    
    # Helper methods
    
    async def _query_ai_model(self, prompt: str, model: str = "gpt-4") -> str:
        """Query AI model with prompt"""
        
        try:
            if model.startswith("gpt"):
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            elif model.startswith("claude"):
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            else:
                raise ValueError(f"Unsupported model: {model}")
                
        except Exception as e:
            logger.error(f"Error querying AI model {model}: {e}")
            raise
    
    async def _query_knowledge_graph(self, problem: str, context: Dict[str, Any]) -> List[Dict]:
        """Query knowledge graph for relevant information"""
        
        # Extract key terms from problem
        key_terms = self._extract_key_terms(problem)
        
        # Query for relevant entities and relationships
        query = """
        MATCH (n)
        WHERE ANY(term IN $terms WHERE n.name CONTAINS term OR n.description CONTAINS term)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n.id as entity_id, n.name as entity_name, labels(n) as entity_types,
               collect({relationship: type(r), related_entity: m.name}) as relationships
        LIMIT 50
        """
        
        results = []
        with self.neo4j_driver.session() as session:
            result = session.run(query, terms=key_terms)
            results = [dict(record) for record in result]
        
        return results
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for knowledge retrieval"""
        
        # Simple keyword extraction (can be enhanced with NLP)
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return list(set(key_terms))[:10]  # Return top 10 unique terms
    
    def _search_knowledge_base(self, problem: str) -> Dict[str, Any]:
        """Search knowledge base for relevant patterns"""
        
        key_terms = self._extract_key_terms(problem)
        relevant_patterns = {}
        
        for category, patterns in self.knowledge_base.items():
            for pattern_name, pattern_data in patterns.items():
                # Simple relevance check
                pattern_text = json.dumps(pattern_data).lower()
                relevance_score = sum(1 for term in key_terms if term in pattern_text)
                
                if relevance_score > 0:
                    relevant_patterns[f"{category}.{pattern_name}"] = {
                        "data": pattern_data,
                        "relevance": relevance_score
                    }
        
        return relevant_patterns
    
    def _find_applicable_rules(self, problem: str, context: Dict[str, Any]) -> List[LogicalRule]:
        """Find logical rules applicable to the problem"""
        
        applicable_rules = []
        problem_lower = problem.lower()
        
        for rule in self.logical_rules.values():
            # Check if rule domain is relevant
            if rule.domain in problem_lower or any(
                term in rule.premise.lower() for term in self._extract_key_terms(problem)
            ):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _rule_applies_to_problem(self, rule: LogicalRule, problem: str, context: Dict[str, Any]) -> bool:
        """Check if a logical rule applies to the given problem"""
        
        # Simple heuristic - can be enhanced with more sophisticated matching
        problem_terms = set(self._extract_key_terms(problem))
        rule_terms = set(self._extract_key_terms(rule.premise + " " + rule.conclusion))
        
        # Check for term overlap
        overlap = len(problem_terms.intersection(rule_terms))
        return overlap > 0
    
    def _evaluate_rule_premise(self, rule: LogicalRule, context: Dict[str, Any]) -> bool:
        """Evaluate if a rule's premise is satisfied given the context"""
        
        # Simplified evaluation - in practice, this would use formal logic
        premise_lower = rule.premise.lower()
        
        # Check for context matches
        for key, value in context.items():
            if key.lower() in premise_lower:
                return True
        
        # Default to true for demonstration
        return True
    
    def _combine_deductive_conclusions(self, conclusions: List[Dict]) -> str:
        """Combine multiple deductive conclusions"""
        
        if not conclusions:
            return "No deductive conclusions available"
        
        combined = "Deductive reasoning conclusions:\n"
        for i, conclusion in enumerate(conclusions, 1):
            combined += f"{i}. {conclusion['conclusion']} (confidence: {conclusion['confidence']:.2f})\n"
        
        return combined
    
    def _analyze_causal_chains(self, relationships: List[Dict], problem: str) -> Dict[str, Any]:
        """Analyze causal chains for insights"""
        
        if not relationships:
            return {"message": "No causal relationships found"}
        
        # Build causal graph
        causal_graph = nx.DiGraph()
        for rel in relationships:
            causal_graph.add_edge(rel["cause"], rel["effect"], 
                                weight=rel.get("confidence", 0.5))
        
        # Find paths and cycles
        analysis = {
            "total_relationships": len(relationships),
            "unique_entities": len(causal_graph.nodes()),
            "causal_chains": [],
            "cycles": list(nx.simple_cycles(causal_graph)),
            "central_entities": []
        }
        
        # Find central entities
        if causal_graph.nodes():
            centrality = nx.degree_centrality(causal_graph)
            analysis["central_entities"] = sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )[:5]
        
        return analysis
    
    def _calculate_chain_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence for reasoning chain"""
        
        if not steps:
            return 0.0
        
        # Weighted average with decay for longer chains
        weights = [0.9 ** i for i in range(len(steps))]
        weighted_confidences = [step.confidence * weight for step, weight in zip(steps, weights)]
        
        return sum(weighted_confidences) / sum(weights)
    
    def _detect_contradictions(self, conclusions: List[str]) -> List[str]:
        """Detect contradictions in conclusions"""
        
        # Simplified contradiction detection
        contradictions = []
        
        for i, conclusion1 in enumerate(conclusions):
            for j, conclusion2 in enumerate(conclusions[i+1:], i+1):
                # Look for obvious contradictions
                if ("not" in conclusion1.lower() and 
                    conclusion1.lower().replace("not", "").strip() in conclusion2.lower()):
                    contradictions.append(f"Contradiction between step {i+1} and {j+1}")
        
        return contradictions
    
    def _check_confidence_alignment(self, steps: List[ReasoningStep]) -> List[str]:
        """Check for confidence alignment issues"""
        
        issues = []
        confidences = [step.confidence for step in steps]
        
        # Check for extreme confidence variations
        if max(confidences) - min(confidences) > 0.7:
            issues.append("Large confidence variation across reasoning steps")
        
        return issues
    
    def _check_evidence_quality(self, steps: List[ReasoningStep]) -> List[str]:
        """Check evidence quality across reasoning steps"""
        
        issues = []
        
        for i, step in enumerate(steps):
            if not step.evidence or len(step.evidence) == 0:
                issues.append(f"Step {i+1} lacks supporting evidence")
            elif len(step.evidence) == 1 and "error" in step.evidence[0].lower():
                issues.append(f"Step {i+1} has error-based evidence")
        
        return issues
    
    async def _store_reasoning_chain(self, chain: ReasoningChain):
        """Store reasoning chain in Neo4j"""
        
        query = """
        CREATE (rc:ReasoningChain {
            chain_id: $chain_id,
            problem: $problem,
            final_conclusion: $final_conclusion,
            overall_confidence: $overall_confidence,
            created_at: datetime($created_at)
        })
        """
        
        with self.neo4j_driver.session() as session:
            session.run(query, 
                       chain_id=chain.chain_id,
                       problem=chain.problem,
                       final_conclusion=chain.final_conclusion,
                       overall_confidence=chain.overall_confidence,
                       created_at=chain.created_at.isoformat())
    
    def export_reasoning_results(self, output_dir: str):
        """Export reasoning results and insights"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export reasoning chains
        chains_data = [asdict(chain) for chain in self.reasoning_chains.values()]
        with open(os.path.join(output_dir, "reasoning_chains.json"), 'w') as f:
            json.dump(chains_data, f, indent=2, default=str)
        
        # Export logical rules
        rules_data = [asdict(rule) for rule in self.logical_rules.values()]
        with open(os.path.join(output_dir, "logical_rules.json"), 'w') as f:
            json.dump(rules_data, f, indent=2, default=str)
        
        # Export knowledge base
        with open(os.path.join(output_dir, "knowledge_base.json"), 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
        
        logger.info(f"Reasoning results exported to {output_dir}")
    
    def close(self):
        """Close database connections"""
        self.neo4j_driver.close()

# Example usage
if __name__ == "__main__":
    reasoning_engine = AdvancedReasoningEngine(
        neo4j_uri="bolt://neo4j-lb.nexus-knowledge-graph:7687",
        neo4j_user="neo4j",
        neo4j_password="nexus-architect-graph-password",
        openai_api_key="your-openai-key",
        anthropic_api_key="your-anthropic-key"
    )
    
    async def main():
        try:
            # Example: Reason about a complex architectural problem
            problem = "Our microservices architecture is experiencing performance issues and high latency"
            context = {
                "current_architecture": "microservices",
                "performance_metrics": {"latency": "high", "throughput": "low"},
                "constraints": ["budget_limited", "timeline_tight"]
            }
            
            reasoning_chain = await reasoning_engine.reason_about_problem(
                problem, context, [ReasoningType.DEDUCTIVE, ReasoningType.CAUSAL, ReasoningType.STRATEGIC]
            )
            
            print(f"Reasoning completed: {reasoning_chain.final_conclusion}")
            print(f"Confidence: {reasoning_chain.overall_confidence:.2f}")
            
            # Generate strategic insights
            insights = await reasoning_engine.generate_strategic_insights("architecture")
            print(f"Generated {len(insights)} strategic insights")
            
            # Solve complex problem
            solution = await reasoning_engine.solve_complex_problem(
                "Optimize system performance while maintaining security and reducing costs",
                constraints=["security_compliance", "budget_constraints"],
                objectives=["performance_optimization", "cost_reduction", "security_maintenance"]
            )
            
            print(f"Complex problem solution confidence: {solution['confidence']:.2f}")
            
            # Export results
            reasoning_engine.export_reasoning_results("/tmp/reasoning_results")
            
        finally:
            reasoning_engine.close()
    
    # Run the example
    asyncio.run(main())

