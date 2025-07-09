#!/usr/bin/env python3
"""
Nexus Architect - WS4 Phase 6: Multi-Agent Coordinator
Advanced autonomous capabilities with multi-agent coordination, task distribution, and collaborative problem solving
"""

import asyncio
import json
import logging
import time
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
import redis
import psycopg2
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml
import heapq
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of autonomous agents"""
    DECISION_ENGINE = "decision_engine"
    QA_AUTOMATION = "qa_automation"
    TRANSFORMATION = "transformation"
    BUG_FIXING = "bug_fixing"
    MONITORING = "monitoring"
    OPERATIONS = "operations"
    SECURITY = "security"
    PERFORMANCE = "performance"

class TaskType(Enum):
    """Types of tasks that can be distributed"""
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    REMEDIATION = "remediation"
    MONITORING = "monitoring"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_TUNE = "performance_tune"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELEGATED = "delegated"

class ConflictType(Enum):
    """Types of conflicts between agents"""
    RESOURCE_CONFLICT = "resource_conflict"
    PRIORITY_CONFLICT = "priority_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    CAPABILITY_CONFLICT = "capability_conflict"
    TIMING_CONFLICT = "timing_conflict"

@dataclass
class Agent:
    """Autonomous agent representation"""
    id: str
    agent_type: AgentType
    name: str
    capabilities: List[str]
    current_load: float = 0.0
    max_capacity: float = 100.0
    status: str = "active"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    performance_score: float = 1.0
    success_rate: float = 1.0
    average_response_time: float = 0.0
    specializations: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return (self.status == "active" and 
                self.current_load < self.max_capacity and
                (datetime.now() - self.last_heartbeat).seconds < 60)
    
    def can_handle_task(self, task_type: TaskType, requirements: Dict[str, Any]) -> bool:
        """Check if agent can handle a specific task"""
        task_capability = task_type.value
        if task_capability not in self.capabilities:
            return False
        
        # Check resource requirements
        for resource, required_amount in requirements.get('resources', {}).items():
            available = self.resource_requirements.get(resource, 0)
            if available < required_amount:
                return False
        
        return True

@dataclass
class Task:
    """Task representation for multi-agent coordination"""
    id: str
    task_type: TaskType
    priority: TaskPriority
    description: str
    requirements: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    
    def is_ready_for_execution(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are completed"""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)
    
    def is_overdue(self) -> bool:
        """Check if task is overdue"""
        return self.deadline and datetime.now() > self.deadline

@dataclass
class Conflict:
    """Conflict representation between agents or tasks"""
    id: str
    conflict_type: ConflictType
    description: str
    involved_agents: List[str]
    involved_tasks: List[str]
    severity: float
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None

class ConsensusEngine:
    """Handles consensus building between agents"""
    
    def __init__(self):
        self.voting_sessions = {}
        self.consensus_threshold = 0.7
        
    def initiate_vote(self, proposal_id: str, proposal: Dict[str, Any], 
                     eligible_agents: List[str], timeout: int = 300) -> str:
        """Initiate a voting session for consensus building"""
        session_id = str(uuid.uuid4())
        
        self.voting_sessions[session_id] = {
            'proposal_id': proposal_id,
            'proposal': proposal,
            'eligible_agents': set(eligible_agents),
            'votes': {},
            'created_at': datetime.now(),
            'timeout': timeout,
            'status': 'active'
        }
        
        logger.info(f"Initiated voting session {session_id} for proposal {proposal_id}")
        return session_id
    
    def cast_vote(self, session_id: str, agent_id: str, vote: bool, 
                  reasoning: str = "") -> bool:
        """Cast a vote in a consensus session"""
        if session_id not in self.voting_sessions:
            return False
        
        session = self.voting_sessions[session_id]
        
        if (session['status'] != 'active' or 
            agent_id not in session['eligible_agents']):
            return False
        
        session['votes'][agent_id] = {
            'vote': vote,
            'reasoning': reasoning,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Agent {agent_id} voted {vote} in session {session_id}")
        return True
    
    def check_consensus(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Check if consensus has been reached"""
        if session_id not in self.voting_sessions:
            return None
        
        session = self.voting_sessions[session_id]
        
        # Check timeout
        if (datetime.now() - session['created_at']).seconds > session['timeout']:
            session['status'] = 'timeout'
            return {'consensus': False, 'reason': 'timeout'}
        
        # Check if all agents have voted
        total_agents = len(session['eligible_agents'])
        votes_cast = len(session['votes'])
        
        if votes_cast < total_agents:
            return None  # Still waiting for votes
        
        # Calculate consensus
        positive_votes = sum(1 for vote_data in session['votes'].values() 
                           if vote_data['vote'])
        consensus_ratio = positive_votes / total_agents
        
        consensus_reached = consensus_ratio >= self.consensus_threshold
        
        session['status'] = 'completed'
        session['consensus_ratio'] = consensus_ratio
        
        return {
            'consensus': consensus_reached,
            'ratio': consensus_ratio,
            'votes': session['votes'],
            'threshold': self.consensus_threshold
        }

class LoadBalancer:
    """Handles task distribution and load balancing across agents"""
    
    def __init__(self):
        self.load_history = defaultdict(list)
        self.performance_weights = {
            'current_load': 0.3,
            'performance_score': 0.25,
            'success_rate': 0.25,
            'response_time': 0.2
        }
    
    def calculate_agent_score(self, agent: Agent, task: Task) -> float:
        """Calculate agent suitability score for a task"""
        if not agent.can_handle_task(task.task_type, task.requirements):
            return 0.0
        
        # Load factor (lower is better)
        load_factor = 1.0 - (agent.current_load / agent.max_capacity)
        
        # Performance factor
        performance_factor = agent.performance_score
        
        # Success rate factor
        success_factor = agent.success_rate
        
        # Response time factor (lower is better)
        response_factor = 1.0 / (1.0 + agent.average_response_time)
        
        # Specialization bonus
        specialization_bonus = 1.0
        if task.task_type.value in agent.specializations:
            specialization_bonus = 1.5
        
        # Calculate weighted score
        score = (
            load_factor * self.performance_weights['current_load'] +
            performance_factor * self.performance_weights['performance_score'] +
            success_factor * self.performance_weights['success_rate'] +
            response_factor * self.performance_weights['response_time']
        ) * specialization_bonus
        
        return score
    
    def select_best_agent(self, agents: List[Agent], task: Task) -> Optional[Agent]:
        """Select the best agent for a task"""
        available_agents = [agent for agent in agents if agent.is_available()]
        
        if not available_agents:
            return None
        
        # Calculate scores for all available agents
        agent_scores = []
        for agent in available_agents:
            score = self.calculate_agent_score(agent, task)
            if score > 0:
                agent_scores.append((score, agent))
        
        if not agent_scores:
            return None
        
        # Sort by score (highest first)
        agent_scores.sort(key=lambda x: x[0], reverse=True)
        
        return agent_scores[0][1]
    
    def distribute_load(self, agents: List[Agent], tasks: List[Task]) -> Dict[str, List[str]]:
        """Distribute tasks across agents optimally"""
        assignment = defaultdict(list)
        
        # Sort tasks by priority and deadline
        sorted_tasks = sorted(tasks, key=lambda t: (
            t.priority.value,
            t.deadline or datetime.max
        ), reverse=True)
        
        for task in sorted_tasks:
            best_agent = self.select_best_agent(agents, task)
            if best_agent:
                assignment[best_agent.id].append(task.id)
                # Update agent load (simplified)
                best_agent.current_load += task.estimated_duration
        
        return dict(assignment)

class ConflictResolver:
    """Handles conflict detection and resolution between agents"""
    
    def __init__(self, consensus_engine: ConsensusEngine):
        self.consensus_engine = consensus_engine
        self.active_conflicts = {}
        
    def detect_conflicts(self, agents: List[Agent], tasks: List[Task]) -> List[Conflict]:
        """Detect conflicts between agents and tasks"""
        conflicts = []
        
        # Resource conflicts
        resource_usage = defaultdict(float)
        resource_assignments = defaultdict(list)
        
        for task in tasks:
            if task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                for resource, amount in task.requirements.get('resources', {}).items():
                    resource_usage[resource] += amount
                    resource_assignments[resource].append(task.id)
        
        # Check for resource over-allocation
        for resource, total_usage in resource_usage.items():
            if total_usage > 100.0:  # Assuming 100% is max capacity
                conflict = Conflict(
                    id=str(uuid.uuid4()),
                    conflict_type=ConflictType.RESOURCE_CONFLICT,
                    description=f"Resource {resource} over-allocated: {total_usage}%",
                    involved_agents=[],
                    involved_tasks=resource_assignments[resource],
                    severity=total_usage / 100.0
                )
                conflicts.append(conflict)
        
        # Priority conflicts
        high_priority_tasks = [t for t in tasks if t.priority.value >= 4]
        if len(high_priority_tasks) > len([a for a in agents if a.is_available()]):
            conflict = Conflict(
                id=str(uuid.uuid4()),
                conflict_type=ConflictType.PRIORITY_CONFLICT,
                description=f"Too many high-priority tasks ({len(high_priority_tasks)}) for available agents",
                involved_agents=[a.id for a in agents if a.is_available()],
                involved_tasks=[t.id for t in high_priority_tasks],
                severity=len(high_priority_tasks) / max(1, len(agents))
            )
            conflicts.append(conflict)
        
        # Dependency conflicts (circular dependencies)
        dependency_graph = {task.id: task.dependencies for task in tasks}
        circular_deps = self._detect_circular_dependencies(dependency_graph)
        
        for cycle in circular_deps:
            conflict = Conflict(
                id=str(uuid.uuid4()),
                conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                description=f"Circular dependency detected: {' -> '.join(cycle)}",
                involved_agents=[],
                involved_tasks=cycle,
                severity=len(cycle) / 10.0  # Severity based on cycle length
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                dfs(neighbor, path + [neighbor])
            
            rec_stack.remove(node)
        
        for node in dependency_graph:
            if node not in visited:
                dfs(node, [node])
        
        return cycles
    
    def resolve_conflict(self, conflict: Conflict, agents: List[Agent], 
                        tasks: List[Task]) -> Dict[str, Any]:
        """Resolve a specific conflict"""
        resolution_strategy = None
        
        if conflict.conflict_type == ConflictType.RESOURCE_CONFLICT:
            resolution_strategy = self._resolve_resource_conflict(conflict, tasks)
        elif conflict.conflict_type == ConflictType.PRIORITY_CONFLICT:
            resolution_strategy = self._resolve_priority_conflict(conflict, agents, tasks)
        elif conflict.conflict_type == ConflictType.DEPENDENCY_CONFLICT:
            resolution_strategy = self._resolve_dependency_conflict(conflict, tasks)
        
        if resolution_strategy:
            # Initiate consensus vote for resolution
            eligible_agents = [a.id for a in agents if a.is_available()]
            session_id = self.consensus_engine.initiate_vote(
                proposal_id=f"resolve_conflict_{conflict.id}",
                proposal=resolution_strategy,
                eligible_agents=eligible_agents
            )
            
            return {
                'strategy': resolution_strategy,
                'consensus_session': session_id,
                'status': 'pending_consensus'
            }
        
        return {'status': 'no_resolution_found'}
    
    def _resolve_resource_conflict(self, conflict: Conflict, tasks: List[Task]) -> Dict[str, Any]:
        """Resolve resource conflicts by task rescheduling or resource reallocation"""
        involved_tasks = [t for t in tasks if t.id in conflict.involved_tasks]
        
        # Sort by priority (lower priority tasks get rescheduled first)
        involved_tasks.sort(key=lambda t: t.priority.value)
        
        return {
            'type': 'resource_reallocation',
            'action': 'reschedule_tasks',
            'tasks_to_reschedule': [t.id for t in involved_tasks[:len(involved_tasks)//2]],
            'reason': 'Resource over-allocation resolution'
        }
    
    def _resolve_priority_conflict(self, conflict: Conflict, agents: List[Agent], 
                                 tasks: List[Task]) -> Dict[str, Any]:
        """Resolve priority conflicts by task queuing or agent scaling"""
        return {
            'type': 'priority_management',
            'action': 'implement_priority_queue',
            'queue_strategy': 'strict_priority_with_aging',
            'reason': 'Too many high-priority tasks for available agents'
        }
    
    def _resolve_dependency_conflict(self, conflict: Conflict, tasks: List[Task]) -> Dict[str, Any]:
        """Resolve dependency conflicts by breaking cycles"""
        return {
            'type': 'dependency_resolution',
            'action': 'break_circular_dependency',
            'tasks_to_modify': conflict.involved_tasks,
            'reason': 'Circular dependency detected'
        }

class CollaborativeProblemSolver:
    """Handles collaborative problem solving between multiple agents"""
    
    def __init__(self, consensus_engine: ConsensusEngine):
        self.consensus_engine = consensus_engine
        self.active_collaborations = {}
        
    def initiate_collaboration(self, problem_id: str, problem_description: str,
                             required_capabilities: List[str], 
                             agents: List[Agent]) -> str:
        """Initiate a collaborative problem-solving session"""
        collaboration_id = str(uuid.uuid4())
        
        # Select agents with required capabilities
        suitable_agents = []
        for agent in agents:
            if any(cap in agent.capabilities for cap in required_capabilities):
                suitable_agents.append(agent)
        
        if len(suitable_agents) < 2:
            logger.warning(f"Not enough suitable agents for collaboration {collaboration_id}")
            return None
        
        self.active_collaborations[collaboration_id] = {
            'problem_id': problem_id,
            'description': problem_description,
            'required_capabilities': required_capabilities,
            'participating_agents': [a.id for a in suitable_agents],
            'solutions': {},
            'consensus_sessions': [],
            'status': 'active',
            'created_at': datetime.now()
        }
        
        logger.info(f"Initiated collaboration {collaboration_id} with {len(suitable_agents)} agents")
        return collaboration_id
    
    def submit_solution(self, collaboration_id: str, agent_id: str, 
                       solution: Dict[str, Any]) -> bool:
        """Submit a solution proposal from an agent"""
        if collaboration_id not in self.active_collaborations:
            return False
        
        collaboration = self.active_collaborations[collaboration_id]
        
        if agent_id not in collaboration['participating_agents']:
            return False
        
        collaboration['solutions'][agent_id] = {
            'solution': solution,
            'submitted_at': datetime.now(),
            'confidence': solution.get('confidence', 0.5)
        }
        
        logger.info(f"Agent {agent_id} submitted solution for collaboration {collaboration_id}")
        
        # Check if we have enough solutions to proceed
        if len(collaboration['solutions']) >= len(collaboration['participating_agents']) // 2:
            self._evaluate_solutions(collaboration_id)
        
        return True
    
    def _evaluate_solutions(self, collaboration_id: str):
        """Evaluate and merge solutions from multiple agents"""
        collaboration = self.active_collaborations[collaboration_id]
        solutions = collaboration['solutions']
        
        # Simple solution merging strategy
        merged_solution = self._merge_solutions(list(solutions.values()))
        
        # Initiate consensus vote on merged solution
        session_id = self.consensus_engine.initiate_vote(
            proposal_id=f"collaboration_solution_{collaboration_id}",
            proposal=merged_solution,
            eligible_agents=collaboration['participating_agents']
        )
        
        collaboration['consensus_sessions'].append(session_id)
        collaboration['merged_solution'] = merged_solution
        
        logger.info(f"Initiated consensus vote for collaboration {collaboration_id}")
    
    def _merge_solutions(self, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple solutions into a single solution"""
        # Weighted average based on confidence scores
        total_confidence = sum(sol['confidence'] for sol in solutions)
        
        if total_confidence == 0:
            return solutions[0]['solution']  # Fallback to first solution
        
        merged = {}
        
        # Merge numerical values using weighted average
        for sol in solutions:
            weight = sol['confidence'] / total_confidence
            solution_data = sol['solution']
            
            for key, value in solution_data.items():
                if isinstance(value, (int, float)):
                    merged[key] = merged.get(key, 0) + value * weight
                elif isinstance(value, str):
                    # For strings, use the one with highest confidence
                    if key not in merged or sol['confidence'] > merged.get(f"{key}_confidence", 0):
                        merged[key] = value
                        merged[f"{key}_confidence"] = sol['confidence']
                elif isinstance(value, list):
                    # Merge lists by union
                    merged[key] = list(set(merged.get(key, []) + value))
        
        return merged

class MultiAgentCoordinator:
    """Main coordinator for multi-agent autonomous operations"""
    
    def __init__(self):
        self.agents = {}
        self.tasks = {}
        self.completed_tasks = set()
        
        # Core components
        self.consensus_engine = ConsensusEngine()
        self.load_balancer = LoadBalancer()
        self.conflict_resolver = ConflictResolver(self.consensus_engine)
        self.problem_solver = CollaborativeProblemSolver(self.consensus_engine)
        
        # Configuration
        self.coordination_interval = 30  # seconds
        self.max_task_retries = 3
        self.task_timeout = 3600  # 1 hour
        
        # State management
        self.running = False
        self.coordination_thread = None
        
        # Database connections
        self.redis_client = None
        self.postgres_conn = None
        self._init_connections()
        
    def _init_connections(self):
        """Initialize database connections"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            
        try:
            self.postgres_conn = psycopg2.connect(
                host="localhost",
                database="nexus_architect",
                user="nexus_user",
                password="nexus_password"
            )
            logger.info("PostgreSQL connection established")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
    
    def register_agent(self, agent: Agent) -> bool:
        """Register a new agent with the coordinator"""
        if agent.id in self.agents:
            logger.warning(f"Agent {agent.id} already registered")
            return False
        
        self.agents[agent.id] = agent
        logger.info(f"Registered agent {agent.id} of type {agent.agent_type.value}")
        
        # Store in Redis for persistence
        if self.redis_client:
            try:
                self.redis_client.hset(
                    "agents", 
                    agent.id, 
                    json.dumps(asdict(agent), default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to store agent in Redis: {e}")
        
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the coordinator"""
        if agent_id not in self.agents:
            return False
        
        # Reassign tasks from this agent
        agent_tasks = [task for task in self.tasks.values() 
                      if task.assigned_agent == agent_id and 
                      task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]]
        
        for task in agent_tasks:
            task.assigned_agent = None
            task.status = TaskStatus.PENDING
            logger.info(f"Reassigned task {task.id} due to agent {agent_id} unregistration")
        
        del self.agents[agent_id]
        
        # Remove from Redis
        if self.redis_client:
            try:
                self.redis_client.hdel("agents", agent_id)
            except Exception as e:
                logger.warning(f"Failed to remove agent from Redis: {e}")
        
        logger.info(f"Unregistered agent {agent_id}")
        return True
    
    def submit_task(self, task: Task) -> bool:
        """Submit a new task for execution"""
        if task.id in self.tasks:
            logger.warning(f"Task {task.id} already exists")
            return False
        
        self.tasks[task.id] = task
        logger.info(f"Submitted task {task.id} of type {task.task_type.value}")
        
        # Store in database
        self._store_task(task)
        
        return True
    
    def _store_task(self, task: Task):
        """Store task in database"""
        if not self.postgres_conn:
            return
        
        try:
            cursor = self.postgres_conn.cursor()
            cursor.execute("""
                INSERT INTO multi_agent_tasks 
                (id, task_type, priority, description, requirements, dependencies, 
                 status, created_at, deadline, estimated_duration)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                assigned_agent = EXCLUDED.assigned_agent,
                result = EXCLUDED.result
            """, (
                task.id, task.task_type.value, task.priority.value,
                task.description, json.dumps(task.requirements),
                json.dumps(task.dependencies), task.status.value,
                task.created_at, task.deadline, task.estimated_duration
            ))
            self.postgres_conn.commit()
            cursor.close()
        except Exception as e:
            logger.warning(f"Failed to store task in database: {e}")
    
    def start_coordination(self):
        """Start the coordination loop"""
        if self.running:
            logger.warning("Coordination already running")
            return
        
        self.running = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
        logger.info("Multi-agent coordination started")
    
    def stop_coordination(self):
        """Stop the coordination loop"""
        self.running = False
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        logger.info("Multi-agent coordination stopped")
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while self.running:
            try:
                # Update agent heartbeats
                self._update_agent_status()
                
                # Detect and resolve conflicts
                conflicts = self.conflict_resolver.detect_conflicts(
                    list(self.agents.values()), 
                    list(self.tasks.values())
                )
                
                for conflict in conflicts:
                    if conflict.id not in self.conflict_resolver.active_conflicts:
                        self.conflict_resolver.active_conflicts[conflict.id] = conflict
                        resolution = self.conflict_resolver.resolve_conflict(
                            conflict, 
                            list(self.agents.values()), 
                            list(self.tasks.values())
                        )
                        logger.info(f"Initiated resolution for conflict {conflict.id}")
                
                # Assign pending tasks
                self._assign_pending_tasks()
                
                # Check task progress and timeouts
                self._monitor_task_progress()
                
                # Process consensus results
                self._process_consensus_results()
                
                logger.debug(f"Coordination cycle completed. Active agents: {len(self.agents)}, "
                           f"Active tasks: {len([t for t in self.tasks.values() if t.status != TaskStatus.COMPLETED])}")
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
            
            time.sleep(self.coordination_interval)
    
    def _update_agent_status(self):
        """Update agent status and remove inactive agents"""
        current_time = datetime.now()
        inactive_agents = []
        
        for agent_id, agent in self.agents.items():
            if (current_time - agent.last_heartbeat).seconds > 120:  # 2 minutes timeout
                agent.status = "inactive"
                inactive_agents.append(agent_id)
        
        # Remove inactive agents
        for agent_id in inactive_agents:
            self.unregister_agent(agent_id)
    
    def _assign_pending_tasks(self):
        """Assign pending tasks to available agents"""
        pending_tasks = [task for task in self.tasks.values() 
                        if task.status == TaskStatus.PENDING and 
                        task.is_ready_for_execution(self.completed_tasks)]
        
        if not pending_tasks:
            return
        
        # Get assignment recommendations from load balancer
        assignment = self.load_balancer.distribute_load(
            list(self.agents.values()), 
            pending_tasks
        )
        
        for agent_id, task_ids in assignment.items():
            agent = self.agents.get(agent_id)
            if not agent or not agent.is_available():
                continue
            
            for task_id in task_ids:
                task = self.tasks.get(task_id)
                if task and task.status == TaskStatus.PENDING:
                    task.assigned_agent = agent_id
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_at = datetime.now()
                    
                    # Update agent load
                    agent.current_load += task.estimated_duration
                    
                    logger.info(f"Assigned task {task_id} to agent {agent_id}")
                    
                    # Store assignment
                    self._store_task(task)
    
    def _monitor_task_progress(self):
        """Monitor task progress and handle timeouts"""
        current_time = datetime.now()
        
        for task in self.tasks.values():
            # Check for overdue tasks
            if task.is_overdue():
                logger.warning(f"Task {task.id} is overdue")
                # Could trigger escalation or reassignment
            
            # Check for stuck tasks
            if (task.status == TaskStatus.IN_PROGRESS and 
                task.started_at and 
                (current_time - task.started_at).seconds > self.task_timeout):
                
                logger.warning(f"Task {task.id} timed out, marking as failed")
                task.status = TaskStatus.FAILED
                task.error_message = "Task timeout"
                
                # Free up agent capacity
                if task.assigned_agent and task.assigned_agent in self.agents:
                    agent = self.agents[task.assigned_agent]
                    agent.current_load = max(0, agent.current_load - task.estimated_duration)
                
                # Retry if possible
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.assigned_agent = None
                    logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
    
    def _process_consensus_results(self):
        """Process results from consensus voting sessions"""
        for session_id, session in list(self.consensus_engine.voting_sessions.items()):
            if session['status'] == 'active':
                result = self.consensus_engine.check_consensus(session_id)
                if result:
                    logger.info(f"Consensus session {session_id} completed: {result}")
                    # Process the consensus result based on proposal type
                    self._handle_consensus_result(session, result)
    
    def _handle_consensus_result(self, session: Dict[str, Any], result: Dict[str, Any]):
        """Handle the result of a consensus vote"""
        proposal_id = session['proposal_id']
        
        if result['consensus']:
            logger.info(f"Consensus reached for proposal {proposal_id}")
            # Implement the agreed-upon action
            if 'resolve_conflict' in proposal_id:
                # Apply conflict resolution
                pass
            elif 'collaboration_solution' in proposal_id:
                # Apply collaborative solution
                pass
        else:
            logger.info(f"Consensus not reached for proposal {proposal_id}")
            # Handle lack of consensus (escalate, retry, etc.)
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        active_agents = [a for a in self.agents.values() if a.status == "active"]
        pending_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
        in_progress_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        
        return {
            'coordination_running': self.running,
            'total_agents': len(self.agents),
            'active_agents': len(active_agents),
            'total_tasks': len(self.tasks),
            'pending_tasks': len(pending_tasks),
            'in_progress_tasks': len(in_progress_tasks),
            'completed_tasks': len(completed_tasks),
            'active_conflicts': len(self.conflict_resolver.active_conflicts),
            'active_collaborations': len(self.problem_solver.active_collaborations),
            'last_update': datetime.now().isoformat()
        }

# Flask API
app = Flask(__name__)
CORS(app)

# Global coordinator instance
coordinator = MultiAgentCoordinator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'multi_agent_coordinator',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/coordination/status', methods=['GET'])
def get_coordination_status():
    """Get coordination status"""
    return jsonify(coordinator.get_coordination_status())

@app.route('/coordination/start', methods=['POST'])
def start_coordination():
    """Start coordination"""
    coordinator.start_coordination()
    return jsonify({
        'message': 'Multi-agent coordination started',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/coordination/stop', methods=['POST'])
def stop_coordination():
    """Stop coordination"""
    coordinator.stop_coordination()
    return jsonify({
        'message': 'Multi-agent coordination stopped',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/agents', methods=['GET'])
def get_agents():
    """Get all registered agents"""
    agents_data = {}
    for agent_id, agent in coordinator.agents.items():
        agents_data[agent_id] = asdict(agent)
    
    return jsonify({
        'agents': agents_data,
        'count': len(agents_data),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/agents', methods=['POST'])
def register_agent():
    """Register a new agent"""
    data = request.get_json()
    
    try:
        agent = Agent(
            id=data['id'],
            agent_type=AgentType(data['agent_type']),
            name=data['name'],
            capabilities=data['capabilities'],
            max_capacity=data.get('max_capacity', 100.0),
            specializations=data.get('specializations', []),
            resource_requirements=data.get('resource_requirements', {})
        )
        
        success = coordinator.register_agent(agent)
        
        if success:
            return jsonify({
                'message': f'Agent {agent.id} registered successfully',
                'agent_id': agent.id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Agent registration failed'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Invalid agent data: {str(e)}'}), 400

@app.route('/agents/<agent_id>', methods=['DELETE'])
def unregister_agent(agent_id):
    """Unregister an agent"""
    success = coordinator.unregister_agent(agent_id)
    
    if success:
        return jsonify({
            'message': f'Agent {agent_id} unregistered successfully',
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({'error': 'Agent not found'}), 404

@app.route('/tasks', methods=['GET'])
def get_tasks():
    """Get all tasks"""
    tasks_data = {}
    for task_id, task in coordinator.tasks.items():
        tasks_data[task_id] = asdict(task)
    
    return jsonify({
        'tasks': tasks_data,
        'count': len(tasks_data),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/tasks', methods=['POST'])
def submit_task():
    """Submit a new task"""
    data = request.get_json()
    
    try:
        task = Task(
            id=data.get('id', str(uuid.uuid4())),
            task_type=TaskType(data['task_type']),
            priority=TaskPriority(data['priority']),
            description=data['description'],
            requirements=data.get('requirements', {}),
            dependencies=data.get('dependencies', []),
            deadline=datetime.fromisoformat(data['deadline']) if data.get('deadline') else None,
            estimated_duration=data.get('estimated_duration', 0.0)
        )
        
        success = coordinator.submit_task(task)
        
        if success:
            return jsonify({
                'message': f'Task {task.id} submitted successfully',
                'task_id': task.id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Task submission failed'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Invalid task data: {str(e)}'}), 400

@app.route('/tasks/<task_id>/status', methods=['PUT'])
def update_task_status(task_id):
    """Update task status"""
    data = request.get_json()
    
    if task_id not in coordinator.tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = coordinator.tasks[task_id]
    
    try:
        new_status = TaskStatus(data['status'])
        task.status = new_status
        
        if new_status == TaskStatus.IN_PROGRESS:
            task.started_at = datetime.now()
        elif new_status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now()
            coordinator.completed_tasks.add(task_id)
            
            # Update agent performance metrics
            if task.assigned_agent and task.assigned_agent in coordinator.agents:
                agent = coordinator.agents[task.assigned_agent]
                if task.started_at:
                    task.actual_duration = (task.completed_at - task.started_at).total_seconds()
                    # Update agent metrics (simplified)
                    agent.current_load = max(0, agent.current_load - task.estimated_duration)
        
        # Store updated task
        coordinator._store_task(task)
        
        return jsonify({
            'message': f'Task {task_id} status updated to {new_status.value}',
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid status: {str(e)}'}), 400

@app.route('/conflicts', methods=['GET'])
def get_conflicts():
    """Get active conflicts"""
    conflicts_data = {}
    for conflict_id, conflict in coordinator.conflict_resolver.active_conflicts.items():
        conflicts_data[conflict_id] = asdict(conflict)
    
    return jsonify({
        'conflicts': conflicts_data,
        'count': len(conflicts_data),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/collaborations', methods=['POST'])
def initiate_collaboration():
    """Initiate a collaborative problem-solving session"""
    data = request.get_json()
    
    collaboration_id = coordinator.problem_solver.initiate_collaboration(
        problem_id=data['problem_id'],
        problem_description=data['description'],
        required_capabilities=data['required_capabilities'],
        agents=list(coordinator.agents.values())
    )
    
    if collaboration_id:
        return jsonify({
            'message': 'Collaboration initiated successfully',
            'collaboration_id': collaboration_id,
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({'error': 'Failed to initiate collaboration'}), 400

@app.route('/collaborations/<collaboration_id>/solutions', methods=['POST'])
def submit_solution(collaboration_id):
    """Submit a solution to a collaboration"""
    data = request.get_json()
    
    success = coordinator.problem_solver.submit_solution(
        collaboration_id=collaboration_id,
        agent_id=data['agent_id'],
        solution=data['solution']
    )
    
    if success:
        return jsonify({
            'message': 'Solution submitted successfully',
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({'error': 'Failed to submit solution'}), 400

if __name__ == '__main__':
    # Start coordination automatically
    coordinator.start_coordination()
    
    try:
        app.run(host='0.0.0.0', port=8070, debug=False)
    finally:
        coordinator.stop_coordination()

