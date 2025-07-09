"""
Nexus Architect Knowledge Graph Schema
Comprehensive schema definition for organizational knowledge representation
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from neo4j import GraphDatabase
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeType(str, Enum):
    # Core Entities
    ORGANIZATION = "Organization"
    PROJECT = "Project"
    SYSTEM = "System"
    COMPONENT = "Component"
    SERVICE = "Service"
    DATABASE = "Database"
    API = "API"
    
    # People and Roles
    PERSON = "Person"
    TEAM = "Team"
    ROLE = "Role"
    STAKEHOLDER = "Stakeholder"
    
    # Architecture Elements
    ARCHITECTURE_PATTERN = "ArchitecturePattern"
    DESIGN_PATTERN = "DesignPattern"
    TECHNOLOGY = "Technology"
    FRAMEWORK = "Framework"
    LIBRARY = "Library"
    TOOL = "Tool"
    
    # Documentation and Knowledge
    DOCUMENT = "Document"
    REQUIREMENT = "Requirement"
    SPECIFICATION = "Specification"
    DECISION = "Decision"
    BEST_PRACTICE = "BestPractice"
    LESSON_LEARNED = "LessonLearned"
    
    # Security and Compliance
    SECURITY_CONTROL = "SecurityControl"
    VULNERABILITY = "Vulnerability"
    THREAT = "Threat"
    COMPLIANCE_FRAMEWORK = "ComplianceFramework"
    POLICY = "Policy"
    
    # Performance and Monitoring
    METRIC = "Metric"
    ALERT = "Alert"
    INCIDENT = "Incident"
    SLA = "SLA"
    
    # Business Context
    BUSINESS_CAPABILITY = "BusinessCapability"
    BUSINESS_PROCESS = "BusinessProcess"
    BUSINESS_RULE = "BusinessRule"
    OBJECTIVE = "Objective"
    
    # Data and Information
    DATA_SOURCE = "DataSource"
    DATA_FLOW = "DataFlow"
    DATA_MODEL = "DataModel"
    SCHEMA = "Schema"

class RelationshipType(str, Enum):
    # Hierarchical Relationships
    CONTAINS = "CONTAINS"
    PART_OF = "PART_OF"
    BELONGS_TO = "BELONGS_TO"
    OWNS = "OWNS"
    
    # Dependencies
    DEPENDS_ON = "DEPENDS_ON"
    REQUIRES = "REQUIRES"
    USES = "USES"
    IMPLEMENTS = "IMPLEMENTS"
    EXTENDS = "EXTENDS"
    
    # Communication
    COMMUNICATES_WITH = "COMMUNICATES_WITH"
    CALLS = "CALLS"
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    
    # People and Roles
    ASSIGNED_TO = "ASSIGNED_TO"
    RESPONSIBLE_FOR = "RESPONSIBLE_FOR"
    REPORTS_TO = "REPORTS_TO"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    
    # Knowledge Relationships
    DOCUMENTS = "DOCUMENTS"
    REFERENCES = "REFERENCES"
    DERIVED_FROM = "DERIVED_FROM"
    INFLUENCES = "INFLUENCES"
    
    # Security Relationships
    PROTECTS = "PROTECTS"
    THREATENS = "THREATENS"
    MITIGATES = "MITIGATES"
    COMPLIES_WITH = "COMPLIES_WITH"
    
    # Temporal Relationships
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    TRIGGERS = "TRIGGERS"
    CAUSED_BY = "CAUSED_BY"
    
    # Quality Relationships
    MONITORS = "MONITORS"
    MEASURES = "MEASURES"
    IMPACTS = "IMPACTS"
    OPTIMIZES = "OPTIMIZES"

@dataclass
class NodeSchema:
    node_type: NodeType
    required_properties: List[str]
    optional_properties: List[str]
    constraints: List[str]
    indexes: List[str]
    description: str

@dataclass
class RelationshipSchema:
    relationship_type: RelationshipType
    source_types: List[NodeType]
    target_types: List[NodeType]
    required_properties: List[str]
    optional_properties: List[str]
    description: str

class NexusGraphSchema:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.node_schemas = self._define_node_schemas()
        self.relationship_schemas = self._define_relationship_schemas()
        
    def _define_node_schemas(self) -> Dict[NodeType, NodeSchema]:
        """Define comprehensive node schemas for the knowledge graph"""
        return {
            NodeType.ORGANIZATION: NodeSchema(
                node_type=NodeType.ORGANIZATION,
                required_properties=["name", "id", "created_at"],
                optional_properties=["description", "industry", "size", "location", "website"],
                constraints=["UNIQUE (o.id)", "UNIQUE (o.name)"],
                indexes=["INDEX FOR (o:Organization) ON (o.name)", "INDEX FOR (o:Organization) ON (o.industry)"],
                description="Root organization entity containing all other entities"
            ),
            
            NodeType.PROJECT: NodeSchema(
                node_type=NodeType.PROJECT,
                required_properties=["name", "id", "status", "created_at"],
                optional_properties=["description", "start_date", "end_date", "budget", "priority", "phase"],
                constraints=["UNIQUE (p.id)"],
                indexes=["INDEX FOR (p:Project) ON (p.name)", "INDEX FOR (p:Project) ON (p.status)"],
                description="Project entities representing development initiatives"
            ),
            
            NodeType.SYSTEM: NodeSchema(
                node_type=NodeType.SYSTEM,
                required_properties=["name", "id", "type", "status", "created_at"],
                optional_properties=["description", "version", "environment", "criticality", "owner"],
                constraints=["UNIQUE (s.id)"],
                indexes=["INDEX FOR (s:System) ON (s.name)", "INDEX FOR (s:System) ON (s.type)"],
                description="High-level system entities in the architecture"
            ),
            
            NodeType.COMPONENT: NodeSchema(
                node_type=NodeType.COMPONENT,
                required_properties=["name", "id", "type", "created_at"],
                optional_properties=["description", "version", "language", "framework", "repository"],
                constraints=["UNIQUE (c.id)"],
                indexes=["INDEX FOR (c:Component) ON (c.name)", "INDEX FOR (c:Component) ON (c.type)"],
                description="Software components and modules"
            ),
            
            NodeType.SERVICE: NodeSchema(
                node_type=NodeType.SERVICE,
                required_properties=["name", "id", "type", "status", "created_at"],
                optional_properties=["description", "version", "endpoint", "protocol", "sla"],
                constraints=["UNIQUE (s.id)"],
                indexes=["INDEX FOR (s:Service) ON (s.name)", "INDEX FOR (s:Service) ON (s.type)"],
                description="Microservices and API services"
            ),
            
            NodeType.PERSON: NodeSchema(
                node_type=NodeType.PERSON,
                required_properties=["name", "id", "email", "created_at"],
                optional_properties=["title", "department", "skills", "experience", "location"],
                constraints=["UNIQUE (p.id)", "UNIQUE (p.email)"],
                indexes=["INDEX FOR (p:Person) ON (p.name)", "INDEX FOR (p:Person) ON (p.email)"],
                description="People involved in the organization"
            ),
            
            NodeType.TEAM: NodeSchema(
                node_type=NodeType.TEAM,
                required_properties=["name", "id", "type", "created_at"],
                optional_properties=["description", "size", "location", "focus_area"],
                constraints=["UNIQUE (t.id)"],
                indexes=["INDEX FOR (t:Team) ON (t.name)", "INDEX FOR (t:Team) ON (t.type)"],
                description="Teams and organizational units"
            ),
            
            NodeType.TECHNOLOGY: NodeSchema(
                node_type=NodeType.TECHNOLOGY,
                required_properties=["name", "id", "category", "created_at"],
                optional_properties=["description", "version", "vendor", "license", "maturity"],
                constraints=["UNIQUE (t.id)"],
                indexes=["INDEX FOR (t:Technology) ON (t.name)", "INDEX FOR (t:Technology) ON (t.category)"],
                description="Technologies, frameworks, and tools used"
            ),
            
            NodeType.DOCUMENT: NodeSchema(
                node_type=NodeType.DOCUMENT,
                required_properties=["title", "id", "type", "created_at"],
                optional_properties=["description", "content", "author", "version", "url", "tags"],
                constraints=["UNIQUE (d.id)"],
                indexes=["INDEX FOR (d:Document) ON (d.title)", "INDEX FOR (d:Document) ON (d.type)"],
                description="Documentation and knowledge artifacts"
            ),
            
            NodeType.REQUIREMENT: NodeSchema(
                node_type=NodeType.REQUIREMENT,
                required_properties=["title", "id", "type", "priority", "status", "created_at"],
                optional_properties=["description", "acceptance_criteria", "source", "rationale"],
                constraints=["UNIQUE (r.id)"],
                indexes=["INDEX FOR (r:Requirement) ON (r.title)", "INDEX FOR (r:Requirement) ON (r.type)"],
                description="Functional and non-functional requirements"
            ),
            
            NodeType.DECISION: NodeSchema(
                node_type=NodeType.DECISION,
                required_properties=["title", "id", "status", "date", "created_at"],
                optional_properties=["description", "context", "options", "rationale", "consequences"],
                constraints=["UNIQUE (d.id)"],
                indexes=["INDEX FOR (d:Decision) ON (d.title)", "INDEX FOR (d:Decision) ON (d.date)"],
                description="Architectural and design decisions"
            ),
            
            NodeType.SECURITY_CONTROL: NodeSchema(
                node_type=NodeType.SECURITY_CONTROL,
                required_properties=["name", "id", "type", "status", "created_at"],
                optional_properties=["description", "implementation", "effectiveness", "cost"],
                constraints=["UNIQUE (sc.id)"],
                indexes=["INDEX FOR (sc:SecurityControl) ON (sc.name)", "INDEX FOR (sc:SecurityControl) ON (sc.type)"],
                description="Security controls and measures"
            ),
            
            NodeType.VULNERABILITY: NodeSchema(
                node_type=NodeType.VULNERABILITY,
                required_properties=["title", "id", "severity", "status", "discovered_date", "created_at"],
                optional_properties=["description", "cvss_score", "cve_id", "remediation", "impact"],
                constraints=["UNIQUE (v.id)"],
                indexes=["INDEX FOR (v:Vulnerability) ON (v.severity)", "INDEX FOR (v:Vulnerability) ON (v.status)"],
                description="Security vulnerabilities and weaknesses"
            ),
            
            NodeType.METRIC: NodeSchema(
                node_type=NodeType.METRIC,
                required_properties=["name", "id", "type", "unit", "created_at"],
                optional_properties=["description", "target_value", "threshold", "frequency"],
                constraints=["UNIQUE (m.id)"],
                indexes=["INDEX FOR (m:Metric) ON (m.name)", "INDEX FOR (m:Metric) ON (m.type)"],
                description="Performance and business metrics"
            ),
            
            NodeType.INCIDENT: NodeSchema(
                node_type=NodeType.INCIDENT,
                required_properties=["title", "id", "severity", "status", "created_at"],
                optional_properties=["description", "impact", "resolution", "root_cause", "lessons_learned"],
                constraints=["UNIQUE (i.id)"],
                indexes=["INDEX FOR (i:Incident) ON (i.severity)", "INDEX FOR (i:Incident) ON (i.status)"],
                description="Operational incidents and issues"
            ),
            
            NodeType.BUSINESS_CAPABILITY: NodeSchema(
                node_type=NodeType.BUSINESS_CAPABILITY,
                required_properties=["name", "id", "level", "created_at"],
                optional_properties=["description", "maturity", "importance", "owner"],
                constraints=["UNIQUE (bc.id)"],
                indexes=["INDEX FOR (bc:BusinessCapability) ON (bc.name)", "INDEX FOR (bc:BusinessCapability) ON (bc.level)"],
                description="Business capabilities and functions"
            ),
            
            NodeType.DATA_SOURCE: NodeSchema(
                node_type=NodeType.DATA_SOURCE,
                required_properties=["name", "id", "type", "status", "created_at"],
                optional_properties=["description", "location", "format", "size", "update_frequency"],
                constraints=["UNIQUE (ds.id)"],
                indexes=["INDEX FOR (ds:DataSource) ON (ds.name)", "INDEX FOR (ds:DataSource) ON (ds.type)"],
                description="Data sources and repositories"
            )
        }
    
    def _define_relationship_schemas(self) -> Dict[RelationshipType, RelationshipSchema]:
        """Define comprehensive relationship schemas for the knowledge graph"""
        return {
            RelationshipType.CONTAINS: RelationshipSchema(
                relationship_type=RelationshipType.CONTAINS,
                source_types=[NodeType.ORGANIZATION, NodeType.PROJECT, NodeType.SYSTEM, NodeType.TEAM],
                target_types=[NodeType.PROJECT, NodeType.SYSTEM, NodeType.COMPONENT, NodeType.PERSON],
                required_properties=["created_at"],
                optional_properties=["relationship_strength", "notes"],
                description="Hierarchical containment relationship"
            ),
            
            RelationshipType.DEPENDS_ON: RelationshipSchema(
                relationship_type=RelationshipType.DEPENDS_ON,
                source_types=[NodeType.SYSTEM, NodeType.COMPONENT, NodeType.SERVICE],
                target_types=[NodeType.SYSTEM, NodeType.COMPONENT, NodeType.SERVICE, NodeType.DATABASE],
                required_properties=["dependency_type", "created_at"],
                optional_properties=["criticality", "coupling_strength", "notes"],
                description="Dependency relationship between components"
            ),
            
            RelationshipType.COMMUNICATES_WITH: RelationshipSchema(
                relationship_type=RelationshipType.COMMUNICATES_WITH,
                source_types=[NodeType.SERVICE, NodeType.COMPONENT, NodeType.SYSTEM],
                target_types=[NodeType.SERVICE, NodeType.COMPONENT, NodeType.SYSTEM],
                required_properties=["protocol", "created_at"],
                optional_properties=["frequency", "data_volume", "latency", "security_level"],
                description="Communication relationship between services"
            ),
            
            RelationshipType.ASSIGNED_TO: RelationshipSchema(
                relationship_type=RelationshipType.ASSIGNED_TO,
                source_types=[NodeType.PERSON],
                target_types=[NodeType.PROJECT, NodeType.TEAM, NodeType.ROLE],
                required_properties=["assignment_date", "created_at"],
                optional_properties=["allocation_percentage", "end_date", "notes"],
                description="Assignment relationship for people"
            ),
            
            RelationshipType.RESPONSIBLE_FOR: RelationshipSchema(
                relationship_type=RelationshipType.RESPONSIBLE_FOR,
                source_types=[NodeType.PERSON, NodeType.TEAM],
                target_types=[NodeType.SYSTEM, NodeType.COMPONENT, NodeType.SERVICE, NodeType.PROJECT],
                required_properties=["responsibility_type", "created_at"],
                optional_properties=["level", "scope", "notes"],
                description="Responsibility relationship"
            ),
            
            RelationshipType.IMPLEMENTS: RelationshipSchema(
                relationship_type=RelationshipType.IMPLEMENTS,
                source_types=[NodeType.COMPONENT, NodeType.SERVICE, NodeType.SYSTEM],
                target_types=[NodeType.REQUIREMENT, NodeType.SPECIFICATION, NodeType.ARCHITECTURE_PATTERN],
                required_properties=["implementation_status", "created_at"],
                optional_properties=["compliance_level", "notes"],
                description="Implementation relationship"
            ),
            
            RelationshipType.DOCUMENTS: RelationshipSchema(
                relationship_type=RelationshipType.DOCUMENTS,
                source_types=[NodeType.DOCUMENT],
                target_types=[NodeType.SYSTEM, NodeType.COMPONENT, NodeType.PROJECT, NodeType.DECISION],
                required_properties=["document_type", "created_at"],
                optional_properties=["coverage_level", "quality_score", "notes"],
                description="Documentation relationship"
            ),
            
            RelationshipType.THREATENS: RelationshipSchema(
                relationship_type=RelationshipType.THREATENS,
                source_types=[NodeType.THREAT, NodeType.VULNERABILITY],
                target_types=[NodeType.SYSTEM, NodeType.COMPONENT, NodeType.SERVICE, NodeType.DATA_SOURCE],
                required_properties=["threat_level", "created_at"],
                optional_properties=["likelihood", "impact", "notes"],
                description="Security threat relationship"
            ),
            
            RelationshipType.MITIGATES: RelationshipSchema(
                relationship_type=RelationshipType.MITIGATES,
                source_types=[NodeType.SECURITY_CONTROL],
                target_types=[NodeType.THREAT, NodeType.VULNERABILITY],
                required_properties=["mitigation_effectiveness", "created_at"],
                optional_properties=["implementation_cost", "notes"],
                description="Security mitigation relationship"
            ),
            
            RelationshipType.MONITORS: RelationshipSchema(
                relationship_type=RelationshipType.MONITORS,
                source_types=[NodeType.METRIC, NodeType.ALERT],
                target_types=[NodeType.SYSTEM, NodeType.COMPONENT, NodeType.SERVICE],
                required_properties=["monitoring_type", "created_at"],
                optional_properties=["frequency", "threshold", "notes"],
                description="Monitoring relationship"
            ),
            
            RelationshipType.TRIGGERS: RelationshipSchema(
                relationship_type=RelationshipType.TRIGGERS,
                source_types=[NodeType.ALERT, NodeType.INCIDENT],
                target_types=[NodeType.INCIDENT, NodeType.BUSINESS_PROCESS],
                required_properties=["trigger_condition", "created_at"],
                optional_properties=["delay", "probability", "notes"],
                description="Trigger relationship for events"
            ),
            
            RelationshipType.INFLUENCES: RelationshipSchema(
                relationship_type=RelationshipType.INFLUENCES,
                source_types=[NodeType.DECISION, NodeType.REQUIREMENT, NodeType.BUSINESS_RULE],
                target_types=[NodeType.SYSTEM, NodeType.COMPONENT, NodeType.ARCHITECTURE_PATTERN],
                required_properties=["influence_type", "created_at"],
                optional_properties=["influence_strength", "notes"],
                description="Influence relationship for decisions and requirements"
            )
        }
    
    def create_schema(self):
        """Create the complete graph schema in Neo4j"""
        logger.info("Creating Nexus Architect knowledge graph schema...")
        
        with self.driver.session() as session:
            # Create constraints
            self._create_constraints(session)
            
            # Create indexes
            self._create_indexes(session)
            
            # Create node labels
            self._create_node_labels(session)
            
            # Create relationship types
            self._create_relationship_types(session)
            
        logger.info("Schema creation completed successfully")
    
    def _create_constraints(self, session):
        """Create uniqueness and existence constraints"""
        logger.info("Creating constraints...")
        
        constraints = []
        for node_schema in self.node_schemas.values():
            constraints.extend(node_schema.constraints)
        
        for constraint in constraints:
            try:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR {constraint}")
                logger.debug(f"Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Failed to create constraint {constraint}: {e}")
    
    def _create_indexes(self, session):
        """Create performance indexes"""
        logger.info("Creating indexes...")
        
        indexes = []
        for node_schema in self.node_schemas.values():
            indexes.extend(node_schema.indexes)
        
        for index in indexes:
            try:
                session.run(f"CREATE {index} IF NOT EXISTS")
                logger.debug(f"Created index: {index}")
            except Exception as e:
                logger.warning(f"Failed to create index {index}: {e}")
    
    def _create_node_labels(self, session):
        """Create sample nodes for each type to establish labels"""
        logger.info("Creating node labels...")
        
        for node_type in NodeType:
            try:
                query = f"""
                MERGE (n:{node_type.value} {{id: 'schema_placeholder_{node_type.value.lower()}'}})
                SET n.created_at = datetime()
                SET n.schema_placeholder = true
                """
                session.run(query)
                logger.debug(f"Created label: {node_type.value}")
            except Exception as e:
                logger.warning(f"Failed to create label {node_type.value}: {e}")
    
    def _create_relationship_types(self, session):
        """Create sample relationships to establish types"""
        logger.info("Creating relationship types...")
        
        for rel_schema in self.relationship_schemas.values():
            if rel_schema.source_types and rel_schema.target_types:
                source_type = rel_schema.source_types[0].value
                target_type = rel_schema.target_types[0].value
                rel_type = rel_schema.relationship_type.value
                
                try:
                    query = f"""
                    MATCH (s:{source_type} {{schema_placeholder: true}})
                    MATCH (t:{target_type} {{schema_placeholder: true}})
                    MERGE (s)-[r:{rel_type}]->(t)
                    SET r.created_at = datetime()
                    SET r.schema_placeholder = true
                    """
                    session.run(query)
                    logger.debug(f"Created relationship type: {rel_type}")
                except Exception as e:
                    logger.warning(f"Failed to create relationship type {rel_type}: {e}")
    
    def validate_schema(self) -> Dict[str, Any]:
        """Validate the created schema"""
        logger.info("Validating schema...")
        
        validation_results = {
            "constraints": [],
            "indexes": [],
            "node_labels": [],
            "relationship_types": [],
            "validation_passed": True
        }
        
        with self.driver.session() as session:
            # Check constraints
            result = session.run("SHOW CONSTRAINTS")
            validation_results["constraints"] = [record["name"] for record in result]
            
            # Check indexes
            result = session.run("SHOW INDEXES")
            validation_results["indexes"] = [record["name"] for record in result]
            
            # Check node labels
            result = session.run("CALL db.labels()")
            validation_results["node_labels"] = [record["label"] for record in result]
            
            # Check relationship types
            result = session.run("CALL db.relationshipTypes()")
            validation_results["relationship_types"] = [record["relationshipType"] for record in result]
        
        # Validate completeness
        expected_labels = [node_type.value for node_type in NodeType]
        expected_relationships = [rel_type.value for rel_type in RelationshipType]
        
        missing_labels = set(expected_labels) - set(validation_results["node_labels"])
        missing_relationships = set(expected_relationships) - set(validation_results["relationship_types"])
        
        if missing_labels or missing_relationships:
            validation_results["validation_passed"] = False
            validation_results["missing_labels"] = list(missing_labels)
            validation_results["missing_relationships"] = list(missing_relationships)
        
        logger.info(f"Schema validation completed. Passed: {validation_results['validation_passed']}")
        return validation_results
    
    def export_schema_documentation(self, output_path: str):
        """Export schema documentation to YAML file"""
        logger.info(f"Exporting schema documentation to {output_path}")
        
        schema_doc = {
            "nexus_architect_knowledge_graph_schema": {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "description": "Comprehensive knowledge graph schema for Nexus Architect",
                "node_types": {},
                "relationship_types": {}
            }
        }
        
        # Export node schemas
        for node_type, schema in self.node_schemas.items():
            schema_doc["nexus_architect_knowledge_graph_schema"]["node_types"][node_type.value] = {
                "description": schema.description,
                "required_properties": schema.required_properties,
                "optional_properties": schema.optional_properties,
                "constraints": schema.constraints,
                "indexes": schema.indexes
            }
        
        # Export relationship schemas
        for rel_type, schema in self.relationship_schemas.items():
            schema_doc["nexus_architect_knowledge_graph_schema"]["relationship_types"][rel_type.value] = {
                "description": schema.description,
                "source_types": [t.value for t in schema.source_types],
                "target_types": [t.value for t in schema.target_types],
                "required_properties": schema.required_properties,
                "optional_properties": schema.optional_properties
            }
        
        with open(output_path, 'w') as f:
            yaml.dump(schema_doc, f, default_flow_style=False, sort_keys=False)
        
        logger.info("Schema documentation exported successfully")
    
    def cleanup_schema_placeholders(self):
        """Remove schema placeholder nodes and relationships"""
        logger.info("Cleaning up schema placeholders...")
        
        with self.driver.session() as session:
            # Remove placeholder relationships
            session.run("MATCH ()-[r {schema_placeholder: true}]-() DELETE r")
            
            # Remove placeholder nodes
            session.run("MATCH (n {schema_placeholder: true}) DELETE n")
        
        logger.info("Schema placeholders cleaned up")
    
    def close(self):
        """Close the database connection"""
        self.driver.close()

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-lb.nexus-knowledge-graph:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "nexus-architect-graph-password")
    
    # Initialize schema
    schema = NexusGraphSchema(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Create schema
        schema.create_schema()
        
        # Validate schema
        validation_results = schema.validate_schema()
        print(f"Schema validation: {validation_results}")
        
        # Export documentation
        schema.export_schema_documentation("/tmp/nexus_graph_schema.yaml")
        
        # Clean up placeholders
        schema.cleanup_schema_placeholders()
        
    finally:
        schema.close()

