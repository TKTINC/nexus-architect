# WS1: Core Foundation & Security - Execution Prompts

## Overview
This document contains execution-ready prompts for each phase of WS1: Core Foundation & Security. Each prompt can be executed directly when the development team is ready to start that specific phase.

## How to Use These Prompts
1. **Copy the entire prompt** for the phase you want to execute
2. **Paste it into your AI assistant** (Claude, ChatGPT, etc.)
3. **Follow the systematic execution** as the AI guides you through each step
4. **Use the validation checklist** to ensure completion before moving to the next phase

---

## Phase 1: Infrastructure Foundation and Basic Security
**Duration:** 4 weeks | **Team:** 3 backend engineers, 2 DevOps engineers, 1 security engineer

### 🚀 EXECUTION PROMPT - PHASE 1

```
You are a senior DevOps architect tasked with implementing Phase 1 of the Nexus Architect Core Foundation. Your goal is to deploy production-ready Kubernetes infrastructure with basic security controls.

CONTEXT:
- Building foundation for an AI-powered architectural teammate platform
- Need enterprise-grade infrastructure that can scale to 1000+ concurrent users
- Security-first approach with zero-trust principles
- Must support future AI workloads and multi-tenant architecture

TECHNICAL REQUIREMENTS:
Infrastructure Components:
- Kubernetes 1.28+ cluster (3 control plane, 3 worker nodes)
- PostgreSQL 15+ cluster with high availability
- Redis 7.0+ cluster for caching and sessions
- MinIO cluster for object storage
- HashiCorp Vault for secrets management
- Rook-Ceph for persistent storage
- Calico CNI with network policies
- Prometheus + Grafana for monitoring
- cert-manager for SSL/TLS certificates

EXECUTION STEPS:
1. **Week 1: Kubernetes Cluster Deployment**
   - Set up Kubernetes cluster with kubeadm or managed service
   - Configure Calico CNI with network policies
   - Deploy Rook-Ceph for persistent storage
   - Validate cluster networking and storage

2. **Week 2: Database Cluster Setup**
   - Deploy PostgreSQL cluster with Patroni for HA
   - Configure automated backups and point-in-time recovery
   - Deploy Redis cluster with persistence and replication
   - Create initial database schemas for users and basic entities

3. **Week 3: Storage and Security Baseline**
   - Deploy MinIO cluster with encryption and access controls
   - Set up HashiCorp Vault for secrets management
   - Configure cert-manager for automatic SSL certificate management
   - Implement basic network policies and firewall rules

4. **Week 4: Monitoring and Validation**
   - Deploy Prometheus and Grafana monitoring stack
   - Configure alerting rules for infrastructure components
   - Perform load testing and failover scenarios
   - Document infrastructure and create operational procedures

DELIVERABLES CHECKLIST:
□ Kubernetes cluster with 99.9% availability target
□ PostgreSQL cluster with automated backup (RPO: 1 hour)
□ Redis cluster with persistence and high availability
□ MinIO object storage with encryption enabled
□ Vault secrets management with auto-unseal
□ Monitoring stack capturing all infrastructure metrics
□ Network security policies blocking unauthorized traffic
□ SSL certificates automatically managed and renewed
□ Load testing results showing 1000+ concurrent connection support
□ Complete infrastructure documentation and runbooks

VALIDATION CRITERIA:
- Kubernetes cluster passes conformance tests
- Database cluster achieves <100ms query response times
- All security scans pass with zero critical vulnerabilities
- Backup and recovery procedures tested and documented
- Monitoring captures all metrics with <1 minute latency

NEXT STEPS:
Upon completion, proceed to Phase 2: Authentication, Authorization & API Foundation

Please execute this phase systematically, providing detailed implementation steps, configuration files, and validation procedures for each component.
```

---

## Phase 2: Authentication, Authorization & API Foundation
**Duration:** 4 weeks | **Team:** 3 backend engineers, 1 security engineer, 1 AI/ML engineer

### 🚀 EXECUTION PROMPT - PHASE 2

```
You are a senior backend architect implementing Phase 2 of the Nexus Architect Core Foundation. Your goal is to create a comprehensive authentication, authorization, and API foundation with basic AI integration.

CONTEXT:
- Building on the infrastructure from Phase 1
- Need enterprise-grade auth system supporting SSO and MFA
- API-first architecture for all system interactions
- Foundation for AI-powered conversational interfaces
- Must support role-based access control for different user types

TECHNICAL REQUIREMENTS:
Authentication System:
- Keycloak identity provider with OAuth 2.0/OpenID Connect
- Multi-factor authentication (TOTP, SMS, email)
- JWT token management with refresh tokens
- Enterprise SSO integration readiness (SAML, LDAP)

Authorization Framework:
- Role-based access control (RBAC) with granular permissions
- Attribute-based access control (ABAC) foundation
- API-level authorization enforcement
- Fine-grained permissions for different resources

API Foundation:
- FastAPI framework with async endpoints
- OpenAPI/Swagger documentation
- GraphQL layer for flexible data querying
- Request validation, error handling, rate limiting
- Real-time subscriptions for live updates

Basic AI Integration:
- OpenAI GPT-4 and Anthropic Claude integration
- Basic conversational AI framework
- AI model serving infrastructure preparation
- Content filtering and safety checks

EXECUTION STEPS:
1. **Week 1: Authentication System**
   - Deploy Keycloak with PostgreSQL backend
   - Configure OAuth 2.0/OpenID Connect flows
   - Implement JWT token management with refresh
   - Set up MFA with TOTP and backup methods
   - Create user registration and management APIs

2. **Week 2: Authorization Framework**
   - Design RBAC system with roles and permissions
   - Implement permission-based access control
   - Create authorization middleware for APIs
   - Set up fine-grained resource permissions
   - Build admin interface for role management

3. **Week 3: API Foundation**
   - Set up FastAPI with async endpoints
   - Implement OpenAPI documentation generation
   - Create GraphQL schema and resolvers
   - Add request validation and error handling
   - Implement rate limiting and throttling

4. **Week 4: Basic AI Integration**
   - Integrate OpenAI GPT-4 API
   - Add Anthropic Claude API integration
   - Create conversational AI framework
   - Implement content filtering and safety checks
   - Build basic chat interface for testing

DELIVERABLES CHECKLIST:
□ Keycloak identity provider with OAuth 2.0/OIDC
□ Multi-factor authentication with multiple methods
□ RBAC system with granular permissions
□ Authorization middleware for all API endpoints
□ FastAPI foundation with comprehensive documentation
□ GraphQL layer with real-time subscriptions
□ Rate limiting and security controls
□ Basic AI conversational interface
□ User management system with role assignment
□ API security testing and validation

VALIDATION CRITERIA:
- Authentication system supports 1000+ concurrent users
- Authorization decisions complete in <50ms
- API endpoints achieve <200ms response times
- Security testing passes with zero auth bypasses
- Basic AI conversations achieve >90% user satisfaction

INTEGRATION POINTS:
- User authentication for all future workstream interfaces
- API foundation for all system interactions
- AI services integration points for other workstreams
- Authorization framework for secure data access

Please execute this phase systematically, providing detailed implementation steps, code examples, and security validation procedures.
```

---

## Phase 3: Advanced Security & Compliance Framework
**Duration:** 4 weeks | **Team:** 2 security engineers, 2 backend engineers, 1 DevOps engineer

### 🚀 EXECUTION PROMPT - PHASE 3

```
You are a senior security architect implementing Phase 3 of the Nexus Architect Core Foundation. Your goal is to establish advanced security controls and compliance frameworks for enterprise deployment.

CONTEXT:
- Building on authentication and API foundation from Phase 2
- Need zero-trust security architecture for enterprise environments
- Must support GDPR, SOC 2, and HIPAA compliance requirements
- Handling sensitive organizational data and AI processing
- Real-time threat detection and incident response required

TECHNICAL REQUIREMENTS:
Zero-Trust Architecture:
- Istio service mesh with mTLS encryption
- Network micro-segmentation with Calico policies
- Identity-based access controls for all services
- Continuous security validation and monitoring

Data Protection:
- End-to-end encryption for data in transit and at rest
- HashiCorp Vault for encryption key management
- Data classification and labeling system
- Privacy-preserving data processing techniques

Compliance Framework:
- GDPR compliance controls and data subject rights
- SOC 2 Type II preparation and controls
- HIPAA compliance capabilities for healthcare data
- Audit trail and immutable logging system

Security Monitoring:
- Real-time threat detection with SIEM integration
- Anomaly detection and behavioral analysis
- Security incident response automation
- Vulnerability scanning and management

EXECUTION STEPS:
1. **Week 1: Zero-Trust Architecture**
   - Deploy Istio service mesh with mTLS
   - Configure network micro-segmentation policies
   - Implement identity-based service authentication
   - Set up continuous security validation

2. **Week 2: Data Encryption and Key Management**
   - Configure end-to-end encryption for all data flows
   - Set up Vault for encryption key management
   - Implement data classification and labeling
   - Deploy privacy-preserving processing techniques

3. **Week 3: Compliance Framework**
   - Implement GDPR compliance controls
   - Set up SOC 2 control framework
   - Configure HIPAA compliance capabilities
   - Deploy immutable audit logging system

4. **Week 4: Security Monitoring and Response**
   - Set up real-time threat detection
   - Configure anomaly detection and alerting
   - Implement security incident response automation
   - Deploy vulnerability scanning and management

DELIVERABLES CHECKLIST:
□ Istio service mesh with mTLS encryption
□ Network micro-segmentation with security policies
□ End-to-end data encryption and key management
□ Data classification and privacy controls
□ GDPR compliance framework and controls
□ SOC 2 Type II control implementation
□ HIPAA compliance capabilities
□ Immutable audit logging system
□ Real-time threat detection and monitoring
□ Security incident response automation
□ Vulnerability management system
□ Security documentation and procedures

VALIDATION CRITERIA:
- Penetration testing finds zero critical vulnerabilities
- Compliance audit simulation passes with 100% score
- Security incident response completes in <15 minutes
- Encryption performance impact <5% on operations
- Threat detection achieves <1% false positive rate

COMPLIANCE REQUIREMENTS:
GDPR:
- Data subject rights (access, rectification, erasure)
- Privacy by design and default
- Data protection impact assessments
- Breach notification procedures

SOC 2:
- Security controls and monitoring
- Availability and performance monitoring
- Processing integrity validation
- Confidentiality and privacy protection

HIPAA:
- Administrative, physical, and technical safeguards
- Access controls and audit logs
- Data encryption and transmission security
- Business associate agreements

Please execute this phase systematically, providing detailed security configurations, compliance procedures, and validation testing.
```

---

## Phase 4: Enhanced AI Services & Knowledge Foundation
**Duration:** 4 weeks | **Team:** 2 AI/ML engineers, 2 backend engineers, 1 DevOps engineer

### 🚀 EXECUTION PROMPT - PHASE 4

```
You are a senior AI/ML architect implementing Phase 4 of the Nexus Architect Core Foundation. Your goal is to enhance AI capabilities with multi-model support and establish the knowledge processing foundation.

CONTEXT:
- Building on secure infrastructure and API foundation from previous phases
- Need sophisticated AI capabilities for architectural expertise
- Multi-persona AI architecture (Security, Performance, Application, Ops experts)
- Knowledge processing for organizational data and documentation
- Foundation for autonomous capabilities and decision-making

TECHNICAL REQUIREMENTS:
AI Model Infrastructure:
- TorchServe and TensorFlow Serving for model deployment
- Model versioning and A/B testing capabilities
- Auto-scaling based on inference demand
- GPU resource management and optimization

Knowledge Processing:
- Vector database integration (Pinecone or Weaviate)
- Text embedding generation and storage
- Neo4j preparation for knowledge graph
- Document processing and indexing pipeline

Multi-Model Support:
- OpenAI GPT-4 for general conversations
- Anthropic Claude for safety-focused interactions
- Code-specific models (CodeT5, StarCoder)
- Model routing and orchestration framework

AI Safety Framework:
- Content filtering and safety checks
- Response validation and quality control
- Usage monitoring and rate limiting
- Bias detection and mitigation

EXECUTION STEPS:
1. **Week 1: AI Model Serving Infrastructure**
   - Deploy TorchServe and TensorFlow Serving
   - Set up model versioning and registry
   - Configure auto-scaling for inference workloads
   - Implement GPU resource management

2. **Week 2: Vector Database and Knowledge Processing**
   - Deploy vector database (Pinecone/Weaviate)
   - Set up text embedding generation pipeline
   - Create document processing and indexing system
   - Prepare Neo4j schema for knowledge graph

3. **Week 3: Multi-Model Framework**
   - Integrate multiple AI models (GPT-4, Claude, Code models)
   - Implement intelligent model routing
   - Create model orchestration framework
   - Set up A/B testing for model performance

4. **Week 4: AI Safety and Quality Controls**
   - Implement content filtering and safety checks
   - Set up response validation and quality control
   - Deploy usage monitoring and rate limiting
   - Create bias detection and mitigation system

DELIVERABLES CHECKLIST:
□ Production-ready AI model serving infrastructure
□ Vector database with semantic search capabilities
□ Multi-model AI framework with intelligent routing
□ Document processing pipeline for knowledge indexing
□ AI safety framework with content filtering
□ Model performance monitoring and optimization
□ Knowledge graph schema and preparation
□ AI service APIs with comprehensive documentation
□ Model versioning and A/B testing system
□ GPU resource management and auto-scaling

VALIDATION CRITERIA:
- AI model inference completes in <2 seconds for 95% of queries
- Vector search achieves >90% relevance for semantic queries
- Multi-model routing selects optimal model with >95% accuracy
- AI safety framework blocks 100% of inappropriate content
- Knowledge processing handles 1000+ documents per hour

AI MODEL SPECIFICATIONS:
General Conversation:
- OpenAI GPT-4 for general architectural discussions
- Context window: 32k tokens
- Temperature: 0.7 for balanced creativity and accuracy

Safety-Focused:
- Anthropic Claude for security and compliance discussions
- Constitutional AI for safety alignment
- Temperature: 0.3 for conservative responses

Code-Specific:
- StarCoder for code analysis and generation
- CodeT5 for code summarization and documentation
- Fine-tuned models for specific programming languages

KNOWLEDGE PROCESSING PIPELINE:
1. Document ingestion and preprocessing
2. Text extraction and cleaning
3. Chunking and embedding generation
4. Vector storage and indexing
5. Knowledge graph entity extraction
6. Relationship identification and storage

Please execute this phase systematically, providing detailed AI model configurations, knowledge processing pipelines, and performance optimization strategies.
```

---

## Phase 5: Performance Optimization & Monitoring
**Duration:** 4 weeks | **Team:** 2 DevOps engineers, 2 backend engineers, 1 AI/ML engineer

### 🚀 EXECUTION PROMPT - PHASE 5

```
You are a senior performance engineer implementing Phase 5 of the Nexus Architect Core Foundation. Your goal is to optimize system performance and implement comprehensive monitoring for production readiness.

CONTEXT:
- Building on AI-enhanced infrastructure from previous phases
- Need to handle 1000+ concurrent users with <200ms response times
- AI workloads require special performance considerations
- Enterprise monitoring and alerting requirements
- Preparation for autonomous operations and scaling

TECHNICAL REQUIREMENTS:
Performance Optimization:
- Horizontal Pod Autoscaler (HPA) for application scaling
- Vertical Pod Autoscaler (VPA) for resource optimization
- Cluster autoscaler for node management
- Custom metrics-based scaling for AI workloads

Load Balancing:
- NGINX Ingress with intelligent routing
- Istio service mesh load balancing
- Database connection pooling and optimization
- CDN integration for static content delivery

Caching Strategy:
- Multi-level caching with Redis
- Application-level caching for frequent queries
- AI model result caching for performance
- Database query result caching

Monitoring & Observability:
- Prometheus for metrics collection and storage
- Grafana dashboards for visualization
- Custom metrics for AI model performance
- AlertManager for intelligent alerting
- Distributed tracing with Jaeger

EXECUTION STEPS:
1. **Week 1: Auto-Scaling and Load Balancing**
   - Configure HPA, VPA, and cluster autoscaler
   - Set up NGINX Ingress with intelligent routing
   - Optimize database connection pooling
   - Implement CDN for static content

2. **Week 2: Caching Strategy Implementation**
   - Deploy multi-level Redis caching
   - Implement application-level caching
   - Set up AI model result caching
   - Configure database query caching

3. **Week 3: Monitoring and Metrics Collection**
   - Deploy Prometheus monitoring stack
   - Create Grafana dashboards for all components
   - Set up custom metrics for AI performance
   - Configure distributed tracing with Jaeger

4. **Week 4: Alerting and Incident Management**
   - Configure AlertManager with intelligent routing
   - Set up PagerDuty integration
   - Create Slack/Teams notification channels
   - Implement escalation procedures

DELIVERABLES CHECKLIST:
□ Auto-scaling system with intelligent resource management
□ Load balancing with optimal traffic distribution
□ Multi-level caching for performance optimization
□ Comprehensive monitoring with Prometheus and Grafana
□ Real-time alerting and incident management
□ Performance benchmarking and testing frameworks
□ Capacity planning and resource optimization
□ Custom dashboards for different stakeholders
□ Distributed tracing for performance analysis
□ SLA monitoring and reporting

VALIDATION CRITERIA:
- System handles 10x baseline load with auto-scaling
- Response times improve by 50% with caching
- Monitoring captures 100% of metrics with <30 second latency
- Alerting achieves <2% false positive rate
- Incident response procedures complete in <10 minutes

PERFORMANCE TARGETS:
API Response Times:
- 95% of requests: <200ms
- 99% of requests: <500ms
- 99.9% of requests: <1000ms

AI Model Inference:
- Simple queries: <1 second
- Complex queries: <3 seconds
- Batch processing: <30 seconds per batch

Database Performance:
- Simple queries: <50ms
- Complex queries: <200ms
- Bulk operations: <5 seconds

MONITORING DASHBOARDS:
Infrastructure Dashboard:
- CPU, memory, disk, network utilization
- Kubernetes cluster health and resource usage
- Database performance and connection pools
- Storage utilization and I/O metrics

Application Dashboard:
- API response times and error rates
- User session and authentication metrics
- Feature usage and adoption rates
- Business KPIs and user engagement

AI Performance Dashboard:
- Model inference times and accuracy
- Token usage and cost tracking
- Model routing and selection metrics
- Knowledge processing performance

ALERTING RULES:
Critical Alerts (PagerDuty):
- System downtime or unavailability
- Database connection failures
- AI model service failures
- Security incidents or breaches

Warning Alerts (Slack/Teams):
- High response times or error rates
- Resource utilization above 80%
- Unusual traffic patterns
- Performance degradation trends

Please execute this phase systematically, providing detailed performance optimization configurations, monitoring setups, and alerting procedures.
```

---

## Phase 6: System Integration & Production Readiness
**Duration:** 4 weeks | **Team:** Full team (8 engineers) for final integration and testing

### 🚀 EXECUTION PROMPT - PHASE 6

```
You are the technical lead for Phase 6 of the Nexus Architect Core Foundation. Your goal is to complete system integration, validate production readiness, and prepare for other workstream integration.

CONTEXT:
- Final phase of Core Foundation workstream
- All previous phases must be integrated and working together
- Production deployment readiness validation required
- Integration points for other workstreams must be prepared
- Comprehensive testing and documentation completion

TECHNICAL REQUIREMENTS:
System Integration:
- End-to-end workflow testing and validation
- Cross-component integration verification
- Performance testing under production load
- Security testing and vulnerability assessment

Production Deployment:
- Blue-green deployment strategy implementation
- Database migration procedures and testing
- Rollback and recovery procedures validation
- Production environment configuration

Operational Procedures:
- Backup and recovery procedures
- Incident response and escalation
- Maintenance and update procedures
- Capacity planning and scaling guidelines

Documentation:
- Architecture documentation and diagrams
- API documentation with examples
- Security procedures and compliance guides
- Operational runbooks and procedures

EXECUTION STEPS:
1. **Week 1: End-to-End Integration and Testing**
   - Complete system integration testing
   - Validate all component interactions
   - Perform comprehensive security testing
   - Execute performance testing under load

2. **Week 2: Production Deployment Procedures**
   - Implement blue-green deployment strategy
   - Test database migration procedures
   - Validate rollback and recovery procedures
   - Configure production environment

3. **Week 3: Documentation and Procedures**
   - Complete technical documentation
   - Finalize operational procedures
   - Create troubleshooting guides
   - Validate all documentation accuracy

4. **Week 4: Final Validation and Handoff**
   - Execute final validation testing
   - Prepare integration points for other workstreams
   - Complete security and compliance validation
   - Conduct knowledge transfer sessions

DELIVERABLES CHECKLIST:
□ Complete system integration with all components
□ Production-ready deployment procedures
□ Comprehensive end-to-end testing suite
□ Security validation and compliance certification
□ Performance benchmarks and capacity planning
□ Complete technical and operational documentation
□ Backup and recovery procedures tested
□ Incident response plans validated
□ Integration points prepared for other workstreams
□ Knowledge transfer and training completed

VALIDATION CRITERIA:
- End-to-end testing passes with 100% success rate
- Production deployment completes without issues
- Security audit passes with zero critical findings
- Performance benchmarks meet all requirements
- Documentation review passes with 100% accuracy

INTEGRATION TESTING SCENARIOS:
User Authentication Flow:
1. User registration and email verification
2. Login with MFA authentication
3. JWT token generation and validation
4. Role-based access control verification
5. Session management and logout

AI Conversation Flow:
1. User authentication and authorization
2. Conversation initiation and context setup
3. AI model selection and routing
4. Response generation and safety filtering
5. Conversation history and knowledge storage

Data Processing Flow:
1. Document upload and validation
2. Content extraction and processing
3. Vector embedding generation
4. Knowledge graph entity extraction
5. Search and retrieval validation

PRODUCTION READINESS CHECKLIST:
Infrastructure:
□ Kubernetes cluster with 99.9% availability
□ Database cluster with automated backup
□ Monitoring and alerting fully operational
□ Security controls and compliance validated
□ Auto-scaling and load balancing tested

Application:
□ All APIs documented and tested
□ Authentication and authorization working
□ AI services integrated and performing
□ Caching and performance optimized
□ Error handling and logging complete

Operations:
□ Deployment procedures tested
□ Backup and recovery validated
□ Incident response procedures ready
□ Monitoring dashboards operational
□ Documentation complete and accurate

INTEGRATION POINTS FOR OTHER WORKSTREAMS:
WS2 (AI Intelligence):
- AI model serving infrastructure ready
- Knowledge processing pipeline operational
- Vector database and search capabilities
- Multi-persona AI foundation prepared

WS3 (Data Ingestion):
- Database schemas for organizational data
- API endpoints for data ingestion
- Security controls for data processing
- Knowledge graph foundation ready

WS4 (Autonomous Capabilities):
- Infrastructure for autonomous operations
- Security framework for automated actions
- Monitoring for autonomous decision tracking
- API foundation for autonomous services

WS5 (Multi-Role Interfaces):
- Authentication and authorization ready
- API foundation for user interfaces
- Real-time capabilities for live updates
- Security controls for user access

WS6 (Integration & Deployment):
- CI/CD pipeline foundation ready
- Monitoring and alerting infrastructure
- Security controls for enterprise integration
- Production deployment procedures validated

Please execute this phase systematically, ensuring all integration points are validated and the foundation is ready for the next workstreams to build upon.
```

---

## 📋 Phase Execution Checklist

### Before Starting Any Phase:
- [ ] Previous phase completed and validated
- [ ] Team members assigned and available
- [ ] Required infrastructure and tools ready
- [ ] Dependencies from other workstreams resolved

### During Phase Execution:
- [ ] Daily standup meetings with progress updates
- [ ] Weekly milestone reviews and validation
- [ ] Continuous integration and testing
- [ ] Documentation updated in real-time

### After Phase Completion:
- [ ] All deliverables completed and validated
- [ ] Success criteria met and documented
- [ ] Integration points tested and verified
- [ ] Knowledge transfer to next phase team
- [ ] Lessons learned documented and shared

## 🔗 Integration Dependencies

### WS1 → WS2 Dependencies:
- AI model serving infrastructure
- Vector database and knowledge processing
- Authentication for AI services
- Security controls for AI operations

### WS1 → WS3 Dependencies:
- Database schemas for organizational data
- API endpoints for data ingestion
- Security framework for data processing
- Infrastructure for real-time processing

### WS1 → WS4 Dependencies:
- Infrastructure for autonomous operations
- Security framework for automated actions
- Monitoring for decision tracking
- API foundation for autonomous services

### WS1 → WS5 Dependencies:
- Authentication and authorization system
- API foundation for user interfaces
- Real-time capabilities for live updates
- Security controls for user access

### WS1 → WS6 Dependencies:
- CI/CD pipeline foundation
- Monitoring and alerting infrastructure
- Security controls for enterprise integration
- Production deployment procedures

---

**Note:** Each execution prompt is designed to be self-contained and can be executed independently when the team is ready. The prompts include all necessary context, requirements, and validation criteria for successful completion.

