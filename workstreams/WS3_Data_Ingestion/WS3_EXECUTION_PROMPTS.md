# WS3: Data Ingestion & Processing - Execution Prompts

## Overview
This document contains execution-ready prompts for each phase of WS3: Data Ingestion & Processing. Each prompt can be executed directly when the development team is ready to start that specific phase.

## Prerequisites
- WS1 Core Foundation must be completed (infrastructure, databases, security framework)
- WS2 AI Intelligence Phase 2 knowledge graph foundation ready
- Apache Kafka, Spark, and data processing infrastructure from WS1 operational

---

## Phase 1: Git Repository Integration & Code Analysis
**Duration:** 4 weeks | **Team:** 2 data engineers, 2 backend engineers, 1 DevOps engineer

### 🚀 EXECUTION PROMPT - PHASE 1

```
You are a senior data engineer implementing Phase 1 of the Nexus Architect Data Ingestion workstream. Your goal is to implement comprehensive Git repository integration with intelligent code analysis capabilities.

CONTEXT:
- Building foundation for organizational knowledge extraction from code repositories
- Need real-time integration with major Git platforms (GitHub, GitLab, Bitbucket, Azure DevOps)
- Creating intelligent code analysis for dependency mapping and quality assessment
- Foundation for AI-powered code understanding and architectural insights
- Enterprise-scale processing for large codebases and high change velocity

TECHNICAL REQUIREMENTS:
Git Integration Platforms:
- GitHub (REST API v4, GraphQL API v4, Webhooks)
- GitLab (REST API v4, GraphQL, Webhooks)
- Bitbucket (REST API 2.0, Webhooks)
- Azure DevOps (REST API 7.0, Webhooks)

Data Extraction Capabilities:
Repository Metadata:
- Repository information (name, description, topics, languages)
- Branch and tag information with history
- Contributor and collaboration data
- Issue and pull request metadata
- Release and deployment information

Code Analysis:
- File structure and organization mapping
- Function and class definitions extraction
- Import and dependency relationships
- Code complexity and quality metrics
- Documentation and comment analysis

Change Tracking:
- Commit history and authorship analysis
- Code change patterns and frequency
- Merge and branching strategies
- Code review and approval processes
- Deployment and release patterns

Code Processing Pipeline:
Static Analysis:
- Abstract Syntax Tree (AST) parsing for multiple languages
- Dependency graph construction and analysis
- Code quality metrics (cyclomatic complexity, maintainability)
- Security vulnerability scanning and identification
- Documentation coverage and quality assessment

Language Support:
- Python, JavaScript/TypeScript, Java, C#, Go, Rust
- SQL, HTML/CSS, JSON, YAML, XML
- Shell scripts, Dockerfile, Kubernetes manifests
- Configuration files and infrastructure as code

Real-Time Processing:
- Webhook-based real-time updates
- Incremental processing for code changes
- Conflict detection and resolution
- Change impact analysis and propagation

EXECUTION STEPS:
1. **Week 1: Git Platform API Integration**
   - Set up authentication and API clients for all Git platforms
   - Implement rate limiting and error handling for API calls
   - Create repository discovery and metadata extraction
   - Test integration with sample repositories

2. **Week 2: Repository Metadata Extraction**
   - Build comprehensive metadata extraction pipelines
   - Implement contributor and collaboration data processing
   - Create issue and pull request analysis
   - Set up repository monitoring and tracking

3. **Week 3: Code Analysis Pipeline**
   - Implement AST parsing for multiple programming languages
   - Build dependency graph construction and analysis
   - Create code quality metrics and security scanning
   - Develop documentation analysis and coverage assessment

4. **Week 4: Real-Time Updates and Change Tracking**
   - Deploy webhook processing for real-time updates
   - Implement incremental processing for code changes
   - Create change impact analysis and propagation
   - Set up monitoring and alerting for processing pipeline

DELIVERABLES CHECKLIST:
□ Git repository connectors for all major platforms
□ Code analysis pipeline with multi-language support
□ Repository metadata extraction and storage systems
□ Real-time webhook processing for code changes
□ Dependency graph construction and analysis
□ Code quality metrics and security scanning
□ Change tracking and impact analysis capabilities
□ Git integration APIs and management interfaces

VALIDATION CRITERIA:
- Successfully integrate with 100+ repositories across platforms
- Code analysis processes 10,000+ files per hour
- Real-time updates processed within 30 seconds of changes
- Dependency graph construction achieves >95% accuracy
- Security scanning identifies vulnerabilities with <1% false positives

INTEGRATION POINTS:
- WS2 Knowledge Graph: Population with code entities and relationships
- WS2 AI Intelligence: Code understanding and analysis capabilities
- WS5 User Interfaces: Repository exploration and insights
- WS4 Autonomous Capabilities: Code quality and security assessment

Please execute this phase systematically, providing detailed API integrations, code analysis implementations, and real-time processing capabilities.
```

---

## Phase 2: Documentation Systems Integration
**Duration:** 4 weeks | **Team:** 2 data engineers, 1 backend engineer, 1 integration specialist, 1 security engineer

### 🚀 EXECUTION PROMPT - PHASE 2

```
You are a senior integration engineer implementing Phase 2 of the Nexus Architect Data Ingestion workstream. Your goal is to integrate with major documentation platforms and create intelligent document processing capabilities.

CONTEXT:
- Building on Git repository integration from Phase 1
- Need comprehensive integration with documentation platforms (Confluence, SharePoint, Notion, etc.)
- Creating intelligent document parsing and content extraction
- Foundation for AI-powered documentation understanding and Q&A
- Enterprise-scale document processing with format diversity

TECHNICAL REQUIREMENTS:
Documentation Platforms:
Confluence Integration:
- REST API for space and page access
- Content extraction with formatting preservation
- Attachment and media file processing
- User and permission information extraction
- Version history and change tracking

SharePoint Integration:
- Microsoft Graph API for document access
- Office 365 integration for real-time updates
- Document library and folder structure extraction
- Metadata and taxonomy information processing
- Collaboration and sharing information

Modern Platforms:
- Notion API for database and page content
- GitBook API for documentation sites
- Markdown file processing from repositories
- Wiki systems and knowledge bases
- Custom documentation platforms via APIs

Document Processing:
Content Extraction:
- Text extraction from various document formats
- Image and diagram extraction and analysis
- Table and structured data extraction
- Link and reference identification
- Metadata and annotation processing

Format Support:
- Microsoft Office documents (Word, PowerPoint, Excel)
- PDF documents with OCR capabilities
- Markdown and reStructuredText
- HTML and web-based documentation
- Image formats with text extraction (OCR)

Intelligence Processing:
- Natural language processing for content understanding
- Topic modeling and categorization
- Entity extraction and relationship identification
- Summary generation and key point extraction
- Cross-reference and citation analysis

EXECUTION STEPS:
1. **Week 1: Documentation Platform API Integration**
   - Set up authentication and API clients for documentation platforms
   - Implement content discovery and metadata extraction
   - Create permission and access control handling
   - Test integration with sample documentation spaces

2. **Week 2: Document Extraction and Format Processing**
   - Build multi-format document processing pipeline
   - Implement OCR for image-based content extraction
   - Create structured data extraction from tables and forms
   - Develop attachment and media file processing

3. **Week 3: Content Analysis and Intelligence Processing**
   - Deploy natural language processing for content understanding
   - Implement topic modeling and categorization
   - Create entity extraction and relationship identification
   - Build summary generation and key point extraction

4. **Week 4: Version Tracking and Real-Time Updates**
   - Implement document version tracking and change detection
   - Set up real-time updates for document modifications
   - Create cross-reference and citation analysis
   - Deploy document search and retrieval capabilities

DELIVERABLES CHECKLIST:
□ Documentation platform connectors for major systems
□ Multi-format document processing pipeline
□ Content extraction with intelligence analysis
□ Document versioning and change tracking
□ Cross-reference and citation analysis
□ Real-time documentation updates
□ Document search and retrieval capabilities
□ Documentation integration APIs and interfaces

VALIDATION CRITERIA:
- Process 1,000+ documents per hour across formats
- Content extraction accuracy >90% for text and structure
- Real-time updates processed within 60 seconds
- Search relevance >85% for documentation queries
- Cross-reference analysis identifies >80% of document relationships

INTEGRATION POINTS:
- WS2 Knowledge Graph: Population with documentation entities
- WS2 AI Intelligence: Document understanding and Q&A capabilities
- WS5 User Interfaces: Documentation search and exploration
- WS2 Conversational AI: Documentation-based responses

Please execute this phase systematically, providing detailed platform integrations, document processing pipelines, and intelligence analysis capabilities.
```

---

## Phase 3: Project Management & Communication Integration
**Duration:** 4 weeks | **Team:** 2 data engineers, 1 backend engineer, 1 integration specialist, 1 DevOps engineer

### 🚀 EXECUTION PROMPT - PHASE 3

```
You are a senior integration engineer implementing Phase 3 of the Nexus Architect Data Ingestion workstream. Your goal is to integrate with project management and communication platforms for comprehensive organizational data capture.

CONTEXT:
- Building on documentation integration from Phase 2
- Need integration with project management tools (Jira, Linear, Asana) and communication platforms (Slack, Teams)
- Creating unified project and team activity tracking
- Foundation for AI-powered project insights and team collaboration analysis
- Enterprise-scale processing for high-volume project and communication data

TECHNICAL REQUIREMENTS:
Project Management Integration:
Jira Integration:
- REST API for issue and project data
- Workflow and status tracking
- Custom field and metadata extraction
- Sprint and epic information processing
- Time tracking and estimation data

Modern PM Tools:
- Linear API for issue tracking and project management
- Asana API for task and project coordination
- Trello API for Kanban-style project management
- Azure DevOps Work Items for Microsoft ecosystem
- Custom project management tools via APIs

Data Extraction:
- Issue and task information with full history
- Project and epic structure and relationships
- Team assignments and workload distribution
- Timeline and milestone tracking
- Workflow and process analysis

Communication Integration:
Slack Integration:
- Conversations API for channel and direct messages
- User and team information extraction
- File and attachment processing
- Workflow and automation data
- Integration with other tools and services

Microsoft Teams:
- Microsoft Graph API for team and channel data
- Message and conversation extraction
- File sharing and collaboration tracking
- Meeting and call information processing
- Integration with Office 365 ecosystem

Processing Capabilities:
- Natural language processing for message content
- Sentiment analysis and team dynamics insights
- Topic modeling and discussion categorization
- Decision tracking and action item extraction
- Knowledge sharing and expertise identification

EXECUTION STEPS:
1. **Week 1: Project Management Platform Integration**
   - Set up API clients and authentication for PM platforms
   - Implement issue and project data extraction
   - Create workflow and status tracking
   - Test integration with sample projects and workflows

2. **Week 2: Communication Platform Integration**
   - Deploy Slack and Teams API integration
   - Implement message and conversation extraction
   - Create file and attachment processing
   - Set up user and team information extraction

3. **Week 3: Data Processing and Analysis Pipelines**
   - Build natural language processing for communication content
   - Implement sentiment analysis and team dynamics insights
   - Create topic modeling and discussion categorization
   - Develop decision tracking and action item extraction

4. **Week 4: Unified Tracking and Real-Time Updates**
   - Create unified project and activity tracking
   - Implement real-time updates from all platforms
   - Build project insights and analytics
   - Deploy integration APIs for project and communication data

DELIVERABLES CHECKLIST:
□ Project management platform connectors for major tools
□ Communication platform integration (Slack, Teams)
□ Project data extraction and processing pipelines
□ Team collaboration and communication analysis
□ Unified project and activity tracking systems
□ Real-time updates from all platforms
□ Project insights and analytics capabilities
□ Integration APIs for project and communication data

VALIDATION CRITERIA:
- Integrate with 10+ project management and communication platforms
- Process 10,000+ messages and tasks per hour
- Real-time updates processed within 45 seconds
- Project insights accuracy >80% for timeline and resource predictions
- Communication analysis identifies key decisions with >85% accuracy

INTEGRATION POINTS:
- WS2 Knowledge Graph: Population with project and team entities
- WS2 AI Intelligence: Project insights and recommendations
- WS5 User Interfaces: Project tracking and team collaboration
- WS4 Autonomous Capabilities: Project management assistance

Please execute this phase systematically, providing detailed platform integrations, data processing pipelines, and analytics capabilities.
```

---

## Phase 4: Advanced Data Processing & Knowledge Extraction
**Duration:** 4 weeks | **Team:** 2 data engineers, 2 backend engineers, 1 integration specialist, 1 security engineer

### 🚀 EXECUTION PROMPT - PHASE 4

```
You are a senior data processing engineer implementing Phase 4 of the Nexus Architect Data Ingestion workstream. Your goal is to implement advanced data processing pipelines and intelligent knowledge extraction capabilities.

CONTEXT:
- Building on project management and communication integration from Phase 3
- Need advanced data processing with Apache Spark for large-scale transformation
- Creating intelligent knowledge extraction from all organizational data sources
- Foundation for AI-powered insights and automated data classification
- Enterprise-scale processing with comprehensive data quality monitoring

TECHNICAL REQUIREMENTS:
Advanced Processing Pipeline:
Apache Spark Integration:
- Distributed processing for large-scale data transformation
- Machine learning pipelines for data classification
- Graph processing for relationship analysis
- Stream processing for real-time data transformation

Knowledge Extraction:
- Named entity recognition (NER) for person, organization, location
- Relationship extraction for entity connections
- Event extraction for timeline and process understanding
- Concept extraction for domain knowledge identification

Data Transformation:
- Schema normalization across different data sources
- Data deduplication and conflict resolution
- Temporal data alignment and synchronization
- Cross-reference resolution and entity linking

Machine Learning:
- Automated data classification and categorization
- Anomaly detection for data quality issues
- Pattern recognition for process and workflow analysis
- Predictive modeling for data trends and insights

Data Quality Framework:
Quality Metrics:
- Completeness: Percentage of required fields populated
- Accuracy: Correctness of extracted information
- Consistency: Alignment across different data sources
- Timeliness: Freshness and currency of data
- Validity: Conformance to expected formats and ranges

Monitoring Systems:
- Real-time data quality dashboards
- Automated quality alerts and notifications
- Data lineage tracking and impact analysis
- Quality trend analysis and reporting

Validation Procedures:
- Schema validation for structured data
- Content validation for extracted information
- Cross-source validation for consistency checking
- Expert validation for domain-specific accuracy

EXECUTION STEPS:
1. **Week 1: Advanced Processing Pipeline with Apache Spark**
   - Deploy Apache Spark cluster for distributed processing
   - Implement large-scale data transformation pipelines
   - Create machine learning pipelines for data classification
   - Set up graph processing for relationship analysis

2. **Week 2: Knowledge Extraction and Machine Learning**
   - Deploy named entity recognition and relationship extraction
   - Implement event and concept extraction capabilities
   - Create automated data classification and categorization
   - Build anomaly detection for data quality issues

3. **Week 3: Data Quality Framework and Monitoring**
   - Implement comprehensive data quality metrics
   - Deploy real-time data quality dashboards
   - Create automated quality alerts and notifications
   - Set up data lineage tracking and impact analysis

4. **Week 4: Validation Procedures and Optimization**
   - Deploy schema and content validation procedures
   - Implement cross-source validation for consistency
   - Create expert validation workflows
   - Optimize processing performance and resource utilization

DELIVERABLES CHECKLIST:
□ Advanced data processing pipeline with Apache Spark
□ Intelligent knowledge extraction from all data sources
□ Machine learning models for automated data classification
□ Comprehensive data quality monitoring framework
□ Data validation and consistency checking procedures
□ Real-time data quality dashboards and alerts
□ Data lineage tracking and impact analysis
□ Advanced processing APIs and management interfaces

VALIDATION CRITERIA:
- Process 100,000+ data records per hour with Apache Spark
- Knowledge extraction achieves >85% accuracy for entity recognition
- Machine learning classification achieves >90% accuracy
- Data quality monitoring detects >95% of quality issues
- Cross-source validation identifies >80% of inconsistencies

INTEGRATION POINTS:
- WS2 Knowledge Graph: Population with extracted entities and relationships
- WS2 AI Intelligence: Enhanced understanding and reasoning capabilities
- WS5 User Interfaces: Data quality monitoring and insights
- WS4 Autonomous Capabilities: Data-driven decision making

Please execute this phase systematically, providing detailed processing pipelines, knowledge extraction implementations, and data quality frameworks.
```

---

## Phase 5: Real-Time Streaming & Event Processing
**Duration:** 4 weeks | **Team:** 2 data engineers, 1 backend engineer, 1 DevOps engineer, 1 integration specialist

### 🚀 EXECUTION PROMPT - PHASE 5

```
You are a senior streaming engineer implementing Phase 5 of the Nexus Architect Data Ingestion workstream. Your goal is to implement comprehensive real-time streaming architecture and event-driven processing capabilities.

CONTEXT:
- Building on advanced data processing from Phase 4
- Need real-time streaming with Apache Kafka for immediate knowledge updates
- Creating event-driven processing for continuous data transformation
- Foundation for real-time AI insights and autonomous decision-making
- Enterprise-scale streaming with high throughput and low latency requirements

TECHNICAL REQUIREMENTS:
Streaming Architecture:
Apache Kafka:
- Multi-topic event streaming for different data sources
- Partitioning strategy for scalable processing
- Replication and fault tolerance configuration
- Schema registry for event structure management

Stream Processing:
- Kafka Streams for real-time data transformation
- Apache Flink for complex event processing
- Real-time aggregation and windowing operations
- Event correlation and pattern detection

Event Types:
- Code changes and repository updates
- Documentation modifications and additions
- Project management updates and status changes
- Communication messages and collaboration events
- System metrics and performance data

Real-Time Analytics:
Stream Analytics:
- Real-time trend analysis and pattern detection
- Anomaly detection for unusual activity patterns
- Performance monitoring and alerting
- User behavior analysis and insights

Event Correlation:
- Cross-source event correlation and analysis
- Timeline reconstruction for incident analysis
- Impact analysis for change propagation
- Causal relationship identification

Monitoring Dashboards:
- Real-time data flow visualization
- Processing performance and throughput metrics
- Error rates and quality indicators
- System health and resource utilization

EXECUTION STEPS:
1. **Week 1: Kafka Streaming Infrastructure**
   - Deploy production-ready Kafka cluster
   - Configure topics and partitioning strategy
   - Set up replication and fault tolerance
   - Implement schema registry for event management

2. **Week 2: Stream Processing Implementation**
   - Deploy Kafka Streams for real-time transformation
   - Implement Apache Flink for complex event processing
   - Create real-time aggregation and windowing
   - Build event correlation and pattern detection

3. **Week 3: Real-Time Analytics and Event Correlation**
   - Implement real-time trend analysis and pattern detection
   - Deploy anomaly detection for activity patterns
   - Create cross-source event correlation
   - Build timeline reconstruction and impact analysis

4. **Week 4: Monitoring and Performance Optimization**
   - Deploy real-time monitoring dashboards
   - Implement performance and throughput monitoring
   - Create alerting and notification systems
   - Optimize streaming performance and resource utilization

DELIVERABLES CHECKLIST:
□ Production-ready Kafka streaming infrastructure
□ Real-time stream processing for all data sources
□ Event correlation and pattern detection capabilities
□ Real-time analytics and monitoring dashboards
□ Stream processing performance optimization
□ Event-driven knowledge graph updates
□ Real-time alerting and notification systems
□ Streaming APIs and management interfaces

VALIDATION CRITERIA:
- Process 100,000+ events per second with Kafka
- Stream processing latency <100ms for 95% of events
- Event correlation accuracy >90% for related events
- Real-time analytics provide insights within 5 seconds
- System maintains 99.9% uptime under peak load

INTEGRATION POINTS:
- WS2 Knowledge Graph: Real-time updates from all data sources
- WS2 AI Intelligence: Immediate insights and responses
- WS5 User Interfaces: Real-time monitoring and alerts
- WS4 Autonomous Capabilities: Real-time decision making

Please execute this phase systematically, providing detailed streaming architectures, event processing implementations, and real-time analytics capabilities.
```

---

## Phase 6: Data Privacy, Security & Production Optimization
**Duration:** 4 weeks | **Team:** Full team (7 engineers) for final optimization and security hardening

### 🚀 EXECUTION PROMPT - PHASE 6

```
You are the technical lead for Phase 6 of the Nexus Architect Data Ingestion workstream. Your goal is to implement comprehensive data privacy and security controls, optimize performance, and prepare for production deployment.

CONTEXT:
- Final phase of Data Ingestion workstream with all processing capabilities
- Need comprehensive data privacy and security for enterprise compliance
- Production optimization for enterprise-scale data processing
- Integration with all other workstreams for complete system functionality
- Full compliance with major data protection regulations

TECHNICAL REQUIREMENTS:
Privacy & Security:
Data Privacy:
- Personal data identification and classification
- Privacy-preserving data processing techniques
- Data anonymization and pseudonymization
- Consent management and data subject rights

Security Controls:
- End-to-end encryption for data in transit and at rest
- Access controls and authentication for all data sources
- Audit logging for all data access and processing
- Data loss prevention and monitoring

Compliance Framework:
- GDPR compliance for European data subjects
- CCPA compliance for California residents
- HIPAA compliance for healthcare data
- SOC 2 compliance for enterprise customers

Performance Optimization:
Processing Optimization:
- Pipeline performance tuning and optimization
- Resource allocation and scaling strategies
- Caching strategies for frequently accessed data
- Batch processing optimization for large datasets

Storage Optimization:
- Data compression and storage efficiency
- Archival strategies for historical data
- Backup and recovery optimization
- Cost optimization for cloud storage

Monitoring Enhancement:
- Advanced performance monitoring and alerting
- Predictive analytics for capacity planning
- Cost monitoring and optimization recommendations
- SLA monitoring and compliance tracking

EXECUTION STEPS:
1. **Week 1: Data Privacy and Security Controls**
   - Implement personal data identification and classification
   - Deploy privacy-preserving data processing techniques
   - Create data anonymization and pseudonymization
   - Set up end-to-end encryption for all data flows

2. **Week 2: Compliance Framework and Audit Procedures**
   - Implement GDPR, CCPA, HIPAA compliance controls
   - Deploy comprehensive audit logging systems
   - Create data subject rights and consent management
   - Set up compliance monitoring and reporting

3. **Week 3: Performance Optimization and Resource Tuning**
   - Optimize data processing pipelines for performance
   - Implement advanced caching and storage strategies
   - Deploy predictive analytics for capacity planning
   - Create cost optimization and monitoring

4. **Week 4: Final Integration and Production Preparation**
   - Complete integration testing with all workstreams
   - Validate production readiness and performance
   - Finalize compliance and security validation
   - Deploy comprehensive monitoring and alerting

DELIVERABLES CHECKLIST:
□ Comprehensive data privacy and security framework
□ Full compliance with major data protection regulations
□ Optimized data processing performance and resource utilization
□ Complete system integration with all workstreams
□ Production-ready deployment with monitoring and alerting
□ Comprehensive testing and validation results
□ Data governance and compliance documentation
□ Performance benchmarks and optimization guidelines

VALIDATION CRITERIA:
- Privacy controls protect 100% of personal data
- Security testing passes with zero critical vulnerabilities
- Compliance audit achieves 100% pass rate
- Performance optimization improves throughput by 40%
- End-to-end integration testing passes all scenarios
- Production deployment ready with full monitoring

INTEGRATION POINTS:
- Complete integration with all other workstreams
- WS2 Knowledge Graph: Privacy-compliant data population
- WS2 AI Intelligence: Secure and compliant data access
- WS5 User Interfaces: Privacy controls and transparency
- WS4 Autonomous Capabilities: Secure automated data processing

Please execute this phase systematically, ensuring all privacy and security controls are implemented and the system is ready for enterprise production deployment with full compliance.
```

---

## 📋 Phase Execution Checklist

### Before Starting Any Phase:
- [ ] Previous phase completed and validated
- [ ] WS1 Core Foundation dependencies met
- [ ] WS2 AI Intelligence knowledge graph foundation ready
- [ ] Team members assigned and available
- [ ] Required infrastructure and tools ready

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

### WS3 → WS2 Dependencies:
- Knowledge graph population with organizational data
- Training data for AI model improvement
- Real-time data for knowledge updates
- Document processing for AI understanding

### WS3 → WS4 Dependencies:
- Data sources for autonomous decision-making
- Real-time data streams for automated actions
- Quality data for reliable autonomous operations
- Historical data for pattern recognition

### WS3 → WS5 Dependencies:
- Data for user interface dashboards and insights
- Real-time updates for live user experiences
- Search and retrieval capabilities for user queries
- Analytics data for user behavior insights

### WS3 → WS6 Dependencies:
- Data integration capabilities for enterprise systems
- Real-time data flows for production monitoring
- Compliance data for regulatory reporting
- Performance data for system optimization

---

**Note:** Each execution prompt is designed to be self-contained and can be executed independently when the team is ready. The prompts include all necessary context, requirements, and validation criteria for successful completion.

