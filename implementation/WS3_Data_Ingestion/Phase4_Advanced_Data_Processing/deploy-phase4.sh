#!/bin/bash

# WS3 Phase 4: Advanced Data Processing & Knowledge Extraction Deployment Script
# This script deploys the advanced data processing infrastructure with Apache Spark,
# knowledge extraction capabilities, and comprehensive data quality monitoring.

set -e

echo "🚀 Starting WS3 Phase 4 Deployment: Advanced Data Processing & Knowledge Extraction"
echo "=============================================================================="

# Configuration
NAMESPACE="nexus-data-processing"
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-"production"}
SPARK_VERSION="3.4.0"
POSTGRES_VERSION="15"
REDIS_VERSION="7.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        print_warning "helm is not installed - using kubectl for deployment"
    fi
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        print_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    print_success "Prerequisites check completed"
}

# Create namespace
create_namespace() {
    print_status "Creating namespace: $NAMESPACE"
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    print_success "Namespace created/updated: $NAMESPACE"
}

# Deploy PostgreSQL database
deploy_postgresql() {
    print_status "Deploying PostgreSQL database..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: $NAMESPACE
data:
  POSTGRES_DB: nexus_architect
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: nexus_secure_password_2024
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:$POSTGRES_VERSION
        ports:
        - containerPort: 5432
        envFrom:
        - configMapRef:
            name: postgres-config
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: $NAMESPACE
spec:
  selector:
    app: postgres
  ports:
    - protocol: TCP
      port: 5432
      targetPort: 5432
  type: ClusterIP
EOF

    print_success "PostgreSQL deployed"
}

# Deploy Redis cluster
deploy_redis() {
    print_status "Deploying Redis cluster..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:$REDIS_VERSION
        ports:
        - containerPort: 6379
        command:
          - redis-server
          - --appendonly
          - "yes"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
    - protocol: TCP
      port: 6379
      targetPort: 6379
  type: ClusterIP
EOF

    print_success "Redis deployed"
}

# Deploy Apache Spark cluster
deploy_spark() {
    print_status "Deploying Apache Spark cluster..."
    
    # Spark Master
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spark-master
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spark-master
  template:
    metadata:
      labels:
        app: spark-master
    spec:
      containers:
      - name: spark-master
        image: bitnami/spark:$SPARK_VERSION
        ports:
        - containerPort: 7077
        - containerPort: 8080
        env:
        - name: SPARK_MODE
          value: master
        - name: SPARK_MASTER_HOST
          value: spark-master-service
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: spark-master-service
  namespace: $NAMESPACE
spec:
  selector:
    app: spark-master
  ports:
    - name: spark
      protocol: TCP
      port: 7077
      targetPort: 7077
    - name: web-ui
      protocol: TCP
      port: 8080
      targetPort: 8080
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spark-worker
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: spark-worker
  template:
    metadata:
      labels:
        app: spark-worker
    spec:
      containers:
      - name: spark-worker
        image: bitnami/spark:$SPARK_VERSION
        ports:
        - containerPort: 8081
        env:
        - name: SPARK_MODE
          value: worker
        - name: SPARK_MASTER_URL
          value: spark://spark-master-service:7077
        - name: SPARK_WORKER_MEMORY
          value: 2g
        - name: SPARK_WORKER_CORES
          value: "2"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
EOF

    print_success "Apache Spark cluster deployed"
}

# Deploy Advanced Data Processor
deploy_data_processor() {
    print_status "Deploying Advanced Data Processor..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: advanced-data-processor
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: advanced-data-processor
  template:
    metadata:
      labels:
        app: advanced-data-processor
    spec:
      containers:
      - name: advanced-data-processor
        image: python:3.11-slim
        ports:
        - containerPort: 8001
        env:
        - name: DB_HOST
          value: postgres-service
        - name: DB_PORT
          value: "5432"
        - name: DB_NAME
          value: nexus_architect
        - name: DB_USER
          value: postgres
        - name: DB_PASSWORD
          value: nexus_secure_password_2024
        - name: REDIS_HOST
          value: redis-service
        - name: REDIS_PORT
          value: "6379"
        - name: SPARK_MASTER
          value: spark://spark-master-service:7077
        - name: PORT
          value: "8001"
        command:
        - /bin/bash
        - -c
        - |
          pip install --no-cache-dir pyspark pandas numpy scikit-learn flask flask-cors psycopg2-binary redis prometheus-client
          python /app/advanced_data_processor.py
        volumeMounts:
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: app-code
        configMap:
          name: data-processor-code
---
apiVersion: v1
kind: Service
metadata:
  name: advanced-data-processor-service
  namespace: $NAMESPACE
spec:
  selector:
    app: advanced-data-processor
  ports:
    - protocol: TCP
      port: 8001
      targetPort: 8001
  type: ClusterIP
EOF

    print_success "Advanced Data Processor deployed"
}

# Deploy Knowledge Extractor
deploy_knowledge_extractor() {
    print_status "Deploying Knowledge Extractor..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: knowledge-extractor
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: knowledge-extractor
  template:
    metadata:
      labels:
        app: knowledge-extractor
    spec:
      containers:
      - name: knowledge-extractor
        image: python:3.11-slim
        ports:
        - containerPort: 8002
        env:
        - name: DB_HOST
          value: postgres-service
        - name: DB_PORT
          value: "5432"
        - name: DB_NAME
          value: nexus_architect
        - name: DB_USER
          value: postgres
        - name: DB_PASSWORD
          value: nexus_secure_password_2024
        - name: REDIS_HOST
          value: redis-service
        - name: REDIS_PORT
          value: "6379"
        - name: PORT
          value: "8002"
        command:
        - /bin/bash
        - -c
        - |
          apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
          pip install --no-cache-dir spacy nltk scikit-learn networkx pandas numpy flask flask-cors psycopg2-binary redis prometheus-client
          python -m spacy download en_core_web_sm
          python /app/knowledge_extractor.py
        volumeMounts:
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: app-code
        configMap:
          name: knowledge-extractor-code
---
apiVersion: v1
kind: Service
metadata:
  name: knowledge-extractor-service
  namespace: $NAMESPACE
spec:
  selector:
    app: knowledge-extractor
  ports:
    - protocol: TCP
      port: 8002
      targetPort: 8002
  type: ClusterIP
EOF

    print_success "Knowledge Extractor deployed"
}

# Deploy Data Quality Monitor
deploy_quality_monitor() {
    print_status "Deploying Data Quality Monitor..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-quality-monitor
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: data-quality-monitor
  template:
    metadata:
      labels:
        app: data-quality-monitor
    spec:
      containers:
      - name: data-quality-monitor
        image: python:3.11-slim
        ports:
        - containerPort: 8003
        env:
        - name: DB_HOST
          value: postgres-service
        - name: DB_PORT
          value: "5432"
        - name: DB_NAME
          value: nexus_architect
        - name: DB_USER
          value: postgres
        - name: DB_PASSWORD
          value: nexus_secure_password_2024
        - name: REDIS_HOST
          value: redis-service
        - name: REDIS_PORT
          value: "6379"
        - name: PORT
          value: "8003"
        command:
        - /bin/bash
        - -c
        - |
          apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
          pip install --no-cache-dir pandas numpy scipy scikit-learn great-expectations flask flask-cors psycopg2-binary redis prometheus-client
          python /app/data_quality_monitor.py
        volumeMounts:
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8003
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8003
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: app-code
        configMap:
          name: quality-monitor-code
---
apiVersion: v1
kind: Service
metadata:
  name: data-quality-monitor-service
  namespace: $NAMESPACE
spec:
  selector:
    app: data-quality-monitor
  ports:
    - protocol: TCP
      port: 8003
      targetPort: 8003
  type: ClusterIP
EOF

    print_success "Data Quality Monitor deployed"
}

# Create ConfigMaps for application code
create_configmaps() {
    print_status "Creating ConfigMaps for application code..."
    
    # Advanced Data Processor ConfigMap
    kubectl create configmap data-processor-code \
        --from-file=advanced_data_processor.py=spark-processing/advanced_data_processor.py \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Knowledge Extractor ConfigMap
    kubectl create configmap knowledge-extractor-code \
        --from-file=knowledge_extractor.py=knowledge-extraction/knowledge_extractor.py \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Data Quality Monitor ConfigMap
    kubectl create configmap quality-monitor-code \
        --from-file=data_quality_monitor.py=data-quality/data_quality_monitor.py \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    print_success "ConfigMaps created"
}

# Deploy monitoring and observability
deploy_monitoring() {
    print_status "Deploying monitoring and observability..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: $NAMESPACE
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'advanced-data-processor'
      static_configs:
      - targets: ['advanced-data-processor-service:8001']
      metrics_path: /metrics
    - job_name: 'knowledge-extractor'
      static_configs:
      - targets: ['knowledge-extractor-service:8002']
      metrics_path: /metrics
    - job_name: 'data-quality-monitor'
      static_configs:
      - targets: ['data-quality-monitor-service:8003']
      metrics_path: /metrics
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-service
  namespace: $NAMESPACE
spec:
  selector:
    app: prometheus
  ports:
    - protocol: TCP
      port: 9090
      targetPort: 9090
  type: ClusterIP
EOF

    print_success "Monitoring deployed"
}

# Initialize database schema
initialize_database() {
    print_status "Initializing database schema..."
    
    # Wait for PostgreSQL to be ready
    kubectl wait --for=condition=ready pod -l app=postgres --namespace=$NAMESPACE --timeout=300s
    
    # Create database schema
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: db-init
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: db-init
        image: postgres:$POSTGRES_VERSION
        env:
        - name: PGHOST
          value: postgres-service
        - name: PGPORT
          value: "5432"
        - name: PGDATABASE
          value: nexus_architect
        - name: PGUSER
          value: postgres
        - name: PGPASSWORD
          value: nexus_secure_password_2024
        command:
        - /bin/bash
        - -c
        - |
          # Wait for PostgreSQL to be ready
          until pg_isready; do
            echo "Waiting for PostgreSQL..."
            sleep 2
          done
          
          # Create tables
          psql -c "
          CREATE TABLE IF NOT EXISTS data_sources (
            source_id VARCHAR(255) PRIMARY KEY,
            source_type VARCHAR(100) NOT NULL,
            source_name VARCHAR(255) NOT NULL,
            connection_config JSONB,
            schema_config JSONB,
            processing_config JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          CREATE TABLE IF NOT EXISTS processed_data (
            record_id VARCHAR(255) PRIMARY KEY,
            source_id VARCHAR(255) REFERENCES data_sources(source_id),
            data_type VARCHAR(100),
            content JSONB,
            metadata JSONB,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          CREATE TABLE IF NOT EXISTS extracted_entities (
            entity_id VARCHAR(255) PRIMARY KEY,
            entity_type VARCHAR(100) NOT NULL,
            entity_text TEXT NOT NULL,
            confidence FLOAT,
            start_pos INTEGER,
            end_pos INTEGER,
            properties JSONB,
            source_document VARCHAR(255),
            extraction_method VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          CREATE TABLE IF NOT EXISTS extracted_relationships (
            relationship_id VARCHAR(255) PRIMARY KEY,
            source_entity_id VARCHAR(255) REFERENCES extracted_entities(entity_id),
            target_entity_id VARCHAR(255) REFERENCES extracted_entities(entity_id),
            relationship_type VARCHAR(100) NOT NULL,
            confidence FLOAT,
            evidence_text TEXT,
            properties JSONB,
            source_document VARCHAR(255),
            extraction_method VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          CREATE TABLE IF NOT EXISTS extracted_events (
            event_id VARCHAR(255) PRIMARY KEY,
            event_type VARCHAR(100) NOT NULL,
            event_text TEXT NOT NULL,
            timestamp TIMESTAMP,
            participants JSONB,
            location VARCHAR(255),
            confidence FLOAT,
            properties JSONB,
            source_document VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          CREATE TABLE IF NOT EXISTS extracted_concepts (
            concept_id VARCHAR(255) PRIMARY KEY,
            concept_text TEXT NOT NULL,
            concept_type VARCHAR(100) NOT NULL,
            definition TEXT,
            related_terms JSONB,
            confidence FLOAT,
            properties JSONB,
            source_document VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          CREATE TABLE IF NOT EXISTS quality_reports (
            report_id VARCHAR(255) PRIMARY KEY,
            source_name VARCHAR(255) NOT NULL,
            overall_score FLOAT,
            dimension_scores JSONB,
            recommendations JSONB,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          CREATE TABLE IF NOT EXISTS quality_metrics (
            metric_id VARCHAR(255) PRIMARY KEY,
            report_id VARCHAR(255) REFERENCES quality_reports(report_id),
            metric_name VARCHAR(255) NOT NULL,
            metric_type VARCHAR(100) NOT NULL,
            value FLOAT,
            threshold_warning FLOAT,
            threshold_critical FLOAT,
            status VARCHAR(50),
            details JSONB,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          );
          
          CREATE TABLE IF NOT EXISTS quality_issues (
            issue_id VARCHAR(255) PRIMARY KEY,
            report_id VARCHAR(255) REFERENCES quality_reports(report_id),
            issue_type VARCHAR(100) NOT NULL,
            severity VARCHAR(50) NOT NULL,
            description TEXT,
            affected_records INTEGER,
            source_table VARCHAR(255),
            source_column VARCHAR(255),
            detection_rule VARCHAR(255),
            suggested_action TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP
          );
          
          -- Create indexes for performance
          CREATE INDEX IF NOT EXISTS idx_entities_type ON extracted_entities(entity_type);
          CREATE INDEX IF NOT EXISTS idx_entities_source ON extracted_entities(source_document);
          CREATE INDEX IF NOT EXISTS idx_relationships_type ON extracted_relationships(relationship_type);
          CREATE INDEX IF NOT EXISTS idx_events_type ON extracted_events(event_type);
          CREATE INDEX IF NOT EXISTS idx_concepts_type ON extracted_concepts(concept_type);
          CREATE INDEX IF NOT EXISTS idx_quality_reports_source ON quality_reports(source_name);
          CREATE INDEX IF NOT EXISTS idx_quality_metrics_type ON quality_metrics(metric_type);
          CREATE INDEX IF NOT EXISTS idx_quality_issues_type ON quality_issues(issue_type);
          
          -- Insert sample data sources
          INSERT INTO data_sources (source_id, source_type, source_name, connection_config, schema_config, processing_config)
          VALUES 
          ('git-repos', 'git', 'Git Repositories', '{\"platforms\": [\"github\", \"gitlab\"]}', '{}', '{}'),
          ('docs', 'documentation', 'Documentation Systems', '{\"platforms\": [\"confluence\", \"notion\"]}', '{}', '{}'),
          ('projects', 'project_management', 'Project Management', '{\"platforms\": [\"jira\", \"linear\"]}', '{}', '{}'),
          ('communications', 'communication', 'Communication Platforms', '{\"platforms\": [\"slack\", \"teams\"]}', '{}', '{}')
          ON CONFLICT (source_id) DO NOTHING;
          
          GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
          "
          
          echo "Database schema initialized successfully"
      restartPolicy: Never
  backoffLimit: 3
EOF

    # Wait for job completion
    kubectl wait --for=condition=complete job/db-init --namespace=$NAMESPACE --timeout=300s
    
    print_success "Database schema initialized"
}

# Verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Check pod status
    echo "Pod Status:"
    kubectl get pods -n $NAMESPACE
    
    # Check service status
    echo -e "\nService Status:"
    kubectl get services -n $NAMESPACE
    
    # Wait for all deployments to be ready
    print_status "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available deployment --all --namespace=$NAMESPACE --timeout=600s
    
    # Test service endpoints
    print_status "Testing service endpoints..."
    
    # Port forward for testing (run in background)
    kubectl port-forward -n $NAMESPACE service/advanced-data-processor-service 8001:8001 &
    PF_PID1=$!
    kubectl port-forward -n $NAMESPACE service/knowledge-extractor-service 8002:8002 &
    PF_PID2=$!
    kubectl port-forward -n $NAMESPACE service/data-quality-monitor-service 8003:8003 &
    PF_PID3=$!
    
    sleep 10
    
    # Test health endpoints
    if curl -s http://localhost:8001/health > /dev/null; then
        print_success "Advanced Data Processor health check passed"
    else
        print_warning "Advanced Data Processor health check failed"
    fi
    
    if curl -s http://localhost:8002/health > /dev/null; then
        print_success "Knowledge Extractor health check passed"
    else
        print_warning "Knowledge Extractor health check failed"
    fi
    
    if curl -s http://localhost:8003/health > /dev/null; then
        print_success "Data Quality Monitor health check passed"
    else
        print_warning "Data Quality Monitor health check failed"
    fi
    
    # Clean up port forwards
    kill $PF_PID1 $PF_PID2 $PF_PID3 2>/dev/null || true
    
    print_success "Deployment verification completed"
}

# Generate deployment summary
generate_summary() {
    print_status "Generating deployment summary..."
    
    cat <<EOF

🎉 WS3 Phase 4 Deployment Summary
================================

✅ Infrastructure Deployed:
   • PostgreSQL Database (with comprehensive schema)
   • Redis Cache Cluster
   • Apache Spark Cluster (1 master, 2 workers)
   • Prometheus Monitoring

✅ Services Deployed:
   • Advanced Data Processor (Port 8001)
   • Knowledge Extractor (Port 8002)
   • Data Quality Monitor (Port 8003)

✅ Capabilities Available:
   • Large-scale data processing with Apache Spark
   • Multi-language knowledge extraction (NER, relationships, events, concepts)
   • Comprehensive data quality monitoring and validation
   • Real-time metrics and monitoring
   • Machine learning pipelines for data classification

📊 Performance Targets:
   • Process 100,000+ data records per hour
   • Knowledge extraction >85% accuracy
   • Data quality monitoring >95% issue detection
   • ML classification >90% accuracy

🔗 Service Endpoints:
   • Data Processor: http://advanced-data-processor-service:8001
   • Knowledge Extractor: http://knowledge-extractor-service:8002
   • Quality Monitor: http://data-quality-monitor-service:8003
   • Prometheus: http://prometheus-service:9090

🚀 Next Steps:
   1. Configure data sources and validation rules
   2. Set up automated quality monitoring schedules
   3. Integrate with WS2 Knowledge Graph for entity population
   4. Configure alerting and notification systems
   5. Optimize processing performance based on workload

📚 Documentation:
   • API documentation available at service endpoints
   • Prometheus metrics at /metrics endpoints
   • Health checks at /health endpoints

Namespace: $NAMESPACE
Deployment Environment: $DEPLOYMENT_ENV
Deployment Time: $(date)

EOF

    print_success "WS3 Phase 4 deployment completed successfully!"
}

# Main deployment flow
main() {
    echo "Starting WS3 Phase 4 deployment..."
    
    check_prerequisites
    create_namespace
    
    # Deploy infrastructure
    deploy_postgresql
    deploy_redis
    deploy_spark
    
    # Create application code ConfigMaps
    create_configmaps
    
    # Deploy services
    deploy_data_processor
    deploy_knowledge_extractor
    deploy_quality_monitor
    
    # Deploy monitoring
    deploy_monitoring
    
    # Initialize database
    initialize_database
    
    # Verify deployment
    verify_deployment
    
    # Generate summary
    generate_summary
}

# Run main function
main "$@"

