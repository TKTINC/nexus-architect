#!/bin/bash

# WS3 Phase 2: Documentation Systems Integration Deployment Script
# Comprehensive deployment for documentation platform integration

set -e

echo "🚀 Starting WS3 Phase 2: Documentation Systems Integration Deployment"

# Configuration
NAMESPACE="nexus-architect"
PHASE_DIR="$(dirname "$0")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is available
    if ! command -v docker &> /dev/null; then
        error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Python 3.8+ is available
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        error "Python 3.8+ is required"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        success "Namespace $NAMESPACE created"
    fi
}

# Install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    # Create requirements file
    cat > "$PHASE_DIR/requirements.txt" << EOF
# Core dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
asyncio-mqtt==0.16.1
aiofiles==23.2.1
aiohttp==3.9.1

# Database
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23
alembic==1.13.1

# Redis
redis[hiredis]==5.0.1
aioredis==2.0.1

# Document processing
python-docx==1.1.0
python-pptx==0.6.23
openpyxl==3.1.2
PyPDF2==3.0.1
PyMuPDF==1.23.8
beautifulsoup4==4.12.2
markdown==3.5.1
Pillow==10.1.0
pytesseract==0.3.10

# NLP and ML
spacy==3.7.2
nltk==3.8.1
scikit-learn==1.3.2
numpy==1.25.2
networkx==3.2.1

# Monitoring
prometheus-client==0.19.0
structlog==23.2.0

# Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Configuration
pyyaml==6.0.1
python-dotenv==1.0.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
EOF

    # Install dependencies
    pip3 install -r "$PHASE_DIR/requirements.txt"
    
    success "Python dependencies installed"
}

# Create ConfigMaps
create_configmaps() {
    log "Creating ConfigMaps..."
    
    # Documentation Platform Manager ConfigMap
    cat > "$PHASE_DIR/configmap-platform-manager.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: documentation-platform-config
  namespace: $NAMESPACE
data:
  config.yaml: |
    platforms:
      - platform: confluence
        auth_type: basic
        base_url: "https://company.atlassian.net/wiki"
        rate_limit:
          requests_per_minute: 100
          burst_limit: 20
      - platform: sharepoint
        auth_type: oauth
        base_url: "https://graph.microsoft.com/v1.0"
        rate_limit:
          requests_per_minute: 200
          burst_limit: 50
      - platform: notion
        auth_type: token
        base_url: "https://api.notion.com/v1"
        rate_limit:
          requests_per_minute: 30
          burst_limit: 10
    
    processing:
      max_concurrent_documents: 10
      content_extraction_timeout: 300
      ocr_enabled: true
      nlp_enabled: true
    
    storage:
      max_versions_per_document: 100
      auto_cleanup_enabled: true
      retention_days: 365
    
    monitoring:
      metrics_enabled: true
      metrics_port: 8090
      health_check_interval: 30
EOF

    # Document Processor ConfigMap
    cat > "$PHASE_DIR/configmap-document-processor.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: document-processor-config
  namespace: $NAMESPACE
data:
  config.yaml: |
    supported_formats:
      - pdf
      - docx
      - pptx
      - xlsx
      - markdown
      - html
      - txt
    
    processing:
      max_file_size_mb: 100
      ocr_enabled: true
      nlp_enabled: true
      parallel_processing: true
      max_workers: 4
    
    quality:
      min_confidence_threshold: 0.7
      accuracy_target: 0.9
      performance_target_ms: 2000
    
    cache:
      enabled: true
      ttl_seconds: 3600
      max_size_mb: 1000
EOF

    # Semantic Analyzer ConfigMap
    cat > "$PHASE_DIR/configmap-semantic-analyzer.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: semantic-analyzer-config
  namespace: $NAMESPACE
data:
  config.yaml: |
    nlp:
      spacy_model: "en_core_web_sm"
      nltk_data_path: "/opt/nltk_data"
      max_text_length: 1000000
    
    entity_extraction:
      confidence_threshold: 0.7
      custom_patterns_enabled: true
      entity_types:
        - person
        - organization
        - location
        - technology
        - concept
        - process
        - product
    
    relationship_extraction:
      proximity_threshold: 100
      confidence_threshold: 0.6
      dependency_parsing_enabled: true
    
    topic_modeling:
      num_topics: 10
      min_documents: 5
      coherence_threshold: 0.5
    
    knowledge_graph:
      enabled: true
      max_nodes: 100000
      max_edges: 500000
EOF

    # Apply ConfigMaps
    kubectl apply -f "$PHASE_DIR/configmap-platform-manager.yaml"
    kubectl apply -f "$PHASE_DIR/configmap-document-processor.yaml"
    kubectl apply -f "$PHASE_DIR/configmap-semantic-analyzer.yaml"
    
    success "ConfigMaps created"
}

# Create Secrets
create_secrets() {
    log "Creating Secrets..."
    
    # Documentation Platform Credentials Secret
    cat > "$PHASE_DIR/secret-platform-credentials.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: documentation-platform-credentials
  namespace: $NAMESPACE
type: Opaque
data:
  confluence_username: $(echo -n "admin@company.com" | base64)
  confluence_password: $(echo -n "api_token_placeholder" | base64)
  sharepoint_client_id: $(echo -n "client_id_placeholder" | base64)
  sharepoint_client_secret: $(echo -n "client_secret_placeholder" | base64)
  notion_token: $(echo -n "notion_token_placeholder" | base64)
  github_token: $(echo -n "github_token_placeholder" | base64)
  gitlab_token: $(echo -n "gitlab_token_placeholder" | base64)
EOF

    # Database Secret
    cat > "$PHASE_DIR/secret-database.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: documentation-database-secret
  namespace: $NAMESPACE
type: Opaque
data:
  username: $(echo -n "nexus_docs" | base64)
  password: $(echo -n "secure_password_placeholder" | base64)
  database: $(echo -n "nexus_documentation" | base64)
EOF

    # Redis Secret
    cat > "$PHASE_DIR/secret-redis.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: documentation-redis-secret
  namespace: $NAMESPACE
type: Opaque
data:
  password: $(echo -n "redis_password_placeholder" | base64)
EOF

    # Apply Secrets
    kubectl apply -f "$PHASE_DIR/secret-platform-credentials.yaml"
    kubectl apply -f "$PHASE_DIR/secret-database.yaml"
    kubectl apply -f "$PHASE_DIR/secret-redis.yaml"
    
    success "Secrets created"
}

# Deploy PostgreSQL for document metadata
deploy_postgresql() {
    log "Deploying PostgreSQL for document metadata..."
    
    cat > "$PHASE_DIR/postgresql-deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: documentation-postgresql
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: documentation-postgresql
  template:
    metadata:
      labels:
        app: documentation-postgresql
    spec:
      containers:
      - name: postgresql
        image: postgres:15
        env:
        - name: POSTGRES_DB
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: database
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgresql-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: postgresql-storage
        persistentVolumeClaim:
          claimName: documentation-postgresql-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: documentation-postgresql-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: documentation-postgresql-service
  namespace: $NAMESPACE
spec:
  selector:
    app: documentation-postgresql
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
EOF

    kubectl apply -f "$PHASE_DIR/postgresql-deployment.yaml"
    success "PostgreSQL deployed"
}

# Deploy Redis for caching
deploy_redis() {
    log "Deploying Redis for caching..."
    
    cat > "$PHASE_DIR/redis-deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: documentation-redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: documentation-redis
  template:
    metadata:
      labels:
        app: documentation-redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        - --requirepass
        - \$(REDIS_PASSWORD)
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-redis-secret
              key: password
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: documentation-redis-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: documentation-redis-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: documentation-redis-service
  namespace: $NAMESPACE
spec:
  selector:
    app: documentation-redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
EOF

    kubectl apply -f "$PHASE_DIR/redis-deployment.yaml"
    success "Redis deployed"
}

# Deploy Documentation Platform Manager
deploy_platform_manager() {
    log "Deploying Documentation Platform Manager..."
    
    cat > "$PHASE_DIR/platform-manager-deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: documentation-platform-manager
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: documentation-platform-manager
  template:
    metadata:
      labels:
        app: documentation-platform-manager
    spec:
      containers:
      - name: platform-manager
        image: python:3.11-slim
        command:
        - /bin/bash
        - -c
        - |
          pip install -r /app/requirements.txt
          python -m uvicorn main:app --host 0.0.0.0 --port 8080
        workingDir: /app
        ports:
        - containerPort: 8080
        - containerPort: 8090  # Metrics
        env:
        - name: DATABASE_URL
          value: "postgresql://\$(DB_USER):\$(DB_PASSWORD)@documentation-postgresql-service:5432/\$(DB_NAME)"
        - name: REDIS_URL
          value: "redis://:\$(REDIS_PASSWORD)@documentation-redis-service:6379/0"
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: password
        - name: DB_NAME
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: database
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-redis-secret
              key: password
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: config
          mountPath: /app/config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: app-code
        configMap:
          name: platform-manager-code
      - name: config
        configMap:
          name: documentation-platform-config
---
apiVersion: v1
kind: Service
metadata:
  name: documentation-platform-manager-service
  namespace: $NAMESPACE
spec:
  selector:
    app: documentation-platform-manager
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 8090
    targetPort: 8090
  type: ClusterIP
EOF

    kubectl apply -f "$PHASE_DIR/platform-manager-deployment.yaml"
    success "Documentation Platform Manager deployed"
}

# Deploy Document Processor
deploy_document_processor() {
    log "Deploying Document Processor..."
    
    cat > "$PHASE_DIR/document-processor-deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: documentation-processor
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: documentation-processor
  template:
    metadata:
      labels:
        app: documentation-processor
    spec:
      containers:
      - name: processor
        image: python:3.11-slim
        command:
        - /bin/bash
        - -c
        - |
          apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng
          pip install -r /app/requirements.txt
          python -m spacy download en_core_web_sm
          python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
          python -m uvicorn main:app --host 0.0.0.0 --port 8081
        workingDir: /app
        ports:
        - containerPort: 8081
        - containerPort: 8091  # Metrics
        env:
        - name: DATABASE_URL
          value: "postgresql://\$(DB_USER):\$(DB_PASSWORD)@documentation-postgresql-service:5432/\$(DB_NAME)"
        - name: REDIS_URL
          value: "redis://:\$(REDIS_PASSWORD)@documentation-redis-service:6379/1"
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: password
        - name: DB_NAME
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: database
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-redis-secret
              key: password
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: config
          mountPath: /app/config
        - name: temp-storage
          mountPath: /tmp/processing
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
            port: 8081
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: app-code
        configMap:
          name: document-processor-code
      - name: config
        configMap:
          name: document-processor-config
      - name: temp-storage
        emptyDir:
          sizeLimit: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: documentation-processor-service
  namespace: $NAMESPACE
spec:
  selector:
    app: documentation-processor
  ports:
  - name: http
    port: 8081
    targetPort: 8081
  - name: metrics
    port: 8091
    targetPort: 8091
  type: ClusterIP
EOF

    kubectl apply -f "$PHASE_DIR/document-processor-deployment.yaml"
    success "Document Processor deployed"
}

# Deploy Semantic Analyzer
deploy_semantic_analyzer() {
    log "Deploying Semantic Analyzer..."
    
    cat > "$PHASE_DIR/semantic-analyzer-deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: documentation-semantic-analyzer
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: documentation-semantic-analyzer
  template:
    metadata:
      labels:
        app: documentation-semantic-analyzer
    spec:
      containers:
      - name: semantic-analyzer
        image: python:3.11-slim
        command:
        - /bin/bash
        - -c
        - |
          pip install -r /app/requirements.txt
          python -m spacy download en_core_web_sm
          python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
          python -m uvicorn main:app --host 0.0.0.0 --port 8082
        workingDir: /app
        ports:
        - containerPort: 8082
        - containerPort: 8092  # Metrics
        env:
        - name: DATABASE_URL
          value: "postgresql://\$(DB_USER):\$(DB_PASSWORD)@documentation-postgresql-service:5432/\$(DB_NAME)"
        - name: REDIS_URL
          value: "redis://:\$(REDIS_PASSWORD)@documentation-redis-service:6379/2"
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: password
        - name: DB_NAME
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: database
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-redis-secret
              key: password
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: config
          mountPath: /app/config
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
            port: 8082
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8082
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: app-code
        configMap:
          name: semantic-analyzer-code
      - name: config
        configMap:
          name: semantic-analyzer-config
---
apiVersion: v1
kind: Service
metadata:
  name: documentation-semantic-analyzer-service
  namespace: $NAMESPACE
spec:
  selector:
    app: documentation-semantic-analyzer
  ports:
  - name: http
    port: 8082
    targetPort: 8082
  - name: metrics
    port: 8092
    targetPort: 8092
  type: ClusterIP
EOF

    kubectl apply -f "$PHASE_DIR/semantic-analyzer-deployment.yaml"
    success "Semantic Analyzer deployed"
}

# Deploy Version Tracker
deploy_version_tracker() {
    log "Deploying Version Tracker..."
    
    cat > "$PHASE_DIR/version-tracker-deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: documentation-version-tracker
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: documentation-version-tracker
  template:
    metadata:
      labels:
        app: documentation-version-tracker
    spec:
      containers:
      - name: version-tracker
        image: python:3.11-slim
        command:
        - /bin/bash
        - -c
        - |
          pip install -r /app/requirements.txt
          python -m uvicorn main:app --host 0.0.0.0 --port 8083
        workingDir: /app
        ports:
        - containerPort: 8083
        - containerPort: 8093  # Metrics
        env:
        - name: DATABASE_URL
          value: "postgresql://\$(DB_USER):\$(DB_PASSWORD)@documentation-postgresql-service:5432/\$(DB_NAME)"
        - name: REDIS_URL
          value: "redis://:\$(REDIS_PASSWORD)@documentation-redis-service:6379/3"
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: password
        - name: DB_NAME
          valueFrom:
            secretKeyRef:
              name: documentation-database-secret
              key: database
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: documentation-redis-secret
              key: password
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: config
          mountPath: /app/config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8083
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8083
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: app-code
        configMap:
          name: version-tracker-code
      - name: config
        configMap:
          name: documentation-platform-config
---
apiVersion: v1
kind: Service
metadata:
  name: documentation-version-tracker-service
  namespace: $NAMESPACE
spec:
  selector:
    app: documentation-version-tracker
  ports:
  - name: http
    port: 8083
    targetPort: 8083
  - name: metrics
    port: 8093
    targetPort: 8093
  type: ClusterIP
EOF

    kubectl apply -f "$PHASE_DIR/version-tracker-deployment.yaml"
    success "Version Tracker deployed"
}

# Deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring components..."
    
    cat > "$PHASE_DIR/monitoring-deployment.yaml" << EOF
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
    - job_name: 'documentation-platform-manager'
      static_configs:
      - targets: ['documentation-platform-manager-service:8090']
    
    - job_name: 'documentation-processor'
      static_configs:
      - targets: ['documentation-processor-service:8091']
    
    - job_name: 'documentation-semantic-analyzer'
      static_configs:
      - targets: ['documentation-semantic-analyzer-service:8092']
    
    - job_name: 'documentation-version-tracker'
      static_configs:
      - targets: ['documentation-version-tracker-service:8093']
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: documentation-prometheus
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: documentation-prometheus
  template:
    metadata:
      labels:
        app: documentation-prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
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
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: documentation-prometheus-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: documentation-prometheus-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: documentation-prometheus-service
  namespace: $NAMESPACE
spec:
  selector:
    app: documentation-prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
EOF

    kubectl apply -f "$PHASE_DIR/monitoring-deployment.yaml"
    success "Monitoring deployed"
}

# Wait for deployments
wait_for_deployments() {
    log "Waiting for deployments to be ready..."
    
    deployments=(
        "documentation-postgresql"
        "documentation-redis"
        "documentation-platform-manager"
        "documentation-processor"
        "documentation-semantic-analyzer"
        "documentation-version-tracker"
        "documentation-prometheus"
    )
    
    for deployment in "${deployments[@]}"; do
        log "Waiting for $deployment..."
        kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n $NAMESPACE
        success "$deployment is ready"
    done
}

# Run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Wait a bit for services to stabilize
    sleep 30
    
    services=(
        "documentation-platform-manager-service:8080"
        "documentation-processor-service:8081"
        "documentation-semantic-analyzer-service:8082"
        "documentation-version-tracker-service:8083"
    )
    
    for service in "${services[@]}"; do
        log "Health checking $service..."
        if kubectl exec -n $NAMESPACE deployment/documentation-platform-manager -- curl -f "http://$service/health" &> /dev/null; then
            success "$service health check passed"
        else
            warning "$service health check failed"
        fi
    done
}

# Create ingress
create_ingress() {
    log "Creating ingress..."
    
    cat > "$PHASE_DIR/ingress.yaml" << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: documentation-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
    nginx.ingress.kubernetes.io/use-regex: "true"
spec:
  rules:
  - host: nexus-docs.local
    http:
      paths:
      - path: /platform(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: documentation-platform-manager-service
            port:
              number: 8080
      - path: /processor(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: documentation-processor-service
            port:
              number: 8081
      - path: /semantic(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: documentation-semantic-analyzer-service
            port:
              number: 8082
      - path: /versions(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: documentation-version-tracker-service
            port:
              number: 8083
      - path: /metrics(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: documentation-prometheus-service
            port:
              number: 9090
EOF

    kubectl apply -f "$PHASE_DIR/ingress.yaml"
    success "Ingress created"
}

# Print deployment summary
print_summary() {
    log "Deployment Summary"
    echo "===================="
    echo
    echo "🚀 WS3 Phase 2: Documentation Systems Integration deployed successfully!"
    echo
    echo "📊 Services deployed:"
    echo "  • Documentation Platform Manager: Port 8080"
    echo "  • Document Processor: Port 8081"
    echo "  • Semantic Analyzer: Port 8082"
    echo "  • Version Tracker: Port 8083"
    echo "  • PostgreSQL Database: Port 5432"
    echo "  • Redis Cache: Port 6379"
    echo "  • Prometheus Monitoring: Port 9090"
    echo
    echo "🔗 Access URLs (if ingress is configured):"
    echo "  • Platform Manager: http://nexus-docs.local/platform/"
    echo "  • Document Processor: http://nexus-docs.local/processor/"
    echo "  • Semantic Analyzer: http://nexus-docs.local/semantic/"
    echo "  • Version Tracker: http://nexus-docs.local/versions/"
    echo "  • Metrics: http://nexus-docs.local/metrics/"
    echo
    echo "📈 Performance Targets:"
    echo "  • Document Processing: 1,000+ docs/hour"
    echo "  • Content Extraction: >90% accuracy"
    echo "  • Real-time Updates: <60 seconds"
    echo "  • Search Relevance: >85%"
    echo "  • Cross-reference Analysis: >80%"
    echo
    echo "🔧 Next Steps:"
    echo "  1. Configure platform credentials in secrets"
    echo "  2. Test document ingestion from platforms"
    echo "  3. Verify semantic analysis and knowledge graph"
    echo "  4. Set up monitoring dashboards"
    echo "  5. Configure backup and disaster recovery"
    echo
    success "WS3 Phase 2 deployment completed successfully!"
}

# Main deployment function
main() {
    log "Starting WS3 Phase 2 deployment..."
    
    check_prerequisites
    create_namespace
    install_dependencies
    create_configmaps
    create_secrets
    deploy_postgresql
    deploy_redis
    deploy_platform_manager
    deploy_document_processor
    deploy_semantic_analyzer
    deploy_version_tracker
    deploy_monitoring
    wait_for_deployments
    run_health_checks
    create_ingress
    print_summary
}

# Run main function
main "$@"

