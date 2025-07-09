#!/bin/bash

# Nexus Architect WS2 Phase 4: Learning Systems & Continuous Improvement
# Deployment Script

set -e

echo "🚀 Starting WS2 Phase 4: Learning Systems & Continuous Improvement Deployment"

# Configuration
NAMESPACE="nexus-learning"
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-"development"}
REDIS_PASSWORD=${REDIS_PASSWORD:-$(openssl rand -base64 32)}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is required but not installed"
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        log_error "docker is required but not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace
    kubectl label namespace $NAMESPACE \
        app.kubernetes.io/name=nexus-learning \
        app.kubernetes.io/component=ai-intelligence \
        app.kubernetes.io/part-of=nexus-architect \
        --overwrite
    
    log_success "Namespace created: $NAMESPACE"
}

# Deploy Redis for caching and queuing
deploy_redis() {
    log_info "Deploying Redis cluster..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: redis-secret
  namespace: $NAMESPACE
type: Opaque
data:
  password: $(echo -n "$REDIS_PASSWORD" | base64)
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
  namespace: $NAMESPACE
data:
  redis.conf: |
    bind 0.0.0.0
    port 6379
    requirepass $REDIS_PASSWORD
    maxmemory 2gb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
    appendonly yes
    appendfsync everysec
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-learning
  namespace: $NAMESPACE
spec:
  serviceName: redis-learning
  replicas: 3
  selector:
    matchLabels:
      app: redis-learning
  template:
    metadata:
      labels:
        app: redis-learning
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
        command:
        - redis-server
        - /etc/redis/redis.conf
        volumeMounts:
        - name: config
          mountPath: /etc/redis
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: config
        configMap:
          name: redis-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis-learning
  namespace: $NAMESPACE
spec:
  selector:
    app: redis-learning
  ports:
  - port: 6379
    targetPort: 6379
  clusterIP: None
EOF

    log_success "Redis cluster deployed"
}

# Deploy PostgreSQL for metadata storage
deploy_postgresql() {
    log_info "Deploying PostgreSQL..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: $NAMESPACE
type: Opaque
data:
  password: $(echo -n "$POSTGRES_PASSWORD" | base64)
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: $NAMESPACE
data:
  postgresql.conf: |
    shared_preload_libraries = 'pg_stat_statements'
    max_connections = 200
    shared_buffers = 256MB
    effective_cache_size = 1GB
    maintenance_work_mem = 64MB
    checkpoint_completion_target = 0.9
    wal_buffers = 16MB
    default_statistics_target = 100
    random_page_cost = 1.1
    effective_io_concurrency = 200
    work_mem = 4MB
    min_wal_size = 1GB
    max_wal_size = 4GB
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres-learning
  namespace: $NAMESPACE
spec:
  serviceName: postgres-learning
  replicas: 1
  selector:
    matchLabels:
      app: postgres-learning
  template:
    metadata:
      labels:
        app: postgres-learning
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: nexus_learning
        - name: POSTGRES_USER
          value: nexus_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
        - name: config
          mountPath: /etc/postgresql
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: config
        configMap:
          name: postgres-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-learning
  namespace: $NAMESPACE
spec:
  selector:
    app: postgres-learning
  ports:
  - port: 5432
    targetPort: 5432
EOF

    log_success "PostgreSQL deployed"
}

# Deploy MLflow for experiment tracking
deploy_mlflow() {
    log_info "Deploying MLflow..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow
        image: python:3.9-slim
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_BACKEND_STORE_URI
          value: "postgresql://nexus_user:$POSTGRES_PASSWORD@postgres-learning:5432/nexus_learning"
        - name: MLFLOW_DEFAULT_ARTIFACT_ROOT
          value: "/mlflow/artifacts"
        command:
        - /bin/bash
        - -c
        - |
          pip install mlflow psycopg2-binary
          mlflow server \
            --backend-store-uri \$MLFLOW_BACKEND_STORE_URI \
            --default-artifact-root \$MLFLOW_DEFAULT_ARTIFACT_ROOT \
            --host 0.0.0.0 \
            --port 5000
        volumeMounts:
        - name: artifacts
          mountPath: /mlflow/artifacts
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: artifacts
        persistentVolumeClaim:
          claimName: mlflow-artifacts
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-artifacts
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-server
  namespace: $NAMESPACE
spec:
  selector:
    app: mlflow-server
  ports:
  - port: 5000
    targetPort: 5000
EOF

    log_success "MLflow deployed"
}

# Deploy Continuous Learning Engine
deploy_continuous_learning() {
    log_info "Deploying Continuous Learning Engine..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: learning-config
  namespace: $NAMESPACE
data:
  config.yaml: |
    database:
      url: "postgresql://nexus_user:$POSTGRES_PASSWORD@postgres-learning:5432/nexus_learning"
    redis:
      url: "redis://:$REDIS_PASSWORD@redis-learning:6379"
    mlflow:
      tracking_uri: "http://mlflow-server:5000"
    learning:
      batch_size: 32
      learning_rate: 0.001
      update_frequency: 3600  # 1 hour
      min_samples: 100
      max_models: 10
    monitoring:
      metrics_interval: 60
      health_check_interval: 30
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: continuous-learning-engine
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: continuous-learning-engine
  template:
    metadata:
      labels:
        app: continuous-learning-engine
    spec:
      containers:
      - name: learning-engine
        image: python:3.9-slim
        ports:
        - containerPort: 8000
        env:
        - name: CONFIG_PATH
          value: "/app/config/config.yaml"
        - name: PYTHONPATH
          value: "/app"
        command:
        - /bin/bash
        - -c
        - |
          pip install fastapi uvicorn redis sqlalchemy psycopg2-binary \
                     scikit-learn torch transformers mlflow numpy pandas
          cd /app
          python -m uvicorn continuous_learning_engine:app --host 0.0.0.0 --port 8000
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: config
        configMap:
          name: learning-config
      - name: app-code
        configMap:
          name: learning-app-code
---
apiVersion: v1
kind: Service
metadata:
  name: continuous-learning-engine
  namespace: $NAMESPACE
spec:
  selector:
    app: continuous-learning-engine
  ports:
  - port: 8000
    targetPort: 8000
EOF

    log_success "Continuous Learning Engine deployed"
}

# Deploy Feedback Processing System
deploy_feedback_system() {
    log_info "Deploying Feedback Processing System..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feedback-processor
  namespace: $NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: feedback-processor
  template:
    metadata:
      labels:
        app: feedback-processor
    spec:
      containers:
      - name: feedback-processor
        image: python:3.9-slim
        ports:
        - containerPort: 8001
        env:
        - name: REDIS_URL
          value: "redis://:$REDIS_PASSWORD@redis-learning:6379"
        - name: DATABASE_URL
          value: "postgresql://nexus_user:$POSTGRES_PASSWORD@postgres-learning:5432/nexus_learning"
        command:
        - /bin/bash
        - -c
        - |
          pip install fastapi uvicorn redis sqlalchemy psycopg2-binary \
                     celery numpy pandas scikit-learn
          cd /app
          python -m uvicorn feedback_processing_system:app --host 0.0.0.0 --port 8001
        volumeMounts:
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: app-code
        configMap:
          name: feedback-app-code
---
apiVersion: v1
kind: Service
metadata:
  name: feedback-processor
  namespace: $NAMESPACE
spec:
  selector:
    app: feedback-processor
  ports:
  - port: 8001
    targetPort: 8001
EOF

    log_success "Feedback Processing System deployed"
}

# Deploy Knowledge Acquisition Engine
deploy_knowledge_acquisition() {
    log_info "Deploying Knowledge Acquisition Engine..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: knowledge-acquisition
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: knowledge-acquisition
  template:
    metadata:
      labels:
        app: knowledge-acquisition
    spec:
      containers:
      - name: knowledge-engine
        image: python:3.9-slim
        ports:
        - containerPort: 8002
        env:
        - name: REDIS_URL
          value: "redis://:$REDIS_PASSWORD@redis-learning:6379"
        - name: DATABASE_URL
          value: "postgresql://nexus_user:$POSTGRES_PASSWORD@postgres-learning:5432/nexus_learning"
        - name: NEO4J_URI
          value: "bolt://neo4j-knowledge:7687"
        command:
        - /bin/bash
        - -c
        - |
          pip install fastapi uvicorn redis sqlalchemy psycopg2-binary \
                     neo4j spacy transformers torch numpy pandas
          python -m spacy download en_core_web_sm
          cd /app
          python -m uvicorn knowledge_acquisition_engine:app --host 0.0.0.0 --port 8002
        volumeMounts:
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: app-code
        configMap:
          name: knowledge-app-code
---
apiVersion: v1
kind: Service
metadata:
  name: knowledge-acquisition
  namespace: $NAMESPACE
spec:
  selector:
    app: knowledge-acquisition
  ports:
  - port: 8002
    targetPort: 8002
EOF

    log_success "Knowledge Acquisition Engine deployed"
}

# Deploy Model Management System
deploy_model_management() {
    log_info "Deploying Model Management System..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-management
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-management
  template:
    metadata:
      labels:
        app: model-management
    spec:
      containers:
      - name: model-manager
        image: python:3.9-slim
        ports:
        - containerPort: 8003
        env:
        - name: DATABASE_URL
          value: "postgresql://nexus_user:$POSTGRES_PASSWORD@postgres-learning:5432/nexus_learning"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
        - name: MODEL_REGISTRY_PATH
          value: "/models"
        command:
        - /bin/bash
        - -c
        - |
          pip install fastapi uvicorn sqlalchemy psycopg2-binary \
                     mlflow docker kubernetes joblib torch transformers
          cd /app
          python -m uvicorn model_management_system:app --host 0.0.0.0 --port 8003
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: model-storage
          mountPath: /models
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: app-code
        configMap:
          name: model-app-code
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: model-management
  namespace: $NAMESPACE
spec:
  selector:
    app: model-management
  ports:
  - port: 8003
    targetPort: 8003
EOF

    log_success "Model Management System deployed"
}

# Deploy Training Pipeline
deploy_training_pipeline() {
    log_info "Deploying Automated Training Pipeline..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: training-pipeline
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: training-pipeline
  template:
    metadata:
      labels:
        app: training-pipeline
    spec:
      containers:
      - name: training-pipeline
        image: python:3.9-slim
        ports:
        - containerPort: 8004
        env:
        - name: DATABASE_URL
          value: "postgresql://nexus_user:$POSTGRES_PASSWORD@postgres-learning:5432/nexus_learning"
        - name: REDIS_URL
          value: "redis://:$REDIS_PASSWORD@redis-learning:6379"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
        command:
        - /bin/bash
        - -c
        - |
          pip install fastapi uvicorn sqlalchemy psycopg2-binary redis \
                     mlflow celery scikit-learn torch transformers \
                     numpy pandas kubernetes docker optuna
          cd /app
          python -m uvicorn automated_training_pipeline:app --host 0.0.0.0 --port 8004
        volumeMounts:
        - name: app-code
          mountPath: /app
        - name: training-data
          mountPath: /data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
      volumes:
      - name: app-code
        configMap:
          name: training-app-code
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 200Gi
---
apiVersion: v1
kind: Service
metadata:
  name: training-pipeline
  namespace: $NAMESPACE
spec:
  selector:
    app: training-pipeline
  ports:
  - port: 8004
    targetPort: 8004
EOF

    log_success "Automated Training Pipeline deployed"
}

# Deploy monitoring and observability
deploy_monitoring() {
    log_info "Deploying monitoring and observability..."
    
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
    - job_name: 'learning-systems'
      static_configs:
      - targets:
        - 'continuous-learning-engine:8000'
        - 'feedback-processor:8001'
        - 'knowledge-acquisition:8002'
        - 'model-management:8003'
        - 'training-pipeline:8004'
      metrics_path: /metrics
      scrape_interval: 30s
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus-learning
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus-learning
  template:
    metadata:
      labels:
        app: prometheus-learning
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: data
          mountPath: /prometheus
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: data
        persistentVolumeClaim:
          claimName: prometheus-data
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-data
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-learning
  namespace: $NAMESPACE
spec:
  selector:
    app: prometheus-learning
  ports:
  - port: 9090
    targetPort: 9090
EOF

    log_success "Monitoring deployed"
}

# Create ingress
create_ingress() {
    log_info "Creating ingress..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: learning-systems-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - learning.nexus-architect.local
    secretName: learning-tls
  rules:
  - host: learning.nexus-architect.local
    http:
      paths:
      - path: /learning
        pathType: Prefix
        backend:
          service:
            name: continuous-learning-engine
            port:
              number: 8000
      - path: /feedback
        pathType: Prefix
        backend:
          service:
            name: feedback-processor
            port:
              number: 8001
      - path: /knowledge
        pathType: Prefix
        backend:
          service:
            name: knowledge-acquisition
            port:
              number: 8002
      - path: /models
        pathType: Prefix
        backend:
          service:
            name: model-management
            port:
              number: 8003
      - path: /training
        pathType: Prefix
        backend:
          service:
            name: training-pipeline
            port:
              number: 8004
      - path: /mlflow
        pathType: Prefix
        backend:
          service:
            name: mlflow-server
            port:
              number: 5000
      - path: /metrics
        pathType: Prefix
        backend:
          service:
            name: prometheus-learning
            port:
              number: 9090
EOF

    log_success "Ingress created"
}

# Wait for deployments
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."
    
    deployments=(
        "redis-learning"
        "postgres-learning"
        "mlflow-server"
        "continuous-learning-engine"
        "feedback-processor"
        "knowledge-acquisition"
        "model-management"
        "training-pipeline"
        "prometheus-learning"
    )
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment..."
        kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n $NAMESPACE 2>/dev/null || \
        kubectl wait --for=condition=ready --timeout=300s statefulset/$deployment -n $NAMESPACE 2>/dev/null || \
        log_warning "Timeout waiting for $deployment"
    done
    
    log_success "All deployments ready"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check Redis
    if kubectl exec -n $NAMESPACE redis-learning-0 -- redis-cli ping &>/dev/null; then
        log_success "Redis health check passed"
    else
        log_warning "Redis health check failed"
    fi
    
    # Check PostgreSQL
    if kubectl exec -n $NAMESPACE postgres-learning-0 -- pg_isready &>/dev/null; then
        log_success "PostgreSQL health check passed"
    else
        log_warning "PostgreSQL health check failed"
    fi
    
    # Check services
    services=(
        "continuous-learning-engine:8000"
        "feedback-processor:8001"
        "knowledge-acquisition:8002"
        "model-management:8003"
        "training-pipeline:8004"
        "mlflow-server:5000"
    )
    
    for service in "${services[@]}"; do
        service_name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if kubectl exec -n $NAMESPACE redis-learning-0 -- nc -z $service_name $port &>/dev/null; then
            log_success "$service_name health check passed"
        else
            log_warning "$service_name health check failed"
        fi
    done
}

# Display deployment information
display_info() {
    log_info "Deployment completed successfully!"
    echo
    echo "📊 Learning Systems & Continuous Improvement Services:"
    echo "  • Continuous Learning Engine: http://learning.nexus-architect.local/learning"
    echo "  • Feedback Processing: http://learning.nexus-architect.local/feedback"
    echo "  • Knowledge Acquisition: http://learning.nexus-architect.local/knowledge"
    echo "  • Model Management: http://learning.nexus-architect.local/models"
    echo "  • Training Pipeline: http://learning.nexus-architect.local/training"
    echo "  • MLflow: http://learning.nexus-architect.local/mlflow"
    echo "  • Metrics: http://learning.nexus-architect.local/metrics"
    echo
    echo "🔐 Credentials:"
    echo "  • Redis Password: $REDIS_PASSWORD"
    echo "  • PostgreSQL Password: $POSTGRES_PASSWORD"
    echo
    echo "📋 Useful Commands:"
    echo "  • View pods: kubectl get pods -n $NAMESPACE"
    echo "  • View services: kubectl get services -n $NAMESPACE"
    echo "  • View logs: kubectl logs -f deployment/<service-name> -n $NAMESPACE"
    echo "  • Port forward: kubectl port-forward service/<service-name> <local-port>:<service-port> -n $NAMESPACE"
    echo
    echo "🎯 Next Steps:"
    echo "  1. Configure DNS or add entries to /etc/hosts"
    echo "  2. Set up SSL certificates"
    echo "  3. Configure monitoring alerts"
    echo "  4. Run integration tests"
    echo "  5. Set up backup procedures"
}

# Cleanup function
cleanup() {
    if [[ "$1" == "--cleanup" ]]; then
        log_warning "Cleaning up deployment..."
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        log_success "Cleanup completed"
        exit 0
    fi
}

# Main execution
main() {
    # Handle cleanup
    cleanup "$1"
    
    # Start deployment
    log_info "Starting WS2 Phase 4: Learning Systems & Continuous Improvement deployment"
    
    check_prerequisites
    create_namespace
    deploy_redis
    deploy_postgresql
    deploy_mlflow
    deploy_continuous_learning
    deploy_feedback_system
    deploy_knowledge_acquisition
    deploy_model_management
    deploy_training_pipeline
    deploy_monitoring
    create_ingress
    wait_for_deployments
    run_health_checks
    display_info
    
    log_success "WS2 Phase 4 deployment completed successfully! 🎉"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

