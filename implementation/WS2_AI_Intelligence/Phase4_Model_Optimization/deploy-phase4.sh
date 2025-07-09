#!/bin/bash

# Nexus Architect WS2 Phase 4: AI Model Fine-tuning & Optimization
# Deployment Script for Model Optimization Infrastructure

set -euo pipefail

# Configuration
NAMESPACE="nexus-ai-intelligence"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/nexus-ws2-phase4-deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        error "helm is not installed or not in PATH"
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    success "Prerequisites check completed"
}

# Deploy fine-tuning infrastructure
deploy_fine_tuning_infrastructure() {
    log "Deploying fine-tuning infrastructure..."
    
    # Create ConfigMap for fine-tuning configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: fine-tuning-config
  namespace: $NAMESPACE
data:
  config.yaml: |
    fine_tuning:
      batch_size: 16
      learning_rate: 5e-5
      num_epochs: 3
      warmup_steps: 500
      max_seq_length: 512
      gradient_accumulation_steps: 4
      fp16: true
      dataloader_num_workers: 4
      save_steps: 1000
      eval_steps: 500
      logging_steps: 100
      
    models:
      base_models:
        - name: "gpt-3.5-turbo"
          type: "language_model"
          provider: "openai"
        - name: "claude-3-sonnet"
          type: "language_model"
          provider: "anthropic"
        - name: "llama-2-7b"
          type: "language_model"
          provider: "huggingface"
      
      specialized_models:
        - name: "nexus-security-architect"
          base_model: "llama-2-7b"
          domain: "security"
          fine_tuning_data: "/data/security_training.json"
        - name: "nexus-performance-engineer"
          base_model: "llama-2-7b"
          domain: "performance"
          fine_tuning_data: "/data/performance_training.json"
        - name: "nexus-application-architect"
          base_model: "llama-2-7b"
          domain: "application"
          fine_tuning_data: "/data/application_training.json"
    
    optimization:
      quantization:
        enabled: true
        method: "dynamic"
        dtype: "int8"
      
      pruning:
        enabled: true
        sparsity: 0.1
        structured: false
      
      distillation:
        enabled: true
        teacher_model: "gpt-4"
        temperature: 3.0
        alpha: 0.7
EOF
    
    # Deploy fine-tuning job template
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: model-fine-tuning-job
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: fine-tuning
        image: nexus-architect/model-fine-tuning:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: "2"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1"
        - name: PYTHONPATH
          value: "/app"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: data-storage
          mountPath: /data
        - name: config-volume
          mountPath: /config
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: data-storage
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: config-volume
        configMap:
          name: fine-tuning-config
EOF
    
    success "Fine-tuning infrastructure deployed"
}

# Deploy model serving infrastructure
deploy_model_serving() {
    log "Deploying model serving infrastructure..."
    
    # Deploy TorchServe for model serving
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchserve-deployment
  namespace: $NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: torchserve
  template:
    metadata:
      labels:
        app: torchserve
    spec:
      containers:
      - name: torchserve
        image: pytorch/torchserve:latest-gpu
        ports:
        - containerPort: 8080
          name: inference
        - containerPort: 8081
          name: management
        - containerPort: 8082
          name: metrics
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        env:
        - name: TS_CONFIG_FILE
          value: "/config/config.properties"
        volumeMounts:
        - name: model-store
          mountPath: /home/model-server/model-store
        - name: torchserve-config
          mountPath: /config
        livenessProbe:
          httpGet:
            path: /ping
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ping
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 5
      volumes:
      - name: model-store
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: torchserve-config
        configMap:
          name: torchserve-config
---
apiVersion: v1
kind: Service
metadata:
  name: torchserve-service
  namespace: $NAMESPACE
spec:
  selector:
    app: torchserve
  ports:
  - name: inference
    port: 8080
    targetPort: 8080
  - name: management
    port: 8081
    targetPort: 8081
  - name: metrics
    port: 8082
    targetPort: 8082
  type: ClusterIP
EOF
    
    # Create TorchServe configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: torchserve-config
  namespace: $NAMESPACE
data:
  config.properties: |
    inference_address=http://0.0.0.0:8080
    management_address=http://0.0.0.0:8081
    metrics_address=http://0.0.0.0:8082
    grpc_inference_port=7070
    grpc_management_port=7071
    enable_envvars_config=true
    install_py_dep_per_model=true
    enable_metrics_api=true
    metrics_format=prometheus
    number_of_netty_threads=4
    job_queue_size=10
    number_of_gpu=1
    batch_size=1
    max_batch_delay=5000
    response_timeout=120
    unregister_model_timeout=120
    decode_input_request=true
    prefer_direct_buffer=true
    allowed_urls=file://,http://,https://
EOF
    
    success "Model serving infrastructure deployed"
}

# Deploy evaluation and monitoring system
deploy_evaluation_monitoring() {
    log "Deploying evaluation and monitoring system..."
    
    # Deploy model evaluation service
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-evaluation-service
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-evaluation
  template:
    metadata:
      labels:
        app: model-evaluation
    spec:
      containers:
      - name: evaluation-service
        image: nexus-architect/model-evaluation:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: NEO4J_URI
          value: "bolt://neo4j-lb.nexus-knowledge-graph:7687"
        - name: NEO4J_USER
          value: "neo4j"
        - name: NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neo4j-credentials
              key: password
        - name: REDIS_HOST
          value: "redis-lb.nexus-core-foundation"
        - name: REDIS_PORT
          value: "6379"
        - name: ELASTICSEARCH_HOST
          value: "elasticsearch-lb.nexus-monitoring"
        - name: ELASTICSEARCH_PORT
          value: "9200"
        volumeMounts:
        - name: evaluation-data
          mountPath: /data
        - name: model-storage
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 5
      volumes:
      - name: evaluation-data
        persistentVolumeClaim:
          claimName: evaluation-data-pvc
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: model-evaluation-service
  namespace: $NAMESPACE
spec:
  selector:
    app: model-evaluation
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
EOF
    
    success "Evaluation and monitoring system deployed"
}

# Deploy optimization frameworks
deploy_optimization_frameworks() {
    log "Deploying optimization frameworks..."
    
    # Deploy model optimization service
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-optimization-service
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-optimization
  template:
    metadata:
      labels:
        app: model-optimization
    spec:
      containers:
      - name: optimization-service
        image: nexus-architect/model-optimization:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: OPTIMIZATION_CONFIG
          value: "/config/optimization.yaml"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: optimization-config
          mountPath: /config
        - name: temp-storage
          mountPath: /tmp
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: optimization-config
        configMap:
          name: optimization-config
      - name: temp-storage
        emptyDir:
          sizeLimit: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: model-optimization-service
  namespace: $NAMESPACE
spec:
  selector:
    app: model-optimization
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
EOF
    
    # Create optimization configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: optimization-config
  namespace: $NAMESPACE
data:
  optimization.yaml: |
    quantization:
      methods:
        - name: "dynamic_quantization"
          dtype: "int8"
          backend: "fbgemm"
        - name: "static_quantization"
          dtype: "int8"
          backend: "qnnpack"
        - name: "qat"
          dtype: "int8"
          backend: "fbgemm"
    
    pruning:
      methods:
        - name: "magnitude_pruning"
          sparsity: [0.1, 0.3, 0.5, 0.7]
          structured: false
        - name: "structured_pruning"
          sparsity: [0.1, 0.2, 0.3]
          structured: true
    
    distillation:
      teacher_models:
        - "gpt-4"
        - "claude-3-opus"
      temperature: [2.0, 3.0, 4.0, 5.0]
      alpha: [0.5, 0.7, 0.9]
    
    hyperparameter_optimization:
      method: "optuna"
      n_trials: 100
      parameters:
        learning_rate:
          type: "loguniform"
          low: 1e-6
          high: 1e-3
        batch_size:
          type: "categorical"
          choices: [8, 16, 32, 64]
        warmup_steps:
          type: "int"
          low: 100
          high: 1000
EOF
    
    success "Optimization frameworks deployed"
}

# Create persistent volumes
create_persistent_volumes() {
    log "Creating persistent volumes..."
    
    # Model storage PVC
    kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: fast-ssd
EOF
    
    # Training data PVC
    kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 200Gi
  storageClassName: fast-ssd
EOF
    
    # Evaluation data PVC
    kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: evaluation-data-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: standard
EOF
    
    success "Persistent volumes created"
}

# Setup monitoring and alerting
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Deploy ServiceMonitor for Prometheus
    kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: model-optimization-metrics
  namespace: $NAMESPACE
  labels:
    app: model-optimization
spec:
  selector:
    matchLabels:
      app: model-optimization
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: torchserve-metrics
  namespace: $NAMESPACE
  labels:
    app: torchserve
spec:
  selector:
    matchLabels:
      app: torchserve
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF
    
    # Create PrometheusRule for alerting
    kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: model-optimization-alerts
  namespace: $NAMESPACE
spec:
  groups:
  - name: model_optimization
    rules:
    - alert: ModelInferenceLatencyHigh
      expr: histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m])) > 2
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High model inference latency"
        description: "Model {{ \$labels.model_id }} has high inference latency: {{ \$value }}s"
    
    - alert: ModelAccuracyLow
      expr: model_accuracy_score < 0.8
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Low model accuracy"
        description: "Model {{ \$labels.model_id }} accuracy is below threshold: {{ \$value }}"
    
    - alert: ModelThroughputLow
      expr: model_throughput_requests_per_second < 10
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Low model throughput"
        description: "Model {{ \$labels.model_id }} throughput is low: {{ \$value }} req/s"
    
    - alert: ModelMemoryUsageHigh
      expr: model_memory_usage_bytes / (1024^3) > 8
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High model memory usage"
        description: "Model {{ \$labels.model_id }} memory usage is high: {{ \$value }}GB"
EOF
    
    success "Monitoring and alerting configured"
}

# Deploy ingress and networking
deploy_networking() {
    log "Deploying networking and ingress..."
    
    # Create Ingress for model services
    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-optimization-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  tls:
  - hosts:
    - model-optimization.nexus-architect.local
    secretName: nexus-tls-secret
  rules:
  - host: model-optimization.nexus-architect.local
    http:
      paths:
      - path: /inference
        pathType: Prefix
        backend:
          service:
            name: torchserve-service
            port:
              number: 8080
      - path: /management
        pathType: Prefix
        backend:
          service:
            name: torchserve-service
            port:
              number: 8081
      - path: /evaluation
        pathType: Prefix
        backend:
          service:
            name: model-evaluation-service
            port:
              number: 8000
      - path: /optimization
        pathType: Prefix
        backend:
          service:
            name: model-optimization-service
            port:
              number: 8000
EOF
    
    success "Networking and ingress deployed"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check if all pods are running
    log "Checking pod status..."
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Check if all services are available
    log "Checking service status..."
    kubectl get services -n "$NAMESPACE"
    
    # Check if PVCs are bound
    log "Checking PVC status..."
    kubectl get pvc -n "$NAMESPACE"
    
    # Wait for pods to be ready
    log "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=torchserve -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=model-evaluation -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=model-optimization -n "$NAMESPACE" --timeout=300s
    
    # Test service endpoints
    log "Testing service endpoints..."
    
    # Test TorchServe
    if kubectl exec -n "$NAMESPACE" deployment/torchserve-deployment -- curl -f http://localhost:8080/ping; then
        success "TorchServe is responding"
    else
        warning "TorchServe health check failed"
    fi
    
    # Test evaluation service
    if kubectl exec -n "$NAMESPACE" deployment/model-evaluation-service -- curl -f http://localhost:8000/health; then
        success "Model evaluation service is responding"
    else
        warning "Model evaluation service health check failed"
    fi
    
    success "Deployment verification completed"
}

# Main deployment function
main() {
    log "Starting Nexus Architect WS2 Phase 4 deployment..."
    log "Deployment log: $LOG_FILE"
    
    check_prerequisites
    create_persistent_volumes
    deploy_fine_tuning_infrastructure
    deploy_model_serving
    deploy_evaluation_monitoring
    deploy_optimization_frameworks
    setup_monitoring
    deploy_networking
    verify_deployment
    
    success "WS2 Phase 4 deployment completed successfully!"
    log "Model optimization infrastructure is now operational"
    log "Access points:"
    log "  - Model Inference: https://model-optimization.nexus-architect.local/inference"
    log "  - Model Management: https://model-optimization.nexus-architect.local/management"
    log "  - Model Evaluation: https://model-optimization.nexus-architect.local/evaluation"
    log "  - Model Optimization: https://model-optimization.nexus-architect.local/optimization"
}

# Run main function
main "$@"

