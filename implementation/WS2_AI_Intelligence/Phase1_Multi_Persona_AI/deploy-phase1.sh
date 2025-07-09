#!/bin/bash

# Nexus Architect WS2 Phase 1 Deployment Script
# Multi-Persona AI Foundation Deployment

set -e

echo "🚀 Starting WS2 Phase 1: Multi-Persona AI Foundation Deployment"
echo "================================================================"

# Configuration
NAMESPACE="nexus-ai-intelligence"
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-"development"}
KUBECTL_TIMEOUT="300s"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if WS1 Core Foundation is deployed
    if ! kubectl get namespace nexus-infrastructure &> /dev/null; then
        error "WS1 Core Foundation not found. Please deploy WS1 first."
        exit 1
    fi
    
    # Check if required services are running
    local required_services=("postgresql" "redis" "vault" "weaviate")
    for service in "${required_services[@]}"; do
        if ! kubectl get service "$service" -n nexus-infrastructure &> /dev/null; then
            error "Required service '$service' not found in nexus-infrastructure namespace"
            exit 1
        fi
    done
    
    success "Prerequisites check passed"
}

# Create namespace and RBAC
setup_namespace() {
    log "Setting up namespace and RBAC..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace
    kubectl label namespace $NAMESPACE component=ai-intelligence --overwrite
    kubectl label namespace $NAMESPACE workstream=ws2 --overwrite
    
    # Create service account
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: nexus-ai-intelligence
  namespace: $NAMESPACE
  labels:
    app: nexus-ai-intelligence
    component: ai-intelligence
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: nexus-ai-intelligence
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: nexus-ai-intelligence
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: nexus-ai-intelligence
subjects:
- kind: ServiceAccount
  name: nexus-ai-intelligence
  namespace: $NAMESPACE
EOF
    
    success "Namespace and RBAC configured"
}

# Deploy persona definitions
deploy_persona_definitions() {
    log "Deploying persona definitions..."
    
    kubectl apply -f personas/persona_definitions.yaml
    
    # Wait for ConfigMap to be created
    kubectl wait --for=condition=Ready configmap/persona-definitions -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    success "Persona definitions deployed"
}

# Deploy model serving infrastructure
deploy_model_serving() {
    log "Deploying model serving infrastructure..."
    
    # Apply model serving configuration
    kubectl apply -f models/model_serving_infrastructure.yaml
    
    # Wait for PVCs to be bound
    log "Waiting for persistent volumes to be bound..."
    kubectl wait --for=condition=Bound pvc/model-store-pvc -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    kubectl wait --for=condition=Bound pvc/training-datasets-pvc -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    # Wait for TorchServe deployment to be ready
    log "Waiting for TorchServe deployment to be ready..."
    kubectl wait --for=condition=Available deployment/torchserve-multi-persona -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    # Check if HPA is working
    kubectl get hpa torchserve-multi-persona-hpa -n $NAMESPACE
    
    success "Model serving infrastructure deployed"
}

# Deploy orchestration services
deploy_orchestration() {
    log "Deploying orchestration services..."
    
    # Create ConfigMap for orchestration code
    kubectl create configmap persona-orchestrator-code \
        --from-file=orchestration/persona_orchestrator.py \
        --from-file=orchestration/collaboration_framework.py \
        -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy orchestration service
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: persona-orchestrator
  namespace: $NAMESPACE
  labels:
    app: persona-orchestrator
    component: orchestration
spec:
  replicas: 2
  selector:
    matchLabels:
      app: persona-orchestrator
  template:
    metadata:
      labels:
        app: persona-orchestrator
        component: orchestration
    spec:
      serviceAccountName: nexus-ai-intelligence
      containers:
      - name: orchestrator
        image: python:3.11-slim
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP
        env:
        - name: REDIS_URL
          value: "redis://redis.nexus-infrastructure:6379"
        - name: TORCHSERVE_URL
          value: "http://torchserve-multi-persona-service.nexus-ai-intelligence:8080"
        - name: WEAVIATE_URL
          value: "http://weaviate.nexus-infrastructure:8080"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-keys
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-keys
              key: anthropic-api-key
        volumeMounts:
        - name: orchestrator-code
          mountPath: /app
        - name: config
          mountPath: /app/config
        command:
        - /bin/bash
        - -c
        - |
          cd /app
          pip install fastapi uvicorn redis openai anthropic scikit-learn networkx numpy pyyaml httpx
          python persona_orchestrator.py
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: orchestrator-code
        configMap:
          name: persona-orchestrator-code
      - name: config
        configMap:
          name: persona-definitions
---
apiVersion: v1
kind: Service
metadata:
  name: persona-orchestrator-service
  namespace: $NAMESPACE
  labels:
    app: persona-orchestrator
    component: orchestration
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  selector:
    app: persona-orchestrator
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: persona-orchestrator-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: persona-orchestrator
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
    
    # Wait for orchestrator deployment
    kubectl wait --for=condition=Available deployment/persona-orchestrator -n $NAMESPACE --timeout=$KUBECTL_TIMEOUT
    
    success "Orchestration services deployed"
}

# Setup training data generation
setup_training_data() {
    log "Setting up training data generation..."
    
    # Create job for training data generation
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: training-data-generator
  namespace: $NAMESPACE
  labels:
    app: training-data-generator
    component: training
spec:
  template:
    metadata:
      labels:
        app: training-data-generator
        component: training
    spec:
      serviceAccountName: nexus-ai-intelligence
      restartPolicy: OnFailure
      containers:
      - name: generator
        image: python:3.11-slim
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-keys
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-api-keys
              key: anthropic-api-key
        volumeMounts:
        - name: generator-code
          mountPath: /app
        - name: datasets
          mountPath: /opt/ml/datasets
        command:
        - /bin/bash
        - -c
        - |
          cd /app
          pip install openai anthropic pyyaml
          python training_data_generator.py
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: generator-code
        configMap:
          name: training-data-generator-code
      - name: datasets
        persistentVolumeClaim:
          claimName: training-datasets-pvc
EOF
    
    # Create ConfigMap for training data generator
    kubectl create configmap training-data-generator-code \
        --from-file=knowledge-bases/training_data_generator.py \
        -n $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    success "Training data generation job created"
}

# Create API secrets
create_secrets() {
    log "Creating API secrets..."
    
    # Check if secrets already exist
    if kubectl get secret ai-api-keys -n $NAMESPACE &> /dev/null; then
        warning "API keys secret already exists, skipping creation"
        return
    fi
    
    # Prompt for API keys if not provided via environment
    if [[ -z "$OPENAI_API_KEY" ]]; then
        echo -n "Enter OpenAI API Key: "
        read -s OPENAI_API_KEY
        echo
    fi
    
    if [[ -z "$ANTHROPIC_API_KEY" ]]; then
        echo -n "Enter Anthropic API Key: "
        read -s ANTHROPIC_API_KEY
        echo
    fi
    
    # Create secret
    kubectl create secret generic ai-api-keys \
        --from-literal=openai-api-key="$OPENAI_API_KEY" \
        --from-literal=anthropic-api-key="$ANTHROPIC_API_KEY" \
        -n $NAMESPACE
    
    success "API secrets created"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring for AI services..."
    
    # Create ServiceMonitor for Prometheus
    cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: persona-orchestrator-metrics
  namespace: $NAMESPACE
  labels:
    app: persona-orchestrator
    component: monitoring
spec:
  selector:
    matchLabels:
      app: persona-orchestrator
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
    app: torchserve-multi-persona
    component: monitoring
spec:
  selector:
    matchLabels:
      app: torchserve-multi-persona
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF
    
    success "Monitoring configured"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."
    
    # Check all pods are running
    local pods_ready=true
    
    # Check TorchServe pods
    if ! kubectl get pods -l app=torchserve-multi-persona -n $NAMESPACE | grep -q "Running"; then
        error "TorchServe pods are not running"
        pods_ready=false
    fi
    
    # Check orchestrator pods
    if ! kubectl get pods -l app=persona-orchestrator -n $NAMESPACE | grep -q "Running"; then
        error "Persona orchestrator pods are not running"
        pods_ready=false
    fi
    
    if [[ "$pods_ready" == "false" ]]; then
        error "Some pods are not ready. Check with: kubectl get pods -n $NAMESPACE"
        exit 1
    fi
    
    # Test orchestrator health endpoint
    log "Testing orchestrator health endpoint..."
    local orchestrator_pod=$(kubectl get pods -l app=persona-orchestrator -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n $NAMESPACE "$orchestrator_pod" -- curl -f http://localhost:8080/health &> /dev/null; then
        success "Orchestrator health check passed"
    else
        warning "Orchestrator health check failed, but deployment may still be starting"
    fi
    
    # Display deployment status
    echo
    log "Deployment Status:"
    kubectl get all -n $NAMESPACE
    
    echo
    log "Persistent Volumes:"
    kubectl get pvc -n $NAMESPACE
    
    echo
    log "ConfigMaps and Secrets:"
    kubectl get configmaps,secrets -n $NAMESPACE
    
    success "Deployment validation completed"
}

# Display access information
display_access_info() {
    log "Access Information:"
    echo
    echo "🔗 Service Endpoints:"
    echo "   Persona Orchestrator: http://persona-orchestrator-service.nexus-ai-intelligence:8080"
    echo "   TorchServe Management: http://torchserve-multi-persona-service.nexus-ai-intelligence:8081"
    echo "   TorchServe Inference: http://torchserve-multi-persona-service.nexus-ai-intelligence:8080"
    echo
    echo "📊 Monitoring:"
    echo "   Prometheus metrics available on port 9090"
    echo "   Grafana dashboards can be configured for AI service monitoring"
    echo
    echo "🧠 Available Personas:"
    echo "   - Security Architect (security_architect)"
    echo "   - Performance Engineer (performance_engineer)"
    echo "   - Application Architect (application_architect)"
    echo "   - DevOps Specialist (devops_specialist)"
    echo "   - Compliance Auditor (compliance_auditor)"
    echo
    echo "📚 API Documentation:"
    echo "   OpenAPI docs: http://persona-orchestrator-service.nexus-ai-intelligence:8080/docs"
    echo
    echo "🔧 Management Commands:"
    echo "   kubectl get pods -n $NAMESPACE"
    echo "   kubectl logs -f deployment/persona-orchestrator -n $NAMESPACE"
    echo "   kubectl logs -f deployment/torchserve-multi-persona -n $NAMESPACE"
}

# Cleanup function
cleanup() {
    if [[ "$1" == "true" ]]; then
        log "Cleaning up failed deployment..."
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        error "Deployment failed and was cleaned up"
        exit 1
    fi
}

# Main deployment function
main() {
    local start_time=$(date +%s)
    
    echo "🚀 Nexus Architect WS2 Phase 1: Multi-Persona AI Foundation"
    echo "============================================================"
    echo "Deployment Environment: $DEPLOYMENT_ENV"
    echo "Target Namespace: $NAMESPACE"
    echo "Timestamp: $(date)"
    echo
    
    # Set trap for cleanup on failure
    trap 'cleanup true' ERR
    
    # Execute deployment steps
    check_prerequisites
    setup_namespace
    create_secrets
    deploy_persona_definitions
    deploy_model_serving
    deploy_orchestration
    setup_training_data
    setup_monitoring
    validate_deployment
    
    # Calculate deployment time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo
    success "🎉 WS2 Phase 1 deployment completed successfully!"
    echo "⏱️  Total deployment time: ${duration} seconds"
    echo
    
    display_access_info
    
    echo
    echo "🎯 Next Steps:"
    echo "1. Monitor training data generation job completion"
    echo "2. Test persona interactions via API endpoints"
    echo "3. Configure monitoring dashboards"
    echo "4. Proceed with WS2 Phase 2 when ready"
    echo
    echo "📖 For troubleshooting, check the documentation in docs/README.md"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

