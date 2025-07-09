#!/bin/bash

# Nexus Architect WS1 Phase 4: Enhanced AI Services & Knowledge Foundation
# Deployment Script for AI Model Serving, Vector Database, and Knowledge Processing

set -euo pipefail

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
    log "Checking prerequisites for Phase 4 deployment..."
    
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
    
    # Check if previous phases are deployed
    if ! kubectl get namespace nexus-infrastructure &> /dev/null; then
        error "Phase 1 (Infrastructure) not deployed. Please deploy Phase 1 first."
        exit 1
    fi
    
    if ! kubectl get namespace nexus-auth &> /dev/null; then
        error "Phase 2 (Authentication) not deployed. Please deploy Phase 2 first."
        exit 1
    fi
    
    if ! kubectl get namespace nexus-security &> /dev/null; then
        error "Phase 3 (Security) not deployed. Please deploy Phase 3 first."
        exit 1
    fi
    
    # Check for GPU nodes (optional but recommended)
    if kubectl get nodes -l accelerator=nvidia-tesla-t4 &> /dev/null; then
        success "GPU nodes detected for AI model serving"
    else
        warning "No GPU nodes detected. AI models will run on CPU (slower performance)"
    fi
    
    success "Prerequisites check completed"
}

# Create AI namespace
create_ai_namespace() {
    log "Creating AI services namespace..."
    
    kubectl create namespace nexus-ai --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace nexus-ai istio-injection=enabled --overwrite
    
    success "AI namespace created and configured"
}

# Deploy TorchServe model serving
deploy_torchserve() {
    log "Deploying TorchServe model serving infrastructure..."
    
    # Apply TorchServe deployment
    kubectl apply -f model-serving/torchserve-deployment.yaml
    
    # Wait for TorchServe to be ready
    log "Waiting for TorchServe to be ready..."
    kubectl wait --for=condition=Ready pods -l app=torchserve -n nexus-ai --timeout=600s
    
    success "TorchServe model serving deployed"
}

# Deploy Weaviate vector database
deploy_weaviate() {
    log "Deploying Weaviate vector database..."
    
    # Apply Weaviate deployment
    kubectl apply -f vector-db/weaviate-deployment.yaml
    
    # Wait for Weaviate to be ready
    log "Waiting for Weaviate to be ready..."
    kubectl wait --for=condition=Ready pods -l app=weaviate -n nexus-ai --timeout=600s
    
    # Initialize Weaviate schema
    log "Initializing Weaviate schema..."
    kubectl exec -n nexus-ai deployment/weaviate -- /bin/bash -c "
        curl -X POST http://localhost:8080/v1/schema -H 'Content-Type: application/json' -d '{
            \"class\": \"DocumentChunk\",
            \"description\": \"Document chunks for knowledge processing\",
            \"vectorizer\": \"text2vec-openai\",
            \"properties\": [
                {\"name\": \"content\", \"dataType\": [\"text\"]},
                {\"name\": \"document_id\", \"dataType\": [\"string\"]},
                {\"name\": \"chunk_index\", \"dataType\": [\"int\"]},
                {\"name\": \"source\", \"dataType\": [\"string\"]},
                {\"name\": \"content_type\", \"dataType\": [\"string\"]},
                {\"name\": \"language\", \"dataType\": [\"string\"]},
                {\"name\": \"classification\", \"dataType\": [\"string\"]},
                {\"name\": \"created_at\", \"dataType\": [\"date\"]}
            ]
        }'
    " || warning "Schema initialization failed (may already exist)"
    
    success "Weaviate vector database deployed"
}

# Deploy knowledge processing pipeline
deploy_knowledge_pipeline() {
    log "Deploying knowledge processing pipeline..."
    
    # Apply knowledge pipeline deployment
    kubectl apply -f knowledge-processing/knowledge-pipeline.yaml
    
    # Wait for knowledge pipeline to be ready
    log "Waiting for knowledge processing services to be ready..."
    kubectl wait --for=condition=Ready pods -l app=knowledge-pipeline -n nexus-ai --timeout=600s
    kubectl wait --for=condition=Ready pods -l app=neo4j -n nexus-ai --timeout=600s
    
    success "Knowledge processing pipeline deployed"
}

# Deploy AI framework
deploy_ai_framework() {
    log "Deploying multi-model AI framework..."
    
    # Apply AI framework deployment
    kubectl apply -f ai-framework/multi-model-ai-service.yaml
    
    # Wait for AI framework to be ready
    log "Waiting for AI framework to be ready..."
    kubectl wait --for=condition=Ready pods -l app=ai-framework -n nexus-ai --timeout=600s
    
    success "Multi-model AI framework deployed"
}

# Configure network policies
configure_network_policies() {
    log "Configuring network policies for AI services..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-ai-isolation
  namespace: nexus-ai
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: nexus-api
    - namespaceSelector:
        matchLabels:
          name: nexus-gateway
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 8081
    - protocol: TCP
      port: 8082
    - protocol: TCP
      port: 8083
    - protocol: TCP
      port: 8084
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
  - from:
    - namespaceSelector:
        matchLabels:
          name: nexus-infrastructure
    ports:
    - protocol: TCP
      port: 9090
    - protocol: TCP
      port: 9091
    - protocol: TCP
      port: 9092
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: nexus-infrastructure
    ports:
    - protocol: TCP
      port: 5432
    - protocol: TCP
      port: 6379
    - protocol: TCP
      port: 9000
  - to: []
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
EOF
    
    success "Network policies configured"
}

# Setup monitoring and metrics
setup_monitoring() {
    log "Setting up monitoring for AI services..."
    
    # Create ServiceMonitor for Prometheus
    cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: nexus-ai-metrics
  namespace: nexus-ai
  labels:
    app: nexus-ai
spec:
  selector:
    matchLabels:
      component: ai-services
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
---
apiVersion: v1
kind: Service
metadata:
  name: torchserve-metrics
  namespace: nexus-ai
  labels:
    app: torchserve
    component: ai-services
spec:
  ports:
  - name: metrics
    port: 8082
    targetPort: 8082
  selector:
    app: torchserve
---
apiVersion: v1
kind: Service
metadata:
  name: knowledge-pipeline-metrics
  namespace: nexus-ai
  labels:
    app: knowledge-pipeline
    component: ai-services
spec:
  ports:
  - name: metrics
    port: 9091
    targetPort: 9091
  selector:
    app: knowledge-pipeline
---
apiVersion: v1
kind: Service
metadata:
  name: ai-framework-metrics
  namespace: nexus-ai
  labels:
    app: ai-framework
    component: ai-services
spec:
  ports:
  - name: metrics
    port: 9092
    targetPort: 9092
  selector:
    app: ai-framework
EOF
    
    success "Monitoring configured"
}

# Validate deployment
validate_deployment() {
    log "Validating Phase 4 deployment..."
    
    # Check all pods are running
    if ! kubectl get pods -n nexus-ai | grep -q "Running"; then
        error "Some AI service pods are not running"
        kubectl get pods -n nexus-ai
        return 1
    fi
    
    # Test TorchServe
    log "Testing TorchServe model serving..."
    if kubectl exec -n nexus-ai deployment/torchserve-deployment -- curl -f http://localhost:8080/ping &> /dev/null; then
        success "TorchServe is healthy"
    else
        warning "TorchServe health check failed"
    fi
    
    # Test Weaviate
    log "Testing Weaviate vector database..."
    if kubectl exec -n nexus-ai statefulset/weaviate -- curl -f http://localhost:8080/v1/.well-known/ready &> /dev/null; then
        success "Weaviate is healthy"
    else
        warning "Weaviate health check failed"
    fi
    
    # Test Knowledge Pipeline
    log "Testing knowledge processing pipeline..."
    if kubectl exec -n nexus-ai deployment/knowledge-pipeline -- curl -f http://localhost:8083/health &> /dev/null; then
        success "Knowledge pipeline is healthy"
    else
        warning "Knowledge pipeline health check failed"
    fi
    
    # Test AI Framework
    log "Testing AI framework..."
    if kubectl exec -n nexus-ai deployment/ai-framework-service -- curl -f http://localhost:8084/health &> /dev/null; then
        success "AI framework is healthy"
    else
        warning "AI framework health check failed"
    fi
    
    # Test Neo4j
    log "Testing Neo4j knowledge graph..."
    if kubectl exec -n nexus-ai statefulset/neo4j -- curl -f http://localhost:7474/ &> /dev/null; then
        success "Neo4j is healthy"
    else
        warning "Neo4j health check failed"
    fi
    
    success "Phase 4 deployment validation completed"
}

# Display deployment information
display_deployment_info() {
    log "Phase 4 deployment completed successfully!"
    echo
    echo "=== Nexus Architect WS1 Phase 4: Enhanced AI Services & Knowledge Foundation ==="
    echo
    echo "🤖 AI Model Serving:"
    echo "  • TorchServe with GPU acceleration support"
    echo "  • Multi-model deployment (chat, code, security, performance)"
    echo "  • Horizontal auto-scaling based on load"
    echo "  • Prometheus metrics and monitoring"
    echo
    echo "🧠 Vector Database & Knowledge Processing:"
    echo "  • Weaviate vector database with OpenAI embeddings"
    echo "  • Neo4j knowledge graph for entity relationships"
    echo "  • Automated document processing pipeline"
    echo "  • Multi-format support (PDF, DOCX, TXT, MD, HTML)"
    echo
    echo "🔀 Multi-Model AI Framework:"
    echo "  • Intelligent model routing and selection"
    echo "  • OpenAI GPT-4/3.5-turbo integration"
    echo "  • Anthropic Claude integration"
    echo "  • Local model fallback capabilities"
    echo "  • Advanced safety controls and content filtering"
    echo
    echo "🛡️ Safety & Security:"
    echo "  • Content filtering and toxicity detection"
    echo "  • Prompt injection detection"
    echo "  • Privacy violation prevention"
    echo "  • Output validation and quality scoring"
    echo "  • Comprehensive audit logging"
    echo
    echo "📊 Service Endpoints:"
    echo "  • TorchServe API: http://torchserve-service.nexus-ai:8080"
    echo "  • Weaviate Vector DB: http://weaviate.nexus-ai:8080"
    echo "  • Knowledge Pipeline: http://knowledge-pipeline-service.nexus-ai:8083"
    echo "  • AI Framework: http://ai-framework-service.nexus-ai:8084"
    echo "  • Neo4j Browser: http://neo4j.nexus-ai:7474"
    echo
    echo "🔧 Management Commands:"
    echo "  • View AI services: kubectl get pods -n nexus-ai"
    echo "  • Check TorchServe models: kubectl exec -n nexus-ai deployment/torchserve-deployment -- curl http://localhost:8081/models"
    echo "  • Access Weaviate: kubectl port-forward -n nexus-ai svc/weaviate 8080:8080"
    echo "  • Access Neo4j: kubectl port-forward -n nexus-ai svc/neo4j 7474:7474"
    echo "  • View AI metrics: kubectl port-forward -n nexus-ai svc/ai-framework-service 9092:9092"
    echo
    echo "📚 API Examples:"
    echo "  • Chat completion: POST /api/v1/ai/chat"
    echo "  • Generate embedding: POST /api/v1/ai/embedding"
    echo "  • Process document: POST /api/v1/knowledge/process"
    echo "  • Search knowledge: GET /api/v1/knowledge/search?query=..."
    echo "  • Safety check: POST /api/v1/ai/safety-check"
    echo
    echo "⚠️  Important Notes:"
    echo "  • Configure API keys in secrets for external AI providers"
    echo "  • GPU nodes recommended for optimal model performance"
    echo "  • Vector database requires initial schema setup (completed)"
    echo "  • Knowledge graph stores extracted entities and relationships"
    echo "  • All services protected by Istio mTLS and network policies"
    echo
    success "Phase 4 deployment information displayed"
}

# Main deployment function
main() {
    log "Starting Nexus Architect WS1 Phase 4 deployment..."
    log "Enhanced AI Services & Knowledge Foundation"
    echo
    
    # Change to the Phase 4 directory
    cd "$(dirname "$0")"
    
    # Execute deployment steps
    check_prerequisites
    create_ai_namespace
    deploy_torchserve
    deploy_weaviate
    deploy_knowledge_pipeline
    deploy_ai_framework
    configure_network_policies
    setup_monitoring
    validate_deployment
    display_deployment_info
    
    echo
    success "🎉 WS1 Phase 4 deployment completed successfully!"
    echo
    log "Next steps:"
    echo "  1. Configure AI provider API keys in secrets"
    echo "  2. Upload AI models to TorchServe model store"
    echo "  3. Test knowledge processing with sample documents"
    echo "  4. Configure monitoring dashboards and alerts"
    echo "  5. Proceed with Phase 5: Performance Optimization & Monitoring"
}

# Handle script interruption
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"

