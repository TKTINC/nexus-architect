#!/bin/bash

# Nexus Architect WS1 Phase 2 Deployment Script
# Authentication, Authorization & API Foundation

set -e

echo "🚀 Starting Nexus Architect WS1 Phase 2 Deployment"
echo "=================================================="

# Configuration
NAMESPACE_AUTH="nexus-auth"
NAMESPACE_GATEWAY="nexus-gateway"
NAMESPACE_API="nexus-api"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check if Phase 1 infrastructure is running
    if ! kubectl get namespace nexus-infrastructure &> /dev/null; then
        error "Phase 1 infrastructure not found. Please deploy Phase 1 first."
    fi
    
    # Check if PostgreSQL is running
    if ! kubectl get pods -n nexus-infrastructure -l app=postgresql-primary | grep -q Running; then
        error "PostgreSQL from Phase 1 is not running"
    fi
    
    # Check if Redis is running
    if ! kubectl get pods -n nexus-infrastructure -l app=redis | grep -q Running; then
        error "Redis from Phase 1 is not running"
    fi
    
    success "Prerequisites check passed"
}

# Deploy Keycloak
deploy_keycloak() {
    log "Deploying Keycloak identity provider..."
    
    # Apply Keycloak configuration
    kubectl apply -f "${SCRIPT_DIR}/keycloak/keycloak-cluster.yaml"
    
    # Wait for Keycloak to be ready
    log "Waiting for Keycloak to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/keycloak -n ${NAMESPACE_AUTH}
    
    # Wait for database initialization job
    kubectl wait --for=condition=complete --timeout=300s job/keycloak-db-init -n ${NAMESPACE_AUTH}
    
    success "Keycloak deployed successfully"
}

# Deploy Kong API Gateway
deploy_kong() {
    log "Deploying Kong API Gateway..."
    
    # Apply Kong configuration
    kubectl apply -f "${SCRIPT_DIR}/kong/kong-gateway.yaml"
    
    # Wait for Kong migration job
    kubectl wait --for=condition=complete --timeout=300s job/kong-migration -n ${NAMESPACE_GATEWAY}
    
    # Wait for Kong to be ready
    log "Waiting for Kong Gateway to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/kong-gateway -n ${NAMESPACE_GATEWAY}
    
    success "Kong API Gateway deployed successfully"
}

# Configure OAuth and RBAC
configure_oauth() {
    log "Configuring OAuth 2.0 and RBAC policies..."
    
    # Apply OAuth configuration
    kubectl apply -f "${SCRIPT_DIR}/oauth/oauth-config.yaml"
    
    # Apply authorization policies
    kubectl apply -f "${SCRIPT_DIR}/policies/authorization-policies.yaml"
    
    # Wait for realm setup job
    kubectl wait --for=condition=complete --timeout=300s job/keycloak-realm-setup -n ${NAMESPACE_AUTH}
    
    success "OAuth and RBAC configured successfully"
}

# Build and deploy FastAPI application
deploy_fastapi() {
    log "Building and deploying FastAPI application..."
    
    # Build Docker image
    cd "${SCRIPT_DIR}/fastapi"
    
    # Check if Docker is available
    if command -v docker &> /dev/null; then
        log "Building FastAPI Docker image..."
        docker build -t nexus-architect/api:latest .
        
        # If using kind or minikube, load image
        if kubectl config current-context | grep -q "kind\|minikube"; then
            log "Loading image into cluster..."
            if command -v kind &> /dev/null; then
                kind load docker-image nexus-architect/api:latest
            elif command -v minikube &> /dev/null; then
                minikube image load nexus-architect/api:latest
            fi
        fi
    else
        warning "Docker not available. Using pre-built image or building in cluster."
    fi
    
    cd "${SCRIPT_DIR}"
    
    # Deploy FastAPI application
    kubectl apply -f "${SCRIPT_DIR}/fastapi/deployment.yaml"
    
    # Wait for database initialization
    kubectl wait --for=condition=complete --timeout=300s job/nexus-api-db-init -n ${NAMESPACE_API}
    
    # Wait for API to be ready
    log "Waiting for Nexus API to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/nexus-api -n ${NAMESPACE_API}
    
    success "FastAPI application deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying Phase 2 deployment..."
    
    # Check all pods are running
    echo ""
    echo "📊 Deployment Status:"
    echo "===================="
    
    # Keycloak status
    echo "🔐 Keycloak:"
    kubectl get pods -n ${NAMESPACE_AUTH} -l app=keycloak
    
    # Kong status
    echo ""
    echo "🌐 Kong Gateway:"
    kubectl get pods -n ${NAMESPACE_GATEWAY} -l app=kong-gateway
    
    # API status
    echo ""
    echo "🚀 Nexus API:"
    kubectl get pods -n ${NAMESPACE_API} -l app=nexus-api
    
    # Check services
    echo ""
    echo "🔗 Services:"
    echo "============"
    kubectl get services -n ${NAMESPACE_AUTH}
    kubectl get services -n ${NAMESPACE_GATEWAY}
    kubectl get services -n ${NAMESPACE_API}
    
    # Check ingresses
    echo ""
    echo "🌍 Ingresses:"
    echo "============="
    kubectl get ingress -n ${NAMESPACE_AUTH}
    kubectl get ingress -n ${NAMESPACE_GATEWAY}
    kubectl get ingress -n ${NAMESPACE_API}
    
    success "Phase 2 deployment verification completed"
}

# Test endpoints
test_endpoints() {
    log "Testing API endpoints..."
    
    # Get cluster IP for testing
    API_SERVICE=$(kubectl get service nexus-api -n ${NAMESPACE_API} -o jsonpath='{.spec.clusterIP}')
    KEYCLOAK_SERVICE=$(kubectl get service keycloak -n ${NAMESPACE_AUTH} -o jsonpath='{.spec.clusterIP}')
    
    # Test API health endpoint
    if kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -- \
        curl -f "http://${API_SERVICE}:8000/health" > /dev/null 2>&1; then
        success "API health endpoint is responding"
    else
        warning "API health endpoint test failed"
    fi
    
    # Test Keycloak health
    if kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -- \
        curl -f "http://${KEYCLOAK_SERVICE}:8080/health/ready" > /dev/null 2>&1; then
        success "Keycloak health endpoint is responding"
    else
        warning "Keycloak health endpoint test failed"
    fi
}

# Display access information
display_access_info() {
    log "Phase 2 deployment completed successfully!"
    
    echo ""
    echo "🎉 Access Information:"
    echo "====================="
    echo ""
    echo "🔐 Keycloak Admin Console:"
    echo "   URL: https://auth.nexus-architect.local"
    echo "   Username: admin"
    echo "   Password: NexusAdmin2024"
    echo ""
    echo "🌐 Kong Admin API:"
    echo "   URL: https://kong-admin.nexus-architect.local"
    echo "   (Internal access only)"
    echo ""
    echo "🚀 Nexus API:"
    echo "   URL: https://api.nexus-architect.local"
    echo "   Health: https://api.nexus-architect.local/health"
    echo "   Docs: https://api.nexus-architect.local/docs"
    echo ""
    echo "📝 Next Steps:"
    echo "=============="
    echo "1. Configure DNS entries for the above domains"
    echo "2. Set up SSL certificates for production"
    echo "3. Configure OAuth clients in Keycloak"
    echo "4. Test authentication flows"
    echo "5. Proceed to WS1 Phase 3 (Advanced Security)"
    echo ""
}

# Main deployment function
main() {
    echo "🚀 Nexus Architect WS1 Phase 2 Deployment"
    echo "=========================================="
    echo "Deploying: Authentication, Authorization & API Foundation"
    echo ""
    
    check_prerequisites
    deploy_keycloak
    deploy_kong
    configure_oauth
    deploy_fastapi
    verify_deployment
    test_endpoints
    display_access_info
    
    success "🎉 WS1 Phase 2 deployment completed successfully!"
}

# Handle script interruption
trap 'error "Deployment interrupted"' INT TERM

# Run main function
main "$@"

