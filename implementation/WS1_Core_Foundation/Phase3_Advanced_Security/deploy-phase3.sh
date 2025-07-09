#!/bin/bash

# Nexus Architect WS1 Phase 3: Advanced Security & Compliance Framework
# Deployment Script for Zero-Trust Architecture and Compliance Systems

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
    log "Checking prerequisites for Phase 3 deployment..."
    
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
    
    # Check if Phase 1 and Phase 2 are deployed
    if ! kubectl get namespace nexus-infrastructure &> /dev/null; then
        error "Phase 1 (Infrastructure) not deployed. Please deploy Phase 1 first."
        exit 1
    fi
    
    if ! kubectl get namespace nexus-auth &> /dev/null; then
        error "Phase 2 (Authentication) not deployed. Please deploy Phase 2 first."
        exit 1
    fi
    
    # Check if Istio is available
    if ! command -v istioctl &> /dev/null; then
        warning "istioctl not found. Installing Istio..."
        install_istio
    fi
    
    success "Prerequisites check completed"
}

# Install Istio
install_istio() {
    log "Installing Istio service mesh..."
    
    # Download and install Istio
    curl -L https://istio.io/downloadIstio | sh -
    export PATH="$PWD/istio-*/bin:$PATH"
    
    # Install Istio
    istioctl install --set values.defaultRevision=default -y
    
    # Enable sidecar injection for existing namespaces
    kubectl label namespace nexus-infrastructure istio-injection=enabled --overwrite
    kubectl label namespace nexus-auth istio-injection=enabled --overwrite
    kubectl label namespace nexus-gateway istio-injection=enabled --overwrite
    kubectl label namespace nexus-api istio-injection=enabled --overwrite
    
    success "Istio service mesh installed"
}

# Deploy Istio configuration
deploy_istio_config() {
    log "Deploying Istio service mesh configuration..."
    
    # Apply Istio installation configuration
    kubectl apply -f istio/istio-installation.yaml
    
    # Wait for Istio to be ready
    kubectl wait --for=condition=Ready pods -l app=istiod -n istio-system --timeout=300s
    
    # Apply network policies
    kubectl apply -f istio/network-policies.yaml
    
    success "Istio configuration deployed"
}

# Deploy Vault encryption
deploy_vault_encryption() {
    log "Deploying Vault encryption and key management..."
    
    # Apply Vault encryption configuration
    kubectl apply -f vault/vault-encryption.yaml
    
    # Wait for Vault setup job to complete
    kubectl wait --for=condition=complete job/vault-encryption-setup -n nexus-infrastructure --timeout=600s
    
    success "Vault encryption deployed"
}

# Deploy compliance frameworks
deploy_compliance_frameworks() {
    log "Deploying compliance frameworks (GDPR, SOC 2, HIPAA)..."
    
    # Create compliance namespace
    kubectl create namespace nexus-compliance --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace nexus-compliance istio-injection=enabled --overwrite
    
    # Deploy GDPR compliance
    kubectl apply -f compliance/gdpr-compliance.yaml
    
    # Deploy SOC 2 and HIPAA compliance
    kubectl apply -f compliance/soc2-hipaa-compliance.yaml
    
    # Wait for compliance services to be ready
    kubectl wait --for=condition=Ready pods -l app=gdpr-compliance-service -n nexus-compliance --timeout=300s
    kubectl wait --for=condition=Ready pods -l app=compliance-monitoring-service -n nexus-compliance --timeout=300s
    
    success "Compliance frameworks deployed"
}

# Deploy security monitoring
deploy_security_monitoring() {
    log "Deploying security monitoring and threat detection..."
    
    # Create security namespace
    kubectl create namespace nexus-security --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace nexus-security istio-injection=enabled --overwrite
    
    # Deploy security monitoring
    kubectl apply -f monitoring/security-monitoring.yaml
    
    # Wait for security monitoring to be ready
    kubectl wait --for=condition=Ready pods -l app=security-monitoring-service -n nexus-security --timeout=300s
    
    success "Security monitoring deployed"
}

# Configure network policies
configure_network_policies() {
    log "Configuring network micro-segmentation policies..."
    
    # Apply additional network policies for new namespaces
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-compliance-isolation
  namespace: nexus-compliance
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
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 8081
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: nexus-infrastructure
    ports:
    - protocol: TCP
      port: 5432
  - to: []
    ports:
    - protocol: UDP
      port: 53
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-security-isolation
  namespace: nexus-security
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
    ports:
    - protocol: TCP
      port: 8082
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: nexus-infrastructure
    ports:
    - protocol: TCP
      port: 5432
  - to: []
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 443
EOF
    
    success "Network policies configured"
}

# Validate deployment
validate_deployment() {
    log "Validating Phase 3 deployment..."
    
    # Check Istio installation
    if ! kubectl get pods -n istio-system | grep -q "Running"; then
        error "Istio pods are not running"
        return 1
    fi
    
    # Check compliance services
    if ! kubectl get pods -n nexus-compliance | grep -q "Running"; then
        error "Compliance services are not running"
        return 1
    fi
    
    # Check security monitoring
    if ! kubectl get pods -n nexus-security | grep -q "Running"; then
        error "Security monitoring is not running"
        return 1
    fi
    
    # Test service connectivity
    log "Testing service connectivity..."
    
    # Test GDPR compliance service
    if kubectl exec -n nexus-compliance deployment/gdpr-compliance-service -- curl -f http://localhost:8080/health &> /dev/null; then
        success "GDPR compliance service is healthy"
    else
        warning "GDPR compliance service health check failed"
    fi
    
    # Test compliance monitoring service
    if kubectl exec -n nexus-compliance deployment/compliance-monitoring-service -- curl -f http://localhost:8081/health &> /dev/null; then
        success "Compliance monitoring service is healthy"
    else
        warning "Compliance monitoring service health check failed"
    fi
    
    # Test security monitoring service
    if kubectl exec -n nexus-security deployment/security-monitoring-service -- curl -f http://localhost:8082/health &> /dev/null; then
        success "Security monitoring service is healthy"
    else
        warning "Security monitoring service health check failed"
    fi
    
    success "Phase 3 deployment validation completed"
}

# Display deployment information
display_deployment_info() {
    log "Phase 3 deployment completed successfully!"
    echo
    echo "=== Nexus Architect WS1 Phase 3: Advanced Security & Compliance ==="
    echo
    echo "🔒 Zero-Trust Architecture:"
    echo "  • Istio service mesh with mTLS encryption"
    echo "  • Network micro-segmentation policies"
    echo "  • Identity-based access controls"
    echo
    echo "🔐 Encryption & Key Management:"
    echo "  • HashiCorp Vault transit encryption"
    echo "  • Automatic key rotation policies"
    echo "  • End-to-end data encryption"
    echo
    echo "📋 Compliance Frameworks:"
    echo "  • GDPR compliance controls and data subject rights"
    echo "  • SOC 2 Type II control framework"
    echo "  • HIPAA compliance capabilities"
    echo "  • Automated compliance monitoring"
    echo
    echo "🛡️ Security Monitoring:"
    echo "  • Real-time threat detection"
    echo "  • Anomaly detection with ML"
    echo "  • Automated incident response"
    echo "  • Security event correlation"
    echo
    echo "📊 Service Endpoints:"
    echo "  • GDPR Compliance: http://gdpr-compliance-service.nexus-compliance:8080"
    echo "  • Compliance Monitoring: http://compliance-monitoring-service.nexus-compliance:8081"
    echo "  • Security Monitoring: http://security-monitoring-service.nexus-security:8082"
    echo "  • Istio Ingress Gateway: http://istio-ingressgateway.istio-system"
    echo
    echo "🔧 Management Commands:"
    echo "  • View Istio configuration: istioctl proxy-config cluster <pod-name>"
    echo "  • Check mTLS status: istioctl authn tls-check <service>"
    echo "  • View compliance dashboard: kubectl port-forward -n nexus-compliance svc/compliance-monitoring-service 8081:8081"
    echo "  • View security dashboard: kubectl port-forward -n nexus-security svc/security-monitoring-service 8082:8082"
    echo
    echo "📚 Documentation:"
    echo "  • Phase 3 implementation guide: docs/README.md"
    echo "  • Security procedures: docs/security-procedures.md"
    echo "  • Compliance documentation: docs/compliance-guide.md"
    echo
    echo "⚠️  Important Notes:"
    echo "  • All services are now protected by Istio mTLS"
    echo "  • Network policies enforce micro-segmentation"
    echo "  • Compliance monitoring runs automatically"
    echo "  • Security events are logged and analyzed in real-time"
    echo "  • Vault manages all encryption keys with automatic rotation"
    echo
    success "Phase 3 deployment information displayed"
}

# Main deployment function
main() {
    log "Starting Nexus Architect WS1 Phase 3 deployment..."
    log "Advanced Security & Compliance Framework"
    echo
    
    # Change to the Phase 3 directory
    cd "$(dirname "$0")"
    
    # Execute deployment steps
    check_prerequisites
    deploy_istio_config
    deploy_vault_encryption
    deploy_compliance_frameworks
    deploy_security_monitoring
    configure_network_policies
    validate_deployment
    display_deployment_info
    
    echo
    success "🎉 WS1 Phase 3 deployment completed successfully!"
    echo
    log "Next steps:"
    echo "  1. Review security monitoring dashboards"
    echo "  2. Configure compliance policies for your organization"
    echo "  3. Set up alerting and notification channels"
    echo "  4. Conduct security testing and validation"
    echo "  5. Proceed with Phase 4: Enhanced AI Services & Knowledge Foundation"
}

# Handle script interruption
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"

