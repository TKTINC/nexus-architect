#!/bin/bash

# Nexus Architect - WS1 Phase 1 Deployment Script
# Infrastructure Foundation and Basic Security

set -euo pipefail

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
    
    # Check if helm is available
    if ! command -v helm &> /dev/null; then
        warning "Helm is not installed. Some components may require manual installation."
    fi
    
    success "Prerequisites check completed"
}

# Deploy Calico CNI
deploy_calico() {
    log "Deploying Calico CNI..."
    
    # Install Tigera operator
    kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.4/manifests/tigera-operator.yaml || true
    
    # Wait for operator to be ready
    kubectl wait --for=condition=Ready pod -l name=tigera-operator -n tigera-operator --timeout=300s
    
    # Apply Calico configuration
    kubectl apply -f kubernetes/calico-config.yaml
    
    # Wait for Calico to be ready
    kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=calico-node -n calico-system --timeout=300s
    
    success "Calico CNI deployed successfully"
}

# Create namespace
create_namespace() {
    log "Creating nexus-architect namespace..."
    
    kubectl apply -f postgresql/postgresql-cluster.yaml
    
    success "Namespace created successfully"
}

# Deploy PostgreSQL
deploy_postgresql() {
    log "Deploying PostgreSQL cluster..."
    
    kubectl apply -f postgresql/postgresql-cluster.yaml
    
    # Wait for PostgreSQL to be ready
    kubectl wait --for=condition=Ready pod -l app=postgresql,role=primary -n nexus-architect --timeout=600s
    
    # Verify database initialization
    kubectl exec -n nexus-architect postgresql-primary-0 -- psql -U postgres -d nexus_architect -c "SELECT 1;" > /dev/null
    
    success "PostgreSQL cluster deployed successfully"
}

# Deploy Redis
deploy_redis() {
    log "Deploying Redis cluster..."
    
    kubectl apply -f redis/redis-cluster.yaml
    
    # Wait for Redis master to be ready
    kubectl wait --for=condition=Ready pod -l app=redis,role=master -n nexus-architect --timeout=300s
    
    # Wait for Redis replicas to be ready
    kubectl wait --for=condition=Ready pod -l app=redis,role=replica -n nexus-architect --timeout=300s
    
    # Verify Redis connectivity
    kubectl exec -n nexus-architect redis-master-0 -- redis-cli -a nexus_redis_2024! ping
    
    success "Redis cluster deployed successfully"
}

# Deploy MinIO
deploy_minio() {
    log "Deploying MinIO object storage..."
    
    kubectl apply -f minio/minio-cluster.yaml
    
    # Wait for MinIO to be ready
    kubectl wait --for=condition=Ready pod -l app=minio -n nexus-architect --timeout=600s
    
    # Wait for setup job to complete
    kubectl wait --for=condition=Complete job/minio-setup -n nexus-architect --timeout=300s
    
    success "MinIO object storage deployed successfully"
}

# Deploy Vault
deploy_vault() {
    log "Deploying HashiCorp Vault..."
    
    kubectl apply -f vault/vault-cluster.yaml
    
    # Wait for Vault to be ready
    kubectl wait --for=condition=Ready pod -l app=vault -n nexus-architect --timeout=600s
    
    # Wait for initialization job to complete
    kubectl wait --for=condition=Complete job/vault-init -n nexus-architect --timeout=300s
    
    success "HashiCorp Vault deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    kubectl apply -f monitoring/prometheus-grafana.yaml
    
    # Wait for Prometheus to be ready
    kubectl wait --for=condition=Ready pod -l app=prometheus -n monitoring --timeout=300s
    
    # Wait for Grafana to be ready
    kubectl wait --for=condition=Ready pod -l app=grafana -n monitoring --timeout=300s
    
    success "Monitoring stack deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check all pods are running
    echo "Checking pod status in nexus-architect namespace:"
    kubectl get pods -n nexus-architect
    
    echo "Checking pod status in monitoring namespace:"
    kubectl get pods -n monitoring
    
    # Check services
    echo "Checking services in nexus-architect namespace:"
    kubectl get svc -n nexus-architect
    
    echo "Checking services in monitoring namespace:"
    kubectl get svc -n monitoring
    
    # Check persistent volumes
    echo "Checking persistent volume claims:"
    kubectl get pvc -n nexus-architect
    kubectl get pvc -n monitoring
    
    # Verify database connectivity
    log "Verifying database connectivity..."
    kubectl exec -n nexus-architect postgresql-primary-0 -- psql -U postgres -d nexus_architect -c "SELECT COUNT(*) FROM auth.users;"
    
    # Verify Redis connectivity
    log "Verifying Redis connectivity..."
    kubectl exec -n nexus-architect redis-master-0 -- redis-cli -a nexus_redis_2024! info replication
    
    # Verify MinIO connectivity
    log "Verifying MinIO connectivity..."
    kubectl exec -n nexus-architect minio-0 -- mc ls nexus/
    
    # Verify Vault status
    log "Verifying Vault status..."
    kubectl exec -n nexus-architect vault-0 -- vault status
    
    success "Deployment verification completed"
}

# Generate access information
generate_access_info() {
    log "Generating access information..."
    
    cat > access-info.txt << EOF
# Nexus Architect - Phase 1 Access Information
# Generated on: $(date)

## Database Access
PostgreSQL Primary: postgresql-primary.nexus-architect.svc.cluster.local:5432
Database: nexus_architect
Username: nexus_app
Password: nexus_app_pass_2024!

Admin Username: postgres
Admin Password: nexus_admin_2024!

## Redis Access
Redis Master: redis-master.nexus-architect.svc.cluster.local:6379
Redis Replicas: redis-replica.nexus-architect.svc.cluster.local:6379
Password: nexus_redis_2024!

## MinIO Access
MinIO API: minio-api.nexus-architect.svc.cluster.local:9000
MinIO Console: minio-console.nexus-architect.svc.cluster.local:9001
Access Key: minioadmin
Secret Key: nexus_minio_2024!

## Vault Access
Vault API: vault-active.nexus-architect.svc.cluster.local:8200
Root Token: (Check vault-init job logs for root token)

## Monitoring Access
Prometheus: prometheus.monitoring.svc.cluster.local:9090
Grafana: grafana.monitoring.svc.cluster.local:3000
Grafana Username: admin
Grafana Password: nexus_grafana_2024!

## Port Forwarding Commands
# PostgreSQL
kubectl port-forward -n nexus-architect svc/postgresql-primary 5432:5432

# Redis
kubectl port-forward -n nexus-architect svc/redis-master 6379:6379

# MinIO API
kubectl port-forward -n nexus-architect svc/minio-api 9000:9000

# MinIO Console
kubectl port-forward -n nexus-architect svc/minio-console 9001:9001

# Vault
kubectl port-forward -n nexus-architect svc/vault-active 8200:8200

# Prometheus
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

## Useful Commands
# Check all pods
kubectl get pods -n nexus-architect
kubectl get pods -n monitoring

# Check logs
kubectl logs -n nexus-architect -l app=postgresql
kubectl logs -n nexus-architect -l app=redis
kubectl logs -n nexus-architect -l app=minio
kubectl logs -n nexus-architect -l app=vault
kubectl logs -n monitoring -l app=prometheus
kubectl logs -n monitoring -l app=grafana

# Scale deployments
kubectl scale deployment redis-replica -n nexus-architect --replicas=3
kubectl scale statefulset minio -n nexus-architect --replicas=6

EOF

    success "Access information generated in access-info.txt"
}

# Main deployment function
main() {
    log "Starting Nexus Architect WS1 Phase 1 deployment..."
    
    check_prerequisites
    
    # Deploy components in order
    deploy_calico
    create_namespace
    deploy_postgresql
    deploy_redis
    deploy_minio
    deploy_vault
    deploy_monitoring
    
    # Verify everything is working
    verify_deployment
    
    # Generate access information
    generate_access_info
    
    success "WS1 Phase 1 deployment completed successfully!"
    
    log "Next steps:"
    echo "1. Review the access-info.txt file for connection details"
    echo "2. Set up port forwarding to access services locally"
    echo "3. Verify all services are functioning correctly"
    echo "4. Proceed to WS1 Phase 2: Authentication, Authorization & API Foundation"
}

# Handle script interruption
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"

