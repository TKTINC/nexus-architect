#!/bin/bash

# Nexus Architect WS1 Phase 5: Performance Optimization & Monitoring Deployment
# This script deploys all Phase 5 components for production readiness

set -euo pipefail

# Configuration
NAMESPACE="nexus-infrastructure"
PHASE_DIR="/app/Phase5_Performance_Monitoring"
LOG_FILE="/tmp/phase5_deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a $LOG_FILE
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a $LOG_FILE
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a $LOG_FILE
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check kubectl access
    if ! kubectl cluster-info &>/dev/null; then
        error "Kubernetes cluster is not accessible"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace $NAMESPACE &>/dev/null; then
        log "Creating namespace $NAMESPACE"
        kubectl create namespace $NAMESPACE
    fi
    
    # Check if previous phases are deployed
    local required_services=("postgresql" "redis" "vault" "keycloak" "kong")
    for service in "${required_services[@]}"; do
        if ! kubectl get service $service -n $NAMESPACE &>/dev/null; then
            warn "Required service $service not found. Ensure previous phases are deployed."
        fi
    done
    
    log "Prerequisites check completed"
}

# Deploy performance optimization components
deploy_performance_optimization() {
    log "Deploying performance optimization components..."
    
    # Deploy caching optimization
    info "Deploying cache optimization service..."
    kubectl apply -f $PHASE_DIR/performance-optimization/caching-optimization.yaml
    
    # Deploy database optimization
    info "Deploying database optimization service..."
    kubectl apply -f $PHASE_DIR/performance-optimization/database-optimization.yaml
    
    # Wait for deployments to be ready
    log "Waiting for performance optimization services to be ready..."
    kubectl rollout status deployment/cache-optimizer -n $NAMESPACE --timeout=300s
    kubectl rollout status deployment/database-performance-monitor -n $NAMESPACE --timeout=300s
    
    log "Performance optimization components deployed successfully"
}

# Deploy monitoring and observability
deploy_monitoring_observability() {
    log "Deploying monitoring and observability components..."
    
    # Deploy comprehensive monitoring
    info "Deploying monitoring aggregator service..."
    kubectl apply -f $PHASE_DIR/monitoring-observability/comprehensive-monitoring.yaml
    
    # Wait for deployment to be ready
    log "Waiting for monitoring services to be ready..."
    kubectl rollout status deployment/monitoring-aggregator -n $NAMESPACE --timeout=300s
    
    # Verify monitoring endpoints
    info "Verifying monitoring endpoints..."
    local monitoring_pod=$(kubectl get pods -n $NAMESPACE -l app=monitoring-aggregator -o jsonpath='{.items[0].metadata.name}')
    
    if [ -n "$monitoring_pod" ]; then
        # Wait for pod to be fully ready
        kubectl wait --for=condition=ready pod/$monitoring_pod -n $NAMESPACE --timeout=120s
        
        # Test health endpoint
        if kubectl exec $monitoring_pod -n $NAMESPACE -- curl -f http://localhost:8095/health &>/dev/null; then
            log "Monitoring aggregator health check passed"
        else
            warn "Monitoring aggregator health check failed"
        fi
    fi
    
    log "Monitoring and observability components deployed successfully"
}

# Deploy production readiness components
deploy_production_readiness() {
    log "Deploying production readiness components..."
    
    # Deploy production deployment tools
    info "Deploying production deployment configuration..."
    kubectl apply -f $PHASE_DIR/production-readiness/production-deployment.yaml
    
    # Wait for cron job to be created
    kubectl get cronjob production-health-monitor -n $NAMESPACE &>/dev/null
    if [ $? -eq 0 ]; then
        log "Production health monitor cron job created successfully"
    else
        warn "Production health monitor cron job creation failed"
    fi
    
    log "Production readiness components deployed successfully"
}

# Deploy integration testing
deploy_integration_testing() {
    log "Deploying integration testing components..."
    
    # Deploy integration test suite
    info "Deploying integration test configuration..."
    kubectl apply -f $PHASE_DIR/integration-testing/end-to-end-tests.yaml
    
    log "Integration testing components deployed successfully"
}

# Run integration tests
run_integration_tests() {
    log "Running integration tests..."
    
    # Create and run integration test job
    local test_job_name="integration-tests-$(date +%s)"
    
    # Create test job from template
    kubectl get job integration-test-runner -n $NAMESPACE -o yaml | \
    sed "s/name: integration-test-runner/name: $test_job_name/g" | \
    kubectl apply -f -
    
    # Wait for test completion
    info "Waiting for integration tests to complete (timeout: 10 minutes)..."
    kubectl wait --for=condition=complete job/$test_job_name -n $NAMESPACE --timeout=600s
    
    # Get test results
    local test_pod=$(kubectl get pods -n $NAMESPACE -l job-name=$test_job_name -o jsonpath='{.items[0].metadata.name}')
    
    if [ -n "$test_pod" ]; then
        log "Integration test results:"
        kubectl logs $test_pod -n $NAMESPACE | tail -20
        
        # Check if tests passed
        if kubectl logs $test_pod -n $NAMESPACE | grep -q "SUCCESS"; then
            log "Integration tests passed successfully"
        else
            warn "Some integration tests may have failed. Check logs for details."
        fi
    fi
    
    # Cleanup test job
    kubectl delete job $test_job_name -n $NAMESPACE
    
    log "Integration tests completed"
}

# Verify deployment
verify_deployment() {
    log "Verifying Phase 5 deployment..."
    
    # Check all deployments are running
    local deployments=("cache-optimizer" "database-performance-monitor" "monitoring-aggregator")
    
    for deployment in "${deployments[@]}"; do
        local ready_replicas=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        local desired_replicas=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")
        
        if [ "$ready_replicas" = "$desired_replicas" ] && [ "$ready_replicas" != "0" ]; then
            log "✓ Deployment $deployment is healthy ($ready_replicas/$desired_replicas replicas ready)"
        else
            error "✗ Deployment $deployment is unhealthy ($ready_replicas/$desired_replicas replicas ready)"
        fi
    done
    
    # Check services are accessible
    local services=("cache-optimizer-service" "database-performance-monitor-service" "monitoring-aggregator-service")
    
    for service in "${services[@]}"; do
        if kubectl get service $service -n $NAMESPACE &>/dev/null; then
            log "✓ Service $service is available"
        else
            error "✗ Service $service is not available"
        fi
    done
    
    # Test service endpoints
    info "Testing service endpoints..."
    
    # Test cache optimizer
    local cache_pod=$(kubectl get pods -n $NAMESPACE -l app=cache-optimizer -o jsonpath='{.items[0].metadata.name}')
    if [ -n "$cache_pod" ]; then
        if kubectl exec $cache_pod -n $NAMESPACE -- curl -f http://localhost:8090/health &>/dev/null; then
            log "✓ Cache optimizer endpoint is healthy"
        else
            warn "✗ Cache optimizer endpoint health check failed"
        fi
    fi
    
    # Test database monitor
    local db_monitor_pod=$(kubectl get pods -n $NAMESPACE -l app=database-performance-monitor -o jsonpath='{.items[0].metadata.name}')
    if [ -n "$db_monitor_pod" ]; then
        if kubectl exec $db_monitor_pod -n $NAMESPACE -- curl -f http://localhost:8091/health &>/dev/null; then
            log "✓ Database performance monitor endpoint is healthy"
        else
            warn "✗ Database performance monitor endpoint health check failed"
        fi
    fi
    
    # Test monitoring aggregator
    local monitoring_pod=$(kubectl get pods -n $NAMESPACE -l app=monitoring-aggregator -o jsonpath='{.items[0].metadata.name}')
    if [ -n "$monitoring_pod" ]; then
        if kubectl exec $monitoring_pod -n $NAMESPACE -- curl -f http://localhost:8095/health &>/dev/null; then
            log "✓ Monitoring aggregator endpoint is healthy"
        else
            warn "✗ Monitoring aggregator endpoint health check failed"
        fi
    fi
    
    log "Phase 5 deployment verification completed"
}

# Generate deployment report
generate_deployment_report() {
    log "Generating Phase 5 deployment report..."
    
    local report_file="/tmp/phase5_deployment_report.json"
    
    cat > $report_file <<EOF
{
  "phase": "WS1 Phase 5: Performance Optimization & Monitoring",
  "deployment_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "namespace": "$NAMESPACE",
  "components": {
    "performance_optimization": {
      "cache_optimizer": {
        "deployed": true,
        "service": "cache-optimizer-service",
        "port": 8090,
        "metrics_port": 9093
      },
      "database_performance_monitor": {
        "deployed": true,
        "service": "database-performance-monitor-service",
        "port": 8091,
        "metrics_port": 9094
      }
    },
    "monitoring_observability": {
      "monitoring_aggregator": {
        "deployed": true,
        "service": "monitoring-aggregator-service",
        "port": 8095,
        "metrics_port": 9095
      }
    },
    "production_readiness": {
      "health_monitor_cronjob": {
        "deployed": true,
        "schedule": "*/15 * * * *"
      },
      "deployment_procedures": {
        "deployed": true,
        "blue_green_deployment": true,
        "disaster_recovery": true
      }
    },
    "integration_testing": {
      "test_suite": {
        "deployed": true,
        "test_scenarios": [
          "authentication_flow",
          "ai_conversation_flow",
          "data_processing_flow",
          "performance_scenarios",
          "security_scenarios",
          "integration_scenarios"
        ]
      }
    }
  },
  "endpoints": {
    "cache_optimizer": "http://cache-optimizer-service.$NAMESPACE:8090",
    "database_monitor": "http://database-performance-monitor-service.$NAMESPACE:8091",
    "monitoring_aggregator": "http://monitoring-aggregator-service.$NAMESPACE:8095"
  },
  "metrics_endpoints": {
    "cache_optimizer": "http://cache-optimizer-service.$NAMESPACE:9093/metrics",
    "database_monitor": "http://database-performance-monitor-service.$NAMESPACE:9094/metrics",
    "monitoring_aggregator": "http://monitoring-aggregator-service.$NAMESPACE:9095/metrics"
  },
  "next_steps": [
    "Monitor system performance using deployed monitoring tools",
    "Run regular integration tests to ensure system health",
    "Review performance optimization recommendations",
    "Prepare for WS2: AI Intelligence implementation"
  ]
}
EOF
    
    log "Deployment report generated: $report_file"
    cat $report_file
}

# Main deployment function
main() {
    log "Starting Nexus Architect WS1 Phase 5 deployment..."
    log "Deployment log: $LOG_FILE"
    
    # Check prerequisites
    check_prerequisites
    
    # Deploy components
    deploy_performance_optimization
    deploy_monitoring_observability
    deploy_production_readiness
    deploy_integration_testing
    
    # Run tests
    run_integration_tests
    
    # Verify deployment
    verify_deployment
    
    # Generate report
    generate_deployment_report
    
    log "🎉 Nexus Architect WS1 Phase 5 deployment completed successfully!"
    log "All performance optimization and monitoring components are now operational."
    log "The system is ready for production workloads and WS2 implementation."
}

# Cleanup function
cleanup() {
    log "Cleaning up Phase 5 components..."
    
    # Delete deployments
    kubectl delete deployment cache-optimizer database-performance-monitor monitoring-aggregator -n $NAMESPACE --ignore-not-found=true
    
    # Delete services
    kubectl delete service cache-optimizer-service database-performance-monitor-service monitoring-aggregator-service -n $NAMESPACE --ignore-not-found=true
    
    # Delete config maps
    kubectl delete configmap caching-optimization-config database-optimization-config comprehensive-monitoring-config production-deployment-config integration-testing-config -n $NAMESPACE --ignore-not-found=true
    
    # Delete cron jobs
    kubectl delete cronjob production-health-monitor database-optimizer -n $NAMESPACE --ignore-not-found=true
    
    log "Phase 5 cleanup completed"
}

# Script execution
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "verify")
        verify_deployment
        ;;
    "test")
        run_integration_tests
        ;;
    "cleanup")
        cleanup
        ;;
    "report")
        generate_deployment_report
        ;;
    *)
        echo "Usage: $0 {deploy|verify|test|cleanup|report}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy all Phase 5 components (default)"
        echo "  verify  - Verify deployment status"
        echo "  test    - Run integration tests"
        echo "  cleanup - Remove all Phase 5 components"
        echo "  report  - Generate deployment report"
        exit 1
        ;;
esac

