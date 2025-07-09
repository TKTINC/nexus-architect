#!/bin/bash

# Nexus Architect WS2 Phase 3: Advanced Conversational AI & Context Management
# Deployment Script

set -e

echo "🚀 Starting WS2 Phase 3 Deployment: Advanced Conversational AI & Context Management"
echo "=============================================================================="

# Configuration
NAMESPACE="nexus-ai-intelligence"
REDIS_NAMESPACE="nexus-infrastructure"
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-"development"}

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
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
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
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check if required namespaces exist
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log "Creating namespace $NAMESPACE..."
        kubectl create namespace $NAMESPACE
    fi
    
    # Check if Redis is available
    if ! kubectl get service redis-service -n $REDIS_NAMESPACE &> /dev/null; then
        warning "Redis service not found in $REDIS_NAMESPACE namespace"
        warning "Conversational AI requires Redis for context management"
    fi
    
    success "Prerequisites check completed"
}

# Deploy context management system
deploy_context_management() {
    log "Deploying Context Management System..."
    
    # Create ConfigMap for context management configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: context-management-config
  namespace: $NAMESPACE
data:
  redis_url: "redis://redis-service.$REDIS_NAMESPACE.svc.cluster.local:6379"
  session_timeout: "86400"  # 24 hours
  max_context_window: "8000"
  overlap_tokens: "500"
  cleanup_interval: "3600"  # 1 hour
EOF

    # Create Secret for Redis authentication (if needed)
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: context-management-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  redis_password: ""  # Base64 encoded password if Redis requires auth
EOF

    # Deploy Context Management Service
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: context-management-service
  namespace: $NAMESPACE
  labels:
    app: context-management
    component: conversational-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: context-management
  template:
    metadata:
      labels:
        app: context-management
        component: conversational-ai
    spec:
      containers:
      - name: context-manager
        image: nexus-architect/context-management:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: context-management-config
              key: redis_url
        - name: SESSION_TIMEOUT
          valueFrom:
            configMapKeyRef:
              name: context-management-config
              key: session_timeout
        - name: MAX_CONTEXT_WINDOW
          valueFrom:
            configMapKeyRef:
              name: context-management-config
              key: max_context_window
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
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
---
apiVersion: v1
kind: Service
metadata:
  name: context-management-service
  namespace: $NAMESPACE
  labels:
    app: context-management
spec:
  selector:
    app: context-management
  ports:
  - port: 8080
    targetPort: 8080
    name: http
  type: ClusterIP
EOF

    success "Context Management System deployed"
}

# Deploy NLU processing system
deploy_nlu_processing() {
    log "Deploying Natural Language Understanding System..."
    
    # Create ConfigMap for NLU configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: nlu-processing-config
  namespace: $NAMESPACE
data:
  spacy_model: "en_core_web_sm"
  sentiment_model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  emotion_model: "j-hartmann/emotion-english-distilroberta-base"
  intent_confidence_threshold: "0.7"
  entity_confidence_threshold: "0.8"
  max_batch_size: "32"
EOF

    # Deploy NLU Processing Service
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlu-processing-service
  namespace: $NAMESPACE
  labels:
    app: nlu-processing
    component: conversational-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nlu-processing
  template:
    metadata:
      labels:
        app: nlu-processing
        component: conversational-ai
    spec:
      containers:
      - name: nlu-processor
        image: nexus-architect/nlu-processing:latest
        ports:
        - containerPort: 8081
        env:
        - name: SPACY_MODEL
          valueFrom:
            configMapKeyRef:
              name: nlu-processing-config
              key: spacy_model
        - name: SENTIMENT_MODEL
          valueFrom:
            configMapKeyRef:
              name: nlu-processing-config
              key: sentiment_model
        - name: EMOTION_MODEL
          valueFrom:
            configMapKeyRef:
              name: nlu-processing-config
              key: emotion_model
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
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: nlu-processing-service
  namespace: $NAMESPACE
  labels:
    app: nlu-processing
spec:
  selector:
    app: nlu-processing
  ports:
  - port: 8081
    targetPort: 8081
    name: http
  type: ClusterIP
EOF

    success "NLU Processing System deployed"
}

# Deploy role adaptation system
deploy_role_adaptation() {
    log "Deploying Role Adaptation System..."
    
    # Create ConfigMap for role adaptation configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: role-adaptation-config
  namespace: $NAMESPACE
data:
  supported_roles: "executive,developer,project_manager,product_leader,architect,devops_engineer,security_engineer"
  expertise_levels: "beginner,intermediate,advanced,expert"
  communication_styles: "formal,casual,technical,business,educational"
  template_cache_size: "1000"
  adaptation_cache_ttl: "3600"
EOF

    # Deploy Role Adaptation Service
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: role-adaptation-service
  namespace: $NAMESPACE
  labels:
    app: role-adaptation
    component: conversational-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: role-adaptation
  template:
    metadata:
      labels:
        app: role-adaptation
        component: conversational-ai
    spec:
      containers:
      - name: role-adapter
        image: nexus-architect/role-adaptation:latest
        ports:
        - containerPort: 8082
        env:
        - name: SUPPORTED_ROLES
          valueFrom:
            configMapKeyRef:
              name: role-adaptation-config
              key: supported_roles
        - name: EXPERTISE_LEVELS
          valueFrom:
            configMapKeyRef:
              name: role-adaptation-config
              key: expertise_levels
        - name: COMMUNICATION_STYLES
          valueFrom:
            configMapKeyRef:
              name: role-adaptation-config
              key: communication_styles
        resources:
          requests:
            memory: "512Mi"
            cpu: "300m"
          limits:
            memory: "1Gi"
            cpu: "600m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8082
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8082
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: role-adaptation-service
  namespace: $NAMESPACE
  labels:
    app: role-adaptation
spec:
  selector:
    app: role-adaptation
  ports:
  - port: 8082
    targetPort: 8082
    name: http
  type: ClusterIP
EOF

    success "Role Adaptation System deployed"
}

# Deploy quality monitoring system
deploy_quality_monitoring() {
    log "Deploying Quality Monitoring System..."
    
    # Create ConfigMap for quality monitoring configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: quality-monitoring-config
  namespace: $NAMESPACE
data:
  redis_url: "redis://redis-service.$REDIS_NAMESPACE.svc.cluster.local:6379"
  relevance_threshold: "0.6"
  coherence_threshold: "0.7"
  helpfulness_threshold: "0.6"
  accuracy_threshold: "0.8"
  satisfaction_threshold: "0.7"
  alert_webhook_url: ""
  metrics_retention_days: "7"
EOF

    # Deploy Quality Monitoring Service
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quality-monitoring-service
  namespace: $NAMESPACE
  labels:
    app: quality-monitoring
    component: conversational-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: quality-monitoring
  template:
    metadata:
      labels:
        app: quality-monitoring
        component: conversational-ai
    spec:
      containers:
      - name: quality-monitor
        image: nexus-architect/quality-monitoring:latest
        ports:
        - containerPort: 8083
        env:
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: quality-monitoring-config
              key: redis_url
        - name: RELEVANCE_THRESHOLD
          valueFrom:
            configMapKeyRef:
              name: quality-monitoring-config
              key: relevance_threshold
        - name: COHERENCE_THRESHOLD
          valueFrom:
            configMapKeyRef:
              name: quality-monitoring-config
              key: coherence_threshold
        - name: HELPFULNESS_THRESHOLD
          valueFrom:
            configMapKeyRef:
              name: quality-monitoring-config
              key: helpfulness_threshold
        resources:
          requests:
            memory: "512Mi"
            cpu: "300m"
          limits:
            memory: "1Gi"
            cpu: "600m"
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
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: quality-monitoring-service
  namespace: $NAMESPACE
  labels:
    app: quality-monitoring
spec:
  selector:
    app: quality-monitoring
  ports:
  - port: 8083
    targetPort: 8083
    name: http
  type: ClusterIP
EOF

    success "Quality Monitoring System deployed"
}

# Deploy conversational AI orchestrator
deploy_conversational_orchestrator() {
    log "Deploying Conversational AI Orchestrator..."
    
    # Create ConfigMap for orchestrator configuration
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: conversational-orchestrator-config
  namespace: $NAMESPACE
data:
  context_service_url: "http://context-management-service:8080"
  nlu_service_url: "http://nlu-processing-service:8081"
  role_adaptation_service_url: "http://role-adaptation-service:8082"
  quality_monitoring_service_url: "http://quality-monitoring-service:8083"
  max_conversation_turns: "50"
  response_timeout: "30"
  enable_quality_monitoring: "true"
  enable_role_adaptation: "true"
EOF

    # Deploy Conversational AI Orchestrator
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: conversational-orchestrator
  namespace: $NAMESPACE
  labels:
    app: conversational-orchestrator
    component: conversational-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: conversational-orchestrator
  template:
    metadata:
      labels:
        app: conversational-orchestrator
        component: conversational-ai
    spec:
      containers:
      - name: orchestrator
        image: nexus-architect/conversational-orchestrator:latest
        ports:
        - containerPort: 8084
        env:
        - name: CONTEXT_SERVICE_URL
          valueFrom:
            configMapKeyRef:
              name: conversational-orchestrator-config
              key: context_service_url
        - name: NLU_SERVICE_URL
          valueFrom:
            configMapKeyRef:
              name: conversational-orchestrator-config
              key: nlu_service_url
        - name: ROLE_ADAPTATION_SERVICE_URL
          valueFrom:
            configMapKeyRef:
              name: conversational-orchestrator-config
              key: role_adaptation_service_url
        - name: QUALITY_MONITORING_SERVICE_URL
          valueFrom:
            configMapKeyRef:
              name: conversational-orchestrator-config
              key: quality_monitoring_service_url
        resources:
          requests:
            memory: "512Mi"
            cpu: "400m"
          limits:
            memory: "1Gi"
            cpu: "800m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8084
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8084
          initialDelaySeconds: 15
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: conversational-orchestrator
  namespace: $NAMESPACE
  labels:
    app: conversational-orchestrator
spec:
  selector:
    app: conversational-orchestrator
  ports:
  - port: 8084
    targetPort: 8084
    name: http
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: conversational-ai-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - conversational-ai.nexus-architect.local
    secretName: conversational-ai-tls
  rules:
  - host: conversational-ai.nexus-architect.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: conversational-orchestrator
            port:
              number: 8084
EOF

    success "Conversational AI Orchestrator deployed"
}

# Deploy monitoring and observability
deploy_monitoring() {
    log "Deploying Monitoring and Observability..."
    
    # Create ServiceMonitor for Prometheus
    kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: conversational-ai-metrics
  namespace: $NAMESPACE
  labels:
    app: conversational-ai
spec:
  selector:
    matchLabels:
      component: conversational-ai
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
EOF

    # Create Grafana Dashboard ConfigMap
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: conversational-ai-dashboard
  namespace: $NAMESPACE
  labels:
    grafana_dashboard: "1"
data:
  conversational-ai.json: |
    {
      "dashboard": {
        "title": "Nexus Architect - Conversational AI",
        "panels": [
          {
            "title": "Response Quality Scores",
            "type": "stat",
            "targets": [
              {
                "expr": "avg(conversational_ai_quality_score)",
                "legendFormat": "Average Quality"
              }
            ]
          },
          {
            "title": "Conversation Volume",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(conversational_ai_conversations_total[5m])",
                "legendFormat": "Conversations/sec"
              }
            ]
          },
          {
            "title": "Response Times",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, conversational_ai_response_duration_seconds_bucket)",
                "legendFormat": "95th percentile"
              },
              {
                "expr": "histogram_quantile(0.50, conversational_ai_response_duration_seconds_bucket)",
                "legendFormat": "50th percentile"
              }
            ]
          }
        ]
      }
    }
EOF

    success "Monitoring and Observability deployed"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check if all pods are running
    log "Checking pod status..."
    kubectl get pods -n $NAMESPACE -l component=conversational-ai
    
    # Wait for pods to be ready
    log "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l component=conversational-ai -n $NAMESPACE --timeout=300s
    
    # Check services
    log "Checking services..."
    kubectl get services -n $NAMESPACE
    
    # Test connectivity
    log "Testing service connectivity..."
    
    # Test context management service
    if kubectl run test-context --image=curlimages/curl --rm -i --restart=Never -n $NAMESPACE -- \
       curl -s http://context-management-service:8080/health > /dev/null; then
        success "Context Management Service is healthy"
    else
        error "Context Management Service health check failed"
    fi
    
    # Test NLU processing service
    if kubectl run test-nlu --image=curlimages/curl --rm -i --restart=Never -n $NAMESPACE -- \
       curl -s http://nlu-processing-service:8081/health > /dev/null; then
        success "NLU Processing Service is healthy"
    else
        error "NLU Processing Service health check failed"
    fi
    
    # Test role adaptation service
    if kubectl run test-role --image=curlimages/curl --rm -i --restart=Never -n $NAMESPACE -- \
       curl -s http://role-adaptation-service:8082/health > /dev/null; then
        success "Role Adaptation Service is healthy"
    else
        error "Role Adaptation Service health check failed"
    fi
    
    # Test quality monitoring service
    if kubectl run test-quality --image=curlimages/curl --rm -i --restart=Never -n $NAMESPACE -- \
       curl -s http://quality-monitoring-service:8083/health > /dev/null; then
        success "Quality Monitoring Service is healthy"
    else
        error "Quality Monitoring Service health check failed"
    fi
    
    # Test orchestrator
    if kubectl run test-orchestrator --image=curlimages/curl --rm -i --restart=Never -n $NAMESPACE -- \
       curl -s http://conversational-orchestrator:8084/health > /dev/null; then
        success "Conversational Orchestrator is healthy"
    else
        error "Conversational Orchestrator health check failed"
    fi
    
    success "All services are healthy and ready"
}

# Performance test
run_performance_test() {
    log "Running performance tests..."
    
    # Create a test conversation
    kubectl run perf-test --image=curlimages/curl --rm -i --restart=Never -n $NAMESPACE -- \
    curl -X POST http://conversational-orchestrator:8084/api/v1/conversations \
    -H "Content-Type: application/json" \
    -d '{
      "user_id": "test_user",
      "user_profile": {
        "role": "developer",
        "expertise_level": "intermediate",
        "communication_style": "technical"
      },
      "message": "How do I implement OAuth authentication in microservices?",
      "session_metadata": {
        "test": true
      }
    }'
    
    success "Performance test completed"
}

# Main deployment function
main() {
    log "Starting WS2 Phase 3 Deployment: Advanced Conversational AI & Context Management"
    
    check_prerequisites
    deploy_context_management
    deploy_nlu_processing
    deploy_role_adaptation
    deploy_quality_monitoring
    deploy_conversational_orchestrator
    deploy_monitoring
    verify_deployment
    
    if [[ "${RUN_PERFORMANCE_TEST:-false}" == "true" ]]; then
        run_performance_test
    fi
    
    echo ""
    echo "🎉 WS2 Phase 3 Deployment Completed Successfully!"
    echo "=============================================="
    echo ""
    echo "📊 Deployment Summary:"
    echo "- Context Management Service: ✅ Deployed"
    echo "- NLU Processing Service: ✅ Deployed"
    echo "- Role Adaptation Service: ✅ Deployed"
    echo "- Quality Monitoring Service: ✅ Deployed"
    echo "- Conversational Orchestrator: ✅ Deployed"
    echo "- Monitoring & Observability: ✅ Deployed"
    echo ""
    echo "🔗 Access Points:"
    echo "- Conversational AI API: http://conversational-ai.nexus-architect.local"
    echo "- Health Checks: http://conversational-ai.nexus-architect.local/health"
    echo "- API Documentation: http://conversational-ai.nexus-architect.local/docs"
    echo "- Metrics: http://conversational-ai.nexus-architect.local/metrics"
    echo ""
    echo "📈 Performance Targets:"
    echo "- Response Time: <2s (P95)"
    echo "- Conversation Coherence: >70%"
    echo "- Role Adaptation Accuracy: >90%"
    echo "- Quality Score: >4.0/5.0"
    echo "- System Availability: >99.9%"
    echo ""
    echo "🔧 Next Steps:"
    echo "1. Configure role-specific templates and adaptations"
    echo "2. Set up quality monitoring alerts and thresholds"
    echo "3. Train and fine-tune NLU models with domain data"
    echo "4. Integrate with existing persona AI services"
    echo "5. Conduct user acceptance testing"
    echo ""
    success "WS2 Phase 3: Advanced Conversational AI & Context Management is ready for production!"
}

# Run main function
main "$@"

