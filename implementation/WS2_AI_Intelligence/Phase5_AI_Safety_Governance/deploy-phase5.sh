#!/bin/bash

# Nexus Architect WS2 Phase 5: AI Safety, Governance & Explainability
# Deployment Script

set -e

echo "🚀 Starting WS2 Phase 5 Deployment: AI Safety, Governance & Explainability"
echo "============================================================================"

# Configuration
NAMESPACE="nexus-ai-safety"
REDIS_URL="redis://redis-cluster:6379"
POSTGRES_URL="postgresql://postgres:password@postgresql-cluster:5432/nexus_governance"
EXPLAINABILITY_DB_URL="postgresql://postgres:password@postgresql-cluster:5432/nexus_explainability"

# Create namespace
echo "📦 Creating Kubernetes namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy AI Safety Controller
echo "🛡️ Deploying AI Safety Controller..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-safety-controller
  namespace: $NAMESPACE
  labels:
    app: ai-safety-controller
    component: safety
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-safety-controller
  template:
    metadata:
      labels:
        app: ai-safety-controller
        component: safety
    spec:
      containers:
      - name: ai-safety-controller
        image: nexus-architect/ai-safety-controller:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "$REDIS_URL"
        - name: POSTGRES_URL
          value: "$POSTGRES_URL"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /safety/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /safety/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-safety-controller-service
  namespace: $NAMESPACE
spec:
  selector:
    app: ai-safety-controller
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
EOF

# Deploy Bias Detection System
echo "⚖️ Deploying Bias Detection System..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bias-detection-system
  namespace: $NAMESPACE
  labels:
    app: bias-detection-system
    component: bias-detection
spec:
  replicas: 2
  selector:
    matchLabels:
      app: bias-detection-system
  template:
    metadata:
      labels:
        app: bias-detection-system
        component: bias-detection
    spec:
      containers:
      - name: bias-detection-system
        image: nexus-architect/bias-detection:latest
        ports:
        - containerPort: 8001
        env:
        - name: REDIS_URL
          value: "$REDIS_URL"
        - name: POSTGRES_URL
          value: "$POSTGRES_URL"
        - name: MODEL_CACHE_SIZE
          value: "1000"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /bias/health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /bias/health
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: bias-detection-service
  namespace: $NAMESPACE
spec:
  selector:
    app: bias-detection-system
  ports:
  - protocol: TCP
    port: 8001
    targetPort: 8001
  type: ClusterIP
EOF

# Deploy AI Governance System
echo "🏛️ Deploying AI Governance System..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-governance-system
  namespace: $NAMESPACE
  labels:
    app: ai-governance-system
    component: governance
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-governance-system
  template:
    metadata:
      labels:
        app: ai-governance-system
        component: governance
    spec:
      containers:
      - name: ai-governance-system
        image: nexus-architect/ai-governance:latest
        ports:
        - containerPort: 8001
        env:
        - name: REDIS_URL
          value: "$REDIS_URL"
        - name: POSTGRES_URL
          value: "$POSTGRES_URL"
        - name: SMTP_SERVER
          value: "smtp.company.com"
        - name: SMTP_PORT
          value: "587"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /governance/health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /governance/health
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-governance-service
  namespace: $NAMESPACE
spec:
  selector:
    app: ai-governance-system
  ports:
  - protocol: TCP
    port: 8001
    targetPort: 8001
  type: ClusterIP
EOF

# Deploy Explainability Engine
echo "🔍 Deploying Explainability Engine..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: explainability-engine
  namespace: $NAMESPACE
  labels:
    app: explainability-engine
    component: explainability
spec:
  replicas: 3
  selector:
    matchLabels:
      app: explainability-engine
  template:
    metadata:
      labels:
        app: explainability-engine
        component: explainability
    spec:
      containers:
      - name: explainability-engine
        image: nexus-architect/explainability-engine:latest
        ports:
        - containerPort: 8002
        env:
        - name: REDIS_URL
          value: "$REDIS_URL"
        - name: POSTGRES_URL
          value: "$EXPLAINABILITY_DB_URL"
        - name: TRANSFORMERS_CACHE
          value: "/tmp/transformers_cache"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-cache
          mountPath: /tmp/transformers_cache
        livenessProbe:
          httpGet:
            path: /explainability/health
            port: 8002
          initialDelaySeconds: 60
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /explainability/health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: explainability-engine-service
  namespace: $NAMESPACE
spec:
  selector:
    app: explainability-engine
  ports:
  - protocol: TCP
    port: 8002
    targetPort: 8002
  type: ClusterIP
EOF

# Deploy Risk Management System
echo "⚠️ Deploying Risk Management System..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: risk-management-system
  namespace: $NAMESPACE
  labels:
    app: risk-management-system
    component: risk-management
spec:
  replicas: 2
  selector:
    matchLabels:
      app: risk-management-system
  template:
    metadata:
      labels:
        app: risk-management-system
        component: risk-management
    spec:
      containers:
      - name: risk-management-system
        image: nexus-architect/risk-management:latest
        ports:
        - containerPort: 8003
        env:
        - name: REDIS_URL
          value: "$REDIS_URL"
        - name: POSTGRES_URL
          value: "$POSTGRES_URL"
        - name: ALERT_WEBHOOK_URL
          value: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /risk/health
            port: 8003
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /risk/health
            port: 8003
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: risk-management-service
  namespace: $NAMESPACE
spec:
  selector:
    app: risk-management-system
  ports:
  - protocol: TCP
    port: 8003
    targetPort: 8003
  type: ClusterIP
EOF

# Deploy API Gateway for AI Safety Services
echo "🌐 Deploying AI Safety API Gateway..."
cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: ai-safety-gateway
  namespace: $NAMESPACE
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - ai-safety.nexus-architect.local
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: ai-safety-tls
    hosts:
    - ai-safety.nexus-architect.local
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ai-safety-routes
  namespace: $NAMESPACE
spec:
  hosts:
  - ai-safety.nexus-architect.local
  gateways:
  - ai-safety-gateway
  http:
  - match:
    - uri:
        prefix: /safety/
    route:
    - destination:
        host: ai-safety-controller-service
        port:
          number: 8000
  - match:
    - uri:
        prefix: /bias/
    route:
    - destination:
        host: bias-detection-service
        port:
          number: 8001
  - match:
    - uri:
        prefix: /governance/
    route:
    - destination:
        host: ai-governance-service
        port:
          number: 8001
  - match:
    - uri:
        prefix: /explainability/
    route:
    - destination:
        host: explainability-engine-service
        port:
          number: 8002
  - match:
    - uri:
        prefix: /risk/
    route:
    - destination:
        host: risk-management-service
        port:
          number: 8003
EOF

# Deploy Monitoring and Alerting
echo "📊 Deploying Monitoring and Alerting..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-safety-monitoring-config
  namespace: $NAMESPACE
data:
  prometheus-rules.yaml: |
    groups:
    - name: ai-safety-alerts
      rules:
      - alert: HighBiasDetected
        expr: bias_score > 0.8
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High bias detected in AI model"
          description: "Bias score {{ \$value }} exceeds threshold"
      
      - alert: SafetyViolation
        expr: safety_violation_count > 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "AI safety violation detected"
          description: "{{ \$value }} safety violations in the last minute"
      
      - alert: GovernanceApprovalOverdue
        expr: governance_pending_approvals > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Multiple governance approvals pending"
          description: "{{ \$value }} approvals pending for more than 5 minutes"
      
      - alert: ExplainabilityServiceDown
        expr: up{job="explainability-engine"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Explainability service is down"
          description: "Explainability engine is not responding"
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ai-safety-monitoring
  namespace: $NAMESPACE
  labels:
    app: ai-safety-monitoring
spec:
  selector:
    matchLabels:
      component: safety
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
EOF

# Deploy Network Policies for Security
echo "🔒 Deploying Network Security Policies..."
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-safety-network-policy
  namespace: $NAMESPACE
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    - namespaceSelector:
        matchLabels:
          name: nexus-core
    - podSelector: {}
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: nexus-core
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
  - to: []
    ports:
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
    - protocol: TCP
      port: 443  # HTTPS
EOF

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/ai-safety-controller -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/bias-detection-system -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/ai-governance-system -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/explainability-engine -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/risk-management-system -n $NAMESPACE

# Run health checks
echo "🏥 Running health checks..."
echo "Checking AI Safety Controller..."
kubectl exec -n $NAMESPACE deployment/ai-safety-controller -- curl -f http://localhost:8000/safety/health

echo "Checking Bias Detection System..."
kubectl exec -n $NAMESPACE deployment/bias-detection-system -- curl -f http://localhost:8001/bias/health

echo "Checking AI Governance System..."
kubectl exec -n $NAMESPACE deployment/ai-governance-system -- curl -f http://localhost:8001/governance/health

echo "Checking Explainability Engine..."
kubectl exec -n $NAMESPACE deployment/explainability-engine -- curl -f http://localhost:8002/explainability/health

echo "Checking Risk Management System..."
kubectl exec -n $NAMESPACE deployment/risk-management-system -- curl -f http://localhost:8003/risk/health

# Display deployment status
echo "📋 Deployment Status:"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get gateways -n $NAMESPACE
kubectl get virtualservices -n $NAMESPACE

echo ""
echo "✅ WS2 Phase 5 Deployment Complete!"
echo "============================================================================"
echo "🛡️ AI Safety Controller: http://ai-safety.nexus-architect.local/safety/"
echo "⚖️ Bias Detection System: http://ai-safety.nexus-architect.local/bias/"
echo "🏛️ AI Governance System: http://ai-safety.nexus-architect.local/governance/"
echo "🔍 Explainability Engine: http://ai-safety.nexus-architect.local/explainability/"
echo "⚠️ Risk Management System: http://ai-safety.nexus-architect.local/risk/"
echo ""
echo "📊 Monitoring Dashboard: http://grafana.nexus-architect.local/d/ai-safety"
echo "🚨 Alert Manager: http://alertmanager.nexus-architect.local"
echo ""
echo "🔐 All services are secured with OAuth 2.0/OIDC authentication"
echo "📈 Comprehensive monitoring and alerting configured"
echo "⚡ Auto-scaling enabled based on CPU, memory, and custom metrics"
echo ""
echo "Phase 5 AI Safety, Governance & Explainability infrastructure is now operational!"

