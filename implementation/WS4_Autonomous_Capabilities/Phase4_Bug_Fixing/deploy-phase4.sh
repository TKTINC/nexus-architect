#!/bin/bash

# WS4 Phase 4: Autonomous Bug Fixing & Ticket Resolution
# Deployment Script

set -e

echo "🚀 Deploying WS4 Phase 4: Autonomous Bug Fixing & Ticket Resolution"

# Configuration
NAMESPACE="nexus-architect"
PHASE_DIR="implementation/WS4_Autonomous_Capabilities/Phase4_Bug_Fixing"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

echo "📦 Building and deploying Bug Analysis Engine..."

# Bug Analysis Engine Deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bug-analyzer
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: bug-analyzer
  template:
    metadata:
      labels:
        app: bug-analyzer
    spec:
      containers:
      - name: bug-analyzer
        image: nexus-architect/bug-analyzer:latest
        ports:
        - containerPort: 8050
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          value: "postgresql://postgres:password@postgres-service:5432/nexus_architect"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: bug-analyzer-service
  namespace: $NAMESPACE
spec:
  selector:
    app: bug-analyzer
  ports:
  - port: 8050
    targetPort: 8050
  type: ClusterIP
EOF

echo "🔧 Deploying Fix Generation Engine..."

# Fix Generation Engine Deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fix-generator
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fix-generator
  template:
    metadata:
      labels:
        app: fix-generator
    spec:
      containers:
      - name: fix-generator
        image: nexus-architect/fix-generator:latest
        ports:
        - containerPort: 8051
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          value: "postgresql://postgres:password@postgres-service:5432/nexus_architect"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: fix-generator-service
  namespace: $NAMESPACE
spec:
  selector:
    app: fix-generator
  ports:
  - port: 8051
    targetPort: 8051
  type: ClusterIP
EOF

echo "🔄 Deploying Workflow Manager..."

# Workflow Manager Deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: workflow-manager
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: workflow-manager
  template:
    metadata:
      labels:
        app: workflow-manager
    spec:
      containers:
      - name: workflow-manager
        image: nexus-architect/workflow-manager:latest
        ports:
        - containerPort: 8052
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          value: "postgresql://postgres:password@postgres-service:5432/nexus_architect"
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-service:9092"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: workflow-manager-service
  namespace: $NAMESPACE
spec:
  selector:
    app: workflow-manager
  ports:
  - port: 8052
    targetPort: 8052
  type: ClusterIP
EOF

echo "📊 Deploying Success Tracker..."

# Success Tracker Deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: success-tracker
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: success-tracker
  template:
    metadata:
      labels:
        app: success-tracker
    spec:
      containers:
      - name: success-tracker
        image: nexus-architect/success-tracker:latest
        ports:
        - containerPort: 8053
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          value: "postgresql://postgres:password@postgres-service:5432/nexus_architect"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: success-tracker-service
  namespace: $NAMESPACE
spec:
  selector:
    app: success-tracker
  ports:
  - port: 8053
    targetPort: 8053
  type: ClusterIP
EOF

echo "🔐 Creating AI Secrets..."

# Create AI secrets if they don't exist
kubectl create secret generic ai-secrets \
  --from-literal=openai-api-key="${OPENAI_API_KEY:-dummy-key}" \
  --from-literal=anthropic-api-key="${ANTHROPIC_API_KEY:-dummy-key}" \
  --namespace=$NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

echo "📈 Setting up monitoring..."

# ServiceMonitor for Prometheus
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: bug-fixing-metrics
  namespace: $NAMESPACE
spec:
  selector:
    matchLabels:
      monitoring: enabled
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF

echo "🔍 Configuring health checks..."

# Health check endpoints
for service in bug-analyzer fix-generator workflow-manager success-tracker; do
  kubectl patch deployment $service -n $NAMESPACE -p '{
    "spec": {
      "template": {
        "spec": {
          "containers": [{
            "name": "'$service'",
            "livenessProbe": {
              "httpGet": {
                "path": "/health",
                "port": 8050
              },
              "initialDelaySeconds": 30,
              "periodSeconds": 10
            },
            "readinessProbe": {
              "httpGet": {
                "path": "/ready",
                "port": 8050
              },
              "initialDelaySeconds": 5,
              "periodSeconds": 5
            }
          }]
        }
      }
    }
  }'
done

echo "🌐 Setting up ingress..."

# Ingress configuration
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bug-fixing-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - bug-fixing.nexus-architect.local
    secretName: nexus-tls
  rules:
  - host: bug-fixing.nexus-architect.local
    http:
      paths:
      - path: /analyzer
        pathType: Prefix
        backend:
          service:
            name: bug-analyzer-service
            port:
              number: 8050
      - path: /generator
        pathType: Prefix
        backend:
          service:
            name: fix-generator-service
            port:
              number: 8051
      - path: /workflow
        pathType: Prefix
        backend:
          service:
            name: workflow-manager-service
            port:
              number: 8052
      - path: /tracker
        pathType: Prefix
        backend:
          service:
            name: success-tracker-service
            port:
              number: 8053
EOF

echo "⚙️ Configuring auto-scaling..."

# Horizontal Pod Autoscaler
for service in bug-analyzer fix-generator workflow-manager; do
  cat <<EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${service}-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $service
  minReplicas: 2
  maxReplicas: 10
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
done

echo "🔄 Waiting for deployments to be ready..."

# Wait for deployments
kubectl wait --for=condition=available --timeout=300s deployment/bug-analyzer -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/fix-generator -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/workflow-manager -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/success-tracker -n $NAMESPACE

echo "✅ Verifying deployment..."

# Check pod status
kubectl get pods -n $NAMESPACE -l app=bug-analyzer
kubectl get pods -n $NAMESPACE -l app=fix-generator
kubectl get pods -n $NAMESPACE -l app=workflow-manager
kubectl get pods -n $NAMESPACE -l app=success-tracker

# Check services
kubectl get services -n $NAMESPACE

echo "🎉 WS4 Phase 4 deployment completed successfully!"
echo ""
echo "📋 Deployment Summary:"
echo "- Bug Analyzer: http://bug-fixing.nexus-architect.local/analyzer"
echo "- Fix Generator: http://bug-fixing.nexus-architect.local/generator"
echo "- Workflow Manager: http://bug-fixing.nexus-architect.local/workflow"
echo "- Success Tracker: http://bug-fixing.nexus-architect.local/tracker"
echo ""
echo "🔧 Next Steps:"
echo "1. Configure ticket system integrations (Jira, Linear, etc.)"
echo "2. Set up CI/CD pipeline integrations"
echo "3. Configure notification channels"
echo "4. Run integration tests"
echo "5. Monitor performance metrics"
echo ""
echo "📊 Monitoring:"
echo "- Prometheus metrics: /metrics endpoint on each service"
echo "- Health checks: /health and /ready endpoints"
echo "- Logs: kubectl logs -f deployment/<service-name> -n $NAMESPACE"

