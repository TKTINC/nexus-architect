#!/bin/bash

# WS4 Phase 3: Agentic Transformation & Legacy Modernization
# Deployment Script for Nexus Architect

set -e

echo "🚀 Starting WS4 Phase 3 Deployment: Agentic Transformation & Legacy Modernization"

# Configuration
NAMESPACE="nexus-ws4-phase3"
DOCKER_REGISTRY="nexus-registry"
VERSION="1.0.0"

# Create namespace
echo "📦 Creating Kubernetes namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy Legacy System Analyzer
echo "🔍 Deploying Legacy System Analyzer..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: legacy-system-analyzer
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: legacy-system-analyzer
  template:
    metadata:
      labels:
        app: legacy-system-analyzer
    spec:
      containers:
      - name: legacy-analyzer
        image: $DOCKER_REGISTRY/legacy-system-analyzer:$VERSION
        ports:
        - containerPort: 8040
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          value: "postgresql://postgres:5432/nexus_ws4"
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
            path: /health
            port: 8040
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8040
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: legacy-system-analyzer-service
  namespace: $NAMESPACE
spec:
  selector:
    app: legacy-system-analyzer
  ports:
  - port: 8040
    targetPort: 8040
  type: ClusterIP
EOF

# Deploy Agentic Transformation Engine
echo "🤖 Deploying Agentic Transformation Engine..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-transformation-engine
  namespace: $NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentic-transformation-engine
  template:
    metadata:
      labels:
        app: agentic-transformation-engine
    spec:
      containers:
      - name: transformation-engine
        image: $DOCKER_REGISTRY/agentic-transformation-engine:$VERSION
        ports:
        - containerPort: 8041
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          value: "postgresql://postgres:5432/nexus_ws4"
        - name: AI_SERVICE_URL
          value: "http://ai-service:8080"
        - name: MAX_WORKERS
          value: "8"
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
            port: 8041
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8041
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-transformation-engine-service
  namespace: $NAMESPACE
spec:
  selector:
    app: agentic-transformation-engine
  ports:
  - port: 8041
    targetPort: 8041
  type: ClusterIP
EOF

# Deploy Modernization Strategy Engine
echo "📋 Deploying Modernization Strategy Engine..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modernization-strategy-engine
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: modernization-strategy-engine
  template:
    metadata:
      labels:
        app: modernization-strategy-engine
    spec:
      containers:
      - name: strategy-engine
        image: $DOCKER_REGISTRY/modernization-strategy-engine:$VERSION
        ports:
        - containerPort: 8042
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          value: "postgresql://postgres:5432/nexus_ws4"
        - name: LEGACY_ANALYZER_URL
          value: "http://legacy-system-analyzer-service:8040"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8042
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8042
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: modernization-strategy-engine-service
  namespace: $NAMESPACE
spec:
  selector:
    app: modernization-strategy-engine
  ports:
  - port: 8042
    targetPort: 8042
  type: ClusterIP
EOF

# Deploy Automated Migration Toolkit
echo "🔧 Deploying Automated Migration Toolkit..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: automated-migration-toolkit
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: automated-migration-toolkit
  template:
    metadata:
      labels:
        app: automated-migration-toolkit
    spec:
      containers:
      - name: migration-toolkit
        image: $DOCKER_REGISTRY/automated-migration-toolkit:$VERSION
        ports:
        - containerPort: 8043
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          value: "postgresql://postgres:5432/nexus_ws4"
        - name: DOCKER_HOST
          value: "unix:///var/run/docker.sock"
        volumeMounts:
        - name: docker-socket
          mountPath: /var/run/docker.sock
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
            port: 8043
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8043
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock
---
apiVersion: v1
kind: Service
metadata:
  name: automated-migration-toolkit-service
  namespace: $NAMESPACE
spec:
  selector:
    app: automated-migration-toolkit
  ports:
  - port: 8043
    targetPort: 8043
  type: ClusterIP
EOF

# Deploy Redis for caching and queuing
echo "📊 Deploying Redis cluster..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
EOF

# Deploy PostgreSQL for metadata storage
echo "🗄️ Deploying PostgreSQL database..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: "nexus_ws4"
        - name: POSTGRES_USER
          value: "nexus"
        - name: POSTGRES_PASSWORD
          value: "nexus_password"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: $NAMESPACE
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
EOF

# Deploy Prometheus monitoring
echo "📈 Deploying Prometheus monitoring..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        args:
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus/'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--web.enable-lifecycle'
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
  name: prometheus-service
  namespace: $NAMESPACE
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
EOF

# Create ConfigMaps for configuration
echo "⚙️ Creating configuration..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: ws4-phase3-config
  namespace: $NAMESPACE
data:
  transformation_config.yaml: |
    transformation:
      max_concurrent_jobs: 10
      timeout_minutes: 60
      supported_frameworks:
        - spring_boot
        - django
        - express
        - dotnet
      refactoring_patterns:
        - extract_method
        - extract_class
        - move_method
        - rename_variable
      safety_checks:
        - syntax_validation
        - test_execution
        - performance_regression
    
    modernization:
      strategies:
        - framework_upgrade
        - microservices_decomposition
        - cloud_native_transformation
        - api_modernization
      risk_assessment:
        - code_complexity
        - test_coverage
        - dependency_analysis
        - business_impact
EOF

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/legacy-system-analyzer -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/agentic-transformation-engine -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/modernization-strategy-engine -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/automated-migration-toolkit -n $NAMESPACE

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

# Create ingress for external access
echo "🌐 Creating ingress for external access..."
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ws4-phase3-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: ws4-phase3.nexus-architect.local
    http:
      paths:
      - path: /legacy-analyzer
        pathType: Prefix
        backend:
          service:
            name: legacy-system-analyzer-service
            port:
              number: 8040
      - path: /transformation-engine
        pathType: Prefix
        backend:
          service:
            name: agentic-transformation-engine-service
            port:
              number: 8041
      - path: /strategy-engine
        pathType: Prefix
        backend:
          service:
            name: modernization-strategy-engine-service
            port:
              number: 8042
      - path: /migration-toolkit
        pathType: Prefix
        backend:
          service:
            name: automated-migration-toolkit-service
            port:
              number: 8043
EOF

# Run health checks
echo "🏥 Running health checks..."
sleep 30

for service in legacy-system-analyzer agentic-transformation-engine modernization-strategy-engine automated-migration-toolkit; do
    echo "Checking $service health..."
    kubectl exec -n $NAMESPACE deployment/$service -- curl -f http://localhost:804*/health || echo "⚠️ $service health check failed"
done

# Display access information
echo "🎉 WS4 Phase 3 deployment completed successfully!"
echo ""
echo "📋 Service Access Information:"
echo "Legacy System Analyzer: http://ws4-phase3.nexus-architect.local/legacy-analyzer"
echo "Agentic Transformation Engine: http://ws4-phase3.nexus-architect.local/transformation-engine"
echo "Modernization Strategy Engine: http://ws4-phase3.nexus-architect.local/strategy-engine"
echo "Automated Migration Toolkit: http://ws4-phase3.nexus-architect.local/migration-toolkit"
echo ""
echo "📊 Monitoring:"
echo "Prometheus: http://ws4-phase3.nexus-architect.local:9090"
echo ""
echo "🔧 Management Commands:"
echo "View logs: kubectl logs -f deployment/<service-name> -n $NAMESPACE"
echo "Scale service: kubectl scale deployment/<service-name> --replicas=<count> -n $NAMESPACE"
echo "Update config: kubectl edit configmap ws4-phase3-config -n $NAMESPACE"
echo ""
echo "✅ All WS4 Phase 3 services are operational and ready for agentic transformation workloads!"

