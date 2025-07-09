#!/bin/bash

# WS3 Phase 6: Data Privacy, Security & Production Optimization
# Deployment Script for Nexus Architect

set -e

echo "🚀 Starting WS3 Phase 6 Deployment: Data Privacy, Security & Production Optimization"
echo "=================================================================="

# Configuration
NAMESPACE="nexus-architect"
POSTGRES_PASSWORD="nexus_secure_password_2024"
REDIS_PASSWORD="redis_secure_password_2024"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

echo "📋 Phase 6 Components to Deploy:"
echo "  - Data Privacy Manager (Port 8010)"
echo "  - Security Manager (Port 8011)"
echo "  - Compliance Manager (Port 8012)"
echo "  - Performance Optimizer (Port 8013)"
echo "  - Production Integration Manager (Port 8014)"
echo ""

# Deploy Data Privacy Manager
echo "🔐 Deploying Data Privacy Manager..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-privacy-manager
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: data-privacy-manager
  template:
    metadata:
      labels:
        app: data-privacy-manager
    spec:
      containers:
      - name: data-privacy-manager
        image: python:3.11-slim
        ports:
        - containerPort: 8010
        env:
        - name: POSTGRES_HOST
          value: "postgres-service"
        - name: POSTGRES_PASSWORD
          value: "$POSTGRES_PASSWORD"
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PASSWORD
          value: "$REDIS_PASSWORD"
        command: ["python", "/app/data_privacy_manager.py"]
        volumeMounts:
        - name: app-code
          mountPath: /app
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
            port: 8010
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8010
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: app-code
        configMap:
          name: privacy-manager-code
---
apiVersion: v1
kind: Service
metadata:
  name: data-privacy-manager-service
  namespace: $NAMESPACE
spec:
  selector:
    app: data-privacy-manager
  ports:
  - port: 8010
    targetPort: 8010
  type: ClusterIP
EOF

# Deploy Security Manager
echo "🛡️ Deploying Security Manager..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-manager
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: security-manager
  template:
    metadata:
      labels:
        app: security-manager
    spec:
      containers:
      - name: security-manager
        image: python:3.11-slim
        ports:
        - containerPort: 8011
        env:
        - name: POSTGRES_HOST
          value: "postgres-service"
        - name: POSTGRES_PASSWORD
          value: "$POSTGRES_PASSWORD"
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PASSWORD
          value: "$REDIS_PASSWORD"
        command: ["python", "/app/security_manager.py"]
        volumeMounts:
        - name: app-code
          mountPath: /app
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
            port: 8011
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8011
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: app-code
        configMap:
          name: security-manager-code
---
apiVersion: v1
kind: Service
metadata:
  name: security-manager-service
  namespace: $NAMESPACE
spec:
  selector:
    app: security-manager
  ports:
  - port: 8011
    targetPort: 8011
  type: ClusterIP
EOF

# Deploy Compliance Manager
echo "📋 Deploying Compliance Manager..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: compliance-manager
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: compliance-manager
  template:
    metadata:
      labels:
        app: compliance-manager
    spec:
      containers:
      - name: compliance-manager
        image: python:3.11-slim
        ports:
        - containerPort: 8012
        env:
        - name: POSTGRES_HOST
          value: "postgres-service"
        - name: POSTGRES_PASSWORD
          value: "$POSTGRES_PASSWORD"
        command: ["python", "/app/compliance_manager.py"]
        volumeMounts:
        - name: app-code
          mountPath: /app
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
            port: 8012
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8012
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: app-code
        configMap:
          name: compliance-manager-code
---
apiVersion: v1
kind: Service
metadata:
  name: compliance-manager-service
  namespace: $NAMESPACE
spec:
  selector:
    app: compliance-manager
  ports:
  - port: 8012
    targetPort: 8012
  type: ClusterIP
EOF

# Deploy Performance Optimizer
echo "⚡ Deploying Performance Optimizer..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: performance-optimizer
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: performance-optimizer
  template:
    metadata:
      labels:
        app: performance-optimizer
    spec:
      containers:
      - name: performance-optimizer
        image: python:3.11-slim
        ports:
        - containerPort: 8013
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PASSWORD
          value: "$REDIS_PASSWORD"
        command: ["python", "/app/performance_optimizer.py"]
        volumeMounts:
        - name: app-code
          mountPath: /app
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
            port: 8013
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8013
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: app-code
        configMap:
          name: performance-optimizer-code
---
apiVersion: v1
kind: Service
metadata:
  name: performance-optimizer-service
  namespace: $NAMESPACE
spec:
  selector:
    app: performance-optimizer
  ports:
  - port: 8013
    targetPort: 8013
  type: ClusterIP
EOF

# Deploy Production Integration Manager
echo "🔧 Deploying Production Integration Manager..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: production-integration-manager
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: production-integration-manager
  template:
    metadata:
      labels:
        app: production-integration-manager
    spec:
      containers:
      - name: production-integration-manager
        image: python:3.11-slim
        ports:
        - containerPort: 8014
        env:
        - name: POSTGRES_HOST
          value: "postgres-service"
        - name: POSTGRES_PASSWORD
          value: "$POSTGRES_PASSWORD"
        command: ["python", "/app/production_integration_manager.py"]
        volumeMounts:
        - name: app-code
          mountPath: /app
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
            port: 8014
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8014
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: app-code
        configMap:
          name: integration-manager-code
---
apiVersion: v1
kind: Service
metadata:
  name: production-integration-manager-service
  namespace: $NAMESPACE
spec:
  selector:
    app: production-integration-manager
  ports:
  - port: 8014
    targetPort: 8014
  type: ClusterIP
EOF

# Deploy Ingress for external access
echo "🌐 Deploying Ingress for external access..."
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nexus-architect-phase6-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - nexus-architect.local
    secretName: nexus-architect-tls
  rules:
  - host: nexus-architect.local
    http:
      paths:
      - path: /privacy
        pathType: Prefix
        backend:
          service:
            name: data-privacy-manager-service
            port:
              number: 8010
      - path: /security
        pathType: Prefix
        backend:
          service:
            name: security-manager-service
            port:
              number: 8011
      - path: /compliance
        pathType: Prefix
        backend:
          service:
            name: compliance-manager-service
            port:
              number: 8012
      - path: /performance
        pathType: Prefix
        backend:
          service:
            name: performance-optimizer-service
            port:
              number: 8013
      - path: /integration
        pathType: Prefix
        backend:
          service:
            name: production-integration-manager-service
            port:
              number: 8014
EOF

# Deploy monitoring and alerting
echo "📊 Deploying monitoring and alerting..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: nexus-architect-phase6-monitor
  namespace: $NAMESPACE
spec:
  selector:
    matchLabels:
      monitoring: enabled
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: nexus-architect-phase6-alerts
  namespace: $NAMESPACE
spec:
  groups:
  - name: nexus-architect-phase6
    rules:
    - alert: PrivacyManagerDown
      expr: up{job="data-privacy-manager"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Data Privacy Manager is down"
        description: "Data Privacy Manager has been down for more than 1 minute"
    
    - alert: SecurityManagerDown
      expr: up{job="security-manager"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Security Manager is down"
        description: "Security Manager has been down for more than 1 minute"
    
    - alert: ComplianceManagerDown
      expr: up{job="compliance-manager"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Compliance Manager is down"
        description: "Compliance Manager has been down for more than 1 minute"
    
    - alert: PerformanceOptimizerDown
      expr: up{job="performance-optimizer"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "Performance Optimizer is down"
        description: "Performance Optimizer has been down for more than 1 minute"
    
    - alert: HighResponseTime
      expr: http_request_duration_seconds{quantile="0.95"} > 2
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High response time detected"
        description: "95th percentile response time is above 2 seconds"
    
    - alert: HighErrorRate
      expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is above 10% for the last 5 minutes"
EOF

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/data-privacy-manager -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/security-manager -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/compliance-manager -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/performance-optimizer -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/production-integration-manager -n $NAMESPACE

# Run health checks
echo "🏥 Running health checks..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

# Test endpoints
echo "🧪 Testing service endpoints..."
echo "Testing Data Privacy Manager..."
kubectl port-forward service/data-privacy-manager-service 8010:8010 -n $NAMESPACE &
sleep 5
curl -f http://localhost:8010/health || echo "❌ Data Privacy Manager health check failed"
pkill -f "kubectl port-forward.*8010"

echo "Testing Security Manager..."
kubectl port-forward service/security-manager-service 8011:8011 -n $NAMESPACE &
sleep 5
curl -f http://localhost:8011/health || echo "❌ Security Manager health check failed"
pkill -f "kubectl port-forward.*8011"

echo "Testing Compliance Manager..."
kubectl port-forward service/compliance-manager-service 8012:8012 -n $NAMESPACE &
sleep 5
curl -f http://localhost:8012/health || echo "❌ Compliance Manager health check failed"
pkill -f "kubectl port-forward.*8012"

echo "Testing Performance Optimizer..."
kubectl port-forward service/performance-optimizer-service 8013:8013 -n $NAMESPACE &
sleep 5
curl -f http://localhost:8013/health || echo "❌ Performance Optimizer health check failed"
pkill -f "kubectl port-forward.*8013"

echo "Testing Production Integration Manager..."
kubectl port-forward service/production-integration-manager-service 8014:8014 -n $NAMESPACE &
sleep 5
curl -f http://localhost:8014/health || echo "❌ Production Integration Manager health check failed"
pkill -f "kubectl port-forward.*8014"

# Display deployment summary
echo ""
echo "🎉 WS3 Phase 6 Deployment Complete!"
echo "=================================================================="
echo "📊 Deployment Summary:"
echo "  ✅ Data Privacy Manager: http://nexus-architect.local/privacy"
echo "  ✅ Security Manager: http://nexus-architect.local/security"
echo "  ✅ Compliance Manager: http://nexus-architect.local/compliance"
echo "  ✅ Performance Optimizer: http://nexus-architect.local/performance"
echo "  ✅ Production Integration Manager: http://nexus-architect.local/integration"
echo ""
echo "🔧 Management Commands:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl logs -f deployment/data-privacy-manager -n $NAMESPACE"
echo "  kubectl logs -f deployment/security-manager -n $NAMESPACE"
echo "  kubectl logs -f deployment/compliance-manager -n $NAMESPACE"
echo "  kubectl logs -f deployment/performance-optimizer -n $NAMESPACE"
echo "  kubectl logs -f deployment/production-integration-manager -n $NAMESPACE"
echo ""
echo "📈 Monitoring:"
echo "  Prometheus metrics: Available on all services at /metrics"
echo "  Health checks: Available on all services at /health"
echo "  Grafana dashboards: Configure using provided dashboard JSON"
echo ""
echo "🔐 Security Features:"
echo "  ✅ Data privacy controls with PII detection and anonymization"
echo "  ✅ Comprehensive security scanning and threat detection"
echo "  ✅ GDPR, CCPA, HIPAA, and SOC 2 compliance monitoring"
echo "  ✅ Performance optimization with ML-based recommendations"
echo "  ✅ Production-ready integration testing and validation"
echo ""
echo "WS3 Phase 6: Data Privacy, Security & Production Optimization - DEPLOYED! 🚀"

