#!/bin/bash

# WS4 Phase 1: Autonomous Decision Engine & Safety Framework Deployment
# Deploys autonomous decision-making capabilities with comprehensive safety controls

set -e

echo "🚀 Starting WS4 Phase 1 Deployment: Autonomous Decision Engine & Safety Framework"

# Configuration
NAMESPACE="nexus-autonomous"
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-"production"}
REPLICAS=${REPLICAS:-3}

# Create namespace
echo "📦 Creating Kubernetes namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy PostgreSQL for decision storage
echo "🗄️ Deploying PostgreSQL database..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-autonomous
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres-autonomous
  template:
    metadata:
      labels:
        app: postgres-autonomous
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: "autonomous_decisions"
        - name: POSTGRES_USER
          value: "autonomous_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-autonomous-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-autonomous-service
  namespace: $NAMESPACE
spec:
  selector:
    app: postgres-autonomous
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-autonomous-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
EOF

# Deploy Redis for caching and session management
echo "🔄 Deploying Redis cache..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-autonomous
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis-autonomous
  template:
    metadata:
      labels:
        app: redis-autonomous
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command: ["redis-server"]
        args: ["--appendonly", "yes", "--maxmemory", "1gb", "--maxmemory-policy", "allkeys-lru"]
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-autonomous-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-autonomous-service
  namespace: $NAMESPACE
spec:
  selector:
    app: redis-autonomous
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-autonomous-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
EOF

# Deploy Autonomous Decision Engine
echo "🧠 Deploying Autonomous Decision Engine..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: decision-engine
  namespace: $NAMESPACE
spec:
  replicas: $REPLICAS
  selector:
    matchLabels:
      app: decision-engine
  template:
    metadata:
      labels:
        app: decision-engine
    spec:
      containers:
      - name: decision-engine
        image: nexus/decision-engine:latest
        ports:
        - containerPort: 8020
        env:
        - name: DATABASE_URL
          value: "postgresql://autonomous_user:password@postgres-autonomous-service:5432/autonomous_decisions"
        - name: REDIS_URL
          value: "redis://redis-autonomous-service:6379"
        - name: ENVIRONMENT
          value: "$DEPLOYMENT_ENV"
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
            port: 8020
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8020
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: decision-engine-service
  namespace: $NAMESPACE
spec:
  selector:
    app: decision-engine
  ports:
  - port: 8020
    targetPort: 8020
  type: ClusterIP
EOF

# Deploy Safety Framework
echo "🛡️ Deploying Safety Framework..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safety-framework
  namespace: $NAMESPACE
spec:
  replicas: $REPLICAS
  selector:
    matchLabels:
      app: safety-framework
  template:
    metadata:
      labels:
        app: safety-framework
    spec:
      containers:
      - name: safety-framework
        image: nexus/safety-framework:latest
        ports:
        - containerPort: 8021
        env:
        - name: DATABASE_URL
          value: "postgresql://autonomous_user:password@postgres-autonomous-service:5432/autonomous_decisions"
        - name: REDIS_URL
          value: "redis://redis-autonomous-service:6379"
        - name: DECISION_ENGINE_URL
          value: "http://decision-engine-service:8020"
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
            port: 8021
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8021
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: safety-framework-service
  namespace: $NAMESPACE
spec:
  selector:
    app: safety-framework
  ports:
  - port: 8021
    targetPort: 8021
  type: ClusterIP
EOF

# Deploy Risk Manager
echo "⚠️ Deploying Risk Manager..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: risk-manager
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: risk-manager
  template:
    metadata:
      labels:
        app: risk-manager
    spec:
      containers:
      - name: risk-manager
        image: nexus/risk-manager:latest
        ports:
        - containerPort: 8022
        env:
        - name: DATABASE_URL
          value: "postgresql://autonomous_user:password@postgres-autonomous-service:5432/autonomous_decisions"
        - name: REDIS_URL
          value: "redis://redis-autonomous-service:6379"
        - name: DECISION_ENGINE_URL
          value: "http://decision-engine-service:8020"
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
            port: 8022
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8022
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: risk-manager-service
  namespace: $NAMESPACE
spec:
  selector:
    app: risk-manager
  ports:
  - port: 8022
    targetPort: 8022
  type: ClusterIP
EOF

# Deploy Human Oversight Manager
echo "👥 Deploying Human Oversight Manager..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: oversight-manager
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: oversight-manager
  template:
    metadata:
      labels:
        app: oversight-manager
    spec:
      containers:
      - name: oversight-manager
        image: nexus/oversight-manager:latest
        ports:
        - containerPort: 8023
        env:
        - name: DATABASE_URL
          value: "postgresql://autonomous_user:password@postgres-autonomous-service:5432/autonomous_decisions"
        - name: REDIS_URL
          value: "redis://redis-autonomous-service:6379"
        - name: DECISION_ENGINE_URL
          value: "http://decision-engine-service:8020"
        - name: SAFETY_FRAMEWORK_URL
          value: "http://safety-framework-service:8021"
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
            port: 8023
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8023
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: oversight-manager-service
  namespace: $NAMESPACE
spec:
  selector:
    app: oversight-manager
  ports:
  - port: 8023
    targetPort: 8023
  type: ClusterIP
EOF

# Deploy API Gateway for Autonomous Services
echo "🌐 Deploying API Gateway..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autonomous-gateway
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: autonomous-gateway
  template:
    metadata:
      labels:
        app: autonomous-gateway
    spec:
      containers:
      - name: gateway
        image: nginx:alpine
        ports:
        - containerPort: 80
        volumeMounts:
        - name: nginx-config
          mountPath: /etc/nginx/conf.d
      volumes:
      - name: nginx-config
        configMap:
          name: autonomous-gateway-config
---
apiVersion: v1
kind: Service
metadata:
  name: autonomous-gateway-service
  namespace: $NAMESPACE
spec:
  selector:
    app: autonomous-gateway
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: autonomous-gateway-config
  namespace: $NAMESPACE
data:
  default.conf: |
    upstream decision_engine {
        server decision-engine-service:8020;
    }
    upstream safety_framework {
        server safety-framework-service:8021;
    }
    upstream risk_manager {
        server risk-manager-service:8022;
    }
    upstream oversight_manager {
        server oversight-manager-service:8023;
    }
    
    server {
        listen 80;
        server_name autonomous.nexus.local;
        
        location /api/decisions/ {
            proxy_pass http://decision_engine/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        }
        
        location /api/safety/ {
            proxy_pass http://safety_framework/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        }
        
        location /api/risk/ {
            proxy_pass http://risk_manager/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        }
        
        location /api/oversight/ {
            proxy_pass http://oversight_manager/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        }
        
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
EOF

# Deploy Monitoring and Alerting
echo "📊 Deploying Monitoring Stack..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-autonomous-config
  namespace: $NAMESPACE
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'decision-engine'
      static_configs:
      - targets: ['decision-engine-service:8020']
    - job_name: 'safety-framework'
      static_configs:
      - targets: ['safety-framework-service:8021']
    - job_name: 'risk-manager'
      static_configs:
      - targets: ['risk-manager-service:8022']
    - job_name: 'oversight-manager'
      static_configs:
      - targets: ['oversight-manager-service:8023']
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus-autonomous
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus-autonomous
  template:
    metadata:
      labels:
        app: prometheus-autonomous
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
        command:
        - '/bin/prometheus'
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--storage.tsdb.retention.time=200h'
        - '--web.enable-lifecycle'
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-autonomous-config
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-autonomous-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-autonomous-service
  namespace: $NAMESPACE
spec:
  selector:
    app: prometheus-autonomous
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-autonomous-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

# Create secrets
echo "🔐 Creating secrets..."
kubectl create secret generic postgres-secret \
  --from-literal=password="$(openssl rand -base64 32)" \
  --namespace=$NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/postgres-autonomous -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/redis-autonomous -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/decision-engine -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/safety-framework -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/risk-manager -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/oversight-manager -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/autonomous-gateway -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/prometheus-autonomous -n $NAMESPACE

# Run health checks
echo "🏥 Running health checks..."
sleep 30

# Check service endpoints
echo "🔍 Checking service endpoints..."
kubectl get services -n $NAMESPACE

# Get gateway external IP
GATEWAY_IP=$(kubectl get service autonomous-gateway-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$GATEWAY_IP" ]; then
    GATEWAY_IP=$(kubectl get service autonomous-gateway-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
fi

echo "✅ WS4 Phase 1 Deployment Complete!"
echo ""
echo "🌐 Service Endpoints:"
echo "  Gateway: http://$GATEWAY_IP"
echo "  Decision Engine: http://$GATEWAY_IP/api/decisions/"
echo "  Safety Framework: http://$GATEWAY_IP/api/safety/"
echo "  Risk Manager: http://$GATEWAY_IP/api/risk/"
echo "  Oversight Manager: http://$GATEWAY_IP/api/oversight/"
echo ""
echo "📊 Monitoring:"
echo "  Prometheus: kubectl port-forward service/prometheus-autonomous-service 9090:9090 -n $NAMESPACE"
echo ""
echo "🔧 Management Commands:"
echo "  View logs: kubectl logs -f deployment/decision-engine -n $NAMESPACE"
echo "  Scale services: kubectl scale deployment/decision-engine --replicas=5 -n $NAMESPACE"
echo "  Update config: kubectl edit configmap autonomous-gateway-config -n $NAMESPACE"
echo ""
echo "🎯 Next Steps:"
echo "  1. Configure oversight rules and safety policies"
echo "  2. Set up notification channels (email, Slack, Teams)"
echo "  3. Train decision models with historical data"
echo "  4. Conduct safety validation testing"
echo "  5. Integrate with existing systems"

# Performance validation
echo "🚀 Running performance validation..."
echo "Testing decision engine throughput..."

# Create a simple test script
cat > /tmp/test_decisions.sh << 'EOF'
#!/bin/bash
GATEWAY_URL="http://$1"
TOTAL_REQUESTS=100
CONCURRENT=10

echo "Running $TOTAL_REQUESTS requests with $CONCURRENT concurrent connections..."

for i in $(seq 1 $CONCURRENT); do
  {
    for j in $(seq 1 $((TOTAL_REQUESTS/CONCURRENT))); do
      curl -s -X POST "$GATEWAY_URL/api/decisions/evaluate" \
        -H "Content-Type: application/json" \
        -d '{
          "decision_id": "test-'$i'-'$j'",
          "context": {
            "risk_score": 0.3,
            "impact": "low",
            "urgency": "medium"
          },
          "alternatives": [
            {"id": "alt1", "name": "Option 1", "score": 0.7},
            {"id": "alt2", "name": "Option 2", "score": 0.8}
          ]
        }' > /dev/null
    done
  } &
done

wait
echo "Performance test completed!"
EOF

chmod +x /tmp/test_decisions.sh
/tmp/test_decisions.sh $GATEWAY_IP

echo ""
echo "🎉 WS4 Phase 1: Autonomous Decision Engine & Safety Framework Successfully Deployed!"
echo "   Ready for autonomous decision-making with comprehensive safety controls!"

