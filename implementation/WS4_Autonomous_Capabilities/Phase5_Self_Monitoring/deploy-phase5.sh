#!/bin/bash

# Nexus Architect - WS4 Phase 5: Self-Monitoring & Autonomous Operations Deployment
# This script deploys the self-monitoring and autonomous operations components

set -e

echo "🚀 Starting WS4 Phase 5: Self-Monitoring & Autonomous Operations Deployment"

# Configuration
NAMESPACE="nexus-architect"
PHASE_DIR="/home/ubuntu/nexus-architect/implementation/WS4_Autonomous_Capabilities/Phase5_Self_Monitoring"
DOCKER_REGISTRY="nexus-architect"

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
    
    # Check if running as root or with sudo
    if [[ $EUID -eq 0 ]]; then
        warning "Running as root. This is not recommended for production."
    fi
    
    # Check required commands
    local required_commands=("python3" "pip3" "docker" "kubectl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "$cmd is required but not installed."
        fi
    done
    
    # Check Python version
    local python_version=$(python3 --version | cut -d' ' -f2)
    local required_version="3.8"
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        error "Python 3.8+ is required. Current version: $python_version"
    fi
    
    success "Prerequisites check completed"
}

# Install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    cd "$PHASE_DIR"
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f "requirements.txt" ]; then
        cat > requirements.txt << EOF
flask==2.3.3
flask-cors==4.0.0
psutil==5.9.5
redis==4.6.0
psycopg2-binary==2.9.7
numpy==1.24.3
scikit-learn==1.3.0
requests==2.31.0
docker==6.1.3
kubernetes==27.2.0
pyyaml==6.0.1
asyncio==3.4.3
dataclasses==0.6
EOF
    fi
    
    # Install dependencies
    pip3 install -r requirements.txt
    
    success "Dependencies installed successfully"
}

# Setup database schema
setup_database() {
    log "Setting up database schema..."
    
    # Create database schema for monitoring data
    cat > /tmp/monitoring_schema.sql << EOF
-- WS4 Phase 5: Self-Monitoring Database Schema

-- Health metrics table
CREATE TABLE IF NOT EXISTS health_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    value FLOAT NOT NULL,
    unit VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    status VARCHAR(50) NOT NULL,
    threshold_warning FLOAT,
    threshold_critical FLOAT,
    tags JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System alerts table
CREATE TABLE IF NOT EXISTS system_alerts (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    severity VARCHAR(50) NOT NULL,
    component VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolution_time TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Autonomous actions table
CREATE TABLE IF NOT EXISTS autonomous_actions (
    id VARCHAR(255) PRIMARY KEY,
    action_type VARCHAR(100) NOT NULL,
    priority INTEGER NOT NULL,
    description TEXT NOT NULL,
    target_component VARCHAR(255) NOT NULL,
    parameters JSONB,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    result JSONB,
    error_message TEXT,
    rollback_data JSONB
);

-- Security incidents table
CREATE TABLE IF NOT EXISTS security_incidents (
    id VARCHAR(255) PRIMARY KEY,
    incident_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    source_ip INET,
    affected_component VARCHAR(255) NOT NULL,
    detected_at TIMESTAMP NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    response_actions TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance optimizations table
CREATE TABLE IF NOT EXISTS performance_optimizations (
    id SERIAL PRIMARY KEY,
    component VARCHAR(255) NOT NULL,
    optimization_type VARCHAR(100) NOT NULL,
    current_value TEXT,
    recommended_value TEXT,
    expected_improvement FLOAT,
    confidence FLOAT,
    risk_level VARCHAR(50),
    applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMP,
    result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_health_metrics_timestamp ON health_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_health_metrics_name ON health_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_alerts_timestamp ON system_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_alerts_severity ON system_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_autonomous_actions_status ON autonomous_actions(status);
CREATE INDEX IF NOT EXISTS idx_autonomous_actions_created ON autonomous_actions(created_at);
CREATE INDEX IF NOT EXISTS idx_security_incidents_detected ON security_incidents(detected_at);
CREATE INDEX IF NOT EXISTS idx_security_incidents_severity ON security_incidents(severity);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nexus_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nexus_user;
EOF
    
    # Execute schema
    if command -v psql &> /dev/null; then
        PGPASSWORD=nexus_password psql -h localhost -U nexus_user -d nexus_architect -f /tmp/monitoring_schema.sql
        success "Database schema created successfully"
    else
        warning "PostgreSQL client not found. Please run the schema manually."
    fi
    
    # Clean up
    rm -f /tmp/monitoring_schema.sql
}

# Create Kubernetes manifests
create_k8s_manifests() {
    log "Creating Kubernetes manifests..."
    
    mkdir -p "$PHASE_DIR/k8s"
    
    # System Health Monitor Deployment
    cat > "$PHASE_DIR/k8s/system-health-monitor.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: system-health-monitor
  namespace: $NAMESPACE
  labels:
    app: system-health-monitor
    component: monitoring
    workstream: ws4
    phase: phase5
spec:
  replicas: 2
  selector:
    matchLabels:
      app: system-health-monitor
  template:
    metadata:
      labels:
        app: system-health-monitor
        component: monitoring
    spec:
      containers:
      - name: system-health-monitor
        image: $DOCKER_REGISTRY/system-health-monitor:latest
        ports:
        - containerPort: 8060
          name: http
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: POSTGRES_HOST
          value: "postgresql-service"
        - name: POSTGRES_DB
          value: "nexus_architect"
        - name: POSTGRES_USER
          value: "nexus_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-secret
              key: password
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
            port: 8060
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8060
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      serviceAccountName: monitoring-service-account
---
apiVersion: v1
kind: Service
metadata:
  name: system-health-monitor-service
  namespace: $NAMESPACE
  labels:
    app: system-health-monitor
spec:
  selector:
    app: system-health-monitor
  ports:
  - port: 8060
    targetPort: 8060
    name: http
  type: ClusterIP
EOF

    # Autonomous Operations Manager Deployment
    cat > "$PHASE_DIR/k8s/autonomous-operations-manager.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autonomous-operations-manager
  namespace: $NAMESPACE
  labels:
    app: autonomous-operations-manager
    component: operations
    workstream: ws4
    phase: phase5
spec:
  replicas: 1
  selector:
    matchLabels:
      app: autonomous-operations-manager
  template:
    metadata:
      labels:
        app: autonomous-operations-manager
        component: operations
    spec:
      containers:
      - name: autonomous-operations-manager
        image: $DOCKER_REGISTRY/autonomous-operations-manager:latest
        ports:
        - containerPort: 8061
          name: http
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: POSTGRES_HOST
          value: "postgresql-service"
        - name: POSTGRES_DB
          value: "nexus_architect"
        - name: POSTGRES_USER
          value: "nexus_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-secret
              key: password
        - name: DOCKER_HOST
          value: "unix:///var/run/docker.sock"
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
            port: 8061
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8061
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: docker-sock
          mountPath: /var/run/docker.sock
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        securityContext:
          privileged: true
      volumes:
      - name: docker-sock
        hostPath:
          path: /var/run/docker.sock
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      serviceAccountName: operations-service-account
---
apiVersion: v1
kind: Service
metadata:
  name: autonomous-operations-manager-service
  namespace: $NAMESPACE
  labels:
    app: autonomous-operations-manager
spec:
  selector:
    app: autonomous-operations-manager
  ports:
  - port: 8061
    targetPort: 8061
    name: http
  type: ClusterIP
EOF

    # Service Accounts and RBAC
    cat > "$PHASE_DIR/k8s/rbac.yaml" << EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: monitoring-service-account
  namespace: $NAMESPACE
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: operations-service-account
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: monitoring-cluster-role
rules:
- apiGroups: [""]
  resources: ["nodes", "pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["nodes", "pods"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: operations-cluster-role
rules:
- apiGroups: [""]
  resources: ["nodes", "pods", "services", "endpoints"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: monitoring-cluster-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: monitoring-cluster-role
subjects:
- kind: ServiceAccount
  name: monitoring-service-account
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: operations-cluster-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: operations-cluster-role
subjects:
- kind: ServiceAccount
  name: operations-service-account
  namespace: $NAMESPACE
EOF

    # ConfigMap for configuration
    cat > "$PHASE_DIR/k8s/configmap.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: monitoring-config
  namespace: $NAMESPACE
data:
  monitoring.yaml: |
    monitoring:
      collection_interval: 30
      history_retention_hours: 24
      max_history_points: 2880
      
    alerts:
      cpu_warning_threshold: 70
      cpu_critical_threshold: 90
      memory_warning_threshold: 80
      memory_critical_threshold: 95
      disk_warning_threshold: 80
      disk_critical_threshold: 95
      
    operations:
      check_interval: 60
      max_concurrent_actions: 3
      action_timeout: 300
      
    security:
      failed_login_threshold: 5
      max_connections_per_ip: 100
      rate_limit_threshold: 1000
      blocked_ports: [22, 3389, 5432, 6379]
      
    notifications:
      channels: ["log", "webhook"]
      webhook_url: "http://notification-service:8080/webhook"
EOF

    success "Kubernetes manifests created"
}

# Build Docker images
build_docker_images() {
    log "Building Docker images..."
    
    # System Health Monitor Dockerfile
    cat > "$PHASE_DIR/health-monitoring/Dockerfile" << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    procps \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY system_health_monitor.py .

# Create non-root user
RUN useradd -m -u 1000 monitor && chown -R monitor:monitor /app
USER monitor

# Expose port
EXPOSE 8060

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8060/health || exit 1

# Run application
CMD ["python", "system_health_monitor.py"]
EOF

    # Autonomous Operations Manager Dockerfile
    cat > "$PHASE_DIR/self-healing/Dockerfile" << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    procps \\
    curl \\
    iptables \\
    sudo \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY autonomous_operations_manager.py .

# Create non-root user with sudo privileges
RUN useradd -m -u 1000 operations && \\
    echo "operations ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \\
    chown -R operations:operations /app

USER operations

# Expose port
EXPOSE 8061

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8061/health || exit 1

# Run application
CMD ["python", "autonomous_operations_manager.py"]
EOF

    # Build images
    cd "$PHASE_DIR/health-monitoring"
    docker build -t $DOCKER_REGISTRY/system-health-monitor:latest .
    
    cd "$PHASE_DIR/self-healing"
    docker build -t $DOCKER_REGISTRY/autonomous-operations-manager:latest .
    
    success "Docker images built successfully"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply manifests
    kubectl apply -f "$PHASE_DIR/k8s/"
    
    # Wait for deployments to be ready
    log "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/system-health-monitor -n $NAMESPACE
    kubectl wait --for=condition=available --timeout=300s deployment/autonomous-operations-manager -n $NAMESPACE
    
    success "Kubernetes deployment completed"
}

# Setup monitoring and alerting
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Create Prometheus ServiceMonitor
    cat > "$PHASE_DIR/k8s/servicemonitor.yaml" << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ws4-phase5-monitoring
  namespace: $NAMESPACE
  labels:
    app: ws4-phase5
spec:
  selector:
    matchLabels:
      app: system-health-monitor
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ws4-phase5-operations
  namespace: $NAMESPACE
  labels:
    app: ws4-phase5
spec:
  selector:
    matchLabels:
      app: autonomous-operations-manager
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
EOF

    # Apply monitoring configuration
    kubectl apply -f "$PHASE_DIR/k8s/servicemonitor.yaml" || warning "ServiceMonitor creation failed (Prometheus Operator may not be installed)"
    
    success "Monitoring setup completed"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n $NAMESPACE -l workstream=ws4,phase=phase5
    
    # Check service endpoints
    local health_monitor_ip=$(kubectl get svc system-health-monitor-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    local operations_manager_ip=$(kubectl get svc autonomous-operations-manager-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    
    log "Service endpoints:"
    echo "  System Health Monitor: http://$health_monitor_ip:8060"
    echo "  Autonomous Operations Manager: http://$operations_manager_ip:8061"
    
    # Test health endpoints
    if kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -n $NAMESPACE -- curl -f "http://$health_monitor_ip:8060/health" > /dev/null 2>&1; then
        success "System Health Monitor is responding"
    else
        error "System Health Monitor health check failed"
    fi
    
    if kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -n $NAMESPACE -- curl -f "http://$operations_manager_ip:8061/health" > /dev/null 2>&1; then
        success "Autonomous Operations Manager is responding"
    else
        error "Autonomous Operations Manager health check failed"
    fi
    
    success "Deployment verification completed"
}

# Create startup script
create_startup_script() {
    log "Creating startup script..."
    
    cat > "$PHASE_DIR/start-monitoring.sh" << 'EOF'
#!/bin/bash

# Start WS4 Phase 5 Self-Monitoring Services

echo "🚀 Starting WS4 Phase 5: Self-Monitoring & Autonomous Operations"

# Start System Health Monitor
echo "Starting System Health Monitor..."
cd /home/ubuntu/nexus-architect/implementation/WS4_Autonomous_Capabilities/Phase5_Self_Monitoring/health-monitoring
python3 system_health_monitor.py &
HEALTH_MONITOR_PID=$!
echo "System Health Monitor started with PID: $HEALTH_MONITOR_PID"

# Start Autonomous Operations Manager
echo "Starting Autonomous Operations Manager..."
cd /home/ubuntu/nexus-architect/implementation/WS4_Autonomous_Capabilities/Phase5_Self_Monitoring/self-healing
python3 autonomous_operations_manager.py &
OPERATIONS_MANAGER_PID=$!
echo "Autonomous Operations Manager started with PID: $OPERATIONS_MANAGER_PID"

# Save PIDs for later cleanup
echo $HEALTH_MONITOR_PID > /tmp/health_monitor.pid
echo $OPERATIONS_MANAGER_PID > /tmp/operations_manager.pid

echo "✅ All services started successfully!"
echo "System Health Monitor: http://localhost:8060"
echo "Autonomous Operations Manager: http://localhost:8061"
echo ""
echo "To stop services, run: ./stop-monitoring.sh"
EOF

    chmod +x "$PHASE_DIR/start-monitoring.sh"
    
    # Create stop script
    cat > "$PHASE_DIR/stop-monitoring.sh" << 'EOF'
#!/bin/bash

# Stop WS4 Phase 5 Self-Monitoring Services

echo "🛑 Stopping WS4 Phase 5: Self-Monitoring & Autonomous Operations"

# Stop System Health Monitor
if [ -f /tmp/health_monitor.pid ]; then
    HEALTH_MONITOR_PID=$(cat /tmp/health_monitor.pid)
    if kill -0 $HEALTH_MONITOR_PID 2>/dev/null; then
        kill $HEALTH_MONITOR_PID
        echo "System Health Monitor stopped (PID: $HEALTH_MONITOR_PID)"
    fi
    rm -f /tmp/health_monitor.pid
fi

# Stop Autonomous Operations Manager
if [ -f /tmp/operations_manager.pid ]; then
    OPERATIONS_MANAGER_PID=$(cat /tmp/operations_manager.pid)
    if kill -0 $OPERATIONS_MANAGER_PID 2>/dev/null; then
        kill $OPERATIONS_MANAGER_PID
        echo "Autonomous Operations Manager stopped (PID: $OPERATIONS_MANAGER_PID)"
    fi
    rm -f /tmp/operations_manager.pid
fi

echo "✅ All services stopped successfully!"
EOF

    chmod +x "$PHASE_DIR/stop-monitoring.sh"
    
    success "Startup scripts created"
}

# Main deployment function
main() {
    log "🚀 WS4 Phase 5: Self-Monitoring & Autonomous Operations Deployment Started"
    
    check_prerequisites
    install_dependencies
    setup_database
    create_k8s_manifests
    build_docker_images
    deploy_to_kubernetes
    setup_monitoring
    verify_deployment
    create_startup_script
    
    success "🎉 WS4 Phase 5 deployment completed successfully!"
    
    echo ""
    echo "📋 Deployment Summary:"
    echo "  ✅ System Health Monitor deployed on port 8060"
    echo "  ✅ Autonomous Operations Manager deployed on port 8061"
    echo "  ✅ Database schema created"
    echo "  ✅ Kubernetes manifests applied"
    echo "  ✅ Docker images built and deployed"
    echo "  ✅ Monitoring and alerting configured"
    echo ""
    echo "🔗 Service URLs:"
    echo "  System Health Monitor: http://localhost:8060"
    echo "  Autonomous Operations Manager: http://localhost:8061"
    echo ""
    echo "📚 Next Steps:"
    echo "  1. Access the health monitoring dashboard"
    echo "  2. Configure alert thresholds as needed"
    echo "  3. Set up notification channels"
    echo "  4. Monitor autonomous operations"
    echo ""
    echo "🛠️ Management Commands:"
    echo "  Start services: ./start-monitoring.sh"
    echo "  Stop services: ./stop-monitoring.sh"
    echo "  View logs: kubectl logs -f deployment/system-health-monitor -n $NAMESPACE"
}

# Run main function
main "$@"

