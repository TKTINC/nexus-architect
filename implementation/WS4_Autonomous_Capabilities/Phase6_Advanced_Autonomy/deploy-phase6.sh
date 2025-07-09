#!/bin/bash

# Nexus Architect - WS4 Phase 6: Advanced Autonomy & Production Optimization
# Deployment Script for Multi-Agent Coordination, Adaptive Learning, and Production Optimization
# Version: 1.0.0
# Author: Nexus Architect Team

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-"production"}
NAMESPACE=${NAMESPACE:-"nexus-architect"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"nexus-registry.local"}
VERSION=${VERSION:-"1.0.0"}

# Service ports
MULTI_AGENT_PORT=8070
ADAPTIVE_LEARNING_PORT=8071
PRODUCTION_OPTIMIZER_PORT=8072

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Nexus Architect WS4 Phase 6 Deployment${NC}"
echo -e "${BLUE}  Advanced Autonomy & Production Optimization${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1

    print_status "Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$port/health" >/dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "python3" "pip3" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            print_error "Required command '$cmd' not found. Please install it first."
            exit 1
        fi
    done
    
    # Check if Kubernetes is available (optional)
    if command_exists "kubectl"; then
        print_status "Kubernetes CLI detected - will deploy to cluster"
        DEPLOY_TO_K8S=true
    else
        print_warning "Kubernetes CLI not found - will deploy locally with Docker"
        DEPLOY_TO_K8S=false
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_status "Prerequisites check completed successfully"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install required packages
    pip3 install --upgrade pip
    pip3 install \
        flask \
        flask-cors \
        redis \
        psycopg2-binary \
        numpy \
        pandas \
        scikit-learn \
        psutil \
        docker \
        kubernetes \
        joblib \
        pyyaml \
        requests
    
    print_status "Python dependencies installed successfully"
}

# Setup databases
setup_databases() {
    print_status "Setting up databases..."
    
    # Start Redis if not running
    if ! docker ps | grep -q "nexus-redis"; then
        print_status "Starting Redis container..."
        docker run -d \
            --name nexus-redis \
            --restart unless-stopped \
            -p 6379:6379 \
            redis:7-alpine \
            redis-server --appendonly yes
    else
        print_status "Redis container already running"
    fi
    
    # Start PostgreSQL if not running
    if ! docker ps | grep -q "nexus-postgres"; then
        print_status "Starting PostgreSQL container..."
        docker run -d \
            --name nexus-postgres \
            --restart unless-stopped \
            -p 5432:5432 \
            -e POSTGRES_DB=nexus_architect \
            -e POSTGRES_USER=nexus_user \
            -e POSTGRES_PASSWORD=nexus_password \
            -v nexus_postgres_data:/var/lib/postgresql/data \
            postgres:15-alpine
        
        # Wait for PostgreSQL to be ready
        sleep 10
        
        # Create database schema
        print_status "Creating database schema..."
        docker exec nexus-postgres psql -U nexus_user -d nexus_architect -c "
            CREATE TABLE IF NOT EXISTS multi_agent_tasks (
                id VARCHAR(255) PRIMARY KEY,
                task_type VARCHAR(100) NOT NULL,
                priority INTEGER NOT NULL,
                description TEXT,
                requirements JSONB,
                dependencies JSONB,
                assigned_agent VARCHAR(255),
                status VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                assigned_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                deadline TIMESTAMP,
                estimated_duration FLOAT,
                actual_duration FLOAT,
                result JSONB,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS learning_experiences (
                id VARCHAR(255) PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                context JSONB NOT NULL,
                action_taken JSONB NOT NULL,
                outcome JSONB NOT NULL,
                success BOOLEAN NOT NULL,
                performance_metrics JSONB,
                feedback_score FLOAT,
                learning_type VARCHAR(50),
                tags JSONB
            );
            
            CREATE TABLE IF NOT EXISTS decision_history (
                scenario_id VARCHAR(255) PRIMARY KEY,
                scenario_type VARCHAR(100) NOT NULL,
                selected_action JSONB NOT NULL,
                confidence FLOAT,
                reasoning TEXT,
                expected_outcomes JSONB,
                risk_assessment JSONB,
                decision_time FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS production_optimization_status (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performance_metrics JSONB,
                optimization_running BOOLEAN,
                total_optimizations INTEGER,
                total_enhancements INTEGER
            );
            
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON multi_agent_tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON multi_agent_tasks(created_at);
            CREATE INDEX IF NOT EXISTS idx_experiences_timestamp ON learning_experiences(timestamp);
            CREATE INDEX IF NOT EXISTS idx_experiences_success ON learning_experiences(success);
            CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decision_history(timestamp);
            CREATE INDEX IF NOT EXISTS idx_optimization_timestamp ON production_optimization_status(timestamp);
        "
    else
        print_status "PostgreSQL container already running"
    fi
    
    print_status "Database setup completed"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Create Dockerfile for Multi-Agent Coordinator
    cat > Dockerfile.multi-agent << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY multi-agent-coordination/ ./multi-agent-coordination/

# Expose port
EXPOSE 8070

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8070/health || exit 1

# Run application
CMD ["python", "multi-agent-coordination/multi_agent_coordinator.py"]
EOF

    # Create Dockerfile for Adaptive Learning Engine
    cat > Dockerfile.adaptive-learning << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY adaptive-learning/ ./adaptive-learning/

# Expose port
EXPOSE 8071

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8071/health || exit 1

# Run application
CMD ["python", "adaptive-learning/adaptive_learning_engine.py"]
EOF

    # Create Dockerfile for Production Optimizer
    cat > Dockerfile.production-optimizer << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY production-optimization/ ./production-optimization/

# Expose port
EXPOSE 8072

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8072/health || exit 1

# Run application
CMD ["python", "production-optimization/production_optimizer.py"]
EOF

    # Create requirements.txt
    cat > requirements.txt << 'EOF'
flask==2.3.3
flask-cors==4.0.0
redis==5.0.1
psycopg2-binary==2.9.7
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
psutil==5.9.5
docker==6.1.3
kubernetes==27.2.0
joblib==1.3.2
pyyaml==6.0.1
requests==2.31.0
EOF

    # Build images
    docker build -f Dockerfile.multi-agent -t ${DOCKER_REGISTRY}/nexus-multi-agent:${VERSION} .
    docker build -f Dockerfile.adaptive-learning -t ${DOCKER_REGISTRY}/nexus-adaptive-learning:${VERSION} .
    docker build -f Dockerfile.production-optimizer -t ${DOCKER_REGISTRY}/nexus-production-optimizer:${VERSION} .
    
    print_status "Docker images built successfully"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    print_status "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Create ConfigMap for configuration
    kubectl create configmap nexus-phase6-config \
        --from-literal=REDIS_HOST=nexus-redis \
        --from-literal=POSTGRES_HOST=nexus-postgres \
        --from-literal=POSTGRES_DB=nexus_architect \
        --from-literal=POSTGRES_USER=nexus_user \
        --from-literal=POSTGRES_PASSWORD=nexus_password \
        --namespace=${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Redis
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-redis
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nexus-redis
  template:
    metadata:
      labels:
        app: nexus-redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        args: ["redis-server", "--appendonly", "yes"]
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-redis
  namespace: ${NAMESPACE}
spec:
  selector:
    app: nexus-redis
  ports:
  - port: 6379
    targetPort: 6379
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

    # Deploy PostgreSQL
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-postgres
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nexus-postgres
  template:
    metadata:
      labels:
        app: nexus-postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: nexus_architect
        - name: POSTGRES_USER
          value: nexus_user
        - name: POSTGRES_PASSWORD
          value: nexus_password
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-data
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-postgres
  namespace: ${NAMESPACE}
spec:
  selector:
    app: nexus-postgres
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
EOF

    # Deploy Multi-Agent Coordinator
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-multi-agent
  namespace: ${NAMESPACE}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nexus-multi-agent
  template:
    metadata:
      labels:
        app: nexus-multi-agent
    spec:
      containers:
      - name: multi-agent
        image: ${DOCKER_REGISTRY}/nexus-multi-agent:${VERSION}
        ports:
        - containerPort: 8070
        envFrom:
        - configMapRef:
            name: nexus-phase6-config
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
  name: nexus-multi-agent
  namespace: ${NAMESPACE}
spec:
  selector:
    app: nexus-multi-agent
  ports:
  - port: 8070
    targetPort: 8070
  type: LoadBalancer
EOF

    # Deploy Adaptive Learning Engine
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-adaptive-learning
  namespace: ${NAMESPACE}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nexus-adaptive-learning
  template:
    metadata:
      labels:
        app: nexus-adaptive-learning
    spec:
      containers:
      - name: adaptive-learning
        image: ${DOCKER_REGISTRY}/nexus-adaptive-learning:${VERSION}
        ports:
        - containerPort: 8071
        envFrom:
        - configMapRef:
            name: nexus-phase6-config
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
  name: nexus-adaptive-learning
  namespace: ${NAMESPACE}
spec:
  selector:
    app: nexus-adaptive-learning
  ports:
  - port: 8071
    targetPort: 8071
  type: LoadBalancer
EOF

    # Deploy Production Optimizer
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexus-production-optimizer
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nexus-production-optimizer
  template:
    metadata:
      labels:
        app: nexus-production-optimizer
    spec:
      containers:
      - name: production-optimizer
        image: ${DOCKER_REGISTRY}/nexus-production-optimizer:${VERSION}
        ports:
        - containerPort: 8072
        envFrom:
        - configMapRef:
            name: nexus-phase6-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        securityContext:
          privileged: true  # Required for system monitoring
---
apiVersion: v1
kind: Service
metadata:
  name: nexus-production-optimizer
  namespace: ${NAMESPACE}
spec:
  selector:
    app: nexus-production-optimizer
  ports:
  - port: 8072
    targetPort: 8072
  type: LoadBalancer
EOF

    print_status "Kubernetes deployment completed"
}

# Deploy locally with Docker
deploy_locally() {
    print_status "Deploying locally with Docker..."
    
    # Create Docker network
    docker network create nexus-network 2>/dev/null || true
    
    # Start services
    docker run -d \
        --name nexus-multi-agent \
        --network nexus-network \
        --restart unless-stopped \
        -p ${MULTI_AGENT_PORT}:8070 \
        -e REDIS_HOST=nexus-redis \
        -e POSTGRES_HOST=nexus-postgres \
        ${DOCKER_REGISTRY}/nexus-multi-agent:${VERSION}
    
    docker run -d \
        --name nexus-adaptive-learning \
        --network nexus-network \
        --restart unless-stopped \
        -p ${ADAPTIVE_LEARNING_PORT}:8071 \
        -e REDIS_HOST=nexus-redis \
        -e POSTGRES_HOST=nexus-postgres \
        ${DOCKER_REGISTRY}/nexus-adaptive-learning:${VERSION}
    
    docker run -d \
        --name nexus-production-optimizer \
        --network nexus-network \
        --restart unless-stopped \
        -p ${PRODUCTION_OPTIMIZER_PORT}:8072 \
        -e REDIS_HOST=nexus-redis \
        -e POSTGRES_HOST=nexus-postgres \
        --privileged \
        -v /var/run/docker.sock:/var/run/docker.sock \
        ${DOCKER_REGISTRY}/nexus-production-optimizer:${VERSION}
    
    print_status "Local deployment completed"
}

# Run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    # Wait for all services to be ready
    wait_for_service "Multi-Agent Coordinator" ${MULTI_AGENT_PORT}
    wait_for_service "Adaptive Learning Engine" ${ADAPTIVE_LEARNING_PORT}
    wait_for_service "Production Optimizer" ${PRODUCTION_OPTIMIZER_PORT}
    
    # Test Multi-Agent Coordinator
    print_status "Testing Multi-Agent Coordinator..."
    response=$(curl -s "http://localhost:${MULTI_AGENT_PORT}/coordination/status")
    if echo "$response" | grep -q "coordination_running"; then
        print_status "Multi-Agent Coordinator test passed"
    else
        print_error "Multi-Agent Coordinator test failed"
        return 1
    fi
    
    # Test Adaptive Learning Engine
    print_status "Testing Adaptive Learning Engine..."
    response=$(curl -s "http://localhost:${ADAPTIVE_LEARNING_PORT}/learning/status")
    if echo "$response" | grep -q "total_experiences"; then
        print_status "Adaptive Learning Engine test passed"
    else
        print_error "Adaptive Learning Engine test failed"
        return 1
    fi
    
    # Test Production Optimizer
    print_status "Testing Production Optimizer..."
    response=$(curl -s "http://localhost:${PRODUCTION_OPTIMIZER_PORT}/optimization/status")
    if echo "$response" | grep -q "optimization_running"; then
        print_status "Production Optimizer test passed"
    else
        print_error "Production Optimizer test failed"
        return 1
    fi
    
    print_status "All integration tests passed successfully"
}

# Generate deployment report
generate_report() {
    print_status "Generating deployment report..."
    
    cat > deployment_report.md << EOF
# WS4 Phase 6 Deployment Report

## Deployment Summary
- **Date**: $(date)
- **Environment**: ${DEPLOYMENT_ENV}
- **Version**: ${VERSION}
- **Deployment Type**: $([ "$DEPLOY_TO_K8S" = true ] && echo "Kubernetes" || echo "Docker Local")

## Services Deployed

### Multi-Agent Coordinator
- **Port**: ${MULTI_AGENT_PORT}
- **URL**: http://localhost:${MULTI_AGENT_PORT}
- **Health Check**: http://localhost:${MULTI_AGENT_PORT}/health
- **Status**: $(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${MULTI_AGENT_PORT}/health" 2>/dev/null || echo "Unknown")

### Adaptive Learning Engine
- **Port**: ${ADAPTIVE_LEARNING_PORT}
- **URL**: http://localhost:${ADAPTIVE_LEARNING_PORT}
- **Health Check**: http://localhost:${ADAPTIVE_LEARNING_PORT}/health
- **Status**: $(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${ADAPTIVE_LEARNING_PORT}/health" 2>/dev/null || echo "Unknown")

### Production Optimizer
- **Port**: ${PRODUCTION_OPTIMIZER_PORT}
- **URL**: http://localhost:${PRODUCTION_OPTIMIZER_PORT}
- **Health Check**: http://localhost:${PRODUCTION_OPTIMIZER_PORT}/health
- **Status**: $(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PRODUCTION_OPTIMIZER_PORT}/health" 2>/dev/null || echo "Unknown")

## Database Services
- **Redis**: Port 6379
- **PostgreSQL**: Port 5432

## Next Steps
1. Access the services using the URLs above
2. Monitor logs: \`docker logs <container_name>\`
3. Scale services as needed
4. Configure monitoring and alerting
5. Set up backup and disaster recovery

## Troubleshooting
- Check service logs: \`docker logs nexus-<service-name>\`
- Verify database connectivity
- Ensure all ports are accessible
- Check resource usage and scaling needs

EOF

    print_status "Deployment report generated: deployment_report.md"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up temporary files..."
    rm -f Dockerfile.* requirements.txt
}

# Main deployment function
main() {
    echo -e "${BLUE}Starting WS4 Phase 6 deployment...${NC}"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Install dependencies
    install_dependencies
    
    # Setup databases
    setup_databases
    
    # Build Docker images
    build_images
    
    # Deploy based on environment
    if [ "$DEPLOY_TO_K8S" = true ]; then
        deploy_to_kubernetes
    else
        deploy_locally
    fi
    
    # Run integration tests
    run_integration_tests
    
    # Generate deployment report
    generate_report
    
    # Cleanup
    cleanup
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  WS4 Phase 6 Deployment Completed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Services are now running:${NC}"
    echo -e "  • Multi-Agent Coordinator: http://localhost:${MULTI_AGENT_PORT}"
    echo -e "  • Adaptive Learning Engine: http://localhost:${ADAPTIVE_LEARNING_PORT}"
    echo -e "  • Production Optimizer: http://localhost:${PRODUCTION_OPTIMIZER_PORT}"
    echo ""
    echo -e "${BLUE}Check deployment_report.md for detailed information${NC}"
    echo ""
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@"

