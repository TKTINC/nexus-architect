#!/bin/bash

# Nexus Architect - Automated Dependency Installation
# BDT-P1 Deliverable #4: Automated dependency installation for all components
# Version: 1.0
# Author: Nexus DevOps Team

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    exit 1
}

# Check if running in project directory
check_project_directory() {
    if [[ ! -f "workstreams/WS1_Core_Foundation/WS1_EXECUTION_PROMPTS.md" ]]; then
        error "Please run this script from the nexus-architect project root directory"
    fi
    success "Project directory validated ✓"
}

# Install Node.js dependencies
install_nodejs_dependencies() {
    log "Installing Node.js dependencies..."
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Please run setup-local-env.sh first"
    fi
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        error "npm is not installed. Please run setup-local-env.sh first"
    fi
    
    success "Node.js $(node --version) and npm $(npm --version) detected ✓"
    
    # Find all package.json files and install dependencies
    local package_files=($(find implementation/ -name "package.json" -type f))
    
    for package_file in "${package_files[@]}"; do
        local package_dir=$(dirname "$package_file")
        log "Installing dependencies in $package_dir"
        
        (
            cd "$package_dir"
            
            # Check if package-lock.json exists
            if [[ -f "package-lock.json" ]]; then
                log "Using npm ci for faster, reliable installation"
                npm ci
            else
                log "Using npm install"
                npm install
            fi
            
            # Install development dependencies if in development mode
            if [[ "${NODE_ENV:-development}" == "development" ]]; then
                npm install --only=dev
            fi
            
            success "Dependencies installed in $package_dir ✓"
        )
    done
    
    # Install global development tools
    log "Installing global Node.js development tools..."
    npm install -g \
        nodemon \
        pm2 \
        eslint \
        prettier \
        typescript \
        ts-node \
        @types/node \
        concurrently \
        cross-env \
        dotenv-cli
    
    success "Global Node.js tools installed ✓"
}

# Install Python dependencies
install_python_dependencies() {
    log "Installing Python dependencies..."
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please run setup-local-env.sh first"
    fi
    
    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is not installed. Please run setup-local-env.sh first"
    fi
    
    success "Python $(python3 --version) and pip $(pip3 --version) detected ✓"
    
    # Find all requirements.txt files and install dependencies
    local requirements_files=($(find implementation/ -name "requirements.txt" -type f))
    
    for req_file in "${requirements_files[@]}"; do
        local req_dir=$(dirname "$req_file")
        log "Installing Python dependencies in $req_dir"
        
        (
            cd "$req_dir"
            
            # Create virtual environment if it doesn't exist
            if [[ ! -d "venv" ]]; then
                log "Creating virtual environment"
                python3 -m venv venv
            fi
            
            # Activate virtual environment
            source venv/bin/activate
            
            # Upgrade pip
            pip install --upgrade pip
            
            # Install dependencies
            pip install -r requirements.txt
            
            # Install development dependencies if they exist
            if [[ -f "requirements-dev.txt" ]]; then
                pip install -r requirements-dev.txt
            fi
            
            success "Python dependencies installed in $req_dir ✓"
        )
    done
    
    # Install global Python development tools
    log "Installing global Python development tools..."
    pip3 install --user \
        black \
        flake8 \
        mypy \
        pytest \
        pytest-cov \
        bandit \
        safety \
        pre-commit \
        poetry \
        pipenv
    
    success "Global Python tools installed ✓"
}

# Install Docker dependencies
install_docker_dependencies() {
    log "Installing Docker dependencies..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please run setup-local-env.sh first"
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please run setup-local-env.sh first"
    fi
    
    success "Docker $(docker --version) and Docker Compose $(docker-compose --version) detected ✓"
    
    # Pull base images
    log "Pulling Docker base images..."
    docker pull node:20-alpine
    docker pull python:3.11-slim
    docker pull postgres:15-alpine
    docker pull redis:7-alpine
    docker pull nginx:alpine
    docker pull prom/prometheus:latest
    docker pull grafana/grafana:latest
    docker pull docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    docker pull docker.elastic.co/kibana/kibana:8.11.0
    
    success "Docker base images pulled ✓"
}

# Install system dependencies
install_system_dependencies() {
    log "Installing system dependencies..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            # Debian/Ubuntu
            sudo apt-get update
            sudo apt-get install -y \
                postgresql-client \
                redis-tools \
                curl \
                wget \
                jq \
                htop \
                tree \
                vim \
                nano \
                git \
                unzip \
                zip \
                build-essential \
                ca-certificates \
                gnupg \
                lsb-release \
                software-properties-common
        elif command -v yum &> /dev/null; then
            # RHEL/CentOS
            sudo yum update -y
            sudo yum install -y \
                postgresql \
                redis \
                curl \
                wget \
                jq \
                htop \
                tree \
                vim \
                nano \
                git \
                unzip \
                zip \
                gcc \
                gcc-c++ \
                make
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew update
            brew install \
                postgresql \
                redis \
                curl \
                wget \
                jq \
                htop \
                tree \
                vim \
                nano \
                git
        else
            warning "Homebrew not found. Please install system dependencies manually"
        fi
    fi
    
    success "System dependencies installed ✓"
}

# Create Dockerfiles for development
create_dockerfiles() {
    log "Creating Dockerfiles for development..."
    
    # WS1 Core Foundation Dockerfile
    mkdir -p implementation/WS1_Core_Foundation
    cat > implementation/WS1_Core_Foundation/Dockerfile.dev << 'EOF'
FROM node:20-alpine

WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    postgresql-client \
    curl \
    bash

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["npm", "run", "dev"]
EOF

    # WS2 AI Intelligence Dockerfile
    mkdir -p implementation/WS2_AI_Intelligence
    cat > implementation/WS2_AI_Intelligence/Dockerfile.dev << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

    # WS3 Data Ingestion Dockerfile
    mkdir -p implementation/WS3_Data_Ingestion
    cat > implementation/WS3_Data_Ingestion/Dockerfile.dev << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

    # WS4 Autonomous Capabilities Dockerfile
    mkdir -p implementation/WS4_Autonomous_Capabilities
    cat > implementation/WS4_Autonomous_Capabilities/Dockerfile.dev << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

    # WS5 Frontend Dockerfile
    mkdir -p implementation/WS5_Multi_Role_Interfaces/Phase1_Design_System/nexus-ui-framework
    cat > implementation/WS5_Multi_Role_Interfaces/Phase1_Design_System/nexus-ui-framework/Dockerfile.dev << 'EOF'
FROM node:20-alpine

WORKDIR /app

# Install system dependencies
RUN apk add --no-cache curl bash

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Start application
CMD ["npm", "start"]
EOF

    success "Development Dockerfiles created ✓"
}

# Create package.json files for missing components
create_package_files() {
    log "Creating package.json files for missing components..."
    
    # WS1 Core Foundation package.json
    if [[ ! -f "implementation/WS1_Core_Foundation/package.json" ]]; then
        cat > implementation/WS1_Core_Foundation/package.json << 'EOF'
{
  "name": "nexus-ws1-core-foundation",
  "version": "1.0.0",
  "description": "Nexus Architect WS1 Core Foundation",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js",
    "test": "jest",
    "lint": "eslint .",
    "format": "prettier --write ."
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "helmet": "^7.1.0",
    "morgan": "^1.10.0",
    "dotenv": "^16.3.1",
    "pg": "^8.11.3",
    "redis": "^4.6.10",
    "jsonwebtoken": "^9.0.2",
    "bcryptjs": "^2.4.3",
    "joi": "^17.11.0",
    "winston": "^3.11.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.7.0",
    "eslint": "^8.54.0",
    "prettier": "^3.1.0"
  }
}
EOF
    fi
    
    # Create requirements.txt for Python services
    for ws in WS2_AI_Intelligence WS3_Data_Ingestion WS4_Autonomous_Capabilities; do
        if [[ ! -f "implementation/$ws/requirements.txt" ]]; then
            cat > "implementation/$ws/requirements.txt" << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
psycopg2-binary==2.9.9
redis==5.0.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
python-dotenv==1.0.0
requests==2.31.0
aiofiles==23.2.1
celery==5.3.4
prometheus-client==0.19.0
structlog==23.2.0
EOF
        fi
    done
    
    success "Package files created ✓"
}

# Verify installations
verify_installations() {
    log "Verifying installations..."
    
    # Check Node.js dependencies
    local node_projects=($(find implementation/ -name "package.json" -type f))
    for package_file in "${node_projects[@]}"; do
        local package_dir=$(dirname "$package_file")
        if [[ -d "$package_dir/node_modules" ]]; then
            success "Node.js dependencies verified in $package_dir ✓"
        else
            warning "Node.js dependencies missing in $package_dir"
        fi
    done
    
    # Check Python dependencies
    local python_projects=($(find implementation/ -name "requirements.txt" -type f))
    for req_file in "${python_projects[@]}"; do
        local req_dir=$(dirname "$req_file")
        if [[ -d "$req_dir/venv" ]]; then
            success "Python virtual environment verified in $req_dir ✓"
        else
            warning "Python virtual environment missing in $req_dir"
        fi
    done
    
    # Check Docker images
    local required_images=(
        "node:20-alpine"
        "python:3.11-slim"
        "postgres:15-alpine"
        "redis:7-alpine"
        "nginx:alpine"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
    )
    
    for image in "${required_images[@]}"; do
        if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "$image"; then
            success "Docker image $image verified ✓"
        else
            warning "Docker image $image missing"
        fi
    done
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #4: Automated dependency installation for all components"
    
    check_project_directory
    install_system_dependencies
    create_package_files
    install_nodejs_dependencies
    install_python_dependencies
    install_docker_dependencies
    create_dockerfiles
    verify_installations
    
    success "🎉 All dependencies installed successfully!"
    success "📦 Node.js: Global and project dependencies installed"
    success "🐍 Python: Virtual environments and packages configured"
    success "🐳 Docker: Base images pulled and Dockerfiles created"
    success "🔧 System: Development tools and utilities installed"
    
    log "📋 Next steps:"
    log "   1. Run docker-compose up to start services"
    log "   2. Run database setup script"
    log "   3. Start individual services for development"
}

# Run main function
main "$@"

