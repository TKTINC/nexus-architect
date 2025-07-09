#!/bin/bash

# Nexus Architect - Local Development Environment Setup
# BDT-P1 Deliverable #1: Complete local environment automation
# Version: 1.0
# Author: Nexus DevOps Team

set -e  # Exit on any error

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root for security reasons"
fi

log "🚀 Starting Nexus Architect Local Development Environment Setup"

# Check system requirements
check_system_requirements() {
    log "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        error "Unsupported operating system: $OSTYPE"
    fi
    
    # Check available memory (minimum 8GB)
    if [[ "$OS" == "linux" ]]; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    else
        MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    fi
    
    if [[ $MEMORY_GB -lt 8 ]]; then
        warning "System has ${MEMORY_GB}GB RAM. Recommended minimum is 8GB"
    else
        success "System memory: ${MEMORY_GB}GB ✓"
    fi
    
    # Check available disk space (minimum 20GB)
    DISK_GB=$(df -BG . | awk 'NR==2 {print int($4)}')
    if [[ $DISK_GB -lt 20 ]]; then
        error "Insufficient disk space. Available: ${DISK_GB}GB, Required: 20GB"
    else
        success "Available disk space: ${DISK_GB}GB ✓"
    fi
}

# Install Docker and Docker Compose
install_docker() {
    log "Installing Docker and Docker Compose..."
    
    if command -v docker &> /dev/null; then
        success "Docker already installed: $(docker --version)"
    else
        if [[ "$OS" == "linux" ]]; then
            # Install Docker on Linux
            curl -fsSL https://get.docker.com -o get-docker.sh
            sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
        else
            error "Please install Docker Desktop for macOS manually from https://docker.com"
        fi
    fi
    
    if command -v docker-compose &> /dev/null; then
        success "Docker Compose already installed: $(docker-compose --version)"
    else
        # Install Docker Compose
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
}

# Install Node.js and npm
install_nodejs() {
    log "Installing Node.js and npm..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        success "Node.js already installed: $NODE_VERSION"
    else
        # Install Node.js using NodeSource repository
        if [[ "$OS" == "linux" ]]; then
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
            sudo apt-get install -y nodejs
        else
            error "Please install Node.js manually from https://nodejs.org"
        fi
    fi
    
    # Install global packages
    npm install -g pnpm yarn pm2
    success "Global npm packages installed ✓"
}

# Install Python and pip
install_python() {
    log "Installing Python and pip..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        success "Python already installed: $PYTHON_VERSION"
    else
        if [[ "$OS" == "linux" ]]; then
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv
        else
            error "Please install Python manually from https://python.org"
        fi
    fi
    
    # Install global Python packages
    pip3 install --user virtualenv pipenv poetry
    success "Python development tools installed ✓"
}

# Install development tools
install_dev_tools() {
    log "Installing development tools..."
    
    if [[ "$OS" == "linux" ]]; then
        sudo apt-get update
        sudo apt-get install -y \
            git \
            curl \
            wget \
            unzip \
            jq \
            htop \
            tree \
            vim \
            nano \
            build-essential \
            ca-certificates \
            gnupg \
            lsb-release
    fi
    
    # Install kubectl
    if ! command -v kubectl &> /dev/null; then
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        rm kubectl
    fi
    
    # Install Terraform
    if ! command -v terraform &> /dev/null; then
        wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
        sudo apt update && sudo apt install terraform
    fi
    
    success "Development tools installed ✓"
}

# Setup project directories
setup_project_structure() {
    log "Setting up project directory structure..."
    
    # Create main directories
    mkdir -p ~/nexus-dev/{data,logs,certs,backups,scripts}
    mkdir -p ~/nexus-dev/data/{postgres,redis,elasticsearch}
    mkdir -p ~/nexus-dev/logs/{app,nginx,postgres,redis}
    mkdir -p ~/nexus-dev/certs/{ssl,ca}
    
    # Set permissions
    chmod 755 ~/nexus-dev
    chmod 700 ~/nexus-dev/certs
    
    success "Project directory structure created ✓"
}

# Generate SSL certificates for local development
generate_ssl_certs() {
    log "Generating SSL certificates for local development..."
    
    CERT_DIR="$HOME/nexus-dev/certs/ssl"
    
    # Generate CA private key
    openssl genrsa -out "$CERT_DIR/ca-key.pem" 4096
    
    # Generate CA certificate
    openssl req -new -x509 -days 365 -key "$CERT_DIR/ca-key.pem" -sha256 -out "$CERT_DIR/ca.pem" -subj "/C=US/ST=CA/L=San Francisco/O=Nexus Architect/OU=Development/CN=Nexus CA"
    
    # Generate server private key
    openssl genrsa -out "$CERT_DIR/server-key.pem" 4096
    
    # Generate server certificate signing request
    openssl req -subj "/C=US/ST=CA/L=San Francisco/O=Nexus Architect/OU=Development/CN=localhost" -sha256 -new -key "$CERT_DIR/server-key.pem" -out "$CERT_DIR/server.csr"
    
    # Generate server certificate
    echo "subjectAltName = DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:0.0.0.0" > "$CERT_DIR/extfile.cnf"
    openssl x509 -req -days 365 -in "$CERT_DIR/server.csr" -CA "$CERT_DIR/ca.pem" -CAkey "$CERT_DIR/ca-key.pem" -out "$CERT_DIR/server-cert.pem" -extfile "$CERT_DIR/extfile.cnf"
    
    # Clean up
    rm "$CERT_DIR/server.csr" "$CERT_DIR/extfile.cnf"
    
    # Set permissions
    chmod 400 "$CERT_DIR/ca-key.pem" "$CERT_DIR/server-key.pem"
    chmod 444 "$CERT_DIR/ca.pem" "$CERT_DIR/server-cert.pem"
    
    success "SSL certificates generated ✓"
}

# Create environment configuration
create_env_config() {
    log "Creating environment configuration..."
    
    cat > ~/nexus-dev/.env << EOF
# Nexus Architect Local Development Environment
# Generated on $(date)

# Application Configuration
NODE_ENV=development
PYTHON_ENV=development
DEBUG=true
LOG_LEVEL=debug

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=nexus_dev
POSTGRES_USER=nexus_dev
POSTGRES_PASSWORD=nexus_dev_password

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=nexus_redis_password

# Security Configuration
JWT_SECRET=nexus_jwt_secret_key_for_development_only
ENCRYPTION_KEY=nexus_encryption_key_for_development_only
SSL_CERT_PATH=/home/$(whoami)/nexus-dev/certs/ssl/server-cert.pem
SSL_KEY_PATH=/home/$(whoami)/nexus-dev/certs/ssl/server-key.pem

# API Configuration
API_BASE_URL=https://localhost:8000
FRONTEND_URL=https://localhost:3000
ADMIN_URL=https://localhost:3001

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3002
ELASTICSEARCH_PORT=9200
KIBANA_PORT=5601

# Development Tools
HOT_RELOAD=true
AUTO_RESTART=true
WATCH_FILES=true
EOF
    
    success "Environment configuration created ✓"
}

# Install project dependencies
install_project_dependencies() {
    log "Installing project dependencies..."
    
    # Navigate to project root
    cd /home/ubuntu/nexus-architect
    
    # Install dependencies for each workstream
    for ws in implementation/WS*/; do
        if [[ -d "$ws" ]]; then
            log "Installing dependencies for $(basename "$ws")..."
            
            # Find and install Node.js dependencies
            find "$ws" -name "package.json" -type f | while read package_file; do
                package_dir=$(dirname "$package_file")
                log "Installing npm dependencies in $package_dir"
                (cd "$package_dir" && npm install)
            done
            
            # Find and install Python dependencies
            find "$ws" -name "requirements.txt" -type f | while read req_file; do
                req_dir=$(dirname "$req_file")
                log "Installing Python dependencies in $req_dir"
                (cd "$req_dir" && pip3 install -r requirements.txt)
            done
        fi
    done
    
    success "Project dependencies installed ✓"
}

# Create startup script
create_startup_script() {
    log "Creating startup script..."
    
    cat > ~/nexus-dev/start-nexus.sh << 'EOF'
#!/bin/bash

# Nexus Architect Local Development Startup Script

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log "🚀 Starting Nexus Architect Local Development Environment"

# Load environment variables
if [[ -f ~/nexus-dev/.env ]]; then
    source ~/nexus-dev/.env
    log "Environment variables loaded ✓"
fi

# Start Docker services
log "Starting Docker services..."
cd /home/ubuntu/nexus-architect/bdt/BDT-P1/docker
docker-compose up -d

# Wait for services to be ready
log "Waiting for services to be ready..."
sleep 30

# Check service health
log "Checking service health..."
docker-compose ps

success "🎉 Nexus Architect Local Development Environment is ready!"
success "🌐 Frontend: https://localhost:3000"
success "🔧 Admin: https://localhost:3001"
success "📊 Grafana: https://localhost:3002"
success "🔍 Kibana: https://localhost:5601"
success "📈 Prometheus: https://localhost:9090"

EOF
    
    chmod +x ~/nexus-dev/start-nexus.sh
    success "Startup script created ✓"
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #1: Complete Local Environment Automation"
    
    check_system_requirements
    install_docker
    install_nodejs
    install_python
    install_dev_tools
    setup_project_structure
    generate_ssl_certs
    create_env_config
    install_project_dependencies
    create_startup_script
    
    success "🎉 Local development environment setup completed successfully!"
    success "📁 Configuration directory: ~/nexus-dev"
    success "🚀 Start environment: ~/nexus-dev/start-nexus.sh"
    success "📖 Next step: Run docker-compose setup and database initialization"
    
    warning "⚠️  Please log out and log back in to apply Docker group permissions"
}

# Run main function
main "$@"

