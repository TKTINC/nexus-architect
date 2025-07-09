# Nexus Architect - Local Deployment Guide

**BDT-P1 Deliverable #19: Deployment instructions**  
**Version:** 1.0  
**Last Updated:** $(date)  
**Author:** Nexus DevOps Team

## Quick Start Deployment

### One-Command Deployment

```bash
# Clone and deploy in one command
git clone https://github.com/TKTINC/nexus-architect.git && \
cd nexus-architect && \
chmod +x bdt/BDT-P1/scripts/setup-local-env.sh && \
./bdt/BDT-P1/scripts/setup-local-env.sh
```

### Verification

```bash
# Verify deployment success
./bdt/BDT-P1/scripts/run-local-tests.sh health
open http://localhost:3000  # Executive Dashboard
```

---

## Detailed Deployment Instructions

### Prerequisites

#### System Requirements
- **OS:** Ubuntu 20.04+, macOS 10.15+, or Windows 10+ with WSL2
- **CPU:** 4+ cores (8+ recommended)
- **Memory:** 16GB+ (32GB recommended)
- **Storage:** 50GB+ free space (SSD recommended)
- **Network:** Stable internet connection

#### Required Software
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y \
  docker.io docker-compose \
  nodejs npm \
  python3 python3-pip \
  git curl wget jq \
  postgresql-client redis-tools

# macOS (with Homebrew)
brew install docker docker-compose node python3 git postgresql redis

# Verify installations
docker --version && docker-compose --version
node --version && npm --version
python3 --version && pip3 --version
```

### Step-by-Step Deployment

#### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/TKTINC/nexus-architect.git
cd nexus-architect

# Verify repository structure
ls -la
echo "Repository cloned successfully ✅"
```

#### 2. Environment Configuration

```bash
# Create environment file
cp .env.example .env

# Edit environment variables (optional - defaults work for local development)
nano .env

# Verify environment setup
source .env
echo "Environment configured ✅"
```

#### 3. Dependency Installation

```bash
# Install all dependencies
./bdt/BDT-P1/scripts/install-dependencies.sh

# Verify dependency installation
echo "Dependencies installed ✅"
```

#### 4. SSL Certificate Setup

```bash
# Generate SSL certificates for local development
./bdt/BDT-P1/security/local-ssl-setup.sh

# Verify SSL setup
ls -la ssl/certs/
echo "SSL certificates generated ✅"
```

#### 5. Database Initialization

```bash
# Start database containers
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d postgres redis

# Wait for databases to be ready
sleep 30

# Initialize databases with sample data
./bdt/BDT-P1/scripts/local-database-setup.sh

# Verify database setup
pg_isready -h localhost -p 5432
redis-cli -h localhost -p 6379 ping
echo "Databases initialized ✅"
```

#### 6. Authentication Services

```bash
# Start authentication services
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d keycloak ldap

# Wait for services to be ready
sleep 60

# Configure authentication
./bdt/BDT-P1/security/local-auth-setup.sh

# Verify authentication setup
curl -f http://localhost:8080/auth/realms/nexus
echo "Authentication services configured ✅"
```

#### 7. Monitoring Stack

```bash
# Deploy monitoring stack
./bdt/BDT-P1/monitoring/monitoring-stack-local.sh

# Verify monitoring services
curl -f http://localhost:9090/-/healthy  # Prometheus
curl -f http://localhost:3000/api/health  # Grafana
curl -f http://localhost:9200/_cluster/health  # Elasticsearch
echo "Monitoring stack deployed ✅"
```

#### 8. Application Services

```bash
# Start all application services
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d

# Start frontend applications
cd implementation/WS5_Multi_Role_Interfaces/Phase2_Executive_Dashboard/executive-dashboard
npm install && npm run dev &

cd ../Phase3_Developer_Tools/developer-dashboard/nexus-dev-dashboard
npm install && npm run dev &

cd ../Phase4_Project_Management/project-management-dashboard
npm install && npm run dev &

cd ../Phase5_Mobile_Applications/mobile-app/nexus-mobile-app
npm install && npm run dev &

# Return to project root
cd /home/ubuntu/nexus-architect

echo "Application services started ✅"
```

#### 9. Health Verification

```bash
# Run comprehensive health check
./bdt/BDT-P1/scripts/run-local-tests.sh health

# Verify all services are responding
curl -f http://localhost:3000  # Executive Dashboard
curl -f http://localhost:3001  # Developer Tools
curl -f http://localhost:3002  # Project Management
curl -f http://localhost:3003  # Mobile Interface
curl -f http://localhost:8001/health  # Core API

echo "Health verification completed ✅"
```

#### 10. Integration Testing

```bash
# Run integration tests
./bdt/BDT-P1/scripts/integration-test-local.sh

# Run performance tests
./bdt/BDT-P1/scripts/performance-test-local.sh quick

echo "Integration testing completed ✅"
```

---

## Service Access Information

### Frontend Applications

| Application | URL | Description |
|-------------|-----|-------------|
| Executive Dashboard | http://localhost:3000 | Strategic insights and KPIs |
| Developer Tools | http://localhost:3001 | Development workflow and IDE integration |
| Project Management | http://localhost:3002 | Project tracking and team collaboration |
| Mobile Interface | http://localhost:3003 | Mobile-optimized interface |

### Backend Services

| Service | URL | Description |
|---------|-----|-------------|
| Core API | http://localhost:8001 | Main application API |
| Authentication API | http://localhost:8002 | User authentication service |
| Analytics API | http://localhost:8003 | Analytics and reporting service |
| Workflow API | http://localhost:8004 | Workflow automation service |

### Infrastructure Services

| Service | URL | Credentials | Description |
|---------|-----|-------------|-------------|
| PostgreSQL | localhost:5432 | postgres/postgres | Primary database |
| Redis | localhost:6379 | (no auth) | Cache and session store |
| Keycloak | http://localhost:8080 | admin/admin | SSO and identity management |
| OpenLDAP | localhost:389 | cn=admin,dc=nexus,dc=dev/nexus_ldap_admin | Directory service |

### Monitoring Services

| Service | URL | Credentials | Description |
|---------|-----|-------------|-------------|
| Prometheus | http://localhost:9090 | (no auth) | Metrics collection |
| Grafana | http://localhost:3000 | admin/admin | Metrics visualization |
| Elasticsearch | http://localhost:9200 | (no auth) | Log storage |
| Kibana | http://localhost:5601 | (no auth) | Log visualization |

---

## Configuration Options

### Environment Variables

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=nexus_dev
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

REDIS_HOST=localhost
REDIS_PORT=6379

# Authentication
KEYCLOAK_URL=http://localhost:8080
KEYCLOAK_REALM=nexus
KEYCLOAK_CLIENT_ID=nexus-frontend
KEYCLOAK_CLIENT_SECRET=nexus-frontend-secret

LDAP_URL=ldap://localhost:389
LDAP_BIND_DN=cn=admin,dc=nexus,dc=dev
LDAP_BIND_PASSWORD=nexus_ldap_admin

# API Configuration
API_BASE_URL=http://localhost:8001
JWT_SECRET=your-jwt-secret-key-here

# SSL Configuration
SSL_CERT_PATH=./ssl/certs/server.crt
SSL_KEY_PATH=./ssl/private/server.key

# Development Settings
NODE_ENV=development
DEBUG=true
LOG_LEVEL=debug

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
ELASTICSEARCH_URL=http://localhost:9200
```

### Docker Compose Overrides

Create `docker-compose.override.yml` for custom configurations:

```yaml
version: '3.8'

services:
  postgres:
    environment:
      - POSTGRES_PASSWORD=custom_password
    volumes:
      - ./custom-data:/var/lib/postgresql/data

  redis:
    command: redis-server --requirepass custom_redis_password

  nginx:
    ports:
      - "8080:80"  # Custom port mapping
```

### Port Customization

```bash
# Change default ports if conflicts exist
export FRONTEND_PORT=3010
export API_PORT=8010
export POSTGRES_PORT=5433
export REDIS_PORT=6380

# Update docker-compose.yml accordingly
sed -i "s/3000:3000/${FRONTEND_PORT}:3000/" bdt/BDT-P1/docker/docker-compose.dev.yml
```

---

## Deployment Modes

### Development Mode (Default)

```bash
# Full development environment with hot reloading
./bdt/BDT-P1/scripts/setup-local-env.sh

# Features:
# - Hot reloading enabled
# - Debug logging
# - Development tools accessible
# - Sample data loaded
```

### Testing Mode

```bash
# Optimized for testing with minimal resources
NODE_ENV=test ./bdt/BDT-P1/scripts/setup-local-env.sh

# Features:
# - Reduced resource allocation
# - Test data sets
# - Faster startup
# - Automated testing enabled
```

### Demo Mode

```bash
# Production-like environment for demonstrations
NODE_ENV=demo ./bdt/BDT-P1/scripts/setup-local-env.sh

# Features:
# - Production builds
# - Realistic data sets
# - Performance optimizations
# - Monitoring enabled
```

### Minimal Mode

```bash
# Lightweight deployment for resource-constrained environments
DEPLOYMENT_MODE=minimal ./bdt/BDT-P1/scripts/setup-local-env.sh

# Features:
# - Core services only
# - Reduced memory usage
# - Essential features only
# - Faster deployment
```

---

## Troubleshooting Deployment

### Common Issues

#### 1. Port Conflicts

```bash
# Check for port conflicts
netstat -tulpn | grep -E ':(3000|3001|8001|5432|6379)'

# Kill conflicting processes
sudo fuser -k 3000/tcp
sudo fuser -k 8001/tcp

# Or use alternative ports
PORT=3010 npm run dev
```

#### 2. Docker Issues

```bash
# Docker daemon not running
sudo systemctl start docker
sudo systemctl enable docker

# Permission issues
sudo usermod -aG docker $USER
newgrp docker

# Clean up Docker resources
docker system prune -a
docker volume prune
```

#### 3. Database Connection Issues

```bash
# PostgreSQL not ready
docker logs nexus-postgres
docker restart nexus-postgres

# Redis connection failed
docker logs nexus-redis
redis-cli -h localhost -p 6379 ping

# Reset databases
./bdt/BDT-P1/scripts/local-database-setup.sh reset
```

#### 4. SSL Certificate Issues

```bash
# Certificate generation failed
rm -rf ssl/
./bdt/BDT-P1/security/local-ssl-setup.sh

# Trust certificate (macOS)
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain ssl/certs/ca.crt

# Trust certificate (Linux)
sudo cp ssl/certs/ca.crt /usr/local/share/ca-certificates/nexus-ca.crt
sudo update-ca-certificates
```

#### 5. Memory Issues

```bash
# Insufficient memory
free -h
sudo swapon --show

# Add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Reduce Docker memory limits
# Edit docker-compose.yml to add memory limits
```

### Diagnostic Commands

```bash
# System health check
./bdt/BDT-P1/scripts/run-local-tests.sh health

# Service status
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml ps

# Resource usage
docker stats --no-stream
top -bn1 | head -20

# Network connectivity
curl -I http://localhost:3000
curl -I http://localhost:8001/health

# Log analysis
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml logs --tail=50
tail -f logs/application.log
```

---

## Deployment Automation

### Automated Deployment Script

```bash
#!/bin/bash
# deploy-nexus-local.sh

set -e  # Exit on any error

echo "🚀 Starting Nexus Architect Local Deployment"
echo "============================================="

# Check prerequisites
echo "Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }

# Clone repository if not exists
if [ ! -d "nexus-architect" ]; then
    echo "Cloning repository..."
    git clone https://github.com/TKTINC/nexus-architect.git
fi

cd nexus-architect

# Run deployment
echo "Running deployment..."
chmod +x bdt/BDT-P1/scripts/setup-local-env.sh
./bdt/BDT-P1/scripts/setup-local-env.sh

# Verify deployment
echo "Verifying deployment..."
./bdt/BDT-P1/scripts/run-local-tests.sh health

echo "🎉 Deployment completed successfully!"
echo "Access the applications at:"
echo "  - Executive Dashboard: http://localhost:3000"
echo "  - Developer Tools: http://localhost:3001"
echo "  - Project Management: http://localhost:3002"
echo "  - Mobile Interface: http://localhost:3003"
```

### CI/CD Integration

```yaml
# .github/workflows/local-deployment-test.yml
name: Local Deployment Test

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-local-deployment:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker
      uses: docker/setup-buildx-action@v2
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Run deployment
      run: |
        chmod +x bdt/BDT-P1/scripts/setup-local-env.sh
        ./bdt/BDT-P1/scripts/setup-local-env.sh
    
    - name: Run health checks
      run: |
        ./bdt/BDT-P1/scripts/run-local-tests.sh health
    
    - name: Run integration tests
      run: |
        ./bdt/BDT-P1/scripts/integration-test-local.sh
```

---

## Maintenance and Updates

### Regular Maintenance

```bash
# Weekly maintenance script
#!/bin/bash
# weekly-maintenance.sh

echo "🔧 Running weekly maintenance..."

# Update dependencies
./bdt/BDT-P1/scripts/install-dependencies.sh update

# Clean up Docker resources
docker system prune -f
docker volume prune -f

# Backup databases
./bdt/BDT-P1/scripts/backup-restore-local.sh backup

# Run security scans
./bdt/BDT-P1/security/security-scan-local.sh

# Performance testing
./bdt/BDT-P1/scripts/performance-test-local.sh quick

echo "✅ Weekly maintenance completed"
```

### Updates and Upgrades

```bash
# Update to latest version
git pull origin main

# Update dependencies
npm update
pip install --upgrade -r requirements.txt

# Rebuild containers
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml build --no-cache

# Restart services
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml restart

# Verify updates
./bdt/BDT-P1/scripts/run-local-tests.sh health
```

---

## Support and Documentation

### Getting Help

1. **Documentation:** Check the comprehensive guides in `bdt/BDT-P1/docs/`
2. **Troubleshooting:** Review `troubleshooting-guide.md` for common issues
3. **GitHub Issues:** Create an issue at https://github.com/TKTINC/nexus-architect/issues
4. **Community:** Join the Discord server for community support

### Additional Resources

- **Local Development Guide:** `bdt/BDT-P1/docs/local-dev-guide.md`
- **Security Checklist:** `bdt/BDT-P1/docs/security-checklist.md`
- **Performance Benchmarks:** `bdt/BDT-P1/docs/performance-benchmarks.md`
- **Validation Checklist:** `bdt/BDT-P1/docs/validation-checklist.md`

---

## Deployment Checklist

### Pre-Deployment

- [ ] System requirements verified ✅
- [ ] Required software installed ✅
- [ ] Network connectivity confirmed ✅
- [ ] Sufficient disk space available ✅

### During Deployment

- [ ] Repository cloned successfully ✅
- [ ] Environment configured ✅
- [ ] Dependencies installed ✅
- [ ] SSL certificates generated ✅
- [ ] Databases initialized ✅
- [ ] Services started ✅

### Post-Deployment

- [ ] Health checks passed ✅
- [ ] Integration tests passed ✅
- [ ] Performance tests passed ✅
- [ ] Security scans passed ✅
- [ ] Documentation reviewed ✅

### Success Criteria

✅ **Deployment Successful When:**
- All services respond to health checks
- Frontend applications load without errors
- API endpoints return expected responses
- Database connections established
- Authentication services functional
- Monitoring stack operational

---

**🎉 Congratulations! Your Nexus Architect local development environment is now deployed and ready for use.**

For next steps, proceed to BDT-P2 (Staging Environment & Enterprise Integration) when ready to deploy to staging infrastructure.

