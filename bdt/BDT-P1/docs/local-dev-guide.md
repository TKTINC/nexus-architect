# Nexus Architect - Local Development Guide

**BDT-P1 Deliverable #14: Comprehensive local development guide**  
**Version:** 1.0  
**Last Updated:** $(date)  
**Author:** Nexus DevOps Team

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Environment Setup](#environment-setup)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Debugging](#debugging)
8. [Performance Optimization](#performance-optimization)
9. [Security Guidelines](#security-guidelines)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

## Overview

This guide provides comprehensive instructions for setting up and working with the Nexus Architect local development environment. The local environment includes all components necessary for full-stack development, testing, and validation.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Nexus Architect Local Environment        │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer                                             │
│  ├── Executive Dashboard (Port 3000)                        │
│  ├── Developer Tools (Port 3001)                           │
│  ├── Project Management (Port 3002)                        │
│  └── Mobile Interface (Port 3003)                          │
├─────────────────────────────────────────────────────────────┤
│  Backend Services                                           │
│  ├── Core API (Port 8001)                                  │
│  ├── Authentication Service (Port 8002)                    │
│  ├── Analytics Engine (Port 8003)                          │
│  └── Workflow Engine (Port 8004)                           │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── PostgreSQL (Port 5432)                                │
│  ├── Redis (Port 6379)                                     │
│  └── Elasticsearch (Port 9200)                             │
├─────────────────────────────────────────────────────────────┤
│  Security & Auth                                            │
│  ├── Keycloak SSO (Port 8080)                              │
│  ├── OpenLDAP (Port 389)                                   │
│  └── SSL/TLS Certificates                                  │
├─────────────────────────────────────────────────────────────┤
│  Monitoring & Observability                                │
│  ├── Prometheus (Port 9090)                                │
│  ├── Grafana (Port 3000)                                   │
│  ├── Elasticsearch (Port 9200)                             │
│  └── Kibana (Port 5601)                                    │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements

- **Operating System:** Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+ with WSL2
- **Memory:** Minimum 16GB RAM (32GB recommended)
- **Storage:** Minimum 50GB free space (SSD recommended)
- **CPU:** 4+ cores (8+ cores recommended)
- **Network:** Stable internet connection for initial setup

### Required Software

1. **Docker & Docker Compose**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   
   # macOS
   brew install docker docker-compose
   
   # Or install Docker Desktop
   ```

2. **Node.js & npm**
   ```bash
   # Install Node.js 18+ and npm
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   
   # Verify installation
   node --version
   npm --version
   ```

3. **Python 3.8+**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3 python3-pip
   
   # macOS
   brew install python3
   
   # Verify installation
   python3 --version
   pip3 --version
   ```

4. **Git**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git
   
   # macOS
   brew install git
   
   # Configure Git
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

5. **Additional Tools**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install curl wget jq postgresql-client redis-tools
   
   # macOS
   brew install curl wget jq postgresql redis
   ```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/TKTINC/nexus-architect.git
cd nexus-architect
```

### 2. Run the Setup Script

```bash
# Make the setup script executable
chmod +x bdt/BDT-P1/scripts/setup-local-env.sh

# Run the complete setup
./bdt/BDT-P1/scripts/setup-local-env.sh
```

This script will:
- Install all dependencies
- Set up Docker containers
- Initialize databases
- Configure SSL certificates
- Start all services
- Run initial tests

### 3. Verify Installation

```bash
# Check service health
./bdt/BDT-P1/scripts/run-local-tests.sh health

# Run integration tests
./bdt/BDT-P1/scripts/integration-test-local.sh

# Access the applications
open http://localhost:3000  # Executive Dashboard
open http://localhost:3001  # Developer Tools
```

## Environment Setup

### Manual Setup (Alternative to Quick Start)

If you prefer to set up the environment manually or need to customize the setup:

#### 1. Install Dependencies

```bash
# Install Node.js dependencies for all frontend applications
./bdt/BDT-P1/scripts/install-dependencies.sh frontend

# Install Python dependencies for backend services
./bdt/BDT-P1/scripts/install-dependencies.sh backend

# Install development tools
./bdt/BDT-P1/scripts/install-dependencies.sh tools
```

#### 2. Database Setup

```bash
# Start database containers
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d postgres redis

# Initialize databases with sample data
./bdt/BDT-P1/scripts/local-database-setup.sh
```

#### 3. SSL Configuration

```bash
# Generate SSL certificates for local development
./bdt/BDT-P1/security/local-ssl-setup.sh

# Verify SSL setup
curl -k https://localhost:3000
```

#### 4. Authentication Setup

```bash
# Set up Keycloak and LDAP for authentication testing
./bdt/BDT-P1/security/local-auth-setup.sh

# Test authentication
curl -X POST http://localhost:8080/auth/realms/nexus/protocol/openid-connect/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password&client_id=nexus-frontend&username=admin&password=password"
```

#### 5. Monitoring Setup

```bash
# Start monitoring stack
./bdt/BDT-P1/monitoring/monitoring-stack-local.sh

# Access monitoring dashboards
open http://localhost:9090  # Prometheus
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:5601  # Kibana
```

### Environment Variables

Create a `.env` file in the project root with the following variables:

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

## Development Workflow

### Starting the Development Environment

```bash
# Start all services
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d

# Start frontend applications in development mode
cd implementation/WS5_Multi_Role_Interfaces/Phase1_Design_System/nexus-ui-framework
npm run dev

cd ../Phase2_Executive_Dashboard/executive-dashboard
npm run dev

cd ../Phase3_Developer_Tools/developer-dashboard/nexus-dev-dashboard
npm run dev
```

### Code Development

#### Frontend Development

1. **React Applications**
   ```bash
   # Navigate to the specific application
   cd implementation/WS5_Multi_Role_Interfaces/Phase2_Executive_Dashboard/executive-dashboard
   
   # Install dependencies
   npm install
   
   # Start development server
   npm run dev
   
   # Run tests
   npm test
   
   # Build for production
   npm run build
   ```

2. **Shared Components**
   ```bash
   # Work on shared UI components
   cd implementation/WS5_Multi_Role_Interfaces/Phase1_Design_System/nexus-ui-framework
   
   # Start Storybook for component development
   npm run storybook
   ```

#### Backend Development

1. **Python Services**
   ```bash
   # Navigate to backend service
   cd implementation/WS1_Core_Infrastructure/Phase2_Backend_Services/api-gateway
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start development server
   python app.py
   
   # Run tests
   pytest
   ```

2. **API Development**
   ```bash
   # Test API endpoints
   curl -X GET http://localhost:8001/api/health
   
   # Use Postman collection for comprehensive testing
   newman run tests/postman/nexus-api-tests.json
   ```

### Hot Reloading

All development servers support hot reloading:

- **Frontend:** Changes to React components automatically refresh the browser
- **Backend:** Python services restart automatically on file changes
- **Styles:** CSS/SCSS changes apply immediately
- **Configuration:** Environment variable changes require service restart

### Database Development

#### PostgreSQL

```bash
# Connect to PostgreSQL
psql -h localhost -U postgres -d nexus_dev

# Run migrations
python manage.py migrate

# Create sample data
python manage.py loaddata fixtures/sample_data.json

# Backup development database
pg_dump -h localhost -U postgres nexus_dev > backup.sql
```

#### Redis

```bash
# Connect to Redis
redis-cli -h localhost -p 6379

# Monitor Redis operations
redis-cli -h localhost -p 6379 monitor

# Clear Redis cache
redis-cli -h localhost -p 6379 flushdb
```

## Testing

### Running Tests

#### Unit Tests

```bash
# Frontend unit tests
cd implementation/WS5_Multi_Role_Interfaces/Phase2_Executive_Dashboard/executive-dashboard
npm test

# Backend unit tests
cd implementation/WS1_Core_Infrastructure/Phase2_Backend_Services/api-gateway
pytest tests/unit/
```

#### Integration Tests

```bash
# Run comprehensive integration tests
./bdt/BDT-P1/scripts/integration-test-local.sh

# Run specific test categories
./bdt/BDT-P1/scripts/integration-test-local.sh api
./bdt/BDT-P1/scripts/integration-test-local.sh frontend
./bdt/BDT-P1/scripts/integration-test-local.sh database
```

#### Performance Tests

```bash
# Run performance testing suite
./bdt/BDT-P1/scripts/performance-test-local.sh

# Run specific performance tests
./bdt/BDT-P1/scripts/performance-test-local.sh load
./bdt/BDT-P1/scripts/performance-test-local.sh stress
```

#### End-to-End Tests

```bash
# Run Cypress E2E tests
cd tests/e2e
npm install
npm run cypress:run

# Run specific test suites
npm run cypress:run --spec "cypress/integration/auth.spec.js"
```

### Test Data Management

```bash
# Reset test databases
./bdt/BDT-P1/scripts/local-database-setup.sh reset

# Load specific test fixtures
./bdt/BDT-P1/scripts/local-database-setup.sh load-fixtures user-management

# Create test data snapshots
./bdt/BDT-P1/scripts/backup-restore-local.sh backup
```

## Debugging

### Frontend Debugging

#### React DevTools

1. Install React DevTools browser extension
2. Open browser developer tools
3. Navigate to "React" tab
4. Inspect component state and props

#### Browser Debugging

```javascript
// Add breakpoints in code
debugger;

// Console logging
console.log('Debug info:', data);
console.table(arrayData);
console.group('API Response');
console.log('Status:', response.status);
console.log('Data:', response.data);
console.groupEnd();
```

#### Network Debugging

```bash
# Monitor network requests
# Open browser DevTools > Network tab

# Test API endpoints directly
curl -v http://localhost:8001/api/health

# Monitor WebSocket connections
wscat -c ws://localhost:8001/ws
```

### Backend Debugging

#### Python Debugging

```python
# Add breakpoints in Python code
import pdb; pdb.set_trace()

# Or use ipdb for enhanced debugging
import ipdb; ipdb.set_trace()

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug('Debug message')
```

#### API Debugging

```bash
# Enable debug mode in Flask
export FLASK_DEBUG=1
export FLASK_ENV=development

# View detailed error traces
curl -v http://localhost:8001/api/endpoint

# Monitor API logs
docker logs -f nexus-api-gateway
```

### Database Debugging

#### PostgreSQL

```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
SELECT pg_reload_conf();

-- Monitor slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check active connections
SELECT * FROM pg_stat_activity;
```

#### Redis

```bash
# Monitor Redis commands
redis-cli -h localhost -p 6379 monitor

# Check Redis memory usage
redis-cli -h localhost -p 6379 info memory

# Debug Redis keys
redis-cli -h localhost -p 6379 keys "*"
```

### Performance Debugging

```bash
# Profile application performance
npm run build:analyze  # Frontend bundle analysis

# Monitor system resources
htop
iotop
nethogs

# Database performance
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'user@example.com';
```

## Performance Optimization

### Frontend Optimization

#### Bundle Optimization

```bash
# Analyze bundle size
npm run build:analyze

# Optimize images
npm install imagemin-cli -g
imagemin src/assets/images/* --out-dir=src/assets/images/optimized

# Enable compression
# Add to nginx.conf or server configuration
gzip on;
gzip_types text/css application/javascript application/json;
```

#### Code Splitting

```javascript
// Implement lazy loading
const LazyComponent = React.lazy(() => import('./LazyComponent'));

// Route-based code splitting
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const Projects = React.lazy(() => import('./pages/Projects'));
```

#### Caching Strategies

```javascript
// Service Worker for caching
// Register in index.js
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js');
}

// API response caching
const cache = new Map();
const getCachedData = (key) => {
  if (cache.has(key)) {
    return cache.get(key);
  }
  // Fetch and cache data
};
```

### Backend Optimization

#### Database Optimization

```sql
-- Add indexes for frequently queried columns
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_projects_status ON projects(status);

-- Optimize queries
EXPLAIN ANALYZE SELECT * FROM users 
JOIN projects ON users.id = projects.owner_id 
WHERE users.active = true;
```

#### Caching

```python
# Redis caching
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

def get_user_data(user_id):
    cache_key = f"user:{user_id}"
    cached_data = r.get(cache_key)
    
    if cached_data:
        return json.loads(cached_data)
    
    # Fetch from database
    user_data = fetch_user_from_db(user_id)
    r.setex(cache_key, 3600, json.dumps(user_data))  # Cache for 1 hour
    return user_data
```

#### Connection Pooling

```python
# PostgreSQL connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:password@localhost/nexus_dev',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)
```

### Monitoring Performance

```bash
# Set up performance monitoring
./bdt/BDT-P1/monitoring/monitoring-stack-local.sh

# View performance metrics
open http://localhost:3000  # Grafana dashboards

# Monitor application metrics
curl http://localhost:8001/metrics  # Prometheus metrics
```

## Security Guidelines

### Development Security

#### Environment Security

```bash
# Never commit sensitive data
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
echo "*.pem" >> .gitignore

# Use environment variables for secrets
export DATABASE_PASSWORD="secure_password"
export JWT_SECRET="your-secret-key"
```

#### Code Security

```javascript
// Input validation
const validateEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// Sanitize user input
import DOMPurify from 'dompurify';
const cleanHTML = DOMPurify.sanitize(userInput);
```

#### API Security

```python
# Input validation
from marshmallow import Schema, fields, validate

class UserSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=8))

# Rate limiting
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@app.route('/api/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    # Login logic
    pass
```

### Security Testing

```bash
# Run security scans
./bdt/BDT-P1/security/security-scan-local.sh

# Check for vulnerabilities
npm audit
pip-audit

# Test SSL configuration
./bdt/BDT-P1/security/local-ssl-setup.sh verify
```

## Troubleshooting

### Common Issues

#### Port Conflicts

```bash
# Check which process is using a port
lsof -i :3000
netstat -tulpn | grep :3000

# Kill process using port
kill -9 $(lsof -t -i:3000)
```

#### Docker Issues

```bash
# Restart Docker services
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml restart

# Clean up Docker resources
docker system prune -a
docker volume prune

# Check Docker logs
docker logs nexus-postgres
docker logs nexus-redis
```

#### Database Connection Issues

```bash
# Test PostgreSQL connection
pg_isready -h localhost -p 5432

# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Reset database
./bdt/BDT-P1/scripts/local-database-setup.sh reset
```

#### SSL Certificate Issues

```bash
# Regenerate SSL certificates
./bdt/BDT-P1/security/local-ssl-setup.sh regenerate

# Trust certificates (macOS)
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain ssl/certs/ca.crt
```

### Getting Help

1. **Check Logs**
   ```bash
   # Application logs
   tail -f logs/application.log
   
   # Docker logs
   docker-compose logs -f
   
   # System logs
   journalctl -f
   ```

2. **Run Diagnostics**
   ```bash
   # Health checks
   ./bdt/BDT-P1/scripts/run-local-tests.sh health
   
   # Integration tests
   ./bdt/BDT-P1/scripts/integration-test-local.sh
   ```

3. **Community Support**
   - GitHub Issues: [https://github.com/TKTINC/nexus-architect/issues](https://github.com/TKTINC/nexus-architect/issues)
   - Documentation: [https://docs.nexus-architect.dev](https://docs.nexus-architect.dev)
   - Discord: [https://discord.gg/nexus-architect](https://discord.gg/nexus-architect)

## Best Practices

### Code Quality

```bash
# Use linting and formatting
npm run lint
npm run format

# Pre-commit hooks
npm install husky lint-staged --save-dev
```

### Git Workflow

```bash
# Feature branch workflow
git checkout -b feature/new-feature
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# Create pull request for code review
```

### Testing Strategy

```bash
# Test-driven development
# 1. Write failing test
# 2. Write minimal code to pass
# 3. Refactor

# Maintain test coverage
npm run test:coverage
```

### Documentation

```markdown
# Keep documentation updated
# - README files for each component
# - API documentation
# - Code comments for complex logic
# - Architecture decision records (ADRs)
```

### Performance

```bash
# Regular performance testing
./bdt/BDT-P1/scripts/performance-test-local.sh

# Monitor bundle sizes
npm run build:analyze

# Database query optimization
EXPLAIN ANALYZE your_query;
```

---

**Next Steps:**
1. Complete the environment setup using this guide
2. Review the [Troubleshooting Guide](troubleshooting-guide.md)
3. Check the [Security Checklist](security-checklist.md)
4. Run performance benchmarks using [Performance Benchmarks](performance-benchmarks.md)

**Support:**
For additional help, refer to the troubleshooting guide or contact the development team through the official channels listed above.

