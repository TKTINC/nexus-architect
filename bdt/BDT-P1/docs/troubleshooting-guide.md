# Nexus Architect - Troubleshooting Guide

**BDT-P1 Deliverable #15: Detailed troubleshooting procedures**  
**Version:** 1.0  
**Last Updated:** $(date)  
**Author:** Nexus DevOps Team

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Service Issues](#service-issues)
3. [Database Problems](#database-problems)
4. [Authentication Issues](#authentication-issues)
5. [Network & Connectivity](#network--connectivity)
6. [Performance Issues](#performance-issues)
7. [Security Problems](#security-problems)
8. [Development Environment](#development-environment)
9. [Docker & Containerization](#docker--containerization)
10. [Monitoring & Logging](#monitoring--logging)
11. [Emergency Procedures](#emergency-procedures)

## Quick Diagnostics

### Health Check Commands

```bash
# Run comprehensive health check
./bdt/BDT-P1/scripts/run-local-tests.sh health

# Check all services status
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml ps

# Test database connectivity
pg_isready -h localhost -p 5432
redis-cli -h localhost -p 6379 ping

# Check port availability
netstat -tulpn | grep -E ':(3000|3001|8001|8002|5432|6379)'
```

### System Resource Check

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check disk space
df -h
du -sh /home/ubuntu/nexus-dev/*

# Check CPU usage
top -bn1 | grep "Cpu(s)"
ps aux --sort=-%cpu | head -10

# Check network connectivity
ping -c 3 google.com
curl -I http://localhost:3000
```

## Service Issues

### Frontend Services Not Starting

**Symptoms:**
- Browser shows "This site can't be reached"
- Port 3000, 3001, 3002, 3003 not responding
- npm start fails

**Diagnosis:**
```bash
# Check if Node.js is installed
node --version
npm --version

# Check for port conflicts
lsof -i :3000
netstat -tulpn | grep :3000

# Check npm logs
npm run dev 2>&1 | tee debug.log
```

**Solutions:**

1. **Port Conflict Resolution:**
   ```bash
   # Kill process using the port
   kill -9 $(lsof -t -i:3000)
   
   # Or use different port
   PORT=3010 npm run dev
   ```

2. **Node Modules Issues:**
   ```bash
   # Clear npm cache
   npm cache clean --force
   
   # Remove and reinstall dependencies
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Permission Issues:**
   ```bash
   # Fix npm permissions
   sudo chown -R $(whoami) ~/.npm
   sudo chown -R $(whoami) /usr/local/lib/node_modules
   ```

### Backend Services Failing

**Symptoms:**
- API endpoints return 500 errors
- Services crash on startup
- Database connection errors

**Diagnosis:**
```bash
# Check Python environment
python3 --version
pip3 list

# Check service logs
docker logs nexus-api-gateway
tail -f logs/application.log

# Test API endpoints
curl -v http://localhost:8001/health
```

**Solutions:**

1. **Python Environment Issues:**
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Missing Dependencies:**
   ```bash
   # Install missing packages
   pip install -r requirements.txt
   
   # Update pip
   pip install --upgrade pip
   ```

3. **Configuration Issues:**
   ```bash
   # Check environment variables
   env | grep -E '(DATABASE|REDIS|JWT)'
   
   # Validate configuration
   python -c "import config; print(config.DATABASE_URL)"
   ```

## Database Problems

### PostgreSQL Connection Issues

**Symptoms:**
- "Connection refused" errors
- "Role does not exist" errors
- "Database does not exist" errors

**Diagnosis:**
```bash
# Check PostgreSQL status
pg_isready -h localhost -p 5432
docker logs nexus-postgres

# Test connection
psql -h localhost -U postgres -d postgres -c "SELECT version();"

# Check database existence
psql -h localhost -U postgres -l
```

**Solutions:**

1. **Service Not Running:**
   ```bash
   # Start PostgreSQL container
   docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d postgres
   
   # Check container status
   docker ps | grep postgres
   ```

2. **Authentication Issues:**
   ```bash
   # Reset PostgreSQL password
   docker exec -it nexus-postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'postgres';"
   
   # Update .env file
   echo "POSTGRES_PASSWORD=postgres" >> .env
   ```

3. **Database Initialization:**
   ```bash
   # Recreate databases
   ./bdt/BDT-P1/scripts/local-database-setup.sh reset
   
   # Load sample data
   ./bdt/BDT-P1/scripts/local-database-setup.sh load-fixtures
   ```

### Redis Connection Issues

**Symptoms:**
- Redis connection timeouts
- Cache operations failing
- Session storage errors

**Diagnosis:**
```bash
# Check Redis status
redis-cli -h localhost -p 6379 ping
docker logs nexus-redis

# Check Redis configuration
redis-cli -h localhost -p 6379 config get "*"

# Monitor Redis operations
redis-cli -h localhost -p 6379 monitor
```

**Solutions:**

1. **Service Not Running:**
   ```bash
   # Start Redis container
   docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d redis
   
   # Check Redis logs
   docker logs nexus-redis
   ```

2. **Memory Issues:**
   ```bash
   # Check Redis memory usage
   redis-cli -h localhost -p 6379 info memory
   
   # Clear Redis cache
   redis-cli -h localhost -p 6379 flushdb
   ```

3. **Configuration Issues:**
   ```bash
   # Reset Redis configuration
   docker exec -it nexus-redis redis-cli config resetstat
   
   # Restart Redis
   docker restart nexus-redis
   ```

## Authentication Issues

### Keycloak SSO Problems

**Symptoms:**
- Login redirects fail
- Token validation errors
- SSO integration not working

**Diagnosis:**
```bash
# Check Keycloak status
curl -f http://localhost:8080/health/ready

# Test realm configuration
curl http://localhost:8080/auth/realms/nexus

# Check Keycloak logs
docker logs nexus-keycloak
```

**Solutions:**

1. **Service Not Ready:**
   ```bash
   # Wait for Keycloak to start
   while ! curl -f http://localhost:8080/health/ready; do
     echo "Waiting for Keycloak..."
     sleep 5
   done
   ```

2. **Realm Configuration:**
   ```bash
   # Recreate Keycloak configuration
   ./bdt/BDT-P1/security/local-auth-setup.sh reset-keycloak
   
   # Import realm configuration
   docker exec -it nexus-keycloak /opt/jboss/keycloak/bin/standalone.sh \
     -Djboss.socket.binding.port-offset=100 -Dkeycloak.import=/tmp/realm-export.json
   ```

3. **Client Configuration:**
   ```bash
   # Update client secrets
   export KEYCLOAK_CLIENT_SECRET="new-secret"
   
   # Test token endpoint
   curl -X POST http://localhost:8080/auth/realms/nexus/protocol/openid-connect/token \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "grant_type=password&client_id=nexus-frontend&client_secret=$KEYCLOAK_CLIENT_SECRET&username=admin&password=password"
   ```

### LDAP Authentication Issues

**Symptoms:**
- LDAP bind failures
- User authentication errors
- Directory search problems

**Diagnosis:**
```bash
# Test LDAP connection
ldapsearch -x -H "ldap://localhost:389" -b "dc=nexus,dc=dev" -D "cn=admin,dc=nexus,dc=dev" -w "nexus_ldap_admin"

# Check LDAP logs
docker logs nexus-ldap

# Test user authentication
ldapwhoami -x -H "ldap://localhost:389" -D "uid=admin,ou=people,dc=nexus,dc=dev" -w "password"
```

**Solutions:**

1. **Service Issues:**
   ```bash
   # Restart LDAP service
   docker restart nexus-ldap
   
   # Check LDAP configuration
   docker exec -it nexus-ldap slapcat
   ```

2. **User Management:**
   ```bash
   # Reset LDAP users
   ./bdt/BDT-P1/security/local-auth-setup.sh reset-ldap
   
   # Add test users
   ./bdt/BDT-P1/security/local-auth-setup.sh create-test-users
   ```

## Network & Connectivity

### Port Conflicts

**Symptoms:**
- "Address already in use" errors
- Services fail to start
- Connection refused errors

**Diagnosis:**
```bash
# Check port usage
netstat -tulpn | grep -E ':(3000|3001|8001|8002|5432|6379|8080)'
lsof -i :3000

# Check Docker port mappings
docker port nexus-frontend
```

**Solutions:**

1. **Kill Conflicting Processes:**
   ```bash
   # Find and kill process
   kill -9 $(lsof -t -i:3000)
   
   # Or use fuser
   fuser -k 3000/tcp
   ```

2. **Change Port Configuration:**
   ```bash
   # Update docker-compose.yml
   sed -i 's/3000:3000/3010:3000/' bdt/BDT-P1/docker/docker-compose.dev.yml
   
   # Update environment variables
   export FRONTEND_PORT=3010
   ```

### SSL Certificate Issues

**Symptoms:**
- "Certificate not trusted" warnings
- HTTPS connections fail
- SSL handshake errors

**Diagnosis:**
```bash
# Test SSL certificate
openssl s_client -connect localhost:443 -servername localhost

# Check certificate validity
openssl x509 -in ssl/certs/server.crt -text -noout

# Verify certificate chain
openssl verify -CAfile ssl/certs/ca.crt ssl/certs/server.crt
```

**Solutions:**

1. **Regenerate Certificates:**
   ```bash
   # Remove old certificates
   rm -rf ssl/
   
   # Generate new certificates
   ./bdt/BDT-P1/security/local-ssl-setup.sh
   ```

2. **Trust CA Certificate:**
   ```bash
   # macOS
   sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain ssl/certs/ca.crt
   
   # Linux
   sudo cp ssl/certs/ca.crt /usr/local/share/ca-certificates/nexus-ca.crt
   sudo update-ca-certificates
   ```

## Performance Issues

### Slow Application Response

**Symptoms:**
- Pages load slowly
- API responses take too long
- Database queries timeout

**Diagnosis:**
```bash
# Run performance tests
./bdt/BDT-P1/scripts/performance-test-local.sh

# Check system resources
top
iotop
nethogs

# Monitor database performance
psql -h localhost -U postgres -d nexus_dev -c "SELECT * FROM pg_stat_activity;"
```

**Solutions:**

1. **Database Optimization:**
   ```sql
   -- Add missing indexes
   CREATE INDEX idx_users_email ON users(email);
   CREATE INDEX idx_projects_status ON projects(status);
   
   -- Analyze query performance
   EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'user@example.com';
   
   -- Update table statistics
   ANALYZE;
   ```

2. **Application Optimization:**
   ```bash
   # Enable production build
   npm run build
   
   # Use compression
   # Add to nginx configuration
   gzip on;
   gzip_types text/css application/javascript;
   ```

3. **Resource Allocation:**
   ```bash
   # Increase Docker memory limits
   # Update docker-compose.yml
   services:
     postgres:
       deploy:
         resources:
           limits:
             memory: 2G
   ```

### Memory Issues

**Symptoms:**
- Out of memory errors
- System becomes unresponsive
- Services crash randomly

**Diagnosis:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check Docker memory usage
docker stats

# Monitor memory over time
vmstat 1 10
```

**Solutions:**

1. **Increase System Memory:**
   ```bash
   # Add swap space
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Optimize Application Memory:**
   ```bash
   # Reduce Node.js memory usage
   export NODE_OPTIONS="--max-old-space-size=2048"
   
   # Optimize Docker containers
   docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml down
   docker system prune -a
   ```

## Security Problems

### Security Scan Failures

**Symptoms:**
- Vulnerability scan alerts
- Security tests failing
- Compliance check errors

**Diagnosis:**
```bash
# Run security scans
./bdt/BDT-P1/security/security-scan-local.sh

# Check for vulnerabilities
npm audit
pip-audit

# Test SSL configuration
./bdt/BDT-P1/security/local-ssl-setup.sh verify
```

**Solutions:**

1. **Update Dependencies:**
   ```bash
   # Update npm packages
   npm audit fix
   npm update
   
   # Update Python packages
   pip install --upgrade -r requirements.txt
   ```

2. **Fix Security Issues:**
   ```bash
   # Update SSL configuration
   ./bdt/BDT-P1/security/local-ssl-setup.sh regenerate
   
   # Update security headers
   # Add to nginx configuration
   add_header X-Frame-Options DENY;
   add_header X-Content-Type-Options nosniff;
   ```

### Authentication Bypass

**Symptoms:**
- Unauthorized access to protected routes
- JWT token validation failing
- Session management issues

**Diagnosis:**
```bash
# Test authentication
curl -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"password"}'

# Test protected endpoints
curl -H "Authorization: Bearer invalid_token" http://localhost:8001/api/user/profile
```

**Solutions:**

1. **Reset Authentication:**
   ```bash
   # Regenerate JWT secrets
   export JWT_SECRET=$(openssl rand -base64 32)
   
   # Reset authentication services
   ./bdt/BDT-P1/security/local-auth-setup.sh reset
   ```

2. **Update Security Configuration:**
   ```python
   # Update JWT configuration
   JWT_ALGORITHM = 'HS256'
   JWT_EXPIRATION_DELTA = timedelta(hours=1)
   JWT_REFRESH_EXPIRATION_DELTA = timedelta(days=7)
   ```

## Development Environment

### IDE and Editor Issues

**Symptoms:**
- Code completion not working
- Linting errors
- Debugging breakpoints not hit

**Solutions:**

1. **VS Code Configuration:**
   ```json
   // .vscode/settings.json
   {
     "python.defaultInterpreterPath": "./venv/bin/python",
     "eslint.workingDirectories": ["frontend"],
     "typescript.preferences.importModuleSpecifier": "relative"
   }
   ```

2. **ESLint Configuration:**
   ```bash
   # Install ESLint
   npm install -g eslint
   
   # Initialize ESLint
   eslint --init
   ```

### Git Issues

**Symptoms:**
- Merge conflicts
- Large file issues
- Permission problems

**Solutions:**

1. **Resolve Merge Conflicts:**
   ```bash
   # Check conflict status
   git status
   
   # Resolve conflicts manually, then
   git add .
   git commit -m "resolve merge conflicts"
   ```

2. **Large File Handling:**
   ```bash
   # Use Git LFS for large files
   git lfs install
   git lfs track "*.zip"
   git add .gitattributes
   ```

## Docker & Containerization

### Container Startup Issues

**Symptoms:**
- Containers exit immediately
- "No such file or directory" errors
- Permission denied errors

**Diagnosis:**
```bash
# Check container status
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml ps

# Check container logs
docker logs nexus-postgres
docker logs nexus-redis

# Inspect container
docker inspect nexus-postgres
```

**Solutions:**

1. **Fix Dockerfile Issues:**
   ```dockerfile
   # Ensure proper base image
   FROM node:18-alpine
   
   # Set working directory
   WORKDIR /app
   
   # Copy and install dependencies
   COPY package*.json ./
   RUN npm ci --only=production
   ```

2. **Volume Permissions:**
   ```bash
   # Fix volume permissions
   sudo chown -R $(id -u):$(id -g) ./data
   
   # Update docker-compose.yml
   volumes:
     - ./data:/var/lib/postgresql/data:Z
   ```

### Docker Compose Issues

**Symptoms:**
- Services not communicating
- Network connectivity problems
- Environment variable issues

**Solutions:**

1. **Network Configuration:**
   ```yaml
   # docker-compose.yml
   networks:
     nexus-network:
       driver: bridge
   
   services:
     postgres:
       networks:
         - nexus-network
   ```

2. **Environment Variables:**
   ```bash
   # Check environment variables
   docker-compose config
   
   # Update .env file
   echo "DATABASE_URL=postgresql://postgres:postgres@postgres:5432/nexus_dev" >> .env
   ```

## Monitoring & Logging

### Log Analysis

**Symptoms:**
- Missing log entries
- Log rotation issues
- Performance degradation

**Solutions:**

1. **Configure Logging:**
   ```python
   # Python logging configuration
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('app.log'),
           logging.StreamHandler()
       ]
   )
   ```

2. **Log Rotation:**
   ```bash
   # Configure logrotate
   sudo tee /etc/logrotate.d/nexus << EOF
   /home/ubuntu/nexus-dev/logs/*.log {
       daily
       rotate 30
       compress
       delaycompress
       missingok
       notifempty
   }
   EOF
   ```

### Monitoring Stack Issues

**Symptoms:**
- Prometheus not collecting metrics
- Grafana dashboards not loading
- Elasticsearch indexing problems

**Solutions:**

1. **Restart Monitoring Stack:**
   ```bash
   # Restart monitoring services
   ./bdt/BDT-P1/monitoring/monitoring-stack-local.sh restart
   
   # Check service status
   curl http://localhost:9090/targets  # Prometheus
   curl http://localhost:3000/api/health  # Grafana
   ```

2. **Fix Configuration:**
   ```yaml
   # prometheus.yml
   scrape_configs:
     - job_name: 'nexus-api'
       static_configs:
         - targets: ['localhost:8001']
   ```

## Emergency Procedures

### Complete System Reset

**When to use:** When multiple systems are failing and quick recovery is needed.

```bash
# 1. Stop all services
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml down

# 2. Clean Docker resources
docker system prune -a -f
docker volume prune -f

# 3. Reset databases
rm -rf data/postgres data/redis

# 4. Regenerate certificates
rm -rf ssl/
./bdt/BDT-P1/security/local-ssl-setup.sh

# 5. Restart everything
./bdt/BDT-P1/scripts/setup-local-env.sh
```

### Data Recovery

**When to use:** When data corruption or accidental deletion occurs.

```bash
# 1. Stop services
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml down

# 2. List available backups
./bdt/BDT-P1/scripts/backup-restore-local.sh list

# 3. Restore from backup
./bdt/BDT-P1/scripts/backup-restore-local.sh restore BACKUP_ID

# 4. Verify restoration
./bdt/BDT-P1/scripts/integration-test-local.sh database

# 5. Restart services
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d
```

### Performance Emergency

**When to use:** When system performance is severely degraded.

```bash
# 1. Identify resource bottlenecks
top
iotop
nethogs

# 2. Kill resource-intensive processes
pkill -f "node.*memory-intensive"

# 3. Clear caches
redis-cli -h localhost -p 6379 flushdb
npm cache clean --force

# 4. Restart with resource limits
docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d --scale frontend=1

# 5. Monitor recovery
./bdt/BDT-P1/scripts/performance-test-local.sh load
```

---

## Getting Additional Help

### Diagnostic Information Collection

Before seeking help, collect the following information:

```bash
# System information
uname -a
cat /etc/os-release

# Docker information
docker version
docker-compose version

# Service status
./bdt/BDT-P1/scripts/run-local-tests.sh health > diagnostic-report.txt

# Recent logs
tail -100 logs/application.log >> diagnostic-report.txt
docker-compose logs --tail=100 >> diagnostic-report.txt
```

### Support Channels

1. **GitHub Issues:** [https://github.com/TKTINC/nexus-architect/issues](https://github.com/TKTINC/nexus-architect/issues)
2. **Documentation:** [https://docs.nexus-architect.dev](https://docs.nexus-architect.dev)
3. **Community Discord:** [https://discord.gg/nexus-architect](https://discord.gg/nexus-architect)
4. **Stack Overflow:** Tag questions with `nexus-architect`

### Escalation Process

1. **Level 1:** Check this troubleshooting guide
2. **Level 2:** Search GitHub issues and documentation
3. **Level 3:** Create detailed GitHub issue with diagnostic information
4. **Level 4:** Contact development team directly for critical issues

---

**Remember:** Always backup your data before attempting major fixes, and test solutions in a separate environment when possible.

