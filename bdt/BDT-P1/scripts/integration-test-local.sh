#!/bin/bash

# Nexus Architect - Integration Testing Suite
# BDT-P1 Deliverable #12: Integration testing suite
# Version: 1.0
# Author: Nexus DevOps Team

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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
}

info() {
    echo -e "${PURPLE}[INFO]${NC} $1"
}

# Integration testing configuration
INTEGRATION_TEST_DIR="$HOME/nexus-dev/integration-tests"
TEST_RESULTS_DIR="$INTEGRATION_TEST_DIR/results"
TEST_DATE=$(date +%Y%m%d_%H%M%S)
TEST_REPORT="$TEST_RESULTS_DIR/integration-test-report-$TEST_DATE.html"

# Test configuration
FRONTEND_URLS=(
    "http://localhost:3000"
    "http://localhost:3001"
)

BACKEND_URLS=(
    "http://localhost:8001"
    "http://localhost:8002"
    "http://localhost:8003"
    "http://localhost:8004"
)

DATABASE_URLS=(
    "postgresql://postgres:postgres@localhost:5432/nexus_dev"
    "redis://localhost:6379"
)

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Initialize integration testing environment
init_integration_testing() {
    log "Initializing integration testing environment..."
    
    # Create integration testing directory structure
    mkdir -p "$INTEGRATION_TEST_DIR"/{tests,results,reports,fixtures,mocks}
    mkdir -p "$TEST_RESULTS_DIR"/{api,frontend,database,auth,workflow}
    
    # Install testing dependencies
    install_testing_dependencies
    
    # Create test fixtures
    create_test_fixtures
    
    success "Integration testing environment initialized ✓"
}

# Install testing dependencies
install_testing_dependencies() {
    log "Installing testing dependencies..."
    
    # Install Node.js testing tools
    npm install -g newman jest supertest cypress
    
    # Install Python testing tools
    pip3 install --user pytest requests pytest-html pytest-json-report
    
    # Install curl and jq if not available
    if ! command -v jq &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update && sudo apt-get install -y jq curl
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install jq curl
            fi
        fi
    fi
    
    success "Testing dependencies installed ✓"
}

# Create test fixtures
create_test_fixtures() {
    log "Creating test fixtures..."
    
    # Create test user data
    cat > "$INTEGRATION_TEST_DIR/fixtures/test-users.json" << 'EOF'
{
  "users": [
    {
      "id": "test-admin-001",
      "username": "admin",
      "email": "admin@nexus.test",
      "password": "password",
      "role": "admin",
      "firstName": "Admin",
      "lastName": "User"
    },
    {
      "id": "test-dev-001",
      "username": "developer",
      "email": "developer@nexus.test",
      "password": "password",
      "role": "developer",
      "firstName": "Developer",
      "lastName": "User"
    },
    {
      "id": "test-mgr-001",
      "username": "manager",
      "email": "manager@nexus.test",
      "password": "password",
      "role": "manager",
      "firstName": "Project",
      "lastName": "Manager"
    },
    {
      "id": "test-exec-001",
      "username": "executive",
      "email": "executive@nexus.test",
      "password": "password",
      "role": "executive",
      "firstName": "Executive",
      "lastName": "User"
    }
  ]
}
EOF

    # Create test project data
    cat > "$INTEGRATION_TEST_DIR/fixtures/test-projects.json" << 'EOF'
{
  "projects": [
    {
      "id": "proj-001",
      "name": "Test Project Alpha",
      "description": "Integration test project for validation",
      "status": "active",
      "owner": "test-dev-001",
      "team": ["test-dev-001", "test-mgr-001"],
      "created": "2024-01-01T00:00:00Z"
    },
    {
      "id": "proj-002",
      "name": "Test Project Beta",
      "description": "Secondary test project for workflow testing",
      "status": "planning",
      "owner": "test-mgr-001",
      "team": ["test-dev-001", "test-mgr-001", "test-exec-001"],
      "created": "2024-01-02T00:00:00Z"
    }
  ]
}
EOF

    success "Test fixtures created ✓"
}

# Test service health and connectivity
test_service_health() {
    log "Testing service health and connectivity..."
    
    local health_results="$TEST_RESULTS_DIR/service-health-$TEST_DATE.json"
    local health_status=0
    
    echo '{"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'","tests":[' > "$health_results"
    
    # Test frontend services
    for url in "${FRONTEND_URLS[@]}"; do
        ((TOTAL_TESTS++))
        log "Testing frontend service: $url"
        
        if curl -f -s --max-time 10 "$url" > /dev/null; then
            success "Frontend service $url is healthy ✓"
            ((PASSED_TESTS++))
            echo '{"service":"'$url'","type":"frontend","status":"pass","message":"Service is healthy"},' >> "$health_results"
        else
            error "Frontend service $url is not responding"
            ((FAILED_TESTS++))
            health_status=1
            echo '{"service":"'$url'","type":"frontend","status":"fail","message":"Service not responding"},' >> "$health_results"
        fi
    done
    
    # Test backend services
    for url in "${BACKEND_URLS[@]}"; do
        ((TOTAL_TESTS++))
        log "Testing backend service: $url"
        
        # Test health endpoint
        local health_endpoint="$url/health"
        if curl -f -s --max-time 10 "$health_endpoint" > /dev/null; then
            success "Backend service $url is healthy ✓"
            ((PASSED_TESTS++))
            echo '{"service":"'$url'","type":"backend","status":"pass","message":"Health endpoint responding"},' >> "$health_results"
        else
            # Try root endpoint if health endpoint fails
            if curl -f -s --max-time 10 "$url" > /dev/null; then
                warning "Backend service $url responding but no health endpoint"
                ((PASSED_TESTS++))
                echo '{"service":"'$url'","type":"backend","status":"pass","message":"Service responding (no health endpoint)"},' >> "$health_results"
            else
                error "Backend service $url is not responding"
                ((FAILED_TESTS++))
                health_status=1
                echo '{"service":"'$url'","type":"backend","status":"fail","message":"Service not responding"},' >> "$health_results"
            fi
        fi
    done
    
    # Test database connectivity
    for db_url in "${DATABASE_URLS[@]}"; do
        ((TOTAL_TESTS++))
        log "Testing database connectivity: $db_url"
        
        if [[ "$db_url" == postgresql* ]]; then
            # Test PostgreSQL connection
            if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
                success "PostgreSQL database is accessible ✓"
                ((PASSED_TESTS++))
                echo '{"service":"'$db_url'","type":"database","status":"pass","message":"PostgreSQL connection successful"},' >> "$health_results"
            else
                error "PostgreSQL database is not accessible"
                ((FAILED_TESTS++))
                health_status=1
                echo '{"service":"'$db_url'","type":"database","status":"fail","message":"PostgreSQL connection failed"},' >> "$health_results"
            fi
        elif [[ "$db_url" == redis* ]]; then
            # Test Redis connection
            if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
                success "Redis database is accessible ✓"
                ((PASSED_TESTS++))
                echo '{"service":"'$db_url'","type":"database","status":"pass","message":"Redis connection successful"},' >> "$health_results"
            else
                error "Redis database is not accessible"
                ((FAILED_TESTS++))
                health_status=1
                echo '{"service":"'$db_url'","type":"database","status":"fail","message":"Redis connection failed"},' >> "$health_results"
            fi
        fi
    done
    
    # Close JSON array
    sed -i '$ s/,$//' "$health_results"
    echo ']}' >> "$health_results"
    
    if [[ $health_status -eq 0 ]]; then
        success "All service health checks passed ✓"
    else
        warning "Some service health checks failed"
    fi
    
    return $health_status
}

# Test API endpoints
test_api_endpoints() {
    log "Testing API endpoints..."
    
    local api_results="$TEST_RESULTS_DIR/api-tests-$TEST_DATE.json"
    local api_status=0
    
    echo '{"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'","api_tests":[' > "$api_results"
    
    # Test authentication endpoints
    for backend_url in "${BACKEND_URLS[@]}"; do
        if curl -f -s --max-time 5 "$backend_url" > /dev/null; then
            log "Testing authentication API on $backend_url"
            
            # Test login endpoint
            ((TOTAL_TESTS++))
            local login_response=$(curl -s -X POST "$backend_url/api/auth/login" \
                -H "Content-Type: application/json" \
                -d '{"username":"admin","password":"password"}' \
                --max-time 10)
            
            if echo "$login_response" | jq -e '.token' > /dev/null 2>&1; then
                success "Login API working ✓"
                ((PASSED_TESTS++))
                echo '{"endpoint":"'$backend_url'/api/auth/login","method":"POST","status":"pass","message":"Login successful"},' >> "$api_results"
                
                # Extract token for further tests
                local token=$(echo "$login_response" | jq -r '.token')
                
                # Test protected endpoint
                ((TOTAL_TESTS++))
                local profile_response=$(curl -s -H "Authorization: Bearer $token" \
                    "$backend_url/api/user/profile" --max-time 10)
                
                if echo "$profile_response" | jq -e '.username' > /dev/null 2>&1; then
                    success "Protected API working ✓"
                    ((PASSED_TESTS++))
                    echo '{"endpoint":"'$backend_url'/api/user/profile","method":"GET","status":"pass","message":"Protected endpoint accessible"},' >> "$api_results"
                else
                    warning "Protected API not working properly"
                    ((FAILED_TESTS++))
                    api_status=1
                    echo '{"endpoint":"'$backend_url'/api/user/profile","method":"GET","status":"fail","message":"Protected endpoint failed"},' >> "$api_results"
                fi
            else
                warning "Login API not working properly"
                ((FAILED_TESTS++))
                api_status=1
                echo '{"endpoint":"'$backend_url'/api/auth/login","method":"POST","status":"fail","message":"Login failed"},' >> "$api_results"
            fi
            
            # Test public endpoints
            local public_endpoints=("/api/health" "/api/version" "/api/status")
            
            for endpoint in "${public_endpoints[@]}"; do
                ((TOTAL_TESTS++))
                log "Testing public endpoint: $backend_url$endpoint"
                
                local response_code=$(curl -s -o /dev/null -w "%{http_code}" "$backend_url$endpoint" --max-time 10)
                
                if [[ "$response_code" == "200" ]]; then
                    success "Public endpoint $endpoint working ✓"
                    ((PASSED_TESTS++))
                    echo '{"endpoint":"'$backend_url$endpoint'","method":"GET","status":"pass","message":"Public endpoint accessible"},' >> "$api_results"
                else
                    warning "Public endpoint $endpoint returned $response_code"
                    ((FAILED_TESTS++))
                    api_status=1
                    echo '{"endpoint":"'$backend_url$endpoint'","method":"GET","status":"fail","message":"Returned HTTP '$response_code'"},' >> "$api_results"
                fi
            done
        fi
    done
    
    # Close JSON array
    sed -i '$ s/,$//' "$api_results"
    echo ']}' >> "$api_results"
    
    if [[ $api_status -eq 0 ]]; then
        success "All API endpoint tests passed ✓"
    else
        warning "Some API endpoint tests failed"
    fi
    
    return $api_status
}

# Test frontend functionality
test_frontend_functionality() {
    log "Testing frontend functionality..."
    
    local frontend_results="$TEST_RESULTS_DIR/frontend-tests-$TEST_DATE.json"
    local frontend_status=0
    
    echo '{"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'","frontend_tests":[' > "$frontend_results"
    
    for frontend_url in "${FRONTEND_URLS[@]}"; do
        if curl -f -s --max-time 5 "$frontend_url" > /dev/null; then
            log "Testing frontend functionality on $frontend_url"
            
            # Test main page load
            ((TOTAL_TESTS++))
            local page_content=$(curl -s "$frontend_url" --max-time 10)
            
            if echo "$page_content" | grep -q "Nexus Architect"; then
                success "Frontend main page loads correctly ✓"
                ((PASSED_TESTS++))
                echo '{"url":"'$frontend_url'","test":"main_page","status":"pass","message":"Main page loads with correct title"},' >> "$frontend_results"
            else
                warning "Frontend main page may not be loading correctly"
                ((FAILED_TESTS++))
                frontend_status=1
                echo '{"url":"'$frontend_url'","test":"main_page","status":"fail","message":"Main page title not found"},' >> "$frontend_results"
            fi
            
            # Test static assets
            ((TOTAL_TESTS++))
            local assets_check=$(curl -s -o /dev/null -w "%{http_code}" "$frontend_url/static/css/main.css" --max-time 10 2>/dev/null || echo "404")
            
            if [[ "$assets_check" == "200" ]]; then
                success "Frontend static assets accessible ✓"
                ((PASSED_TESTS++))
                echo '{"url":"'$frontend_url'","test":"static_assets","status":"pass","message":"CSS assets accessible"},' >> "$frontend_results"
            else
                # Try alternative asset paths
                local alt_assets_check=$(curl -s -o /dev/null -w "%{http_code}" "$frontend_url/assets/index.css" --max-time 10 2>/dev/null || echo "404")
                if [[ "$alt_assets_check" == "200" ]]; then
                    success "Frontend static assets accessible (alternative path) ✓"
                    ((PASSED_TESTS++))
                    echo '{"url":"'$frontend_url'","test":"static_assets","status":"pass","message":"Assets accessible via alternative path"},' >> "$frontend_results"
                else
                    warning "Frontend static assets may not be accessible"
                    ((FAILED_TESTS++))
                    frontend_status=1
                    echo '{"url":"'$frontend_url'","test":"static_assets","status":"fail","message":"Static assets not accessible"},' >> "$frontend_results"
                fi
            fi
            
            # Test API connectivity from frontend
            ((TOTAL_TESTS++))
            local api_connectivity=$(curl -s "$frontend_url/api/health" --max-time 10 2>/dev/null || echo "")
            
            if [[ -n "$api_connectivity" ]]; then
                success "Frontend to API connectivity working ✓"
                ((PASSED_TESTS++))
                echo '{"url":"'$frontend_url'","test":"api_connectivity","status":"pass","message":"Frontend can reach API endpoints"},' >> "$frontend_results"
            else
                warning "Frontend to API connectivity may not be working"
                ((FAILED_TESTS++))
                frontend_status=1
                echo '{"url":"'$frontend_url'","test":"api_connectivity","status":"fail","message":"Frontend cannot reach API endpoints"},' >> "$frontend_results"
            fi
        fi
    done
    
    # Close JSON array
    sed -i '$ s/,$//' "$frontend_results"
    echo ']}' >> "$frontend_results"
    
    if [[ $frontend_status -eq 0 ]]; then
        success "All frontend functionality tests passed ✓"
    else
        warning "Some frontend functionality tests failed"
    fi
    
    return $frontend_status
}

# Test database operations
test_database_operations() {
    log "Testing database operations..."
    
    local db_results="$TEST_RESULTS_DIR/database-tests-$TEST_DATE.json"
    local db_status=0
    
    echo '{"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'","database_tests":[' > "$db_results"
    
    # Test PostgreSQL operations
    if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
        log "Testing PostgreSQL database operations..."
        
        # Test connection and basic query
        ((TOTAL_TESTS++))
        local pg_version=$(psql -h localhost -U postgres -d postgres -t -c "SELECT version();" 2>/dev/null | head -1 || echo "")
        
        if [[ -n "$pg_version" ]]; then
            success "PostgreSQL connection and query working ✓"
            ((PASSED_TESTS++))
            echo '{"database":"postgresql","test":"connection","status":"pass","message":"Connection and basic query successful"},' >> "$db_results"
            
            # Test table creation and operations
            ((TOTAL_TESTS++))
            local table_test=$(psql -h localhost -U postgres -d postgres -c "
                CREATE TABLE IF NOT EXISTS integration_test (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT NOW()
                );
                INSERT INTO integration_test (name) VALUES ('test_record');
                SELECT COUNT(*) FROM integration_test WHERE name = 'test_record';
                DROP TABLE integration_test;
            " 2>/dev/null | grep -o '[0-9]\+' | tail -1 || echo "0")
            
            if [[ "$table_test" -gt 0 ]]; then
                success "PostgreSQL CRUD operations working ✓"
                ((PASSED_TESTS++))
                echo '{"database":"postgresql","test":"crud_operations","status":"pass","message":"Create, insert, select, drop operations successful"},' >> "$db_results"
            else
                warning "PostgreSQL CRUD operations may not be working"
                ((FAILED_TESTS++))
                db_status=1
                echo '{"database":"postgresql","test":"crud_operations","status":"fail","message":"CRUD operations failed"},' >> "$db_results"
            fi
        else
            warning "PostgreSQL connection failed"
            ((FAILED_TESTS++))
            db_status=1
            echo '{"database":"postgresql","test":"connection","status":"fail","message":"Connection failed"},' >> "$db_results"
        fi
    else
        warning "PostgreSQL is not accessible, skipping tests"
        ((SKIPPED_TESTS++))
        echo '{"database":"postgresql","test":"connection","status":"skip","message":"PostgreSQL not accessible"},' >> "$db_results"
    fi
    
    # Test Redis operations
    if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
        log "Testing Redis database operations..."
        
        # Test basic Redis operations
        ((TOTAL_TESTS++))
        local redis_test=$(redis-cli -h localhost -p 6379 eval "
            redis.call('SET', 'integration_test_key', 'test_value')
            local value = redis.call('GET', 'integration_test_key')
            redis.call('DEL', 'integration_test_key')
            return value
        " 0 2>/dev/null || echo "")
        
        if [[ "$redis_test" == "test_value" ]]; then
            success "Redis operations working ✓"
            ((PASSED_TESTS++))
            echo '{"database":"redis","test":"operations","status":"pass","message":"Set, get, delete operations successful"},' >> "$db_results"
        else
            warning "Redis operations may not be working"
            ((FAILED_TESTS++))
            db_status=1
            echo '{"database":"redis","test":"operations","status":"fail","message":"Redis operations failed"},' >> "$db_results"
        fi
    else
        warning "Redis is not accessible, skipping tests"
        ((SKIPPED_TESTS++))
        echo '{"database":"redis","test":"operations","status":"skip","message":"Redis not accessible"},' >> "$db_results"
    fi
    
    # Close JSON array
    sed -i '$ s/,$//' "$db_results"
    echo ']}' >> "$db_results"
    
    if [[ $db_status -eq 0 ]]; then
        success "All database operation tests passed ✓"
    else
        warning "Some database operation tests failed"
    fi
    
    return $db_status
}

# Test authentication workflow
test_authentication_workflow() {
    log "Testing authentication workflow..."
    
    local auth_results="$TEST_RESULTS_DIR/auth-tests-$TEST_DATE.json"
    local auth_status=0
    
    echo '{"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'","auth_tests":[' > "$auth_results"
    
    # Test Keycloak authentication if available
    if curl -f -s "http://localhost:8080/health/ready" > /dev/null 2>&1; then
        log "Testing Keycloak authentication workflow..."
        
        # Test realm accessibility
        ((TOTAL_TESTS++))
        local realm_response=$(curl -s "http://localhost:8080/auth/realms/nexus" --max-time 10)
        
        if echo "$realm_response" | jq -e '.realm' > /dev/null 2>&1; then
            success "Keycloak realm accessible ✓"
            ((PASSED_TESTS++))
            echo '{"service":"keycloak","test":"realm_access","status":"pass","message":"Realm configuration accessible"},' >> "$auth_results"
            
            # Test token endpoint
            ((TOTAL_TESTS++))
            local token_response=$(curl -s -X POST "http://localhost:8080/auth/realms/nexus/protocol/openid-connect/token" \
                -H "Content-Type: application/x-www-form-urlencoded" \
                -d "grant_type=password&client_id=nexus-frontend&client_secret=nexus-frontend-secret&username=admin&password=password" \
                --max-time 10)
            
            if echo "$token_response" | jq -e '.access_token' > /dev/null 2>&1; then
                success "Keycloak token endpoint working ✓"
                ((PASSED_TESTS++))
                echo '{"service":"keycloak","test":"token_endpoint","status":"pass","message":"Token generation successful"},' >> "$auth_results"
            else
                warning "Keycloak token endpoint may not be working"
                ((FAILED_TESTS++))
                auth_status=1
                echo '{"service":"keycloak","test":"token_endpoint","status":"fail","message":"Token generation failed"},' >> "$auth_results"
            fi
        else
            warning "Keycloak realm not accessible"
            ((FAILED_TESTS++))
            auth_status=1
            echo '{"service":"keycloak","test":"realm_access","status":"fail","message":"Realm not accessible"},' >> "$auth_results"
        fi
    else
        warning "Keycloak is not accessible, skipping authentication tests"
        ((SKIPPED_TESTS++))
        echo '{"service":"keycloak","test":"availability","status":"skip","message":"Keycloak not accessible"},' >> "$auth_results"
    fi
    
    # Test LDAP authentication if available
    if ldapsearch -x -H "ldap://localhost:389" -b "dc=nexus,dc=dev" -D "cn=admin,dc=nexus,dc=dev" -w "nexus_ldap_admin" "(objectClass=*)" dn > /dev/null 2>&1; then
        log "Testing LDAP authentication workflow..."
        
        # Test user authentication
        ((TOTAL_TESTS++))
        if ldapsearch -x -H "ldap://localhost:389" -b "ou=people,dc=nexus,dc=dev" -D "uid=admin,ou=people,dc=nexus,dc=dev" -w "password" "(uid=admin)" dn > /dev/null 2>&1; then
            success "LDAP user authentication working ✓"
            ((PASSED_TESTS++))
            echo '{"service":"ldap","test":"user_auth","status":"pass","message":"User authentication successful"},' >> "$auth_results"
        else
            warning "LDAP user authentication may not be working"
            ((FAILED_TESTS++))
            auth_status=1
            echo '{"service":"ldap","test":"user_auth","status":"fail","message":"User authentication failed"},' >> "$auth_results"
        fi
    else
        warning "LDAP is not accessible, skipping LDAP tests"
        ((SKIPPED_TESTS++))
        echo '{"service":"ldap","test":"availability","status":"skip","message":"LDAP not accessible"},' >> "$auth_results"
    fi
    
    # Close JSON array
    sed -i '$ s/,$//' "$auth_results"
    echo ']}' >> "$auth_results"
    
    if [[ $auth_status -eq 0 ]]; then
        success "All authentication workflow tests passed ✓"
    else
        warning "Some authentication workflow tests failed"
    fi
    
    return $auth_status
}

# Test end-to-end workflows
test_end_to_end_workflows() {
    log "Testing end-to-end workflows..."
    
    local e2e_results="$TEST_RESULTS_DIR/e2e-tests-$TEST_DATE.json"
    local e2e_status=0
    
    echo '{"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'","e2e_tests":[' > "$e2e_results"
    
    # Test complete user workflow
    for frontend_url in "${FRONTEND_URLS[@]}"; do
        if curl -f -s --max-time 5 "$frontend_url" > /dev/null; then
            log "Testing end-to-end user workflow on $frontend_url"
            
            # Simulate user journey: Login -> Dashboard -> Projects -> Logout
            ((TOTAL_TESTS++))
            
            # Step 1: Access login page
            local login_page=$(curl -s "$frontend_url/login" --max-time 10 2>/dev/null || curl -s "$frontend_url" --max-time 10)
            
            if [[ -n "$login_page" ]]; then
                # Step 2: Attempt to access dashboard (should redirect to login if not authenticated)
                local dashboard_response=$(curl -s -w "%{http_code}" "$frontend_url/dashboard" --max-time 10)
                
                # Step 3: Check if protected routes are properly protected
                if echo "$dashboard_response" | grep -q "401\|403\|login"; then
                    success "Protected routes properly secured ✓"
                    ((PASSED_TESTS++))
                    echo '{"workflow":"user_journey","step":"route_protection","status":"pass","message":"Protected routes require authentication"},' >> "$e2e_results"
                else
                    warning "Protected routes may not be properly secured"
                    ((FAILED_TESTS++))
                    e2e_status=1
                    echo '{"workflow":"user_journey","step":"route_protection","status":"fail","message":"Protected routes accessible without authentication"},' >> "$e2e_results"
                fi
            else
                warning "Cannot access frontend for workflow testing"
                ((FAILED_TESTS++))
                e2e_status=1
                echo '{"workflow":"user_journey","step":"frontend_access","status":"fail","message":"Frontend not accessible"},' >> "$e2e_results"
            fi
        fi
    done
    
    # Test API workflow
    for backend_url in "${BACKEND_URLS[@]}"; do
        if curl -f -s --max-time 5 "$backend_url" > /dev/null; then
            log "Testing API workflow on $backend_url"
            
            ((TOTAL_TESTS++))
            
            # Step 1: Login and get token
            local login_response=$(curl -s -X POST "$backend_url/api/auth/login" \
                -H "Content-Type: application/json" \
                -d '{"username":"admin","password":"password"}' \
                --max-time 10)
            
            if echo "$login_response" | jq -e '.token' > /dev/null 2>&1; then
                local token=$(echo "$login_response" | jq -r '.token')
                
                # Step 2: Use token to access protected resource
                local protected_response=$(curl -s -H "Authorization: Bearer $token" \
                    "$backend_url/api/user/profile" --max-time 10)
                
                if echo "$protected_response" | jq -e '.username' > /dev/null 2>&1; then
                    success "API authentication workflow working ✓"
                    ((PASSED_TESTS++))
                    echo '{"workflow":"api_auth","step":"complete_flow","status":"pass","message":"Login and protected access successful"},' >> "$e2e_results"
                else
                    warning "API protected access may not be working"
                    ((FAILED_TESTS++))
                    e2e_status=1
                    echo '{"workflow":"api_auth","step":"protected_access","status":"fail","message":"Protected access failed"},' >> "$e2e_results"
                fi
            else
                warning "API login may not be working"
                ((FAILED_TESTS++))
                e2e_status=1
                echo '{"workflow":"api_auth","step":"login","status":"fail","message":"API login failed"},' >> "$e2e_results"
            fi
        fi
    done
    
    # Close JSON array
    sed -i '$ s/,$//' "$e2e_results"
    echo ']}' >> "$e2e_results"
    
    if [[ $e2e_status -eq 0 ]]; then
        success "All end-to-end workflow tests passed ✓"
    else
        warning "Some end-to-end workflow tests failed"
    fi
    
    return $e2e_status
}

# Generate integration test report
generate_integration_report() {
    log "Generating integration test report..."
    
    local pass_rate=0
    if [[ $TOTAL_TESTS -gt 0 ]]; then
        pass_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    fi
    
    cat > "$TEST_REPORT" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Nexus Architect Integration Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .pass { color: #27ae60; font-weight: bold; }
        .fail { color: #e74c3c; font-weight: bold; }
        .skip { color: #f39c12; font-weight: bold; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; text-align: center; }
        .metric h3 { margin: 0; font-size: 24px; }
        .metric p { margin: 5px 0 0 0; color: #7f8c8d; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .progress-bar { width: 100%; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #27ae60; transition: width 0.3s ease; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Nexus Architect Integration Test Report</h1>
        <p>Generated: $(date)</p>
        <p>Test ID: $TEST_DATE</p>
    </div>

    <div class="section">
        <h2>Test Summary</h2>
        <div class="metric">
            <h3>$TOTAL_TESTS</h3>
            <p>Total Tests</p>
        </div>
        <div class="metric">
            <h3 class="pass">$PASSED_TESTS</h3>
            <p>Passed</p>
        </div>
        <div class="metric">
            <h3 class="fail">$FAILED_TESTS</h3>
            <p>Failed</p>
        </div>
        <div class="metric">
            <h3 class="skip">$SKIPPED_TESTS</h3>
            <p>Skipped</p>
        </div>
        <div class="metric">
            <h3>$pass_rate%</h3>
            <p>Pass Rate</p>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: ${pass_rate}%;"></div>
        </div>
    </div>

    <div class="section">
        <h2>Test Categories</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Description</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Service Health</td>
                <td>Connectivity and health checks for all services</td>
                <td class="pass">COMPLETED</td>
            </tr>
            <tr>
                <td>API Endpoints</td>
                <td>Authentication and public API endpoint testing</td>
                <td class="pass">COMPLETED</td>
            </tr>
            <tr>
                <td>Frontend Functionality</td>
                <td>Frontend loading, assets, and API connectivity</td>
                <td class="pass">COMPLETED</td>
            </tr>
            <tr>
                <td>Database Operations</td>
                <td>PostgreSQL and Redis CRUD operations</td>
                <td class="pass">COMPLETED</td>
            </tr>
            <tr>
                <td>Authentication Workflow</td>
                <td>Keycloak and LDAP authentication testing</td>
                <td class="pass">COMPLETED</td>
            </tr>
            <tr>
                <td>End-to-End Workflows</td>
                <td>Complete user journey and API workflow testing</td>
                <td class="pass">COMPLETED</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Detailed Results</h2>
        <p>Detailed test results are available in JSON format:</p>
        <ul>
            <li><strong>Service Health:</strong> $TEST_RESULTS_DIR/service-health-$TEST_DATE.json</li>
            <li><strong>API Tests:</strong> $TEST_RESULTS_DIR/api-tests-$TEST_DATE.json</li>
            <li><strong>Frontend Tests:</strong> $TEST_RESULTS_DIR/frontend-tests-$TEST_DATE.json</li>
            <li><strong>Database Tests:</strong> $TEST_RESULTS_DIR/database-tests-$TEST_DATE.json</li>
            <li><strong>Authentication Tests:</strong> $TEST_RESULTS_DIR/auth-tests-$TEST_DATE.json</li>
            <li><strong>End-to-End Tests:</strong> $TEST_RESULTS_DIR/e2e-tests-$TEST_DATE.json</li>
        </ul>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ol>
            <li>Review any failed tests and address underlying issues</li>
            <li>Implement continuous integration testing</li>
            <li>Set up automated test execution on code changes</li>
            <li>Expand test coverage for edge cases</li>
            <li>Monitor test execution times and optimize slow tests</li>
        </ol>
    </div>

    <div class="section">
        <h2>Next Steps</h2>
        <ol>
            <li>Address any failed integration tests</li>
            <li>Implement additional test scenarios</li>
            <li>Set up test automation in CI/CD pipeline</li>
            <li>Create test data management strategy</li>
            <li>Schedule regular integration test execution</li>
        </ol>
    </div>
</body>
</html>
EOF

    success "Integration test report generated: $TEST_REPORT ✓"
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #12: Integration testing suite"
    
    init_integration_testing
    
    # Run all integration tests
    test_service_health
    test_api_endpoints
    test_frontend_functionality
    test_database_operations
    test_authentication_workflow
    test_end_to_end_workflows
    
    # Generate report
    generate_integration_report
    
    success "🎉 Integration testing suite completed successfully!"
    success "📊 Total Tests: $TOTAL_TESTS"
    success "✅ Passed: $PASSED_TESTS"
    success "❌ Failed: $FAILED_TESTS"
    success "⏭️ Skipped: $SKIPPED_TESTS"
    success "📄 Report: $TEST_REPORT"
    
    local pass_rate=0
    if [[ $TOTAL_TESTS -gt 0 ]]; then
        pass_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    fi
    
    log "📋 Integration Test Results:"
    log "   🔍 Service Health Checks"
    log "   🔌 API Endpoint Testing"
    log "   🌐 Frontend Functionality"
    log "   🗄️ Database Operations"
    log "   🔐 Authentication Workflows"
    log "   🔄 End-to-End User Journeys"
    
    info "📊 Pass Rate: $pass_rate%"
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        success "🎉 All integration tests passed!"
    else
        warning "⚠️ $FAILED_TESTS tests failed. Review detailed results."
    fi
    
    info "💡 Next steps:"
    info "   1. Review integration test results"
    info "   2. Address any failed tests"
    info "   3. Implement continuous integration testing"
    info "   4. Expand test coverage"
    info "   5. Set up automated test execution"
    
    # Return appropriate exit code
    if [[ $FAILED_TESTS -gt 0 ]]; then
        return 1
    else
        return 0
    fi
}

# Handle script arguments
case "${1:-all}" in
    "health")
        init_integration_testing
        test_service_health
        ;;
    "api")
        init_integration_testing
        test_api_endpoints
        ;;
    "frontend")
        init_integration_testing
        test_frontend_functionality
        ;;
    "database")
        init_integration_testing
        test_database_operations
        ;;
    "auth")
        init_integration_testing
        test_authentication_workflow
        ;;
    "e2e")
        init_integration_testing
        test_end_to_end_workflows
        ;;
    "report")
        generate_integration_report
        ;;
    "all"|*)
        main
        ;;
esac

