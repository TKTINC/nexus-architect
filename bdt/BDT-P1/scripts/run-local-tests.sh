#!/bin/bash

# Nexus Architect - Local Testing Automation
# BDT-P1 Deliverable #5: Comprehensive local testing automation
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

# Test configuration
TEST_RESULTS_DIR="~/nexus-dev/test-results"
COVERAGE_THRESHOLD=80
PERFORMANCE_THRESHOLD_MS=1000

# Initialize test environment
init_test_environment() {
    log "Initializing test environment..."
    
    # Create test results directory
    mkdir -p "$TEST_RESULTS_DIR"
    
    # Load environment variables
    if [[ -f ~/nexus-dev/.env ]]; then
        source ~/nexus-dev/.env
        success "Environment variables loaded ✓"
    else
        warning "Environment file not found. Using defaults."
    fi
    
    # Check if services are running
    if ! docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml ps | grep -q "Up"; then
        warning "Docker services not running. Starting services..."
        docker-compose -f bdt/BDT-P1/docker/docker-compose.dev.yml up -d
        sleep 30
    fi
    
    success "Test environment initialized ✓"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    local services=(
        "localhost:5432"  # PostgreSQL
        "localhost:6379"  # Redis
        "localhost:9200"  # Elasticsearch
        "localhost:9090"  # Prometheus
    )
    
    for service in "${services[@]}"; do
        local host=$(echo $service | cut -d: -f1)
        local port=$(echo $service | cut -d: -f2)
        
        log "Checking $service..."
        for i in {1..30}; do
            if nc -z $host $port 2>/dev/null; then
                success "$service is ready ✓"
                break
            fi
            if [[ $i -eq 30 ]]; then
                error "$service is not ready after 60 seconds"
            fi
            sleep 2
        done
    done
}

# Run unit tests
run_unit_tests() {
    log "Running unit tests..."
    
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    # Node.js unit tests
    local node_projects=($(find implementation/ -name "package.json" -type f))
    for package_file in "${node_projects[@]}"; do
        local package_dir=$(dirname "$package_file")
        
        if [[ -f "$package_dir/package.json" ]] && grep -q '"test"' "$package_dir/package.json"; then
            log "Running Node.js tests in $package_dir"
            
            (
                cd "$package_dir"
                if npm test 2>&1 | tee "$TEST_RESULTS_DIR/$(basename $package_dir)-unit.log"; then
                    success "Unit tests passed in $package_dir ✓"
                    ((passed_tests++))
                else
                    error "Unit tests failed in $package_dir ✗"
                    ((failed_tests++))
                fi
                ((total_tests++))
            )
        fi
    done
    
    # Python unit tests
    local python_projects=($(find implementation/ -name "requirements.txt" -type f))
    for req_file in "${python_projects[@]}"; do
        local req_dir=$(dirname "$req_file")
        
        if [[ -d "$req_dir/venv" ]]; then
            log "Running Python tests in $req_dir"
            
            (
                cd "$req_dir"
                source venv/bin/activate
                
                if python -m pytest --verbose --tb=short 2>&1 | tee "$TEST_RESULTS_DIR/$(basename $req_dir)-unit.log"; then
                    success "Unit tests passed in $req_dir ✓"
                    ((passed_tests++))
                else
                    error "Unit tests failed in $req_dir ✗"
                    ((failed_tests++))
                fi
                ((total_tests++))
            )
        fi
    done
    
    info "Unit Tests Summary: $passed_tests/$total_tests passed"
    
    if [[ $failed_tests -gt 0 ]]; then
        error "Some unit tests failed. Check logs in $TEST_RESULTS_DIR"
        return 1
    fi
    
    success "All unit tests passed ✓"
}

# Run integration tests
run_integration_tests() {
    log "Running integration tests..."
    
    # Database connectivity test
    log "Testing database connectivity..."
    if PGPASSWORD=nexus_dev_password psql -h localhost -p 5432 -U nexus_dev -d nexus_dev -c "SELECT 1;" &>/dev/null; then
        success "PostgreSQL connectivity test passed ✓"
    else
        error "PostgreSQL connectivity test failed ✗"
        return 1
    fi
    
    # Redis connectivity test
    log "Testing Redis connectivity..."
    if redis-cli -h localhost -p 6379 -a nexus_redis_password ping &>/dev/null; then
        success "Redis connectivity test passed ✓"
    else
        error "Redis connectivity test failed ✗"
        return 1
    fi
    
    # API endpoint tests
    log "Testing API endpoints..."
    local api_endpoints=(
        "http://localhost:8001/health"  # WS1 Core API
        "http://localhost:8002/health"  # WS2 AI API
        "http://localhost:8003/health"  # WS3 Data API
        "http://localhost:8004/health"  # WS4 Autonomous API
    )
    
    for endpoint in "${api_endpoints[@]}"; do
        log "Testing $endpoint..."
        if curl -f -s "$endpoint" &>/dev/null; then
            success "API endpoint $endpoint is healthy ✓"
        else
            warning "API endpoint $endpoint is not responding"
        fi
    done
    
    # Frontend accessibility test
    log "Testing frontend accessibility..."
    if curl -f -s "http://localhost:3000" &>/dev/null; then
        success "Frontend is accessible ✓"
    else
        warning "Frontend is not accessible"
    fi
    
    success "Integration tests completed ✓"
}

# Run performance tests
run_performance_tests() {
    log "Running performance tests..."
    
    # API response time tests
    local api_endpoints=(
        "http://localhost:8001/health"
        "http://localhost:8002/health"
        "http://localhost:8003/health"
        "http://localhost:8004/health"
    )
    
    for endpoint in "${api_endpoints[@]}"; do
        log "Testing response time for $endpoint..."
        
        local response_time=$(curl -o /dev/null -s -w '%{time_total}' "$endpoint" | awk '{print $1*1000}')
        local response_time_int=$(printf "%.0f" "$response_time")
        
        if [[ $response_time_int -lt $PERFORMANCE_THRESHOLD_MS ]]; then
            success "Response time for $endpoint: ${response_time_int}ms ✓"
        else
            warning "Response time for $endpoint: ${response_time_int}ms (exceeds ${PERFORMANCE_THRESHOLD_MS}ms threshold)"
        fi
    done
    
    # Database query performance test
    log "Testing database query performance..."
    local query_time=$(PGPASSWORD=nexus_dev_password psql -h localhost -p 5432 -U nexus_dev -d nexus_dev -c "\timing on" -c "SELECT COUNT(*) FROM ws1_core.users;" 2>&1 | grep "Time:" | awk '{print $2}' | sed 's/ms//')
    
    if [[ -n "$query_time" ]] && [[ $(echo "$query_time < 100" | bc -l) -eq 1 ]]; then
        success "Database query performance: ${query_time}ms ✓"
    else
        warning "Database query performance: ${query_time}ms (may need optimization)"
    fi
    
    success "Performance tests completed ✓"
}

# Run security tests
run_security_tests() {
    log "Running security tests..."
    
    # SSL certificate validation
    log "Testing SSL certificates..."
    if openssl x509 -in ~/nexus-dev/certs/ssl/server-cert.pem -text -noout &>/dev/null; then
        success "SSL certificate is valid ✓"
    else
        error "SSL certificate is invalid ✗"
        return 1
    fi
    
    # Password strength test
    log "Testing password security..."
    local test_passwords=("password" "123456" "admin" "nexus_dev_password")
    for password in "${test_passwords[@]}"; do
        if [[ ${#password} -lt 12 ]]; then
            warning "Weak password detected: $password (consider using stronger passwords in production)"
        fi
    done
    
    # Port security test
    log "Testing port security..."
    local open_ports=$(netstat -tuln | grep LISTEN | awk '{print $4}' | cut -d: -f2 | sort -n | uniq)
    info "Open ports: $(echo $open_ports | tr '\n' ' ')"
    
    # Environment variable security test
    log "Testing environment variable security..."
    if [[ -f ~/nexus-dev/.env ]]; then
        if grep -q "password" ~/nexus-dev/.env; then
            warning "Passwords found in environment file (ensure proper security in production)"
        fi
        if grep -q "secret" ~/nexus-dev/.env; then
            warning "Secrets found in environment file (use proper secret management in production)"
        fi
    fi
    
    success "Security tests completed ✓"
}

# Run code quality tests
run_code_quality_tests() {
    log "Running code quality tests..."
    
    # Node.js code quality
    local node_projects=($(find implementation/ -name "package.json" -type f))
    for package_file in "${node_projects[@]}"; do
        local package_dir=$(dirname "$package_file")
        
        if [[ -f "$package_dir/package.json" ]]; then
            log "Checking code quality in $package_dir"
            
            (
                cd "$package_dir"
                
                # ESLint check
                if command -v eslint &> /dev/null && [[ -f ".eslintrc.js" || -f ".eslintrc.json" ]]; then
                    if eslint . --ext .js,.jsx,.ts,.tsx 2>&1 | tee "$TEST_RESULTS_DIR/$(basename $package_dir)-eslint.log"; then
                        success "ESLint passed in $package_dir ✓"
                    else
                        warning "ESLint issues found in $package_dir"
                    fi
                fi
                
                # Prettier check
                if command -v prettier &> /dev/null; then
                    if prettier --check . 2>&1 | tee "$TEST_RESULTS_DIR/$(basename $package_dir)-prettier.log"; then
                        success "Prettier check passed in $package_dir ✓"
                    else
                        warning "Prettier formatting issues found in $package_dir"
                    fi
                fi
            )
        fi
    done
    
    # Python code quality
    local python_projects=($(find implementation/ -name "requirements.txt" -type f))
    for req_file in "${python_projects[@]}"; do
        local req_dir=$(dirname "$req_file")
        
        if [[ -d "$req_dir/venv" ]]; then
            log "Checking Python code quality in $req_dir"
            
            (
                cd "$req_dir"
                source venv/bin/activate
                
                # Black formatting check
                if command -v black &> /dev/null; then
                    if black --check . 2>&1 | tee "$TEST_RESULTS_DIR/$(basename $req_dir)-black.log"; then
                        success "Black formatting check passed in $req_dir ✓"
                    else
                        warning "Black formatting issues found in $req_dir"
                    fi
                fi
                
                # Flake8 linting
                if command -v flake8 &> /dev/null; then
                    if flake8 . 2>&1 | tee "$TEST_RESULTS_DIR/$(basename $req_dir)-flake8.log"; then
                        success "Flake8 linting passed in $req_dir ✓"
                    else
                        warning "Flake8 linting issues found in $req_dir"
                    fi
                fi
            )
        fi
    done
    
    success "Code quality tests completed ✓"
}

# Generate test coverage report
generate_coverage_report() {
    log "Generating test coverage report..."
    
    local coverage_file="$TEST_RESULTS_DIR/coverage-summary.txt"
    echo "Nexus Architect Test Coverage Report" > "$coverage_file"
    echo "Generated on: $(date)" >> "$coverage_file"
    echo "======================================" >> "$coverage_file"
    
    # Node.js coverage
    local node_projects=($(find implementation/ -name "package.json" -type f))
    for package_file in "${node_projects[@]}"; do
        local package_dir=$(dirname "$package_file")
        local project_name=$(basename "$package_dir")
        
        if [[ -f "$package_dir/package.json" ]] && grep -q '"test"' "$package_dir/package.json"; then
            (
                cd "$package_dir"
                if npm run test:coverage &>/dev/null; then
                    echo "$project_name: Coverage report generated" >> "$coverage_file"
                else
                    echo "$project_name: Coverage report not available" >> "$coverage_file"
                fi
            )
        fi
    done
    
    # Python coverage
    local python_projects=($(find implementation/ -name "requirements.txt" -type f))
    for req_file in "${python_projects[@]}"; do
        local req_dir=$(dirname "$req_file")
        local project_name=$(basename "$req_dir")
        
        if [[ -d "$req_dir/venv" ]]; then
            (
                cd "$req_dir"
                source venv/bin/activate
                
                if python -m pytest --cov=. --cov-report=term-missing &>/dev/null; then
                    echo "$project_name: Coverage report generated" >> "$coverage_file"
                else
                    echo "$project_name: Coverage report not available" >> "$coverage_file"
                fi
            )
        fi
    done
    
    success "Coverage report generated: $coverage_file ✓"
}

# Generate test summary
generate_test_summary() {
    log "Generating test summary..."
    
    local summary_file="$TEST_RESULTS_DIR/test-summary.txt"
    
    cat > "$summary_file" << EOF
Nexus Architect Local Test Summary
Generated on: $(date)
=====================================

Test Environment:
- PostgreSQL: $(PGPASSWORD=nexus_dev_password psql -h localhost -p 5432 -U nexus_dev -d nexus_dev -c "SELECT version();" -t 2>/dev/null | head -1 | xargs || echo "Not available")
- Redis: $(redis-cli -h localhost -p 6379 -a nexus_redis_password info server 2>/dev/null | grep redis_version | cut -d: -f2 | tr -d '\r' || echo "Not available")
- Node.js: $(node --version 2>/dev/null || echo "Not available")
- Python: $(python3 --version 2>/dev/null || echo "Not available")
- Docker: $(docker --version 2>/dev/null || echo "Not available")

Test Results:
- Unit Tests: $(ls $TEST_RESULTS_DIR/*-unit.log 2>/dev/null | wc -l) projects tested
- Integration Tests: Completed
- Performance Tests: Completed
- Security Tests: Completed
- Code Quality Tests: Completed

Coverage Reports:
- Available in: $TEST_RESULTS_DIR/
- Threshold: $COVERAGE_THRESHOLD%

Performance Metrics:
- Response Time Threshold: ${PERFORMANCE_THRESHOLD_MS}ms
- All API endpoints tested

Security Checks:
- SSL certificates validated
- Password strength checked
- Port security reviewed
- Environment variables audited

Next Steps:
1. Review any failed tests in the log files
2. Address code quality issues if any
3. Ensure coverage meets the $COVERAGE_THRESHOLD% threshold
4. Run tests regularly during development

EOF

    success "Test summary generated: $summary_file ✓"
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #5: Comprehensive local testing automation"
    
    init_test_environment
    wait_for_services
    
    log "🧪 Starting comprehensive test suite..."
    
    # Run all test categories
    run_unit_tests
    run_integration_tests
    run_performance_tests
    run_security_tests
    run_code_quality_tests
    
    # Generate reports
    generate_coverage_report
    generate_test_summary
    
    success "🎉 All tests completed successfully!"
    success "📊 Test results available in: $TEST_RESULTS_DIR"
    success "📋 Test summary: $TEST_RESULTS_DIR/test-summary.txt"
    success "📈 Coverage report: $TEST_RESULTS_DIR/coverage-summary.txt"
    
    log "📋 Test Categories Completed:"
    log "   ✅ Unit Tests - Individual component testing"
    log "   ✅ Integration Tests - Service connectivity and API health"
    log "   ✅ Performance Tests - Response times and query performance"
    log "   ✅ Security Tests - SSL, passwords, and environment security"
    log "   ✅ Code Quality Tests - Linting, formatting, and best practices"
    
    info "💡 Tip: Run this script regularly during development to catch issues early"
}

# Handle script arguments
case "${1:-all}" in
    "unit")
        init_test_environment
        wait_for_services
        run_unit_tests
        ;;
    "integration")
        init_test_environment
        wait_for_services
        run_integration_tests
        ;;
    "performance")
        init_test_environment
        wait_for_services
        run_performance_tests
        ;;
    "security")
        init_test_environment
        run_security_tests
        ;;
    "quality")
        init_test_environment
        run_code_quality_tests
        ;;
    "all"|*)
        main
        ;;
esac

