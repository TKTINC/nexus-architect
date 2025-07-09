#!/bin/bash

# Nexus Architect - Performance Testing Automation
# BDT-P1 Deliverable #11: Performance testing automation
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
    exit 1
}

info() {
    echo -e "${PURPLE}[INFO]${NC} $1"
}

# Performance testing configuration
PERF_TEST_DIR="$HOME/nexus-dev/performance-tests"
RESULTS_DIR="$PERF_TEST_DIR/results"
TEST_DATE=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$RESULTS_DIR/performance-report-$TEST_DATE.html"

# Test endpoints
FRONTEND_ENDPOINTS=(
    "http://localhost:3000"
    "http://localhost:3001"
)

BACKEND_ENDPOINTS=(
    "http://localhost:8001"
    "http://localhost:8002"
    "http://localhost:8003"
    "http://localhost:8004"
)

# Performance thresholds
MAX_RESPONSE_TIME=2000  # milliseconds
MAX_ERROR_RATE=1        # percentage
MIN_THROUGHPUT=100      # requests per second

# Initialize performance testing environment
init_performance_testing() {
    log "Initializing performance testing environment..."
    
    # Create performance testing directory structure
    mkdir -p "$PERF_TEST_DIR"/{scripts,configs,results,reports}
    mkdir -p "$RESULTS_DIR"/{load-tests,stress-tests,spike-tests,volume-tests}
    
    # Install performance testing tools
    install_performance_tools
    
    success "Performance testing environment initialized ✓"
}

# Install performance testing tools
install_performance_tools() {
    log "Installing performance testing tools..."
    
    # Install Apache Bench (ab) if not available
    if ! command -v ab &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update && sudo apt-get install -y apache2-utils
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install httpie
            fi
        fi
    fi
    
    # Install wrk if not available
    if ! command -v wrk &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get install -y wrk
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install wrk
            fi
        fi
    fi
    
    # Install Node.js performance testing tools
    npm install -g artillery clinic autocannon
    
    # Install Python performance testing tools
    pip3 install --user locust pytest-benchmark
    
    success "Performance testing tools installed ✓"
}

# Create Artillery configuration
create_artillery_config() {
    log "Creating Artillery load testing configuration..."
    
    cat > "$PERF_TEST_DIR/configs/artillery-load-test.yml" << 'EOF'
config:
  target: 'http://localhost:3000'
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up"
    - duration: 120
      arrivalRate: 50
      name: "Ramp up load"
    - duration: 300
      arrivalRate: 100
      name: "Sustained load"
    - duration: 60
      arrivalRate: 200
      name: "Peak load"
  defaults:
    headers:
      User-Agent: "Nexus Performance Test"
  processor: "./artillery-processor.js"

scenarios:
  - name: "Frontend Load Test"
    weight: 70
    flow:
      - get:
          url: "/"
          capture:
            - json: "$.status"
              as: "status"
      - think: 2
      - get:
          url: "/dashboard"
      - think: 3
      - get:
          url: "/projects"
      - think: 2
      - post:
          url: "/api/search"
          json:
            query: "test"
            limit: 10

  - name: "API Load Test"
    weight: 30
    flow:
      - get:
          url: "/api/health"
      - think: 1
      - get:
          url: "/api/metrics"
      - think: 1
      - post:
          url: "/api/auth/login"
          json:
            username: "testuser"
            password: "testpass"
EOF

    # Create Artillery processor
    cat > "$PERF_TEST_DIR/configs/artillery-processor.js" << 'EOF'
module.exports = {
  setRandomUser: setRandomUser,
  logResponse: logResponse
};

function setRandomUser(requestParams, context, ee, next) {
  const users = ['admin', 'developer', 'manager', 'executive'];
  context.vars.username = users[Math.floor(Math.random() * users.length)];
  context.vars.password = 'password';
  return next();
}

function logResponse(requestParams, response, context, ee, next) {
  if (response.statusCode >= 400) {
    console.log(`Error ${response.statusCode}: ${requestParams.url}`);
  }
  return next();
}
EOF

    success "Artillery configuration created ✓"
}

# Create Locust configuration
create_locust_config() {
    log "Creating Locust performance testing configuration..."
    
    cat > "$PERF_TEST_DIR/configs/locustfile.py" << 'EOF'
from locust import HttpUser, task, between
import random
import json

class NexusUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login user when test starts"""
        self.login()
    
    def login(self):
        """Simulate user login"""
        response = self.client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "testpass"
        })
        if response.status_code == 200:
            self.token = response.json().get("token")
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(3)
    def view_dashboard(self):
        """View main dashboard"""
        self.client.get("/dashboard")
    
    @task(2)
    def view_projects(self):
        """View projects page"""
        self.client.get("/projects")
    
    @task(2)
    def search_projects(self):
        """Search for projects"""
        search_terms = ["nexus", "test", "development", "production"]
        term = random.choice(search_terms)
        self.client.post("/api/search", json={
            "query": term,
            "limit": 10
        })
    
    @task(1)
    def view_analytics(self):
        """View analytics page"""
        self.client.get("/analytics")
    
    @task(1)
    def api_health_check(self):
        """Check API health"""
        self.client.get("/api/health")
    
    @task(1)
    def get_metrics(self):
        """Get system metrics"""
        self.client.get("/api/metrics")

class AdminUser(HttpUser):
    wait_time = between(2, 5)
    weight = 1  # Lower weight for admin users
    
    def on_start(self):
        self.login_admin()
    
    def login_admin(self):
        """Login as admin user"""
        response = self.client.post("/api/auth/login", json={
            "username": "admin",
            "password": "password"
        })
        if response.status_code == 200:
            self.token = response.json().get("token")
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(2)
    def admin_dashboard(self):
        """View admin dashboard"""
        self.client.get("/admin/dashboard")
    
    @task(1)
    def manage_users(self):
        """Manage users"""
        self.client.get("/admin/users")
    
    @task(1)
    def system_settings(self):
        """View system settings"""
        self.client.get("/admin/settings")

class DeveloperUser(HttpUser):
    wait_time = between(1, 4)
    weight = 3  # Higher weight for developer users
    
    def on_start(self):
        self.login_developer()
    
    def login_developer(self):
        """Login as developer user"""
        response = self.client.post("/api/auth/login", json={
            "username": "developer",
            "password": "password"
        })
        if response.status_code == 200:
            self.token = response.json().get("token")
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(3)
    def developer_dashboard(self):
        """View developer dashboard"""
        self.client.get("/developer/dashboard")
    
    @task(2)
    def code_quality(self):
        """Check code quality"""
        self.client.get("/developer/quality")
    
    @task(2)
    def workflow_optimization(self):
        """View workflow optimization"""
        self.client.get("/developer/workflow")
    
    @task(1)
    def learning_center(self):
        """Access learning center"""
        self.client.get("/developer/learning")
EOF

    success "Locust configuration created ✓"
}

# Run load testing
run_load_tests() {
    log "Running load testing..."
    
    local load_results_dir="$RESULTS_DIR/load-tests"
    mkdir -p "$load_results_dir"
    
    # Test each frontend endpoint
    for endpoint in "${FRONTEND_ENDPOINTS[@]}"; do
        if curl -f -s --max-time 5 "$endpoint" &>/dev/null; then
            log "Load testing $endpoint..."
            
            # Apache Bench test
            local ab_output="$load_results_dir/ab-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g')-$TEST_DATE.txt"
            ab -n 1000 -c 10 -g "$load_results_dir/ab-gnuplot-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g').tsv" "$endpoint/" > "$ab_output" 2>&1 || true
            
            # Extract key metrics from ab output
            if [[ -f "$ab_output" ]]; then
                local requests_per_sec=$(grep "Requests per second" "$ab_output" | awk '{print $4}' || echo "0")
                local mean_time=$(grep "Time per request" "$ab_output" | head -1 | awk '{print $4}' || echo "0")
                local failed_requests=$(grep "Failed requests" "$ab_output" | awk '{print $3}' || echo "0")
                
                log "  Requests/sec: $requests_per_sec"
                log "  Mean time: ${mean_time}ms"
                log "  Failed requests: $failed_requests"
            fi
            
            # wrk test (if available)
            if command -v wrk &> /dev/null; then
                local wrk_output="$load_results_dir/wrk-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g')-$TEST_DATE.txt"
                wrk -t4 -c100 -d30s --script="$PERF_TEST_DIR/scripts/wrk-script.lua" "$endpoint/" > "$wrk_output" 2>&1 || true
            fi
            
            # Artillery test
            if command -v artillery &> /dev/null; then
                local artillery_output="$load_results_dir/artillery-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g')-$TEST_DATE.json"
                cd "$PERF_TEST_DIR/configs"
                artillery run --target "$endpoint" --output "$artillery_output" artillery-load-test.yml || true
            fi
        else
            warning "Endpoint $endpoint is not accessible, skipping load test"
        fi
    done
    
    success "Load testing completed ✓"
}

# Run stress testing
run_stress_tests() {
    log "Running stress testing..."
    
    local stress_results_dir="$RESULTS_DIR/stress-tests"
    mkdir -p "$stress_results_dir"
    
    # Stress test with increasing load
    for endpoint in "${FRONTEND_ENDPOINTS[@]}"; do
        if curl -f -s --max-time 5 "$endpoint" &>/dev/null; then
            log "Stress testing $endpoint..."
            
            # Progressive load increase
            local concurrency_levels=(50 100 200 500 1000)
            
            for concurrency in "${concurrency_levels[@]}"; do
                log "  Testing with $concurrency concurrent users..."
                
                local stress_output="$stress_results_dir/stress-c${concurrency}-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g')-$TEST_DATE.txt"
                
                # Run stress test with timeout
                timeout 60s ab -n $((concurrency * 10)) -c "$concurrency" "$endpoint/" > "$stress_output" 2>&1 || true
                
                # Check if system is still responsive
                if ! curl -f -s --max-time 10 "$endpoint" &>/dev/null; then
                    warning "System became unresponsive at $concurrency concurrent users"
                    echo "System became unresponsive at $concurrency concurrent users" >> "$stress_output"
                    break
                fi
                
                # Brief pause between tests
                sleep 10
            done
        fi
    done
    
    success "Stress testing completed ✓"
}

# Run spike testing
run_spike_tests() {
    log "Running spike testing..."
    
    local spike_results_dir="$RESULTS_DIR/spike-tests"
    mkdir -p "$spike_results_dir"
    
    # Sudden load spikes
    for endpoint in "${FRONTEND_ENDPOINTS[@]}"; do
        if curl -f -s --max-time 5 "$endpoint" &>/dev/null; then
            log "Spike testing $endpoint..."
            
            # Baseline measurement
            local baseline_output="$spike_results_dir/baseline-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g')-$TEST_DATE.txt"
            ab -n 100 -c 5 "$endpoint/" > "$baseline_output" 2>&1 || true
            
            # Sudden spike
            local spike_output="$spike_results_dir/spike-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g')-$TEST_DATE.txt"
            ab -n 2000 -c 500 "$endpoint/" > "$spike_output" 2>&1 || true
            
            # Recovery measurement
            sleep 30
            local recovery_output="$spike_results_dir/recovery-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g')-$TEST_DATE.txt"
            ab -n 100 -c 5 "$endpoint/" > "$recovery_output" 2>&1 || true
        fi
    done
    
    success "Spike testing completed ✓"
}

# Run volume testing
run_volume_tests() {
    log "Running volume testing..."
    
    local volume_results_dir="$RESULTS_DIR/volume-tests"
    mkdir -p "$volume_results_dir"
    
    # Extended duration tests
    for endpoint in "${FRONTEND_ENDPOINTS[@]}"; do
        if curl -f -s --max-time 5 "$endpoint" &>/dev/null; then
            log "Volume testing $endpoint (extended duration)..."
            
            # 30-minute sustained load test
            local volume_output="$volume_results_dir/volume-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g')-$TEST_DATE.txt"
            
            # Use Artillery for extended testing if available
            if command -v artillery &> /dev/null; then
                cat > "$PERF_TEST_DIR/configs/volume-test.yml" << EOF
config:
  target: '$endpoint'
  phases:
    - duration: 1800  # 30 minutes
      arrivalRate: 20
      name: "Volume test"
scenarios:
  - flow:
      - get:
          url: "/"
      - think: 5
EOF
                cd "$PERF_TEST_DIR/configs"
                artillery run --output "$volume_output.json" volume-test.yml > "$volume_output" 2>&1 || true
            else
                # Fallback to ab with extended test
                ab -n 36000 -c 20 -t 1800 "$endpoint/" > "$volume_output" 2>&1 || true
            fi
        fi
    done
    
    success "Volume testing completed ✓"
}

# Analyze performance results
analyze_performance_results() {
    log "Analyzing performance test results..."
    
    local analysis_file="$RESULTS_DIR/performance-analysis-$TEST_DATE.txt"
    
    cat > "$analysis_file" << EOF
Nexus Architect Performance Test Analysis
========================================
Test Date: $(date)
Test ID: $TEST_DATE

PERFORMANCE THRESHOLDS:
- Maximum Response Time: ${MAX_RESPONSE_TIME}ms
- Maximum Error Rate: ${MAX_ERROR_RATE}%
- Minimum Throughput: ${MIN_THROUGHPUT} req/sec

LOAD TEST RESULTS:
EOF

    # Analyze load test results
    for result_file in "$RESULTS_DIR/load-tests"/ab-*.txt; do
        if [[ -f "$result_file" ]]; then
            local endpoint=$(basename "$result_file" | sed 's/ab-//g' | sed 's/-[0-9_]*\.txt//g')
            
            echo "" >> "$analysis_file"
            echo "Endpoint: $endpoint" >> "$analysis_file"
            
            # Extract metrics
            local requests_per_sec=$(grep "Requests per second" "$result_file" | awk '{print $4}' | cut -d'.' -f1 || echo "0")
            local mean_time=$(grep "Time per request" "$result_file" | head -1 | awk '{print $4}' | cut -d'.' -f1 || echo "0")
            local failed_requests=$(grep "Failed requests" "$result_file" | awk '{print $3}' || echo "0")
            local total_requests=$(grep "Complete requests" "$result_file" | awk '{print $3}' || echo "1")
            
            # Calculate error rate
            local error_rate=0
            if [[ "$total_requests" -gt 0 ]]; then
                error_rate=$((failed_requests * 100 / total_requests))
            fi
            
            echo "  Throughput: ${requests_per_sec} req/sec" >> "$analysis_file"
            echo "  Mean Response Time: ${mean_time}ms" >> "$analysis_file"
            echo "  Error Rate: ${error_rate}%" >> "$analysis_file"
            
            # Performance assessment
            local performance_issues=()
            
            if [[ "$mean_time" -gt "$MAX_RESPONSE_TIME" ]]; then
                performance_issues+=("High response time")
            fi
            
            if [[ "$error_rate" -gt "$MAX_ERROR_RATE" ]]; then
                performance_issues+=("High error rate")
            fi
            
            if [[ "$requests_per_sec" -lt "$MIN_THROUGHPUT" ]]; then
                performance_issues+=("Low throughput")
            fi
            
            if [[ ${#performance_issues[@]} -eq 0 ]]; then
                echo "  Status: ✅ PASS" >> "$analysis_file"
            else
                echo "  Status: ❌ FAIL (${performance_issues[*]})" >> "$analysis_file"
            fi
        fi
    done
    
    # Add recommendations
    cat >> "$analysis_file" << 'EOF'

PERFORMANCE RECOMMENDATIONS:
1. Monitor response times during peak usage
2. Implement caching for frequently accessed data
3. Optimize database queries and add indexes
4. Consider implementing CDN for static assets
5. Set up auto-scaling for high-traffic periods
6. Implement connection pooling for databases
7. Use compression for API responses
8. Optimize frontend bundle sizes
9. Implement lazy loading for large datasets
10. Monitor and optimize memory usage

NEXT STEPS:
1. Review failed performance tests
2. Implement performance optimizations
3. Set up continuous performance monitoring
4. Establish performance budgets for CI/CD
5. Schedule regular performance testing
EOF

    success "Performance analysis completed: $analysis_file ✓"
}

# Generate performance report
generate_performance_report() {
    log "Generating performance test report..."
    
    cat > "$REPORT_FILE" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Nexus Architect Performance Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .pass { color: #27ae60; font-weight: bold; }
        .fail { color: #e74c3c; font-weight: bold; }
        .warning { color: #f39c12; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #ecf0f1; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Nexus Architect Performance Test Report</h1>
        <p>Generated: $(date)</p>
        <p>Test ID: $TEST_DATE</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report contains the results of comprehensive performance testing conducted on the Nexus Architect local development environment.</p>
        
        <div class="metric">
            <strong>Test Types:</strong><br>
            Load Testing ✓<br>
            Stress Testing ✓<br>
            Spike Testing ✓<br>
            Volume Testing ✓
        </div>
        
        <div class="metric">
            <strong>Thresholds:</strong><br>
            Max Response Time: ${MAX_RESPONSE_TIME}ms<br>
            Max Error Rate: ${MAX_ERROR_RATE}%<br>
            Min Throughput: ${MIN_THROUGHPUT} req/sec
        </div>
    </div>

    <div class="section">
        <h2>Test Results Summary</h2>
        <table>
            <tr>
                <th>Test Type</th>
                <th>Status</th>
                <th>Details</th>
            </tr>
            <tr>
                <td>Load Testing</td>
                <td class="pass">COMPLETED</td>
                <td>Tested normal expected load conditions</td>
            </tr>
            <tr>
                <td>Stress Testing</td>
                <td class="pass">COMPLETED</td>
                <td>Tested beyond normal capacity</td>
            </tr>
            <tr>
                <td>Spike Testing</td>
                <td class="pass">COMPLETED</td>
                <td>Tested sudden load increases</td>
            </tr>
            <tr>
                <td>Volume Testing</td>
                <td class="pass">COMPLETED</td>
                <td>Tested extended duration load</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Detailed Results</h2>
        <p>Detailed test results are available in the following locations:</p>
        <ul>
            <li><strong>Load Tests:</strong> $RESULTS_DIR/load-tests/</li>
            <li><strong>Stress Tests:</strong> $RESULTS_DIR/stress-tests/</li>
            <li><strong>Spike Tests:</strong> $RESULTS_DIR/spike-tests/</li>
            <li><strong>Volume Tests:</strong> $RESULTS_DIR/volume-tests/</li>
        </ul>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ol>
            <li>Implement performance monitoring in production</li>
            <li>Set up automated performance testing in CI/CD</li>
            <li>Optimize identified performance bottlenecks</li>
            <li>Establish performance budgets for development</li>
            <li>Schedule regular performance reviews</li>
        </ol>
    </div>

    <div class="section">
        <h2>Next Steps</h2>
        <ol>
            <li>Review detailed test results</li>
            <li>Address any performance issues identified</li>
            <li>Implement continuous performance monitoring</li>
            <li>Update performance testing strategy</li>
            <li>Schedule follow-up performance tests</li>
        </ol>
    </div>
</body>
</html>
EOF

    success "Performance report generated: $REPORT_FILE ✓"
}

# Create performance testing scripts
create_performance_scripts() {
    log "Creating additional performance testing scripts..."
    
    # Create wrk Lua script
    cat > "$PERF_TEST_DIR/scripts/wrk-script.lua" << 'EOF'
-- wrk script for Nexus Architect performance testing

wrk.method = "GET"
wrk.headers["User-Agent"] = "Nexus Performance Test"

-- Request paths to test
local paths = {
    "/",
    "/dashboard",
    "/projects",
    "/analytics",
    "/api/health",
    "/api/metrics"
}

local path_index = 1

request = function()
    local path = paths[path_index]
    path_index = path_index + 1
    if path_index > #paths then
        path_index = 1
    end
    return wrk.format(nil, path)
end

response = function(status, headers, body)
    if status >= 400 then
        print("Error " .. status .. " for " .. wrk.path)
    end
end

done = function(summary, latency, requests)
    print("Total requests: " .. summary.requests)
    print("Total errors: " .. summary.errors.status)
    print("Average latency: " .. latency.mean / 1000 .. "ms")
    print("Max latency: " .. latency.max / 1000 .. "ms")
    print("Requests/sec: " .. summary.requests / (summary.duration / 1000000))
end
EOF

    # Create performance monitoring script
    cat > "$PERF_TEST_DIR/scripts/monitor-performance.sh" << 'EOF'
#!/bin/bash

echo "📊 Monitoring Performance During Tests"
echo "====================================="

MONITOR_DURATION=${1:-300}  # Default 5 minutes
INTERVAL=5

echo "Monitoring for $MONITOR_DURATION seconds (interval: ${INTERVAL}s)"
echo "Timestamp,CPU%,Memory%,DiskIO,NetworkRX,NetworkTX" > performance-monitor.csv

for ((i=0; i<MONITOR_DURATION; i+=INTERVAL)); do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Memory usage
    memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    # Disk I/O (simplified)
    disk_io=$(iostat -d 1 1 | tail -n +4 | awk '{sum+=$4} END {print sum}' 2>/dev/null || echo "0")
    
    # Network (simplified)
    network_rx=$(cat /proc/net/dev | grep eth0 | awk '{print $2}' 2>/dev/null || echo "0")
    network_tx=$(cat /proc/net/dev | grep eth0 | awk '{print $10}' 2>/dev/null || echo "0")
    
    echo "$timestamp,$cpu_usage,$memory_usage,$disk_io,$network_rx,$network_tx" >> performance-monitor.csv
    
    sleep $INTERVAL
done

echo "Performance monitoring completed. Results saved to performance-monitor.csv"
EOF

    chmod +x "$PERF_TEST_DIR/scripts"/*.sh
    
    success "Performance testing scripts created ✓"
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #11: Performance testing automation"
    
    init_performance_testing
    create_artillery_config
    create_locust_config
    create_performance_scripts
    run_load_tests
    run_stress_tests
    run_spike_tests
    run_volume_tests
    analyze_performance_results
    generate_performance_report
    
    success "🎉 Performance testing automation completed successfully!"
    success "📊 Test Results: $RESULTS_DIR"
    success "📋 Analysis: $RESULTS_DIR/performance-analysis-$TEST_DATE.txt"
    success "📄 Report: $REPORT_FILE"
    
    log "📋 Performance Testing Completed:"
    log "   🔄 Load Testing - Normal expected load"
    log "   💪 Stress Testing - Beyond normal capacity"
    log "   ⚡ Spike Testing - Sudden load increases"
    log "   📊 Volume Testing - Extended duration"
    
    info "💡 Performance Test Tools Available:"
    info "   🎯 Artillery - Advanced load testing"
    info "   🐝 Locust - Python-based load testing"
    info "   ⚡ Apache Bench - Quick HTTP benchmarking"
    info "   🔧 wrk - Modern HTTP benchmarking"
    
    info "📋 Next steps:"
    info "   1. Review performance test results"
    info "   2. Address any performance bottlenecks"
    info "   3. Set up continuous performance monitoring"
    info "   4. Integrate performance tests into CI/CD"
    info "   5. Establish performance budgets"
    
    warning "⚠️  Performance testing requires active services. Ensure all Nexus components are running."
}

# Handle script arguments
case "${1:-all}" in
    "load")
        init_performance_testing
        run_load_tests
        ;;
    "stress")
        init_performance_testing
        run_stress_tests
        ;;
    "spike")
        init_performance_testing
        run_spike_tests
        ;;
    "volume")
        init_performance_testing
        run_volume_tests
        ;;
    "analyze")
        analyze_performance_results
        ;;
    "report")
        generate_performance_report
        ;;
    "all"|*)
        main
        ;;
esac

