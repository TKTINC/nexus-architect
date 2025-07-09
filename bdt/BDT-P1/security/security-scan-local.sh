#!/bin/bash

# Nexus Architect - Local Security Scanning
# BDT-P1 Deliverable #8: Local security scanning and vulnerability checks
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

# Security scan configuration
SCAN_RESULTS_DIR="$HOME/nexus-dev/security-scans"
SCAN_DATE=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$SCAN_RESULTS_DIR/security-report-$SCAN_DATE.txt"

# Initialize security scanning environment
init_security_scanning() {
    log "Initializing security scanning environment..."
    
    # Create scan results directory
    mkdir -p "$SCAN_RESULTS_DIR"/{reports,logs,evidence}
    
    # Create report header
    cat > "$REPORT_FILE" << EOF
Nexus Architect Local Security Scan Report
==========================================
Scan Date: $(date)
Scan ID: $SCAN_DATE
Environment: Local Development
Scope: All Nexus Architect components

EXECUTIVE SUMMARY
================
This report contains the results of automated security scanning
performed on the local development environment of Nexus Architect.

EOF

    success "Security scanning environment initialized ✓"
}

# Install security scanning tools
install_security_tools() {
    log "Installing security scanning tools..."
    
    # Check if tools are already installed
    local tools_needed=()
    
    if ! command -v nmap &> /dev/null; then
        tools_needed+=("nmap")
    fi
    
    if ! command -v nikto &> /dev/null; then
        tools_needed+=("nikto")
    fi
    
    if ! command -v sqlmap &> /dev/null; then
        tools_needed+=("sqlmap")
    fi
    
    # Install missing tools
    if [[ ${#tools_needed[@]} -gt 0 ]]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update
            for tool in "${tools_needed[@]}"; do
                sudo apt-get install -y "$tool"
            done
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                for tool in "${tools_needed[@]}"; do
                    brew install "$tool"
                done
            else
                warning "Homebrew not found. Please install security tools manually."
            fi
        fi
    fi
    
    # Install Python security tools
    pip3 install --user bandit safety semgrep
    
    # Install Node.js security tools
    npm install -g audit-ci retire snyk
    
    success "Security scanning tools installed ✓"
}

# Perform network security scan
scan_network_security() {
    log "Performing network security scan..."
    
    echo "NETWORK SECURITY SCAN" >> "$REPORT_FILE"
    echo "====================" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Port scan
    log "Scanning open ports..."
    local open_ports=$(nmap -sT -O localhost 2>/dev/null | grep "open" | awk '{print $1}' | cut -d'/' -f1 | tr '\n' ' ')
    
    echo "Open Ports: $open_ports" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Check for unnecessary services
    log "Checking for unnecessary services..."
    local risky_ports=("21" "23" "25" "53" "135" "139" "445" "1433" "3389")
    local found_risky=()
    
    for port in "${risky_ports[@]}"; do
        if echo "$open_ports" | grep -q "$port"; then
            found_risky+=("$port")
        fi
    done
    
    if [[ ${#found_risky[@]} -gt 0 ]]; then
        warning "Potentially risky ports found: ${found_risky[*]}"
        echo "WARNING: Risky ports detected: ${found_risky[*]}" >> "$REPORT_FILE"
    else
        success "No obviously risky ports detected ✓"
        echo "INFO: No obviously risky ports detected" >> "$REPORT_FILE"
    fi
    
    echo "" >> "$REPORT_FILE"
    
    # SSL/TLS security check
    log "Checking SSL/TLS security..."
    if [[ -f "$HOME/nexus-dev/certs/ssl/server-cert.pem" ]]; then
        local ssl_info=$(openssl x509 -in "$HOME/nexus-dev/certs/ssl/server-cert.pem" -text -noout)
        local key_size=$(echo "$ssl_info" | grep "Public-Key:" | awk '{print $2}' | tr -d '()')
        local signature_algo=$(echo "$ssl_info" | grep "Signature Algorithm:" | head -1 | awk '{print $3}')
        
        echo "SSL Certificate Analysis:" >> "$REPORT_FILE"
        echo "  Key Size: $key_size" >> "$REPORT_FILE"
        echo "  Signature Algorithm: $signature_algo" >> "$REPORT_FILE"
        
        if [[ "$key_size" -ge 2048 ]]; then
            success "SSL key size adequate ($key_size bits) ✓"
            echo "  Key Size Status: SECURE" >> "$REPORT_FILE"
        else
            warning "SSL key size may be insufficient ($key_size bits)"
            echo "  Key Size Status: WEAK" >> "$REPORT_FILE"
        fi
        
        if [[ "$signature_algo" == *"sha256"* ]] || [[ "$signature_algo" == *"sha384"* ]] || [[ "$signature_algo" == *"sha512"* ]]; then
            success "SSL signature algorithm secure ($signature_algo) ✓"
            echo "  Signature Algorithm Status: SECURE" >> "$REPORT_FILE"
        else
            warning "SSL signature algorithm may be weak ($signature_algo)"
            echo "  Signature Algorithm Status: WEAK" >> "$REPORT_FILE"
        fi
    fi
    
    echo "" >> "$REPORT_FILE"
    success "Network security scan completed ✓"
}

# Perform web application security scan
scan_web_application() {
    log "Performing web application security scan..."
    
    echo "WEB APPLICATION SECURITY SCAN" >> "$REPORT_FILE"
    echo "=============================" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Check if web services are running
    local web_endpoints=(
        "http://localhost:3000"
        "http://localhost:3001"
        "http://localhost:8001"
        "http://localhost:8002"
        "http://localhost:8003"
        "http://localhost:8004"
    )
    
    local active_endpoints=()
    
    for endpoint in "${web_endpoints[@]}"; do
        if curl -f -s --max-time 5 "$endpoint" &>/dev/null; then
            active_endpoints+=("$endpoint")
        fi
    done
    
    echo "Active Web Endpoints: ${active_endpoints[*]}" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Basic web security headers check
    log "Checking security headers..."
    for endpoint in "${active_endpoints[@]}"; do
        log "Scanning $endpoint..."
        
        local headers=$(curl -I -s --max-time 10 "$endpoint" 2>/dev/null || echo "")
        
        echo "Security Headers Analysis for $endpoint:" >> "$REPORT_FILE"
        
        # Check for security headers
        local security_headers=(
            "Strict-Transport-Security"
            "X-Content-Type-Options"
            "X-Frame-Options"
            "X-XSS-Protection"
            "Content-Security-Policy"
            "Referrer-Policy"
        )
        
        for header in "${security_headers[@]}"; do
            if echo "$headers" | grep -qi "$header"; then
                echo "  ✅ $header: Present" >> "$REPORT_FILE"
            else
                echo "  ❌ $header: Missing" >> "$REPORT_FILE"
            fi
        done
        
        # Check for information disclosure headers
        local disclosure_headers=(
            "Server"
            "X-Powered-By"
            "X-AspNet-Version"
        )
        
        for header in "${disclosure_headers[@]}"; do
            if echo "$headers" | grep -qi "$header"; then
                local value=$(echo "$headers" | grep -i "$header" | cut -d: -f2- | xargs)
                echo "  ⚠️  $header: $value (Information Disclosure)" >> "$REPORT_FILE"
            fi
        done
        
        echo "" >> "$REPORT_FILE"
    done
    
    # Nikto web vulnerability scan (if available and endpoints are active)
    if command -v nikto &> /dev/null && [[ ${#active_endpoints[@]} -gt 0 ]]; then
        log "Running Nikto vulnerability scan..."
        for endpoint in "${active_endpoints[@]}"; do
            local nikto_output="$SCAN_RESULTS_DIR/logs/nikto-$(echo $endpoint | sed 's|[^a-zA-Z0-9]|_|g').log"
            nikto -h "$endpoint" -output "$nikto_output" -Format txt &>/dev/null || true
            
            if [[ -f "$nikto_output" ]]; then
                echo "Nikto Scan Results for $endpoint:" >> "$REPORT_FILE"
                echo "  Detailed results: $nikto_output" >> "$REPORT_FILE"
                
                # Extract summary
                local issues=$(grep -c "OSVDB\|CVE" "$nikto_output" 2>/dev/null || echo "0")
                echo "  Issues Found: $issues" >> "$REPORT_FILE"
                echo "" >> "$REPORT_FILE"
            fi
        done
    fi
    
    success "Web application security scan completed ✓"
}

# Perform dependency vulnerability scan
scan_dependencies() {
    log "Performing dependency vulnerability scan..."
    
    echo "DEPENDENCY VULNERABILITY SCAN" >> "$REPORT_FILE"
    echo "=============================" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Node.js dependency scan
    log "Scanning Node.js dependencies..."
    local node_projects=($(find implementation/ -name "package.json" -type f))
    
    for package_file in "${node_projects[@]}"; do
        local package_dir=$(dirname "$package_file")
        local project_name=$(basename "$package_dir")
        
        log "Scanning $project_name..."
        
        (
            cd "$package_dir"
            
            echo "Node.js Dependency Scan - $project_name:" >> "$REPORT_FILE"
            
            # npm audit
            if [[ -f "package-lock.json" ]]; then
                local audit_output=$(npm audit --json 2>/dev/null || echo '{"vulnerabilities":{}}')
                local vuln_count=$(echo "$audit_output" | jq '.metadata.vulnerabilities.total // 0' 2>/dev/null || echo "0")
                echo "  npm audit vulnerabilities: $vuln_count" >> "$REPORT_FILE"
                
                if [[ "$vuln_count" -gt 0 ]]; then
                    warning "$project_name has $vuln_count npm vulnerabilities"
                    echo "$audit_output" > "$SCAN_RESULTS_DIR/logs/npm-audit-$project_name.json"
                    echo "  Detailed report: $SCAN_RESULTS_DIR/logs/npm-audit-$project_name.json" >> "$REPORT_FILE"
                fi
            fi
            
            # Retire.js scan
            if command -v retire &> /dev/null; then
                local retire_output="$SCAN_RESULTS_DIR/logs/retire-$project_name.log"
                retire --outputformat text --outputpath "$retire_output" . &>/dev/null || true
                
                if [[ -f "$retire_output" ]]; then
                    local retire_issues=$(grep -c "vulnerability" "$retire_output" 2>/dev/null || echo "0")
                    echo "  retire.js vulnerabilities: $retire_issues" >> "$REPORT_FILE"
                fi
            fi
            
            echo "" >> "$REPORT_FILE"
        )
    done
    
    # Python dependency scan
    log "Scanning Python dependencies..."
    local python_projects=($(find implementation/ -name "requirements.txt" -type f))
    
    for req_file in "${python_projects[@]}"; do
        local req_dir=$(dirname "$req_file")
        local project_name=$(basename "$req_dir")
        
        log "Scanning $project_name..."
        
        (
            cd "$req_dir"
            
            echo "Python Dependency Scan - $project_name:" >> "$REPORT_FILE"
            
            # Safety scan
            if command -v safety &> /dev/null; then
                local safety_output="$SCAN_RESULTS_DIR/logs/safety-$project_name.log"
                safety check --json --output "$safety_output" 2>/dev/null || true
                
                if [[ -f "$safety_output" ]]; then
                    local safety_issues=$(jq length "$safety_output" 2>/dev/null || echo "0")
                    echo "  safety vulnerabilities: $safety_issues" >> "$REPORT_FILE"
                    
                    if [[ "$safety_issues" -gt 0 ]]; then
                        warning "$project_name has $safety_issues Python vulnerabilities"
                    fi
                fi
            fi
            
            echo "" >> "$REPORT_FILE"
        )
    done
    
    success "Dependency vulnerability scan completed ✓"
}

# Perform code security analysis
scan_code_security() {
    log "Performing code security analysis..."
    
    echo "CODE SECURITY ANALYSIS" >> "$REPORT_FILE"
    echo "======================" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Python code security with Bandit
    log "Scanning Python code with Bandit..."
    local python_projects=($(find implementation/ -name "*.py" -type f | head -1 | xargs dirname 2>/dev/null || echo ""))
    
    if [[ -n "$python_projects" ]] && command -v bandit &> /dev/null; then
        local bandit_output="$SCAN_RESULTS_DIR/logs/bandit-scan.json"
        bandit -r implementation/ -f json -o "$bandit_output" &>/dev/null || true
        
        if [[ -f "$bandit_output" ]]; then
            local bandit_issues=$(jq '.results | length' "$bandit_output" 2>/dev/null || echo "0")
            echo "Bandit Security Issues: $bandit_issues" >> "$REPORT_FILE"
            
            if [[ "$bandit_issues" -gt 0 ]]; then
                warning "Found $bandit_issues potential security issues in Python code"
                echo "  Detailed report: $bandit_output" >> "$REPORT_FILE"
            else
                success "No security issues found in Python code ✓"
            fi
        fi
    fi
    
    # Semgrep security scan
    if command -v semgrep &> /dev/null; then
        log "Running Semgrep security analysis..."
        local semgrep_output="$SCAN_RESULTS_DIR/logs/semgrep-scan.json"
        semgrep --config=auto --json --output="$semgrep_output" implementation/ &>/dev/null || true
        
        if [[ -f "$semgrep_output" ]]; then
            local semgrep_issues=$(jq '.results | length' "$semgrep_output" 2>/dev/null || echo "0")
            echo "Semgrep Security Issues: $semgrep_issues" >> "$REPORT_FILE"
            
            if [[ "$semgrep_issues" -gt 0 ]]; then
                warning "Found $semgrep_issues potential security issues with Semgrep"
                echo "  Detailed report: $semgrep_output" >> "$REPORT_FILE"
            else
                success "No security issues found with Semgrep ✓"
            fi
        fi
    fi
    
    echo "" >> "$REPORT_FILE"
    success "Code security analysis completed ✓"
}

# Perform configuration security check
scan_configuration_security() {
    log "Performing configuration security check..."
    
    echo "CONFIGURATION SECURITY CHECK" >> "$REPORT_FILE"
    echo "============================" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Check environment files for sensitive data
    log "Checking environment files..."
    local env_files=($(find . -name ".env*" -type f 2>/dev/null))
    
    echo "Environment File Security:" >> "$REPORT_FILE"
    
    for env_file in "${env_files[@]}"; do
        echo "  File: $env_file" >> "$REPORT_FILE"
        
        # Check for weak passwords
        local weak_passwords=$(grep -i "password.*=.*\(password\|123\|admin\|test\)" "$env_file" 2>/dev/null || echo "")
        if [[ -n "$weak_passwords" ]]; then
            warning "Weak passwords found in $env_file"
            echo "    ⚠️  Weak passwords detected" >> "$REPORT_FILE"
        else
            echo "    ✅ No obvious weak passwords" >> "$REPORT_FILE"
        fi
        
        # Check for hardcoded secrets
        local hardcoded_secrets=$(grep -E "(secret|key|token).*=.*[a-zA-Z0-9]{20,}" "$env_file" 2>/dev/null | wc -l)
        echo "    Hardcoded secrets: $hardcoded_secrets" >> "$REPORT_FILE"
        
        # Check file permissions
        local permissions=$(stat -c "%a" "$env_file" 2>/dev/null || stat -f "%A" "$env_file" 2>/dev/null || echo "unknown")
        if [[ "$permissions" == "600" ]] || [[ "$permissions" == "400" ]]; then
            echo "    ✅ File permissions secure ($permissions)" >> "$REPORT_FILE"
        else
            warning "Environment file $env_file has permissive permissions ($permissions)"
            echo "    ⚠️  File permissions may be too permissive ($permissions)" >> "$REPORT_FILE"
        fi
    done
    
    echo "" >> "$REPORT_FILE"
    
    # Check Docker configuration
    log "Checking Docker configuration..."
    if [[ -f "bdt/BDT-P1/docker/docker-compose.dev.yml" ]]; then
        echo "Docker Configuration Security:" >> "$REPORT_FILE"
        
        # Check for privileged containers
        local privileged=$(grep -i "privileged.*true" bdt/BDT-P1/docker/docker-compose.dev.yml || echo "")
        if [[ -n "$privileged" ]]; then
            warning "Privileged containers detected in Docker configuration"
            echo "  ⚠️  Privileged containers detected" >> "$REPORT_FILE"
        else
            echo "  ✅ No privileged containers" >> "$REPORT_FILE"
        fi
        
        # Check for host network mode
        local host_network=$(grep -i "network_mode.*host" bdt/BDT-P1/docker/docker-compose.dev.yml || echo "")
        if [[ -n "$host_network" ]]; then
            warning "Host network mode detected in Docker configuration"
            echo "  ⚠️  Host network mode detected" >> "$REPORT_FILE"
        else
            echo "  ✅ No host network mode usage" >> "$REPORT_FILE"
        fi
        
        # Check for volume mounts
        local sensitive_mounts=$(grep -E ":/etc|:/var|:/usr|:/root" bdt/BDT-P1/docker/docker-compose.dev.yml || echo "")
        if [[ -n "$sensitive_mounts" ]]; then
            warning "Potentially sensitive volume mounts detected"
            echo "  ⚠️  Sensitive volume mounts detected" >> "$REPORT_FILE"
        else
            echo "  ✅ No obviously sensitive volume mounts" >> "$REPORT_FILE"
        fi
    fi
    
    echo "" >> "$REPORT_FILE"
    success "Configuration security check completed ✓"
}

# Generate security recommendations
generate_security_recommendations() {
    log "Generating security recommendations..."
    
    echo "SECURITY RECOMMENDATIONS" >> "$REPORT_FILE"
    echo "========================" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    cat >> "$REPORT_FILE" << 'EOF'
Based on the security scan results, here are the recommended actions:

IMMEDIATE ACTIONS:
1. Review and address any high-severity vulnerabilities found in dependencies
2. Ensure all environment files have restrictive permissions (600 or 400)
3. Replace any weak or default passwords with strong, unique passwords
4. Verify SSL/TLS certificates are using strong algorithms and key sizes

ONGOING SECURITY PRACTICES:
1. Regularly update dependencies to patch known vulnerabilities
2. Implement automated security scanning in CI/CD pipeline
3. Use secrets management tools instead of hardcoded credentials
4. Enable security headers on all web applications
5. Implement proper input validation and output encoding
6. Use principle of least privilege for all services and users
7. Enable comprehensive logging and monitoring
8. Regularly review and update security configurations

PRODUCTION CONSIDERATIONS:
1. Use proper certificate authorities for SSL certificates
2. Implement Web Application Firewall (WAF)
3. Enable DDoS protection and rate limiting
4. Use container security scanning and runtime protection
5. Implement network segmentation and access controls
6. Enable database encryption at rest and in transit
7. Implement backup encryption and secure storage
8. Conduct regular penetration testing and security audits

COMPLIANCE REQUIREMENTS:
1. Ensure data encryption meets regulatory standards
2. Implement audit logging for compliance requirements
3. Establish data retention and deletion policies
4. Implement access controls and user management
5. Document security procedures and incident response plans
EOF

    echo "" >> "$REPORT_FILE"
    success "Security recommendations generated ✓"
}

# Create security scan summary
create_scan_summary() {
    log "Creating security scan summary..."
    
    local summary_file="$SCAN_RESULTS_DIR/security-summary-$SCAN_DATE.txt"
    
    cat > "$summary_file" << EOF
Nexus Architect Security Scan Summary
====================================
Scan Date: $(date)
Scan ID: $SCAN_DATE

SCAN COVERAGE:
✅ Network Security Scan
✅ Web Application Security Scan  
✅ Dependency Vulnerability Scan
✅ Code Security Analysis
✅ Configuration Security Check

RESULTS SUMMARY:
- Network: $(grep -c "WARNING\|ERROR" "$REPORT_FILE" | head -1 || echo "0") issues found
- Dependencies: Scanned Node.js and Python packages
- Code: Static analysis completed
- Configuration: Environment and Docker configs reviewed

REPORT LOCATIONS:
- Full Report: $REPORT_FILE
- Scan Logs: $SCAN_RESULTS_DIR/logs/
- Evidence: $SCAN_RESULTS_DIR/evidence/

NEXT STEPS:
1. Review full security report
2. Address any high-priority findings
3. Implement recommended security controls
4. Schedule regular security scans

EOF

    success "Security scan summary created: $summary_file ✓"
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #8: Local security scanning and vulnerability checks"
    
    init_security_scanning
    install_security_tools
    scan_network_security
    scan_web_application
    scan_dependencies
    scan_code_security
    scan_configuration_security
    generate_security_recommendations
    create_scan_summary
    
    success "🎉 Security scanning completed successfully!"
    success "📊 Full Report: $REPORT_FILE"
    success "📁 Scan Results: $SCAN_RESULTS_DIR"
    success "🔍 Summary: $SCAN_RESULTS_DIR/security-summary-$SCAN_DATE.txt"
    
    log "📋 Security Scan Results:"
    log "   🔒 Network security assessed"
    log "   🌐 Web application security checked"
    log "   📦 Dependencies scanned for vulnerabilities"
    log "   💻 Code analyzed for security issues"
    log "   ⚙️  Configuration security reviewed"
    
    info "💡 Next steps:"
    info "   1. Review the full security report"
    info "   2. Address any critical or high-severity findings"
    info "   3. Implement recommended security controls"
    info "   4. Run this scan regularly during development"
    
    warning "⚠️  This scan is for development environment only. Production requires additional security measures."
}

# Handle script arguments
case "${1:-all}" in
    "network")
        init_security_scanning
        scan_network_security
        ;;
    "web")
        init_security_scanning
        scan_web_application
        ;;
    "dependencies")
        init_security_scanning
        scan_dependencies
        ;;
    "code")
        init_security_scanning
        scan_code_security
        ;;
    "config")
        init_security_scanning
        scan_configuration_security
        ;;
    "all"|*)
        main
        ;;
esac

