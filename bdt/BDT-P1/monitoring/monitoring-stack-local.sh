#!/bin/bash

# Nexus Architect - Local Monitoring Stack
# BDT-P1 Deliverable #10: Local monitoring stack (Prometheus, Grafana, ELK)
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

# Monitoring configuration
MONITORING_DIR="$HOME/nexus-dev/monitoring"
PROMETHEUS_VERSION="v2.47.2"
GRAFANA_VERSION="10.2.0"
ELASTICSEARCH_VERSION="8.11.0"
KIBANA_VERSION="8.11.0"
LOGSTASH_VERSION="8.11.0"

# Initialize monitoring environment
init_monitoring_environment() {
    log "Initializing monitoring environment..."
    
    # Create monitoring directory structure
    mkdir -p "$MONITORING_DIR"/{prometheus,grafana,elasticsearch,kibana,logstash,alertmanager}
    mkdir -p "$MONITORING_DIR/prometheus"/{config,data,rules}
    mkdir -p "$MONITORING_DIR/grafana"/{data,dashboards,provisioning/{dashboards,datasources}}
    mkdir -p "$MONITORING_DIR/elasticsearch"/{data,config}
    mkdir -p "$MONITORING_DIR/kibana"/{config,data}
    mkdir -p "$MONITORING_DIR/logstash"/{config,pipeline}
    mkdir -p "$MONITORING_DIR/alertmanager"/{config,data}
    
    # Set proper permissions
    chmod 755 "$MONITORING_DIR"
    chmod 777 "$MONITORING_DIR/prometheus/data"
    chmod 777 "$MONITORING_DIR/grafana/data"
    chmod 777 "$MONITORING_DIR/elasticsearch/data"
    
    success "Monitoring environment initialized ✓"
}

# Setup Prometheus configuration
setup_prometheus() {
    log "Setting up Prometheus configuration..."
    
    # Create Prometheus configuration
    cat > "$MONITORING_DIR/prometheus/config/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # cAdvisor for container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # Nexus Frontend Applications
  - job_name: 'nexus-frontend'
    static_configs:
      - targets: ['host.docker.internal:3000', 'host.docker.internal:3001']
    metrics_path: '/metrics'
    scrape_interval: 30s

  # Nexus Backend Services
  - job_name: 'nexus-backend'
    static_configs:
      - targets: ['host.docker.internal:8001', 'host.docker.internal:8002', 'host.docker.internal:8003']
    metrics_path: '/metrics'
    scrape_interval: 30s

  # Database Metrics
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  # NGINX Metrics
  - job_name: 'nginx-exporter'
    static_configs:
      - targets: ['nginx-exporter:9113']

  # Elasticsearch Metrics
  - job_name: 'elasticsearch-exporter'
    static_configs:
      - targets: ['elasticsearch-exporter:9114']
EOF

    # Create alerting rules
    cat > "$MONITORING_DIR/prometheus/rules/nexus-alerts.yml" << 'EOF'
groups:
  - name: nexus-alerts
    rules:
      # High CPU Usage
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes on {{ $labels.instance }}"

      # High Memory Usage
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes on {{ $labels.instance }}"

      # High Disk Usage
      - alert: HighDiskUsage
        expr: (node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100 > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High disk usage detected"
          description: "Disk usage is above 90% on {{ $labels.instance }} filesystem {{ $labels.mountpoint }}"

      # Service Down
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.job }} service is down on {{ $labels.instance }}"

      # High Response Time
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is above 1 second for {{ $labels.job }}"

      # Database Connection Issues
      - alert: DatabaseConnectionHigh
        expr: pg_stat_activity_count > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High database connections"
          description: "PostgreSQL has more than 80 active connections"

      # Redis Memory Usage
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis memory usage high"
          description: "Redis memory usage is above 90%"
EOF

    success "Prometheus configuration created ✓"
}

# Setup Grafana configuration
setup_grafana() {
    log "Setting up Grafana configuration..."
    
    # Create Grafana datasource configuration
    cat > "$MONITORING_DIR/grafana/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    database: "logstash-*"
    interval: Daily
    timeField: "@timestamp"
    editable: true
EOF

    # Create Grafana dashboard provisioning
    cat > "$MONITORING_DIR/grafana/provisioning/dashboards/dashboards.yml" << 'EOF'
apiVersion: 1

providers:
  - name: 'nexus-dashboards'
    orgId: 1
    folder: 'Nexus Architect'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    # Create Nexus Overview Dashboard
    cat > "$MONITORING_DIR/grafana/dashboards/nexus-overview.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Nexus Architect - System Overview",
    "tags": ["nexus", "overview"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "System CPU Usage",
        "type": "stat",
        "targets": [
          {
            "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 70},
                {"color": "red", "value": 90}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Memory Usage",
        "type": "stat",
        "targets": [
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "Memory Usage %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 70},
                {"color": "red", "value": 90}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
      },
      {
        "id": 3,
        "title": "Active Services",
        "type": "stat",
        "targets": [
          {
            "expr": "count(up == 1)",
            "legendFormat": "Services Up"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "thresholds": {
              "steps": [
                {"color": "red", "value": null},
                {"color": "yellow", "value": 5},
                {"color": "green", "value": 8}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
      },
      {
        "id": 4,
        "title": "Response Time (95th percentile)",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "Response Time"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 0.5},
                {"color": "red", "value": 1}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "30s"
  }
}
EOF

    success "Grafana configuration created ✓"
}

# Setup ELK Stack configuration
setup_elk_stack() {
    log "Setting up ELK Stack configuration..."
    
    # Create Elasticsearch configuration
    cat > "$MONITORING_DIR/elasticsearch/config/elasticsearch.yml" << 'EOF'
cluster.name: "nexus-logs"
node.name: "nexus-es-node"
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
xpack.monitoring.collection.enabled: true
EOF

    # Create Kibana configuration
    cat > "$MONITORING_DIR/kibana/config/kibana.yml" << 'EOF'
server.name: nexus-kibana
server.host: "0.0.0.0"
elasticsearch.hosts: ["http://elasticsearch:9200"]
monitoring.ui.container.elasticsearch.enabled: true
EOF

    # Create Logstash configuration
    cat > "$MONITORING_DIR/logstash/config/logstash.yml" << 'EOF'
http.host: "0.0.0.0"
xpack.monitoring.elasticsearch.hosts: ["http://elasticsearch:9200"]
EOF

    # Create Logstash pipeline configuration
    cat > "$MONITORING_DIR/logstash/pipeline/nexus.conf" << 'EOF'
input {
  beats {
    port => 5044
  }
  
  tcp {
    port => 5000
    codec => json_lines
  }
  
  udp {
    port => 5000
    codec => json_lines
  }
}

filter {
  if [fields][service] {
    mutate {
      add_field => { "service_name" => "%{[fields][service]}" }
    }
  }
  
  if [message] =~ /ERROR/ {
    mutate {
      add_field => { "log_level" => "ERROR" }
    }
  } else if [message] =~ /WARN/ {
    mutate {
      add_field => { "log_level" => "WARN" }
    }
  } else if [message] =~ /INFO/ {
    mutate {
      add_field => { "log_level" => "INFO" }
    }
  } else {
    mutate {
      add_field => { "log_level" => "DEBUG" }
    }
  }
  
  date {
    match => [ "timestamp", "ISO8601" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "nexus-logs-%{+YYYY.MM.dd}"
  }
  
  stdout {
    codec => rubydebug
  }
}
EOF

    success "ELK Stack configuration created ✓"
}

# Create monitoring Docker Compose
create_monitoring_compose() {
    log "Creating monitoring Docker Compose configuration..."
    
    cat > "$MONITORING_DIR/docker-compose.monitoring.yml" << EOF
version: '3.8'

services:
  # Prometheus
  prometheus:
    image: prom/prometheus:$PROMETHEUS_VERSION
    container_name: nexus-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/config:/etc/prometheus
      - ./prometheus/data:/prometheus
      - ./prometheus/rules:/etc/prometheus/rules
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - nexus-monitoring

  # Grafana
  grafana:
    image: grafana/grafana:$GRAFANA_VERSION
    container_name: nexus-grafana
    restart: unless-stopped
    ports:
      - "3001:3000"
    volumes:
      - ./grafana/data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=nexus_grafana_admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    networks:
      - nexus-monitoring

  # Node Exporter
  node-exporter:
    image: prom/node-exporter:latest
    container_name: nexus-node-exporter
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - nexus-monitoring

  # cAdvisor
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: nexus-cadvisor
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg:/dev/kmsg
    networks:
      - nexus-monitoring

  # Elasticsearch
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:$ELASTICSEARCH_VERSION
    container_name: nexus-elasticsearch
    restart: unless-stopped
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - ./elasticsearch/data:/usr/share/elasticsearch/data
      - ./elasticsearch/config/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    networks:
      - nexus-monitoring

  # Kibana
  kibana:
    image: docker.elastic.co/kibana/kibana:$KIBANA_VERSION
    container_name: nexus-kibana
    restart: unless-stopped
    ports:
      - "5601:5601"
    volumes:
      - ./kibana/config/kibana.yml:/usr/share/kibana/config/kibana.yml
    depends_on:
      - elasticsearch
    networks:
      - nexus-monitoring

  # Logstash
  logstash:
    image: docker.elastic.co/logstash/logstash:$LOGSTASH_VERSION
    container_name: nexus-logstash
    restart: unless-stopped
    ports:
      - "5044:5044"
      - "5000:5000/tcp"
      - "5000:5000/udp"
      - "9600:9600"
    volumes:
      - ./logstash/config/logstash.yml:/usr/share/logstash/config/logstash.yml
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    depends_on:
      - elasticsearch
    networks:
      - nexus-monitoring

  # AlertManager
  alertmanager:
    image: prom/alertmanager:latest
    container_name: nexus-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager/config:/etc/alertmanager
      - ./alertmanager/data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    networks:
      - nexus-monitoring

networks:
  nexus-monitoring:
    driver: bridge

volumes:
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  elasticsearch_data:
    driver: local
EOF

    success "Monitoring Docker Compose created ✓"
}

# Create monitoring startup script
create_monitoring_startup() {
    log "Creating monitoring startup script..."
    
    cat > "$HOME/nexus-dev/start-monitoring.sh" << 'EOF'
#!/bin/bash

echo "📊 Starting Nexus Architect Monitoring Stack"
echo "============================================"

MONITORING_DIR="$HOME/nexus-dev/monitoring"

# Start monitoring services
echo "🚀 Starting monitoring services..."
cd "$MONITORING_DIR"
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 60

# Check service health
echo "🏥 Checking service health..."

# Check Prometheus
if curl -f -s "http://localhost:9090/-/healthy" > /dev/null; then
    echo "✅ Prometheus is ready"
else
    echo "⚠️  Prometheus is not ready yet"
fi

# Check Grafana
if curl -f -s "http://localhost:3001/api/health" > /dev/null; then
    echo "✅ Grafana is ready"
else
    echo "⚠️  Grafana is not ready yet"
fi

# Check Elasticsearch
if curl -f -s "http://localhost:9200/_cluster/health" > /dev/null; then
    echo "✅ Elasticsearch is ready"
else
    echo "⚠️  Elasticsearch is not ready yet"
fi

# Check Kibana
if curl -f -s "http://localhost:5601/api/status" > /dev/null; then
    echo "✅ Kibana is ready"
else
    echo "⚠️  Kibana is not ready yet"
fi

echo ""
echo "🎉 Monitoring stack started!"
echo "📊 Prometheus: http://localhost:9090"
echo "📈 Grafana: http://localhost:3001 (admin/nexus_grafana_admin)"
echo "🔍 Kibana: http://localhost:5601"
echo "🚨 AlertManager: http://localhost:9093"
echo "📊 Node Exporter: http://localhost:9100"
echo "🐳 cAdvisor: http://localhost:8080"
EOF

    chmod +x "$HOME/nexus-dev/start-monitoring.sh"
    success "Monitoring startup script created ✓"
}

# Create monitoring test script
create_monitoring_tests() {
    log "Creating monitoring test script..."
    
    cat > "$MONITORING_DIR/test-monitoring.sh" << 'EOF'
#!/bin/bash

echo "🧪 Testing Nexus Architect Monitoring Stack"
echo "==========================================="

# Test Prometheus
echo "📊 Testing Prometheus..."
if curl -f -s "http://localhost:9090/api/v1/query?query=up" | jq '.status' | grep -q "success"; then
    echo "✅ Prometheus API working"
    
    # Check targets
    targets=$(curl -s "http://localhost:9090/api/v1/targets" | jq '.data.activeTargets | length')
    echo "📋 Active targets: $targets"
else
    echo "❌ Prometheus API failed"
fi

# Test Grafana
echo "📈 Testing Grafana..."
if curl -f -s "http://localhost:3001/api/health" | jq '.database' | grep -q "ok"; then
    echo "✅ Grafana health check passed"
    
    # Check datasources
    datasources=$(curl -s -u admin:nexus_grafana_admin "http://localhost:3001/api/datasources" | jq 'length')
    echo "📋 Configured datasources: $datasources"
else
    echo "❌ Grafana health check failed"
fi

# Test Elasticsearch
echo "🔍 Testing Elasticsearch..."
if curl -f -s "http://localhost:9200/_cluster/health" | jq '.status' | grep -q "green\|yellow"; then
    echo "✅ Elasticsearch cluster healthy"
    
    # Check indices
    indices=$(curl -s "http://localhost:9200/_cat/indices?format=json" | jq 'length')
    echo "📋 Available indices: $indices"
else
    echo "❌ Elasticsearch cluster unhealthy"
fi

# Test Kibana
echo "📊 Testing Kibana..."
if curl -f -s "http://localhost:5601/api/status" | jq '.status.overall.state' | grep -q "green"; then
    echo "✅ Kibana status green"
else
    echo "❌ Kibana status not green"
fi

# Test log ingestion
echo "📝 Testing log ingestion..."
test_log='{"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'","level":"INFO","service":"test","message":"Monitoring test log entry"}'
echo "$test_log" | nc localhost 5000

sleep 5

# Check if log was ingested
log_count=$(curl -s "http://localhost:9200/nexus-logs-*/_search?q=service:test" | jq '.hits.total.value // .hits.total')
if [[ "$log_count" -gt 0 ]]; then
    echo "✅ Log ingestion working ($log_count logs found)"
else
    echo "⚠️  Log ingestion may not be working"
fi

echo ""
echo "🎉 Monitoring stack test completed!"
EOF

    chmod +x "$MONITORING_DIR/test-monitoring.sh"
    success "Monitoring test script created ✓"
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #10: Local monitoring stack (Prometheus, Grafana, ELK)"
    
    init_monitoring_environment
    setup_prometheus
    setup_grafana
    setup_elk_stack
    create_monitoring_compose
    create_monitoring_startup
    create_monitoring_tests
    
    success "🎉 Monitoring stack setup completed successfully!"
    success "📊 Configuration: $MONITORING_DIR"
    success "🚀 Startup script: $HOME/nexus-dev/start-monitoring.sh"
    success "🧪 Test script: $MONITORING_DIR/test-monitoring.sh"
    
    log "📋 Monitoring Stack Components:"
    log "   📊 Prometheus - Metrics collection and alerting"
    log "   📈 Grafana - Visualization and dashboards"
    log "   🔍 Elasticsearch - Log storage and search"
    log "   📊 Kibana - Log visualization and analysis"
    log "   📝 Logstash - Log processing and ingestion"
    log "   🚨 AlertManager - Alert routing and management"
    log "   📊 Node Exporter - System metrics"
    log "   🐳 cAdvisor - Container metrics"
    
    info "💡 Next steps:"
    info "   1. Start monitoring stack: ~/nexus-dev/start-monitoring.sh"
    info "   2. Test monitoring: ~/nexus-dev/monitoring/test-monitoring.sh"
    info "   3. Access Grafana: http://localhost:3001 (admin/nexus_grafana_admin)"
    info "   4. Access Prometheus: http://localhost:9090"
    info "   5. Access Kibana: http://localhost:5601"
    
    warning "⚠️  Allow 2-3 minutes for all services to fully start up"
}

# Run main function
main "$@"

