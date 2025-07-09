#!/bin/bash

# Nexus Architect - Local Database Setup
# BDT-P1 Deliverable #3: PostgreSQL/Redis initialization with sample data
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

# Database configuration
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="nexus_dev"
DB_USER="nexus_dev"
DB_PASSWORD="nexus_dev_password"

REDIS_HOST="localhost"
REDIS_PORT="6379"
REDIS_PASSWORD="nexus_redis_password"

# Wait for database to be ready
wait_for_postgres() {
    log "Waiting for PostgreSQL to be ready..."
    
    for i in {1..30}; do
        if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1;" &>/dev/null; then
            success "PostgreSQL is ready ✓"
            return 0
        fi
        log "Waiting for PostgreSQL... (attempt $i/30)"
        sleep 2
    done
    
    error "PostgreSQL is not ready after 60 seconds"
}

# Wait for Redis to be ready
wait_for_redis() {
    log "Waiting for Redis to be ready..."
    
    for i in {1..30}; do
        if redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping &>/dev/null; then
            success "Redis is ready ✓"
            return 0
        fi
        log "Waiting for Redis... (attempt $i/30)"
        sleep 2
    done
    
    error "Redis is not ready after 60 seconds"
}

# Create database schemas
create_schemas() {
    log "Creating database schemas..."
    
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << 'EOF'
-- Nexus Architect Database Schema
-- Created for local development environment

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas for each workstream
CREATE SCHEMA IF NOT EXISTS ws1_core;
CREATE SCHEMA IF NOT EXISTS ws2_ai;
CREATE SCHEMA IF NOT EXISTS ws3_data;
CREATE SCHEMA IF NOT EXISTS ws4_autonomous;
CREATE SCHEMA IF NOT EXISTS ws5_interfaces;
CREATE SCHEMA IF NOT EXISTS shared;

-- WS1 Core Foundation Tables
CREATE TABLE IF NOT EXISTS ws1_core.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ws1_core.organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ws1_core.user_organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES ws1_core.users(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES ws1_core.organizations(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'member',
    permissions JSONB DEFAULT '{}',
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, organization_id)
);

-- WS2 AI Intelligence Tables
CREATE TABLE IF NOT EXISTS ws2_ai.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    config JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ws2_ai.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES ws2_ai.models(id) ON DELETE CASCADE,
    input_data JSONB NOT NULL,
    output_data JSONB NOT NULL,
    confidence DECIMAL(5,4),
    processing_time_ms INTEGER,
    user_id UUID REFERENCES ws1_core.users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- WS3 Data Ingestion Tables
CREATE TABLE IF NOT EXISTS ws3_data.sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    connection_config JSONB NOT NULL,
    schema_config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    last_sync TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ws3_data.ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES ws3_data.sources(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'pending',
    records_processed INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- WS4 Autonomous Capabilities Tables
CREATE TABLE IF NOT EXISTS ws4_autonomous.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    state JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    last_action TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ws4_autonomous.actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES ws4_autonomous.agents(id) ON DELETE CASCADE,
    action_type VARCHAR(100) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    execution_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- WS5 Multi-Role Interfaces Tables
CREATE TABLE IF NOT EXISTS ws5_interfaces.dashboards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES ws1_core.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    layout JSONB DEFAULT '{}',
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ws5_interfaces.widgets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dashboard_id UUID REFERENCES ws5_interfaces.dashboards(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    config JSONB NOT NULL,
    position JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shared Tables
CREATE TABLE IF NOT EXISTS shared.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES ws1_core.users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shared.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON ws1_core.users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON ws1_core.users(username);
CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON ws2_ai.predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON ws2_ai.predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_source_id ON ws3_data.ingestion_jobs(source_id);
CREATE INDEX IF NOT EXISTS idx_actions_agent_id ON ws4_autonomous.actions(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON shared.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON shared.audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON shared.system_metrics(metric_name, timestamp);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON ws1_core.users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON ws1_core.organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON ws2_ai.models FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_sources_updated_at BEFORE UPDATE ON ws3_data.sources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON ws4_autonomous.agents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dashboards_updated_at BEFORE UPDATE ON ws5_interfaces.dashboards FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

EOF

    success "Database schemas created ✓"
}

# Insert sample data
insert_sample_data() {
    log "Inserting sample data..."
    
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << 'EOF'
-- Sample data for local development

-- Sample organizations
INSERT INTO ws1_core.organizations (id, name, slug, description) VALUES
    ('550e8400-e29b-41d4-a716-446655440001', 'Nexus Corporation', 'nexus-corp', 'Main organization for Nexus Architect platform'),
    ('550e8400-e29b-41d4-a716-446655440002', 'Demo Company', 'demo-company', 'Demo organization for testing purposes')
ON CONFLICT (slug) DO NOTHING;

-- Sample users
INSERT INTO ws1_core.users (id, email, username, password_hash, first_name, last_name, role, is_verified) VALUES
    ('550e8400-e29b-41d4-a716-446655440010', 'admin@nexus.dev', 'admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.PmvlDO', 'Admin', 'User', 'admin', true),
    ('550e8400-e29b-41d4-a716-446655440011', 'developer@nexus.dev', 'developer', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.PmvlDO', 'Developer', 'User', 'developer', true),
    ('550e8400-e29b-41d4-a716-446655440012', 'manager@nexus.dev', 'manager', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.PmvlDO', 'Project', 'Manager', 'manager', true),
    ('550e8400-e29b-41d4-a716-446655440013', 'executive@nexus.dev', 'executive', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj/RK.PmvlDO', 'Executive', 'User', 'executive', true)
ON CONFLICT (email) DO NOTHING;

-- Sample user-organization relationships
INSERT INTO ws1_core.user_organizations (user_id, organization_id, role) VALUES
    ('550e8400-e29b-41d4-a716-446655440010', '550e8400-e29b-41d4-a716-446655440001', 'admin'),
    ('550e8400-e29b-41d4-a716-446655440011', '550e8400-e29b-41d4-a716-446655440001', 'developer'),
    ('550e8400-e29b-41d4-a716-446655440012', '550e8400-e29b-41d4-a716-446655440001', 'manager'),
    ('550e8400-e29b-41d4-a716-446655440013', '550e8400-e29b-41d4-a716-446655440001', 'executive')
ON CONFLICT (user_id, organization_id) DO NOTHING;

-- Sample AI models
INSERT INTO ws2_ai.models (id, name, type, version, description, config) VALUES
    ('550e8400-e29b-41d4-a716-446655440020', 'Sentiment Analysis', 'nlp', '1.0.0', 'Analyzes sentiment in text data', '{"accuracy": 0.92, "language": "en"}'),
    ('550e8400-e29b-41d4-a716-446655440021', 'Demand Forecasting', 'regression', '2.1.0', 'Predicts future demand based on historical data', '{"horizon": 30, "confidence": 0.85}'),
    ('550e8400-e29b-41d4-a716-446655440022', 'Anomaly Detection', 'classification', '1.5.0', 'Detects anomalies in system metrics', '{"threshold": 0.95, "window": 300}')
ON CONFLICT (id) DO NOTHING;

-- Sample data sources
INSERT INTO ws3_data.sources (id, name, type, connection_config) VALUES
    ('550e8400-e29b-41d4-a716-446655440030', 'Customer Database', 'postgresql', '{"host": "customer-db.example.com", "port": 5432, "database": "customers"}'),
    ('550e8400-e29b-41d4-a716-446655440031', 'Sales API', 'rest_api', '{"base_url": "https://api.sales.example.com", "auth_type": "bearer"}'),
    ('550e8400-e29b-41d4-a716-446655440032', 'IoT Sensors', 'mqtt', '{"broker": "mqtt.iot.example.com", "port": 1883, "topics": ["sensors/+/data"]}')
ON CONFLICT (id) DO NOTHING;

-- Sample autonomous agents
INSERT INTO ws4_autonomous.agents (id, name, type, config) VALUES
    ('550e8400-e29b-41d4-a716-446655440040', 'System Monitor', 'monitoring', '{"check_interval": 60, "thresholds": {"cpu": 80, "memory": 85}}'),
    ('550e8400-e29b-41d4-a716-446655440041', 'Auto Scaler', 'scaling', '{"min_instances": 2, "max_instances": 10, "target_cpu": 70}'),
    ('550e8400-e29b-41d4-a716-446655440042', 'Backup Manager', 'backup', '{"schedule": "0 2 * * *", "retention_days": 30}')
ON CONFLICT (id) DO NOTHING;

-- Sample dashboards
INSERT INTO ws5_interfaces.dashboards (id, user_id, name, type, config) VALUES
    ('550e8400-e29b-41d4-a716-446655440050', '550e8400-e29b-41d4-a716-446655440013', 'Executive Overview', 'executive', '{"refresh_interval": 300, "theme": "dark"}'),
    ('550e8400-e29b-41d4-a716-446655440051', '550e8400-e29b-41d4-a716-446655440011', 'Developer Console', 'developer', '{"refresh_interval": 30, "theme": "light"}'),
    ('550e8400-e29b-41d4-a716-446655440052', '550e8400-e29b-41d4-a716-446655440012', 'Project Dashboard', 'manager', '{"refresh_interval": 120, "theme": "auto"}')
ON CONFLICT (id) DO NOTHING;

-- Sample system metrics
INSERT INTO shared.system_metrics (metric_name, metric_value, tags) VALUES
    ('cpu_usage_percent', 45.2, '{"host": "web-01", "environment": "development"}'),
    ('memory_usage_percent', 62.8, '{"host": "web-01", "environment": "development"}'),
    ('disk_usage_percent', 34.1, '{"host": "web-01", "environment": "development", "mount": "/"}'),
    ('response_time_ms', 125.5, '{"endpoint": "/api/health", "method": "GET"}'),
    ('request_count', 1250, '{"endpoint": "/api/users", "method": "GET", "status": "200"}')
ON CONFLICT DO NOTHING;

EOF

    success "Sample data inserted ✓"
}

# Setup Redis data
setup_redis_data() {
    log "Setting up Redis data..."
    
    # Sample cache data
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD << 'EOF'
SET "session:550e8400-e29b-41d4-a716-446655440010" "{\"user_id\":\"550e8400-e29b-41d4-a716-446655440010\",\"role\":\"admin\",\"expires\":1735689600}"
SET "cache:user:550e8400-e29b-41d4-a716-446655440010" "{\"id\":\"550e8400-e29b-41d4-a716-446655440010\",\"email\":\"admin@nexus.dev\",\"username\":\"admin\",\"role\":\"admin\"}"
SET "config:app:theme" "dark"
SET "config:app:refresh_interval" "30"
HSET "metrics:realtime" "active_users" "42" "cpu_usage" "45.2" "memory_usage" "62.8"
LPUSH "notifications:550e8400-e29b-41d4-a716-446655440010" "{\"id\":\"1\",\"message\":\"Welcome to Nexus Architect!\",\"type\":\"info\",\"timestamp\":\"2024-01-01T00:00:00Z\"}"
SADD "active_sessions" "550e8400-e29b-41d4-a716-446655440010"
ZADD "leaderboard:performance" 95.5 "ws1-core-api" 92.3 "ws2-ai-api" 88.7 "ws3-data-api"
EOF

    success "Redis data setup completed ✓"
}

# Create database backup
create_backup() {
    log "Creating initial database backup..."
    
    mkdir -p ~/nexus-dev/backups
    
    # PostgreSQL backup
    PGPASSWORD=$DB_PASSWORD pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME > ~/nexus-dev/backups/nexus_dev_initial.sql
    
    # Redis backup
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD BGSAVE
    
    success "Initial backup created ✓"
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #3: PostgreSQL/Redis initialization with sample data"
    
    wait_for_postgres
    wait_for_redis
    create_schemas
    insert_sample_data
    setup_redis_data
    create_backup
    
    success "🎉 Database setup completed successfully!"
    success "📊 PostgreSQL: nexus_dev database with sample data"
    success "🔄 Redis: Cache and session data configured"
    success "💾 Backup: Initial backup created in ~/nexus-dev/backups"
    
    log "📋 Sample credentials:"
    log "   Admin: admin@nexus.dev / password"
    log "   Developer: developer@nexus.dev / password"
    log "   Manager: manager@nexus.dev / password"
    log "   Executive: executive@nexus.dev / password"
}

# Run main function
main "$@"

