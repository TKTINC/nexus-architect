#!/bin/bash

# Nexus Architect - Backup and Restore Procedures
# BDT-P1 Deliverable #13: Backup and restore procedures
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

# Backup configuration
BACKUP_BASE_DIR="$HOME/nexus-dev/backups"
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_BASE_DIR/$BACKUP_DATE"
RETENTION_DAYS=30

# Database configuration
PG_HOST="localhost"
PG_PORT="5432"
PG_USER="postgres"
PG_PASSWORD="postgres"
PG_DATABASES=("nexus_dev" "nexus_test")

REDIS_HOST="localhost"
REDIS_PORT="6379"

# Application directories
APP_DIRS=(
    "$HOME/nexus-dev/implementation"
    "$HOME/nexus-dev/config"
    "$HOME/nexus-dev/logs"
    "$HOME/nexus-dev/uploads"
)

# Initialize backup environment
init_backup_environment() {
    log "Initializing backup environment..."
    
    # Create backup directory structure
    mkdir -p "$BACKUP_BASE_DIR"/{daily,weekly,monthly,restore}
    mkdir -p "$BACKUP_DIR"/{databases,applications,configs,logs}
    
    # Install backup tools if needed
    install_backup_tools
    
    success "Backup environment initialized ✓"
}

# Install backup tools
install_backup_tools() {
    log "Installing backup tools..."
    
    # Install PostgreSQL client tools if not available
    if ! command -v pg_dump &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update && sudo apt-get install -y postgresql-client
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install postgresql
            fi
        fi
    fi
    
    # Install Redis tools if not available
    if ! command -v redis-cli &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get install -y redis-tools
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install redis
            fi
        fi
    fi
    
    # Install compression tools
    if ! command -v pigz &> /dev/null; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get install -y pigz
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install pigz
            fi
        fi
    fi
    
    success "Backup tools installed ✓"
}

# Backup PostgreSQL databases
backup_postgresql() {
    log "Backing up PostgreSQL databases..."
    
    local pg_backup_dir="$BACKUP_DIR/databases/postgresql"
    mkdir -p "$pg_backup_dir"
    
    # Set password for pg_dump
    export PGPASSWORD="$PG_PASSWORD"
    
    for database in "${PG_DATABASES[@]}"; do
        log "Backing up PostgreSQL database: $database"
        
        # Check if database exists
        if psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -lqt | cut -d \| -f 1 | grep -qw "$database"; then
            # Create database dump
            local dump_file="$pg_backup_dir/${database}_${BACKUP_DATE}.sql"
            
            if pg_dump -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$database" \
                --verbose --clean --if-exists --create --format=plain > "$dump_file" 2>/dev/null; then
                
                # Compress the dump
                if command -v pigz &> /dev/null; then
                    pigz "$dump_file"
                    dump_file="${dump_file}.gz"
                else
                    gzip "$dump_file"
                    dump_file="${dump_file}.gz"
                fi
                
                local dump_size=$(du -h "$dump_file" | cut -f1)
                success "PostgreSQL database $database backed up successfully ($dump_size) ✓"
                
                # Create metadata file
                cat > "$pg_backup_dir/${database}_${BACKUP_DATE}.meta" << EOF
{
    "database": "$database",
    "backup_date": "$BACKUP_DATE",
    "backup_type": "postgresql",
    "file_size": "$dump_size",
    "compression": "gzip",
    "pg_version": "$(psql -h $PG_HOST -p $PG_PORT -U $PG_USER -d $database -t -c 'SELECT version();' | head -1 | xargs)"
}
EOF
            else
                error "Failed to backup PostgreSQL database: $database"
            fi
        else
            warning "PostgreSQL database $database does not exist, skipping"
        fi
    done
    
    # Backup PostgreSQL configuration
    if [[ -f "/etc/postgresql/*/main/postgresql.conf" ]]; then
        cp /etc/postgresql/*/main/postgresql.conf "$pg_backup_dir/postgresql.conf" 2>/dev/null || true
    fi
    
    unset PGPASSWORD
    success "PostgreSQL backup completed ✓"
}

# Backup Redis data
backup_redis() {
    log "Backing up Redis data..."
    
    local redis_backup_dir="$BACKUP_DIR/databases/redis"
    mkdir -p "$redis_backup_dir"
    
    # Check if Redis is accessible
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
        log "Creating Redis backup..."
        
        # Trigger Redis save
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE > /dev/null
        
        # Wait for background save to complete
        while [[ "$(redis-cli -h $REDIS_HOST -p $REDIS_PORT LASTSAVE)" == "$(redis-cli -h $REDIS_HOST -p $REDIS_PORT LASTSAVE)" ]]; do
            sleep 1
        done
        
        # Copy Redis dump file
        local redis_dump_file="/var/lib/redis/dump.rdb"
        if [[ -f "$redis_dump_file" ]]; then
            cp "$redis_dump_file" "$redis_backup_dir/redis_${BACKUP_DATE}.rdb"
            
            # Compress the dump
            if command -v pigz &> /dev/null; then
                pigz "$redis_backup_dir/redis_${BACKUP_DATE}.rdb"
            else
                gzip "$redis_backup_dir/redis_${BACKUP_DATE}.rdb"
            fi
            
            local dump_size=$(du -h "$redis_backup_dir/redis_${BACKUP_DATE}.rdb.gz" | cut -f1)
            success "Redis data backed up successfully ($dump_size) ✓"
            
            # Create metadata file
            cat > "$redis_backup_dir/redis_${BACKUP_DATE}.meta" << EOF
{
    "backup_date": "$BACKUP_DATE",
    "backup_type": "redis",
    "file_size": "$dump_size",
    "compression": "gzip",
    "redis_version": "$(redis-cli -h $REDIS_HOST -p $REDIS_PORT INFO server | grep redis_version | cut -d: -f2 | tr -d '\r')"
}
EOF
        else
            # Alternative: use Redis SAVE command and export
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --rdb "$redis_backup_dir/redis_${BACKUP_DATE}.rdb"
            
            if [[ -f "$redis_backup_dir/redis_${BACKUP_DATE}.rdb" ]]; then
                # Compress the dump
                if command -v pigz &> /dev/null; then
                    pigz "$redis_backup_dir/redis_${BACKUP_DATE}.rdb"
                else
                    gzip "$redis_backup_dir/redis_${BACKUP_DATE}.rdb"
                fi
                
                local dump_size=$(du -h "$redis_backup_dir/redis_${BACKUP_DATE}.rdb.gz" | cut -f1)
                success "Redis data backed up successfully ($dump_size) ✓"
            else
                warning "Could not create Redis backup"
            fi
        fi
    else
        warning "Redis is not accessible, skipping backup"
    fi
    
    success "Redis backup completed ✓"
}

# Backup application files
backup_applications() {
    log "Backing up application files..."
    
    local app_backup_dir="$BACKUP_DIR/applications"
    mkdir -p "$app_backup_dir"
    
    for app_dir in "${APP_DIRS[@]}"; do
        if [[ -d "$app_dir" ]]; then
            local dir_name=$(basename "$app_dir")
            log "Backing up application directory: $app_dir"
            
            # Create tar archive with compression
            local archive_file="$app_backup_dir/${dir_name}_${BACKUP_DATE}.tar.gz"
            
            if tar -czf "$archive_file" -C "$(dirname "$app_dir")" "$(basename "$app_dir")" 2>/dev/null; then
                local archive_size=$(du -h "$archive_file" | cut -f1)
                success "Application directory $dir_name backed up successfully ($archive_size) ✓"
                
                # Create metadata file
                cat > "$app_backup_dir/${dir_name}_${BACKUP_DATE}.meta" << EOF
{
    "directory": "$app_dir",
    "backup_date": "$BACKUP_DATE",
    "backup_type": "application",
    "file_size": "$archive_size",
    "compression": "gzip",
    "file_count": "$(find "$app_dir" -type f | wc -l)"
}
EOF
            else
                warning "Failed to backup application directory: $app_dir"
            fi
        else
            warning "Application directory does not exist: $app_dir"
        fi
    done
    
    success "Application backup completed ✓"
}

# Backup configuration files
backup_configurations() {
    log "Backing up configuration files..."
    
    local config_backup_dir="$BACKUP_DIR/configs"
    mkdir -p "$config_backup_dir"
    
    # Backup Docker configurations
    if [[ -d "$HOME/nexus-dev/docker" ]]; then
        tar -czf "$config_backup_dir/docker_configs_${BACKUP_DATE}.tar.gz" -C "$HOME/nexus-dev" docker
        success "Docker configurations backed up ✓"
    fi
    
    # Backup environment files
    local env_files=(
        "$HOME/nexus-dev/.env"
        "$HOME/nexus-dev/.env.local"
        "$HOME/nexus-dev/.env.development"
    )
    
    for env_file in "${env_files[@]}"; do
        if [[ -f "$env_file" ]]; then
            cp "$env_file" "$config_backup_dir/"
            success "Environment file $(basename "$env_file") backed up ✓"
        fi
    done
    
    # Backup SSL certificates
    if [[ -d "$HOME/nexus-dev/ssl" ]]; then
        tar -czf "$config_backup_dir/ssl_certificates_${BACKUP_DATE}.tar.gz" -C "$HOME/nexus-dev" ssl
        success "SSL certificates backed up ✓"
    fi
    
    # Backup monitoring configurations
    if [[ -d "$HOME/nexus-dev/monitoring" ]]; then
        tar -czf "$config_backup_dir/monitoring_configs_${BACKUP_DATE}.tar.gz" -C "$HOME/nexus-dev" monitoring
        success "Monitoring configurations backed up ✓"
    fi
    
    success "Configuration backup completed ✓"
}

# Create backup manifest
create_backup_manifest() {
    log "Creating backup manifest..."
    
    local manifest_file="$BACKUP_DIR/backup_manifest.json"
    local total_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    
    cat > "$manifest_file" << EOF
{
    "backup_id": "$BACKUP_DATE",
    "backup_date": "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)",
    "backup_type": "full",
    "total_size": "$total_size",
    "components": {
        "databases": {
            "postgresql": $(find "$BACKUP_DIR/databases/postgresql" -name "*.gz" 2>/dev/null | wc -l),
            "redis": $(find "$BACKUP_DIR/databases/redis" -name "*.gz" 2>/dev/null | wc -l)
        },
        "applications": $(find "$BACKUP_DIR/applications" -name "*.tar.gz" 2>/dev/null | wc -l),
        "configurations": $(find "$BACKUP_DIR/configs" -name "*.tar.gz" 2>/dev/null | wc -l)
    },
    "files": [
$(find "$BACKUP_DIR" -type f -name "*.gz" -o -name "*.tar.gz" | sed 's/.*/"&"/' | paste -sd, -)
    ],
    "retention_policy": {
        "retention_days": $RETENTION_DAYS,
        "cleanup_date": "$(date -d "+$RETENTION_DAYS days" -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
    }
}
EOF

    success "Backup manifest created: $manifest_file ✓"
}

# Restore PostgreSQL database
restore_postgresql() {
    local backup_file="$1"
    local target_database="$2"
    
    if [[ -z "$backup_file" || -z "$target_database" ]]; then
        error "Usage: restore_postgresql <backup_file> <target_database>"
    fi
    
    log "Restoring PostgreSQL database: $target_database from $backup_file"
    
    # Set password for psql
    export PGPASSWORD="$PG_PASSWORD"
    
    # Check if backup file exists
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file does not exist: $backup_file"
    fi
    
    # Decompress if needed
    local sql_file="$backup_file"
    if [[ "$backup_file" == *.gz ]]; then
        sql_file="${backup_file%.gz}"
        gunzip -c "$backup_file" > "$sql_file"
    fi
    
    # Drop existing database if it exists
    psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d postgres -c "DROP DATABASE IF EXISTS $target_database;" 2>/dev/null || true
    
    # Restore database
    if psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d postgres < "$sql_file"; then
        success "PostgreSQL database $target_database restored successfully ✓"
        
        # Clean up temporary file if we decompressed
        if [[ "$backup_file" == *.gz ]]; then
            rm -f "$sql_file"
        fi
    else
        error "Failed to restore PostgreSQL database: $target_database"
    fi
    
    unset PGPASSWORD
}

# Restore Redis data
restore_redis() {
    local backup_file="$1"
    
    if [[ -z "$backup_file" ]]; then
        error "Usage: restore_redis <backup_file>"
    fi
    
    log "Restoring Redis data from: $backup_file"
    
    # Check if backup file exists
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file does not exist: $backup_file"
    fi
    
    # Stop Redis service temporarily
    if systemctl is-active --quiet redis-server; then
        sudo systemctl stop redis-server
        local redis_was_running=true
    fi
    
    # Decompress if needed
    local rdb_file="$backup_file"
    if [[ "$backup_file" == *.gz ]]; then
        rdb_file="${backup_file%.gz}"
        gunzip -c "$backup_file" > "$rdb_file"
    fi
    
    # Copy RDB file to Redis data directory
    sudo cp "$rdb_file" /var/lib/redis/dump.rdb
    sudo chown redis:redis /var/lib/redis/dump.rdb
    
    # Start Redis service
    if [[ "$redis_was_running" == true ]]; then
        sudo systemctl start redis-server
    fi
    
    # Wait for Redis to start
    sleep 5
    
    # Verify restore
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
        success "Redis data restored successfully ✓"
        
        # Clean up temporary file if we decompressed
        if [[ "$backup_file" == *.gz ]]; then
            rm -f "$rdb_file"
        fi
    else
        error "Failed to restore Redis data"
    fi
}

# Restore application files
restore_applications() {
    local backup_file="$1"
    local target_directory="$2"
    
    if [[ -z "$backup_file" || -z "$target_directory" ]]; then
        error "Usage: restore_applications <backup_file> <target_directory>"
    fi
    
    log "Restoring application files from: $backup_file to: $target_directory"
    
    # Check if backup file exists
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file does not exist: $backup_file"
    fi
    
    # Create target directory if it doesn't exist
    mkdir -p "$(dirname "$target_directory")"
    
    # Extract archive
    if tar -xzf "$backup_file" -C "$(dirname "$target_directory")"; then
        success "Application files restored successfully ✓"
    else
        error "Failed to restore application files"
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up old backups (retention: $RETENTION_DAYS days)..."
    
    # Find and remove backups older than retention period
    local deleted_count=0
    
    while IFS= read -r -d '' backup_dir; do
        local backup_date=$(basename "$backup_dir")
        local backup_timestamp=$(date -d "${backup_date:0:8} ${backup_date:9:2}:${backup_date:11:2}:${backup_date:13:2}" +%s 2>/dev/null || echo "0")
        local current_timestamp=$(date +%s)
        local age_days=$(( (current_timestamp - backup_timestamp) / 86400 ))
        
        if [[ $age_days -gt $RETENTION_DAYS ]]; then
            log "Removing old backup: $backup_dir (age: $age_days days)"
            rm -rf "$backup_dir"
            ((deleted_count++))
        fi
    done < <(find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "????????_??????" -print0)
    
    if [[ $deleted_count -gt 0 ]]; then
        success "Cleaned up $deleted_count old backups ✓"
    else
        info "No old backups to clean up"
    fi
}

# List available backups
list_backups() {
    log "Available backups:"
    
    if [[ ! -d "$BACKUP_BASE_DIR" ]]; then
        warning "No backup directory found"
        return
    fi
    
    echo ""
    printf "%-20s %-15s %-10s %s\n" "Backup ID" "Date" "Size" "Components"
    printf "%-20s %-15s %-10s %s\n" "--------" "----" "----" "----------"
    
    for backup_dir in "$BACKUP_BASE_DIR"/????????_??????; do
        if [[ -d "$backup_dir" ]]; then
            local backup_id=$(basename "$backup_dir")
            local backup_date="${backup_id:0:4}-${backup_id:4:2}-${backup_id:6:2} ${backup_id:9:2}:${backup_id:11:2}:${backup_id:13:2}"
            local backup_size=$(du -sh "$backup_dir" 2>/dev/null | cut -f1 || echo "Unknown")
            
            local components=""
            [[ -d "$backup_dir/databases" ]] && components+="DB "
            [[ -d "$backup_dir/applications" ]] && components+="APP "
            [[ -d "$backup_dir/configs" ]] && components+="CFG "
            
            printf "%-20s %-15s %-10s %s\n" "$backup_id" "$backup_date" "$backup_size" "$components"
        fi
    done
    echo ""
}

# Verify backup integrity
verify_backup() {
    local backup_id="$1"
    
    if [[ -z "$backup_id" ]]; then
        error "Usage: verify_backup <backup_id>"
    fi
    
    local backup_path="$BACKUP_BASE_DIR/$backup_id"
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup not found: $backup_id"
    fi
    
    log "Verifying backup integrity: $backup_id"
    
    local verification_errors=0
    
    # Verify manifest file
    if [[ -f "$backup_path/backup_manifest.json" ]]; then
        if jq empty "$backup_path/backup_manifest.json" 2>/dev/null; then
            success "Backup manifest is valid ✓"
        else
            error "Backup manifest is corrupted"
            ((verification_errors++))
        fi
    else
        warning "Backup manifest not found"
        ((verification_errors++))
    fi
    
    # Verify compressed files
    for compressed_file in $(find "$backup_path" -name "*.gz" -o -name "*.tar.gz"); do
        if gzip -t "$compressed_file" 2>/dev/null; then
            success "File integrity verified: $(basename "$compressed_file") ✓"
        else
            error "File corruption detected: $(basename "$compressed_file")"
            ((verification_errors++))
        fi
    done
    
    if [[ $verification_errors -eq 0 ]]; then
        success "Backup verification completed successfully ✓"
        return 0
    else
        error "Backup verification failed with $verification_errors errors"
        return 1
    fi
}

# Main backup function
perform_backup() {
    log "🎯 BDT-P1 Deliverable #13: Starting comprehensive backup..."
    
    init_backup_environment
    backup_postgresql
    backup_redis
    backup_applications
    backup_configurations
    create_backup_manifest
    cleanup_old_backups
    
    local total_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    
    success "🎉 Backup completed successfully!"
    success "📁 Backup location: $BACKUP_DIR"
    success "📊 Total size: $total_size"
    success "🗓️ Backup ID: $BACKUP_DATE"
    
    log "📋 Backup includes:"
    log "   🗄️ PostgreSQL databases"
    log "   🔴 Redis data"
    log "   📱 Application files"
    log "   ⚙️ Configuration files"
    log "   📄 SSL certificates"
    log "   📊 Monitoring configs"
    
    info "💡 To restore from this backup:"
    info "   ./backup-restore-local.sh restore $BACKUP_DATE"
    
    info "💡 To verify this backup:"
    info "   ./backup-restore-local.sh verify $BACKUP_DATE"
}

# Main restore function
perform_restore() {
    local backup_id="$1"
    
    if [[ -z "$backup_id" ]]; then
        error "Usage: perform_restore <backup_id>"
    fi
    
    local backup_path="$BACKUP_BASE_DIR/$backup_id"
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup not found: $backup_id"
    fi
    
    log "🔄 Starting restore from backup: $backup_id"
    
    # Verify backup before restore
    if ! verify_backup "$backup_id"; then
        error "Backup verification failed. Aborting restore."
    fi
    
    warning "⚠️ This will overwrite existing data. Continue? (y/N)"
    read -r confirmation
    if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
        info "Restore cancelled by user"
        return 0
    fi
    
    # Restore databases
    for pg_backup in "$backup_path/databases/postgresql"/*.sql.gz; do
        if [[ -f "$pg_backup" ]]; then
            local db_name=$(basename "$pg_backup" | sed 's/_[0-9]*_[0-9]*\.sql\.gz$//')
            restore_postgresql "$pg_backup" "$db_name"
        fi
    done
    
    for redis_backup in "$backup_path/databases/redis"/*.rdb.gz; do
        if [[ -f "$redis_backup" ]]; then
            restore_redis "$redis_backup"
            break  # Only restore the first Redis backup found
        fi
    done
    
    # Restore applications
    for app_backup in "$backup_path/applications"/*.tar.gz; do
        if [[ -f "$app_backup" ]]; then
            local app_name=$(basename "$app_backup" | sed 's/_[0-9]*_[0-9]*\.tar\.gz$//')
            local target_dir="$HOME/nexus-dev/$app_name"
            restore_applications "$app_backup" "$target_dir"
        fi
    done
    
    success "🎉 Restore completed successfully!"
    warning "⚠️ Please restart all services to ensure proper operation"
}

# Main execution
main() {
    case "${1:-backup}" in
        "backup")
            perform_backup
            ;;
        "restore")
            perform_restore "$2"
            ;;
        "list")
            list_backups
            ;;
        "verify")
            verify_backup "$2"
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "help"|"-h"|"--help")
            echo "Nexus Architect Backup and Restore Tool"
            echo ""
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  backup              Create a full backup (default)"
            echo "  restore <backup_id> Restore from a specific backup"
            echo "  list                List available backups"
            echo "  verify <backup_id>  Verify backup integrity"
            echo "  cleanup             Remove old backups based on retention policy"
            echo "  help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 backup                    # Create a new backup"
            echo "  $0 restore 20241201_143022   # Restore from specific backup"
            echo "  $0 list                      # Show all available backups"
            echo "  $0 verify 20241201_143022    # Verify backup integrity"
            echo "  $0 cleanup                   # Clean old backups"
            ;;
        *)
            error "Unknown command: $1. Use '$0 help' for usage information."
            ;;
    esac
}

# Execute main function with all arguments
main "$@"

