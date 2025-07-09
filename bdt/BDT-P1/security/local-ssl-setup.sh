#!/bin/bash

# Nexus Architect - Local SSL Setup
# BDT-P1 Deliverable #6: Local SSL certificates and HTTPS configuration
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

# SSL Configuration
SSL_DIR="$HOME/nexus-dev/certs/ssl"
CA_DIR="$HOME/nexus-dev/certs/ca"
DOMAIN="localhost"
COUNTRY="US"
STATE="CA"
CITY="San Francisco"
ORGANIZATION="Nexus Architect"
ORG_UNIT="Development"

# Create SSL directory structure
create_ssl_directories() {
    log "Creating SSL directory structure..."
    
    mkdir -p "$SSL_DIR"
    mkdir -p "$CA_DIR"
    mkdir -p "$HOME/nexus-dev/certs/backup"
    
    # Set secure permissions
    chmod 700 "$HOME/nexus-dev/certs"
    chmod 700 "$SSL_DIR"
    chmod 700 "$CA_DIR"
    
    success "SSL directories created ✓"
}

# Generate Certificate Authority (CA)
generate_ca() {
    log "Generating Certificate Authority (CA)..."
    
    # Generate CA private key
    openssl genrsa -out "$CA_DIR/ca-key.pem" 4096
    chmod 400 "$CA_DIR/ca-key.pem"
    
    # Generate CA certificate
    openssl req -new -x509 -days 3650 -key "$CA_DIR/ca-key.pem" -sha256 -out "$CA_DIR/ca.pem" -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORGANIZATION/OU=$ORG_UNIT/CN=Nexus Architect CA"
    chmod 444 "$CA_DIR/ca.pem"
    
    success "Certificate Authority generated ✓"
}

# Generate server certificates
generate_server_certificates() {
    log "Generating server certificates..."
    
    # Generate server private key
    openssl genrsa -out "$SSL_DIR/server-key.pem" 4096
    chmod 400 "$SSL_DIR/server-key.pem"
    
    # Generate server certificate signing request
    openssl req -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORGANIZATION/OU=$ORG_UNIT/CN=$DOMAIN" -sha256 -new -key "$SSL_DIR/server-key.pem" -out "$SSL_DIR/server.csr"
    
    # Create extensions file for server certificate
    cat > "$SSL_DIR/server-extfile.cnf" << EOF
subjectAltName = DNS:localhost,DNS:*.localhost,DNS:nexus.local,DNS:*.nexus.local,IP:127.0.0.1,IP:0.0.0.0,IP:172.20.0.1
extendedKeyUsage = serverAuth
keyUsage = keyEncipherment,dataEncipherment
EOF
    
    # Generate server certificate
    openssl x509 -req -days 365 -in "$SSL_DIR/server.csr" -CA "$CA_DIR/ca.pem" -CAkey "$CA_DIR/ca-key.pem" -out "$SSL_DIR/server-cert.pem" -extfile "$SSL_DIR/server-extfile.cnf" -CAcreateserial
    chmod 444 "$SSL_DIR/server-cert.pem"
    
    # Clean up
    rm "$SSL_DIR/server.csr" "$SSL_DIR/server-extfile.cnf"
    
    success "Server certificates generated ✓"
}

# Generate client certificates
generate_client_certificates() {
    log "Generating client certificates..."
    
    # Generate client private key
    openssl genrsa -out "$SSL_DIR/client-key.pem" 4096
    chmod 400 "$SSL_DIR/client-key.pem"
    
    # Generate client certificate signing request
    openssl req -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORGANIZATION/OU=$ORG_UNIT/CN=Nexus Client" -new -key "$SSL_DIR/client-key.pem" -out "$SSL_DIR/client.csr"
    
    # Create extensions file for client certificate
    cat > "$SSL_DIR/client-extfile.cnf" << EOF
extendedKeyUsage = clientAuth
keyUsage = keyEncipherment,dataEncipherment
EOF
    
    # Generate client certificate
    openssl x509 -req -days 365 -in "$SSL_DIR/client.csr" -CA "$CA_DIR/ca.pem" -CAkey "$CA_DIR/ca-key.pem" -out "$SSL_DIR/client-cert.pem" -extfile "$SSL_DIR/client-extfile.cnf" -CAcreateserial
    chmod 444 "$SSL_DIR/client-cert.pem"
    
    # Clean up
    rm "$SSL_DIR/client.csr" "$SSL_DIR/client-extfile.cnf"
    
    success "Client certificates generated ✓"
}

# Create certificate bundles
create_certificate_bundles() {
    log "Creating certificate bundles..."
    
    # Create full chain certificate
    cat "$SSL_DIR/server-cert.pem" "$CA_DIR/ca.pem" > "$SSL_DIR/server-fullchain.pem"
    chmod 444 "$SSL_DIR/server-fullchain.pem"
    
    # Create PKCS#12 bundle for browsers
    openssl pkcs12 -export -out "$SSL_DIR/server.p12" -inkey "$SSL_DIR/server-key.pem" -in "$SSL_DIR/server-cert.pem" -certfile "$CA_DIR/ca.pem" -passout pass:nexus-dev
    chmod 400 "$SSL_DIR/server.p12"
    
    # Create Java keystore
    if command -v keytool &> /dev/null; then
        keytool -importkeystore -srckeystore "$SSL_DIR/server.p12" -srcstoretype PKCS12 -destkeystore "$SSL_DIR/server.jks" -deststoretype JKS -srcstorepass nexus-dev -deststorepass nexus-dev -noprompt
        chmod 400 "$SSL_DIR/server.jks"
        success "Java keystore created ✓"
    fi
    
    success "Certificate bundles created ✓"
}

# Install CA certificate in system trust store
install_ca_certificate() {
    log "Installing CA certificate in system trust store..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v update-ca-certificates &> /dev/null; then
            sudo cp "$CA_DIR/ca.pem" /usr/local/share/ca-certificates/nexus-ca.crt
            sudo update-ca-certificates
            success "CA certificate installed in Linux trust store ✓"
        else
            warning "update-ca-certificates not found. Manual CA installation required."
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v security &> /dev/null; then
            sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "$CA_DIR/ca.pem"
            success "CA certificate installed in macOS trust store ✓"
        else
            warning "security command not found. Manual CA installation required."
        fi
    else
        warning "Unsupported OS for automatic CA installation. Manual installation required."
    fi
}

# Create NGINX SSL configuration
create_nginx_ssl_config() {
    log "Creating NGINX SSL configuration..."
    
    mkdir -p bdt/BDT-P1/docker/nginx/conf.d
    
    cat > bdt/BDT-P1/docker/nginx/conf.d/ssl.conf << EOF
# Nexus Architect NGINX SSL Configuration
# Generated on $(date)

# SSL Configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384:ECDHE-RSA-AES128-SHA:ECDHE-RSA-AES256-SHA:DHE-RSA-AES128-SHA256:DHE-RSA-AES256-SHA256:DHE-RSA-AES128-SHA:DHE-RSA-AES256-SHA:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA;
ssl_prefer_server_ciphers on;
ssl_dhparam /etc/nginx/ssl/dhparam.pem;

# SSL Session Configuration
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_session_tickets off;

# OCSP Stapling
ssl_stapling on;
ssl_stapling_verify on;

# Security Headers
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
add_header X-Content-Type-Options nosniff always;
add_header X-Frame-Options DENY always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' wss: https:;" always;

# Certificate Configuration
ssl_certificate /etc/nginx/ssl/server-cert.pem;
ssl_certificate_key /etc/nginx/ssl/server-key.pem;
ssl_trusted_certificate /etc/nginx/ssl/ca.pem;
EOF

    success "NGINX SSL configuration created ✓"
}

# Generate Diffie-Hellman parameters
generate_dhparam() {
    log "Generating Diffie-Hellman parameters..."
    
    if [[ ! -f "$SSL_DIR/dhparam.pem" ]]; then
        openssl dhparam -out "$SSL_DIR/dhparam.pem" 2048
        chmod 444 "$SSL_DIR/dhparam.pem"
        success "Diffie-Hellman parameters generated ✓"
    else
        success "Diffie-Hellman parameters already exist ✓"
    fi
}

# Create SSL verification script
create_ssl_verification_script() {
    log "Creating SSL verification script..."
    
    cat > "$HOME/nexus-dev/verify-ssl.sh" << 'EOF'
#!/bin/bash

# SSL Certificate Verification Script

echo "🔒 Nexus Architect SSL Certificate Verification"
echo "=============================================="

SSL_DIR="$HOME/nexus-dev/certs/ssl"
CA_DIR="$HOME/nexus-dev/certs/ca"

# Check certificate validity
echo "📋 Certificate Information:"
openssl x509 -in "$SSL_DIR/server-cert.pem" -text -noout | grep -E "(Subject:|Issuer:|Not Before:|Not After:|DNS:|IP Address:)"

echo ""
echo "🔍 Certificate Verification:"

# Verify certificate against CA
if openssl verify -CAfile "$CA_DIR/ca.pem" "$SSL_DIR/server-cert.pem"; then
    echo "✅ Certificate verification: PASSED"
else
    echo "❌ Certificate verification: FAILED"
fi

# Check certificate expiration
EXPIRY_DATE=$(openssl x509 -in "$SSL_DIR/server-cert.pem" -noout -enddate | cut -d= -f2)
EXPIRY_EPOCH=$(date -d "$EXPIRY_DATE" +%s)
CURRENT_EPOCH=$(date +%s)
DAYS_UNTIL_EXPIRY=$(( (EXPIRY_EPOCH - CURRENT_EPOCH) / 86400 ))

if [[ $DAYS_UNTIL_EXPIRY -gt 30 ]]; then
    echo "✅ Certificate expiry: $DAYS_UNTIL_EXPIRY days remaining"
elif [[ $DAYS_UNTIL_EXPIRY -gt 0 ]]; then
    echo "⚠️  Certificate expiry: $DAYS_UNTIL_EXPIRY days remaining (renewal recommended)"
else
    echo "❌ Certificate expiry: EXPIRED"
fi

# Test SSL connection
echo ""
echo "🌐 SSL Connection Test:"
if curl -k --cert "$SSL_DIR/client-cert.pem" --key "$SSL_DIR/client-key.pem" --cacert "$CA_DIR/ca.pem" https://localhost:443 &>/dev/null; then
    echo "✅ SSL connection: SUCCESS"
else
    echo "⚠️  SSL connection: Could not connect (services may not be running)"
fi

echo ""
echo "📁 Certificate Files:"
echo "   CA Certificate: $CA_DIR/ca.pem"
echo "   Server Certificate: $SSL_DIR/server-cert.pem"
echo "   Server Private Key: $SSL_DIR/server-key.pem"
echo "   Client Certificate: $SSL_DIR/client-cert.pem"
echo "   Full Chain: $SSL_DIR/server-fullchain.pem"
echo "   PKCS#12 Bundle: $SSL_DIR/server.p12"
EOF

    chmod +x "$HOME/nexus-dev/verify-ssl.sh"
    success "SSL verification script created ✓"
}

# Create certificate renewal script
create_renewal_script() {
    log "Creating certificate renewal script..."
    
    cat > "$HOME/nexus-dev/renew-ssl.sh" << 'EOF'
#!/bin/bash

# SSL Certificate Renewal Script

echo "🔄 Nexus Architect SSL Certificate Renewal"
echo "=========================================="

SSL_DIR="$HOME/nexus-dev/certs/ssl"
CA_DIR="$HOME/nexus-dev/certs/ca"
BACKUP_DIR="$HOME/nexus-dev/certs/backup/$(date +%Y%m%d_%H%M%S)"

# Create backup
echo "📦 Creating backup..."
mkdir -p "$BACKUP_DIR"
cp -r "$SSL_DIR"/* "$BACKUP_DIR/"
cp -r "$CA_DIR"/* "$BACKUP_DIR/"
echo "✅ Backup created: $BACKUP_DIR"

# Regenerate certificates
echo "🔑 Regenerating certificates..."
cd /home/ubuntu/nexus-architect
./bdt/BDT-P1/security/local-ssl-setup.sh

echo "✅ SSL certificates renewed successfully!"
echo "📋 Remember to restart services to use new certificates"
EOF

    chmod +x "$HOME/nexus-dev/renew-ssl.sh"
    success "Certificate renewal script created ✓"
}

# Validate certificates
validate_certificates() {
    log "Validating generated certificates..."
    
    # Check if all required files exist
    local required_files=(
        "$CA_DIR/ca.pem"
        "$CA_DIR/ca-key.pem"
        "$SSL_DIR/server-cert.pem"
        "$SSL_DIR/server-key.pem"
        "$SSL_DIR/client-cert.pem"
        "$SSL_DIR/client-key.pem"
        "$SSL_DIR/server-fullchain.pem"
        "$SSL_DIR/dhparam.pem"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            success "Certificate file exists: $(basename "$file") ✓"
        else
            error "Certificate file missing: $file"
        fi
    done
    
    # Verify certificate chain
    if openssl verify -CAfile "$CA_DIR/ca.pem" "$SSL_DIR/server-cert.pem" &>/dev/null; then
        success "Certificate chain validation passed ✓"
    else
        error "Certificate chain validation failed"
    fi
    
    # Check certificate expiration
    local expiry_date=$(openssl x509 -in "$SSL_DIR/server-cert.pem" -noout -enddate | cut -d= -f2)
    success "Certificate expires: $expiry_date ✓"
    
    # Verify private key matches certificate
    local cert_modulus=$(openssl x509 -noout -modulus -in "$SSL_DIR/server-cert.pem" | openssl md5)
    local key_modulus=$(openssl rsa -noout -modulus -in "$SSL_DIR/server-key.pem" | openssl md5)
    
    if [[ "$cert_modulus" == "$key_modulus" ]]; then
        success "Private key matches certificate ✓"
    else
        error "Private key does not match certificate"
    fi
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #6: Local SSL certificates and HTTPS configuration"
    
    create_ssl_directories
    generate_ca
    generate_server_certificates
    generate_client_certificates
    create_certificate_bundles
    generate_dhparam
    create_nginx_ssl_config
    create_ssl_verification_script
    create_renewal_script
    validate_certificates
    
    # Optional: Install CA certificate
    read -p "Install CA certificate in system trust store? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_ca_certificate
    fi
    
    success "🎉 SSL setup completed successfully!"
    success "🔒 CA Certificate: $CA_DIR/ca.pem"
    success "🌐 Server Certificate: $SSL_DIR/server-cert.pem"
    success "🔑 Server Private Key: $SSL_DIR/server-key.pem"
    success "📋 Full Chain: $SSL_DIR/server-fullchain.pem"
    
    log "📋 Next steps:"
    log "   1. Verify certificates: ~/nexus-dev/verify-ssl.sh"
    log "   2. Start Docker services with SSL enabled"
    log "   3. Access services via HTTPS://localhost"
    log "   4. Import CA certificate in browsers if needed"
    
    warning "⚠️  These certificates are for development only. Use proper certificates in production."
}

# Run main function
main "$@"

