#!/bin/bash

# Nexus Architect - Local Authentication Setup
# BDT-P1 Deliverable #7: Local SSO simulation and authentication testing
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

# Authentication configuration
AUTH_DIR="$HOME/nexus-dev/auth"
KEYCLOAK_VERSION="22.0.5"
LDAP_VERSION="1.5.0"

# Create authentication directory structure
create_auth_directories() {
    log "Creating authentication directory structure..."
    
    mkdir -p "$AUTH_DIR"/{keycloak,ldap,saml,oauth,config,scripts,data}
    mkdir -p "$AUTH_DIR/keycloak"/{themes,providers,import}
    mkdir -p "$AUTH_DIR/ldap"/{data,config}
    mkdir -p "$AUTH_DIR/saml"/{metadata,certificates}
    mkdir -p "$AUTH_DIR/oauth"/{keys,config}
    
    # Set permissions
    chmod 755 "$AUTH_DIR"
    chmod 700 "$AUTH_DIR/oauth/keys"
    chmod 700 "$AUTH_DIR/saml/certificates"
    
    success "Authentication directories created ✓"
}

# Setup Keycloak for local SSO simulation
setup_keycloak() {
    log "Setting up Keycloak for local SSO simulation..."
    
    # Create Keycloak Docker Compose configuration
    cat > "$AUTH_DIR/keycloak/docker-compose.keycloak.yml" << EOF
version: '3.8'

services:
  keycloak-db:
    image: postgres:15-alpine
    container_name: nexus-keycloak-db
    restart: unless-stopped
    environment:
      POSTGRES_DB: keycloak
      POSTGRES_USER: keycloak
      POSTGRES_PASSWORD: keycloak_password
    volumes:
      - keycloak_db_data:/var/lib/postgresql/data
    networks:
      - nexus-auth-network

  keycloak:
    image: quay.io/keycloak/keycloak:$KEYCLOAK_VERSION
    container_name: nexus-keycloak
    restart: unless-stopped
    environment:
      KC_DB: postgres
      KC_DB_URL: jdbc:postgresql://keycloak-db:5432/keycloak
      KC_DB_USERNAME: keycloak
      KC_DB_PASSWORD: keycloak_password
      KC_HOSTNAME: localhost
      KC_HOSTNAME_PORT: 8080
      KC_HOSTNAME_STRICT: false
      KC_HOSTNAME_STRICT_HTTPS: false
      KC_HTTP_ENABLED: true
      KC_HEALTH_ENABLED: true
      KC_METRICS_ENABLED: true
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: nexus_admin_password
    ports:
      - "8080:8080"
    volumes:
      - ./themes:/opt/keycloak/themes
      - ./providers:/opt/keycloak/providers
      - ./import:/opt/keycloak/data/import
    depends_on:
      - keycloak-db
    networks:
      - nexus-auth-network
    command: start-dev --import-realm
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8080/health/ready || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5

networks:
  nexus-auth-network:
    driver: bridge

volumes:
  keycloak_db_data:
    driver: local
EOF

    # Create Keycloak realm configuration
    cat > "$AUTH_DIR/keycloak/import/nexus-realm.json" << 'EOF'
{
  "realm": "nexus",
  "displayName": "Nexus Architect",
  "enabled": true,
  "sslRequired": "external",
  "registrationAllowed": true,
  "registrationEmailAsUsername": true,
  "rememberMe": true,
  "verifyEmail": false,
  "loginWithEmailAllowed": true,
  "duplicateEmailsAllowed": false,
  "resetPasswordAllowed": true,
  "editUsernameAllowed": false,
  "bruteForceProtected": true,
  "permanentLockout": false,
  "maxFailureWaitSeconds": 900,
  "minimumQuickLoginWaitSeconds": 60,
  "waitIncrementSeconds": 60,
  "quickLoginCheckMilliSeconds": 1000,
  "maxDeltaTimeSeconds": 43200,
  "failureFactor": 30,
  "roles": {
    "realm": [
      {
        "name": "admin",
        "description": "Administrator role"
      },
      {
        "name": "developer",
        "description": "Developer role"
      },
      {
        "name": "manager",
        "description": "Project manager role"
      },
      {
        "name": "executive",
        "description": "Executive role"
      },
      {
        "name": "user",
        "description": "Standard user role"
      }
    ]
  },
  "users": [
    {
      "username": "admin",
      "email": "admin@nexus.dev",
      "firstName": "Admin",
      "lastName": "User",
      "enabled": true,
      "emailVerified": true,
      "credentials": [
        {
          "type": "password",
          "value": "password",
          "temporary": false
        }
      ],
      "realmRoles": ["admin", "user"]
    },
    {
      "username": "developer",
      "email": "developer@nexus.dev",
      "firstName": "Developer",
      "lastName": "User",
      "enabled": true,
      "emailVerified": true,
      "credentials": [
        {
          "type": "password",
          "value": "password",
          "temporary": false
        }
      ],
      "realmRoles": ["developer", "user"]
    },
    {
      "username": "manager",
      "email": "manager@nexus.dev",
      "firstName": "Project",
      "lastName": "Manager",
      "enabled": true,
      "emailVerified": true,
      "credentials": [
        {
          "type": "password",
          "value": "password",
          "temporary": false
        }
      ],
      "realmRoles": ["manager", "user"]
    },
    {
      "username": "executive",
      "email": "executive@nexus.dev",
      "firstName": "Executive",
      "lastName": "User",
      "enabled": true,
      "emailVerified": true,
      "credentials": [
        {
          "type": "password",
          "value": "password",
          "temporary": false
        }
      ],
      "realmRoles": ["executive", "user"]
    }
  ],
  "clients": [
    {
      "clientId": "nexus-frontend",
      "name": "Nexus Frontend Application",
      "enabled": true,
      "clientAuthenticatorType": "client-secret",
      "secret": "nexus-frontend-secret",
      "redirectUris": [
        "http://localhost:3000/*",
        "https://localhost:3000/*",
        "http://localhost:3001/*",
        "https://localhost:3001/*"
      ],
      "webOrigins": [
        "http://localhost:3000",
        "https://localhost:3000",
        "http://localhost:3001",
        "https://localhost:3001"
      ],
      "protocol": "openid-connect",
      "publicClient": false,
      "standardFlowEnabled": true,
      "implicitFlowEnabled": false,
      "directAccessGrantsEnabled": true,
      "serviceAccountsEnabled": true,
      "fullScopeAllowed": true
    },
    {
      "clientId": "nexus-api",
      "name": "Nexus API Services",
      "enabled": true,
      "clientAuthenticatorType": "client-secret",
      "secret": "nexus-api-secret",
      "protocol": "openid-connect",
      "publicClient": false,
      "bearerOnly": true,
      "serviceAccountsEnabled": true,
      "fullScopeAllowed": true
    }
  ]
}
EOF

    success "Keycloak configuration created ✓"
}

# Setup OpenLDAP for directory simulation
setup_openldap() {
    log "Setting up OpenLDAP for directory simulation..."
    
    # Create OpenLDAP Docker Compose configuration
    cat > "$AUTH_DIR/ldap/docker-compose.ldap.yml" << EOF
version: '3.8'

services:
  openldap:
    image: osixia/openldap:$LDAP_VERSION
    container_name: nexus-openldap
    restart: unless-stopped
    environment:
      LDAP_ORGANISATION: "Nexus Architect"
      LDAP_DOMAIN: "nexus.dev"
      LDAP_ADMIN_PASSWORD: "nexus_ldap_admin"
      LDAP_CONFIG_PASSWORD: "nexus_ldap_config"
      LDAP_READONLY_USER: "true"
      LDAP_READONLY_USER_USERNAME: "readonly"
      LDAP_READONLY_USER_PASSWORD: "nexus_ldap_readonly"
      LDAP_RFC2307BIS_SCHEMA: "false"
      LDAP_BACKEND: "mdb"
      LDAP_TLS: "true"
      LDAP_TLS_CRT_FILENAME: "ldap.crt"
      LDAP_TLS_KEY_FILENAME: "ldap.key"
      LDAP_TLS_DH_PARAM_FILENAME: "dhparam.pem"
      LDAP_TLS_CA_CRT_FILENAME: "ca.crt"
      LDAP_TLS_ENFORCE: "false"
      LDAP_TLS_CIPHER_SUITE: "SECURE256:-VERS-SSL3.0"
      LDAP_TLS_VERIFY_CLIENT: "demand"
      LDAP_REPLICATION: "false"
      KEEP_EXISTING_CONFIG: "false"
      LDAP_REMOVE_CONFIG_AFTER_SETUP: "true"
      LDAP_SSL_HELPER_PREFIX: "ldap"
    ports:
      - "389:389"
      - "636:636"
    volumes:
      - ldap_data:/var/lib/ldap
      - ldap_config:/etc/ldap/slapd.d
      - ./data:/container/service/slapd/assets/config/bootstrap/ldif/custom
      - ~/nexus-dev/certs/ssl:/container/service/slapd/assets/certs:ro
    networks:
      - nexus-auth-network
    healthcheck:
      test: ["CMD-SHELL", "ldapsearch -x -H ldap://localhost -b dc=nexus,dc=dev -D 'cn=admin,dc=nexus,dc=dev' -w nexus_ldap_admin || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  phpldapadmin:
    image: osixia/phpldapadmin:latest
    container_name: nexus-phpldapadmin
    restart: unless-stopped
    environment:
      PHPLDAPADMIN_LDAP_HOSTS: "openldap"
      PHPLDAPADMIN_HTTPS: "false"
    ports:
      - "8081:80"
    depends_on:
      - openldap
    networks:
      - nexus-auth-network

networks:
  nexus-auth-network:
    external: true

volumes:
  ldap_data:
    driver: local
  ldap_config:
    driver: local
EOF

    # Create LDAP bootstrap data
    cat > "$AUTH_DIR/ldap/data/01-users.ldif" << 'EOF'
# Organizational Units
dn: ou=people,dc=nexus,dc=dev
objectClass: organizationalUnit
ou: people

dn: ou=groups,dc=nexus,dc=dev
objectClass: organizationalUnit
ou: groups

# Groups
dn: cn=admins,ou=groups,dc=nexus,dc=dev
objectClass: groupOfNames
cn: admins
description: Administrator group
member: uid=admin,ou=people,dc=nexus,dc=dev

dn: cn=developers,ou=groups,dc=nexus,dc=dev
objectClass: groupOfNames
cn: developers
description: Developer group
member: uid=developer,ou=people,dc=nexus,dc=dev

dn: cn=managers,ou=groups,dc=nexus,dc=dev
objectClass: groupOfNames
cn: managers
description: Manager group
member: uid=manager,ou=people,dc=nexus,dc=dev

dn: cn=executives,ou=groups,dc=nexus,dc=dev
objectClass: groupOfNames
cn: executives
description: Executive group
member: uid=executive,ou=people,dc=nexus,dc=dev

# Users
dn: uid=admin,ou=people,dc=nexus,dc=dev
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: admin
sn: User
givenName: Admin
cn: Admin User
displayName: Admin User
uidNumber: 1001
gidNumber: 1001
userPassword: {SSHA}password
gecos: Admin User
loginShell: /bin/bash
homeDirectory: /home/admin
mail: admin@nexus.dev

dn: uid=developer,ou=people,dc=nexus,dc=dev
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: developer
sn: User
givenName: Developer
cn: Developer User
displayName: Developer User
uidNumber: 1002
gidNumber: 1002
userPassword: {SSHA}password
gecos: Developer User
loginShell: /bin/bash
homeDirectory: /home/developer
mail: developer@nexus.dev

dn: uid=manager,ou=people,dc=nexus,dc=dev
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: manager
sn: Manager
givenName: Project
cn: Project Manager
displayName: Project Manager
uidNumber: 1003
gidNumber: 1003
userPassword: {SSHA}password
gecos: Project Manager
loginShell: /bin/bash
homeDirectory: /home/manager
mail: manager@nexus.dev

dn: uid=executive,ou=people,dc=nexus,dc=dev
objectClass: inetOrgPerson
objectClass: posixAccount
objectClass: shadowAccount
uid: executive
sn: User
givenName: Executive
cn: Executive User
displayName: Executive User
uidNumber: 1004
gidNumber: 1004
userPassword: {SSHA}password
gecos: Executive User
loginShell: /bin/bash
homeDirectory: /home/executive
mail: executive@nexus.dev
EOF

    success "OpenLDAP configuration created ✓"
}

# Setup SAML simulation
setup_saml() {
    log "Setting up SAML simulation..."
    
    # Generate SAML certificates
    mkdir -p "$AUTH_DIR/saml/certificates"
    
    # Generate SAML signing certificate
    openssl req -new -x509 -days 365 -nodes -out "$AUTH_DIR/saml/certificates/saml-signing.crt" -keyout "$AUTH_DIR/saml/certificates/saml-signing.key" -subj "/C=US/ST=CA/L=San Francisco/O=Nexus Architect/OU=Development/CN=SAML Signing"
    
    # Generate SAML encryption certificate
    openssl req -new -x509 -days 365 -nodes -out "$AUTH_DIR/saml/certificates/saml-encryption.crt" -keyout "$AUTH_DIR/saml/certificates/saml-encryption.key" -subj "/C=US/ST=CA/L=San Francisco/O=Nexus Architect/OU=Development/CN=SAML Encryption"
    
    # Create SAML metadata
    cat > "$AUTH_DIR/saml/metadata/idp-metadata.xml" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata" 
                     entityID="http://localhost:8080/auth/realms/nexus">
  <md:IDPSSODescriptor WantAuthnRequestsSigned="false" 
                       protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
    <md:KeyDescriptor use="signing">
      <ds:KeyInfo xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
        <ds:KeyName>saml-signing</ds:KeyName>
      </ds:KeyInfo>
    </md:KeyDescriptor>
    <md:KeyDescriptor use="encryption">
      <ds:KeyInfo xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
        <ds:KeyName>saml-encryption</ds:KeyName>
      </ds:KeyInfo>
    </md:KeyDescriptor>
    <md:SingleLogoutService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                            Location="http://localhost:8080/auth/realms/nexus/protocol/saml"/>
    <md:SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                            Location="http://localhost:8080/auth/realms/nexus/protocol/saml"/>
    <md:SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
                            Location="http://localhost:8080/auth/realms/nexus/protocol/saml"/>
  </md:IDPSSODescriptor>
</md:EntityDescriptor>
EOF

    success "SAML configuration created ✓"
}

# Setup OAuth2/OIDC simulation
setup_oauth() {
    log "Setting up OAuth2/OIDC simulation..."
    
    # Generate OAuth2 keys
    mkdir -p "$AUTH_DIR/oauth/keys"
    
    # Generate RSA key pair for JWT signing
    openssl genrsa -out "$AUTH_DIR/oauth/keys/private.pem" 2048
    openssl rsa -in "$AUTH_DIR/oauth/keys/private.pem" -pubout -out "$AUTH_DIR/oauth/keys/public.pem"
    
    # Create OAuth2 configuration
    cat > "$AUTH_DIR/oauth/config/oauth-config.json" << 'EOF'
{
  "issuer": "http://localhost:8080/auth/realms/nexus",
  "authorization_endpoint": "http://localhost:8080/auth/realms/nexus/protocol/openid-connect/auth",
  "token_endpoint": "http://localhost:8080/auth/realms/nexus/protocol/openid-connect/token",
  "userinfo_endpoint": "http://localhost:8080/auth/realms/nexus/protocol/openid-connect/userinfo",
  "jwks_uri": "http://localhost:8080/auth/realms/nexus/protocol/openid-connect/certs",
  "end_session_endpoint": "http://localhost:8080/auth/realms/nexus/protocol/openid-connect/logout",
  "response_types_supported": [
    "code",
    "id_token",
    "token",
    "id_token token",
    "code id_token",
    "code token",
    "code id_token token"
  ],
  "subject_types_supported": [
    "public",
    "pairwise"
  ],
  "id_token_signing_alg_values_supported": [
    "PS384",
    "ES384",
    "RS384",
    "HS256",
    "HS512",
    "ES256",
    "RS256",
    "HS384",
    "ES512",
    "PS256",
    "PS512",
    "RS512"
  ],
  "scopes_supported": [
    "openid",
    "profile",
    "email",
    "roles",
    "web-origins",
    "microprofile-jwt"
  ],
  "claims_supported": [
    "aud",
    "sub",
    "iss",
    "auth_time",
    "name",
    "given_name",
    "family_name",
    "preferred_username",
    "email",
    "acr"
  ]
}
EOF

    success "OAuth2/OIDC configuration created ✓"
}

# Create authentication testing scripts
create_auth_test_scripts() {
    log "Creating authentication testing scripts..."
    
    # Create Keycloak test script
    cat > "$AUTH_DIR/scripts/test-keycloak.sh" << 'EOF'
#!/bin/bash

echo "🔐 Testing Keycloak Authentication"
echo "================================="

KEYCLOAK_URL="http://localhost:8080"
REALM="nexus"
CLIENT_ID="nexus-frontend"
CLIENT_SECRET="nexus-frontend-secret"

# Test realm accessibility
echo "📋 Testing realm accessibility..."
if curl -f -s "$KEYCLOAK_URL/auth/realms/$REALM" > /dev/null; then
    echo "✅ Realm accessible"
else
    echo "❌ Realm not accessible"
    exit 1
fi

# Test token endpoint
echo "📋 Testing token endpoint..."
TOKEN_RESPONSE=$(curl -s -X POST "$KEYCLOAK_URL/auth/realms/$REALM/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=$CLIENT_ID" \
  -d "client_secret=$CLIENT_SECRET" \
  -d "username=admin" \
  -d "password=password")

if echo "$TOKEN_RESPONSE" | grep -q "access_token"; then
    echo "✅ Token endpoint working"
    ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.access_token')
    echo "🔑 Access token obtained"
else
    echo "❌ Token endpoint failed"
    echo "Response: $TOKEN_RESPONSE"
    exit 1
fi

# Test userinfo endpoint
echo "📋 Testing userinfo endpoint..."
USERINFO_RESPONSE=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" "$KEYCLOAK_URL/auth/realms/$REALM/protocol/openid-connect/userinfo")

if echo "$USERINFO_RESPONSE" | grep -q "preferred_username"; then
    echo "✅ Userinfo endpoint working"
    echo "👤 User: $(echo "$USERINFO_RESPONSE" | jq -r '.preferred_username')"
else
    echo "❌ Userinfo endpoint failed"
    echo "Response: $USERINFO_RESPONSE"
fi

echo "🎉 Keycloak authentication test completed!"
EOF

    # Create LDAP test script
    cat > "$AUTH_DIR/scripts/test-ldap.sh" << 'EOF'
#!/bin/bash

echo "📁 Testing LDAP Authentication"
echo "=============================="

LDAP_HOST="localhost"
LDAP_PORT="389"
LDAP_BASE_DN="dc=nexus,dc=dev"
LDAP_ADMIN_DN="cn=admin,dc=nexus,dc=dev"
LDAP_ADMIN_PASSWORD="nexus_ldap_admin"

# Test LDAP connectivity
echo "📋 Testing LDAP connectivity..."
if ldapsearch -x -H "ldap://$LDAP_HOST:$LDAP_PORT" -b "$LDAP_BASE_DN" -D "$LDAP_ADMIN_DN" -w "$LDAP_ADMIN_PASSWORD" "(objectClass=*)" dn > /dev/null 2>&1; then
    echo "✅ LDAP connectivity successful"
else
    echo "❌ LDAP connectivity failed"
    exit 1
fi

# Test user authentication
echo "📋 Testing user authentication..."
TEST_USERS=("admin" "developer" "manager" "executive")

for user in "${TEST_USERS[@]}"; do
    if ldapsearch -x -H "ldap://$LDAP_HOST:$LDAP_PORT" -b "ou=people,$LDAP_BASE_DN" -D "uid=$user,ou=people,$LDAP_BASE_DN" -w "password" "(uid=$user)" dn > /dev/null 2>&1; then
        echo "✅ User $user authentication successful"
    else
        echo "❌ User $user authentication failed"
    fi
done

# Test group membership
echo "📋 Testing group membership..."
GROUPS=$(ldapsearch -x -H "ldap://$LDAP_HOST:$LDAP_PORT" -b "ou=groups,$LDAP_BASE_DN" -D "$LDAP_ADMIN_DN" -w "$LDAP_ADMIN_PASSWORD" "(objectClass=groupOfNames)" cn | grep "^cn:" | cut -d' ' -f2)

echo "📋 Available groups:"
for group in $GROUPS; do
    echo "   - $group"
done

echo "🎉 LDAP authentication test completed!"
EOF

    # Create comprehensive auth test script
    cat > "$AUTH_DIR/scripts/test-all-auth.sh" << 'EOF'
#!/bin/bash

echo "🔒 Comprehensive Authentication Test Suite"
echo "=========================================="

AUTH_DIR="$HOME/nexus-dev/auth"

# Test Keycloak
echo "🔐 Testing Keycloak..."
if [[ -f "$AUTH_DIR/scripts/test-keycloak.sh" ]]; then
    bash "$AUTH_DIR/scripts/test-keycloak.sh"
else
    echo "⚠️  Keycloak test script not found"
fi

echo ""

# Test LDAP
echo "📁 Testing LDAP..."
if [[ -f "$AUTH_DIR/scripts/test-ldap.sh" ]]; then
    bash "$AUTH_DIR/scripts/test-ldap.sh"
else
    echo "⚠️  LDAP test script not found"
fi

echo ""

# Test SSL certificates
echo "🔒 Testing SSL certificates..."
SSL_DIR="$HOME/nexus-dev/certs/ssl"
if [[ -f "$SSL_DIR/server-cert.pem" ]]; then
    if openssl x509 -in "$SSL_DIR/server-cert.pem" -text -noout > /dev/null 2>&1; then
        echo "✅ SSL certificates valid"
    else
        echo "❌ SSL certificates invalid"
    fi
else
    echo "⚠️  SSL certificates not found"
fi

echo ""
echo "🎉 Comprehensive authentication test completed!"
EOF

    # Make scripts executable
    chmod +x "$AUTH_DIR/scripts"/*.sh
    
    success "Authentication test scripts created ✓"
}

# Create authentication startup script
create_auth_startup_script() {
    log "Creating authentication startup script..."
    
    cat > "$HOME/nexus-dev/start-auth.sh" << 'EOF'
#!/bin/bash

echo "🔐 Starting Nexus Architect Authentication Services"
echo "=================================================="

AUTH_DIR="$HOME/nexus-dev/auth"

# Start Keycloak
echo "🔐 Starting Keycloak..."
cd "$AUTH_DIR/keycloak"
docker-compose -f docker-compose.keycloak.yml up -d

# Start OpenLDAP
echo "📁 Starting OpenLDAP..."
cd "$AUTH_DIR/ldap"
docker-compose -f docker-compose.ldap.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check Keycloak
if curl -f -s "http://localhost:8080/health/ready" > /dev/null; then
    echo "✅ Keycloak is ready"
else
    echo "⚠️  Keycloak is not ready yet"
fi

# Check LDAP
if ldapsearch -x -H "ldap://localhost:389" -b "dc=nexus,dc=dev" -D "cn=admin,dc=nexus,dc=dev" -w "nexus_ldap_admin" "(objectClass=*)" dn > /dev/null 2>&1; then
    echo "✅ LDAP is ready"
else
    echo "⚠️  LDAP is not ready yet"
fi

echo ""
echo "🎉 Authentication services started!"
echo "🌐 Keycloak Admin: http://localhost:8080/admin (admin/nexus_admin_password)"
echo "📁 phpLDAPadmin: http://localhost:8081 (cn=admin,dc=nexus,dc=dev/nexus_ldap_admin)"
echo "🔍 Test authentication: ~/nexus-dev/auth/scripts/test-all-auth.sh"
EOF

    chmod +x "$HOME/nexus-dev/start-auth.sh"
    success "Authentication startup script created ✓"
}

# Create authentication configuration for applications
create_app_auth_config() {
    log "Creating authentication configuration for applications..."
    
    # Create environment variables for authentication
    cat > "$AUTH_DIR/config/auth.env" << 'EOF'
# Nexus Architect Authentication Configuration
# Generated for local development

# Keycloak Configuration
KEYCLOAK_URL=http://localhost:8080
KEYCLOAK_REALM=nexus
KEYCLOAK_CLIENT_ID=nexus-frontend
KEYCLOAK_CLIENT_SECRET=nexus-frontend-secret
KEYCLOAK_API_CLIENT_ID=nexus-api
KEYCLOAK_API_CLIENT_SECRET=nexus-api-secret

# LDAP Configuration
LDAP_URL=ldap://localhost:389
LDAP_BASE_DN=dc=nexus,dc=dev
LDAP_BIND_DN=cn=admin,dc=nexus,dc=dev
LDAP_BIND_PASSWORD=nexus_ldap_admin
LDAP_USER_SEARCH_BASE=ou=people,dc=nexus,dc=dev
LDAP_GROUP_SEARCH_BASE=ou=groups,dc=nexus,dc=dev

# SAML Configuration
SAML_IDP_METADATA_URL=http://localhost:8080/auth/realms/nexus/protocol/saml/descriptor
SAML_SP_ENTITY_ID=nexus-frontend
SAML_SP_ACS_URL=http://localhost:3000/auth/saml/callback

# OAuth2/OIDC Configuration
OAUTH_ISSUER=http://localhost:8080/auth/realms/nexus
OAUTH_AUTHORIZATION_URL=http://localhost:8080/auth/realms/nexus/protocol/openid-connect/auth
OAUTH_TOKEN_URL=http://localhost:8080/auth/realms/nexus/protocol/openid-connect/token
OAUTH_USERINFO_URL=http://localhost:8080/auth/realms/nexus/protocol/openid-connect/userinfo
OAUTH_JWKS_URL=http://localhost:8080/auth/realms/nexus/protocol/openid-connect/certs

# JWT Configuration
JWT_ALGORITHM=RS256
JWT_PUBLIC_KEY_PATH=/home/$(whoami)/nexus-dev/auth/oauth/keys/public.pem
JWT_PRIVATE_KEY_PATH=/home/$(whoami)/nexus-dev/auth/oauth/keys/private.pem

# Session Configuration
SESSION_SECRET=nexus_session_secret_for_development_only
SESSION_TIMEOUT=3600
REMEMBER_ME_TIMEOUT=604800

# Security Configuration
ENABLE_2FA=false
PASSWORD_MIN_LENGTH=8
PASSWORD_REQUIRE_SPECIAL_CHARS=true
ACCOUNT_LOCKOUT_ATTEMPTS=5
ACCOUNT_LOCKOUT_DURATION=900
EOF

    success "Application authentication configuration created ✓"
}

# Main execution
main() {
    log "🎯 BDT-P1 Deliverable #7: Local SSO simulation and authentication testing"
    
    create_auth_directories
    setup_keycloak
    setup_openldap
    setup_saml
    setup_oauth
    create_auth_test_scripts
    create_auth_startup_script
    create_app_auth_config
    
    success "🎉 Local authentication setup completed successfully!"
    success "🔐 Keycloak: Ready for SSO simulation"
    success "📁 OpenLDAP: Ready for directory services"
    success "🔒 SAML: Metadata and certificates generated"
    success "🔑 OAuth2/OIDC: Keys and configuration ready"
    
    log "📋 Next steps:"
    log "   1. Start authentication services: ~/nexus-dev/start-auth.sh"
    log "   2. Test authentication: ~/nexus-dev/auth/scripts/test-all-auth.sh"
    log "   3. Configure applications to use authentication"
    log "   4. Access Keycloak admin: http://localhost:8080/admin"
    log "   5. Access phpLDAPadmin: http://localhost:8081"
    
    info "📋 Test credentials:"
    info "   Admin: admin/password"
    info "   Developer: developer/password"
    info "   Manager: manager/password"
    info "   Executive: executive/password"
    
    warning "⚠️  These are development credentials only. Use proper authentication in production."
}

# Run main function
main "$@"

