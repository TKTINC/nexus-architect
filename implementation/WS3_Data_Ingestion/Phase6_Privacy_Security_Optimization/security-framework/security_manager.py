"""
Security Manager for Nexus Architect
Implements comprehensive security controls including encryption,
access controls, audit logging, and threat detection.
"""

import hashlib
import hmac
import secrets
import logging
import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import ipaddress
import re
from collections import defaultdict, deque
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AccessAction(Enum):
    """Types of access actions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    PASSWORD = "password"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    CERTIFICATE = "certificate"
    OAUTH = "oauth"
    SAML = "saml"

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    result: str  # success, failure, blocked
    threat_level: ThreatLevel
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessPolicy:
    """Access control policy"""
    policy_id: str
    name: str
    description: str
    resources: Set[str]
    actions: Set[AccessAction]
    users: Set[str] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    ip_restrictions: Set[str] = field(default_factory=set)
    time_restrictions: Optional[Dict[str, Any]] = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

@dataclass
class ThreatIndicator:
    """Threat detection indicator"""
    indicator_id: str
    indicator_type: str  # ip, user_agent, pattern, behavior
    value: str
    threat_level: ThreatLevel
    description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None
    hit_count: int = 0

@dataclass
class UserSession:
    """User session information"""
    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    authentication_method: AuthenticationMethod
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    is_active: bool = True

class SecurityManager:
    """
    Comprehensive security manager implementing enterprise-grade
    security controls, encryption, and threat detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the security manager"""
        self.config = config
        self.database_config = config.get('database', {})
        
        # Initialize encryption
        self.master_key = self._generate_master_key()
        self.cipher_suite = Fernet(self.master_key)
        
        # Generate RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # JWT configuration
        self.jwt_secret = config.get('jwt_secret', secrets.token_hex(32))
        self.jwt_algorithm = 'HS256'
        self.jwt_expiry = timedelta(hours=24)
        
        # Security policies and sessions
        self.access_policies: Dict[str, AccessPolicy] = {}
        self.user_sessions: Dict[str, UserSession] = {}
        self.security_events: deque = deque(maxlen=10000)  # Keep last 10k events
        
        # Threat detection
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.failed_attempts: defaultdict = defaultdict(list)
        self.rate_limits: defaultdict = defaultdict(list)
        
        # Security monitoring
        self.monitoring_enabled = True
        self.monitoring_thread = None
        self._start_monitoring()
        
        logger.info("Security Manager initialized successfully")
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key"""
        password = self.config.get('master_password', 'nexus_security_master_2024').encode()
        salt = self.config.get('master_salt', 'nexus_security_salt').encode()
        
        # Use PBKDF2 for key derivation
        key = hashlib.pbkdf2_hmac('sha256', password, salt, 100000)
        return base64.urlsafe_b64encode(key)
    
    def _start_monitoring(self):
        """Start security monitoring thread"""
        if self.monitoring_enabled and not self.monitoring_thread:
            self.monitoring_thread = threading.Thread(target=self._security_monitor, daemon=True)
            self.monitoring_thread.start()
            logger.info("Security monitoring started")
    
    def _security_monitor(self):
        """Background security monitoring"""
        while self.monitoring_enabled:
            try:
                # Clean expired sessions
                self._cleanup_expired_sessions()
                
                # Clean old failed attempts
                self._cleanup_failed_attempts()
                
                # Clean rate limit tracking
                self._cleanup_rate_limits()
                
                # Analyze threat patterns
                self._analyze_threat_patterns()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in security monitoring: {str(e)}")
                time.sleep(60)
    
    def _cleanup_expired_sessions(self):
        """Clean up expired user sessions"""
        current_time = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session in self.user_sessions.items()
            if session.expires_at < current_time
        ]
        
        for session_id in expired_sessions:
            del self.user_sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    def _cleanup_failed_attempts(self):
        """Clean up old failed login attempts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for identifier in list(self.failed_attempts.keys()):
            self.failed_attempts[identifier] = [
                attempt for attempt in self.failed_attempts[identifier]
                if attempt > cutoff_time
            ]
            
            if not self.failed_attempts[identifier]:
                del self.failed_attempts[identifier]
    
    def _cleanup_rate_limits(self):
        """Clean up old rate limit tracking"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=15)
        
        for identifier in list(self.rate_limits.keys()):
            self.rate_limits[identifier] = [
                request_time for request_time in self.rate_limits[identifier]
                if request_time > cutoff_time
            ]
            
            if not self.rate_limits[identifier]:
                del self.rate_limits[identifier]
    
    def _analyze_threat_patterns(self):
        """Analyze security events for threat patterns"""
        try:
            # Analyze recent security events
            recent_events = [
                event for event in self.security_events
                if event.timestamp > datetime.utcnow() - timedelta(hours=1)
            ]
            
            # Detect suspicious patterns
            ip_counts = defaultdict(int)
            user_counts = defaultdict(int)
            
            for event in recent_events:
                if event.result == "failure":
                    if event.ip_address:
                        ip_counts[event.ip_address] += 1
                    if event.user_id:
                        user_counts[event.user_id] += 1
            
            # Flag suspicious IPs (>10 failures in 1 hour)
            for ip, count in ip_counts.items():
                if count > 10:
                    self._add_threat_indicator(
                        indicator_type="ip",
                        value=ip,
                        threat_level=ThreatLevel.HIGH,
                        description=f"Suspicious IP with {count} failed attempts"
                    )
            
            # Flag suspicious users (>5 failures in 1 hour)
            for user, count in user_counts.items():
                if count > 5:
                    self._add_threat_indicator(
                        indicator_type="user",
                        value=user,
                        threat_level=ThreatLevel.MEDIUM,
                        description=f"Suspicious user with {count} failed attempts"
                    )
                    
        except Exception as e:
            logger.error(f"Error analyzing threat patterns: {str(e)}")
    
    def _add_threat_indicator(self, indicator_type: str, value: str, 
                            threat_level: ThreatLevel, description: str):
        """Add a new threat indicator"""
        indicator_id = hashlib.sha256(f"{indicator_type}:{value}".encode()).hexdigest()[:16]
        
        if indicator_id in self.threat_indicators:
            # Update existing indicator
            indicator = self.threat_indicators[indicator_id]
            indicator.last_seen = datetime.utcnow()
            indicator.hit_count += 1
        else:
            # Create new indicator
            indicator = ThreatIndicator(
                indicator_id=indicator_id,
                indicator_type=indicator_type,
                value=value,
                threat_level=threat_level,
                description=description
            )
            self.threat_indicators[indicator_id] = indicator
        
        logger.warning(f"Threat indicator added: {indicator_type}:{value} - {description}")
    
    async def encrypt_data(self, data: str, use_asymmetric: bool = False) -> str:
        """
        Encrypt data using symmetric or asymmetric encryption
        
        Args:
            data: Data to encrypt
            use_asymmetric: Use RSA asymmetric encryption
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            if use_asymmetric:
                # Use RSA encryption for small data
                encrypted_data = self.public_key.encrypt(
                    data.encode(),
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            else:
                # Use Fernet symmetric encryption
                encrypted_data = self.cipher_suite.encrypt(data.encode())
            
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            raise
    
    async def decrypt_data(self, encrypted_data: str, use_asymmetric: bool = False) -> str:
        """
        Decrypt data using symmetric or asymmetric encryption
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            use_asymmetric: Use RSA asymmetric decryption
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            
            if use_asymmetric:
                # Use RSA decryption
                decrypted_data = self.private_key.decrypt(
                    encrypted_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            else:
                # Use Fernet symmetric decryption
                decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            
            return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            raise
    
    async def hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error hashing password: {str(e)}")
            raise
    
    async def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify password against hash
        
        Args:
            password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            Verification result
        """
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            return False
    
    async def generate_jwt_token(self, user_id: str, roles: List[str], 
                               security_level: SecurityLevel = SecurityLevel.INTERNAL) -> str:
        """
        Generate JWT token for user authentication
        
        Args:
            user_id: User identifier
            roles: User roles
            security_level: Security clearance level
            
        Returns:
            JWT token
        """
        try:
            payload = {
                'user_id': user_id,
                'roles': roles,
                'security_level': security_level.value,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + self.jwt_expiry,
                'jti': secrets.token_hex(16)  # JWT ID for revocation
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            return token
            
        except Exception as e:
            logger.error(f"Error generating JWT token: {str(e)}")
            raise
    
    async def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
        except Exception as e:
            logger.error(f"Error verifying JWT token: {str(e)}")
            return None
    
    async def create_user_session(self, user_id: str, ip_address: str, 
                                user_agent: str, auth_method: AuthenticationMethod,
                                security_level: SecurityLevel = SecurityLevel.INTERNAL) -> UserSession:
        """
        Create a new user session
        
        Args:
            user_id: User identifier
            ip_address: Client IP address
            user_agent: Client user agent
            auth_method: Authentication method used
            security_level: Security clearance level
            
        Returns:
            Created user session
        """
        try:
            session_id = secrets.token_hex(32)
            current_time = datetime.utcnow()
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                authentication_method=auth_method,
                created_at=current_time,
                last_activity=current_time,
                expires_at=current_time + timedelta(hours=8),  # 8 hour session
                security_level=security_level
            )
            
            self.user_sessions[session_id] = session
            
            # Log security event
            await self._log_security_event(
                event_type="session_created",
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                result="success",
                threat_level=ThreatLevel.LOW,
                details={"session_id": session_id, "auth_method": auth_method.value}
            )
            
            logger.info(f"Created session for user {user_id}: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating user session: {str(e)}")
            raise
    
    async def validate_session(self, session_id: str, ip_address: str) -> Optional[UserSession]:
        """
        Validate user session
        
        Args:
            session_id: Session identifier
            ip_address: Client IP address
            
        Returns:
            Valid session or None
        """
        try:
            if session_id not in self.user_sessions:
                return None
            
            session = self.user_sessions[session_id]
            current_time = datetime.utcnow()
            
            # Check if session is expired
            if session.expires_at < current_time:
                del self.user_sessions[session_id]
                return None
            
            # Check if session is active
            if not session.is_active:
                return None
            
            # Validate IP address (optional strict checking)
            if self.config.get('strict_ip_validation', False):
                if session.ip_address != ip_address:
                    logger.warning(f"IP mismatch for session {session_id}: {session.ip_address} vs {ip_address}")
                    return None
            
            # Update last activity
            session.last_activity = current_time
            
            return session
            
        except Exception as e:
            logger.error(f"Error validating session: {str(e)}")
            return None
    
    async def check_access_permission(self, user_id: str, resource: str, 
                                    action: AccessAction, ip_address: str) -> bool:
        """
        Check if user has permission to perform action on resource
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            ip_address: Client IP address
            
        Returns:
            Permission granted status
        """
        try:
            # Check threat indicators
            if await self._is_threat_ip(ip_address):
                await self._log_security_event(
                    event_type="access_blocked_threat",
                    user_id=user_id,
                    ip_address=ip_address,
                    resource=resource,
                    action=action.value,
                    result="blocked",
                    threat_level=ThreatLevel.HIGH,
                    details={"reason": "threat_ip"}
                )
                return False
            
            # Check rate limiting
            if not await self._check_rate_limit(user_id, ip_address):
                await self._log_security_event(
                    event_type="access_blocked_rate_limit",
                    user_id=user_id,
                    ip_address=ip_address,
                    resource=resource,
                    action=action.value,
                    result="blocked",
                    threat_level=ThreatLevel.MEDIUM,
                    details={"reason": "rate_limit"}
                )
                return False
            
            # Check access policies
            for policy in self.access_policies.values():
                if self._policy_matches(policy, user_id, resource, action, ip_address):
                    await self._log_security_event(
                        event_type="access_granted",
                        user_id=user_id,
                        ip_address=ip_address,
                        resource=resource,
                        action=action.value,
                        result="success",
                        threat_level=ThreatLevel.LOW,
                        details={"policy_id": policy.policy_id}
                    )
                    return True
            
            # No matching policy found
            await self._log_security_event(
                event_type="access_denied",
                user_id=user_id,
                ip_address=ip_address,
                resource=resource,
                action=action.value,
                result="failure",
                threat_level=ThreatLevel.MEDIUM,
                details={"reason": "no_policy_match"}
            )
            return False
            
        except Exception as e:
            logger.error(f"Error checking access permission: {str(e)}")
            return False
    
    def _policy_matches(self, policy: AccessPolicy, user_id: str, resource: str, 
                       action: AccessAction, ip_address: str) -> bool:
        """Check if access policy matches the request"""
        try:
            # Check if policy is expired
            if policy.expires_at and policy.expires_at < datetime.utcnow():
                return False
            
            # Check resource match
            if policy.resources and not any(
                resource.startswith(res) for res in policy.resources
            ):
                return False
            
            # Check action match
            if policy.actions and action not in policy.actions:
                return False
            
            # Check user match
            if policy.users and user_id not in policy.users:
                return False
            
            # Check IP restrictions
            if policy.ip_restrictions:
                ip_allowed = False
                for ip_restriction in policy.ip_restrictions:
                    try:
                        if '/' in ip_restriction:
                            # CIDR notation
                            network = ipaddress.ip_network(ip_restriction, strict=False)
                            if ipaddress.ip_address(ip_address) in network:
                                ip_allowed = True
                                break
                        else:
                            # Exact IP match
                            if ip_address == ip_restriction:
                                ip_allowed = True
                                break
                    except ValueError:
                        continue
                
                if not ip_allowed:
                    return False
            
            # Check time restrictions
            if policy.time_restrictions:
                current_time = datetime.utcnow()
                # Implement time-based access control logic here
                # For now, assume time restrictions are met
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching policy: {str(e)}")
            return False
    
    async def _is_threat_ip(self, ip_address: str) -> bool:
        """Check if IP address is flagged as a threat"""
        for indicator in self.threat_indicators.values():
            if indicator.indicator_type == "ip" and indicator.value == ip_address:
                if indicator.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    return True
        return False
    
    async def _check_rate_limit(self, user_id: str, ip_address: str) -> bool:
        """Check rate limiting for user and IP"""
        current_time = datetime.utcnow()
        
        # Rate limit by user (100 requests per 15 minutes)
        user_requests = self.rate_limits[f"user:{user_id}"]
        user_requests.append(current_time)
        if len(user_requests) > 100:
            return False
        
        # Rate limit by IP (200 requests per 15 minutes)
        ip_requests = self.rate_limits[f"ip:{ip_address}"]
        ip_requests.append(current_time)
        if len(ip_requests) > 200:
            return False
        
        return True
    
    async def _log_security_event(self, event_type: str, user_id: Optional[str] = None,
                                ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                                resource: Optional[str] = None, action: Optional[str] = None,
                                result: str = "unknown", threat_level: ThreatLevel = ThreatLevel.LOW,
                                details: Optional[Dict[str, Any]] = None):
        """Log security event for audit trail"""
        try:
            event = SecurityEvent(
                event_id=secrets.token_hex(16),
                timestamp=datetime.utcnow(),
                event_type=event_type,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                action=action,
                result=result,
                threat_level=threat_level,
                details=details or {}
            )
            
            self.security_events.append(event)
            
            # Log to file/database in production
            logger.info(f"Security event: {event_type} - {result} - {threat_level.value}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {str(e)}")
    
    async def create_access_policy(self, policy_data: Dict[str, Any]) -> AccessPolicy:
        """
        Create a new access control policy
        
        Args:
            policy_data: Policy configuration
            
        Returns:
            Created access policy
        """
        try:
            policy_id = policy_data.get('policy_id') or secrets.token_hex(8)
            
            policy = AccessPolicy(
                policy_id=policy_id,
                name=policy_data['name'],
                description=policy_data['description'],
                resources=set(policy_data.get('resources', [])),
                actions=set(AccessAction(action) for action in policy_data.get('actions', [])),
                users=set(policy_data.get('users', [])),
                roles=set(policy_data.get('roles', [])),
                ip_restrictions=set(policy_data.get('ip_restrictions', [])),
                time_restrictions=policy_data.get('time_restrictions'),
                security_level=SecurityLevel(policy_data.get('security_level', 'internal')),
                expires_at=policy_data.get('expires_at')
            )
            
            self.access_policies[policy_id] = policy
            logger.info(f"Created access policy: {policy_id}")
            
            return policy
            
        except Exception as e:
            logger.error(f"Error creating access policy: {str(e)}")
            raise
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """
        Get security metrics and statistics
        
        Returns:
            Security metrics dictionary
        """
        try:
            current_time = datetime.utcnow()
            
            # Count events by type in last 24 hours
            recent_events = [
                event for event in self.security_events
                if event.timestamp > current_time - timedelta(hours=24)
            ]
            
            event_counts = defaultdict(int)
            threat_counts = defaultdict(int)
            
            for event in recent_events:
                event_counts[event.event_type] += 1
                threat_counts[event.threat_level.value] += 1
            
            return {
                "active_sessions": len(self.user_sessions),
                "access_policies": len(self.access_policies),
                "threat_indicators": len(self.threat_indicators),
                "events_24h": len(recent_events),
                "event_types": dict(event_counts),
                "threat_levels": dict(threat_counts),
                "failed_attempts": sum(len(attempts) for attempts in self.failed_attempts.values()),
                "timestamp": current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting security metrics: {str(e)}")
            return {}
    
    async def perform_security_scan(self) -> Dict[str, Any]:
        """
        Perform comprehensive security scan
        
        Returns:
            Security scan results
        """
        try:
            scan_results = {
                "scan_id": secrets.token_hex(16),
                "timestamp": datetime.utcnow().isoformat(),
                "vulnerabilities": [],
                "recommendations": [],
                "risk_score": 0
            }
            
            # Check for expired sessions
            expired_sessions = [
                session for session in self.user_sessions.values()
                if session.expires_at < datetime.utcnow()
            ]
            
            if expired_sessions:
                scan_results["vulnerabilities"].append({
                    "type": "expired_sessions",
                    "severity": "medium",
                    "count": len(expired_sessions),
                    "description": "Expired sessions found in active session store"
                })
                scan_results["recommendations"].append("Clean up expired sessions regularly")
            
            # Check for weak access policies
            weak_policies = [
                policy for policy in self.access_policies.values()
                if not policy.ip_restrictions and not policy.time_restrictions
            ]
            
            if weak_policies:
                scan_results["vulnerabilities"].append({
                    "type": "weak_access_policies",
                    "severity": "low",
                    "count": len(weak_policies),
                    "description": "Access policies without IP or time restrictions"
                })
                scan_results["recommendations"].append("Add IP and time restrictions to access policies")
            
            # Check threat indicators
            high_threat_indicators = [
                indicator for indicator in self.threat_indicators.values()
                if indicator.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ]
            
            if high_threat_indicators:
                scan_results["vulnerabilities"].append({
                    "type": "high_threat_indicators",
                    "severity": "high",
                    "count": len(high_threat_indicators),
                    "description": "High-severity threat indicators detected"
                })
                scan_results["recommendations"].append("Investigate and mitigate high-threat indicators")
            
            # Calculate risk score
            risk_score = 0
            for vuln in scan_results["vulnerabilities"]:
                if vuln["severity"] == "critical":
                    risk_score += 40
                elif vuln["severity"] == "high":
                    risk_score += 20
                elif vuln["severity"] == "medium":
                    risk_score += 10
                elif vuln["severity"] == "low":
                    risk_score += 5
            
            scan_results["risk_score"] = min(risk_score, 100)
            
            return scan_results
            
        except Exception as e:
            logger.error(f"Error performing security scan: {str(e)}")
            return {"error": str(e)}

def create_security_api(security_manager: SecurityManager):
    """Create Flask API for security management"""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "service": "security_manager"})
    
    @app.route('/encrypt', methods=['POST'])
    async def encrypt_data():
        try:
            data = request.get_json()
            text = data.get('text', '')
            use_asymmetric = data.get('use_asymmetric', False)
            
            encrypted_text = await security_manager.encrypt_data(text, use_asymmetric)
            
            return jsonify({
                "status": "success",
                "encrypted_data": encrypted_text,
                "encryption_type": "asymmetric" if use_asymmetric else "symmetric"
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/decrypt', methods=['POST'])
    async def decrypt_data():
        try:
            data = request.get_json()
            encrypted_text = data.get('encrypted_data', '')
            use_asymmetric = data.get('use_asymmetric', False)
            
            decrypted_text = await security_manager.decrypt_data(encrypted_text, use_asymmetric)
            
            return jsonify({
                "status": "success",
                "decrypted_data": decrypted_text
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/generate-token', methods=['POST'])
    async def generate_token():
        try:
            data = request.get_json()
            user_id = data.get('user_id', '')
            roles = data.get('roles', [])
            security_level = SecurityLevel(data.get('security_level', 'internal'))
            
            token = await security_manager.generate_jwt_token(user_id, roles, security_level)
            
            return jsonify({
                "status": "success",
                "token": token,
                "expires_in": int(security_manager.jwt_expiry.total_seconds())
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/verify-token', methods=['POST'])
    async def verify_token():
        try:
            data = request.get_json()
            token = data.get('token', '')
            
            payload = await security_manager.verify_jwt_token(token)
            
            if payload:
                return jsonify({
                    "status": "success",
                    "valid": True,
                    "payload": payload
                })
            else:
                return jsonify({
                    "status": "success",
                    "valid": False
                })
                
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/check-access', methods=['POST'])
    async def check_access():
        try:
            data = request.get_json()
            user_id = data.get('user_id', '')
            resource = data.get('resource', '')
            action = AccessAction(data.get('action', 'read'))
            ip_address = data.get('ip_address', request.remote_addr)
            
            has_access = await security_manager.check_access_permission(
                user_id, resource, action, ip_address
            )
            
            return jsonify({
                "status": "success",
                "access_granted": has_access
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/security-metrics', methods=['GET'])
    async def get_security_metrics():
        try:
            metrics = await security_manager.get_security_metrics()
            return jsonify(metrics)
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/security-scan', methods=['POST'])
    async def perform_security_scan():
        try:
            scan_results = await security_manager.perform_security_scan()
            return jsonify(scan_results)
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    return app

if __name__ == "__main__":
    # Example configuration
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'nexus_architect',
            'user': 'postgres',
            'password': 'nexus_secure_password_2024'
        },
        'master_password': 'nexus_security_master_key_2024',
        'master_salt': 'nexus_security_salt_2024',
        'jwt_secret': 'nexus_jwt_secret_key_2024',
        'strict_ip_validation': False
    }
    
    # Initialize security manager
    security_manager = SecurityManager(config)
    
    # Create Flask API
    app = create_security_api(security_manager)
    
    print("Security Manager API starting...")
    print("Available endpoints:")
    print("  POST /encrypt - Encrypt data")
    print("  POST /decrypt - Decrypt data")
    print("  POST /generate-token - Generate JWT token")
    print("  POST /verify-token - Verify JWT token")
    print("  POST /check-access - Check access permissions")
    print("  GET /security-metrics - Get security metrics")
    print("  POST /security-scan - Perform security scan")
    
    app.run(host='0.0.0.0', port=8011, debug=False)

