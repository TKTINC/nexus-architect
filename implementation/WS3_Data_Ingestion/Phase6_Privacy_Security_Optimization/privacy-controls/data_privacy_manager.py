"""
Data Privacy Manager for Nexus Architect
Implements comprehensive data privacy controls including PII detection,
anonymization, pseudonymization, and consent management.
"""

import re
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PIIType(Enum):
    """Types of personally identifiable information"""
    EMAIL = "EMAIL"
    PHONE = "PHONE_NUMBER"
    SSN = "US_SSN"
    CREDIT_CARD = "CREDIT_CARD"
    PERSON_NAME = "PERSON"
    LOCATION = "LOCATION"
    DATE_TIME = "DATE_TIME"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    IBAN = "IBAN"
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"
    PASSPORT = "US_PASSPORT"
    BANK_ACCOUNT = "US_BANK_NUMBER"

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ConsentStatus(Enum):
    """Consent status for data processing"""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"

@dataclass
class PIIDetectionResult:
    """Result of PII detection analysis"""
    entity_type: PIIType
    text: str
    start: int
    end: int
    confidence: float
    anonymized_value: Optional[str] = None
    pseudonymized_value: Optional[str] = None

@dataclass
class DataSubject:
    """Data subject information for privacy compliance"""
    subject_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    consent_status: ConsentStatus = ConsentStatus.PENDING
    consent_date: Optional[datetime] = None
    consent_expiry: Optional[datetime] = None
    data_sources: Set[str] = field(default_factory=set)
    processing_purposes: Set[str] = field(default_factory=set)
    retention_period: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PrivacyPolicy:
    """Privacy policy configuration"""
    policy_id: str
    name: str
    description: str
    data_types: Set[PIIType]
    processing_purposes: Set[str]
    retention_period: timedelta
    anonymization_required: bool = True
    consent_required: bool = True
    geographic_restrictions: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)

class DataPrivacyManager:
    """
    Comprehensive data privacy manager implementing enterprise-grade
    privacy controls, PII detection, and compliance management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data privacy manager"""
        self.config = config
        self.database_config = config.get('database', {})
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Initialize PII detection engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic detection")
            self.nlp = None
        
        # Privacy policies and data subjects storage
        self.privacy_policies: Dict[str, PrivacyPolicy] = {}
        self.data_subjects: Dict[str, DataSubject] = {}
        
        # PII detection patterns
        self.pii_patterns = self._initialize_pii_patterns()
        
        # Anonymization strategies
        self.anonymization_strategies = self._initialize_anonymization_strategies()
        
        logger.info("Data Privacy Manager initialized successfully")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for data protection"""
        password = self.config.get('encryption_password', 'nexus_privacy_key_2024').encode()
        salt = self.config.get('encryption_salt', 'nexus_salt').encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _initialize_pii_patterns(self) -> Dict[PIIType, List[str]]:
        """Initialize regex patterns for PII detection"""
        return {
            PIIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PIIType.PHONE: [
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',
                r'\b\d{10}\b'
            ],
            PIIType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b'
            ],
            PIIType.CREDIT_CARD: [
                r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
            ],
            PIIType.IP_ADDRESS: [
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            ]
        }
    
    def _initialize_anonymization_strategies(self) -> Dict[PIIType, Dict[str, Any]]:
        """Initialize anonymization strategies for different PII types"""
        return {
            PIIType.EMAIL: {
                "method": "mask",
                "config": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 5, "from_end": True})
            },
            PIIType.PHONE: {
                "method": "mask",
                "config": OperatorConfig("mask", {"masking_char": "X", "chars_to_mask": 6, "from_end": True})
            },
            PIIType.SSN: {
                "method": "mask",
                "config": OperatorConfig("mask", {"masking_char": "X", "chars_to_mask": 5, "from_end": True})
            },
            PIIType.PERSON_NAME: {
                "method": "replace",
                "config": OperatorConfig("replace", {"new_value": "[PERSON]"})
            },
            PIIType.LOCATION: {
                "method": "replace",
                "config": OperatorConfig("replace", {"new_value": "[LOCATION]"})
            },
            PIIType.CREDIT_CARD: {
                "method": "mask",
                "config": OperatorConfig("mask", {"masking_char": "X", "chars_to_mask": 12, "from_end": True})
            }
        }
    
    async def detect_pii(self, text: str, language: str = "en") -> List[PIIDetectionResult]:
        """
        Detect personally identifiable information in text
        
        Args:
            text: Text to analyze for PII
            language: Language of the text
            
        Returns:
            List of PII detection results
        """
        try:
            results = []
            
            # Use Presidio analyzer for comprehensive PII detection
            analyzer_results = self.analyzer.analyze(
                text=text,
                language=language,
                entities=[pii_type.value for pii_type in PIIType]
            )
            
            for result in analyzer_results:
                pii_type = PIIType(result.entity_type)
                detected_text = text[result.start:result.end]
                
                pii_result = PIIDetectionResult(
                    entity_type=pii_type,
                    text=detected_text,
                    start=result.start,
                    end=result.end,
                    confidence=result.score
                )
                
                results.append(pii_result)
            
            # Additional pattern-based detection
            for pii_type, patterns in self.pii_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        # Check if already detected by Presidio
                        if not any(r.start <= match.start() < r.end for r in results):
                            pii_result = PIIDetectionResult(
                                entity_type=pii_type,
                                text=match.group(),
                                start=match.start(),
                                end=match.end(),
                                confidence=0.8  # Pattern-based confidence
                            )
                            results.append(pii_result)
            
            logger.info(f"Detected {len(results)} PII entities in text")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting PII: {str(e)}")
            return []
    
    async def anonymize_text(self, text: str, pii_results: List[PIIDetectionResult]) -> str:
        """
        Anonymize text by replacing PII with anonymized values
        
        Args:
            text: Original text
            pii_results: PII detection results
            
        Returns:
            Anonymized text
        """
        try:
            # Convert PII results to Presidio format
            analyzer_results = []
            for pii_result in pii_results:
                analyzer_results.append(
                    RecognizerResult(
                        entity_type=pii_result.entity_type.value,
                        start=pii_result.start,
                        end=pii_result.end,
                        score=pii_result.confidence
                    )
                )
            
            # Create anonymization operators
            operators = {}
            for pii_result in pii_results:
                pii_type = pii_result.entity_type
                if pii_type in self.anonymization_strategies:
                    strategy = self.anonymization_strategies[pii_type]
                    operators[pii_type.value] = strategy["config"]
            
            # Anonymize text
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )
            
            return anonymized_result.text
            
        except Exception as e:
            logger.error(f"Error anonymizing text: {str(e)}")
            return text
    
    async def pseudonymize_data(self, data: str, salt: Optional[str] = None) -> str:
        """
        Pseudonymize data using cryptographic hashing
        
        Args:
            data: Data to pseudonymize
            salt: Optional salt for hashing
            
        Returns:
            Pseudonymized data
        """
        try:
            if salt is None:
                salt = secrets.token_hex(16)
            
            # Create hash of data with salt
            hash_input = f"{data}{salt}".encode('utf-8')
            hash_object = hashlib.sha256(hash_input)
            pseudonym = hash_object.hexdigest()[:16]  # Use first 16 characters
            
            return f"PSEUDO_{pseudonym}"
            
        except Exception as e:
            logger.error(f"Error pseudonymizing data: {str(e)}")
            return data
    
    async def encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data using Fernet encryption
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            return data
    
    async def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            return encrypted_data
    
    async def classify_data_sensitivity(self, text: str) -> DataClassification:
        """
        Classify data sensitivity level based on content
        
        Args:
            text: Text to classify
            
        Returns:
            Data classification level
        """
        try:
            # Detect PII in text
            pii_results = await self.detect_pii(text)
            
            # Classification logic based on PII presence and types
            if not pii_results:
                return DataClassification.PUBLIC
            
            # Check for high-sensitivity PII
            high_sensitivity_types = {
                PIIType.SSN, PIIType.CREDIT_CARD, PIIType.MEDICAL_LICENSE,
                PIIType.PASSPORT, PIIType.BANK_ACCOUNT
            }
            
            if any(pii.entity_type in high_sensitivity_types for pii in pii_results):
                return DataClassification.RESTRICTED
            
            # Check for medium-sensitivity PII
            medium_sensitivity_types = {
                PIIType.EMAIL, PIIType.PHONE, PIIType.PERSON_NAME,
                PIIType.US_DRIVER_LICENSE
            }
            
            if any(pii.entity_type in medium_sensitivity_types for pii in pii_results):
                return DataClassification.CONFIDENTIAL
            
            # Low-sensitivity PII
            return DataClassification.INTERNAL
            
        except Exception as e:
            logger.error(f"Error classifying data sensitivity: {str(e)}")
            return DataClassification.INTERNAL
    
    async def register_data_subject(self, subject_data: Dict[str, Any]) -> DataSubject:
        """
        Register a new data subject for privacy compliance
        
        Args:
            subject_data: Data subject information
            
        Returns:
            Created data subject
        """
        try:
            subject_id = subject_data.get('subject_id') or secrets.token_hex(16)
            
            data_subject = DataSubject(
                subject_id=subject_id,
                email=subject_data.get('email'),
                name=subject_data.get('name'),
                consent_status=ConsentStatus(subject_data.get('consent_status', 'pending')),
                consent_date=subject_data.get('consent_date'),
                consent_expiry=subject_data.get('consent_expiry'),
                data_sources=set(subject_data.get('data_sources', [])),
                processing_purposes=set(subject_data.get('processing_purposes', [])),
                retention_period=subject_data.get('retention_period')
            )
            
            self.data_subjects[subject_id] = data_subject
            logger.info(f"Registered data subject: {subject_id}")
            
            return data_subject
            
        except Exception as e:
            logger.error(f"Error registering data subject: {str(e)}")
            raise
    
    async def update_consent(self, subject_id: str, consent_status: ConsentStatus,
                           purposes: Optional[Set[str]] = None) -> bool:
        """
        Update consent status for a data subject
        
        Args:
            subject_id: Data subject identifier
            consent_status: New consent status
            purposes: Optional processing purposes
            
        Returns:
            Success status
        """
        try:
            if subject_id not in self.data_subjects:
                logger.error(f"Data subject not found: {subject_id}")
                return False
            
            data_subject = self.data_subjects[subject_id]
            data_subject.consent_status = consent_status
            data_subject.consent_date = datetime.utcnow()
            data_subject.updated_at = datetime.utcnow()
            
            if purposes:
                data_subject.processing_purposes = purposes
            
            # Set expiry for granted consent (1 year default)
            if consent_status == ConsentStatus.GRANTED:
                data_subject.consent_expiry = datetime.utcnow() + timedelta(days=365)
            
            logger.info(f"Updated consent for subject {subject_id}: {consent_status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating consent: {str(e)}")
            return False
    
    async def check_consent_validity(self, subject_id: str, purpose: str) -> bool:
        """
        Check if consent is valid for a specific processing purpose
        
        Args:
            subject_id: Data subject identifier
            purpose: Processing purpose
            
        Returns:
            Consent validity status
        """
        try:
            if subject_id not in self.data_subjects:
                return False
            
            data_subject = self.data_subjects[subject_id]
            
            # Check consent status
            if data_subject.consent_status != ConsentStatus.GRANTED:
                return False
            
            # Check expiry
            if data_subject.consent_expiry and datetime.utcnow() > data_subject.consent_expiry:
                # Update status to expired
                data_subject.consent_status = ConsentStatus.EXPIRED
                return False
            
            # Check purpose
            if purpose not in data_subject.processing_purposes:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking consent validity: {str(e)}")
            return False
    
    async def process_data_subject_request(self, subject_id: str, request_type: str) -> Dict[str, Any]:
        """
        Process data subject rights requests (GDPR Article 15-22)
        
        Args:
            subject_id: Data subject identifier
            request_type: Type of request (access, rectification, erasure, portability)
            
        Returns:
            Request processing result
        """
        try:
            if subject_id not in self.data_subjects:
                return {"status": "error", "message": "Data subject not found"}
            
            data_subject = self.data_subjects[subject_id]
            
            if request_type == "access":
                # Right to access (Article 15)
                return {
                    "status": "success",
                    "data": {
                        "subject_id": data_subject.subject_id,
                        "email": data_subject.email,
                        "name": data_subject.name,
                        "consent_status": data_subject.consent_status.value,
                        "consent_date": data_subject.consent_date.isoformat() if data_subject.consent_date else None,
                        "data_sources": list(data_subject.data_sources),
                        "processing_purposes": list(data_subject.processing_purposes),
                        "created_at": data_subject.created_at.isoformat(),
                        "updated_at": data_subject.updated_at.isoformat()
                    }
                }
            
            elif request_type == "erasure":
                # Right to erasure (Article 17)
                del self.data_subjects[subject_id]
                return {"status": "success", "message": "Data subject data erased"}
            
            elif request_type == "portability":
                # Right to data portability (Article 20)
                export_data = {
                    "subject_id": data_subject.subject_id,
                    "email": data_subject.email,
                    "name": data_subject.name,
                    "consent_history": {
                        "status": data_subject.consent_status.value,
                        "date": data_subject.consent_date.isoformat() if data_subject.consent_date else None,
                        "purposes": list(data_subject.processing_purposes)
                    }
                }
                return {"status": "success", "data": export_data}
            
            else:
                return {"status": "error", "message": f"Unsupported request type: {request_type}"}
                
        except Exception as e:
            logger.error(f"Error processing data subject request: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def create_privacy_policy(self, policy_data: Dict[str, Any]) -> PrivacyPolicy:
        """
        Create a new privacy policy
        
        Args:
            policy_data: Privacy policy configuration
            
        Returns:
            Created privacy policy
        """
        try:
            policy_id = policy_data.get('policy_id') or secrets.token_hex(8)
            
            privacy_policy = PrivacyPolicy(
                policy_id=policy_id,
                name=policy_data['name'],
                description=policy_data['description'],
                data_types=set(PIIType(dt) for dt in policy_data['data_types']),
                processing_purposes=set(policy_data['processing_purposes']),
                retention_period=timedelta(days=policy_data['retention_days']),
                anonymization_required=policy_data.get('anonymization_required', True),
                consent_required=policy_data.get('consent_required', True),
                geographic_restrictions=set(policy_data.get('geographic_restrictions', []))
            )
            
            self.privacy_policies[policy_id] = privacy_policy
            logger.info(f"Created privacy policy: {policy_id}")
            
            return privacy_policy
            
        except Exception as e:
            logger.error(f"Error creating privacy policy: {str(e)}")
            raise
    
    async def get_privacy_metrics(self) -> Dict[str, Any]:
        """
        Get privacy compliance metrics and statistics
        
        Returns:
            Privacy metrics dictionary
        """
        try:
            total_subjects = len(self.data_subjects)
            consent_granted = sum(1 for ds in self.data_subjects.values() 
                                if ds.consent_status == ConsentStatus.GRANTED)
            consent_denied = sum(1 for ds in self.data_subjects.values() 
                               if ds.consent_status == ConsentStatus.DENIED)
            consent_expired = sum(1 for ds in self.data_subjects.values() 
                                if ds.consent_status == ConsentStatus.EXPIRED)
            
            return {
                "total_data_subjects": total_subjects,
                "consent_granted": consent_granted,
                "consent_denied": consent_denied,
                "consent_expired": consent_expired,
                "consent_rate": (consent_granted / total_subjects * 100) if total_subjects > 0 else 0,
                "total_privacy_policies": len(self.privacy_policies),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting privacy metrics: {str(e)}")
            return {}

def create_privacy_api(privacy_manager: DataPrivacyManager):
    """Create Flask API for data privacy management"""
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "service": "data_privacy_manager"})
    
    @app.route('/detect-pii', methods=['POST'])
    async def detect_pii():
        try:
            data = request.get_json()
            text = data.get('text', '')
            language = data.get('language', 'en')
            
            pii_results = await privacy_manager.detect_pii(text, language)
            
            return jsonify({
                "status": "success",
                "pii_detected": len(pii_results),
                "results": [
                    {
                        "entity_type": result.entity_type.value,
                        "text": result.text,
                        "start": result.start,
                        "end": result.end,
                        "confidence": result.confidence
                    }
                    for result in pii_results
                ]
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/anonymize', methods=['POST'])
    async def anonymize_text():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            # Detect PII first
            pii_results = await privacy_manager.detect_pii(text)
            
            # Anonymize text
            anonymized_text = await privacy_manager.anonymize_text(text, pii_results)
            
            return jsonify({
                "status": "success",
                "original_text": text,
                "anonymized_text": anonymized_text,
                "pii_detected": len(pii_results)
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/classify-sensitivity', methods=['POST'])
    async def classify_sensitivity():
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            classification = await privacy_manager.classify_data_sensitivity(text)
            
            return jsonify({
                "status": "success",
                "text": text,
                "classification": classification.value
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/data-subjects', methods=['POST'])
    async def register_data_subject():
        try:
            subject_data = request.get_json()
            data_subject = await privacy_manager.register_data_subject(subject_data)
            
            return jsonify({
                "status": "success",
                "subject_id": data_subject.subject_id,
                "message": "Data subject registered successfully"
            })
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/consent/<subject_id>', methods=['PUT'])
    async def update_consent(subject_id):
        try:
            data = request.get_json()
            consent_status = ConsentStatus(data.get('consent_status'))
            purposes = set(data.get('purposes', []))
            
            success = await privacy_manager.update_consent(subject_id, consent_status, purposes)
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "Consent updated successfully"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to update consent"
                }), 400
                
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/data-subject-request/<subject_id>/<request_type>', methods=['POST'])
    async def process_data_subject_request(subject_id, request_type):
        try:
            result = await privacy_manager.process_data_subject_request(subject_id, request_type)
            
            if result["status"] == "success":
                return jsonify(result)
            else:
                return jsonify(result), 400
                
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/privacy-metrics', methods=['GET'])
    async def get_privacy_metrics():
        try:
            metrics = await privacy_manager.get_privacy_metrics()
            return jsonify(metrics)
            
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
        'encryption_password': 'nexus_privacy_encryption_key_2024',
        'encryption_salt': 'nexus_privacy_salt_2024'
    }
    
    # Initialize privacy manager
    privacy_manager = DataPrivacyManager(config)
    
    # Create Flask API
    app = create_privacy_api(privacy_manager)
    
    print("Data Privacy Manager API starting...")
    print("Available endpoints:")
    print("  POST /detect-pii - Detect PII in text")
    print("  POST /anonymize - Anonymize text with PII")
    print("  POST /classify-sensitivity - Classify data sensitivity")
    print("  POST /data-subjects - Register data subject")
    print("  PUT /consent/<subject_id> - Update consent")
    print("  POST /data-subject-request/<subject_id>/<request_type> - Process data subject request")
    print("  GET /privacy-metrics - Get privacy compliance metrics")
    
    app.run(host='0.0.0.0', port=8010, debug=False)

