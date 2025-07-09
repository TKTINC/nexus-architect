"""
Intelligent Bug Analyzer for Nexus Architect
Comprehensive bug analysis with NLP, root cause analysis, and impact assessment
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from textblob import TextBlob
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BugSeverity(Enum):
    """Bug severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class BugCategory(Enum):
    """Bug category types"""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    UI_UX = "ui_ux"
    INTEGRATION = "integration"
    DATA = "data"
    CONFIGURATION = "configuration"

class ComponentType(Enum):
    """System component types"""
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    API = "api"
    INFRASTRUCTURE = "infrastructure"
    THIRD_PARTY = "third_party"

@dataclass
class BugReport:
    """Bug report data structure"""
    id: str
    title: str
    description: str
    reporter: str
    created_at: datetime
    severity: Optional[BugSeverity] = None
    category: Optional[BugCategory] = None
    component: Optional[ComponentType] = None
    tags: List[str] = None
    attachments: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.attachments is None:
            self.attachments = []

@dataclass
class RootCauseAnalysis:
    """Root cause analysis results"""
    primary_cause: str
    contributing_factors: List[str]
    affected_components: List[str]
    code_locations: List[str]
    log_patterns: List[str]
    confidence_score: float
    analysis_timestamp: datetime

@dataclass
class ImpactAssessment:
    """Bug impact assessment"""
    affected_users: int
    business_impact: str
    system_impact: str
    performance_impact: Dict[str, float]
    security_implications: List[str]
    estimated_fix_time: timedelta
    priority_score: float

@dataclass
class SimilarBug:
    """Similar bug information"""
    bug_id: str
    similarity_score: float
    resolution_method: str
    fix_time: timedelta
    success_rate: float

class IntelligentBugAnalyzer:
    """
    Comprehensive bug analysis engine with NLP, root cause analysis, and impact assessment
    """
    
    def __init__(self):
        self.nlp = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.bug_history = []
        self.component_graph = nx.DiGraph()
        self.pattern_database = {}
        self.performance_baselines = {}
        
        # Initialize NLP model
        self._initialize_nlp()
        
        # Initialize component relationships
        self._initialize_component_graph()
        
        # Initialize pattern database
        self._initialize_pattern_database()
    
    def _initialize_nlp(self):
        """Initialize NLP model for text analysis"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic NLP processing.")
            self.nlp = None
    
    def _initialize_component_graph(self):
        """Initialize system component relationship graph"""
        # Add common system components
        components = [
            "frontend", "backend", "database", "api", "cache",
            "authentication", "authorization", "payment", "notification",
            "search", "analytics", "logging", "monitoring"
        ]
        
        for component in components:
            self.component_graph.add_node(component)
        
        # Add common relationships
        relationships = [
            ("frontend", "api"),
            ("api", "backend"),
            ("backend", "database"),
            ("backend", "cache"),
            ("api", "authentication"),
            ("api", "authorization"),
            ("backend", "payment"),
            ("backend", "notification"),
            ("backend", "search"),
            ("backend", "analytics"),
            ("backend", "logging"),
            ("backend", "monitoring")
        ]
        
        for source, target in relationships:
            self.component_graph.add_edge(source, target)
    
    def _initialize_pattern_database(self):
        """Initialize common bug patterns database"""
        self.pattern_database = {
            "null_pointer": {
                "patterns": [
                    r"NullPointerException",
                    r"null.*reference",
                    r"cannot.*null",
                    r"undefined.*null"
                ],
                "category": BugCategory.FUNCTIONAL,
                "severity": BugSeverity.HIGH,
                "common_causes": [
                    "Uninitialized variable",
                    "Missing null check",
                    "Race condition",
                    "Incorrect API usage"
                ]
            },
            "memory_leak": {
                "patterns": [
                    r"OutOfMemoryError",
                    r"memory.*leak",
                    r"heap.*space",
                    r"memory.*usage.*high"
                ],
                "category": BugCategory.PERFORMANCE,
                "severity": BugSeverity.CRITICAL,
                "common_causes": [
                    "Unclosed resources",
                    "Circular references",
                    "Large object retention",
                    "Inefficient algorithms"
                ]
            },
            "sql_injection": {
                "patterns": [
                    r"SQL.*injection",
                    r"malicious.*query",
                    r"unauthorized.*database",
                    r"SQL.*error.*input"
                ],
                "category": BugCategory.SECURITY,
                "severity": BugSeverity.CRITICAL,
                "common_causes": [
                    "Unsanitized input",
                    "Dynamic query construction",
                    "Missing parameterization",
                    "Insufficient validation"
                ]
            },
            "performance_degradation": {
                "patterns": [
                    r"slow.*response",
                    r"timeout",
                    r"performance.*issue",
                    r"high.*latency"
                ],
                "category": BugCategory.PERFORMANCE,
                "severity": BugSeverity.MEDIUM,
                "common_causes": [
                    "Inefficient queries",
                    "Resource contention",
                    "Network issues",
                    "Algorithmic complexity"
                ]
            },
            "authentication_failure": {
                "patterns": [
                    r"authentication.*failed",
                    r"login.*error",
                    r"unauthorized.*access",
                    r"invalid.*credentials"
                ],
                "category": BugCategory.SECURITY,
                "severity": BugSeverity.HIGH,
                "common_causes": [
                    "Token expiration",
                    "Session management",
                    "Credential validation",
                    "Permission configuration"
                ]
            }
        }
    
    async def analyze_bug_report(self, bug_report: BugReport) -> Dict[str, Any]:
        """
        Comprehensive analysis of bug report
        
        Args:
            bug_report: Bug report to analyze
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        
        try:
            # Perform parallel analysis
            tasks = [
                self._analyze_text_content(bug_report),
                self._classify_bug(bug_report),
                self._identify_components(bug_report),
                self._perform_root_cause_analysis(bug_report),
                self._assess_impact(bug_report),
                self._find_similar_bugs(bug_report)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Combine results
            analysis_result = {
                "bug_id": bug_report.id,
                "text_analysis": results[0],
                "classification": results[1],
                "components": results[2],
                "root_cause": results[3],
                "impact_assessment": results[4],
                "similar_bugs": results[5],
                "analysis_time": time.time() - start_time,
                "timestamp": datetime.now()
            }
            
            # Store for future reference
            self.bug_history.append(analysis_result)
            
            logger.info(f"Bug analysis completed for {bug_report.id} in {analysis_result['analysis_time']:.2f}s")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing bug report {bug_report.id}: {str(e)}")
            raise
    
    async def _analyze_text_content(self, bug_report: BugReport) -> Dict[str, Any]:
        """Analyze text content using NLP"""
        try:
            combined_text = f"{bug_report.title} {bug_report.description}"
            
            # Basic text analysis
            blob = TextBlob(combined_text)
            sentiment = blob.sentiment
            
            # Extract entities if spaCy is available
            entities = []
            if self.nlp:
                doc = self.nlp(combined_text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Extract keywords using TF-IDF
            keywords = self._extract_keywords(combined_text)
            
            # Identify error patterns
            error_patterns = self._identify_error_patterns(combined_text)
            
            return {
                "sentiment": {
                    "polarity": sentiment.polarity,
                    "subjectivity": sentiment.subjectivity
                },
                "entities": entities,
                "keywords": keywords,
                "error_patterns": error_patterns,
                "text_length": len(combined_text),
                "word_count": len(combined_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            return {}
    
    async def _classify_bug(self, bug_report: BugReport) -> Dict[str, Any]:
        """Classify bug severity and category"""
        try:
            combined_text = f"{bug_report.title} {bug_report.description}".lower()
            
            # Severity classification
            severity_keywords = {
                BugSeverity.CRITICAL: [
                    "critical", "crash", "down", "outage", "security",
                    "data loss", "corruption", "breach", "exploit"
                ],
                BugSeverity.HIGH: [
                    "high", "urgent", "important", "major", "broken",
                    "failure", "error", "exception", "bug"
                ],
                BugSeverity.MEDIUM: [
                    "medium", "moderate", "issue", "problem", "slow",
                    "performance", "delay", "timeout"
                ],
                BugSeverity.LOW: [
                    "low", "minor", "cosmetic", "enhancement", "improvement",
                    "suggestion", "nice to have"
                ]
            }
            
            # Category classification
            category_keywords = {
                BugCategory.FUNCTIONAL: [
                    "function", "feature", "behavior", "logic", "calculation",
                    "workflow", "process", "operation"
                ],
                BugCategory.PERFORMANCE: [
                    "performance", "slow", "fast", "speed", "latency",
                    "timeout", "memory", "cpu", "load"
                ],
                BugCategory.SECURITY: [
                    "security", "authentication", "authorization", "permission",
                    "access", "vulnerability", "exploit", "injection"
                ],
                BugCategory.UI_UX: [
                    "ui", "ux", "interface", "display", "layout", "design",
                    "usability", "navigation", "visual"
                ],
                BugCategory.INTEGRATION: [
                    "integration", "api", "service", "connection", "communication",
                    "sync", "external", "third party"
                ],
                BugCategory.DATA: [
                    "data", "database", "query", "record", "field",
                    "validation", "format", "import", "export"
                ]
            }
            
            # Calculate scores
            severity_scores = {}
            for severity, keywords in severity_keywords.items():
                score = sum(1 for keyword in keywords if keyword in combined_text)
                severity_scores[severity.value] = score
            
            category_scores = {}
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in combined_text)
                category_scores[category.value] = score
            
            # Determine classifications
            predicted_severity = max(severity_scores, key=severity_scores.get)
            predicted_category = max(category_scores, key=category_scores.get)
            
            return {
                "severity": {
                    "predicted": predicted_severity,
                    "confidence": severity_scores[predicted_severity] / max(sum(severity_scores.values()), 1),
                    "scores": severity_scores
                },
                "category": {
                    "predicted": predicted_category,
                    "confidence": category_scores[predicted_category] / max(sum(category_scores.values()), 1),
                    "scores": category_scores
                }
            }
            
        except Exception as e:
            logger.error(f"Error in bug classification: {str(e)}")
            return {}
    
    async def _identify_components(self, bug_report: BugReport) -> Dict[str, Any]:
        """Identify affected system components"""
        try:
            combined_text = f"{bug_report.title} {bug_report.description}".lower()
            
            # Component keywords
            component_keywords = {
                ComponentType.FRONTEND: [
                    "frontend", "ui", "interface", "browser", "client",
                    "react", "vue", "angular", "javascript", "css", "html"
                ],
                ComponentType.BACKEND: [
                    "backend", "server", "api", "service", "application",
                    "python", "java", "node", "php", "ruby", "go"
                ],
                ComponentType.DATABASE: [
                    "database", "db", "sql", "query", "table", "record",
                    "mysql", "postgresql", "mongodb", "redis", "elasticsearch"
                ],
                ComponentType.API: [
                    "api", "endpoint", "rest", "graphql", "webhook",
                    "request", "response", "json", "xml"
                ],
                ComponentType.INFRASTRUCTURE: [
                    "infrastructure", "server", "cloud", "aws", "azure",
                    "docker", "kubernetes", "deployment", "network"
                ],
                ComponentType.THIRD_PARTY: [
                    "third party", "external", "integration", "plugin",
                    "library", "dependency", "package", "module"
                ]
            }
            
            # Calculate component scores
            component_scores = {}
            for component, keywords in component_keywords.items():
                score = sum(1 for keyword in keywords if keyword in combined_text)
                component_scores[component.value] = score
            
            # Identify likely components
            likely_components = [
                component for component, score in component_scores.items()
                if score > 0
            ]
            
            # Use component graph for dependency analysis
            affected_components = set(likely_components)
            for component in likely_components:
                if component in self.component_graph:
                    # Add dependent components
                    dependencies = list(self.component_graph.successors(component))
                    affected_components.update(dependencies)
            
            return {
                "primary_components": likely_components,
                "affected_components": list(affected_components),
                "component_scores": component_scores,
                "dependency_analysis": {
                    component: list(self.component_graph.successors(component))
                    for component in likely_components
                    if component in self.component_graph
                }
            }
            
        except Exception as e:
            logger.error(f"Error in component identification: {str(e)}")
            return {}
    
    async def _perform_root_cause_analysis(self, bug_report: BugReport) -> RootCauseAnalysis:
        """Perform root cause analysis"""
        try:
            combined_text = f"{bug_report.title} {bug_report.description}"
            
            # Pattern matching for known issues
            matched_patterns = []
            for pattern_name, pattern_info in self.pattern_database.items():
                for pattern in pattern_info["patterns"]:
                    if re.search(pattern, combined_text, re.IGNORECASE):
                        matched_patterns.append({
                            "pattern": pattern_name,
                            "info": pattern_info
                        })
            
            # Determine primary cause
            if matched_patterns:
                primary_pattern = matched_patterns[0]
                primary_cause = primary_pattern["pattern"]
                contributing_factors = primary_pattern["info"]["common_causes"]
            else:
                primary_cause = "Unknown - requires manual investigation"
                contributing_factors = ["Insufficient information for automated analysis"]
            
            # Extract potential code locations
            code_patterns = [
                r"at\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(",  # Stack trace
                r"in\s+([a-zA-Z_][a-zA-Z0-9_./]*\.py)",   # Python files
                r"([a-zA-Z_][a-zA-Z0-9_./]*\.java)",      # Java files
                r"([a-zA-Z_][a-zA-Z0-9_./]*\.js)",        # JavaScript files
            ]
            
            code_locations = []
            for pattern in code_patterns:
                matches = re.findall(pattern, combined_text)
                code_locations.extend(matches)
            
            # Extract log patterns
            log_patterns = [
                r"ERROR.*",
                r"FATAL.*",
                r"Exception.*",
                r"Error.*",
                r"\d{4}-\d{2}-\d{2}.*ERROR.*"
            ]
            
            log_entries = []
            for pattern in log_patterns:
                matches = re.findall(pattern, combined_text, re.MULTILINE)
                log_entries.extend(matches)
            
            # Calculate confidence score
            confidence_score = 0.8 if matched_patterns else 0.3
            if code_locations:
                confidence_score += 0.1
            if log_entries:
                confidence_score += 0.1
            
            confidence_score = min(confidence_score, 1.0)
            
            return RootCauseAnalysis(
                primary_cause=primary_cause,
                contributing_factors=contributing_factors,
                affected_components=[],  # Will be filled by component analysis
                code_locations=code_locations[:10],  # Limit to top 10
                log_patterns=log_entries[:5],  # Limit to top 5
                confidence_score=confidence_score,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in root cause analysis: {str(e)}")
            return RootCauseAnalysis(
                primary_cause="Analysis failed",
                contributing_factors=[str(e)],
                affected_components=[],
                code_locations=[],
                log_patterns=[],
                confidence_score=0.0,
                analysis_timestamp=datetime.now()
            )
    
    async def _assess_impact(self, bug_report: BugReport) -> ImpactAssessment:
        """Assess bug impact on users and system"""
        try:
            combined_text = f"{bug_report.title} {bug_report.description}".lower()
            
            # Estimate affected users based on keywords
            user_impact_keywords = {
                "all users": 1000000,
                "all customers": 1000000,
                "everyone": 1000000,
                "many users": 10000,
                "some users": 1000,
                "few users": 100,
                "single user": 1
            }
            
            affected_users = 100  # Default estimate
            for keyword, count in user_impact_keywords.items():
                if keyword in combined_text:
                    affected_users = count
                    break
            
            # Business impact assessment
            business_keywords = {
                "revenue": "High - Revenue Impact",
                "payment": "High - Payment System",
                "security": "Critical - Security Breach",
                "data loss": "Critical - Data Loss",
                "outage": "Critical - Service Outage",
                "performance": "Medium - Performance Impact",
                "ui": "Low - User Interface",
                "cosmetic": "Low - Cosmetic Issue"
            }
            
            business_impact = "Medium - Standard Impact"
            for keyword, impact in business_keywords.items():
                if keyword in combined_text:
                    business_impact = impact
                    break
            
            # System impact assessment
            system_impact = "Localized component impact"
            if any(word in combined_text for word in ["crash", "down", "outage"]):
                system_impact = "System-wide impact"
            elif any(word in combined_text for word in ["slow", "performance"]):
                system_impact = "Performance degradation"
            
            # Performance impact (mock data)
            performance_impact = {
                "response_time_increase": 0.1,
                "cpu_usage_increase": 0.05,
                "memory_usage_increase": 0.02,
                "error_rate_increase": 0.01
            }
            
            # Security implications
            security_implications = []
            security_keywords = [
                "authentication", "authorization", "injection", "xss",
                "csrf", "vulnerability", "exploit", "breach"
            ]
            
            for keyword in security_keywords:
                if keyword in combined_text:
                    security_implications.append(f"Potential {keyword} issue")
            
            # Estimate fix time based on complexity
            complexity_indicators = {
                "simple": timedelta(hours=2),
                "easy": timedelta(hours=4),
                "medium": timedelta(hours=8),
                "complex": timedelta(days=2),
                "critical": timedelta(days=1),
                "urgent": timedelta(hours=6)
            }
            
            estimated_fix_time = timedelta(hours=8)  # Default
            for indicator, time_estimate in complexity_indicators.items():
                if indicator in combined_text:
                    estimated_fix_time = time_estimate
                    break
            
            # Calculate priority score (0-100)
            priority_score = 50  # Base score
            
            # Adjust based on severity
            if "critical" in combined_text:
                priority_score += 30
            elif "high" in combined_text:
                priority_score += 20
            elif "urgent" in combined_text:
                priority_score += 25
            
            # Adjust based on user impact
            if affected_users > 10000:
                priority_score += 20
            elif affected_users > 1000:
                priority_score += 10
            
            priority_score = min(priority_score, 100)
            
            return ImpactAssessment(
                affected_users=affected_users,
                business_impact=business_impact,
                system_impact=system_impact,
                performance_impact=performance_impact,
                security_implications=security_implications,
                estimated_fix_time=estimated_fix_time,
                priority_score=priority_score
            )
            
        except Exception as e:
            logger.error(f"Error in impact assessment: {str(e)}")
            return ImpactAssessment(
                affected_users=0,
                business_impact="Unknown",
                system_impact="Unknown",
                performance_impact={},
                security_implications=[],
                estimated_fix_time=timedelta(hours=8),
                priority_score=50
            )
    
    async def _find_similar_bugs(self, bug_report: BugReport) -> List[SimilarBug]:
        """Find similar bugs from history"""
        try:
            if not self.bug_history:
                return []
            
            current_text = f"{bug_report.title} {bug_report.description}"
            
            # Create text corpus
            texts = [current_text]
            for bug in self.bug_history[-100:]:  # Last 100 bugs
                if "text_analysis" in bug and "keywords" in bug["text_analysis"]:
                    bug_text = " ".join(bug["text_analysis"]["keywords"])
                    texts.append(bug_text)
            
            if len(texts) < 2:
                return []
            
            # Calculate similarity using TF-IDF
            try:
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                
                similar_bugs = []
                for i, score in enumerate(similarity_scores):
                    if score > 0.3:  # Similarity threshold
                        bug_data = self.bug_history[-(len(similarity_scores) - i)]
                        similar_bugs.append(SimilarBug(
                            bug_id=bug_data["bug_id"],
                            similarity_score=float(score),
                            resolution_method="Unknown",  # Would come from resolution tracking
                            fix_time=timedelta(hours=8),  # Default estimate
                            success_rate=0.8  # Default success rate
                        ))
                
                # Sort by similarity score
                similar_bugs.sort(key=lambda x: x.similarity_score, reverse=True)
                return similar_bugs[:5]  # Top 5 similar bugs
                
            except Exception as e:
                logger.warning(f"Error in similarity calculation: {str(e)}")
                return []
            
        except Exception as e:
            logger.error(f"Error finding similar bugs: {str(e)}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        try:
            # Simple keyword extraction using TF-IDF
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Filter common words
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'between'
            }
            
            keywords = [word for word in words if word not in stop_words]
            
            # Return most frequent keywords
            from collections import Counter
            word_counts = Counter(keywords)
            return [word for word, count in word_counts.most_common(10)]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _identify_error_patterns(self, text: str) -> List[str]:
        """Identify error patterns in text"""
        try:
            patterns = []
            
            # Common error patterns
            error_patterns = [
                r"Exception.*",
                r"Error.*",
                r"Failed.*",
                r"Unable.*",
                r"Cannot.*",
                r"Invalid.*",
                r"Null.*",
                r"Undefined.*",
                r"Timeout.*",
                r"Connection.*refused",
                r"Access.*denied",
                r"Permission.*denied"
            ]
            
            for pattern in error_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                patterns.extend(matches)
            
            return patterns[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Error identifying error patterns: {str(e)}")
            return []
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        try:
            if not self.bug_history:
                return {"total_analyzed": 0}
            
            total_analyzed = len(self.bug_history)
            
            # Calculate average analysis time
            analysis_times = [bug.get("analysis_time", 0) for bug in self.bug_history]
            avg_analysis_time = sum(analysis_times) / len(analysis_times) if analysis_times else 0
            
            # Category distribution
            categories = {}
            severities = {}
            
            for bug in self.bug_history:
                if "classification" in bug:
                    category = bug["classification"].get("category", {}).get("predicted", "unknown")
                    severity = bug["classification"].get("severity", {}).get("predicted", "unknown")
                    
                    categories[category] = categories.get(category, 0) + 1
                    severities[severity] = severities.get(severity, 0) + 1
            
            return {
                "total_analyzed": total_analyzed,
                "average_analysis_time": avg_analysis_time,
                "category_distribution": categories,
                "severity_distribution": severities,
                "pattern_database_size": len(self.pattern_database),
                "component_graph_nodes": self.component_graph.number_of_nodes(),
                "component_graph_edges": self.component_graph.number_of_edges()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    async def test_bug_analyzer():
        """Test the bug analyzer"""
        analyzer = IntelligentBugAnalyzer()
        
        # Create test bug report
        bug_report = BugReport(
            id="BUG-001",
            title="Application crashes when user clicks submit button",
            description="""
            When users try to submit the payment form, the application crashes with a NullPointerException.
            This happens consistently for all users trying to make payments.
            The error appears in the logs as:
            java.lang.NullPointerException at com.example.payment.PaymentProcessor.process(PaymentProcessor.java:45)
            
            Steps to reproduce:
            1. Navigate to payment page
            2. Fill in payment details
            3. Click submit button
            4. Application crashes
            
            Expected: Payment should be processed successfully
            Actual: Application crashes with null pointer exception
            
            This is affecting all customers and preventing any payments from being processed.
            """,
            reporter="john.doe@example.com",
            created_at=datetime.now()
        )
        
        # Analyze the bug
        print("Analyzing bug report...")
        analysis = await analyzer.analyze_bug_report(bug_report)
        
        # Print results
        print(f"\nBug Analysis Results for {bug_report.id}:")
        print(f"Analysis Time: {analysis['analysis_time']:.2f} seconds")
        
        if "classification" in analysis:
            classification = analysis["classification"]
            print(f"\nClassification:")
            print(f"  Severity: {classification['severity']['predicted']} (confidence: {classification['severity']['confidence']:.2f})")
            print(f"  Category: {classification['category']['predicted']} (confidence: {classification['category']['confidence']:.2f})")
        
        if "root_cause" in analysis:
            root_cause = analysis["root_cause"]
            print(f"\nRoot Cause Analysis:")
            print(f"  Primary Cause: {root_cause.primary_cause}")
            print(f"  Confidence: {root_cause.confidence_score:.2f}")
            print(f"  Contributing Factors: {', '.join(root_cause.contributing_factors[:3])}")
            if root_cause.code_locations:
                print(f"  Code Locations: {', '.join(root_cause.code_locations[:3])}")
        
        if "impact_assessment" in analysis:
            impact = analysis["impact_assessment"]
            print(f"\nImpact Assessment:")
            print(f"  Affected Users: {impact.affected_users}")
            print(f"  Business Impact: {impact.business_impact}")
            print(f"  Priority Score: {impact.priority_score}")
            print(f"  Estimated Fix Time: {impact.estimated_fix_time}")
        
        if "components" in analysis:
            components = analysis["components"]
            print(f"\nAffected Components:")
            print(f"  Primary: {', '.join(components['primary_components'])}")
            print(f"  All Affected: {', '.join(components['affected_components'])}")
        
        # Get statistics
        stats = analyzer.get_analysis_statistics()
        print(f"\nAnalyzer Statistics:")
        print(f"  Total Analyzed: {stats['total_analyzed']}")
        print(f"  Average Analysis Time: {stats['average_analysis_time']:.2f}s")
    
    # Run the test
    asyncio.run(test_bug_analyzer())

