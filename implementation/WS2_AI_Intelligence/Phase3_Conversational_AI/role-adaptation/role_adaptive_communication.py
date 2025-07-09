"""
Nexus Architect Role-Adaptive Communication System

This module provides role-adaptive communication capabilities that adjust
language, content depth, and presentation style based on the user's role
and expertise level.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from jinja2 import Template, Environment, BaseLoader
import re

logger = logging.getLogger(__name__)

class UserRole(str, Enum):
    """User roles for adaptive communication"""
    EXECUTIVE = "executive"
    DEVELOPER = "developer"
    PROJECT_MANAGER = "project_manager"
    PRODUCT_LEADER = "product_leader"
    ARCHITECT = "architect"
    DEVOPS_ENGINEER = "devops_engineer"
    SECURITY_ENGINEER = "security_engineer"
    QA_ENGINEER = "qa_engineer"
    BUSINESS_ANALYST = "business_analyst"
    STAKEHOLDER = "stakeholder"

class ExpertiseLevel(str, Enum):
    """Expertise levels for content adaptation"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class CommunicationStyle(str, Enum):
    """Communication styles"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    BUSINESS = "business"
    EDUCATIONAL = "educational"

class ContentType(str, Enum):
    """Types of content to generate"""
    EXPLANATION = "explanation"
    RECOMMENDATION = "recommendation"
    ANALYSIS = "analysis"
    TUTORIAL = "tutorial"
    SUMMARY = "summary"
    ACTION_PLAN = "action_plan"
    RISK_ASSESSMENT = "risk_assessment"
    TECHNICAL_SPEC = "technical_spec"

@dataclass
class UserProfile:
    """User profile for role adaptation"""
    role: UserRole
    expertise_level: ExpertiseLevel
    communication_style: CommunicationStyle
    preferences: Dict[str, Any]
    context: Dict[str, Any] = None

@dataclass
class AdaptationRules:
    """Rules for adapting communication to specific roles"""
    vocabulary_level: str
    technical_depth: str
    focus_areas: List[str]
    avoid_topics: List[str]
    preferred_formats: List[str]
    tone: str
    examples_type: str

@dataclass
class ResponseTemplate:
    """Template for generating role-adapted responses"""
    template_id: str
    content_type: ContentType
    template_text: str
    variables: List[str]
    role_adaptations: Dict[UserRole, Dict[str, Any]]

class RoleAdaptationEngine:
    """Engine for adapting communication based on user roles"""
    
    def __init__(self):
        self.adaptation_rules = self._load_adaptation_rules()
        self.response_templates = self._load_response_templates()
        self.jinja_env = Environment(loader=BaseLoader())
        
    def _load_adaptation_rules(self) -> Dict[UserRole, AdaptationRules]:
        """Load adaptation rules for different user roles"""
        return {
            UserRole.EXECUTIVE: AdaptationRules(
                vocabulary_level="business",
                technical_depth="high_level",
                focus_areas=["business_impact", "roi", "strategic_value", "risk"],
                avoid_topics=["implementation_details", "code_examples"],
                preferred_formats=["executive_summary", "bullet_points", "metrics"],
                tone="professional",
                examples_type="business_cases"
            ),
            UserRole.DEVELOPER: AdaptationRules(
                vocabulary_level="technical",
                technical_depth="detailed",
                focus_areas=["implementation", "code_quality", "best_practices", "tools"],
                avoid_topics=["business_metrics", "high_level_strategy"],
                preferred_formats=["code_examples", "step_by_step", "technical_docs"],
                tone="collaborative",
                examples_type="code_samples"
            ),
            UserRole.PROJECT_MANAGER: AdaptationRules(
                vocabulary_level="balanced",
                technical_depth="moderate",
                focus_areas=["timeline", "resources", "risks", "dependencies"],
                avoid_topics=["deep_technical_details"],
                preferred_formats=["project_plans", "timelines", "risk_matrices"],
                tone="organized",
                examples_type="project_scenarios"
            ),
            UserRole.PRODUCT_LEADER: AdaptationRules(
                vocabulary_level="product",
                technical_depth="moderate",
                focus_areas=["user_experience", "features", "market_impact", "feasibility"],
                avoid_topics=["infrastructure_details"],
                preferred_formats=["feature_specs", "user_stories", "roadmaps"],
                tone="strategic",
                examples_type="product_cases"
            ),
            UserRole.ARCHITECT: AdaptationRules(
                vocabulary_level="technical",
                technical_depth="architectural",
                focus_areas=["system_design", "patterns", "scalability", "integration"],
                avoid_topics=["business_details"],
                preferred_formats=["architecture_diagrams", "design_docs", "patterns"],
                tone="analytical",
                examples_type="architecture_patterns"
            ),
            UserRole.DEVOPS_ENGINEER: AdaptationRules(
                vocabulary_level="technical",
                technical_depth="operational",
                focus_areas=["deployment", "monitoring", "automation", "infrastructure"],
                avoid_topics=["business_strategy"],
                preferred_formats=["scripts", "configs", "runbooks"],
                tone="practical",
                examples_type="operational_procedures"
            ),
            UserRole.SECURITY_ENGINEER: AdaptationRules(
                vocabulary_level="security",
                technical_depth="security_focused",
                focus_areas=["threats", "vulnerabilities", "compliance", "controls"],
                avoid_topics=["non_security_features"],
                preferred_formats=["security_assessments", "threat_models", "controls"],
                tone="cautious",
                examples_type="security_scenarios"
            )
        }
        
    def _load_response_templates(self) -> Dict[ContentType, List[ResponseTemplate]]:
        """Load response templates for different content types"""
        return {
            ContentType.EXPLANATION: [
                ResponseTemplate(
                    template_id="explanation_executive",
                    content_type=ContentType.EXPLANATION,
                    template_text="""
## {{ topic }}

**Business Impact:** {{ business_impact }}

**Key Points:**
{% for point in key_points %}
- {{ point }}
{% endfor %}

**Recommendation:** {{ recommendation }}

**Next Steps:** {{ next_steps }}
                    """,
                    variables=["topic", "business_impact", "key_points", "recommendation", "next_steps"],
                    role_adaptations={
                        UserRole.EXECUTIVE: {"focus": "business_value", "detail_level": "high_level"}
                    }
                ),
                ResponseTemplate(
                    template_id="explanation_developer",
                    content_type=ContentType.EXPLANATION,
                    template_text="""
# {{ topic }}

## Overview
{{ overview }}

## Technical Details
{{ technical_details }}

## Implementation Example
```{{ language }}
{{ code_example }}
```

## Best Practices
{% for practice in best_practices %}
- {{ practice }}
{% endfor %}

## Related Resources
{% for resource in resources %}
- [{{ resource.title }}]({{ resource.url }})
{% endfor %}
                    """,
                    variables=["topic", "overview", "technical_details", "language", "code_example", "best_practices", "resources"],
                    role_adaptations={
                        UserRole.DEVELOPER: {"focus": "implementation", "detail_level": "detailed"}
                    }
                )
            ],
            ContentType.RECOMMENDATION: [
                ResponseTemplate(
                    template_id="recommendation_pm",
                    content_type=ContentType.RECOMMENDATION,
                    template_text="""
## Recommendation: {{ title }}

**Priority:** {{ priority }}
**Effort:** {{ effort }}
**Timeline:** {{ timeline }}

### Problem Statement
{{ problem }}

### Proposed Solution
{{ solution }}

### Implementation Plan
{% for phase in phases %}
**Phase {{ loop.index }}: {{ phase.name }}** ({{ phase.duration }})
{{ phase.description }}
{% endfor %}

### Risks and Mitigation
{% for risk in risks %}
- **Risk:** {{ risk.description }}
  **Mitigation:** {{ risk.mitigation }}
{% endfor %}

### Success Metrics
{% for metric in metrics %}
- {{ metric }}
{% endfor %}
                    """,
                    variables=["title", "priority", "effort", "timeline", "problem", "solution", "phases", "risks", "metrics"],
                    role_adaptations={
                        UserRole.PROJECT_MANAGER: {"focus": "execution", "detail_level": "project_oriented"}
                    }
                )
            ]
        }

class CommunicationAdapter:
    """Adapts communication style and content based on user profile"""
    
    def __init__(self, adaptation_engine: RoleAdaptationEngine):
        self.adaptation_engine = adaptation_engine
        self.vocabulary_mappings = self._load_vocabulary_mappings()
        
    def _load_vocabulary_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load vocabulary mappings for different roles"""
        return {
            "technical_to_business": {
                "API": "interface",
                "microservices": "modular services",
                "latency": "response time",
                "throughput": "processing capacity",
                "scalability": "ability to handle growth",
                "refactoring": "code improvement",
                "deployment": "release",
                "infrastructure": "technology foundation"
            },
            "business_to_technical": {
                "ROI": "return on investment",
                "KPI": "key performance indicator",
                "stakeholder": "interested party",
                "deliverable": "output",
                "milestone": "checkpoint",
                "scope": "project boundaries"
            }
        }
        
    async def adapt_content(
        self,
        content: str,
        user_profile: UserProfile,
        content_type: ContentType
    ) -> str:
        """Adapt content based on user profile"""
        
        # Get adaptation rules for user role
        rules = self.adaptation_engine.adaptation_rules.get(user_profile.role)
        if not rules:
            return content
            
        # Apply vocabulary adaptation
        adapted_content = self._adapt_vocabulary(content, user_profile.role)
        
        # Apply technical depth adaptation
        adapted_content = self._adapt_technical_depth(adapted_content, rules.technical_depth)
        
        # Apply tone adaptation
        adapted_content = self._adapt_tone(adapted_content, rules.tone)
        
        # Apply format adaptation
        adapted_content = self._adapt_format(adapted_content, rules.preferred_formats, content_type)
        
        return adapted_content
        
    def _adapt_vocabulary(self, content: str, role: UserRole) -> str:
        """Adapt vocabulary based on user role"""
        if role == UserRole.EXECUTIVE:
            # Convert technical terms to business terms
            for tech_term, business_term in self.vocabulary_mappings["technical_to_business"].items():
                content = re.sub(r'\b' + re.escape(tech_term) + r'\b', business_term, content, flags=re.IGNORECASE)
        elif role == UserRole.DEVELOPER:
            # Keep technical terms, possibly add more detail
            pass
        elif role == UserRole.PROJECT_MANAGER:
            # Balance technical and business terms
            pass
            
        return content
        
    def _adapt_technical_depth(self, content: str, depth_level: str) -> str:
        """Adapt technical depth based on requirements"""
        if depth_level == "high_level":
            # Remove detailed technical explanations
            content = re.sub(r'```[\s\S]*?```', '[Code example available]', content)
            content = re.sub(r'Technical details:.*?(?=\n\n|\n#|\Z)', '', content, flags=re.DOTALL)
        elif depth_level == "detailed":
            # Keep all technical details
            pass
        elif depth_level == "moderate":
            # Keep some technical details but simplify
            pass
            
        return content
        
    def _adapt_tone(self, content: str, tone: str) -> str:
        """Adapt tone of communication"""
        tone_patterns = {
            "professional": {
                "casual_phrases": {
                    r"\bguys\b": "team members",
                    r"\bokay\b": "acceptable",
                    r"\bawesome\b": "excellent"
                }
            },
            "collaborative": {
                "formal_phrases": {
                    r"\byou must\b": "you should consider",
                    r"\bit is required\b": "it would be beneficial"
                }
            }
        }
        
        if tone in tone_patterns:
            for pattern, replacement in tone_patterns[tone].get("casual_phrases", {}).items():
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            for pattern, replacement in tone_patterns[tone].get("formal_phrases", {}).items():
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                
        return content
        
    def _adapt_format(self, content: str, preferred_formats: List[str], content_type: ContentType) -> str:
        """Adapt format based on preferences"""
        if "bullet_points" in preferred_formats and content_type == ContentType.SUMMARY:
            # Convert paragraphs to bullet points
            paragraphs = content.split('\n\n')
            bullet_points = []
            for para in paragraphs:
                if para.strip() and not para.startswith('#'):
                    bullet_points.append(f"• {para.strip()}")
                else:
                    bullet_points.append(para)
            content = '\n'.join(bullet_points)
            
        elif "executive_summary" in preferred_formats:
            # Add executive summary format
            if not content.startswith("## Executive Summary"):
                content = "## Executive Summary\n\n" + content
                
        return content

class ResponseGenerator:
    """Generates role-adapted responses using templates"""
    
    def __init__(self, adaptation_engine: RoleAdaptationEngine):
        self.adaptation_engine = adaptation_engine
        self.communication_adapter = CommunicationAdapter(adaptation_engine)
        
    async def generate_response(
        self,
        content_type: ContentType,
        user_profile: UserProfile,
        context_data: Dict[str, Any]
    ) -> str:
        """Generate a role-adapted response"""
        
        # Get appropriate template
        template = self._select_template(content_type, user_profile.role)
        if not template:
            return self._generate_fallback_response(context_data)
            
        # Prepare template variables
        template_vars = self._prepare_template_variables(template, context_data, user_profile)
        
        # Render template
        jinja_template = self.adaptation_engine.jinja_env.from_string(template.template_text)
        rendered_content = jinja_template.render(**template_vars)
        
        # Apply additional adaptations
        adapted_content = await self.communication_adapter.adapt_content(
            rendered_content, user_profile, content_type
        )
        
        return adapted_content.strip()
        
    def _select_template(self, content_type: ContentType, role: UserRole) -> Optional[ResponseTemplate]:
        """Select appropriate template based on content type and role"""
        templates = self.adaptation_engine.response_templates.get(content_type, [])
        
        # Find template with role-specific adaptation
        for template in templates:
            if role in template.role_adaptations:
                return template
                
        # Fallback to first template of the content type
        return templates[0] if templates else None
        
    def _prepare_template_variables(
        self,
        template: ResponseTemplate,
        context_data: Dict[str, Any],
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Prepare variables for template rendering"""
        template_vars = {}
        
        # Map context data to template variables
        for var in template.variables:
            if var in context_data:
                template_vars[var] = context_data[var]
            else:
                template_vars[var] = f"[{var} not provided]"
                
        # Apply role-specific adaptations
        role_adaptations = template.role_adaptations.get(user_profile.role, {})
        template_vars.update(role_adaptations)
        
        return template_vars
        
    def _generate_fallback_response(self, context_data: Dict[str, Any]) -> str:
        """Generate fallback response when no template is available"""
        return f"I understand you're asking about {context_data.get('topic', 'this topic')}. Let me provide you with relevant information based on your role and requirements."

class MultiModalResponseGenerator:
    """Generates responses with multiple modalities (text, code, visualizations)"""
    
    def __init__(self, response_generator: ResponseGenerator):
        self.response_generator = response_generator
        
    async def generate_multimodal_response(
        self,
        content_type: ContentType,
        user_profile: UserProfile,
        context_data: Dict[str, Any],
        include_code: bool = False,
        include_diagrams: bool = False,
        include_metrics: bool = False
    ) -> Dict[str, Any]:
        """Generate response with multiple modalities"""
        
        # Generate base text response
        text_response = await self.response_generator.generate_response(
            content_type, user_profile, context_data
        )
        
        response = {
            "text": text_response,
            "modalities": []
        }
        
        # Add code examples if requested and appropriate for role
        if include_code and self._should_include_code(user_profile.role):
            code_example = self._generate_code_example(context_data, user_profile)
            if code_example:
                response["modalities"].append({
                    "type": "code",
                    "content": code_example
                })
                
        # Add diagrams if requested and appropriate
        if include_diagrams and self._should_include_diagrams(user_profile.role):
            diagram_spec = self._generate_diagram_spec(context_data, user_profile)
            if diagram_spec:
                response["modalities"].append({
                    "type": "diagram",
                    "content": diagram_spec
                })
                
        # Add metrics if requested and appropriate
        if include_metrics and self._should_include_metrics(user_profile.role):
            metrics = self._generate_metrics(context_data, user_profile)
            if metrics:
                response["modalities"].append({
                    "type": "metrics",
                    "content": metrics
                })
                
        return response
        
    def _should_include_code(self, role: UserRole) -> bool:
        """Determine if code examples should be included for this role"""
        return role in [UserRole.DEVELOPER, UserRole.ARCHITECT, UserRole.DEVOPS_ENGINEER]
        
    def _should_include_diagrams(self, role: UserRole) -> bool:
        """Determine if diagrams should be included for this role"""
        return role in [UserRole.ARCHITECT, UserRole.PROJECT_MANAGER, UserRole.PRODUCT_LEADER]
        
    def _should_include_metrics(self, role: UserRole) -> bool:
        """Determine if metrics should be included for this role"""
        return role in [UserRole.EXECUTIVE, UserRole.PROJECT_MANAGER, UserRole.PRODUCT_LEADER]
        
    def _generate_code_example(self, context_data: Dict[str, Any], user_profile: UserProfile) -> Optional[Dict[str, str]]:
        """Generate appropriate code example"""
        topic = context_data.get("topic", "")
        
        if "authentication" in topic.lower():
            return {
                "language": "python",
                "code": """
# OAuth 2.0 Authentication Example
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Validate token and return user
    user = validate_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.get("/protected")
async def protected_route(current_user = Depends(get_current_user)):
    return {"message": f"Hello {current_user.username}"}
                """,
                "description": "FastAPI OAuth 2.0 implementation example"
            }
            
        return None
        
    def _generate_diagram_spec(self, context_data: Dict[str, Any], user_profile: UserProfile) -> Optional[Dict[str, Any]]:
        """Generate diagram specification"""
        topic = context_data.get("topic", "")
        
        if "architecture" in topic.lower():
            return {
                "type": "architecture_diagram",
                "components": ["API Gateway", "Microservices", "Database", "Cache"],
                "connections": [
                    {"from": "API Gateway", "to": "Microservices"},
                    {"from": "Microservices", "to": "Database"},
                    {"from": "Microservices", "to": "Cache"}
                ],
                "description": "High-level system architecture"
            }
            
        return None
        
    def _generate_metrics(self, context_data: Dict[str, Any], user_profile: UserProfile) -> Optional[Dict[str, Any]]:
        """Generate relevant metrics"""
        return {
            "performance_metrics": {
                "response_time": "< 200ms",
                "throughput": "1000 req/sec",
                "availability": "99.9%"
            },
            "business_metrics": {
                "cost_reduction": "30%",
                "time_to_market": "2 weeks faster",
                "user_satisfaction": "4.5/5"
            }
        }

# Example usage and testing
async def main():
    """Example usage of role-adaptive communication"""
    adaptation_engine = RoleAdaptationEngine()
    response_generator = ResponseGenerator(adaptation_engine)
    multimodal_generator = MultiModalResponseGenerator(response_generator)
    
    # Test different user profiles
    profiles = [
        UserProfile(
            role=UserRole.EXECUTIVE,
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            communication_style=CommunicationStyle.BUSINESS,
            preferences={"format": "executive_summary"}
        ),
        UserProfile(
            role=UserRole.DEVELOPER,
            expertise_level=ExpertiseLevel.ADVANCED,
            communication_style=CommunicationStyle.TECHNICAL,
            preferences={"include_code": True}
        ),
        UserProfile(
            role=UserRole.PROJECT_MANAGER,
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            communication_style=CommunicationStyle.FORMAL,
            preferences={"format": "project_plan"}
        )
    ]
    
    context_data = {
        "topic": "OAuth Authentication Implementation",
        "business_impact": "Improved security and user experience",
        "key_points": [
            "Centralized authentication",
            "Single sign-on capability",
            "Enhanced security"
        ],
        "recommendation": "Implement OAuth 2.0 with PKCE",
        "next_steps": "Begin with pilot implementation"
    }
    
    for profile in profiles:
        print(f"\n{'='*50}")
        print(f"Response for {profile.role.value}:")
        print(f"{'='*50}")
        
        response = await multimodal_generator.generate_multimodal_response(
            ContentType.EXPLANATION,
            profile,
            context_data,
            include_code=(profile.role == UserRole.DEVELOPER),
            include_diagrams=(profile.role in [UserRole.ARCHITECT, UserRole.PROJECT_MANAGER]),
            include_metrics=(profile.role == UserRole.EXECUTIVE)
        )
        
        print(response["text"])
        
        if response["modalities"]:
            print("\nAdditional Content:")
            for modality in response["modalities"]:
                print(f"- {modality['type']}: {modality['content']}")

if __name__ == "__main__":
    asyncio.run(main())

