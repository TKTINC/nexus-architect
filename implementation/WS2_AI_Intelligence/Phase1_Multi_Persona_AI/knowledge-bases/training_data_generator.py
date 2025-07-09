"""
Nexus Architect Training Data Generator
Generates domain-specific training datasets for persona fine-tuning
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import uuid

import yaml
import openai
import anthropic
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

openai.api_key = OPENAI_API_KEY
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

@dataclass
class TrainingExample:
    instruction: str
    input: str
    output: str
    persona: str
    domain: str
    complexity: str
    metadata: Dict[str, Any]

class TrainingDataGenerator:
    def __init__(self):
        self.personas = {
            "security_architect": {
                "name": "Security Architect",
                "domains": [
                    "Threat Modeling and Risk Assessment",
                    "Security Architecture Design",
                    "Compliance Frameworks (NIST, ISO 27001, SOC 2)",
                    "Identity and Access Management",
                    "Cryptography and Data Protection",
                    "Application Security (OWASP)",
                    "Network Security and Firewalls",
                    "Cloud Security and DevSecOps",
                    "Incident Response and Forensics",
                    "Security Monitoring and SIEM"
                ],
                "scenarios": [
                    "threat_modeling", "security_review", "compliance_assessment",
                    "vulnerability_analysis", "security_architecture", "incident_response"
                ]
            },
            "performance_engineer": {
                "name": "Performance Engineer",
                "domains": [
                    "System Performance Analysis",
                    "Database Performance Tuning",
                    "Caching Strategies",
                    "Load Testing and Capacity Planning",
                    "Application Performance Monitoring",
                    "Scalability Architecture",
                    "Memory and CPU Optimization",
                    "Network Performance",
                    "Frontend Performance",
                    "Cloud Performance Optimization"
                ],
                "scenarios": [
                    "performance_analysis", "optimization_strategy", "capacity_planning",
                    "bottleneck_identification", "monitoring_setup", "scalability_design"
                ]
            },
            "application_architect": {
                "name": "Application Architect",
                "domains": [
                    "Software Architecture Patterns",
                    "Microservices Design",
                    "API Design and Integration",
                    "Domain-Driven Design",
                    "Event-Driven Architecture",
                    "System Design and Integration",
                    "Technology Stack Evaluation",
                    "Code Quality and Best Practices",
                    "Architecture Governance",
                    "Cloud-Native Patterns"
                ],
                "scenarios": [
                    "architecture_design", "pattern_selection", "technology_evaluation",
                    "system_integration", "code_review", "migration_strategy"
                ]
            },
            "devops_specialist": {
                "name": "DevOps Specialist",
                "domains": [
                    "CI/CD Pipeline Design",
                    "Infrastructure as Code",
                    "Container Orchestration",
                    "Monitoring and Observability",
                    "Site Reliability Engineering",
                    "Configuration Management",
                    "Security Integration (DevSecOps)",
                    "Cloud Infrastructure",
                    "Automation and Scripting",
                    "Incident Response"
                ],
                "scenarios": [
                    "pipeline_optimization", "infrastructure_automation", "monitoring_setup",
                    "deployment_strategy", "operational_excellence", "troubleshooting"
                ]
            },
            "compliance_auditor": {
                "name": "Compliance Auditor",
                "domains": [
                    "GDPR Data Protection",
                    "HIPAA Healthcare Compliance",
                    "SOX Financial Controls",
                    "ISO 27001 Security Standards",
                    "SOC 2 Service Controls",
                    "PCI DSS Payment Security",
                    "Risk Assessment and Management",
                    "Audit Preparation",
                    "Policy Development",
                    "Regulatory Reporting"
                ],
                "scenarios": [
                    "compliance_assessment", "audit_preparation", "risk_analysis",
                    "policy_review", "control_implementation", "regulatory_guidance"
                ]
            }
        }
        
        self.complexity_levels = ["simple", "moderate", "complex", "expert"]
        self.output_dir = Path("/opt/ml/datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_all_training_data(self, examples_per_persona: int = 500):
        """Generate training data for all personas"""
        logger.info(f"Starting training data generation for {len(self.personas)} personas")
        
        for persona_id, persona_config in self.personas.items():
            logger.info(f"Generating training data for {persona_config['name']}")
            await self.generate_persona_training_data(persona_id, examples_per_persona)
        
        logger.info("Training data generation completed")
    
    async def generate_persona_training_data(self, persona_id: str, num_examples: int):
        """Generate training data for a specific persona"""
        persona_config = self.personas[persona_id]
        training_examples = []
        
        examples_per_domain = num_examples // len(persona_config["domains"])
        
        for domain in persona_config["domains"]:
            logger.info(f"Generating examples for {persona_id} - {domain}")
            
            for scenario in persona_config["scenarios"]:
                examples_per_scenario = examples_per_domain // len(persona_config["scenarios"])
                
                for complexity in self.complexity_levels:
                    examples_per_complexity = max(1, examples_per_scenario // len(self.complexity_levels))
                    
                    for _ in range(examples_per_complexity):
                        try:
                            example = await self.generate_training_example(
                                persona_id, domain, scenario, complexity
                            )
                            training_examples.append(example)
                        except Exception as e:
                            logger.error(f"Error generating example: {e}")
                            continue
        
        # Save training data
        await self.save_training_data(persona_id, training_examples)
        logger.info(f"Generated {len(training_examples)} training examples for {persona_id}")
    
    async def generate_training_example(self, persona_id: str, domain: str, scenario: str, complexity: str) -> TrainingExample:
        """Generate a single training example"""
        persona_config = self.personas[persona_id]
        
        # Generate instruction and input
        instruction_prompt = f"""
        Generate a realistic instruction for a {persona_config['name']} in the domain of {domain}.
        The scenario is {scenario} with {complexity} complexity level.
        The instruction should be professional, specific, and require domain expertise.
        Return only the instruction text, no additional formatting.
        """
        
        instruction = await self._call_gpt4(instruction_prompt)
        
        # Generate input context
        input_prompt = f"""
        For the instruction: "{instruction}"
        Generate a realistic input context that a {persona_config['name']} would receive.
        This should include relevant technical details, constraints, and background information.
        The complexity level is {complexity}.
        Return only the input context, no additional formatting.
        """
        
        input_text = await self._call_gpt4(input_prompt)
        
        # Generate expert output
        output_prompt = f"""
        You are a senior {persona_config['name']} with 15+ years of experience in {domain}.
        
        Instruction: {instruction}
        Input: {input_text}
        
        Provide a comprehensive, expert-level response that demonstrates deep domain knowledge.
        Include specific recommendations, best practices, and implementation guidance.
        The response should be authoritative and actionable.
        Complexity level: {complexity}
        """
        
        output = await self._call_gpt4(output_prompt, max_tokens=1024)
        
        return TrainingExample(
            instruction=instruction.strip(),
            input=input_text.strip(),
            output=output.strip(),
            persona=persona_id,
            domain=domain,
            complexity=complexity,
            metadata={
                "scenario": scenario,
                "generated_at": datetime.utcnow().isoformat(),
                "generator_version": "1.0.0"
            }
        )
    
    async def _call_gpt4(self, prompt: str, max_tokens: int = 512) -> str:
        """Call GPT-4 API"""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            raise
    
    async def save_training_data(self, persona_id: str, examples: List[TrainingExample]):
        """Save training data in JSONL format for fine-tuning"""
        output_file = self.output_dir / f"{persona_id}_training.jsonl"
        
        with open(output_file, 'w') as f:
            for example in examples:
                # Format for fine-tuning
                training_record = {
                    "messages": [
                        {"role": "system", "content": f"You are a {self.personas[persona_id]['name']} with expertise in {example.domain}."},
                        {"role": "user", "content": f"{example.instruction}\n\nContext: {example.input}"},
                        {"role": "assistant", "content": example.output}
                    ],
                    "metadata": example.metadata
                }
                f.write(json.dumps(training_record) + '\n')
        
        logger.info(f"Saved training data to {output_file}")
    
    async def generate_knowledge_base_content(self, persona_id: str):
        """Generate knowledge base content for persona"""
        persona_config = self.personas[persona_id]
        knowledge_base = {}
        
        for domain in persona_config["domains"]:
            logger.info(f"Generating knowledge base content for {domain}")
            
            # Generate comprehensive domain knowledge
            knowledge_prompt = f"""
            Create comprehensive knowledge base content for a {persona_config['name']} 
            in the domain of {domain}.
            
            Include:
            1. Key concepts and definitions
            2. Best practices and methodologies
            3. Common patterns and anti-patterns
            4. Tools and technologies
            5. Industry standards and frameworks
            6. Troubleshooting guides
            7. Implementation examples
            
            Format as structured content with clear sections.
            """
            
            content = await self._call_gpt4(knowledge_prompt, max_tokens=2048)
            knowledge_base[domain] = content
        
        # Save knowledge base
        kb_file = self.output_dir / f"{persona_id}_knowledge_base.json"
        with open(kb_file, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
        
        logger.info(f"Generated knowledge base for {persona_id}")
    
    async def create_validation_dataset(self, persona_id: str, num_examples: int = 100):
        """Create validation dataset for persona evaluation"""
        persona_config = self.personas[persona_id]
        validation_examples = []
        
        for domain in persona_config["domains"][:3]:  # Limit to top 3 domains
            for complexity in ["moderate", "complex"]:
                for _ in range(num_examples // 6):  # Distribute across domains and complexity
                    try:
                        # Generate validation question
                        question_prompt = f"""
                        Generate a challenging validation question for a {persona_config['name']} 
                        in the domain of {domain} with {complexity} complexity.
                        The question should test deep domain knowledge and practical application.
                        Return only the question, no additional formatting.
                        """
                        
                        question = await self._call_gpt4(question_prompt)
                        
                        # Generate expert answer
                        answer_prompt = f"""
                        You are a senior {persona_config['name']} with 15+ years of experience.
                        Answer this question with expert-level detail and practical guidance:
                        
                        {question}
                        
                        Provide a comprehensive answer that demonstrates deep domain expertise.
                        """
                        
                        answer = await self._call_gpt4(answer_prompt, max_tokens=1024)
                        
                        validation_examples.append({
                            "question": question.strip(),
                            "expert_answer": answer.strip(),
                            "domain": domain,
                            "complexity": complexity,
                            "persona": persona_id
                        })
                    except Exception as e:
                        logger.error(f"Error generating validation example: {e}")
                        continue
        
        # Save validation dataset
        validation_file = self.output_dir / f"{persona_id}_validation.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_examples, f, indent=2)
        
        logger.info(f"Created validation dataset for {persona_id} with {len(validation_examples)} examples")

class PersonaKnowledgeBase:
    def __init__(self, persona_id: str):
        self.persona_id = persona_id
        self.knowledge_base = {}
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load knowledge base from file"""
        kb_file = Path(f"/opt/ml/datasets/{self.persona_id}_knowledge_base.json")
        if kb_file.exists():
            with open(kb_file, 'r') as f:
                self.knowledge_base = json.load(f)
    
    def search_knowledge(self, query: str, domain: str = None) -> List[str]:
        """Search knowledge base for relevant information"""
        results = []
        
        search_domains = [domain] if domain else self.knowledge_base.keys()
        
        for domain_name in search_domains:
            if domain_name in self.knowledge_base:
                content = self.knowledge_base[domain_name]
                if any(term.lower() in content.lower() for term in query.split()):
                    results.append(content)
        
        return results
    
    def get_domain_expertise(self, domain: str) -> str:
        """Get expertise content for specific domain"""
        return self.knowledge_base.get(domain, "")

async def main():
    """Main function to generate all training data"""
    generator = TrainingDataGenerator()
    
    # Generate training data for all personas
    await generator.generate_all_training_data(examples_per_persona=500)
    
    # Generate knowledge bases
    for persona_id in generator.personas.keys():
        await generator.generate_knowledge_base_content(persona_id)
        await generator.create_validation_dataset(persona_id)
    
    logger.info("All training data generation completed")

if __name__ == "__main__":
    asyncio.run(main())

