"""
Nexus Architect Model Management System

This module implements comprehensive model lifecycle management including
versioning, deployment, monitoring, and automated rollback capabilities.
"""

import asyncio
import json
import time
import logging
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pickle
import joblib

import redis.asyncio as redis
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from kubernetes import client, config
import docker

logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    """Types of models"""
    LANGUAGE_MODEL = "language_model"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    EMBEDDING = "embedding"
    GENERATIVE = "generative"
    ENSEMBLE = "ensemble"

class ModelStatus(str, Enum):
    """Model deployment status"""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class DeploymentStrategy(str, Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    IMMEDIATE = "immediate"

class ModelFramework(str, Enum):
    """Model frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    TRANSFORMERS = "transformers"
    CUSTOM = "custom"

@dataclass
class ModelMetadata:
    """Model metadata"""
    model_id: str
    model_name: str
    version: str
    model_type: ModelType
    framework: ModelFramework
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    parameters: Dict[str, Any]
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    dependencies: List[str]
    model_size_mb: float
    checksum: str

@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_id: str
    version: str
    status: ModelStatus
    deployment_strategy: DeploymentStrategy
    model_path: str
    config_path: str
    performance_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    deployment_config: Dict[str, Any]
    created_at: datetime
    deployed_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    deployment_id: str
    model_version_id: str
    strategy: DeploymentStrategy
    target_environment: str
    resource_requirements: Dict[str, Any]
    scaling_config: Dict[str, Any]
    health_check_config: Dict[str, Any]
    rollback_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    created_at: datetime

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    performance_id: str
    model_version_id: str
    metric_name: str
    metric_value: float
    measurement_time: datetime
    environment: str
    data_sample_size: int
    metadata: Dict[str, Any]

@dataclass
class ModelComparison:
    """Model comparison results"""
    comparison_id: str
    model_a_id: str
    model_b_id: str
    comparison_metrics: Dict[str, Dict[str, float]]
    winner: str
    confidence: float
    test_dataset_info: Dict[str, Any]
    comparison_time: datetime

# Database models
Base = declarative_base()

class ModelMetadataDB(Base):
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True)
    model_id = Column(String(255), unique=True, nullable=False)
    model_name = Column(String(255), nullable=False)
    version = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    framework = Column(String(50), nullable=False)
    description = Column(Text)
    author = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    tags = Column(Text)
    parameters = Column(Text)
    training_config = Column(Text)
    performance_metrics = Column(Text)
    dependencies = Column(Text)
    model_size_mb = Column(Float)
    checksum = Column(String(255))

class ModelVersionDB(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True)
    version_id = Column(String(255), unique=True, nullable=False)
    model_id = Column(String(255), nullable=False)
    version = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    deployment_strategy = Column(String(50), nullable=False)
    model_path = Column(String(500), nullable=False)
    config_path = Column(String(500), nullable=False)
    performance_metrics = Column(Text)
    validation_results = Column(Text)
    deployment_config = Column(Text)
    created_at = Column(DateTime, nullable=False)
    deployed_at = Column(DateTime)
    deprecated_at = Column(DateTime)

class DeploymentConfigDB(Base):
    __tablename__ = "deployment_configs"
    
    id = Column(Integer, primary_key=True)
    deployment_id = Column(String(255), unique=True, nullable=False)
    model_version_id = Column(String(255), nullable=False)
    strategy = Column(String(50), nullable=False)
    target_environment = Column(String(100), nullable=False)
    resource_requirements = Column(Text)
    scaling_config = Column(Text)
    health_check_config = Column(Text)
    rollback_config = Column(Text)
    monitoring_config = Column(Text)
    created_at = Column(DateTime, nullable=False)

class ModelPerformanceDB(Base):
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True)
    performance_id = Column(String(255), unique=True, nullable=False)
    model_version_id = Column(String(255), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    measurement_time = Column(DateTime, nullable=False)
    environment = Column(String(100), nullable=False)
    data_sample_size = Column(Integer)
    metadata = Column(Text)

class ModelComparisonDB(Base):
    __tablename__ = "model_comparisons"
    
    id = Column(Integer, primary_key=True)
    comparison_id = Column(String(255), unique=True, nullable=False)
    model_a_id = Column(String(255), nullable=False)
    model_b_id = Column(String(255), nullable=False)
    comparison_metrics = Column(Text, nullable=False)
    winner = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    test_dataset_info = Column(Text)
    comparison_time = Column(DateTime, nullable=False)

class ModelValidator:
    """Model validation and testing"""
    
    def __init__(self):
        self.validation_metrics = {
            'classification': ['accuracy', 'precision', 'recall', 'f1_score'],
            'regression': ['mse', 'mae', 'r2_score'],
            'language_model': ['perplexity', 'bleu_score', 'rouge_score'],
            'embedding': ['cosine_similarity', 'clustering_score']
        }
        
    async def validate_model(self, 
                           model_path: str, 
                           model_type: ModelType,
                           test_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate model performance"""
        
        try:
            # Load model based on framework
            model = await self._load_model(model_path)
            
            # Run validation based on model type
            if model_type == ModelType.CLASSIFICATION:
                return await self._validate_classification_model(model, test_data)
            elif model_type == ModelType.REGRESSION:
                return await self._validate_regression_model(model, test_data)
            elif model_type == ModelType.LANGUAGE_MODEL:
                return await self._validate_language_model(model, test_data)
            elif model_type == ModelType.EMBEDDING:
                return await self._validate_embedding_model(model, test_data)
            else:
                return await self._validate_generic_model(model, test_data)
                
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return {"validation_error": 1.0}
            
    async def _load_model(self, model_path: str):
        """Load model from path"""
        model_path = Path(model_path)
        
        if model_path.suffix == '.pkl':
            return joblib.load(model_path)
        elif model_path.suffix == '.pt' or model_path.suffix == '.pth':
            return torch.load(model_path, map_location='cpu')
        elif model_path.is_dir():
            # Assume transformers model
            from transformers import AutoModel
            return AutoModel.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
            
    async def _validate_classification_model(self, model, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate classification model"""
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            # Handle PyTorch models
            model.eval()
            with torch.no_grad():
                if isinstance(X_test, np.ndarray):
                    X_test = torch.FloatTensor(X_test)
                y_pred = model(X_test).argmax(dim=1).numpy()
                
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
        
    async def _validate_regression_model(self, model, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate regression model"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            # Handle PyTorch models
            model.eval()
            with torch.no_grad():
                if isinstance(X_test, np.ndarray):
                    X_test = torch.FloatTensor(X_test)
                y_pred = model(X_test).numpy()
                
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        return metrics
        
    async def _validate_language_model(self, model, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate language model"""
        # Simplified validation for language models
        test_texts = test_data.get('test_texts', [])
        
        if not test_texts:
            return {"perplexity": float('inf')}
            
        # Calculate perplexity (simplified)
        total_log_prob = 0
        total_tokens = 0
        
        for text in test_texts[:100]:  # Limit for performance
            # This is a simplified perplexity calculation
            # In practice, use proper tokenization and model evaluation
            tokens = text.split()
            total_tokens += len(tokens)
            # Placeholder calculation
            total_log_prob += len(tokens) * -2.0  # Simplified
            
        perplexity = np.exp(-total_log_prob / total_tokens) if total_tokens > 0 else float('inf')
        
        return {
            'perplexity': perplexity,
            'avg_tokens_per_text': total_tokens / len(test_texts) if test_texts else 0
        }
        
    async def _validate_embedding_model(self, model, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate embedding model"""
        test_texts = test_data.get('test_texts', [])
        
        if not test_texts:
            return {"embedding_error": 1.0}
            
        # Generate embeddings
        embeddings = []
        for text in test_texts[:100]:  # Limit for performance
            # Simplified embedding generation
            if hasattr(model, 'encode'):
                embedding = model.encode(text)
            else:
                # Placeholder embedding
                embedding = np.random.normal(0, 1, 768)
            embeddings.append(embedding)
            
        embeddings = np.array(embeddings)
        
        # Calculate similarity metrics
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        
        return {
            'avg_cosine_similarity': avg_similarity,
            'embedding_dimension': embeddings.shape[1],
            'embedding_variance': np.var(embeddings)
        }
        
    async def _validate_generic_model(self, model, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Generic model validation"""
        return {
            'model_loaded': 1.0,
            'validation_timestamp': time.time()
        }

class ModelDeployer:
    """Model deployment management"""
    
    def __init__(self):
        self.docker_client = None
        self.k8s_client = None
        
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker client not available: {e}")
            
        try:
            config.load_incluster_config()
            self.k8s_client = client.AppsV1Api()
        except Exception:
            try:
                config.load_kube_config()
                self.k8s_client = client.AppsV1Api()
            except Exception as e:
                logger.warning(f"Kubernetes client not available: {e}")
                
    async def deploy_model(self, 
                         model_version: ModelVersion,
                         deployment_config: DeploymentConfig) -> bool:
        """Deploy model to target environment"""
        
        try:
            if deployment_config.strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._deploy_blue_green(model_version, deployment_config)
            elif deployment_config.strategy == DeploymentStrategy.CANARY:
                return await self._deploy_canary(model_version, deployment_config)
            elif deployment_config.strategy == DeploymentStrategy.ROLLING:
                return await self._deploy_rolling(model_version, deployment_config)
            else:
                return await self._deploy_immediate(model_version, deployment_config)
                
        except Exception as e:
            logger.error(f"Deployment error: {e}")
            return False
            
    async def _deploy_blue_green(self, 
                               model_version: ModelVersion,
                               deployment_config: DeploymentConfig) -> bool:
        """Blue-green deployment"""
        
        if not self.k8s_client:
            logger.error("Kubernetes client not available for blue-green deployment")
            return False
            
        try:
            # Create new deployment (green)
            green_deployment = await self._create_k8s_deployment(
                model_version, 
                deployment_config,
                suffix="green"
            )
            
            # Wait for green deployment to be ready
            await self._wait_for_deployment_ready(green_deployment.metadata.name)
            
            # Run health checks
            if await self._run_health_checks(green_deployment.metadata.name, deployment_config):
                # Switch traffic to green
                await self._switch_service_to_deployment(
                    deployment_config.target_environment,
                    green_deployment.metadata.name
                )
                
                # Clean up old blue deployment
                await self._cleanup_old_deployment(deployment_config.target_environment, "blue")
                
                logger.info(f"Blue-green deployment successful: {model_version.version_id}")
                return True
            else:
                # Rollback - delete green deployment
                await self._delete_deployment(green_deployment.metadata.name)
                logger.error(f"Blue-green deployment failed health checks: {model_version.version_id}")
                return False
                
        except Exception as e:
            logger.error(f"Blue-green deployment error: {e}")
            return False
            
    async def _deploy_canary(self, 
                           model_version: ModelVersion,
                           deployment_config: DeploymentConfig) -> bool:
        """Canary deployment"""
        
        if not self.k8s_client:
            logger.error("Kubernetes client not available for canary deployment")
            return False
            
        try:
            # Create canary deployment with small traffic percentage
            canary_deployment = await self._create_k8s_deployment(
                model_version,
                deployment_config,
                suffix="canary",
                replicas=1  # Start with single replica
            )
            
            # Wait for canary to be ready
            await self._wait_for_deployment_ready(canary_deployment.metadata.name)
            
            # Gradually increase traffic
            traffic_percentages = [10, 25, 50, 75, 100]
            
            for percentage in traffic_percentages:
                # Update traffic split
                await self._update_traffic_split(
                    deployment_config.target_environment,
                    canary_deployment.metadata.name,
                    percentage
                )
                
                # Wait and monitor
                await asyncio.sleep(300)  # 5 minutes
                
                # Check metrics
                if not await self._check_canary_metrics(canary_deployment.metadata.name):
                    # Rollback
                    await self._rollback_canary(deployment_config.target_environment)
                    logger.error(f"Canary deployment failed at {percentage}%: {model_version.version_id}")
                    return False
                    
            # Canary successful, promote to full deployment
            await self._promote_canary(deployment_config.target_environment, canary_deployment.metadata.name)
            
            logger.info(f"Canary deployment successful: {model_version.version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Canary deployment error: {e}")
            return False
            
    async def _deploy_rolling(self, 
                            model_version: ModelVersion,
                            deployment_config: DeploymentConfig) -> bool:
        """Rolling deployment"""
        
        if not self.k8s_client:
            logger.error("Kubernetes client not available for rolling deployment")
            return False
            
        try:
            # Update existing deployment with new image
            deployment_name = f"{deployment_config.target_environment}-deployment"
            
            # Get current deployment
            current_deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace="default"
            )
            
            # Update image
            container_image = await self._build_model_image(model_version)
            current_deployment.spec.template.spec.containers[0].image = container_image
            
            # Apply update
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace="default",
                body=current_deployment
            )
            
            # Wait for rollout to complete
            await self._wait_for_rollout_complete(deployment_name)
            
            # Run health checks
            if await self._run_health_checks(deployment_name, deployment_config):
                logger.info(f"Rolling deployment successful: {model_version.version_id}")
                return True
            else:
                # Rollback
                await self._rollback_deployment(deployment_name)
                logger.error(f"Rolling deployment failed health checks: {model_version.version_id}")
                return False
                
        except Exception as e:
            logger.error(f"Rolling deployment error: {e}")
            return False
            
    async def _deploy_immediate(self, 
                              model_version: ModelVersion,
                              deployment_config: DeploymentConfig) -> bool:
        """Immediate deployment (for development/testing)"""
        
        try:
            # Build and deploy immediately
            if self.docker_client:
                # Docker deployment
                container_image = await self._build_model_image(model_version)
                
                # Stop existing container
                try:
                    existing_container = self.docker_client.containers.get(deployment_config.target_environment)
                    existing_container.stop()
                    existing_container.remove()
                except docker.errors.NotFound:
                    pass
                    
                # Start new container
                container = self.docker_client.containers.run(
                    container_image,
                    name=deployment_config.target_environment,
                    ports={'8000/tcp': 8000},
                    detach=True,
                    environment=deployment_config.resource_requirements
                )
                
                # Wait for container to be ready
                await asyncio.sleep(10)
                
                # Basic health check
                if container.status == 'running':
                    logger.info(f"Immediate deployment successful: {model_version.version_id}")
                    return True
                else:
                    logger.error(f"Immediate deployment failed: {model_version.version_id}")
                    return False
                    
            else:
                # Local deployment (for testing)
                logger.info(f"Local deployment simulated: {model_version.version_id}")
                return True
                
        except Exception as e:
            logger.error(f"Immediate deployment error: {e}")
            return False
            
    async def _build_model_image(self, model_version: ModelVersion) -> str:
        """Build Docker image for model"""
        
        image_name = f"nexus-model:{model_version.version}"
        
        # Create Dockerfile
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY {model_version.model_path} ./model/
COPY {model_version.config_path} ./config/

COPY serve.py .

EXPOSE 8000

CMD ["python", "serve.py"]
"""
        
        # Build image
        if self.docker_client:
            try:
                image = self.docker_client.images.build(
                    fileobj=dockerfile_content.encode(),
                    tag=image_name,
                    rm=True
                )
                return image_name
            except Exception as e:
                logger.error(f"Image build error: {e}")
                return "default-model-image:latest"
        else:
            return "default-model-image:latest"
            
    async def _create_k8s_deployment(self, 
                                   model_version: ModelVersion,
                                   deployment_config: DeploymentConfig,
                                   suffix: str = "",
                                   replicas: int = 3) -> client.V1Deployment:
        """Create Kubernetes deployment"""
        
        deployment_name = f"{deployment_config.target_environment}-{suffix}" if suffix else deployment_config.target_environment
        container_image = await self._build_model_image(model_version)
        
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=deployment_name),
            spec=client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": deployment_name}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": deployment_name}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="model-server",
                                image=container_image,
                                ports=[client.V1ContainerPort(container_port=8000)],
                                resources=client.V1ResourceRequirements(
                                    requests=deployment_config.resource_requirements.get("requests", {}),
                                    limits=deployment_config.resource_requirements.get("limits", {})
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        return self.k8s_client.create_namespaced_deployment(
            namespace="default",
            body=deployment
        )
        
    async def _wait_for_deployment_ready(self, deployment_name: str, timeout: int = 300):
        """Wait for deployment to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.k8s_client.read_namespaced_deployment(
                    name=deployment_name,
                    namespace="default"
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    return True
                    
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error checking deployment status: {e}")
                await asyncio.sleep(10)
                
        raise TimeoutError(f"Deployment {deployment_name} not ready within {timeout} seconds")
        
    async def _run_health_checks(self, deployment_name: str, deployment_config: DeploymentConfig) -> bool:
        """Run health checks on deployment"""
        
        health_config = deployment_config.health_check_config
        
        # Basic health check
        try:
            # In practice, make HTTP requests to health endpoints
            await asyncio.sleep(5)  # Simulate health check
            
            # Check if pods are running
            if self.k8s_client:
                deployment = self.k8s_client.read_namespaced_deployment(
                    name=deployment_name,
                    namespace="default"
                )
                
                if deployment.status.ready_replicas == deployment.spec.replicas:
                    return True
                    
            return True  # Simplified for demo
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

class ModelManagementSystem:
    """Main model management system"""
    
    def __init__(self, 
                 database_url: str,
                 redis_url: str = "redis://localhost:6379",
                 model_storage_path: str = "/models"):
        
        self.database_url = database_url
        self.redis_url = redis_url
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(exist_ok=True)
        
        # Initialize database
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize Redis
        self.redis_client = None
        
        # Initialize components
        self.validator = ModelValidator()
        self.deployer = ModelDeployer()
        
        # MLflow tracking
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Configuration
        self.config = {
            "auto_validation": True,
            "auto_deployment": False,
            "performance_threshold": 0.8,
            "rollback_threshold": 0.7,
            "monitoring_interval": 300,  # 5 minutes
            "cleanup_retention_days": 30
        }
        
    async def initialize(self):
        """Initialize the model management system"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        
        # Start background monitoring
        asyncio.create_task(self._background_monitor())
        
        logger.info("Model management system initialized")
        
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def register_model(self, 
                           model_name: str,
                           model_type: ModelType,
                           framework: ModelFramework,
                           model_path: str,
                           config_path: str,
                           description: str = "",
                           author: str = "system",
                           tags: List[str] = None,
                           parameters: Dict[str, Any] = None,
                           training_config: Dict[str, Any] = None) -> str:
        """Register a new model"""
        
        # Generate model ID and version
        model_id = f"model_{int(time.time())}_{hash(model_name) % 10000}"
        version = "1.0.0"
        
        # Calculate checksum
        checksum = await self._calculate_model_checksum(model_path)
        
        # Get model size
        model_size_mb = await self._get_model_size(model_path)
        
        # Copy model to storage
        stored_model_path = await self._store_model(model_id, version, model_path)
        stored_config_path = await self._store_config(model_id, version, config_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            model_type=model_type,
            framework=framework,
            description=description,
            author=author,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=tags or [],
            parameters=parameters or {},
            training_config=training_config or {},
            performance_metrics={},
            dependencies=[],
            model_size_mb=model_size_mb,
            checksum=checksum
        )
        
        # Store metadata
        await self._store_model_metadata(metadata)
        
        # Create initial version
        model_version = ModelVersion(
            version_id=f"{model_id}_v{version}",
            model_id=model_id,
            version=version,
            status=ModelStatus.STAGING,
            deployment_strategy=DeploymentStrategy.IMMEDIATE,
            model_path=stored_model_path,
            config_path=stored_config_path,
            performance_metrics={},
            validation_results={},
            deployment_config={},
            created_at=datetime.utcnow()
        )
        
        await self._store_model_version(model_version)
        
        # Auto-validate if enabled
        if self.config["auto_validation"]:
            asyncio.create_task(self._auto_validate_model(model_version))
            
        logger.info(f"Registered model: {model_id}")
        return model_id
        
    async def create_model_version(self, 
                                 model_id: str,
                                 model_path: str,
                                 config_path: str,
                                 version: str = None,
                                 deployment_strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN) -> str:
        """Create a new version of existing model"""
        
        # Get existing model metadata
        metadata = await self._get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
            
        # Generate version if not provided
        if not version:
            existing_versions = await self._get_model_versions(model_id)
            version_numbers = [float(v.version) for v in existing_versions]
            next_version = max(version_numbers) + 0.1 if version_numbers else 1.0
            version = f"{next_version:.1f}"
            
        # Store new model files
        stored_model_path = await self._store_model(model_id, version, model_path)
        stored_config_path = await self._store_config(model_id, version, config_path)
        
        # Create version
        version_id = f"{model_id}_v{version}"
        model_version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            version=version,
            status=ModelStatus.STAGING,
            deployment_strategy=deployment_strategy,
            model_path=stored_model_path,
            config_path=stored_config_path,
            performance_metrics={},
            validation_results={},
            deployment_config={},
            created_at=datetime.utcnow()
        )
        
        await self._store_model_version(model_version)
        
        # Auto-validate if enabled
        if self.config["auto_validation"]:
            asyncio.create_task(self._auto_validate_model(model_version))
            
        logger.info(f"Created model version: {version_id}")
        return version_id
        
    async def validate_model_version(self, 
                                   version_id: str,
                                   test_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate a model version"""
        
        model_version = await self._get_model_version(version_id)
        if not model_version:
            raise ValueError(f"Model version not found: {version_id}")
            
        metadata = await self._get_model_metadata(model_version.model_id)
        
        # Run validation
        validation_results = await self.validator.validate_model(
            model_version.model_path,
            metadata.model_type,
            test_data
        )
        
        # Update model version
        model_version.validation_results = validation_results
        model_version.performance_metrics = validation_results
        
        # Update status based on performance
        if self._meets_performance_threshold(validation_results):
            model_version.status = ModelStatus.VALIDATING
        else:
            model_version.status = ModelStatus.FAILED
            
        await self._update_model_version(model_version)
        
        # Store performance metrics
        for metric_name, metric_value in validation_results.items():
            await self._store_performance_metric(
                version_id, metric_name, metric_value, "validation"
            )
            
        logger.info(f"Validated model version: {version_id}")
        return validation_results
        
    async def deploy_model_version(self, 
                                 version_id: str,
                                 target_environment: str,
                                 deployment_strategy: DeploymentStrategy = None) -> bool:
        """Deploy a model version"""
        
        model_version = await self._get_model_version(version_id)
        if not model_version:
            raise ValueError(f"Model version not found: {version_id}")
            
        if model_version.status != ModelStatus.VALIDATING:
            raise ValueError(f"Model version not ready for deployment: {model_version.status}")
            
        # Use provided strategy or default from version
        strategy = deployment_strategy or model_version.deployment_strategy
        
        # Create deployment config
        deployment_config = DeploymentConfig(
            deployment_id=f"deploy_{version_id}_{int(time.time())}",
            model_version_id=version_id,
            strategy=strategy,
            target_environment=target_environment,
            resource_requirements={
                "requests": {"cpu": "500m", "memory": "1Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"}
            },
            scaling_config={
                "min_replicas": 1,
                "max_replicas": 10,
                "target_cpu_utilization": 70
            },
            health_check_config={
                "path": "/health",
                "interval": 30,
                "timeout": 10,
                "retries": 3
            },
            rollback_config={
                "auto_rollback": True,
                "rollback_threshold": self.config["rollback_threshold"]
            },
            monitoring_config={
                "metrics_enabled": True,
                "logging_enabled": True,
                "alerting_enabled": True
            },
            created_at=datetime.utcnow()
        )
        
        # Store deployment config
        await self._store_deployment_config(deployment_config)
        
        # Deploy
        success = await self.deployer.deploy_model(model_version, deployment_config)
        
        if success:
            model_version.status = ModelStatus.PRODUCTION
            model_version.deployed_at = datetime.utcnow()
            await self._update_model_version(model_version)
            
            logger.info(f"Deployed model version: {version_id}")
        else:
            logger.error(f"Failed to deploy model version: {version_id}")
            
        return success
        
    async def compare_models(self, 
                           model_a_id: str,
                           model_b_id: str,
                           test_data: Dict[str, Any]) -> ModelComparison:
        """Compare two models"""
        
        # Validate both models
        results_a = await self.validate_model_version(model_a_id, test_data)
        results_b = await self.validate_model_version(model_b_id, test_data)
        
        # Compare metrics
        comparison_metrics = {
            "model_a": results_a,
            "model_b": results_b
        }
        
        # Determine winner (simplified)
        score_a = sum(results_a.values()) / len(results_a) if results_a else 0
        score_b = sum(results_b.values()) / len(results_b) if results_b else 0
        
        winner = model_a_id if score_a > score_b else model_b_id
        confidence = abs(score_a - score_b) / max(score_a, score_b) if max(score_a, score_b) > 0 else 0
        
        comparison = ModelComparison(
            comparison_id=f"comp_{int(time.time())}",
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            comparison_metrics=comparison_metrics,
            winner=winner,
            confidence=confidence,
            test_dataset_info={"size": len(test_data.get("X_test", []))},
            comparison_time=datetime.utcnow()
        )
        
        await self._store_model_comparison(comparison)
        
        logger.info(f"Compared models: {model_a_id} vs {model_b_id}, winner: {winner}")
        return comparison
        
    async def rollback_deployment(self, 
                                target_environment: str,
                                to_version_id: str = None) -> bool:
        """Rollback deployment to previous version"""
        
        try:
            if to_version_id:
                # Rollback to specific version
                target_version = await self._get_model_version(to_version_id)
                if not target_version:
                    raise ValueError(f"Target version not found: {to_version_id}")
            else:
                # Rollback to previous production version
                target_version = await self._get_previous_production_version(target_environment)
                if not target_version:
                    raise ValueError("No previous production version found")
                    
            # Create rollback deployment config
            deployment_config = DeploymentConfig(
                deployment_id=f"rollback_{target_environment}_{int(time.time())}",
                model_version_id=target_version.version_id,
                strategy=DeploymentStrategy.IMMEDIATE,
                target_environment=target_environment,
                resource_requirements={
                    "requests": {"cpu": "500m", "memory": "1Gi"},
                    "limits": {"cpu": "2", "memory": "4Gi"}
                },
                scaling_config={},
                health_check_config={},
                rollback_config={},
                monitoring_config={},
                created_at=datetime.utcnow()
            )
            
            # Deploy previous version
            success = await self.deployer.deploy_model(target_version, deployment_config)
            
            if success:
                target_version.status = ModelStatus.PRODUCTION
                target_version.deployed_at = datetime.utcnow()
                await self._update_model_version(target_version)
                
                logger.info(f"Rollback successful: {target_environment} -> {target_version.version_id}")
            else:
                logger.error(f"Rollback failed: {target_environment}")
                
            return success
            
        except Exception as e:
            logger.error(f"Rollback error: {e}")
            return False
            
    async def _calculate_model_checksum(self, model_path: str) -> str:
        """Calculate model file checksum"""
        hash_md5 = hashlib.md5()
        
        if Path(model_path).is_file():
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            # Directory checksum
            for file_path in Path(model_path).rglob("*"):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                            
        return hash_md5.hexdigest()
        
    async def _get_model_size(self, model_path: str) -> float:
        """Get model size in MB"""
        path = Path(model_path)
        
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        elif path.is_dir():
            total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            return total_size / (1024 * 1024)
        else:
            return 0.0
            
    async def _store_model(self, model_id: str, version: str, model_path: str) -> str:
        """Store model in managed storage"""
        storage_dir = self.model_storage_path / model_id / version
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        stored_path = storage_dir / "model"
        
        if Path(model_path).is_file():
            shutil.copy2(model_path, stored_path)
        else:
            shutil.copytree(model_path, stored_path, dirs_exist_ok=True)
            
        return str(stored_path)
        
    async def _store_config(self, model_id: str, version: str, config_path: str) -> str:
        """Store config in managed storage"""
        storage_dir = self.model_storage_path / model_id / version
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        stored_path = storage_dir / "config.json"
        shutil.copy2(config_path, stored_path)
        
        return str(stored_path)
        
    def _meets_performance_threshold(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics meet performance threshold"""
        if not metrics:
            return False
            
        # Simple threshold check (customize based on model type)
        avg_score = sum(v for v in metrics.values() if not np.isnan(v) and v != float('inf'))
        avg_score = avg_score / len(metrics) if metrics else 0
        
        return avg_score >= self.config["performance_threshold"]
        
    async def _auto_validate_model(self, model_version: ModelVersion):
        """Auto-validate model with default test data"""
        try:
            # Generate synthetic test data (in practice, use real test sets)
            test_data = {
                "X_test": np.random.random((100, 10)),
                "y_test": np.random.randint(0, 2, 100)
            }
            
            await self.validate_model_version(model_version.version_id, test_data)
            
        except Exception as e:
            logger.error(f"Auto-validation error: {e}")
            
    async def _background_monitor(self):
        """Background monitoring of deployed models"""
        while True:
            try:
                # Monitor production models
                production_versions = await self._get_production_versions()
                
                for version in production_versions:
                    # Check performance metrics
                    recent_metrics = await self._get_recent_performance_metrics(version.version_id)
                    
                    if recent_metrics:
                        avg_performance = sum(recent_metrics.values()) / len(recent_metrics)
                        
                        if avg_performance < self.config["rollback_threshold"]:
                            logger.warning(f"Performance degradation detected: {version.version_id}")
                            # Auto-rollback could be implemented here
                            
                await asyncio.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Background monitor error: {e}")
                await asyncio.sleep(60)
                
    # Database operations (simplified implementations)
    async def _store_model_metadata(self, metadata: ModelMetadata):
        """Store model metadata in database"""
        session = self.SessionLocal()
        try:
            metadata_db = ModelMetadataDB(
                model_id=metadata.model_id,
                model_name=metadata.model_name,
                version=metadata.version,
                model_type=metadata.model_type.value,
                framework=metadata.framework.value,
                description=metadata.description,
                author=metadata.author,
                created_at=metadata.created_at,
                updated_at=metadata.updated_at,
                tags=json.dumps(metadata.tags),
                parameters=json.dumps(metadata.parameters),
                training_config=json.dumps(metadata.training_config),
                performance_metrics=json.dumps(metadata.performance_metrics),
                dependencies=json.dumps(metadata.dependencies),
                model_size_mb=metadata.model_size_mb,
                checksum=metadata.checksum
            )
            session.add(metadata_db)
            session.commit()
        finally:
            session.close()
            
    async def _store_model_version(self, model_version: ModelVersion):
        """Store model version in database"""
        session = self.SessionLocal()
        try:
            version_db = ModelVersionDB(
                version_id=model_version.version_id,
                model_id=model_version.model_id,
                version=model_version.version,
                status=model_version.status.value,
                deployment_strategy=model_version.deployment_strategy.value,
                model_path=model_version.model_path,
                config_path=model_version.config_path,
                performance_metrics=json.dumps(model_version.performance_metrics),
                validation_results=json.dumps(model_version.validation_results),
                deployment_config=json.dumps(model_version.deployment_config),
                created_at=model_version.created_at,
                deployed_at=model_version.deployed_at,
                deprecated_at=model_version.deprecated_at
            )
            session.add(version_db)
            session.commit()
        finally:
            session.close()
            
    async def _get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata from database"""
        session = self.SessionLocal()
        try:
            metadata_db = session.query(ModelMetadataDB).filter(
                ModelMetadataDB.model_id == model_id
            ).first()
            
            if not metadata_db:
                return None
                
            return ModelMetadata(
                model_id=metadata_db.model_id,
                model_name=metadata_db.model_name,
                version=metadata_db.version,
                model_type=ModelType(metadata_db.model_type),
                framework=ModelFramework(metadata_db.framework),
                description=metadata_db.description,
                author=metadata_db.author,
                created_at=metadata_db.created_at,
                updated_at=metadata_db.updated_at,
                tags=json.loads(metadata_db.tags),
                parameters=json.loads(metadata_db.parameters),
                training_config=json.loads(metadata_db.training_config),
                performance_metrics=json.loads(metadata_db.performance_metrics),
                dependencies=json.loads(metadata_db.dependencies),
                model_size_mb=metadata_db.model_size_mb,
                checksum=metadata_db.checksum
            )
            
        finally:
            session.close()
            
    async def _get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get model version from database"""
        session = self.SessionLocal()
        try:
            version_db = session.query(ModelVersionDB).filter(
                ModelVersionDB.version_id == version_id
            ).first()
            
            if not version_db:
                return None
                
            return ModelVersion(
                version_id=version_db.version_id,
                model_id=version_db.model_id,
                version=version_db.version,
                status=ModelStatus(version_db.status),
                deployment_strategy=DeploymentStrategy(version_db.deployment_strategy),
                model_path=version_db.model_path,
                config_path=version_db.config_path,
                performance_metrics=json.loads(version_db.performance_metrics),
                validation_results=json.loads(version_db.validation_results),
                deployment_config=json.loads(version_db.deployment_config),
                created_at=version_db.created_at,
                deployed_at=version_db.deployed_at,
                deprecated_at=version_db.deprecated_at
            )
            
        finally:
            session.close()
            
    async def _update_model_version(self, model_version: ModelVersion):
        """Update model version in database"""
        session = self.SessionLocal()
        try:
            version_db = session.query(ModelVersionDB).filter(
                ModelVersionDB.version_id == model_version.version_id
            ).first()
            
            if version_db:
                version_db.status = model_version.status.value
                version_db.performance_metrics = json.dumps(model_version.performance_metrics)
                version_db.validation_results = json.dumps(model_version.validation_results)
                version_db.deployment_config = json.dumps(model_version.deployment_config)
                version_db.deployed_at = model_version.deployed_at
                version_db.deprecated_at = model_version.deprecated_at
                session.commit()
                
        finally:
            session.close()
            
    async def _store_performance_metric(self, 
                                      version_id: str,
                                      metric_name: str,
                                      metric_value: float,
                                      environment: str):
        """Store performance metric"""
        session = self.SessionLocal()
        try:
            performance_db = ModelPerformanceDB(
                performance_id=f"perf_{int(time.time())}_{version_id}_{metric_name}",
                model_version_id=version_id,
                metric_name=metric_name,
                metric_value=metric_value,
                measurement_time=datetime.utcnow(),
                environment=environment,
                data_sample_size=100,  # Placeholder
                metadata=json.dumps({})
            )
            session.add(performance_db)
            session.commit()
        finally:
            session.close()
            
    async def _store_deployment_config(self, deployment_config: DeploymentConfig):
        """Store deployment config"""
        session = self.SessionLocal()
        try:
            config_db = DeploymentConfigDB(
                deployment_id=deployment_config.deployment_id,
                model_version_id=deployment_config.model_version_id,
                strategy=deployment_config.strategy.value,
                target_environment=deployment_config.target_environment,
                resource_requirements=json.dumps(deployment_config.resource_requirements),
                scaling_config=json.dumps(deployment_config.scaling_config),
                health_check_config=json.dumps(deployment_config.health_check_config),
                rollback_config=json.dumps(deployment_config.rollback_config),
                monitoring_config=json.dumps(deployment_config.monitoring_config),
                created_at=deployment_config.created_at
            )
            session.add(config_db)
            session.commit()
        finally:
            session.close()
            
    async def _store_model_comparison(self, comparison: ModelComparison):
        """Store model comparison"""
        session = self.SessionLocal()
        try:
            comparison_db = ModelComparisonDB(
                comparison_id=comparison.comparison_id,
                model_a_id=comparison.model_a_id,
                model_b_id=comparison.model_b_id,
                comparison_metrics=json.dumps(comparison.comparison_metrics),
                winner=comparison.winner,
                confidence=comparison.confidence,
                test_dataset_info=json.dumps(comparison.test_dataset_info),
                comparison_time=comparison.comparison_time
            )
            session.add(comparison_db)
            session.commit()
        finally:
            session.close()
            
    async def _get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        session = self.SessionLocal()
        try:
            versions_db = session.query(ModelVersionDB).filter(
                ModelVersionDB.model_id == model_id
            ).order_by(ModelVersionDB.created_at.desc()).all()
            
            versions = []
            for version_db in versions_db:
                version = ModelVersion(
                    version_id=version_db.version_id,
                    model_id=version_db.model_id,
                    version=version_db.version,
                    status=ModelStatus(version_db.status),
                    deployment_strategy=DeploymentStrategy(version_db.deployment_strategy),
                    model_path=version_db.model_path,
                    config_path=version_db.config_path,
                    performance_metrics=json.loads(version_db.performance_metrics),
                    validation_results=json.loads(version_db.validation_results),
                    deployment_config=json.loads(version_db.deployment_config),
                    created_at=version_db.created_at,
                    deployed_at=version_db.deployed_at,
                    deprecated_at=version_db.deprecated_at
                )
                versions.append(version)
                
            return versions
            
        finally:
            session.close()
            
    async def _get_production_versions(self) -> List[ModelVersion]:
        """Get all production model versions"""
        session = self.SessionLocal()
        try:
            versions_db = session.query(ModelVersionDB).filter(
                ModelVersionDB.status == ModelStatus.PRODUCTION.value
            ).all()
            
            versions = []
            for version_db in versions_db:
                version = ModelVersion(
                    version_id=version_db.version_id,
                    model_id=version_db.model_id,
                    version=version_db.version,
                    status=ModelStatus(version_db.status),
                    deployment_strategy=DeploymentStrategy(version_db.deployment_strategy),
                    model_path=version_db.model_path,
                    config_path=version_db.config_path,
                    performance_metrics=json.loads(version_db.performance_metrics),
                    validation_results=json.loads(version_db.validation_results),
                    deployment_config=json.loads(version_db.deployment_config),
                    created_at=version_db.created_at,
                    deployed_at=version_db.deployed_at,
                    deprecated_at=version_db.deprecated_at
                )
                versions.append(version)
                
            return versions
            
        finally:
            session.close()
            
    async def _get_recent_performance_metrics(self, version_id: str) -> Dict[str, float]:
        """Get recent performance metrics for a version"""
        session = self.SessionLocal()
        try:
            recent_time = datetime.utcnow() - timedelta(hours=1)
            
            metrics_db = session.query(ModelPerformanceDB).filter(
                ModelPerformanceDB.model_version_id == version_id,
                ModelPerformanceDB.measurement_time >= recent_time
            ).all()
            
            metrics = {}
            for metric_db in metrics_db:
                metrics[metric_db.metric_name] = metric_db.metric_value
                
            return metrics
            
        finally:
            session.close()
            
    async def _get_previous_production_version(self, environment: str) -> Optional[ModelVersion]:
        """Get previous production version for environment"""
        # Simplified implementation
        session = self.SessionLocal()
        try:
            version_db = session.query(ModelVersionDB).filter(
                ModelVersionDB.status == ModelStatus.PRODUCTION.value
            ).order_by(ModelVersionDB.deployed_at.desc()).offset(1).first()
            
            if not version_db:
                return None
                
            return ModelVersion(
                version_id=version_db.version_id,
                model_id=version_db.model_id,
                version=version_db.version,
                status=ModelStatus(version_db.status),
                deployment_strategy=DeploymentStrategy(version_db.deployment_strategy),
                model_path=version_db.model_path,
                config_path=version_db.config_path,
                performance_metrics=json.loads(version_db.performance_metrics),
                validation_results=json.loads(version_db.validation_results),
                deployment_config=json.loads(version_db.deployment_config),
                created_at=version_db.created_at,
                deployed_at=version_db.deployed_at,
                deprecated_at=version_db.deprecated_at
            )
            
        finally:
            session.close()

# Example usage and testing
async def main():
    """Example usage of model management system"""
    system = ModelManagementSystem(
        database_url="sqlite:///model_management.db",
        redis_url="redis://localhost:6379",
        model_storage_path="/tmp/models"
    )
    
    await system.initialize()
    
    try:
        # Register a model
        model_id = await system.register_model(
            model_name="sentiment_classifier",
            model_type=ModelType.CLASSIFICATION,
            framework=ModelFramework.SKLEARN,
            model_path="/tmp/dummy_model.pkl",
            config_path="/tmp/dummy_config.json",
            description="Sentiment classification model",
            author="data_scientist",
            tags=["nlp", "sentiment", "classification"],
            parameters={"n_estimators": 100, "max_depth": 10},
            training_config={"dataset": "imdb", "epochs": 10}
        )
        
        print(f"Registered model: {model_id}")
        
        # Create test data
        test_data = {
            "X_test": np.random.random((100, 10)),
            "y_test": np.random.randint(0, 2, 100)
        }
        
        # Get model versions
        versions = await system._get_model_versions(model_id)
        if versions:
            version_id = versions[0].version_id
            
            # Validate model
            validation_results = await system.validate_model_version(version_id, test_data)
            print(f"Validation results: {validation_results}")
            
            # Deploy model (if validation passes)
            if system._meets_performance_threshold(validation_results):
                success = await system.deploy_model_version(
                    version_id, 
                    "production",
                    DeploymentStrategy.BLUE_GREEN
                )
                print(f"Deployment success: {success}")
                
        # Wait for background monitoring
        await asyncio.sleep(10)
        
    finally:
        await system.close()

if __name__ == "__main__":
    # Create dummy files for testing
    Path("/tmp/dummy_model.pkl").touch()
    Path("/tmp/dummy_config.json").write_text('{"model_type": "classification"}')
    
    asyncio.run(main())

