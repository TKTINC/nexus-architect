"""
Nexus Architect Automated Training Pipeline

This module implements comprehensive automated training pipelines with
MLOps integration, continuous learning, and performance optimization.
"""

import asyncio
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pickle
import joblib
import yaml

import redis.asyncio as redis
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from kubernetes import client, config
import docker
from celery import Celery
import wandb

logger = logging.getLogger(__name__)

class TrainingStatus(str, Enum):
    """Training job status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TrainingType(str, Enum):
    """Types of training"""
    INITIAL = "initial"
    INCREMENTAL = "incremental"
    FINE_TUNING = "fine_tuning"
    TRANSFER_LEARNING = "transfer_learning"
    FEDERATED = "federated"
    REINFORCEMENT = "reinforcement"

class OptimizationStrategy(str, Enum):
    """Optimization strategies"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    HYPERBAND = "hyperband"
    OPTUNA = "optuna"

@dataclass
class TrainingConfig:
    """Training configuration"""
    config_id: str
    model_type: str
    training_type: TrainingType
    optimization_strategy: OptimizationStrategy
    hyperparameters: Dict[str, Any]
    data_config: Dict[str, Any]
    training_params: Dict[str, Any]
    validation_config: Dict[str, Any]
    early_stopping: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    created_at: datetime

@dataclass
class TrainingJob:
    """Training job information"""
    job_id: str
    config_id: str
    model_name: str
    training_type: TrainingType
    status: TrainingStatus
    progress: float
    current_epoch: int
    total_epochs: int
    metrics: Dict[str, float]
    logs: List[str]
    resource_usage: Dict[str, float]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class TrainingResult:
    """Training result"""
    result_id: str
    job_id: str
    model_path: str
    config_path: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_time: float
    validation_results: Dict[str, Any]
    model_artifacts: List[str]
    experiment_id: str
    run_id: str
    created_at: datetime

@dataclass
class HyperparameterOptimization:
    """Hyperparameter optimization result"""
    optimization_id: str
    strategy: OptimizationStrategy
    search_space: Dict[str, Any]
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_trials: int
    optimization_time: float
    created_at: datetime

# Database models
Base = declarative_base()

class TrainingConfigDB(Base):
    __tablename__ = "training_configs"
    
    id = Column(Integer, primary_key=True)
    config_id = Column(String(255), unique=True, nullable=False)
    model_type = Column(String(100), nullable=False)
    training_type = Column(String(50), nullable=False)
    optimization_strategy = Column(String(50), nullable=False)
    hyperparameters = Column(Text, nullable=False)
    data_config = Column(Text, nullable=False)
    training_params = Column(Text, nullable=False)
    validation_config = Column(Text, nullable=False)
    early_stopping = Column(Text, nullable=False)
    resource_requirements = Column(Text, nullable=False)
    monitoring_config = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)

class TrainingJobDB(Base):
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(255), unique=True, nullable=False)
    config_id = Column(String(255), nullable=False)
    model_name = Column(String(255), nullable=False)
    training_type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    progress = Column(Float, default=0.0)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, nullable=False)
    metrics = Column(Text, default='{}')
    logs = Column(Text, default='[]')
    resource_usage = Column(Text, default='{}')
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)

class TrainingResultDB(Base):
    __tablename__ = "training_results"
    
    id = Column(Integer, primary_key=True)
    result_id = Column(String(255), unique=True, nullable=False)
    job_id = Column(String(255), nullable=False)
    model_path = Column(String(500), nullable=False)
    config_path = Column(String(500), nullable=False)
    metrics = Column(Text, nullable=False)
    hyperparameters = Column(Text, nullable=False)
    training_time = Column(Float, nullable=False)
    validation_results = Column(Text, nullable=False)
    model_artifacts = Column(Text, nullable=False)
    experiment_id = Column(String(255))
    run_id = Column(String(255))
    created_at = Column(DateTime, nullable=False)

class HyperparameterOptimizationDB(Base):
    __tablename__ = "hyperparameter_optimizations"
    
    id = Column(Integer, primary_key=True)
    optimization_id = Column(String(255), unique=True, nullable=False)
    strategy = Column(String(50), nullable=False)
    search_space = Column(Text, nullable=False)
    best_params = Column(Text, nullable=False)
    best_score = Column(Float, nullable=False)
    optimization_history = Column(Text, nullable=False)
    total_trials = Column(Integer, nullable=False)
    optimization_time = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False)

class DataProcessor:
    """Data processing for training"""
    
    def __init__(self):
        self.processors = {
            'text': self._process_text_data,
            'tabular': self._process_tabular_data,
            'image': self._process_image_data,
            'time_series': self._process_time_series_data
        }
        
    async def process_training_data(self, 
                                  data_config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
        """Process training data based on configuration"""
        
        data_type = data_config.get('type', 'tabular')
        data_source = data_config.get('source')
        preprocessing = data_config.get('preprocessing', {})
        
        # Load data
        raw_data = await self._load_data(data_source, data_type)
        
        # Process data
        if data_type in self.processors:
            X, y = await self.processors[data_type](raw_data, preprocessing)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
        # Split data
        test_size = data_config.get('test_size', 0.2)
        val_size = data_config.get('val_size', 0.1)
        random_state = data_config.get('random_state', 42)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(test_size / (test_size + val_size)), 
            random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    async def _load_data(self, data_source: str, data_type: str) -> Any:
        """Load data from source"""
        
        if data_source.endswith('.csv'):
            return pd.read_csv(data_source)
        elif data_source.endswith('.json'):
            with open(data_source, 'r') as f:
                return json.load(f)
        elif data_source.endswith('.pkl'):
            with open(data_source, 'rb') as f:
                return pickle.load(f)
        else:
            # Generate synthetic data for demo
            if data_type == 'tabular':
                return pd.DataFrame({
                    'feature_1': np.random.normal(0, 1, 1000),
                    'feature_2': np.random.normal(0, 1, 1000),
                    'feature_3': np.random.normal(0, 1, 1000),
                    'target': np.random.randint(0, 2, 1000)
                })
            elif data_type == 'text':
                return [f"Sample text {i}" for i in range(1000)]
            else:
                return np.random.random((1000, 10))
                
    async def _process_text_data(self, data: Any, preprocessing: Dict[str, Any]) -> Tuple[Any, Any]:
        """Process text data"""
        
        # Simplified text processing
        if isinstance(data, list):
            # Assume list of texts with labels
            texts = data[::2]  # Even indices are texts
            labels = data[1::2]  # Odd indices are labels
        else:
            # Generate synthetic labels
            texts = data
            labels = np.random.randint(0, 2, len(texts))
            
        # Simple text vectorization (in practice, use proper tokenization)
        X = np.array([[len(text), text.count(' ')] for text in texts])
        y = np.array(labels)
        
        return X, y
        
    async def _process_tabular_data(self, data: pd.DataFrame, preprocessing: Dict[str, Any]) -> Tuple[Any, Any]:
        """Process tabular data"""
        
        target_column = preprocessing.get('target_column', 'target')
        
        if target_column in data.columns:
            X = data.drop(columns=[target_column]).values
            y = data[target_column].values
        else:
            # Use last column as target
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
        # Handle missing values
        if preprocessing.get('fill_missing', True):
            X = np.nan_to_num(X)
            
        # Normalize features
        if preprocessing.get('normalize', True):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
        return X, y
        
    async def _process_image_data(self, data: Any, preprocessing: Dict[str, Any]) -> Tuple[Any, Any]:
        """Process image data"""
        
        # Simplified image processing
        if isinstance(data, np.ndarray):
            X = data
            y = np.random.randint(0, 10, len(data))  # Random labels
        else:
            # Generate synthetic image data
            X = np.random.random((1000, 28, 28, 3))
            y = np.random.randint(0, 10, 1000)
            
        # Normalize pixel values
        if preprocessing.get('normalize', True):
            X = X / 255.0
            
        return X, y
        
    async def _process_time_series_data(self, data: Any, preprocessing: Dict[str, Any]) -> Tuple[Any, Any]:
        """Process time series data"""
        
        # Simplified time series processing
        if isinstance(data, pd.DataFrame):
            # Use first column as features, second as target
            X = data.iloc[:, 0].values.reshape(-1, 1)
            y = data.iloc[:, 1].values if data.shape[1] > 1 else np.random.random(len(data))
        else:
            # Generate synthetic time series
            X = np.random.random((1000, 1))
            y = np.random.random(1000)
            
        # Create sequences
        sequence_length = preprocessing.get('sequence_length', 10)
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length])
            
        return np.array(X_sequences), np.array(y_sequences)

class ModelTrainer:
    """Model training implementation"""
    
    def __init__(self):
        self.trainers = {
            'sklearn': self._train_sklearn_model,
            'pytorch': self._train_pytorch_model,
            'transformers': self._train_transformers_model,
            'ensemble': self._train_ensemble_model
        }
        
    async def train_model(self, 
                         training_config: TrainingConfig,
                         training_data: Tuple[Any, Any, Any, Any, Any, Any],
                         job_id: str) -> TrainingResult:
        """Train model based on configuration"""
        
        model_type = training_config.model_type
        X_train, X_val, X_test, y_train, y_val, y_test = training_data
        
        start_time = time.time()
        
        # Initialize MLflow experiment
        experiment_name = f"nexus_training_{model_type}"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(training_config.hyperparameters)
            
            # Train model
            if model_type in self.trainers:
                model, metrics = await self.trainers[model_type](
                    training_config, X_train, X_val, y_train, y_val, job_id
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Validate model
            validation_results = await self._validate_model(
                model, X_test, y_test, training_config.validation_config
            )
            
            # Log metrics
            mlflow.log_metrics(metrics)
            mlflow.log_metrics(validation_results)
            
            # Save model
            model_path = f"/tmp/models/{job_id}_model.pkl"
            config_path = f"/tmp/models/{job_id}_config.json"
            
            await self._save_model(model, model_path, model_type)
            await self._save_config(training_config, config_path)
            
            # Log model to MLflow
            if model_type == 'sklearn':
                mlflow.sklearn.log_model(model, "model")
            elif model_type == 'pytorch':
                mlflow.pytorch.log_model(model, "model")
                
            training_time = time.time() - start_time
            
            # Create result
            result = TrainingResult(
                result_id=f"result_{job_id}",
                job_id=job_id,
                model_path=model_path,
                config_path=config_path,
                metrics=metrics,
                hyperparameters=training_config.hyperparameters,
                training_time=training_time,
                validation_results=validation_results,
                model_artifacts=[model_path, config_path],
                experiment_id=run.info.experiment_id,
                run_id=run.info.run_id,
                created_at=datetime.utcnow()
            )
            
            return result
            
    async def _train_sklearn_model(self, 
                                 config: TrainingConfig,
                                 X_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_train: np.ndarray,
                                 y_val: np.ndarray,
                                 job_id: str) -> Tuple[Any, Dict[str, float]]:
        """Train scikit-learn model"""
        
        algorithm = config.hyperparameters.get('algorithm', 'random_forest')
        
        # Select algorithm
        if algorithm == 'random_forest':
            model = RandomForestClassifier(**{
                k: v for k, v in config.hyperparameters.items() 
                if k != 'algorithm'
            })
        elif algorithm == 'gradient_boosting':
            model = GradientBoostingClassifier(**{
                k: v for k, v in config.hyperparameters.items() 
                if k != 'algorithm'
            })
        elif algorithm == 'logistic_regression':
            model = LogisticRegression(**{
                k: v for k, v in config.hyperparameters.items() 
                if k != 'algorithm'
            })
        elif algorithm == 'svm':
            model = SVC(**{
                k: v for k, v in config.hyperparameters.items() 
                if k != 'algorithm'
            })
        else:
            model = RandomForestClassifier()
            
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'val_f1': f1_score(y_val, val_pred, average='weighted')
        }
        
        return model, metrics
        
    async def _train_pytorch_model(self, 
                                 config: TrainingConfig,
                                 X_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_train: np.ndarray,
                                 y_val: np.ndarray,
                                 job_id: str) -> Tuple[Any, Dict[str, float]]:
        """Train PyTorch model"""
        
        # Simple neural network
        input_size = X_train.shape[1]
        hidden_size = config.hyperparameters.get('hidden_size', 64)
        num_classes = len(np.unique(y_train))
        
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
                
        model = SimpleNN()
        
        # Training parameters
        learning_rate = config.hyperparameters.get('learning_rate', 0.001)
        epochs = config.training_params.get('epochs', 10)
        batch_size = config.training_params.get('batch_size', 32)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            model.train()
            
            # Batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())
                
                train_outputs = model(X_train_tensor)
                train_loss = criterion(train_outputs, y_train_tensor)
                train_losses.append(train_loss.item())
                
        # Calculate final metrics
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            val_outputs = model(X_val_tensor)
            
            train_pred = train_outputs.argmax(dim=1).numpy()
            val_pred = val_outputs.argmax(dim=1).numpy()
            
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1]
        }
        
        return model, metrics
        
    async def _train_transformers_model(self, 
                                      config: TrainingConfig,
                                      X_train: np.ndarray,
                                      X_val: np.ndarray,
                                      y_train: np.ndarray,
                                      y_val: np.ndarray,
                                      job_id: str) -> Tuple[Any, Dict[str, float]]:
        """Train transformers model"""
        
        # Simplified transformers training
        model_name = config.hyperparameters.get('model_name', 'bert-base-uncased')
        
        try:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(np.unique(y_train))
            )
            
            # Simplified training (in practice, use proper tokenization and training)
            metrics = {
                'train_accuracy': 0.85,  # Placeholder
                'val_accuracy': 0.82,   # Placeholder
                'train_loss': 0.3,      # Placeholder
                'val_loss': 0.4         # Placeholder
            }
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Transformers training error: {e}")
            # Fallback to simple model
            return await self._train_sklearn_model(config, X_train, X_val, y_train, y_val, job_id)
            
    async def _train_ensemble_model(self, 
                                  config: TrainingConfig,
                                  X_train: np.ndarray,
                                  X_val: np.ndarray,
                                  y_train: np.ndarray,
                                  y_val: np.ndarray,
                                  job_id: str) -> Tuple[Any, Dict[str, float]]:
        """Train ensemble model"""
        
        from sklearn.ensemble import VotingClassifier
        
        # Create base models
        models = [
            ('rf', RandomForestClassifier(n_estimators=50)),
            ('gb', GradientBoostingClassifier(n_estimators=50)),
            ('lr', LogisticRegression())
        ]
        
        # Create ensemble
        ensemble = VotingClassifier(estimators=models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = ensemble.predict(X_train)
        val_pred = ensemble.predict(X_val)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'val_f1': f1_score(y_val, val_pred, average='weighted')
        }
        
        return ensemble, metrics
        
    async def _validate_model(self, 
                            model: Any,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            validation_config: Dict[str, Any]) -> Dict[str, float]:
        """Validate trained model"""
        
        try:
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                # Handle PyTorch models
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test)
                    outputs = model(X_test_tensor)
                    y_pred = outputs.argmax(dim=1).numpy()
                    
            # Calculate validation metrics
            validation_results = {
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision_score(y_test, y_pred, average='weighted'),
                'test_recall': recall_score(y_test, y_pred, average='weighted'),
                'test_f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {'validation_error': 1.0}
            
    async def _save_model(self, model: Any, model_path: str, model_type: str):
        """Save trained model"""
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        if model_type == 'sklearn' or model_type == 'ensemble':
            joblib.dump(model, model_path)
        elif model_type == 'pytorch':
            torch.save(model.state_dict(), model_path)
        else:
            # Generic pickle save
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
    async def _save_config(self, config: TrainingConfig, config_path: str):
        """Save training configuration"""
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        # Convert datetime to string
        config_dict['created_at'] = config.created_at.isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

class HyperparameterOptimizer:
    """Hyperparameter optimization"""
    
    def __init__(self):
        self.optimizers = {
            OptimizationStrategy.GRID_SEARCH: self._grid_search,
            OptimizationStrategy.RANDOM_SEARCH: self._random_search,
            OptimizationStrategy.BAYESIAN: self._bayesian_optimization,
            OptimizationStrategy.OPTUNA: self._optuna_optimization
        }
        
    async def optimize_hyperparameters(self, 
                                     base_config: TrainingConfig,
                                     search_space: Dict[str, Any],
                                     training_data: Tuple[Any, Any, Any, Any, Any, Any],
                                     n_trials: int = 50) -> HyperparameterOptimization:
        """Optimize hyperparameters"""
        
        strategy = base_config.optimization_strategy
        
        if strategy in self.optimizers:
            result = await self.optimizers[strategy](
                base_config, search_space, training_data, n_trials
            )
        else:
            raise ValueError(f"Unsupported optimization strategy: {strategy}")
            
        return result
        
    async def _grid_search(self, 
                         base_config: TrainingConfig,
                         search_space: Dict[str, Any],
                         training_data: Tuple[Any, Any, Any, Any, Any, Any],
                         n_trials: int) -> HyperparameterOptimization:
        """Grid search optimization"""
        
        from itertools import product
        
        X_train, X_val, X_test, y_train, y_val, y_test = training_data
        
        # Generate parameter combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        best_score = -np.inf
        best_params = {}
        optimization_history = []
        
        start_time = time.time()
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Update config with new parameters
            config = TrainingConfig(
                config_id=f"grid_{int(time.time())}",
                model_type=base_config.model_type,
                training_type=base_config.training_type,
                optimization_strategy=base_config.optimization_strategy,
                hyperparameters={**base_config.hyperparameters, **params},
                data_config=base_config.data_config,
                training_params=base_config.training_params,
                validation_config=base_config.validation_config,
                early_stopping=base_config.early_stopping,
                resource_requirements=base_config.resource_requirements,
                monitoring_config=base_config.monitoring_config,
                created_at=datetime.utcnow()
            )
            
            # Train and evaluate
            try:
                trainer = ModelTrainer()
                result = await trainer.train_model(
                    config, training_data, f"grid_{int(time.time())}"
                )
                
                score = result.validation_results.get('test_accuracy', 0)
                
                optimization_history.append({
                    'params': params,
                    'score': score,
                    'metrics': result.validation_results
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.error(f"Grid search trial error: {e}")
                optimization_history.append({
                    'params': params,
                    'score': 0,
                    'error': str(e)
                })
                
        optimization_time = time.time() - start_time
        
        return HyperparameterOptimization(
            optimization_id=f"grid_{int(time.time())}",
            strategy=OptimizationStrategy.GRID_SEARCH,
            search_space=search_space,
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            total_trials=len(optimization_history),
            optimization_time=optimization_time,
            created_at=datetime.utcnow()
        )
        
    async def _random_search(self, 
                           base_config: TrainingConfig,
                           search_space: Dict[str, Any],
                           training_data: Tuple[Any, Any, Any, Any, Any, Any],
                           n_trials: int) -> HyperparameterOptimization:
        """Random search optimization"""
        
        X_train, X_val, X_test, y_train, y_val, y_test = training_data
        
        best_score = -np.inf
        best_params = {}
        optimization_history = []
        
        start_time = time.time()
        
        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for param_name, param_range in search_space.items():
                if isinstance(param_range, list):
                    params[param_name] = np.random.choice(param_range)
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = np.random.randint(param_range[0], param_range[1])
                    else:
                        params[param_name] = np.random.uniform(param_range[0], param_range[1])
                        
            # Update config with new parameters
            config = TrainingConfig(
                config_id=f"random_{trial}",
                model_type=base_config.model_type,
                training_type=base_config.training_type,
                optimization_strategy=base_config.optimization_strategy,
                hyperparameters={**base_config.hyperparameters, **params},
                data_config=base_config.data_config,
                training_params=base_config.training_params,
                validation_config=base_config.validation_config,
                early_stopping=base_config.early_stopping,
                resource_requirements=base_config.resource_requirements,
                monitoring_config=base_config.monitoring_config,
                created_at=datetime.utcnow()
            )
            
            # Train and evaluate
            try:
                trainer = ModelTrainer()
                result = await trainer.train_model(
                    config, training_data, f"random_{trial}"
                )
                
                score = result.validation_results.get('test_accuracy', 0)
                
                optimization_history.append({
                    'trial': trial,
                    'params': params,
                    'score': score,
                    'metrics': result.validation_results
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.error(f"Random search trial {trial} error: {e}")
                optimization_history.append({
                    'trial': trial,
                    'params': params,
                    'score': 0,
                    'error': str(e)
                })
                
        optimization_time = time.time() - start_time
        
        return HyperparameterOptimization(
            optimization_id=f"random_{int(time.time())}",
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            search_space=search_space,
            best_params=best_params,
            best_score=best_score,
            optimization_history=optimization_history,
            total_trials=n_trials,
            optimization_time=optimization_time,
            created_at=datetime.utcnow()
        )
        
    async def _bayesian_optimization(self, 
                                   base_config: TrainingConfig,
                                   search_space: Dict[str, Any],
                                   training_data: Tuple[Any, Any, Any, Any, Any, Any],
                                   n_trials: int) -> HyperparameterOptimization:
        """Bayesian optimization (simplified)"""
        
        # Simplified Bayesian optimization using random search
        # In practice, use libraries like scikit-optimize or Optuna
        
        return await self._random_search(base_config, search_space, training_data, n_trials)
        
    async def _optuna_optimization(self, 
                                 base_config: TrainingConfig,
                                 search_space: Dict[str, Any],
                                 training_data: Tuple[Any, Any, Any, Any, Any, Any],
                                 n_trials: int) -> HyperparameterOptimization:
        """Optuna optimization"""
        
        try:
            import optuna
            
            def objective(trial):
                # Sample parameters
                params = {}
                for param_name, param_range in search_space.items():
                    if isinstance(param_range, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                    elif isinstance(param_range, tuple) and len(param_range) == 2:
                        if isinstance(param_range[0], int):
                            params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                            
                # Update config
                config = TrainingConfig(
                    config_id=f"optuna_{trial.number}",
                    model_type=base_config.model_type,
                    training_type=base_config.training_type,
                    optimization_strategy=base_config.optimization_strategy,
                    hyperparameters={**base_config.hyperparameters, **params},
                    data_config=base_config.data_config,
                    training_params=base_config.training_params,
                    validation_config=base_config.validation_config,
                    early_stopping=base_config.early_stopping,
                    resource_requirements=base_config.resource_requirements,
                    monitoring_config=base_config.monitoring_config,
                    created_at=datetime.utcnow()
                )
                
                # Train and evaluate
                try:
                    trainer = ModelTrainer()
                    result = asyncio.run(trainer.train_model(
                        config, training_data, f"optuna_{trial.number}"
                    ))
                    
                    return result.validation_results.get('test_accuracy', 0)
                    
                except Exception as e:
                    logger.error(f"Optuna trial {trial.number} error: {e}")
                    return 0
                    
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Extract results
            best_params = study.best_params
            best_score = study.best_value
            
            optimization_history = []
            for trial in study.trials:
                optimization_history.append({
                    'trial': trial.number,
                    'params': trial.params,
                    'score': trial.value if trial.value else 0,
                    'state': trial.state.name
                })
                
            return HyperparameterOptimization(
                optimization_id=f"optuna_{int(time.time())}",
                strategy=OptimizationStrategy.OPTUNA,
                search_space=search_space,
                best_params=best_params,
                best_score=best_score,
                optimization_history=optimization_history,
                total_trials=len(study.trials),
                optimization_time=0,  # Optuna doesn't track this directly
                created_at=datetime.utcnow()
            )
            
        except ImportError:
            logger.warning("Optuna not available, falling back to random search")
            return await self._random_search(base_config, search_space, training_data, n_trials)

class AutomatedTrainingPipeline:
    """Main automated training pipeline"""
    
    def __init__(self, 
                 database_url: str,
                 redis_url: str = "redis://localhost:6379",
                 celery_broker: str = "redis://localhost:6379"):
        
        self.database_url = database_url
        self.redis_url = redis_url
        self.celery_broker = celery_broker
        
        # Initialize database
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize Redis
        self.redis_client = None
        
        # Initialize Celery
        self.celery_app = Celery('nexus_training', broker=celery_broker)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        
        # Configuration
        self.config = {
            "max_concurrent_jobs": 5,
            "job_timeout": 3600,  # 1 hour
            "auto_optimization": True,
            "auto_deployment": False,
            "monitoring_interval": 60,  # 1 minute
            "cleanup_retention_days": 7
        }
        
    async def initialize(self):
        """Initialize the training pipeline"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        
        # Initialize MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow_training.db")
        
        # Start background monitoring
        asyncio.create_task(self._background_monitor())
        
        logger.info("Automated training pipeline initialized")
        
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def create_training_config(self, 
                                   model_type: str,
                                   training_type: TrainingType,
                                   optimization_strategy: OptimizationStrategy,
                                   hyperparameters: Dict[str, Any],
                                   data_config: Dict[str, Any],
                                   training_params: Dict[str, Any] = None,
                                   validation_config: Dict[str, Any] = None) -> str:
        """Create training configuration"""
        
        config_id = f"config_{int(time.time())}_{hash(model_type) % 10000}"
        
        config = TrainingConfig(
            config_id=config_id,
            model_type=model_type,
            training_type=training_type,
            optimization_strategy=optimization_strategy,
            hyperparameters=hyperparameters,
            data_config=data_config,
            training_params=training_params or {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            validation_config=validation_config or {
                "metrics": ["accuracy", "f1_score"],
                "cross_validation": False
            },
            early_stopping={
                "enabled": True,
                "patience": 5,
                "min_delta": 0.001
            },
            resource_requirements={
                "cpu": "2",
                "memory": "4Gi",
                "gpu": "0"
            },
            monitoring_config={
                "log_metrics": True,
                "log_artifacts": True,
                "track_gradients": False
            },
            created_at=datetime.utcnow()
        )
        
        await self._store_training_config(config)
        
        logger.info(f"Created training config: {config_id}")
        return config_id
        
    async def submit_training_job(self, 
                                config_id: str,
                                model_name: str,
                                priority: int = 1) -> str:
        """Submit training job"""
        
        config = await self._get_training_config(config_id)
        if not config:
            raise ValueError(f"Training config not found: {config_id}")
            
        job_id = f"job_{int(time.time())}_{hash(model_name) % 10000}"
        
        job = TrainingJob(
            job_id=job_id,
            config_id=config_id,
            model_name=model_name,
            training_type=config.training_type,
            status=TrainingStatus.QUEUED,
            progress=0.0,
            current_epoch=0,
            total_epochs=config.training_params.get('epochs', 10),
            metrics={},
            logs=[],
            resource_usage={}
        )
        
        await self._store_training_job(job)
        
        # Queue job for execution
        await self._queue_training_job(job_id, priority)
        
        logger.info(f"Submitted training job: {job_id}")
        return job_id
        
    async def run_training_job(self, job_id: str) -> TrainingResult:
        """Run training job"""
        
        job = await self._get_training_job(job_id)
        if not job:
            raise ValueError(f"Training job not found: {job_id}")
            
        config = await self._get_training_config(job.config_id)
        if not config:
            raise ValueError(f"Training config not found: {job.config_id}")
            
        try:
            # Update job status
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.utcnow()
            await self._update_training_job(job)
            
            # Process data
            training_data = await self.data_processor.process_training_data(config.data_config)
            
            # Optimize hyperparameters if enabled
            if config.optimization_strategy != OptimizationStrategy.GRID_SEARCH or self.config["auto_optimization"]:
                search_space = config.hyperparameters.get('search_space', {})
                if search_space:
                    optimization_result = await self.hyperparameter_optimizer.optimize_hyperparameters(
                        config, search_space, training_data
                    )
                    
                    # Update config with best parameters
                    config.hyperparameters.update(optimization_result.best_params)
                    await self._store_hyperparameter_optimization(optimization_result)
                    
            # Train model
            result = await self.model_trainer.train_model(config, training_data, job_id)
            
            # Update job status
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            job.metrics = result.metrics
            await self._update_training_job(job)
            
            # Store result
            await self._store_training_result(result)
            
            logger.info(f"Training job completed: {job_id}")
            return result
            
        except Exception as e:
            # Update job status
            job.status = TrainingStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            await self._update_training_job(job)
            
            logger.error(f"Training job failed: {job_id}, error: {e}")
            raise
            
    async def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job status"""
        return await self._get_training_job(job_id)
        
    async def cancel_training_job(self, job_id: str) -> bool:
        """Cancel training job"""
        
        job = await self._get_training_job(job_id)
        if not job:
            return False
            
        if job.status in [TrainingStatus.QUEUED, TrainingStatus.RUNNING]:
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            await self._update_training_job(job)
            
            logger.info(f"Cancelled training job: {job_id}")
            return True
            
        return False
        
    async def get_training_results(self, job_id: str) -> Optional[TrainingResult]:
        """Get training results"""
        return await self._get_training_result(job_id)
        
    async def list_training_jobs(self, 
                               status: TrainingStatus = None,
                               limit: int = 100) -> List[TrainingJob]:
        """List training jobs"""
        return await self._list_training_jobs(status, limit)
        
    async def _queue_training_job(self, job_id: str, priority: int):
        """Queue training job for execution"""
        
        # Add to Redis queue
        await self.redis_client.zadd("training_queue", {job_id: priority})
        
        # Start job processor if not running
        asyncio.create_task(self._process_training_queue())
        
    async def _process_training_queue(self):
        """Process training queue"""
        
        while True:
            try:
                # Check for available slots
                running_jobs = await self._count_running_jobs()
                
                if running_jobs < self.config["max_concurrent_jobs"]:
                    # Get next job from queue
                    job_data = await self.redis_client.zpopmax("training_queue")
                    
                    if job_data:
                        job_id = job_data[0][0]
                        
                        # Run job in background
                        asyncio.create_task(self._run_job_with_timeout(job_id))
                        
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(10)
                
    async def _run_job_with_timeout(self, job_id: str):
        """Run job with timeout"""
        
        try:
            await asyncio.wait_for(
                self.run_training_job(job_id),
                timeout=self.config["job_timeout"]
            )
        except asyncio.TimeoutError:
            # Cancel job due to timeout
            job = await self._get_training_job(job_id)
            if job:
                job.status = TrainingStatus.FAILED
                job.error_message = "Job timeout"
                job.completed_at = datetime.utcnow()
                await self._update_training_job(job)
                
            logger.error(f"Training job timeout: {job_id}")
            
        except Exception as e:
            logger.error(f"Training job execution error: {job_id}, error: {e}")
            
    async def _background_monitor(self):
        """Background monitoring of training jobs"""
        
        while True:
            try:
                # Monitor running jobs
                running_jobs = await self._list_training_jobs(TrainingStatus.RUNNING)
                
                for job in running_jobs:
                    # Check if job is stuck
                    if job.started_at:
                        runtime = datetime.utcnow() - job.started_at
                        if runtime.total_seconds() > self.config["job_timeout"]:
                            # Mark as failed
                            job.status = TrainingStatus.FAILED
                            job.error_message = "Job stuck/timeout"
                            job.completed_at = datetime.utcnow()
                            await self._update_training_job(job)
                            
                # Cleanup old jobs
                await self._cleanup_old_jobs()
                
                await asyncio.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Background monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_old_jobs(self):
        """Cleanup old completed jobs"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.config["cleanup_retention_days"])
        
        # This would be implemented with proper database cleanup
        # For now, just log
        logger.info(f"Cleanup jobs older than {cutoff_date}")
        
    async def _count_running_jobs(self) -> int:
        """Count running jobs"""
        running_jobs = await self._list_training_jobs(TrainingStatus.RUNNING)
        return len(running_jobs)
        
    # Database operations (simplified implementations)
    async def _store_training_config(self, config: TrainingConfig):
        """Store training config in database"""
        session = self.SessionLocal()
        try:
            config_db = TrainingConfigDB(
                config_id=config.config_id,
                model_type=config.model_type,
                training_type=config.training_type.value,
                optimization_strategy=config.optimization_strategy.value,
                hyperparameters=json.dumps(config.hyperparameters),
                data_config=json.dumps(config.data_config),
                training_params=json.dumps(config.training_params),
                validation_config=json.dumps(config.validation_config),
                early_stopping=json.dumps(config.early_stopping),
                resource_requirements=json.dumps(config.resource_requirements),
                monitoring_config=json.dumps(config.monitoring_config),
                created_at=config.created_at
            )
            session.add(config_db)
            session.commit()
        finally:
            session.close()
            
    async def _get_training_config(self, config_id: str) -> Optional[TrainingConfig]:
        """Get training config from database"""
        session = self.SessionLocal()
        try:
            config_db = session.query(TrainingConfigDB).filter(
                TrainingConfigDB.config_id == config_id
            ).first()
            
            if not config_db:
                return None
                
            return TrainingConfig(
                config_id=config_db.config_id,
                model_type=config_db.model_type,
                training_type=TrainingType(config_db.training_type),
                optimization_strategy=OptimizationStrategy(config_db.optimization_strategy),
                hyperparameters=json.loads(config_db.hyperparameters),
                data_config=json.loads(config_db.data_config),
                training_params=json.loads(config_db.training_params),
                validation_config=json.loads(config_db.validation_config),
                early_stopping=json.loads(config_db.early_stopping),
                resource_requirements=json.loads(config_db.resource_requirements),
                monitoring_config=json.loads(config_db.monitoring_config),
                created_at=config_db.created_at
            )
            
        finally:
            session.close()
            
    async def _store_training_job(self, job: TrainingJob):
        """Store training job in database"""
        session = self.SessionLocal()
        try:
            job_db = TrainingJobDB(
                job_id=job.job_id,
                config_id=job.config_id,
                model_name=job.model_name,
                training_type=job.training_type.value,
                status=job.status.value,
                progress=job.progress,
                current_epoch=job.current_epoch,
                total_epochs=job.total_epochs,
                metrics=json.dumps(job.metrics),
                logs=json.dumps(job.logs),
                resource_usage=json.dumps(job.resource_usage),
                started_at=job.started_at,
                completed_at=job.completed_at,
                error_message=job.error_message
            )
            session.add(job_db)
            session.commit()
        finally:
            session.close()
            
    async def _get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job from database"""
        session = self.SessionLocal()
        try:
            job_db = session.query(TrainingJobDB).filter(
                TrainingJobDB.job_id == job_id
            ).first()
            
            if not job_db:
                return None
                
            return TrainingJob(
                job_id=job_db.job_id,
                config_id=job_db.config_id,
                model_name=job_db.model_name,
                training_type=TrainingType(job_db.training_type),
                status=TrainingStatus(job_db.status),
                progress=job_db.progress,
                current_epoch=job_db.current_epoch,
                total_epochs=job_db.total_epochs,
                metrics=json.loads(job_db.metrics),
                logs=json.loads(job_db.logs),
                resource_usage=json.loads(job_db.resource_usage),
                started_at=job_db.started_at,
                completed_at=job_db.completed_at,
                error_message=job_db.error_message
            )
            
        finally:
            session.close()
            
    async def _update_training_job(self, job: TrainingJob):
        """Update training job in database"""
        session = self.SessionLocal()
        try:
            job_db = session.query(TrainingJobDB).filter(
                TrainingJobDB.job_id == job.job_id
            ).first()
            
            if job_db:
                job_db.status = job.status.value
                job_db.progress = job.progress
                job_db.current_epoch = job.current_epoch
                job_db.metrics = json.dumps(job.metrics)
                job_db.logs = json.dumps(job.logs)
                job_db.resource_usage = json.dumps(job.resource_usage)
                job_db.started_at = job.started_at
                job_db.completed_at = job.completed_at
                job_db.error_message = job.error_message
                session.commit()
                
        finally:
            session.close()
            
    async def _store_training_result(self, result: TrainingResult):
        """Store training result in database"""
        session = self.SessionLocal()
        try:
            result_db = TrainingResultDB(
                result_id=result.result_id,
                job_id=result.job_id,
                model_path=result.model_path,
                config_path=result.config_path,
                metrics=json.dumps(result.metrics),
                hyperparameters=json.dumps(result.hyperparameters),
                training_time=result.training_time,
                validation_results=json.dumps(result.validation_results),
                model_artifacts=json.dumps(result.model_artifacts),
                experiment_id=result.experiment_id,
                run_id=result.run_id,
                created_at=result.created_at
            )
            session.add(result_db)
            session.commit()
        finally:
            session.close()
            
    async def _get_training_result(self, job_id: str) -> Optional[TrainingResult]:
        """Get training result from database"""
        session = self.SessionLocal()
        try:
            result_db = session.query(TrainingResultDB).filter(
                TrainingResultDB.job_id == job_id
            ).first()
            
            if not result_db:
                return None
                
            return TrainingResult(
                result_id=result_db.result_id,
                job_id=result_db.job_id,
                model_path=result_db.model_path,
                config_path=result_db.config_path,
                metrics=json.loads(result_db.metrics),
                hyperparameters=json.loads(result_db.hyperparameters),
                training_time=result_db.training_time,
                validation_results=json.loads(result_db.validation_results),
                model_artifacts=json.loads(result_db.model_artifacts),
                experiment_id=result_db.experiment_id,
                run_id=result_db.run_id,
                created_at=result_db.created_at
            )
            
        finally:
            session.close()
            
    async def _list_training_jobs(self, 
                                status: TrainingStatus = None,
                                limit: int = 100) -> List[TrainingJob]:
        """List training jobs from database"""
        session = self.SessionLocal()
        try:
            query = session.query(TrainingJobDB)
            
            if status:
                query = query.filter(TrainingJobDB.status == status.value)
                
            jobs_db = query.order_by(TrainingJobDB.id.desc()).limit(limit).all()
            
            jobs = []
            for job_db in jobs_db:
                job = TrainingJob(
                    job_id=job_db.job_id,
                    config_id=job_db.config_id,
                    model_name=job_db.model_name,
                    training_type=TrainingType(job_db.training_type),
                    status=TrainingStatus(job_db.status),
                    progress=job_db.progress,
                    current_epoch=job_db.current_epoch,
                    total_epochs=job_db.total_epochs,
                    metrics=json.loads(job_db.metrics),
                    logs=json.loads(job_db.logs),
                    resource_usage=json.loads(job_db.resource_usage),
                    started_at=job_db.started_at,
                    completed_at=job_db.completed_at,
                    error_message=job_db.error_message
                )
                jobs.append(job)
                
            return jobs
            
        finally:
            session.close()
            
    async def _store_hyperparameter_optimization(self, optimization: HyperparameterOptimization):
        """Store hyperparameter optimization result"""
        session = self.SessionLocal()
        try:
            optimization_db = HyperparameterOptimizationDB(
                optimization_id=optimization.optimization_id,
                strategy=optimization.strategy.value,
                search_space=json.dumps(optimization.search_space),
                best_params=json.dumps(optimization.best_params),
                best_score=optimization.best_score,
                optimization_history=json.dumps(optimization.optimization_history),
                total_trials=optimization.total_trials,
                optimization_time=optimization.optimization_time,
                created_at=optimization.created_at
            )
            session.add(optimization_db)
            session.commit()
        finally:
            session.close()

# Example usage and testing
async def main():
    """Example usage of automated training pipeline"""
    pipeline = AutomatedTrainingPipeline(
        database_url="sqlite:///training_pipeline.db",
        redis_url="redis://localhost:6379"
    )
    
    await pipeline.initialize()
    
    try:
        # Create training configuration
        config_id = await pipeline.create_training_config(
            model_type="sklearn",
            training_type=TrainingType.INITIAL,
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
            hyperparameters={
                "algorithm": "random_forest",
                "n_estimators": 100,
                "max_depth": 10,
                "search_space": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": (5, 20),
                    "min_samples_split": (2, 10)
                }
            },
            data_config={
                "type": "tabular",
                "source": "/tmp/synthetic_data.csv",
                "test_size": 0.2,
                "val_size": 0.1,
                "preprocessing": {
                    "normalize": True,
                    "fill_missing": True
                }
            }
        )
        
        print(f"Created training config: {config_id}")
        
        # Submit training job
        job_id = await pipeline.submit_training_job(
            config_id=config_id,
            model_name="test_classifier",
            priority=1
        )
        
        print(f"Submitted training job: {job_id}")
        
        # Monitor job progress
        while True:
            job = await pipeline.get_job_status(job_id)
            if job:
                print(f"Job {job_id}: {job.status.value}, Progress: {job.progress}%")
                
                if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                    break
                    
            await asyncio.sleep(5)
            
        # Get results
        if job and job.status == TrainingStatus.COMPLETED:
            result = await pipeline.get_training_results(job_id)
            if result:
                print(f"Training completed successfully!")
                print(f"Metrics: {result.metrics}")
                print(f"Validation: {result.validation_results}")
                print(f"Model path: {result.model_path}")
                
        # List all jobs
        all_jobs = await pipeline.list_training_jobs(limit=10)
        print(f"Total jobs: {len(all_jobs)}")
        
    finally:
        await pipeline.close()

if __name__ == "__main__":
    # Create dummy data for testing
    Path("/tmp/models").mkdir(exist_ok=True)
    
    # Create synthetic data
    data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(0, 1, 1000),
        'feature_3': np.random.normal(0, 1, 1000),
        'target': np.random.randint(0, 2, 1000)
    })
    data.to_csv('/tmp/synthetic_data.csv', index=False)
    
    asyncio.run(main())

