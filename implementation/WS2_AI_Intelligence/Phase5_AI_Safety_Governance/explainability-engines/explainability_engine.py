"""
Nexus Architect Explainability Engine

This module implements comprehensive AI explainability and transparency systems
including LIME, SHAP, attention visualization, and custom explanation methods.
"""

import asyncio
import logging
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import pickle
import base64
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline
import lime
import lime.lime_text
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations available"""
    FEATURE_IMPORTANCE = "feature_importance"
    LIME_TEXT = "lime_text"
    LIME_TABULAR = "lime_tabular"
    SHAP_VALUES = "shap_values"
    ATTENTION_WEIGHTS = "attention_weights"
    COUNTERFACTUAL = "counterfactual"
    EXAMPLE_BASED = "example_based"
    RULE_BASED = "rule_based"
    CAUSAL = "causal"
    GLOBAL_SURROGATE = "global_surrogate"

class ExplanationAudience(Enum):
    """Target audience for explanations"""
    TECHNICAL_EXPERT = "technical_expert"
    DOMAIN_EXPERT = "domain_expert"
    BUSINESS_USER = "business_user"
    END_USER = "end_user"
    REGULATOR = "regulator"
    AUDITOR = "auditor"

class ModelType(Enum):
    """Types of models that can be explained"""
    TRANSFORMER = "transformer"
    TREE_BASED = "tree_based"
    LINEAR = "linear"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    BLACK_BOX = "black_box"

@dataclass
class ExplanationRequest:
    """Request for model explanation"""
    request_id: str
    model_id: str
    model_type: ModelType
    explanation_type: ExplanationType
    audience: ExplanationAudience
    input_data: Dict[str, Any]
    prediction: Dict[str, Any]
    context: Dict[str, Any]
    requested_by: str
    requested_at: datetime
    priority: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExplanationResult:
    """Result of explanation generation"""
    explanation_id: str
    request_id: str
    explanation_type: ExplanationType
    audience: ExplanationAudience
    explanation_data: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    narrative: str
    confidence_score: float
    generated_at: datetime
    generation_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class LIMEExplainer:
    """LIME-based explanation generator"""
    
    def __init__(self):
        self.text_explainer = None
        self.tabular_explainer = None
    
    def explain_text_prediction(self, model_predict_fn: Callable, 
                              text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for text prediction
        
        Args:
            model_predict_fn: Function that takes text and returns prediction probabilities
            text: Input text to explain
            num_features: Number of features to include in explanation
            
        Returns:
            LIME explanation data
        """
        try:
            # Initialize text explainer if not exists
            if self.text_explainer is None:
                self.text_explainer = lime.lime_text.LimeTextExplainer(
                    class_names=['negative', 'positive'],
                    mode='classification'
                )
            
            # Generate explanation
            explanation = self.text_explainer.explain_instance(
                text, 
                model_predict_fn, 
                num_features=num_features,
                num_samples=1000
            )
            
            # Extract explanation data
            explanation_data = {
                'type': 'lime_text',
                'features': [],
                'prediction_probabilities': explanation.predict_proba.tolist(),
                'score': explanation.score,
                'intercept': explanation.intercept[0] if hasattr(explanation, 'intercept') else 0
            }
            
            # Extract feature importance
            for feature, importance in explanation.as_list():
                explanation_data['features'].append({
                    'feature': feature,
                    'importance': importance,
                    'abs_importance': abs(importance)
                })
            
            # Sort by absolute importance
            explanation_data['features'].sort(key=lambda x: x['abs_importance'], reverse=True)
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"LIME text explanation failed: {e}")
            raise
    
    def explain_tabular_prediction(self, model_predict_fn: Callable,
                                 instance: np.ndarray, training_data: np.ndarray,
                                 feature_names: List[str], 
                                 categorical_features: List[int] = None,
                                 num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for tabular prediction
        
        Args:
            model_predict_fn: Function that takes instances and returns predictions
            instance: Single instance to explain
            training_data: Training data for generating perturbations
            feature_names: Names of features
            categorical_features: Indices of categorical features
            num_features: Number of features to include in explanation
            
        Returns:
            LIME explanation data
        """
        try:
            # Initialize tabular explainer if not exists
            if self.tabular_explainer is None:
                self.tabular_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data,
                    feature_names=feature_names,
                    categorical_features=categorical_features or [],
                    mode='classification'
                )
            
            # Generate explanation
            explanation = self.tabular_explainer.explain_instance(
                instance,
                model_predict_fn,
                num_features=num_features,
                num_samples=1000
            )
            
            # Extract explanation data
            explanation_data = {
                'type': 'lime_tabular',
                'features': [],
                'prediction_probabilities': explanation.predict_proba.tolist(),
                'score': explanation.score,
                'intercept': explanation.intercept[0] if hasattr(explanation, 'intercept') else 0
            }
            
            # Extract feature importance
            for feature_idx, importance in explanation.as_list():
                feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
                explanation_data['features'].append({
                    'feature_index': feature_idx,
                    'feature_name': feature_name,
                    'feature_value': instance[feature_idx],
                    'importance': importance,
                    'abs_importance': abs(importance)
                })
            
            # Sort by absolute importance
            explanation_data['features'].sort(key=lambda x: x['abs_importance'], reverse=True)
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"LIME tabular explanation failed: {e}")
            raise

class SHAPExplainer:
    """SHAP-based explanation generator"""
    
    def __init__(self):
        self.explainers = {}
    
    def explain_tree_model(self, model, X: np.ndarray, 
                          feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Generate SHAP explanation for tree-based model
        
        Args:
            model: Tree-based model (RandomForest, XGBoost, etc.)
            X: Input data to explain
            feature_names: Names of features
            
        Returns:
            SHAP explanation data
        """
        try:
            # Create tree explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            explanation_data = {
                'type': 'shap_tree',
                'shap_values': shap_values.tolist(),
                'expected_value': explainer.expected_value,
                'feature_names': feature_names or [f"feature_{i}" for i in range(X.shape[1])],
                'base_values': [explainer.expected_value] * len(X)
            }
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            explanation_data['feature_importance'] = [
                {
                    'feature_name': feature_names[i] if feature_names else f"feature_{i}",
                    'importance': float(feature_importance[i])
                }
                for i in range(len(feature_importance))
            ]
            
            # Sort by importance
            explanation_data['feature_importance'].sort(key=lambda x: x['importance'], reverse=True)
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"SHAP tree explanation failed: {e}")
            raise
    
    def explain_linear_model(self, model, X: np.ndarray,
                           feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Generate SHAP explanation for linear model
        
        Args:
            model: Linear model
            X: Input data to explain
            feature_names: Names of features
            
        Returns:
            SHAP explanation data
        """
        try:
            # Create linear explainer
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
            
            explanation_data = {
                'type': 'shap_linear',
                'shap_values': shap_values.tolist(),
                'expected_value': explainer.expected_value,
                'feature_names': feature_names or [f"feature_{i}" for i in range(X.shape[1])],
                'base_values': [explainer.expected_value] * len(X)
            }
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            explanation_data['feature_importance'] = [
                {
                    'feature_name': feature_names[i] if feature_names else f"feature_{i}",
                    'importance': float(feature_importance[i])
                }
                for i in range(len(feature_importance))
            ]
            
            # Sort by importance
            explanation_data['feature_importance'].sort(key=lambda x: x['importance'], reverse=True)
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"SHAP linear explanation failed: {e}")
            raise
    
    def explain_deep_model(self, model, X: np.ndarray, background_data: np.ndarray,
                          feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Generate SHAP explanation for deep learning model
        
        Args:
            model: Deep learning model
            X: Input data to explain
            background_data: Background dataset for DeepExplainer
            feature_names: Names of features
            
        Returns:
            SHAP explanation data
        """
        try:
            # Create deep explainer
            explainer = shap.DeepExplainer(model, background_data)
            shap_values = explainer.shap_values(X)
            
            # Handle multi-output case
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            explanation_data = {
                'type': 'shap_deep',
                'shap_values': shap_values.tolist(),
                'expected_value': explainer.expected_value,
                'feature_names': feature_names or [f"feature_{i}" for i in range(X.shape[1])],
                'base_values': [explainer.expected_value] * len(X)
            }
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            explanation_data['feature_importance'] = [
                {
                    'feature_name': feature_names[i] if feature_names else f"feature_{i}",
                    'importance': float(feature_importance[i])
                }
                for i in range(len(feature_importance))
            ]
            
            # Sort by importance
            explanation_data['feature_importance'].sort(key=lambda x: x['importance'], reverse=True)
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"SHAP deep explanation failed: {e}")
            raise

class AttentionExplainer:
    """Attention-based explanation for transformer models"""
    
    def __init__(self):
        self.tokenizers = {}
        self.models = {}
    
    def explain_transformer_attention(self, model_name: str, text: str,
                                    layer_idx: int = -1) -> Dict[str, Any]:
        """
        Extract and visualize attention weights from transformer model
        
        Args:
            model_name: Name/path of the transformer model
            text: Input text
            layer_idx: Layer index to extract attention from (-1 for last layer)
            
        Returns:
            Attention explanation data
        """
        try:
            # Load tokenizer and model if not cached
            if model_name not in self.tokenizers:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_name] = AutoModel.from_pretrained(
                    model_name, 
                    output_attentions=True
                )
            
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = model(**inputs)
                attentions = outputs.attentions
            
            # Extract attention from specified layer
            attention = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
            
            # Average across attention heads
            avg_attention = attention.mean(dim=0)  # [seq_len, seq_len]
            
            # Create attention matrix
            attention_matrix = avg_attention.numpy()
            
            explanation_data = {
                'type': 'attention_weights',
                'tokens': tokens,
                'attention_matrix': attention_matrix.tolist(),
                'layer_index': layer_idx,
                'num_heads': attention.shape[0],
                'sequence_length': len(tokens)
            }
            
            # Calculate token importance (sum of attention received)
            token_importance = attention_matrix.sum(axis=0)
            explanation_data['token_importance'] = [
                {
                    'token': token,
                    'importance': float(importance),
                    'position': i
                }
                for i, (token, importance) in enumerate(zip(tokens, token_importance))
            ]
            
            # Sort by importance
            explanation_data['token_importance'].sort(key=lambda x: x['importance'], reverse=True)
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Attention explanation failed: {e}")
            raise

class CounterfactualExplainer:
    """Counterfactual explanation generator"""
    
    def __init__(self):
        pass
    
    def generate_counterfactuals(self, model_predict_fn: Callable,
                               instance: np.ndarray, target_class: int,
                               feature_names: List[str],
                               categorical_features: List[int] = None,
                               num_counterfactuals: int = 5) -> Dict[str, Any]:
        """
        Generate counterfactual explanations
        
        Args:
            model_predict_fn: Function that predicts class probabilities
            instance: Original instance
            target_class: Desired target class
            feature_names: Names of features
            categorical_features: Indices of categorical features
            num_counterfactuals: Number of counterfactuals to generate
            
        Returns:
            Counterfactual explanation data
        """
        try:
            original_prediction = model_predict_fn(instance.reshape(1, -1))[0]
            original_class = np.argmax(original_prediction)
            
            counterfactuals = []
            
            # Simple perturbation-based counterfactual generation
            for _ in range(num_counterfactuals * 10):  # Generate more and select best
                # Create perturbation
                perturbed_instance = instance.copy()
                
                # Randomly select features to perturb
                num_features_to_change = np.random.randint(1, min(5, len(instance)))
                features_to_change = np.random.choice(len(instance), num_features_to_change, replace=False)
                
                for feature_idx in features_to_change:
                    if categorical_features and feature_idx in categorical_features:
                        # For categorical features, randomly change to different category
                        unique_values = [0, 1]  # Simplified binary categorical
                        current_value = int(perturbed_instance[feature_idx])
                        new_value = np.random.choice([v for v in unique_values if v != current_value])
                        perturbed_instance[feature_idx] = new_value
                    else:
                        # For numerical features, add noise
                        noise = np.random.normal(0, 0.1 * np.std(instance))
                        perturbed_instance[feature_idx] += noise
                
                # Check if this creates desired counterfactual
                new_prediction = model_predict_fn(perturbed_instance.reshape(1, -1))[0]
                new_class = np.argmax(new_prediction)
                
                if new_class == target_class:
                    # Calculate distance from original
                    distance = np.linalg.norm(perturbed_instance - instance)
                    
                    # Calculate feature changes
                    feature_changes = []
                    for i, (orig_val, new_val) in enumerate(zip(instance, perturbed_instance)):
                        if abs(orig_val - new_val) > 1e-6:
                            feature_changes.append({
                                'feature_index': i,
                                'feature_name': feature_names[i] if i < len(feature_names) else f"feature_{i}",
                                'original_value': float(orig_val),
                                'counterfactual_value': float(new_val),
                                'change': float(new_val - orig_val)
                            })
                    
                    counterfactuals.append({
                        'instance': perturbed_instance.tolist(),
                        'prediction': new_prediction.tolist(),
                        'predicted_class': int(new_class),
                        'distance': float(distance),
                        'feature_changes': feature_changes,
                        'num_changes': len(feature_changes)
                    })
                
                if len(counterfactuals) >= num_counterfactuals:
                    break
            
            # Sort by distance (prefer closer counterfactuals)
            counterfactuals.sort(key=lambda x: x['distance'])
            
            explanation_data = {
                'type': 'counterfactual',
                'original_instance': instance.tolist(),
                'original_prediction': original_prediction.tolist(),
                'original_class': int(original_class),
                'target_class': int(target_class),
                'counterfactuals': counterfactuals[:num_counterfactuals],
                'feature_names': feature_names
            }
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Counterfactual explanation failed: {e}")
            raise

class VisualizationGenerator:
    """Generate visualizations for explanations"""
    
    def __init__(self):
        pass
    
    def create_feature_importance_plot(self, features: List[Dict[str, Any]],
                                     title: str = "Feature Importance") -> Dict[str, Any]:
        """Create feature importance bar plot"""
        try:
            # Extract data
            feature_names = [f['feature_name'] if 'feature_name' in f else f['feature'] for f in features[:10]]
            importances = [f['importance'] if 'importance' in f else f['abs_importance'] for f in features[:10]]
            
            # Create plotly figure
            fig = go.Figure(data=[
                go.Bar(
                    x=importances,
                    y=feature_names,
                    orientation='h',
                    marker_color='steelblue'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Importance",
                yaxis_title="Features",
                height=400,
                margin=dict(l=150, r=50, t=50, b=50)
            )
            
            # Convert to JSON
            plot_json = fig.to_json()
            
            return {
                'type': 'feature_importance_bar',
                'title': title,
                'data': json.loads(plot_json),
                'description': f"Bar plot showing importance of top {len(feature_names)} features"
            }
            
        except Exception as e:
            logger.error(f"Feature importance plot creation failed: {e}")
            return {'type': 'error', 'message': str(e)}
    
    def create_attention_heatmap(self, attention_matrix: List[List[float]],
                               tokens: List[str], title: str = "Attention Heatmap") -> Dict[str, Any]:
        """Create attention heatmap visualization"""
        try:
            # Create plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=attention_matrix,
                x=tokens,
                y=tokens,
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Tokens (To)",
                yaxis_title="Tokens (From)",
                height=600,
                width=800
            )
            
            # Convert to JSON
            plot_json = fig.to_json()
            
            return {
                'type': 'attention_heatmap',
                'title': title,
                'data': json.loads(plot_json),
                'description': f"Attention heatmap showing relationships between {len(tokens)} tokens"
            }
            
        except Exception as e:
            logger.error(f"Attention heatmap creation failed: {e}")
            return {'type': 'error', 'message': str(e)}
    
    def create_shap_waterfall(self, shap_values: List[float], feature_names: List[str],
                            base_value: float, prediction: float,
                            title: str = "SHAP Waterfall Plot") -> Dict[str, Any]:
        """Create SHAP waterfall plot"""
        try:
            # Prepare data for waterfall plot
            values = [base_value] + shap_values + [prediction]
            labels = ['Base Value'] + feature_names + ['Prediction']
            
            # Create cumulative values for waterfall effect
            cumulative = [base_value]
            for val in shap_values:
                cumulative.append(cumulative[-1] + val)
            cumulative.append(prediction)
            
            # Create plotly waterfall
            fig = go.Figure(go.Waterfall(
                name="SHAP Values",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(shap_values) + ["total"],
                x=labels,
                textposition="outside",
                text=[f"{val:.3f}" for val in values],
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title=title,
                showlegend=True,
                height=500
            )
            
            # Convert to JSON
            plot_json = fig.to_json()
            
            return {
                'type': 'shap_waterfall',
                'title': title,
                'data': json.loads(plot_json),
                'description': f"SHAP waterfall plot showing contribution of {len(feature_names)} features"
            }
            
        except Exception as e:
            logger.error(f"SHAP waterfall plot creation failed: {e}")
            return {'type': 'error', 'message': str(e)}

class NarrativeGenerator:
    """Generate natural language explanations"""
    
    def __init__(self):
        self.templates = {
            ExplanationAudience.TECHNICAL_EXPERT: {
                'feature_importance': "The model's prediction is primarily driven by {top_features}. The feature importance scores indicate {interpretation}.",
                'lime': "LIME analysis reveals that {key_features} are the most influential factors, with {confidence} confidence in the explanation.",
                'shap': "SHAP values show that {positive_features} increase the prediction while {negative_features} decrease it.",
                'attention': "The attention mechanism focuses on {important_tokens}, indicating these are crucial for the model's decision."
            },
            ExplanationAudience.BUSINESS_USER: {
                'feature_importance': "The decision is mainly based on {top_features_simple}. This means {business_interpretation}.",
                'lime': "Our analysis shows that {key_factors} are the most important factors influencing this decision.",
                'shap': "The factors {positive_factors} support this decision, while {negative_factors} work against it.",
                'attention': "The system paid most attention to {key_phrases} when making this decision."
            },
            ExplanationAudience.END_USER: {
                'feature_importance': "This decision is based on {simple_factors}. Here's what this means for you: {user_impact}.",
                'lime': "The main reasons for this decision are {main_reasons}.",
                'shap': "These factors support the decision: {supporting_factors}. These factors work against it: {opposing_factors}.",
                'attention': "The system focused on these key points: {key_points}."
            }
        }
    
    def generate_narrative(self, explanation_data: Dict[str, Any],
                         audience: ExplanationAudience,
                         context: Dict[str, Any] = None) -> str:
        """
        Generate natural language explanation narrative
        
        Args:
            explanation_data: Explanation data from explainer
            audience: Target audience for the narrative
            context: Additional context for narrative generation
            
        Returns:
            Natural language explanation
        """
        try:
            explanation_type = explanation_data.get('type', 'unknown')
            templates = self.templates.get(audience, self.templates[ExplanationAudience.BUSINESS_USER])
            
            if explanation_type == 'feature_importance' or 'feature_importance' in explanation_data:
                return self._generate_feature_importance_narrative(
                    explanation_data, audience, templates, context
                )
            elif explanation_type.startswith('lime'):
                return self._generate_lime_narrative(
                    explanation_data, audience, templates, context
                )
            elif explanation_type.startswith('shap'):
                return self._generate_shap_narrative(
                    explanation_data, audience, templates, context
                )
            elif explanation_type == 'attention_weights':
                return self._generate_attention_narrative(
                    explanation_data, audience, templates, context
                )
            else:
                return f"Explanation of type {explanation_type} generated successfully."
                
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            return "An explanation has been generated for this prediction."
    
    def _generate_feature_importance_narrative(self, explanation_data: Dict[str, Any],
                                             audience: ExplanationAudience,
                                             templates: Dict[str, str],
                                             context: Dict[str, Any]) -> str:
        """Generate narrative for feature importance explanation"""
        features = explanation_data.get('feature_importance', explanation_data.get('features', []))
        top_features = features[:3]
        
        if audience == ExplanationAudience.TECHNICAL_EXPERT:
            feature_names = [f"{f['feature_name']} (importance: {f['importance']:.3f})" for f in top_features]
            top_features_str = ", ".join(feature_names)
            interpretation = "higher values indicate stronger influence on the prediction"
            
            return templates['feature_importance'].format(
                top_features=top_features_str,
                interpretation=interpretation
            )
        
        elif audience == ExplanationAudience.BUSINESS_USER:
            feature_names = [self._simplify_feature_name(f['feature_name']) for f in top_features]
            top_features_simple = ", ".join(feature_names)
            business_interpretation = "these are the key factors driving the decision"
            
            return templates['feature_importance'].format(
                top_features_simple=top_features_simple,
                business_interpretation=business_interpretation
            )
        
        else:  # END_USER
            simple_factors = ", ".join([self._simplify_feature_name(f['feature_name']) for f in top_features])
            user_impact = "understanding these factors can help you make informed decisions"
            
            return templates['feature_importance'].format(
                simple_factors=simple_factors,
                user_impact=user_impact
            )
    
    def _generate_lime_narrative(self, explanation_data: Dict[str, Any],
                               audience: ExplanationAudience,
                               templates: Dict[str, str],
                               context: Dict[str, Any]) -> str:
        """Generate narrative for LIME explanation"""
        features = explanation_data.get('features', [])
        top_features = features[:3]
        confidence = explanation_data.get('score', 0.8)
        
        if audience == ExplanationAudience.TECHNICAL_EXPERT:
            key_features = ", ".join([f"{f['feature']} ({f['importance']:.3f})" for f in top_features])
            return templates['lime'].format(
                key_features=key_features,
                confidence=f"{confidence:.2f}"
            )
        else:
            key_factors = ", ".join([self._simplify_feature_name(f['feature']) for f in top_features])
            return templates['lime'].format(key_factors=key_factors)
    
    def _generate_shap_narrative(self, explanation_data: Dict[str, Any],
                               audience: ExplanationAudience,
                               templates: Dict[str, str],
                               context: Dict[str, Any]) -> str:
        """Generate narrative for SHAP explanation"""
        features = explanation_data.get('feature_importance', [])
        positive_features = [f for f in features if f['importance'] > 0][:2]
        negative_features = [f for f in features if f['importance'] < 0][:2]
        
        if audience == ExplanationAudience.TECHNICAL_EXPERT:
            pos_names = ", ".join([f['feature_name'] for f in positive_features])
            neg_names = ", ".join([f['feature_name'] for f in negative_features])
        else:
            pos_names = ", ".join([self._simplify_feature_name(f['feature_name']) for f in positive_features])
            neg_names = ", ".join([self._simplify_feature_name(f['feature_name']) for f in negative_features])
        
        template_key = 'shap' if audience == ExplanationAudience.TECHNICAL_EXPERT else 'shap'
        return templates[template_key].format(
            positive_features=pos_names,
            negative_features=neg_names,
            positive_factors=pos_names,
            negative_factors=neg_names,
            supporting_factors=pos_names,
            opposing_factors=neg_names
        )
    
    def _generate_attention_narrative(self, explanation_data: Dict[str, Any],
                                    audience: ExplanationAudience,
                                    templates: Dict[str, str],
                                    context: Dict[str, Any]) -> str:
        """Generate narrative for attention explanation"""
        token_importance = explanation_data.get('token_importance', [])
        top_tokens = token_importance[:3]
        
        if audience == ExplanationAudience.TECHNICAL_EXPERT:
            important_tokens = ", ".join([f"'{t['token']}' ({t['importance']:.3f})" for t in top_tokens])
        else:
            important_tokens = ", ".join([f"'{t['token']}'" for t in top_tokens])
        
        template_key = 'attention'
        return templates[template_key].format(
            important_tokens=important_tokens,
            key_phrases=important_tokens,
            key_points=important_tokens
        )
    
    def _simplify_feature_name(self, feature_name: str) -> str:
        """Simplify technical feature names for non-technical audiences"""
        # Simple mapping for common technical terms
        simplifications = {
            'feature_': '',
            '_score': ' score',
            '_count': ' count',
            '_ratio': ' ratio',
            '_avg': ' average',
            '_std': ' variation',
            '_max': ' maximum',
            '_min': ' minimum'
        }
        
        simplified = feature_name.lower()
        for tech_term, simple_term in simplifications.items():
            simplified = simplified.replace(tech_term, simple_term)
        
        return simplified.title()

class ExplainabilityEngine:
    """Main explainability engine"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379",
                 db_url: str = "postgresql://localhost/nexus_explainability"):
        self.redis_client = redis.from_url(redis_url)
        self.db_url = db_url
        
        # Initialize explainers
        self.lime_explainer = LIMEExplainer()
        self.shap_explainer = SHAPExplainer()
        self.attention_explainer = AttentionExplainer()
        self.counterfactual_explainer = CounterfactualExplainer()
        self.visualization_generator = VisualizationGenerator()
        self.narrative_generator = NarrativeGenerator()
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS explanation_requests (
                    request_id VARCHAR(255) PRIMARY KEY,
                    model_id VARCHAR(255) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    explanation_type VARCHAR(50) NOT NULL,
                    audience VARCHAR(50) NOT NULL,
                    input_data JSONB,
                    prediction JSONB,
                    context JSONB,
                    requested_by VARCHAR(255) NOT NULL,
                    requested_at TIMESTAMP NOT NULL,
                    priority VARCHAR(20) DEFAULT 'normal',
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS explanation_results (
                    explanation_id VARCHAR(255) PRIMARY KEY,
                    request_id VARCHAR(255) REFERENCES explanation_requests(request_id),
                    explanation_type VARCHAR(50) NOT NULL,
                    audience VARCHAR(50) NOT NULL,
                    explanation_data JSONB,
                    visualizations JSONB,
                    narrative TEXT,
                    confidence_score FLOAT,
                    generated_at TIMESTAMP NOT NULL,
                    generation_time_ms INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_explanation_requests_requested_at ON explanation_requests(requested_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_explanation_results_request_id ON explanation_results(request_id)")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Explainability database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize explainability database: {e}")
    
    async def generate_explanation(self, request: ExplanationRequest) -> ExplanationResult:
        """
        Generate explanation for model prediction
        
        Args:
            request: Explanation request
            
        Returns:
            Explanation result
        """
        start_time = datetime.utcnow()
        
        try:
            # Store request in database
            await self._store_request(request)
            
            # Generate explanation based on type
            explanation_data = await self._generate_explanation_data(request)
            
            # Generate visualizations
            visualizations = await self._generate_visualizations(explanation_data, request)
            
            # Generate narrative
            narrative = self.narrative_generator.generate_narrative(
                explanation_data, request.audience, request.context
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(explanation_data, request)
            
            # Create result
            end_time = datetime.utcnow()
            generation_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            result = ExplanationResult(
                explanation_id=str(uuid.uuid4()),
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                audience=request.audience,
                explanation_data=explanation_data,
                visualizations=visualizations,
                narrative=narrative,
                confidence_score=confidence_score,
                generated_at=end_time,
                generation_time_ms=generation_time_ms
            )
            
            # Store result in database
            await self._store_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            raise
    
    async def _generate_explanation_data(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Generate explanation data based on request type"""
        explanation_type = request.explanation_type
        input_data = request.input_data
        
        if explanation_type == ExplanationType.LIME_TEXT:
            # Mock model prediction function
            def mock_predict_fn(texts):
                return np.random.rand(len(texts), 2)  # Binary classification
            
            text = input_data.get('text', '')
            return self.lime_explainer.explain_text_prediction(mock_predict_fn, text)
        
        elif explanation_type == ExplanationType.LIME_TABULAR:
            # Mock tabular data
            instance = np.array(input_data.get('features', []))
            training_data = np.random.rand(100, len(instance))
            feature_names = input_data.get('feature_names', [f"feature_{i}" for i in range(len(instance))])
            
            def mock_predict_fn(X):
                return np.random.rand(len(X), 2)
            
            return self.lime_explainer.explain_tabular_prediction(
                mock_predict_fn, instance, training_data, feature_names
            )
        
        elif explanation_type == ExplanationType.SHAP_VALUES:
            # Mock model and data
            X = np.array(input_data.get('features', [])).reshape(1, -1)
            feature_names = input_data.get('feature_names', [f"feature_{i}" for i in range(X.shape[1])])
            
            # Create mock model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            mock_X = np.random.rand(100, X.shape[1])
            mock_y = np.random.randint(0, 2, 100)
            model.fit(mock_X, mock_y)
            
            return self.shap_explainer.explain_tree_model(model, X, feature_names)
        
        elif explanation_type == ExplanationType.ATTENTION_WEIGHTS:
            model_name = input_data.get('model_name', 'bert-base-uncased')
            text = input_data.get('text', '')
            return self.attention_explainer.explain_transformer_attention(model_name, text)
        
        elif explanation_type == ExplanationType.COUNTERFACTUAL:
            instance = np.array(input_data.get('features', []))
            target_class = input_data.get('target_class', 1)
            feature_names = input_data.get('feature_names', [f"feature_{i}" for i in range(len(instance))])
            
            def mock_predict_fn(X):
                return np.random.rand(len(X), 2)
            
            return self.counterfactual_explainer.generate_counterfactuals(
                mock_predict_fn, instance, target_class, feature_names
            )
        
        else:
            # Default feature importance
            features = input_data.get('features', [])
            feature_names = input_data.get('feature_names', [f"feature_{i}" for i in range(len(features))])
            
            # Mock feature importance
            importance_scores = np.random.rand(len(features))
            
            return {
                'type': 'feature_importance',
                'feature_importance': [
                    {
                        'feature_name': name,
                        'importance': float(score)
                    }
                    for name, score in zip(feature_names, importance_scores)
                ]
            }
    
    async def _generate_visualizations(self, explanation_data: Dict[str, Any],
                                     request: ExplanationRequest) -> List[Dict[str, Any]]:
        """Generate visualizations for explanation"""
        visualizations = []
        
        explanation_type = explanation_data.get('type', '')
        
        if 'feature_importance' in explanation_data:
            viz = self.visualization_generator.create_feature_importance_plot(
                explanation_data['feature_importance']
            )
            visualizations.append(viz)
        
        if explanation_type == 'attention_weights':
            viz = self.visualization_generator.create_attention_heatmap(
                explanation_data['attention_matrix'],
                explanation_data['tokens']
            )
            visualizations.append(viz)
        
        if explanation_type.startswith('shap') and 'shap_values' in explanation_data:
            # Create SHAP waterfall plot for first instance
            shap_values = explanation_data['shap_values'][0] if explanation_data['shap_values'] else []
            feature_names = explanation_data.get('feature_names', [])
            base_value = explanation_data.get('expected_value', 0)
            prediction = base_value + sum(shap_values)
            
            viz = self.visualization_generator.create_shap_waterfall(
                shap_values, feature_names, base_value, prediction
            )
            visualizations.append(viz)
        
        return visualizations
    
    def _calculate_confidence_score(self, explanation_data: Dict[str, Any],
                                  request: ExplanationRequest) -> float:
        """Calculate confidence score for explanation"""
        # Simple confidence calculation based on explanation type and data quality
        base_confidence = 0.8
        
        explanation_type = explanation_data.get('type', '')
        
        # Adjust confidence based on explanation type
        if explanation_type.startswith('lime'):
            # LIME has built-in score
            lime_score = explanation_data.get('score', 0.8)
            return min(lime_score, 0.95)
        
        elif explanation_type.startswith('shap'):
            # SHAP is generally reliable
            return 0.9
        
        elif explanation_type == 'attention_weights':
            # Attention explanations are interpretable but may not reflect true importance
            return 0.75
        
        elif explanation_type == 'counterfactual':
            # Confidence based on number of successful counterfactuals
            counterfactuals = explanation_data.get('counterfactuals', [])
            if len(counterfactuals) >= 3:
                return 0.85
            elif len(counterfactuals) >= 1:
                return 0.7
            else:
                return 0.5
        
        return base_confidence
    
    async def _store_request(self, request: ExplanationRequest):
        """Store explanation request in database"""
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            query = """
                INSERT INTO explanation_requests 
                (request_id, model_id, model_type, explanation_type, audience,
                 input_data, prediction, context, requested_by, requested_at,
                 priority, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                request.request_id,
                request.model_id,
                request.model_type.value,
                request.explanation_type.value,
                request.audience.value,
                json.dumps(request.input_data),
                json.dumps(request.prediction),
                json.dumps(request.context),
                request.requested_by,
                request.requested_at,
                request.priority,
                json.dumps(request.metadata)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store explanation request: {e}")
    
    async def _store_result(self, result: ExplanationResult):
        """Store explanation result in database"""
        try:
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            query = """
                INSERT INTO explanation_results 
                (explanation_id, request_id, explanation_type, audience,
                 explanation_data, visualizations, narrative, confidence_score,
                 generated_at, generation_time_ms, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                result.explanation_id,
                result.request_id,
                result.explanation_type.value,
                result.audience.value,
                json.dumps(result.explanation_data),
                json.dumps(result.visualizations),
                result.narrative,
                result.confidence_score,
                result.generated_at,
                result.generation_time_ms,
                json.dumps(result.metadata)
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store explanation result: {e}")

# FastAPI application
app = FastAPI(title="Nexus Architect Explainability Engine", version="1.0.0")
security = HTTPBearer()

# Global explainability engine instance
explainability_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize explainability engine on startup"""
    global explainability_engine
    explainability_engine = ExplainabilityEngine()
    logger.info("Explainability Engine started successfully")

# Pydantic models for API
class ExplanationRequestModel(BaseModel):
    model_id: str = Field(..., description="ID of the model to explain")
    model_type: str = Field(..., description="Type of the model")
    explanation_type: str = Field(..., description="Type of explanation to generate")
    audience: str = Field(..., description="Target audience for explanation")
    input_data: Dict[str, Any] = Field(..., description="Input data for explanation")
    prediction: Dict[str, Any] = Field(..., description="Model prediction to explain")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    requested_by: str = Field(..., description="ID of the requester")
    priority: str = Field(default="normal", description="Priority level")

@app.post("/explainability/generate")
async def generate_explanation(request: ExplanationRequestModel,
                             credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Generate explanation for model prediction"""
    try:
        explanation_request = ExplanationRequest(
            request_id=str(uuid.uuid4()),
            model_id=request.model_id,
            model_type=ModelType(request.model_type),
            explanation_type=ExplanationType(request.explanation_type),
            audience=ExplanationAudience(request.audience),
            input_data=request.input_data,
            prediction=request.prediction,
            context=request.context,
            requested_by=request.requested_by,
            requested_at=datetime.utcnow(),
            priority=request.priority
        )
        
        result = await explainability_engine.generate_explanation(explanation_request)
        
        return {
            'explanation_id': result.explanation_id,
            'request_id': result.request_id,
            'explanation_type': result.explanation_type.value,
            'audience': result.audience.value,
            'explanation_data': result.explanation_data,
            'visualizations': result.visualizations,
            'narrative': result.narrative,
            'confidence_score': result.confidence_score,
            'generated_at': result.generated_at.isoformat(),
            'generation_time_ms': result.generation_time_ms
        }
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate explanation")

@app.get("/explainability/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

