"""
Nexus Architect Graph Neural Network Analytics
Advanced GNN-based analytics for knowledge graph insights and predictions
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx, from_networkx
import torch_geometric.transforms as T

from neo4j import GraphDatabase
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NodePrediction:
    node_id: str
    predicted_class: str
    confidence: float
    actual_class: Optional[str] = None

@dataclass
class LinkPrediction:
    source_id: str
    target_id: str
    predicted_relationship: str
    confidence: float
    actual_relationship: Optional[str] = None

@dataclass
class GraphEmbedding:
    node_id: str
    embedding: List[float]
    embedding_dim: int

class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        # Apply graph convolutions
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling for graph-level tasks
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4, num_layers: int = 3):
        super(GraphAttentionNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.2))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.2))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=0.2))
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        # Apply graph attention layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling for graph-level tasks
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        # Apply GraphSAGE layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling for graph-level tasks
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x

class GNNAnalytics:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.node_classifier = None
        self.link_predictor = None
        self.graph_embedder = None
        
        # Encoders
        self.node_label_encoder = LabelEncoder()
        self.edge_label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
        # Training data
        self.graph_data = None
        self.node_features = None
        self.edge_features = None
        
    def load_graph_data(self, 
                       node_types: Optional[List[str]] = None,
                       relationship_types: Optional[List[str]] = None) -> Data:
        """Load graph data from Neo4j and convert to PyTorch Geometric format"""
        logger.info("Loading graph data from Neo4j...")
        
        with self.driver.session() as session:
            # Load nodes
            node_query = """
            MATCH (n)
            RETURN n.id as id, labels(n) as labels, properties(n) as properties
            """
            
            if node_types:
                type_filter = " OR ".join([f"n:{node_type}" for node_type in node_types])
                node_query = f"""
                MATCH (n)
                WHERE {type_filter}
                RETURN n.id as id, labels(n) as labels, properties(n) as properties
                """
            
            nodes_result = session.run(node_query)
            nodes_data = [record for record in nodes_result]
            
            # Load edges
            edge_query = """
            MATCH (n)-[r]->(m)
            RETURN n.id as source, type(r) as relationship_type, m.id as target, properties(r) as properties
            """
            
            if relationship_types:
                type_filter = " OR ".join([f"type(r) = '{rel_type}'" for rel_type in relationship_types])
                edge_query = f"""
                MATCH (n)-[r]->(m)
                WHERE {type_filter}
                RETURN n.id as source, type(r) as relationship_type, m.id as target, properties(r) as properties
                """
            
            edges_result = session.run(edge_query)
            edges_data = [record for record in edges_result]
        
        # Create node mapping
        node_ids = [node["id"] for node in nodes_data]
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Prepare node features
        node_features = self._prepare_node_features(nodes_data)
        
        # Prepare edge indices and features
        edge_indices, edge_features = self._prepare_edge_features(edges_data, node_to_idx)
        
        # Create PyTorch Geometric data object
        self.graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_features, dtype=torch.float) if edge_features is not None else None,
            node_ids=node_ids
        )
        
        logger.info(f"Loaded graph with {len(node_ids)} nodes and {len(edges_data)} edges")
        return self.graph_data
    
    def _prepare_node_features(self, nodes_data: List[Dict]) -> np.ndarray:
        """Prepare node features for GNN training"""
        features = []
        node_labels = []
        
        for node in nodes_data:
            # Extract node type (primary label)
            primary_label = node["labels"][0] if node["labels"] else "Unknown"
            node_labels.append(primary_label)
            
            # Extract numerical features from properties
            props = node["properties"]
            feature_vector = []
            
            # Basic features
            feature_vector.append(len(node["labels"]))  # Number of labels
            feature_vector.append(len(props))  # Number of properties
            
            # Specific property features
            feature_vector.append(1 if "created_at" in props else 0)
            feature_vector.append(1 if "updated_at" in props else 0)
            feature_vector.append(1 if "name" in props else 0)
            feature_vector.append(1 if "description" in props else 0)
            
            # Text length features
            if "name" in props:
                feature_vector.append(len(str(props["name"])))
            else:
                feature_vector.append(0)
            
            if "description" in props:
                feature_vector.append(len(str(props["description"])))
            else:
                feature_vector.append(0)
            
            # Confidence and score features
            feature_vector.append(props.get("confidence", 0.5))
            feature_vector.append(props.get("score", 0.0))
            feature_vector.append(props.get("priority", 0.0))
            
            features.append(feature_vector)
        
        # Encode node labels
        self.node_label_encoder.fit(node_labels)
        
        # Convert to numpy array and normalize
        features_array = np.array(features)
        features_normalized = self.feature_scaler.fit_transform(features_array)
        
        # Add one-hot encoded labels
        label_encoded = self.node_label_encoder.transform(node_labels)
        label_onehot = np.eye(len(self.node_label_encoder.classes_))[label_encoded]
        
        # Combine features
        final_features = np.concatenate([features_normalized, label_onehot], axis=1)
        
        self.node_features = final_features
        return final_features
    
    def _prepare_edge_features(self, edges_data: List[Dict], node_to_idx: Dict[str, int]) -> Tuple[List[List[int]], Optional[np.ndarray]]:
        """Prepare edge indices and features for GNN training"""
        edge_indices = []
        edge_features = []
        edge_types = []
        
        for edge in edges_data:
            source_id = edge["source"]
            target_id = edge["target"]
            
            if source_id in node_to_idx and target_id in node_to_idx:
                source_idx = node_to_idx[source_id]
                target_idx = node_to_idx[target_id]
                
                edge_indices.append([source_idx, target_idx])
                edge_types.append(edge["relationship_type"])
                
                # Extract edge features
                props = edge["properties"]
                feature_vector = []
                
                # Basic features
                feature_vector.append(len(props))  # Number of properties
                feature_vector.append(props.get("confidence", 0.5))
                feature_vector.append(props.get("weight", 1.0))
                feature_vector.append(props.get("strength", 0.5))
                
                # Temporal features
                if "created_at" in props:
                    try:
                        created_at = datetime.fromisoformat(props["created_at"].replace('Z', '+00:00'))
                        days_ago = (datetime.now() - created_at).days
                        feature_vector.append(days_ago)
                    except:
                        feature_vector.append(0)
                else:
                    feature_vector.append(0)
                
                edge_features.append(feature_vector)
        
        # Encode edge types
        if edge_types:
            self.edge_label_encoder.fit(edge_types)
        
        # Convert edge features to numpy array
        if edge_features:
            edge_features_array = np.array(edge_features)
            edge_features_normalized = StandardScaler().fit_transform(edge_features_array)
            self.edge_features = edge_features_normalized
            return edge_indices, edge_features_normalized
        else:
            return edge_indices, None
    
    def train_node_classifier(self, 
                            target_property: str = "labels",
                            hidden_dim: int = 64,
                            num_epochs: int = 200,
                            learning_rate: float = 0.01) -> Dict[str, float]:
        """Train a node classification model"""
        logger.info("Training node classifier...")
        
        if self.graph_data is None:
            raise ValueError("Graph data not loaded. Call load_graph_data() first.")
        
        # Prepare target labels
        with self.driver.session() as session:
            query = f"""
            MATCH (n)
            RETURN n.id as id, n.{target_property} as target
            """
            result = session.run(query)
            targets = {record["id"]: record["target"] for record in result}
        
        # Create target tensor
        node_targets = []
        for node_id in self.graph_data.node_ids:
            target = targets.get(node_id, "Unknown")
            if isinstance(target, list):
                target = target[0] if target else "Unknown"
            node_targets.append(target)
        
        # Encode targets
        target_encoder = LabelEncoder()
        encoded_targets = target_encoder.fit_transform(node_targets)
        y = torch.tensor(encoded_targets, dtype=torch.long)
        
        # Split data
        num_nodes = self.graph_data.x.size(0)
        indices = np.arange(num_nodes)
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        
        # Initialize model
        input_dim = self.graph_data.x.size(1)
        output_dim = len(target_encoder.classes_)
        
        self.node_classifier = GraphConvolutionalNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(self.device)
        
        # Move data to device
        data = self.graph_data.to(self.device)
        y = y.to(self.device)
        train_mask = train_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        
        # Training setup
        optimizer = optim.Adam(self.node_classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.node_classifier.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            out = self.node_classifier(data.x, data.edge_index)
            loss = criterion(out[train_mask], y[train_mask])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluation
        self.node_classifier.eval()
        with torch.no_grad():
            out = self.node_classifier(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            train_acc = accuracy_score(y[train_mask].cpu(), pred[train_mask].cpu())
            test_acc = accuracy_score(y[test_mask].cpu(), pred[test_mask].cpu())
            
            # Additional metrics
            test_precision = precision_score(y[test_mask].cpu(), pred[test_mask].cpu(), average='weighted')
            test_recall = recall_score(y[test_mask].cpu(), pred[test_mask].cpu(), average='weighted')
            test_f1 = f1_score(y[test_mask].cpu(), pred[test_mask].cpu(), average='weighted')
        
        metrics = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }
        
        logger.info(f"Node classifier training completed. Test accuracy: {test_acc:.4f}")
        return metrics
    
    def predict_node_properties(self, node_ids: Optional[List[str]] = None) -> List[NodePrediction]:
        """Predict properties for nodes using trained classifier"""
        if self.node_classifier is None:
            raise ValueError("Node classifier not trained. Call train_node_classifier() first.")
        
        predictions = []
        
        # Move data to device
        data = self.graph_data.to(self.device)
        
        self.node_classifier.eval()
        with torch.no_grad():
            out = self.node_classifier(data.x, data.edge_index)
            probabilities = F.softmax(out, dim=1)
            pred_classes = out.argmax(dim=1)
            
            for i, node_id in enumerate(self.graph_data.node_ids):
                if node_ids is None or node_id in node_ids:
                    predicted_class = self.node_label_encoder.inverse_transform([pred_classes[i].cpu().item()])[0]
                    confidence = probabilities[i].max().cpu().item()
                    
                    prediction = NodePrediction(
                        node_id=node_id,
                        predicted_class=predicted_class,
                        confidence=confidence
                    )
                    predictions.append(prediction)
        
        return predictions
    
    def train_link_predictor(self, 
                           hidden_dim: int = 64,
                           num_epochs: int = 200,
                           learning_rate: float = 0.01) -> Dict[str, float]:
        """Train a link prediction model"""
        logger.info("Training link predictor...")
        
        if self.graph_data is None:
            raise ValueError("Graph data not loaded. Call load_graph_data() first.")
        
        # Prepare training data for link prediction
        edge_index = self.graph_data.edge_index
        num_edges = edge_index.size(1)
        
        # Create negative samples
        num_nodes = self.graph_data.x.size(0)
        neg_edge_index = self._create_negative_samples(edge_index, num_nodes, num_edges)
        
        # Combine positive and negative edges
        all_edges = torch.cat([edge_index, neg_edge_index], dim=1)
        labels = torch.cat([torch.ones(num_edges), torch.zeros(num_edges)])
        
        # Split data
        num_total_edges = all_edges.size(1)
        indices = np.arange(num_total_edges)
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        train_edges = all_edges[:, train_idx]
        test_edges = all_edges[:, test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        # Initialize model (using node embeddings for link prediction)
        input_dim = self.graph_data.x.size(1)
        
        self.link_predictor = GraphConvolutionalNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim  # Output embeddings
        ).to(self.device)
        
        # Link prediction head
        self.link_pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Move data to device
        data = self.graph_data.to(self.device)
        train_edges = train_edges.to(self.device)
        test_edges = test_edges.to(self.device)
        train_labels = train_labels.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # Training setup
        params = list(self.link_predictor.parameters()) + list(self.link_pred_head.parameters())
        optimizer = optim.Adam(params, lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        self.link_predictor.train()
        self.link_pred_head.train()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Get node embeddings
            node_embeddings = self.link_predictor(data.x, data.edge_index)
            
            # Get edge embeddings by concatenating node embeddings
            source_embeddings = node_embeddings[train_edges[0]]
            target_embeddings = node_embeddings[train_edges[1]]
            edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
            
            # Predict link probabilities
            link_probs = self.link_pred_head(edge_embeddings).squeeze()
            
            # Calculate loss
            loss = criterion(link_probs, train_labels.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluation
        self.link_predictor.eval()
        self.link_pred_head.eval()
        
        with torch.no_grad():
            node_embeddings = self.link_predictor(data.x, data.edge_index)
            
            source_embeddings = node_embeddings[test_edges[0]]
            target_embeddings = node_embeddings[test_edges[1]]
            edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
            
            link_probs = self.link_pred_head(edge_embeddings).squeeze()
            predictions = (link_probs > 0.5).float()
            
            accuracy = accuracy_score(test_labels.cpu(), predictions.cpu())
            precision = precision_score(test_labels.cpu(), predictions.cpu())
            recall = recall_score(test_labels.cpu(), predictions.cpu())
            f1 = f1_score(test_labels.cpu(), predictions.cpu())
        
        metrics = {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }
        
        logger.info(f"Link predictor training completed. Test accuracy: {accuracy:.4f}")
        return metrics
    
    def _create_negative_samples(self, edge_index: torch.Tensor, num_nodes: int, num_samples: int) -> torch.Tensor:
        """Create negative edge samples for link prediction"""
        negative_edges = []
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        
        while len(negative_edges) < num_samples:
            source = np.random.randint(0, num_nodes)
            target = np.random.randint(0, num_nodes)
            
            if source != target and (source, target) not in existing_edges:
                negative_edges.append([source, target])
        
        return torch.tensor(negative_edges).t()
    
    def predict_missing_links(self, 
                            node_pairs: Optional[List[Tuple[str, str]]] = None,
                            threshold: float = 0.5) -> List[LinkPrediction]:
        """Predict missing links in the graph"""
        if self.link_predictor is None or self.link_pred_head is None:
            raise ValueError("Link predictor not trained. Call train_link_predictor() first.")
        
        predictions = []
        
        # Move data to device
        data = self.graph_data.to(self.device)
        
        self.link_predictor.eval()
        self.link_pred_head.eval()
        
        with torch.no_grad():
            # Get node embeddings
            node_embeddings = self.link_predictor(data.x, data.edge_index)
            
            if node_pairs is None:
                # Generate all possible pairs (computationally expensive for large graphs)
                num_nodes = len(self.graph_data.node_ids)
                if num_nodes > 1000:
                    logger.warning("Large graph detected. Consider providing specific node pairs.")
                    return predictions
                
                node_pairs = []
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        node_pairs.append((self.graph_data.node_ids[i], self.graph_data.node_ids[j]))
            
            # Create node ID to index mapping
            node_to_idx = {node_id: idx for idx, node_id in enumerate(self.graph_data.node_ids)}
            
            for source_id, target_id in node_pairs:
                if source_id in node_to_idx and target_id in node_to_idx:
                    source_idx = node_to_idx[source_id]
                    target_idx = node_to_idx[target_id]
                    
                    # Get embeddings
                    source_emb = node_embeddings[source_idx]
                    target_emb = node_embeddings[target_idx]
                    edge_emb = torch.cat([source_emb, target_emb], dim=0).unsqueeze(0)
                    
                    # Predict link probability
                    link_prob = self.link_pred_head(edge_emb).squeeze().item()
                    
                    if link_prob > threshold:
                        prediction = LinkPrediction(
                            source_id=source_id,
                            target_id=target_id,
                            predicted_relationship="PREDICTED_LINK",
                            confidence=link_prob
                        )
                        predictions.append(prediction)
        
        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        return predictions
    
    def generate_node_embeddings(self, embedding_dim: int = 128) -> List[GraphEmbedding]:
        """Generate node embeddings using trained models"""
        if self.node_classifier is None:
            logger.warning("No trained model available. Training a basic embedding model...")
            self.train_embedding_model(embedding_dim)
        
        embeddings = []
        
        # Move data to device
        data = self.graph_data.to(self.device)
        
        # Use node classifier as embedding generator
        self.node_classifier.eval()
        with torch.no_grad():
            # Get embeddings from the second-to-last layer
            x = data.x
            edge_index = data.edge_index
            
            # Forward pass through all but last layer
            for i in range(len(self.node_classifier.convs) - 1):
                x = self.node_classifier.convs[i](x, edge_index)
                if i < len(self.node_classifier.batch_norms):
                    x = self.node_classifier.batch_norms[i](x)
                x = F.relu(x)
                x = self.node_classifier.dropout(x)
            
            # Convert to embeddings
            for i, node_id in enumerate(self.graph_data.node_ids):
                embedding = x[i].cpu().numpy().tolist()
                
                graph_embedding = GraphEmbedding(
                    node_id=node_id,
                    embedding=embedding,
                    embedding_dim=len(embedding)
                )
                embeddings.append(graph_embedding)
        
        return embeddings
    
    def train_embedding_model(self, embedding_dim: int = 128):
        """Train a basic embedding model"""
        logger.info("Training embedding model...")
        
        if self.graph_data is None:
            raise ValueError("Graph data not loaded. Call load_graph_data() first.")
        
        input_dim = self.graph_data.x.size(1)
        
        self.node_classifier = GraphConvolutionalNetwork(
            input_dim=input_dim,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim
        ).to(self.device)
        
        # Simple autoencoder training
        data = self.graph_data.to(self.device)
        optimizer = optim.Adam(self.node_classifier.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        self.node_classifier.train()
        for epoch in range(100):
            optimizer.zero_grad()
            
            # Autoencoder loss
            out = self.node_classifier(data.x, data.edge_index)
            loss = criterion(out, data.x[:, :embedding_dim])  # Reconstruct first features
            
            loss.backward()
            optimizer.step()
            
            if epoch % 25 == 0:
                logger.info(f"Embedding training epoch {epoch}, Loss: {loss.item():.4f}")
    
    def analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze graph structure using GNN insights"""
        if self.graph_data is None:
            raise ValueError("Graph data not loaded. Call load_graph_data() first.")
        
        # Convert to NetworkX for analysis
        G = to_networkx(self.graph_data, to_undirected=True)
        
        # Basic graph statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        # Centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, num_nodes))
        closeness_centrality = nx.closeness_centrality(G)
        
        # Community detection
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            modularity = nx.community.modularity(G, communities)
        except:
            communities = []
            modularity = 0
        
        # Clustering
        clustering_coefficient = nx.average_clustering(G)
        
        # Path lengths
        if nx.is_connected(G):
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            # For disconnected graphs
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
            diameter = nx.diameter(subgraph)
        
        # Node importance (using GNN embeddings if available)
        node_importance = {}
        if self.node_classifier is not None:
            embeddings = self.generate_node_embeddings()
            # Calculate importance based on embedding magnitude
            for emb in embeddings:
                importance = np.linalg.norm(emb.embedding)
                node_importance[emb.node_id] = importance
        
        analysis = {
            "basic_statistics": {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "density": density,
                "clustering_coefficient": clustering_coefficient
            },
            "centrality_measures": {
                "top_degree_centrality": sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
                "top_betweenness_centrality": sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
                "top_closeness_centrality": sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            },
            "path_analysis": {
                "average_path_length": avg_path_length,
                "diameter": diameter
            },
            "community_structure": {
                "num_communities": len(communities),
                "modularity": modularity,
                "largest_community_size": max(len(c) for c in communities) if communities else 0
            },
            "node_importance": sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:20] if node_importance else [],
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        return analysis
    
    def export_results(self, output_dir: str):
        """Export GNN analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export node predictions
        if self.node_classifier is not None:
            node_predictions = self.predict_node_properties()
            with open(os.path.join(output_dir, "node_predictions.json"), 'w') as f:
                json.dump([asdict(pred) for pred in node_predictions], f, indent=2)
        
        # Export link predictions
        if self.link_predictor is not None:
            # Sample a few node pairs for demonstration
            sample_pairs = [
                (self.graph_data.node_ids[i], self.graph_data.node_ids[j])
                for i in range(min(10, len(self.graph_data.node_ids)))
                for j in range(i + 1, min(i + 5, len(self.graph_data.node_ids)))
            ]
            link_predictions = self.predict_missing_links(sample_pairs)
            with open(os.path.join(output_dir, "link_predictions.json"), 'w') as f:
                json.dump([asdict(pred) for pred in link_predictions], f, indent=2)
        
        # Export embeddings
        embeddings = self.generate_node_embeddings()
        with open(os.path.join(output_dir, "node_embeddings.json"), 'w') as f:
            json.dump([asdict(emb) for emb in embeddings], f, indent=2)
        
        # Export graph analysis
        analysis = self.analyze_graph_structure()
        with open(os.path.join(output_dir, "graph_analysis.json"), 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"GNN analysis results exported to {output_dir}")
    
    def close(self):
        """Close database connections"""
        self.driver.close()

# Example usage
if __name__ == "__main__":
    gnn_analytics = GNNAnalytics(
        "bolt://neo4j-lb.nexus-knowledge-graph:7687",
        "neo4j",
        "nexus-architect-graph-password"
    )
    
    try:
        # Load graph data
        graph_data = gnn_analytics.load_graph_data()
        
        # Train models
        node_metrics = gnn_analytics.train_node_classifier()
        print(f"Node classification metrics: {node_metrics}")
        
        link_metrics = gnn_analytics.train_link_predictor()
        print(f"Link prediction metrics: {link_metrics}")
        
        # Generate insights
        analysis = gnn_analytics.analyze_graph_structure()
        print(f"Graph analysis completed: {analysis['basic_statistics']}")
        
        # Export results
        gnn_analytics.export_results("/tmp/gnn_results")
        
    finally:
        gnn_analytics.close()

