"""
Nexus Architect Causal Reasoning Engine
Advanced causal inference and reasoning capabilities for knowledge graphs
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from neo4j import GraphDatabase
import networkx as nx
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CausalRelationType(str, Enum):
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    CONTRIBUTING_FACTOR = "contributing_factor"
    NECESSARY_CONDITION = "necessary_condition"
    SUFFICIENT_CONDITION = "sufficient_condition"
    CORRELATION = "correlation"
    SPURIOUS_CORRELATION = "spurious_correlation"

class TemporalRelationType(str, Enum):
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CONCURRENT = "concurrent"
    PERIODIC = "periodic"
    TRIGGERED_BY = "triggered_by"

@dataclass
class CausalHypothesis:
    id: str
    cause_entity: str
    effect_entity: str
    causal_type: CausalRelationType
    confidence: float
    evidence: List[Dict[str, Any]]
    temporal_pattern: Optional[TemporalRelationType]
    strength: float
    created_at: datetime

@dataclass
class TemporalPattern:
    id: str
    entities: List[str]
    pattern_type: TemporalRelationType
    frequency: Optional[float]
    duration: Optional[timedelta]
    confidence: float
    evidence: List[Dict[str, Any]]

@dataclass
class CausalChain:
    id: str
    entities: List[str]
    relationships: List[str]
    total_strength: float
    confidence: float
    path_length: int

class CausalReasoningEngine:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.causal_hypotheses: Dict[str, CausalHypothesis] = {}
        self.temporal_patterns: Dict[str, TemporalPattern] = {}
        self.causal_chains: Dict[str, CausalChain] = {}
        
        # Initialize causal discovery models
        self.causal_model = None
        self.temporal_model = None
        
    def discover_causal_relationships(self, 
                                   entity_types: Optional[List[str]] = None,
                                   time_window: Optional[Tuple[datetime, datetime]] = None) -> List[CausalHypothesis]:
        """Discover causal relationships in the knowledge graph"""
        logger.info("Starting causal relationship discovery...")
        
        # Get relevant data from graph
        graph_data = self._extract_graph_data(entity_types, time_window)
        
        # Apply causal discovery algorithms
        hypotheses = []
        
        # 1. Correlation-based discovery
        correlation_hypotheses = self._discover_correlations(graph_data)
        hypotheses.extend(correlation_hypotheses)
        
        # 2. Temporal precedence analysis
        temporal_hypotheses = self._discover_temporal_causality(graph_data)
        hypotheses.extend(temporal_hypotheses)
        
        # 3. Granger causality analysis
        granger_hypotheses = self._discover_granger_causality(graph_data)
        hypotheses.extend(granger_hypotheses)
        
        # 4. Structural causal model inference
        structural_hypotheses = self._discover_structural_causality(graph_data)
        hypotheses.extend(structural_hypotheses)
        
        # Store discovered hypotheses
        for hypothesis in hypotheses:
            self.causal_hypotheses[hypothesis.id] = hypothesis
        
        logger.info(f"Discovered {len(hypotheses)} causal hypotheses")
        return hypotheses
    
    def _extract_graph_data(self, 
                           entity_types: Optional[List[str]] = None,
                           time_window: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Extract relevant data from the knowledge graph"""
        
        with self.driver.session() as session:
            # Build query based on filters
            where_clauses = []
            if entity_types:
                type_filter = " OR ".join([f"n:{entity_type}" for entity_type in entity_types])
                where_clauses.append(f"({type_filter})")
            
            if time_window:
                start_time, end_time = time_window
                where_clauses.append(f"n.created_at >= '{start_time.isoformat()}' AND n.created_at <= '{end_time.isoformat()}'")
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "true"
            
            # Extract nodes
            nodes_query = f"""
            MATCH (n)
            WHERE {where_clause}
            RETURN n.id as id, labels(n) as labels, properties(n) as properties
            """
            
            nodes_result = session.run(nodes_query)
            nodes = [record for record in nodes_result]
            
            # Extract relationships
            relationships_query = f"""
            MATCH (n)-[r]->(m)
            WHERE {where_clause.replace('n.', 'n.')} AND {where_clause.replace('n.', 'm.')}
            RETURN n.id as source, type(r) as relationship_type, m.id as target, properties(r) as properties
            """
            
            relationships_result = session.run(relationships_query)
            relationships = [record for record in relationships_result]
            
            # Extract temporal data
            temporal_query = f"""
            MATCH (n)
            WHERE {where_clause} AND exists(n.created_at)
            RETURN n.id as id, n.created_at as timestamp, labels(n) as labels
            ORDER BY n.created_at
            """
            
            temporal_result = session.run(temporal_query)
            temporal_data = [record for record in temporal_result]
            
            # Extract metrics and events
            metrics_query = f"""
            MATCH (n:Metric)-[r:MEASURES]->(target)
            WHERE {where_clause.replace('n.', 'target.')}
            RETURN n.id as metric_id, n.name as metric_name, target.id as target_id, 
                   r.value as value, r.timestamp as timestamp
            ORDER BY r.timestamp
            """
            
            metrics_result = session.run(metrics_query)
            metrics_data = [record for record in metrics_result]
            
            # Extract incidents and their relationships
            incidents_query = f"""
            MATCH (i:Incident)-[r]->(affected)
            WHERE {where_clause.replace('n.', 'affected.')}
            RETURN i.id as incident_id, i.severity as severity, i.created_at as timestamp,
                   type(r) as relationship_type, affected.id as affected_entity
            ORDER BY i.created_at
            """
            
            incidents_result = session.run(incidents_query)
            incidents_data = [record for record in incidents_result]
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "temporal_data": temporal_data,
            "metrics_data": metrics_data,
            "incidents_data": incidents_data
        }
    
    def _discover_correlations(self, graph_data: Dict[str, Any]) -> List[CausalHypothesis]:
        """Discover correlations that might indicate causal relationships"""
        hypotheses = []
        
        # Convert metrics data to time series
        metrics_df = pd.DataFrame(graph_data["metrics_data"])
        if metrics_df.empty:
            return hypotheses
        
        # Group by metric and target
        metric_series = {}
        for _, row in metrics_df.iterrows():
            key = f"{row['metric_id']}_{row['target_id']}"
            if key not in metric_series:
                metric_series[key] = []
            metric_series[key].append({
                "timestamp": row["timestamp"],
                "value": row["value"]
            })
        
        # Calculate correlations between different metrics
        metric_keys = list(metric_series.keys())
        for i in range(len(metric_keys)):
            for j in range(i + 1, len(metric_keys)):
                key1, key2 = metric_keys[i], metric_keys[j]
                
                # Align time series
                series1 = pd.DataFrame(metric_series[key1])
                series2 = pd.DataFrame(metric_series[key2])
                
                if len(series1) < 3 or len(series2) < 3:
                    continue
                
                # Convert timestamps and merge
                series1['timestamp'] = pd.to_datetime(series1['timestamp'])
                series2['timestamp'] = pd.to_datetime(series2['timestamp'])
                
                merged = pd.merge_asof(
                    series1.sort_values('timestamp'),
                    series2.sort_values('timestamp'),
                    on='timestamp',
                    suffixes=('_1', '_2'),
                    tolerance=pd.Timedelta('1 hour')
                )
                
                if len(merged) < 3:
                    continue
                
                # Calculate correlation
                correlation = merged['value_1'].corr(merged['value_2'])
                
                if abs(correlation) > 0.7:  # Strong correlation threshold
                    # Determine potential causality direction
                    entity1 = key1.split('_')[1]
                    entity2 = key2.split('_')[1]
                    
                    # Create hypothesis
                    hypothesis = CausalHypothesis(
                        id=str(uuid.uuid4()),
                        cause_entity=entity1 if correlation > 0 else entity2,
                        effect_entity=entity2 if correlation > 0 else entity1,
                        causal_type=CausalRelationType.CORRELATION,
                        confidence=abs(correlation),
                        evidence=[{
                            "type": "correlation",
                            "correlation_coefficient": correlation,
                            "sample_size": len(merged),
                            "p_value": stats.pearsonr(merged['value_1'], merged['value_2'])[1]
                        }],
                        temporal_pattern=None,
                        strength=abs(correlation),
                        created_at=datetime.utcnow()
                    )
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _discover_temporal_causality(self, graph_data: Dict[str, Any]) -> List[CausalHypothesis]:
        """Discover causal relationships based on temporal precedence"""
        hypotheses = []
        
        # Analyze incident patterns
        incidents_df = pd.DataFrame(graph_data["incidents_data"])
        if incidents_df.empty:
            return hypotheses
        
        incidents_df['timestamp'] = pd.to_datetime(incidents_df['timestamp'])
        incidents_df = incidents_df.sort_values('timestamp')
        
        # Look for temporal patterns in incidents
        for i in range(len(incidents_df) - 1):
            current_incident = incidents_df.iloc[i]
            next_incident = incidents_df.iloc[i + 1]
            
            time_diff = next_incident['timestamp'] - current_incident['timestamp']
            
            # If incidents are close in time (within 1 hour), consider causal relationship
            if time_diff <= timedelta(hours=1):
                hypothesis = CausalHypothesis(
                    id=str(uuid.uuid4()),
                    cause_entity=current_incident['affected_entity'],
                    effect_entity=next_incident['affected_entity'],
                    causal_type=CausalRelationType.DIRECT_CAUSE,
                    confidence=0.8 if time_diff <= timedelta(minutes=15) else 0.6,
                    evidence=[{
                        "type": "temporal_precedence",
                        "time_difference_minutes": time_diff.total_seconds() / 60,
                        "cause_incident": current_incident['incident_id'],
                        "effect_incident": next_incident['incident_id']
                    }],
                    temporal_pattern=TemporalRelationType.PRECEDES,
                    strength=1.0 / (time_diff.total_seconds() / 60 + 1),  # Inverse of time difference
                    created_at=datetime.utcnow()
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _discover_granger_causality(self, graph_data: Dict[str, Any]) -> List[CausalHypothesis]:
        """Apply Granger causality test to discover causal relationships"""
        hypotheses = []
        
        # Convert metrics to time series format
        metrics_df = pd.DataFrame(graph_data["metrics_data"])
        if len(metrics_df) < 10:  # Need sufficient data for Granger test
            return hypotheses
        
        # Group metrics by entity
        entity_metrics = defaultdict(list)
        for _, row in metrics_df.iterrows():
            entity_metrics[row['target_id']].append({
                "timestamp": pd.to_datetime(row["timestamp"]),
                "metric": row["metric_name"],
                "value": row["value"]
            })
        
        # Test Granger causality between entities
        entity_pairs = list(entity_metrics.keys())
        for i in range(len(entity_pairs)):
            for j in range(i + 1, len(entity_pairs)):
                entity1, entity2 = entity_pairs[i], entity_pairs[j]
                
                # Prepare time series data
                series1 = pd.DataFrame(entity_metrics[entity1])
                series2 = pd.DataFrame(entity_metrics[entity2])
                
                if len(series1) < 5 or len(series2) < 5:
                    continue
                
                # Aggregate metrics by time windows
                series1_agg = series1.groupby(pd.Grouper(key='timestamp', freq='1H'))['value'].mean().dropna()
                series2_agg = series2.groupby(pd.Grouper(key='timestamp', freq='1H'))['value'].mean().dropna()
                
                if len(series1_agg) < 5 or len(series2_agg) < 5:
                    continue
                
                # Align time series
                aligned = pd.concat([series1_agg, series2_agg], axis=1, join='inner')
                aligned.columns = ['series1', 'series2']
                aligned = aligned.dropna()
                
                if len(aligned) < 5:
                    continue
                
                # Perform Granger causality test (simplified version)
                try:
                    # Test if series1 Granger-causes series2
                    granger_result_1_to_2 = self._granger_causality_test(
                        aligned['series1'].values, 
                        aligned['series2'].values
                    )
                    
                    # Test if series2 Granger-causes series1
                    granger_result_2_to_1 = self._granger_causality_test(
                        aligned['series2'].values, 
                        aligned['series1'].values
                    )
                    
                    # Create hypotheses based on results
                    if granger_result_1_to_2['p_value'] < 0.05:
                        hypothesis = CausalHypothesis(
                            id=str(uuid.uuid4()),
                            cause_entity=entity1,
                            effect_entity=entity2,
                            causal_type=CausalRelationType.INDIRECT_CAUSE,
                            confidence=1 - granger_result_1_to_2['p_value'],
                            evidence=[{
                                "type": "granger_causality",
                                "f_statistic": granger_result_1_to_2['f_statistic'],
                                "p_value": granger_result_1_to_2['p_value'],
                                "direction": "1_to_2"
                            }],
                            temporal_pattern=TemporalRelationType.PRECEDES,
                            strength=granger_result_1_to_2['f_statistic'] / 10,  # Normalized strength
                            created_at=datetime.utcnow()
                        )
                        hypotheses.append(hypothesis)
                    
                    if granger_result_2_to_1['p_value'] < 0.05:
                        hypothesis = CausalHypothesis(
                            id=str(uuid.uuid4()),
                            cause_entity=entity2,
                            effect_entity=entity1,
                            causal_type=CausalRelationType.INDIRECT_CAUSE,
                            confidence=1 - granger_result_2_to_1['p_value'],
                            evidence=[{
                                "type": "granger_causality",
                                "f_statistic": granger_result_2_to_1['f_statistic'],
                                "p_value": granger_result_2_to_1['p_value'],
                                "direction": "2_to_1"
                            }],
                            temporal_pattern=TemporalRelationType.PRECEDES,
                            strength=granger_result_2_to_1['f_statistic'] / 10,
                            created_at=datetime.utcnow()
                        )
                        hypotheses.append(hypothesis)
                        
                except Exception as e:
                    logger.warning(f"Granger causality test failed for {entity1}-{entity2}: {e}")
                    continue
        
        return hypotheses
    
    def _granger_causality_test(self, cause_series: np.ndarray, effect_series: np.ndarray, max_lag: int = 3) -> Dict[str, float]:
        """Simplified Granger causality test"""
        n = len(cause_series)
        
        # Prepare lagged data
        X = []
        y = []
        
        for i in range(max_lag, n):
            # Include lagged values of both series
            features = []
            
            # Lagged values of effect series (autoregressive component)
            for lag in range(1, max_lag + 1):
                features.append(effect_series[i - lag])
            
            # Lagged values of cause series
            for lag in range(1, max_lag + 1):
                features.append(cause_series[i - lag])
            
            X.append(features)
            y.append(effect_series[i])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 3:
            return {"f_statistic": 0, "p_value": 1.0}
        
        # Fit restricted model (without cause series)
        X_restricted = X[:, :max_lag]  # Only autoregressive terms
        
        # Fit unrestricted model (with cause series)
        X_unrestricted = X  # All terms
        
        try:
            # Calculate RSS for both models
            from sklearn.linear_model import LinearRegression
            
            model_restricted = LinearRegression().fit(X_restricted, y)
            model_unrestricted = LinearRegression().fit(X_unrestricted, y)
            
            rss_restricted = np.sum((y - model_restricted.predict(X_restricted)) ** 2)
            rss_unrestricted = np.sum((y - model_unrestricted.predict(X_unrestricted)) ** 2)
            
            # Calculate F-statistic
            n_obs = len(y)
            n_restricted = X_restricted.shape[1]
            n_unrestricted = X_unrestricted.shape[1]
            
            f_statistic = ((rss_restricted - rss_unrestricted) / (n_unrestricted - n_restricted)) / \
                         (rss_unrestricted / (n_obs - n_unrestricted))
            
            # Calculate p-value using F-distribution
            from scipy.stats import f
            p_value = 1 - f.cdf(f_statistic, n_unrestricted - n_restricted, n_obs - n_unrestricted)
            
            return {"f_statistic": f_statistic, "p_value": p_value}
            
        except Exception as e:
            logger.warning(f"Error in Granger test calculation: {e}")
            return {"f_statistic": 0, "p_value": 1.0}
    
    def _discover_structural_causality(self, graph_data: Dict[str, Any]) -> List[CausalHypothesis]:
        """Discover causal relationships using structural causal models"""
        hypotheses = []
        
        # Build graph structure
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data["nodes"]:
            G.add_node(node["id"], **node["properties"])
        
        # Add edges
        for rel in graph_data["relationships"]:
            G.add_edge(rel["source"], rel["target"], 
                      relationship_type=rel["relationship_type"],
                      **rel["properties"])
        
        # Analyze graph structure for causal patterns
        
        # 1. Common cause patterns (confounders)
        for node in G.nodes():
            out_neighbors = list(G.successors(node))
            if len(out_neighbors) >= 2:
                # Node might be a common cause
                for i in range(len(out_neighbors)):
                    for j in range(i + 1, len(out_neighbors)):
                        neighbor1, neighbor2 = out_neighbors[i], out_neighbors[j]
                        
                        # Check if there's no direct connection between neighbors
                        if not G.has_edge(neighbor1, neighbor2) and not G.has_edge(neighbor2, neighbor1):
                            # Potential spurious correlation due to common cause
                            hypothesis = CausalHypothesis(
                                id=str(uuid.uuid4()),
                                cause_entity=neighbor1,
                                effect_entity=neighbor2,
                                causal_type=CausalRelationType.SPURIOUS_CORRELATION,
                                confidence=0.7,
                                evidence=[{
                                    "type": "common_cause",
                                    "common_cause_entity": node,
                                    "pattern": "confounding"
                                }],
                                temporal_pattern=None,
                                strength=0.5,
                                created_at=datetime.utcnow()
                            )
                            hypotheses.append(hypothesis)
        
        # 2. Mediator patterns
        for path in nx.all_simple_paths(G, source=None, target=None, cutoff=3):
            if len(path) == 3:  # A -> B -> C pattern
                source, mediator, target = path
                
                # Check if there's also a direct path
                if G.has_edge(source, target):
                    # Potential mediation
                    hypothesis = CausalHypothesis(
                        id=str(uuid.uuid4()),
                        cause_entity=source,
                        effect_entity=target,
                        causal_type=CausalRelationType.INDIRECT_CAUSE,
                        confidence=0.8,
                        evidence=[{
                            "type": "mediation",
                            "mediator_entity": mediator,
                            "pattern": "indirect_causation"
                        }],
                        temporal_pattern=TemporalRelationType.PRECEDES,
                        strength=0.7,
                        created_at=datetime.utcnow()
                    )
                    hypotheses.append(hypothesis)
        
        # 3. Collider patterns (selection bias)
        for node in G.nodes():
            in_neighbors = list(G.predecessors(node))
            if len(in_neighbors) >= 2:
                # Node might be a collider
                for i in range(len(in_neighbors)):
                    for j in range(i + 1, len(in_neighbors)):
                        neighbor1, neighbor2 = in_neighbors[i], in_neighbors[j]
                        
                        # Collider can create spurious correlation
                        hypothesis = CausalHypothesis(
                            id=str(uuid.uuid4()),
                            cause_entity=neighbor1,
                            effect_entity=neighbor2,
                            causal_type=CausalRelationType.SPURIOUS_CORRELATION,
                            confidence=0.6,
                            evidence=[{
                                "type": "collider",
                                "collider_entity": node,
                                "pattern": "selection_bias"
                            }],
                            temporal_pattern=None,
                            strength=0.4,
                            created_at=datetime.utcnow()
                        )
                        hypotheses.append(hypothesis)
        
        return hypotheses
    
    def discover_temporal_patterns(self, 
                                 entity_types: Optional[List[str]] = None,
                                 pattern_types: Optional[List[TemporalRelationType]] = None) -> List[TemporalPattern]:
        """Discover temporal patterns in the knowledge graph"""
        logger.info("Starting temporal pattern discovery...")
        
        patterns = []
        
        # Get temporal data
        graph_data = self._extract_graph_data(entity_types)
        temporal_df = pd.DataFrame(graph_data["temporal_data"])
        
        if temporal_df.empty:
            return patterns
        
        temporal_df['timestamp'] = pd.to_datetime(temporal_df['timestamp'])
        temporal_df = temporal_df.sort_values('timestamp')
        
        # 1. Discover periodic patterns
        periodic_patterns = self._discover_periodic_patterns(temporal_df)
        patterns.extend(periodic_patterns)
        
        # 2. Discover sequence patterns
        sequence_patterns = self._discover_sequence_patterns(temporal_df)
        patterns.extend(sequence_patterns)
        
        # 3. Discover concurrent patterns
        concurrent_patterns = self._discover_concurrent_patterns(temporal_df)
        patterns.extend(concurrent_patterns)
        
        # Store discovered patterns
        for pattern in patterns:
            self.temporal_patterns[pattern.id] = pattern
        
        logger.info(f"Discovered {len(patterns)} temporal patterns")
        return patterns
    
    def _discover_periodic_patterns(self, temporal_df: pd.DataFrame) -> List[TemporalPattern]:
        """Discover periodic temporal patterns"""
        patterns = []
        
        # Group by entity type and analyze periodicity
        for label_group in temporal_df.groupby('labels'):
            labels, group_df = label_group
            
            if len(group_df) < 5:  # Need sufficient data
                continue
            
            # Analyze time intervals between events
            group_df = group_df.sort_values('timestamp')
            intervals = group_df['timestamp'].diff().dropna()
            
            if len(intervals) < 3:
                continue
            
            # Convert to hours for analysis
            interval_hours = intervals.dt.total_seconds() / 3600
            
            # Check for periodicity using autocorrelation
            if len(interval_hours) >= 10:
                # Simple periodicity detection
                mean_interval = interval_hours.mean()
                std_interval = interval_hours.std()
                
                # If intervals are relatively consistent, it might be periodic
                if std_interval / mean_interval < 0.5:  # Coefficient of variation < 0.5
                    pattern = TemporalPattern(
                        id=str(uuid.uuid4()),
                        entities=group_df['id'].tolist(),
                        pattern_type=TemporalRelationType.PERIODIC,
                        frequency=1 / mean_interval,  # Events per hour
                        duration=timedelta(hours=mean_interval),
                        confidence=1 - (std_interval / mean_interval),
                        evidence=[{
                            "type": "periodicity_analysis",
                            "mean_interval_hours": mean_interval,
                            "std_interval_hours": std_interval,
                            "coefficient_of_variation": std_interval / mean_interval,
                            "sample_size": len(interval_hours)
                        }]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _discover_sequence_patterns(self, temporal_df: pd.DataFrame) -> List[TemporalPattern]:
        """Discover sequential temporal patterns"""
        patterns = []
        
        # Look for common sequences of entity types
        temporal_df['hour'] = temporal_df['timestamp'].dt.hour
        temporal_df['day'] = temporal_df['timestamp'].dt.date
        
        # Group by day and analyze sequences
        for day, day_group in temporal_df.groupby('day'):
            day_group = day_group.sort_values('timestamp')
            
            if len(day_group) < 3:
                continue
            
            # Extract sequence of entity types
            sequence = day_group['labels'].apply(lambda x: x[0] if x else 'Unknown').tolist()
            
            # Look for repeating subsequences
            for seq_len in range(2, min(5, len(sequence))):
                for i in range(len(sequence) - seq_len + 1):
                    subseq = sequence[i:i + seq_len]
                    
                    # Count occurrences of this subsequence
                    count = 0
                    for j in range(len(sequence) - seq_len + 1):
                        if sequence[j:j + seq_len] == subseq:
                            count += 1
                    
                    # If subsequence appears multiple times, it's a pattern
                    if count >= 2:
                        pattern = TemporalPattern(
                            id=str(uuid.uuid4()),
                            entities=day_group.iloc[i:i + seq_len]['id'].tolist(),
                            pattern_type=TemporalRelationType.PRECEDES,
                            frequency=count,
                            duration=None,
                            confidence=count / (len(sequence) - seq_len + 1),
                            evidence=[{
                                "type": "sequence_analysis",
                                "sequence": subseq,
                                "occurrences": count,
                                "sequence_length": seq_len,
                                "day": str(day)
                            }]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _discover_concurrent_patterns(self, temporal_df: pd.DataFrame) -> List[TemporalPattern]:
        """Discover concurrent temporal patterns"""
        patterns = []
        
        # Group events by time windows
        temporal_df['time_window'] = temporal_df['timestamp'].dt.floor('1H')
        
        for window, window_group in temporal_df.groupby('time_window'):
            if len(window_group) >= 2:  # Multiple events in same window
                entities = window_group['id'].tolist()
                entity_types = window_group['labels'].apply(lambda x: x[0] if x else 'Unknown').tolist()
                
                pattern = TemporalPattern(
                    id=str(uuid.uuid4()),
                    entities=entities,
                    pattern_type=TemporalRelationType.CONCURRENT,
                    frequency=None,
                    duration=timedelta(hours=1),
                    confidence=0.8,
                    evidence=[{
                        "type": "concurrency_analysis",
                        "time_window": str(window),
                        "entity_count": len(entities),
                        "entity_types": entity_types
                    }]
                )
                patterns.append(pattern)
        
        return patterns
    
    def find_causal_chains(self, 
                          start_entity: str, 
                          end_entity: str, 
                          max_length: int = 5) -> List[CausalChain]:
        """Find causal chains between two entities"""
        logger.info(f"Finding causal chains from {start_entity} to {end_entity}")
        
        chains = []
        
        # Build causal graph from hypotheses
        causal_graph = nx.DiGraph()
        
        for hypothesis in self.causal_hypotheses.values():
            if hypothesis.causal_type not in [CausalRelationType.SPURIOUS_CORRELATION]:
                causal_graph.add_edge(
                    hypothesis.cause_entity,
                    hypothesis.effect_entity,
                    weight=hypothesis.strength,
                    confidence=hypothesis.confidence,
                    hypothesis_id=hypothesis.id
                )
        
        # Find all simple paths
        try:
            paths = list(nx.all_simple_paths(
                causal_graph, 
                start_entity, 
                end_entity, 
                cutoff=max_length
            ))
            
            for path in paths:
                # Calculate chain strength and confidence
                total_strength = 1.0
                min_confidence = 1.0
                relationships = []
                
                for i in range(len(path) - 1):
                    edge_data = causal_graph[path[i]][path[i + 1]]
                    total_strength *= edge_data['weight']
                    min_confidence = min(min_confidence, edge_data['confidence'])
                    relationships.append(edge_data['hypothesis_id'])
                
                chain = CausalChain(
                    id=str(uuid.uuid4()),
                    entities=path,
                    relationships=relationships,
                    total_strength=total_strength,
                    confidence=min_confidence,
                    path_length=len(path) - 1
                )
                chains.append(chain)
                self.causal_chains[chain.id] = chain
                
        except nx.NetworkXNoPath:
            logger.info(f"No causal path found from {start_entity} to {end_entity}")
        
        # Sort by strength and confidence
        chains.sort(key=lambda x: (x.total_strength * x.confidence), reverse=True)
        
        logger.info(f"Found {len(chains)} causal chains")
        return chains
    
    def explain_causal_relationship(self, 
                                  cause_entity: str, 
                                  effect_entity: str) -> Dict[str, Any]:
        """Provide detailed explanation of causal relationship"""
        
        # Find direct hypotheses
        direct_hypotheses = [
            h for h in self.causal_hypotheses.values()
            if h.cause_entity == cause_entity and h.effect_entity == effect_entity
        ]
        
        # Find causal chains
        chains = self.find_causal_chains(cause_entity, effect_entity)
        
        # Find temporal patterns
        relevant_patterns = [
            p for p in self.temporal_patterns.values()
            if cause_entity in p.entities and effect_entity in p.entities
        ]
        
        explanation = {
            "cause_entity": cause_entity,
            "effect_entity": effect_entity,
            "direct_hypotheses": [asdict(h) for h in direct_hypotheses],
            "causal_chains": [asdict(c) for c in chains[:5]],  # Top 5 chains
            "temporal_patterns": [asdict(p) for p in relevant_patterns],
            "confidence_score": max([h.confidence for h in direct_hypotheses] + [0]),
            "strength_score": max([h.strength for h in direct_hypotheses] + [0]),
            "explanation_generated_at": datetime.utcnow().isoformat()
        }
        
        return explanation
    
    def get_causal_insights(self, entity: str) -> Dict[str, Any]:
        """Get comprehensive causal insights for an entity"""
        
        # Find what this entity causes
        causes = [
            h for h in self.causal_hypotheses.values()
            if h.cause_entity == entity
        ]
        
        # Find what causes this entity
        effects = [
            h for h in self.causal_hypotheses.values()
            if h.effect_entity == entity
        ]
        
        # Find temporal patterns involving this entity
        patterns = [
            p for p in self.temporal_patterns.values()
            if entity in p.entities
        ]
        
        # Find causal chains starting from this entity
        outgoing_chains = [
            c for c in self.causal_chains.values()
            if c.entities[0] == entity
        ]
        
        # Find causal chains ending at this entity
        incoming_chains = [
            c for c in self.causal_chains.values()
            if c.entities[-1] == entity
        ]
        
        insights = {
            "entity": entity,
            "causes_count": len(causes),
            "effects_count": len(effects),
            "temporal_patterns_count": len(patterns),
            "outgoing_chains_count": len(outgoing_chains),
            "incoming_chains_count": len(incoming_chains),
            "top_causes": [asdict(h) for h in sorted(causes, key=lambda x: x.confidence, reverse=True)[:5]],
            "top_effects": [asdict(h) for h in sorted(effects, key=lambda x: x.confidence, reverse=True)[:5]],
            "temporal_patterns": [asdict(p) for p in patterns],
            "strongest_outgoing_chains": [asdict(c) for c in sorted(outgoing_chains, key=lambda x: x.total_strength, reverse=True)[:3]],
            "strongest_incoming_chains": [asdict(c) for c in sorted(incoming_chains, key=lambda x: x.total_strength, reverse=True)[:3]],
            "insights_generated_at": datetime.utcnow().isoformat()
        }
        
        return insights
    
    def export_causal_model(self, output_path: str):
        """Export the causal model to a file"""
        model_data = {
            "causal_hypotheses": {k: asdict(v) for k, v in self.causal_hypotheses.items()},
            "temporal_patterns": {k: asdict(v) for k, v in self.temporal_patterns.items()},
            "causal_chains": {k: asdict(v) for k, v in self.causal_chains.items()},
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        logger.info(f"Causal model exported to {output_path}")
    
    def close(self):
        """Close database connections"""
        self.driver.close()

# Example usage
if __name__ == "__main__":
    engine = CausalReasoningEngine(
        "bolt://neo4j-lb.nexus-knowledge-graph:7687",
        "neo4j",
        "nexus-architect-graph-password"
    )
    
    try:
        # Discover causal relationships
        hypotheses = engine.discover_causal_relationships()
        print(f"Discovered {len(hypotheses)} causal hypotheses")
        
        # Discover temporal patterns
        patterns = engine.discover_temporal_patterns()
        print(f"Discovered {len(patterns)} temporal patterns")
        
        # Export model
        engine.export_causal_model("/tmp/causal_model.json")
        
    finally:
        engine.close()

