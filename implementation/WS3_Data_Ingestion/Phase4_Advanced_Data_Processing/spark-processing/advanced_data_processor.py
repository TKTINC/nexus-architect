"""
Advanced Data Processor for Nexus Architect
Implements large-scale data processing with Apache Spark for distributed transformation,
machine learning pipelines, and graph processing capabilities.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

# Spark and ML imports
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import *
    from pyspark.ml.classification import *
    from pyspark.ml.clustering import *
    from pyspark.ml.evaluation import *
    from pyspark.ml.tuning import *
    from pyspark.ml.recommendation import *
    from pyspark.graphx import *
except ImportError:
    print("PySpark not available - using mock implementations for development")

# Database and caching
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import pandas as pd
import numpy as np

# Web framework
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
processing_requests = Counter('spark_processing_requests_total', 'Total processing requests', ['operation', 'status'])
processing_duration = Histogram('spark_processing_duration_seconds', 'Processing duration', ['operation'])
active_jobs = Gauge('spark_active_jobs', 'Number of active Spark jobs')
processed_records = Counter('spark_processed_records_total', 'Total processed records', ['source_type'])

@dataclass
class ProcessingConfig:
    """Configuration for data processing operations"""
    spark_master: str = "local[*]"
    app_name: str = "nexus-advanced-processor"
    max_memory: str = "4g"
    executor_cores: int = 2
    executor_memory: str = "2g"
    sql_adaptive_enabled: bool = True
    sql_adaptive_coalesce_enabled: bool = True
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    
@dataclass
class DataSource:
    """Data source configuration"""
    source_id: str
    source_type: str  # git, documentation, project_management, communication
    connection_config: Dict[str, Any]
    schema_config: Dict[str, Any]
    processing_config: Dict[str, Any]

@dataclass
class ProcessingJob:
    """Data processing job definition"""
    job_id: str
    job_type: str  # transformation, classification, clustering, graph_analysis
    source_data: List[str]
    target_schema: Dict[str, Any]
    processing_steps: List[Dict[str, Any]]
    output_config: Dict[str, Any]
    created_at: datetime
    status: str = "pending"

class SparkSessionManager:
    """Manages Spark session lifecycle and configuration"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.spark = None
        self.sql_context = None
        
    def get_session(self) -> SparkSession:
        """Get or create Spark session"""
        if self.spark is None:
            try:
                self.spark = SparkSession.builder \
                    .appName(self.config.app_name) \
                    .master(self.config.spark_master) \
                    .config("spark.executor.memory", self.config.executor_memory) \
                    .config("spark.executor.cores", str(self.config.executor_cores)) \
                    .config("spark.driver.memory", self.config.max_memory) \
                    .config("spark.sql.adaptive.enabled", str(self.config.sql_adaptive_enabled)) \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", str(self.config.sql_adaptive_coalesce_enabled)) \
                    .config("spark.serializer", self.config.serializer) \
                    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                    .getOrCreate()
                
                self.spark.sparkContext.setLogLevel("WARN")
                self.sql_context = self.spark.sql
                
                logger.info(f"Spark session created: {self.spark.version}")
                
            except Exception as e:
                logger.error(f"Failed to create Spark session: {e}")
                # Fallback to pandas for development
                self.spark = None
                
        return self.spark
    
    def stop_session(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
            self.spark = None
            logger.info("Spark session stopped")

class DataTransformationEngine:
    """Advanced data transformation using Spark"""
    
    def __init__(self, spark_manager: SparkSessionManager):
        self.spark_manager = spark_manager
        self.transformations = {}
        
    def register_transformation(self, name: str, transformation_func):
        """Register a custom transformation function"""
        self.transformations[name] = transformation_func
        logger.info(f"Registered transformation: {name}")
    
    def apply_schema_normalization(self, df: DataFrame, target_schema: Dict[str, Any]) -> DataFrame:
        """Apply schema normalization to DataFrame"""
        try:
            spark = self.spark_manager.get_session()
            if not spark:
                return df
            
            # Create target schema
            schema_fields = []
            for field_name, field_config in target_schema.items():
                field_type = field_config.get('type', 'string')
                nullable = field_config.get('nullable', True)
                
                if field_type == 'string':
                    spark_type = StringType()
                elif field_type == 'integer':
                    spark_type = IntegerType()
                elif field_type == 'long':
                    spark_type = LongType()
                elif field_type == 'double':
                    spark_type = DoubleType()
                elif field_type == 'boolean':
                    spark_type = BooleanType()
                elif field_type == 'timestamp':
                    spark_type = TimestampType()
                else:
                    spark_type = StringType()
                
                schema_fields.append(StructField(field_name, spark_type, nullable))
            
            target_spark_schema = StructType(schema_fields)
            
            # Apply transformations to match target schema
            for field in target_spark_schema.fields:
                field_name = field.name
                if field_name not in df.columns:
                    # Add missing column with null values
                    df = df.withColumn(field_name, lit(None).cast(field.dataType))
                else:
                    # Cast existing column to target type
                    df = df.withColumn(field_name, col(field_name).cast(field.dataType))
            
            # Select only target schema columns
            df = df.select(*[field.name for field in target_spark_schema.fields])
            
            return df
            
        except Exception as e:
            logger.error(f"Schema normalization failed: {e}")
            return df
    
    def apply_deduplication(self, df: DataFrame, key_columns: List[str]) -> DataFrame:
        """Apply deduplication based on key columns"""
        try:
            if not key_columns:
                return df
            
            # Add row number for deduplication
            window_spec = Window.partitionBy(*key_columns).orderBy(desc("updated_at"))
            df_with_row_num = df.withColumn("row_num", row_number().over(window_spec))
            
            # Keep only first occurrence
            deduplicated_df = df_with_row_num.filter(col("row_num") == 1).drop("row_num")
            
            return deduplicated_df
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return df
    
    def apply_temporal_alignment(self, df: DataFrame, timestamp_column: str = "timestamp") -> DataFrame:
        """Apply temporal alignment and synchronization"""
        try:
            if timestamp_column not in df.columns:
                return df
            
            # Standardize timestamp format
            df = df.withColumn(
                timestamp_column,
                to_timestamp(col(timestamp_column))
            )
            
            # Add temporal features
            df = df.withColumn("year", year(col(timestamp_column))) \
                   .withColumn("month", month(col(timestamp_column))) \
                   .withColumn("day", dayofmonth(col(timestamp_column))) \
                   .withColumn("hour", hour(col(timestamp_column))) \
                   .withColumn("day_of_week", dayofweek(col(timestamp_column)))
            
            return df
            
        except Exception as e:
            logger.error(f"Temporal alignment failed: {e}")
            return df
    
    def apply_cross_reference_resolution(self, df: DataFrame, reference_mappings: Dict[str, Any]) -> DataFrame:
        """Apply cross-reference resolution and entity linking"""
        try:
            for field_name, mapping_config in reference_mappings.items():
                if field_name in df.columns:
                    mapping_type = mapping_config.get('type', 'direct')
                    mapping_data = mapping_config.get('mapping', {})
                    
                    if mapping_type == 'direct':
                        # Direct value mapping
                        mapping_expr = create_map([lit(x) for x in sum(mapping_data.items(), ())])
                        df = df.withColumn(
                            f"{field_name}_resolved",
                            coalesce(mapping_expr[col(field_name)], col(field_name))
                        )
                    elif mapping_type == 'regex':
                        # Regex-based mapping
                        for pattern, replacement in mapping_data.items():
                            df = df.withColumn(
                                f"{field_name}_resolved",
                                regexp_replace(col(field_name), pattern, replacement)
                            )
            
            return df
            
        except Exception as e:
            logger.error(f"Cross-reference resolution failed: {e}")
            return df

class MachineLearningPipeline:
    """Machine learning pipeline for data classification and analysis"""
    
    def __init__(self, spark_manager: SparkSessionManager):
        self.spark_manager = spark_manager
        self.models = {}
        self.pipelines = {}
        
    def create_classification_pipeline(self, features: List[str], label_column: str) -> Pipeline:
        """Create classification pipeline"""
        try:
            # Feature preparation
            string_indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep") 
                             for col in features if col.endswith("_string")]
            
            vector_assembler = VectorAssembler(
                inputCols=[f"{col}_indexed" if col.endswith("_string") else col for col in features],
                outputCol="features"
            )
            
            # Classification algorithm
            classifier = RandomForestClassifier(
                featuresCol="features",
                labelCol=label_column,
                numTrees=100,
                maxDepth=10,
                seed=42
            )
            
            # Create pipeline
            pipeline_stages = string_indexers + [vector_assembler, classifier]
            pipeline = Pipeline(stages=pipeline_stages)
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create classification pipeline: {e}")
            return None
    
    def create_clustering_pipeline(self, features: List[str], k: int = 10) -> Pipeline:
        """Create clustering pipeline"""
        try:
            # Feature preparation
            string_indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep") 
                             for col in features if col.endswith("_string")]
            
            vector_assembler = VectorAssembler(
                inputCols=[f"{col}_indexed" if col.endswith("_string") else col for col in features],
                outputCol="features"
            )
            
            # Clustering algorithm
            kmeans = KMeans(
                featuresCol="features",
                predictionCol="cluster",
                k=k,
                seed=42
            )
            
            # Create pipeline
            pipeline_stages = string_indexers + [vector_assembler, kmeans]
            pipeline = Pipeline(stages=pipeline_stages)
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create clustering pipeline: {e}")
            return None
    
    def train_model(self, pipeline: Pipeline, training_data: DataFrame, model_name: str) -> Any:
        """Train machine learning model"""
        try:
            with processing_duration.labels(operation="model_training").time():
                model = pipeline.fit(training_data)
                self.models[model_name] = model
                
                logger.info(f"Model trained successfully: {model_name}")
                return model
                
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            processing_requests.labels(operation="model_training", status="error").inc()
            return None
    
    def evaluate_classification_model(self, model: Any, test_data: DataFrame) -> Dict[str, float]:
        """Evaluate classification model performance"""
        try:
            predictions = model.transform(test_data)
            
            evaluator_accuracy = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="accuracy"
            )
            
            evaluator_f1 = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName="f1"
            )
            
            accuracy = evaluator_accuracy.evaluate(predictions)
            f1_score = evaluator_f1.evaluate(predictions)
            
            return {
                "accuracy": accuracy,
                "f1_score": f1_score
            }
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}

class GraphProcessor:
    """Graph processing for relationship analysis"""
    
    def __init__(self, spark_manager: SparkSessionManager):
        self.spark_manager = spark_manager
        
    def create_entity_graph(self, entities_df: DataFrame, relationships_df: DataFrame) -> Dict[str, Any]:
        """Create entity relationship graph"""
        try:
            spark = self.spark_manager.get_session()
            if not spark:
                return {}
            
            # Create vertices (entities)
            vertices = entities_df.select(
                col("entity_id").alias("id"),
                col("entity_type"),
                col("entity_name"),
                col("properties")
            )
            
            # Create edges (relationships)
            edges = relationships_df.select(
                col("source_entity_id").alias("src"),
                col("target_entity_id").alias("dst"),
                col("relationship_type"),
                col("relationship_strength"),
                col("properties")
            )
            
            # Basic graph statistics
            vertex_count = vertices.count()
            edge_count = edges.count()
            
            # Calculate degree distribution
            in_degrees = edges.groupBy("dst").count().withColumnRenamed("count", "in_degree")
            out_degrees = edges.groupBy("src").count().withColumnRenamed("count", "out_degree")
            
            degree_stats = {
                "vertex_count": vertex_count,
                "edge_count": edge_count,
                "avg_in_degree": in_degrees.agg(avg("in_degree")).collect()[0][0] if vertex_count > 0 else 0,
                "avg_out_degree": out_degrees.agg(avg("out_degree")).collect()[0][0] if vertex_count > 0 else 0
            }
            
            return {
                "vertices": vertices,
                "edges": edges,
                "statistics": degree_stats
            }
            
        except Exception as e:
            logger.error(f"Graph creation failed: {e}")
            return {}
    
    def analyze_graph_communities(self, graph_data: Dict[str, Any]) -> DataFrame:
        """Analyze graph communities using connected components"""
        try:
            if not graph_data or "vertices" not in graph_data:
                return None
            
            vertices = graph_data["vertices"]
            edges = graph_data["edges"]
            
            # Simple community detection using connected components
            # In a full implementation, this would use GraphX or NetworkX
            
            # For now, return a placeholder result
            communities = vertices.withColumn("community_id", lit(0))
            
            return communities
            
        except Exception as e:
            logger.error(f"Community analysis failed: {e}")
            return None

class AdvancedDataProcessor:
    """Main advanced data processor orchestrating all processing capabilities"""
    
    def __init__(self):
        self.config = ProcessingConfig()
        self.spark_manager = SparkSessionManager(self.config)
        self.transformation_engine = DataTransformationEngine(self.spark_manager)
        self.ml_pipeline = MachineLearningPipeline(self.spark_manager)
        self.graph_processor = GraphProcessor(self.spark_manager)
        
        # Database connections
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'nexus_architect'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )
        
        # Processing jobs queue
        self.processing_jobs = {}
        self.job_executor = ThreadPoolExecutor(max_workers=4)
        
    def get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None
    
    def load_data_from_source(self, source_config: DataSource) -> Optional[DataFrame]:
        """Load data from configured source"""
        try:
            spark = self.spark_manager.get_session()
            if not spark:
                # Fallback to pandas for development
                return self._load_data_pandas(source_config)
            
            source_type = source_config.source_type
            connection_config = source_config.connection_config
            
            if source_type == "database":
                # Load from database
                df = spark.read \
                    .format("jdbc") \
                    .option("url", connection_config["url"]) \
                    .option("dbtable", connection_config["table"]) \
                    .option("user", connection_config["user"]) \
                    .option("password", connection_config["password"]) \
                    .load()
                    
            elif source_type == "json":
                # Load from JSON files
                df = spark.read.json(connection_config["path"])
                
            elif source_type == "parquet":
                # Load from Parquet files
                df = spark.read.parquet(connection_config["path"])
                
            elif source_type == "csv":
                # Load from CSV files
                df = spark.read \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .csv(connection_config["path"])
            else:
                logger.error(f"Unsupported source type: {source_type}")
                return None
            
            processed_records.labels(source_type=source_type).inc(df.count())
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return None
    
    def _load_data_pandas(self, source_config: DataSource) -> Optional[pd.DataFrame]:
        """Fallback data loading using pandas"""
        try:
            source_type = source_config.source_type
            connection_config = source_config.connection_config
            
            if source_type == "database":
                conn = self.get_db_connection()
                if conn:
                    df = pd.read_sql(connection_config.get("query", "SELECT * FROM data_sources LIMIT 1000"), conn)
                    conn.close()
                    return df
            elif source_type == "json":
                return pd.read_json(connection_config["path"])
            elif source_type == "csv":
                return pd.read_csv(connection_config["path"])
            
            return None
            
        except Exception as e:
            logger.error(f"Pandas data loading failed: {e}")
            return None
    
    def process_data_transformation(self, job: ProcessingJob) -> Dict[str, Any]:
        """Process data transformation job"""
        try:
            with processing_duration.labels(operation="transformation").time():
                active_jobs.inc()
                
                results = []
                for source_id in job.source_data:
                    # Load source data
                    source_config = self._get_source_config(source_id)
                    if not source_config:
                        continue
                    
                    df = self.load_data_from_source(source_config)
                    if df is None:
                        continue
                    
                    # Apply transformations
                    for step in job.processing_steps:
                        step_type = step.get("type")
                        step_config = step.get("config", {})
                        
                        if step_type == "schema_normalization":
                            df = self.transformation_engine.apply_schema_normalization(df, job.target_schema)
                        elif step_type == "deduplication":
                            df = self.transformation_engine.apply_deduplication(df, step_config.get("key_columns", []))
                        elif step_type == "temporal_alignment":
                            df = self.transformation_engine.apply_temporal_alignment(df, step_config.get("timestamp_column", "timestamp"))
                        elif step_type == "cross_reference_resolution":
                            df = self.transformation_engine.apply_cross_reference_resolution(df, step_config.get("mappings", {}))
                    
                    # Store results
                    output_config = job.output_config
                    if output_config.get("format") == "parquet":
                        if hasattr(df, 'write'):  # Spark DataFrame
                            df.write.mode("overwrite").parquet(output_config["path"])
                        else:  # Pandas DataFrame
                            df.to_parquet(output_config["path"])
                    
                    results.append({
                        "source_id": source_id,
                        "record_count": df.count() if hasattr(df, 'count') else len(df),
                        "status": "completed"
                    })
                
                active_jobs.dec()
                processing_requests.labels(operation="transformation", status="success").inc()
                
                return {
                    "job_id": job.job_id,
                    "status": "completed",
                    "results": results,
                    "completed_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            active_jobs.dec()
            processing_requests.labels(operation="transformation", status="error").inc()
            logger.error(f"Data transformation failed: {e}")
            return {
                "job_id": job.job_id,
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat()
            }
    
    def process_classification_job(self, job: ProcessingJob) -> Dict[str, Any]:
        """Process machine learning classification job"""
        try:
            with processing_duration.labels(operation="classification").time():
                active_jobs.inc()
                
                # Load training data
                training_data = None
                for source_id in job.source_data:
                    source_config = self._get_source_config(source_id)
                    if source_config:
                        df = self.load_data_from_source(source_config)
                        if df is not None:
                            training_data = df
                            break
                
                if training_data is None:
                    raise Exception("No training data available")
                
                # Create and train model
                features = job.processing_steps[0].get("config", {}).get("features", [])
                label_column = job.processing_steps[0].get("config", {}).get("label_column", "label")
                
                pipeline = self.ml_pipeline.create_classification_pipeline(features, label_column)
                if pipeline is None:
                    raise Exception("Failed to create classification pipeline")
                
                model = self.ml_pipeline.train_model(pipeline, training_data, job.job_id)
                if model is None:
                    raise Exception("Model training failed")
                
                # Evaluate model if test data is available
                evaluation_results = {}
                if len(job.source_data) > 1:
                    test_source_config = self._get_source_config(job.source_data[1])
                    if test_source_config:
                        test_data = self.load_data_from_source(test_source_config)
                        if test_data is not None:
                            evaluation_results = self.ml_pipeline.evaluate_classification_model(model, test_data)
                
                active_jobs.dec()
                processing_requests.labels(operation="classification", status="success").inc()
                
                return {
                    "job_id": job.job_id,
                    "status": "completed",
                    "model_name": job.job_id,
                    "evaluation": evaluation_results,
                    "completed_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            active_jobs.dec()
            processing_requests.labels(operation="classification", status="error").inc()
            logger.error(f"Classification job failed: {e}")
            return {
                "job_id": job.job_id,
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat()
            }
    
    def process_graph_analysis_job(self, job: ProcessingJob) -> Dict[str, Any]:
        """Process graph analysis job"""
        try:
            with processing_duration.labels(operation="graph_analysis").time():
                active_jobs.inc()
                
                # Load entity and relationship data
                entities_df = None
                relationships_df = None
                
                for i, source_id in enumerate(job.source_data):
                    source_config = self._get_source_config(source_id)
                    if source_config:
                        df = self.load_data_from_source(source_config)
                        if df is not None:
                            if i == 0:
                                entities_df = df
                            elif i == 1:
                                relationships_df = df
                
                if entities_df is None or relationships_df is None:
                    raise Exception("Insufficient data for graph analysis")
                
                # Create and analyze graph
                graph_data = self.graph_processor.create_entity_graph(entities_df, relationships_df)
                if not graph_data:
                    raise Exception("Graph creation failed")
                
                # Analyze communities
                communities = self.graph_processor.analyze_graph_communities(graph_data)
                
                # Store results
                output_config = job.output_config
                if output_config.get("format") == "json":
                    results = {
                        "statistics": graph_data.get("statistics", {}),
                        "community_count": communities.select("community_id").distinct().count() if communities else 0
                    }
                    
                    with open(output_config["path"], 'w') as f:
                        json.dump(results, f, indent=2)
                
                active_jobs.dec()
                processing_requests.labels(operation="graph_analysis", status="success").inc()
                
                return {
                    "job_id": job.job_id,
                    "status": "completed",
                    "graph_statistics": graph_data.get("statistics", {}),
                    "completed_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            active_jobs.dec()
            processing_requests.labels(operation="graph_analysis", status="error").inc()
            logger.error(f"Graph analysis failed: {e}")
            return {
                "job_id": job.job_id,
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat()
            }
    
    def _get_source_config(self, source_id: str) -> Optional[DataSource]:
        """Get source configuration by ID"""
        try:
            # In a real implementation, this would query the database
            # For now, return a mock configuration
            return DataSource(
                source_id=source_id,
                source_type="database",
                connection_config={
                    "url": f"jdbc:postgresql://{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}",
                    "table": "processed_data",
                    "user": self.db_config['user'],
                    "password": self.db_config['password']
                },
                schema_config={},
                processing_config={}
            )
        except Exception as e:
            logger.error(f"Failed to get source config: {e}")
            return None
    
    def submit_processing_job(self, job: ProcessingJob) -> str:
        """Submit processing job for execution"""
        try:
            job.status = "submitted"
            self.processing_jobs[job.job_id] = job
            
            # Submit job to executor
            if job.job_type == "transformation":
                future = self.job_executor.submit(self.process_data_transformation, job)
            elif job.job_type == "classification":
                future = self.job_executor.submit(self.process_classification_job, job)
            elif job.job_type == "graph_analysis":
                future = self.job_executor.submit(self.process_graph_analysis_job, job)
            else:
                raise Exception(f"Unsupported job type: {job.job_type}")
            
            # Store future for result retrieval
            self.redis_client.set(f"job_future:{job.job_id}", "submitted", ex=3600)
            
            logger.info(f"Processing job submitted: {job.job_id}")
            return job.job_id
            
        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get processing job status"""
        try:
            if job_id in self.processing_jobs:
                job = self.processing_jobs[job_id]
                return {
                    "job_id": job_id,
                    "status": job.status,
                    "job_type": job.job_type,
                    "created_at": job.created_at.isoformat(),
                    "source_count": len(job.source_data)
                }
            else:
                return {"job_id": job_id, "status": "not_found"}
                
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {"job_id": job_id, "status": "error", "error": str(e)}

# Flask application
app = Flask(__name__)
CORS(app)

# Initialize processor
processor = AdvancedDataProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check Spark session
        spark_status = "available" if processor.spark_manager.get_session() else "unavailable"
        
        # Check database connection
        db_conn = processor.get_db_connection()
        db_status = "connected" if db_conn else "disconnected"
        if db_conn:
            db_conn.close()
        
        # Check Redis connection
        try:
            processor.redis_client.ping()
            redis_status = "connected"
        except:
            redis_status = "disconnected"
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "spark": spark_status,
                "database": db_status,
                "redis": redis_status
            }
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/v1/processing/jobs', methods=['POST'])
def submit_job():
    """Submit processing job"""
    try:
        data = request.get_json()
        
        job = ProcessingJob(
            job_id=str(uuid.uuid4()),
            job_type=data.get('job_type', 'transformation'),
            source_data=data.get('source_data', []),
            target_schema=data.get('target_schema', {}),
            processing_steps=data.get('processing_steps', []),
            output_config=data.get('output_config', {}),
            created_at=datetime.utcnow()
        )
        
        job_id = processor.submit_processing_job(job)
        
        return jsonify({
            "job_id": job_id,
            "status": "submitted",
            "message": "Processing job submitted successfully"
        })
        
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/processing/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    try:
        status = processor.get_job_status(job_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/processing/jobs', methods=['GET'])
def list_jobs():
    """List all processing jobs"""
    try:
        jobs = []
        for job_id, job in processor.processing_jobs.items():
            jobs.append({
                "job_id": job_id,
                "job_type": job.job_type,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "source_count": len(job.source_data)
            })
        
        return jsonify({"jobs": jobs})
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8001))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Advanced Data Processor on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

