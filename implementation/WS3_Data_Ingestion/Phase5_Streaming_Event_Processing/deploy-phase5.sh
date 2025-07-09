#!/bin/bash

# WS3 Phase 5: Real-Time Streaming & Event Processing Deployment Script
# Deploys Apache Kafka, stream processing, and real-time analytics infrastructure

set -e

echo "🚀 Starting WS3 Phase 5: Real-Time Streaming & Event Processing Deployment"

# Configuration
NAMESPACE="nexus-streaming"
KAFKA_REPLICAS=3
ZOOKEEPER_REPLICAS=3
REDIS_REPLICAS=3

# Create namespace
echo "📦 Creating Kubernetes namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy Zookeeper cluster
echo "🐘 Deploying Zookeeper cluster..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: zookeeper
  namespace: $NAMESPACE
spec:
  serviceName: zookeeper-headless
  replicas: $ZOOKEEPER_REPLICAS
  selector:
    matchLabels:
      app: zookeeper
  template:
    metadata:
      labels:
        app: zookeeper
    spec:
      containers:
      - name: zookeeper
        image: confluentinc/cp-zookeeper:7.4.0
        ports:
        - containerPort: 2181
        - containerPort: 2888
        - containerPort: 3888
        env:
        - name: ZOOKEEPER_CLIENT_PORT
          value: "2181"
        - name: ZOOKEEPER_TICK_TIME
          value: "2000"
        - name: ZOOKEEPER_INIT_LIMIT
          value: "5"
        - name: ZOOKEEPER_SYNC_LIMIT
          value: "2"
        - name: ZOOKEEPER_SERVERS
          value: "zookeeper-0.zookeeper-headless.nexus-streaming.svc.cluster.local:2888:3888;zookeeper-1.zookeeper-headless.nexus-streaming.svc.cluster.local:2888:3888;zookeeper-2.zookeeper-headless.nexus-streaming.svc.cluster.local:2888:3888"
        volumeMounts:
        - name: zookeeper-data
          mountPath: /var/lib/zookeeper/data
        - name: zookeeper-logs
          mountPath: /var/lib/zookeeper/log
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
  volumeClaimTemplates:
  - metadata:
      name: zookeeper-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
  - metadata:
      name: zookeeper-logs
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: zookeeper-headless
  namespace: $NAMESPACE
spec:
  clusterIP: None
  selector:
    app: zookeeper
  ports:
  - name: client
    port: 2181
  - name: follower
    port: 2888
  - name: election
    port: 3888
---
apiVersion: v1
kind: Service
metadata:
  name: zookeeper
  namespace: $NAMESPACE
spec:
  selector:
    app: zookeeper
  ports:
  - name: client
    port: 2181
    targetPort: 2181
EOF

# Wait for Zookeeper to be ready
echo "⏳ Waiting for Zookeeper to be ready..."
kubectl wait --for=condition=ready pod -l app=zookeeper -n $NAMESPACE --timeout=300s

# Deploy Kafka cluster
echo "📡 Deploying Kafka cluster..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: kafka
  namespace: $NAMESPACE
spec:
  serviceName: kafka-headless
  replicas: $KAFKA_REPLICAS
  selector:
    matchLabels:
      app: kafka
  template:
    metadata:
      labels:
        app: kafka
    spec:
      containers:
      - name: kafka
        image: confluentinc/cp-kafka:7.4.0
        ports:
        - containerPort: 9092
        - containerPort: 9093
        env:
        - name: KAFKA_BROKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: KAFKA_ZOOKEEPER_CONNECT
          value: "zookeeper:2181"
        - name: KAFKA_LISTENERS
          value: "PLAINTEXT://0.0.0.0:9092,PLAINTEXT_HOST://0.0.0.0:9093"
        - name: KAFKA_ADVERTISED_LISTENERS
          value: "PLAINTEXT://\$(hostname).kafka-headless.nexus-streaming.svc.cluster.local:9092,PLAINTEXT_HOST://localhost:9093"
        - name: KAFKA_LISTENER_SECURITY_PROTOCOL_MAP
          value: "PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT"
        - name: KAFKA_INTER_BROKER_LISTENER_NAME
          value: "PLAINTEXT"
        - name: KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR
          value: "3"
        - name: KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR
          value: "3"
        - name: KAFKA_TRANSACTION_STATE_LOG_MIN_ISR
          value: "2"
        - name: KAFKA_DEFAULT_REPLICATION_FACTOR
          value: "3"
        - name: KAFKA_MIN_IN_SYNC_REPLICAS
          value: "2"
        - name: KAFKA_LOG_RETENTION_HOURS
          value: "168"
        - name: KAFKA_LOG_SEGMENT_BYTES
          value: "1073741824"
        - name: KAFKA_LOG_RETENTION_CHECK_INTERVAL_MS
          value: "300000"
        - name: KAFKA_NUM_PARTITIONS
          value: "12"
        - name: KAFKA_COMPRESSION_TYPE
          value: "snappy"
        - name: KAFKA_JVM_PERFORMANCE_OPTS
          value: "-server -XX:+UseG1GC -XX:MaxGCPauseMillis=20 -XX:InitiatingHeapOccupancyPercent=35 -XX:+ExplicitGCInvokesConcurrent -Djava.awt.headless=true"
        - name: KAFKA_HEAP_OPTS
          value: "-Xmx1G -Xms1G"
        volumeMounts:
        - name: kafka-data
          mountPath: /var/lib/kafka/data
        resources:
          requests:
            memory: "1.5Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: kafka-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: kafka-headless
  namespace: $NAMESPACE
spec:
  clusterIP: None
  selector:
    app: kafka
  ports:
  - name: kafka
    port: 9092
---
apiVersion: v1
kind: Service
metadata:
  name: kafka
  namespace: $NAMESPACE
spec:
  selector:
    app: kafka
  ports:
  - name: kafka
    port: 9092
    targetPort: 9092
  type: ClusterIP
EOF

# Wait for Kafka to be ready
echo "⏳ Waiting for Kafka to be ready..."
kubectl wait --for=condition=ready pod -l app=kafka -n $NAMESPACE --timeout=300s

# Deploy Redis cluster for caching
echo "🔴 Deploying Redis cluster..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  serviceName: redis-headless
  replicas: $REDIS_REPLICAS
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7.0-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - --appendonly
        - "yes"
        - --cluster-enabled
        - "yes"
        - --cluster-config-file
        - nodes.conf
        - --cluster-node-timeout
        - "5000"
        volumeMounts:
        - name: redis-data
          mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis-headless
  namespace: $NAMESPACE
spec:
  clusterIP: None
  selector:
    app: redis
  ports:
  - name: redis
    port: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: $NAMESPACE
spec:
  selector:
    app: redis
  ports:
  - name: redis
    port: 6379
    targetPort: 6379
EOF

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s

# Deploy Kafka Streaming Manager
echo "🌊 Deploying Kafka Streaming Manager..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kafka-streaming-manager
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: kafka-streaming-manager
  template:
    metadata:
      labels:
        app: kafka-streaming-manager
    spec:
      containers:
      - name: streaming-manager
        image: python:3.11-slim
        ports:
        - containerPort: 8090
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
        - name: DATABASE_HOST
          value: "postgresql.nexus-core.svc.cluster.local"
        - name: DATABASE_PORT
          value: "5432"
        - name: DATABASE_NAME
          value: "nexus_architect"
        - name: DATABASE_USER
          value: "postgres"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-secret
              key: password
        command:
        - /bin/bash
        - -c
        - |
          pip install kafka-python redis psycopg2-binary prometheus-client confluent-kafka
          python -c "
          import sys
          sys.path.append('/app')
          from kafka_streaming_manager import StreamingManager
          import time
          
          config = {
              'kafka': {'bootstrap_servers': 'kafka:9092'},
              'database': {
                  'host': 'postgresql.nexus-core.svc.cluster.local',
                  'port': 5432,
                  'database': 'nexus_architect',
                  'user': 'postgres',
                  'password': 'nexus_secure_password_2024'
              },
              'redis': {'host': 'redis', 'port': 6379, 'db': 0}
          }
          
          manager = StreamingManager(config)
          print('Kafka Streaming Manager started')
          
          while True:
              time.sleep(60)
              health = manager.get_stream_health()
              print(f'Stream health: {health}')
          "
        volumeMounts:
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: app-code
        configMap:
          name: streaming-code
---
apiVersion: v1
kind: Service
metadata:
  name: kafka-streaming-manager
  namespace: $NAMESPACE
spec:
  selector:
    app: kafka-streaming-manager
  ports:
  - name: metrics
    port: 8090
    targetPort: 8090
EOF

# Deploy Stream Processor
echo "⚙️ Deploying Stream Processor..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stream-processor
  namespace: $NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stream-processor
  template:
    metadata:
      labels:
        app: stream-processor
    spec:
      containers:
      - name: processor
        image: python:3.11-slim
        ports:
        - containerPort: 8091
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
        - name: DATABASE_HOST
          value: "postgresql.nexus-core.svc.cluster.local"
        - name: DATABASE_PORT
          value: "5432"
        - name: DATABASE_NAME
          value: "nexus_architect"
        - name: DATABASE_USER
          value: "postgres"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-secret
              key: password
        command:
        - /bin/bash
        - -c
        - |
          pip install kafka-python redis psycopg2-binary prometheus-client scikit-learn numpy pandas networkx
          python -c "
          import sys
          sys.path.append('/app')
          from stream_processor import StreamTransformer, EventCorrelationEngine
          from kafka_streaming_manager import KafkaEventConsumer, EventType
          import time
          
          config = {
              'database': {
                  'host': 'postgresql.nexus-core.svc.cluster.local',
                  'port': 5432,
                  'database': 'nexus_architect',
                  'user': 'postgres',
                  'password': 'nexus_secure_password_2024'
              },
              'redis': {'host': 'redis', 'port': 6379, 'db': 0}
          }
          
          transformer = StreamTransformer(config)
          correlator = EventCorrelationEngine(config)
          
          # Create consumer for all topics
          topics = [
              'nexus-code-changes',
              'nexus-documentation-updates',
              'nexus-project-updates',
              'nexus-communication-messages',
              'nexus-system-metrics',
              'nexus-quality-alerts',
              'nexus-knowledge-extraction',
              'nexus-user-actions'
          ]
          
          consumer = KafkaEventConsumer('kafka:9092', 'stream-processor-group', topics)
          
          def process_event(event):
              processed = transformer.transform_event(event)
              correlations = correlator.correlate_event(processed)
              processed.correlations = correlations
              print(f'Processed event: {event.event_id} with {len(correlations)} correlations')
          
          # Register processing callback
          for event_type in EventType:
              consumer.register_callback(event_type, process_event)
          
          print('Stream Processor started')
          consumer.start_consuming()
          "
        volumeMounts:
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: app-code
        configMap:
          name: streaming-code
---
apiVersion: v1
kind: Service
metadata:
  name: stream-processor
  namespace: $NAMESPACE
spec:
  selector:
    app: stream-processor
  ports:
  - name: metrics
    port: 8091
    targetPort: 8091
EOF

# Deploy Real-Time Analytics Engine
echo "📊 Deploying Real-Time Analytics Engine..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: real-time-analytics
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: real-time-analytics
  template:
    metadata:
      labels:
        app: real-time-analytics
    spec:
      containers:
      - name: analytics
        image: python:3.11-slim
        ports:
        - containerPort: 8004
        - containerPort: 8092
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
        - name: DATABASE_HOST
          value: "postgresql.nexus-core.svc.cluster.local"
        - name: DATABASE_PORT
          value: "5432"
        - name: DATABASE_NAME
          value: "nexus_architect"
        - name: DATABASE_USER
          value: "postgres"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgresql-secret
              key: password
        command:
        - /bin/bash
        - -c
        - |
          pip install kafka-python redis psycopg2-binary prometheus-client scikit-learn numpy pandas flask flask-cors scipy
          python -c "
          import sys
          sys.path.append('/app')
          from real_time_analytics import RealTimeAnalyticsEngine, create_analytics_api
          from prometheus_client import start_http_server
          import threading
          
          config = {
              'database': {
                  'host': 'postgresql.nexus-core.svc.cluster.local',
                  'port': 5432,
                  'database': 'nexus_architect',
                  'user': 'postgres',
                  'password': 'nexus_secure_password_2024'
              },
              'redis': {'host': 'redis', 'port': 6379, 'db': 0}
          }
          
          # Start Prometheus metrics server
          start_http_server(8092)
          
          # Initialize analytics engine
          analytics_engine = RealTimeAnalyticsEngine(config)
          
          # Create Flask API
          app = create_analytics_api(analytics_engine)
          
          print('Real-Time Analytics Engine started')
          print('API: http://0.0.0.0:8004')
          print('Metrics: http://0.0.0.0:8092')
          
          app.run(host='0.0.0.0', port=8004, debug=False)
          "
        volumeMounts:
        - name: app-code
          mountPath: /app
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: app-code
        configMap:
          name: streaming-code
---
apiVersion: v1
kind: Service
metadata:
  name: real-time-analytics
  namespace: $NAMESPACE
spec:
  selector:
    app: real-time-analytics
  ports:
  - name: api
    port: 8004
    targetPort: 8004
  - name: metrics
    port: 8092
    targetPort: 8092
  type: ClusterIP
EOF

# Create ConfigMap with application code
echo "📝 Creating application code ConfigMap..."
kubectl create configmap streaming-code \
  --from-file=kafka_streaming_manager.py=kafka-infrastructure/kafka_streaming_manager.py \
  --from-file=stream_processor.py=stream-processing/stream_processor.py \
  --from-file=real_time_analytics.py=analytics-engine/real_time_analytics.py \
  -n $NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy Prometheus monitoring
echo "📈 Deploying Prometheus monitoring..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:v2.45.0
        ports:
        - containerPort: 9090
        args:
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus/
        - --web.console.libraries=/etc/prometheus/console_libraries
        - --web.console.templates=/etc/prometheus/consoles
        - --web.enable-lifecycle
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus/
        - name: prometheus-storage
          mountPath: /prometheus/
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: $NAMESPACE
spec:
  selector:
    app: prometheus
  ports:
  - name: prometheus
    port: 9090
    targetPort: 9090
  type: ClusterIP
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: $NAMESPACE
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    scrape_configs:
    - job_name: 'kafka-streaming-manager'
      static_configs:
      - targets: ['kafka-streaming-manager:8090']
    
    - job_name: 'stream-processor'
      static_configs:
      - targets: ['stream-processor:8091']
    
    - job_name: 'real-time-analytics'
      static_configs:
      - targets: ['real-time-analytics:8092']
    
    - job_name: 'kafka-exporter'
      static_configs:
      - targets: ['kafka-exporter:9308']
EOF

# Deploy Kafka Exporter for monitoring
echo "📊 Deploying Kafka Exporter..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kafka-exporter
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kafka-exporter
  template:
    metadata:
      labels:
        app: kafka-exporter
    spec:
      containers:
      - name: kafka-exporter
        image: danielqsj/kafka-exporter:v1.6.0
        ports:
        - containerPort: 9308
        args:
        - --kafka.server=kafka:9092
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: kafka-exporter
  namespace: $NAMESPACE
spec:
  selector:
    app: kafka-exporter
  ports:
  - name: metrics
    port: 9308
    targetPort: 9308
EOF

# Create Ingress for external access
echo "🌐 Creating Ingress for external access..."
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: streaming-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/cors-allow-origin: "*"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization"
spec:
  rules:
  - host: streaming.nexus-architect.local
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: real-time-analytics
            port:
              number: 8004
      - path: /metrics
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
EOF

# Wait for all deployments to be ready
echo "⏳ Waiting for all deployments to be ready..."
kubectl wait --for=condition=available deployment --all -n $NAMESPACE --timeout=600s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

# Create topics
echo "📋 Creating Kafka topics..."
kubectl exec -it kafka-0 -n $NAMESPACE -- kafka-topics --create --topic nexus-code-changes --bootstrap-server localhost:9092 --partitions 12 --replication-factor 3 --if-not-exists
kubectl exec -it kafka-0 -n $NAMESPACE -- kafka-topics --create --topic nexus-documentation-updates --bootstrap-server localhost:9092 --partitions 8 --replication-factor 3 --if-not-exists
kubectl exec -it kafka-0 -n $NAMESPACE -- kafka-topics --create --topic nexus-project-updates --bootstrap-server localhost:9092 --partitions 6 --replication-factor 3 --if-not-exists
kubectl exec -it kafka-0 -n $NAMESPACE -- kafka-topics --create --topic nexus-communication-messages --bootstrap-server localhost:9092 --partitions 10 --replication-factor 3 --if-not-exists
kubectl exec -it kafka-0 -n $NAMESPACE -- kafka-topics --create --topic nexus-system-metrics --bootstrap-server localhost:9092 --partitions 4 --replication-factor 3 --if-not-exists
kubectl exec -it kafka-0 -n $NAMESPACE -- kafka-topics --create --topic nexus-quality-alerts --bootstrap-server localhost:9092 --partitions 3 --replication-factor 3 --if-not-exists
kubectl exec -it kafka-0 -n $NAMESPACE -- kafka-topics --create --topic nexus-knowledge-extraction --bootstrap-server localhost:9092 --partitions 8 --replication-factor 3 --if-not-exists
kubectl exec -it kafka-0 -n $NAMESPACE -- kafka-topics --create --topic nexus-user-actions --bootstrap-server localhost:9092 --partitions 6 --replication-factor 3 --if-not-exists

# List topics to verify
echo "📋 Listing Kafka topics..."
kubectl exec -it kafka-0 -n $NAMESPACE -- kafka-topics --list --bootstrap-server localhost:9092

# Display access information
echo ""
echo "🎉 WS3 Phase 5: Real-Time Streaming & Event Processing Deployment Complete!"
echo ""
echo "📊 Access Information:"
echo "  Real-Time Analytics API: http://streaming.nexus-architect.local/api"
echo "  Prometheus Metrics: http://streaming.nexus-architect.local/metrics"
echo "  Kafka Bootstrap Servers: kafka.nexus-streaming.svc.cluster.local:9092"
echo "  Redis: redis.nexus-streaming.svc.cluster.local:6379"
echo ""
echo "🔍 Monitoring:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl logs -f deployment/real-time-analytics -n $NAMESPACE"
echo "  kubectl logs -f deployment/stream-processor -n $NAMESPACE"
echo ""
echo "🧪 Testing:"
echo "  curl http://streaming.nexus-architect.local/api/health"
echo "  curl http://streaming.nexus-architect.local/api/dashboard"
echo ""
echo "✅ Phase 5 Infrastructure Ready for Real-Time Event Processing!"

