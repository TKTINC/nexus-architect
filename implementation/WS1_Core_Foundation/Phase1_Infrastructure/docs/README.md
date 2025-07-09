# WS1 Phase 1: Infrastructure Foundation and Basic Security

## Overview

This phase establishes the core infrastructure foundation for Nexus Architect, including:
- Kubernetes cluster with Calico CNI
- PostgreSQL database cluster with high availability
- Redis cluster for caching and sessions
- MinIO object storage for files and AI models
- HashiCorp Vault for secrets management
- Prometheus and Grafana monitoring stack

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ PostgreSQL  │  │    Redis    │  │    MinIO    │        │
│  │   Cluster   │  │   Cluster   │  │   Storage   │        │
│  │             │  │             │  │             │        │
│  │ Primary +   │  │ Master +    │  │ 4-node      │        │
│  │ 2 Replicas  │  │ 2 Replicas  │  │ Cluster     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────────────────────────────┐  │
│  │   Vault     │  │         Monitoring                  │  │
│  │  Cluster    │  │                                     │  │
│  │             │  │  ┌─────────────┐ ┌─────────────┐   │  │
│  │ 3-node HA   │  │  │ Prometheus  │ │   Grafana   │   │  │
│  │ Auto-unseal │  │  │   Server    │ │  Dashboard  │   │  │
│  └─────────────┘  │  └─────────────┘ └─────────────┘   │  │
│                    └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Kubernetes Infrastructure
- **Calico CNI**: Network policies and security
- **Namespace**: `nexus-architect` for application components
- **RBAC**: Service accounts and role bindings
- **Storage**: Persistent volumes for data persistence

### 2. PostgreSQL Database Cluster
- **Configuration**: Primary-replica setup with automatic failover
- **High Availability**: 1 primary + 2 read replicas
- **Storage**: 100GB persistent storage per instance
- **Security**: SSL/TLS encryption, role-based access
- **Backup**: Automated daily backups to MinIO

**Connection Details:**
- Primary: `postgresql-primary.nexus-architect.svc.cluster.local:5432`
- Replicas: `postgresql-replica.nexus-architect.svc.cluster.local:5432`
- Database: `nexus_architect`
- App User: `nexus_app` / `nexus_app_pass_2024!`
- Admin User: `postgres` / `nexus_admin_2024!`

### 3. Redis Cluster
- **Configuration**: Master-replica setup for high availability
- **Caching**: Application cache and session storage
- **Storage**: 10GB persistent storage for master
- **Security**: Password authentication
- **Performance**: Optimized for low latency

**Connection Details:**
- Master: `redis-master.nexus-architect.svc.cluster.local:6379`
- Replicas: `redis-replica.nexus-architect.svc.cluster.local:6379`
- Password: `nexus_redis_2024!`

### 4. MinIO Object Storage
- **Configuration**: 4-node distributed cluster
- **Storage**: 50GB per node (200GB total)
- **Buckets**: Pre-configured for different data types
- **Security**: Access key authentication, bucket policies
- **Features**: Versioning, lifecycle management

**Connection Details:**
- API: `minio-api.nexus-architect.svc.cluster.local:9000`
- Console: `minio-console.nexus-architect.svc.cluster.local:9001`
- Access Key: `minioadmin`
- Secret Key: `nexus_minio_2024!`

**Pre-configured Buckets:**
- `nexus-architect-data`: Application data
- `nexus-architect-models`: AI models and artifacts
- `nexus-architect-documents`: Document storage
- `nexus-architect-backups`: System backups
- `nexus-architect-logs`: Log archives

### 5. HashiCorp Vault
- **Configuration**: 3-node HA cluster with auto-unseal
- **Storage**: PostgreSQL backend for high availability
- **Security**: Transit seal, audit logging
- **Features**: Dynamic secrets, PKI, encryption

**Connection Details:**
- API: `vault-active.nexus-architect.svc.cluster.local:8200`
- Root Token: Check vault-init job logs

**Configured Engines:**
- KV v2: Application secrets at `nexus-architect/`
- Database: Dynamic PostgreSQL credentials
- PKI: Internal certificate authority
- Transit: Encryption as a service

### 6. Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Exporters**: Node, PostgreSQL, Redis exporters
- **Alerts**: Infrastructure and application monitoring

**Connection Details:**
- Prometheus: `prometheus.monitoring.svc.cluster.local:9090`
- Grafana: `grafana.monitoring.svc.cluster.local:3000`
- Grafana User: `admin` / `nexus_grafana_2024!`

## Deployment

### Prerequisites
- Kubernetes cluster (v1.28+)
- kubectl configured and connected
- Sufficient resources: 16 CPU cores, 32GB RAM, 500GB storage
- StorageClass for persistent volumes

### Quick Start
```bash
# Clone repository and navigate to Phase 1
cd implementation/WS1_Core_Foundation/Phase1_Infrastructure

# Run deployment script
./deploy-phase1.sh

# Check deployment status
kubectl get pods -n nexus-architect
kubectl get pods -n monitoring
```

### Manual Deployment
```bash
# Deploy components individually
kubectl apply -f kubernetes/calico-config.yaml
kubectl apply -f postgresql/postgresql-cluster.yaml
kubectl apply -f redis/redis-cluster.yaml
kubectl apply -f minio/minio-cluster.yaml
kubectl apply -f vault/vault-cluster.yaml
kubectl apply -f monitoring/prometheus-grafana.yaml
```

### Verification
```bash
# Check all services are running
kubectl get all -n nexus-architect
kubectl get all -n monitoring

# Test database connectivity
kubectl exec -n nexus-architect postgresql-primary-0 -- psql -U postgres -d nexus_architect -c "SELECT 1;"

# Test Redis connectivity
kubectl exec -n nexus-architect redis-master-0 -- redis-cli -a nexus_redis_2024! ping

# Test MinIO connectivity
kubectl exec -n nexus-architect minio-0 -- mc ls nexus/

# Check Vault status
kubectl exec -n nexus-architect vault-0 -- vault status
```

## Security Considerations

### Network Security
- Calico network policies isolate traffic
- Pod-to-pod encryption enabled
- Service mesh ready architecture

### Data Security
- All data encrypted at rest
- TLS encryption for data in transit
- Vault manages all secrets and certificates

### Access Control
- RBAC for Kubernetes resources
- Database role-based access
- Vault policy-based access control

### Monitoring Security
- Audit logging enabled
- Security alerts configured
- Compliance monitoring ready

## Performance Tuning

### Database Optimization
- Connection pooling configured
- Read replicas for query distribution
- Automated vacuum and analyze

### Cache Optimization
- Redis memory policies configured
- Eviction strategies optimized
- Replication lag monitoring

### Storage Optimization
- MinIO distributed for performance
- Lifecycle policies for cost optimization
- Compression enabled for text files

## Troubleshooting

### Common Issues

1. **Pod Startup Failures**
   ```bash
   kubectl describe pod <pod-name> -n nexus-architect
   kubectl logs <pod-name> -n nexus-architect
   ```

2. **Storage Issues**
   ```bash
   kubectl get pv,pvc -n nexus-architect
   kubectl describe pvc <pvc-name> -n nexus-architect
   ```

3. **Network Connectivity**
   ```bash
   kubectl exec -it <pod-name> -n nexus-architect -- nslookup <service-name>
   kubectl get networkpolicies -n nexus-architect
   ```

4. **Vault Unsealing**
   ```bash
   kubectl logs job/vault-init -n nexus-architect
   kubectl exec -n nexus-architect vault-0 -- vault operator unseal <key>
   ```

### Health Checks
```bash
# Database health
kubectl exec -n nexus-architect postgresql-primary-0 -- pg_isready

# Redis health
kubectl exec -n nexus-architect redis-master-0 -- redis-cli -a nexus_redis_2024! ping

# MinIO health
kubectl exec -n nexus-architect minio-0 -- curl -f http://localhost:9000/minio/health/live

# Vault health
kubectl exec -n nexus-architect vault-0 -- vault status
```

## Backup and Recovery

### Database Backups
- Automated daily backups to MinIO
- Point-in-time recovery available
- Cross-region backup replication

### Configuration Backups
- Kubernetes manifests in Git
- Vault configuration exported
- Monitoring configuration versioned

### Disaster Recovery
- Multi-zone deployment ready
- Automated failover configured
- Recovery procedures documented

## Next Steps

After successful Phase 1 deployment:

1. **Verify all services are healthy**
2. **Test connectivity between components**
3. **Review monitoring dashboards**
4. **Proceed to WS1 Phase 2**: Authentication, Authorization & API Foundation

## Support

For issues or questions:
- Check logs: `kubectl logs <pod-name> -n nexus-architect`
- Review monitoring: Access Grafana dashboards
- Consult troubleshooting guide above
- Escalate to infrastructure team if needed

