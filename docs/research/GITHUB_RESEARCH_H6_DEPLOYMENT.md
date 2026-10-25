# Hour 6 Research: Production Deployment Patterns

## Overview
Research on production deployment patterns for Shadow AI, based on GitHub projects.

---

## Container Orchestration

### Docker Compose (37.2k stars)
**GitHub**: https://github.com/docker/compose

**Key Concepts**:
- Multi-container application definition
- Service dependencies and networking
- Volume management for persistence
- Environment-based configuration

**Pattern**:
```yaml
services:
  shadow-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://cache:6379
    volumes:
      - shadow-data:/data
    depends_on:
      - db
      - cache
  
  db:
    image: postgres:15
    volumes:
      - db-data:/var/lib/postgresql/data
    
  cache:
    image: redis:7
```

**For Shadow AI**:
- Simple deployment for development/testing
- Service orchestration without K8s complexity
- Volume persistence for memory storage
- Network isolation between components

---

### Kubernetes (121k stars)
**GitHub**: https://github.com/kubernetes/kubernetes

**Key Concepts**:
- Pods: Smallest deployable units
- Deployments: Declarative updates for Pods
- Services: Network abstraction
- ConfigMaps/Secrets: Configuration management
- Ingress: External access routing
- PersistentVolumes: Storage abstraction

**Pattern**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shadow-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: shadow-ai
  template:
    metadata:
      labels:
        app: shadow-ai
    spec:
      containers:
      - name: api
        image: shadow-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: shadow-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
```

**For Shadow AI**:
- Horizontal scaling via replicas
- Self-healing with health checks
- Rolling updates for zero-downtime deployment
- Resource limits for agent processes

---

### Podman (31.2k stars)
**GitHub**: https://github.com/containers/podman

**Key Concepts**:
- Daemonless container management
- Rootless container execution
- Docker-compatible CLI
- Pod concept (similar to K8s pods)
- Systemd integration for services

**Key Advantages**:
- Better security (rootless by default)
- No daemon overhead
- Fork-exec model (more reliable)
- Native systemd integration

**Pattern**:
```bash
# Rootless container
podman run -d --name shadow-api \
  -p 8000:8000 \
  -v shadow-data:/data \
  shadow-ai:latest

# Create as systemd service
podman generate systemd --name shadow-api --files
systemctl --user enable container-shadow-api
```

**For Shadow AI**:
- Secure production deployment
- Systemd integration for auto-start
- Pod grouping for related services
- Rootless operation for security

---

## CI/CD Systems

### GitHub Actions Runner (5.9k stars)
**GitHub**: https://github.com/actions/runner

**Key Concepts**:
- Self-hosted runners for CI/CD
- Workflow automation
- Environment matrix testing
- Artifact caching
- Secrets management

**Pattern**:
```yaml
name: Shadow AI CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.13'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ --cov=shadowd
    
    - name: Build Docker image
      run: docker build -t shadow-ai:${{ github.sha }} .

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        # Deploy script
```

**For Shadow AI**:
- Automated testing on pull requests
- Continuous deployment to staging
- Security scanning integration
- Release automation

---

## Observability

### Prometheus (63.4k stars)
**GitHub**: https://github.com/prometheus/prometheus

**Key Concepts**:
- Time-series database
- Pull-based metrics collection
- PromQL query language
- Alerting with Alertmanager
- Service discovery

**Pattern**:
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'shadow-ai'
    static_configs:
      - targets: ['shadow-api:8000']
    metrics_path: /metrics

# Application metrics
from prometheus_client import Counter, Histogram, Gauge

requests_total = Counter(
    'shadow_requests_total',
    'Total requests processed',
    ['endpoint', 'method']
)

memory_operations = Histogram(
    'shadow_memory_operation_seconds',
    'Memory operation latency',
    ['operation_type']
)

active_agents = Gauge(
    'shadow_active_agents',
    'Number of active agents'
)
```

**For Shadow AI**:
- Track agent execution times
- Monitor memory operations
- Measure LLM API latency
- Alert on error rates

---

### Grafana Loki (27.9k stars)
**GitHub**: https://github.com/grafana/loki

**Key Concepts**:
- Log aggregation system
- Label-based indexing
- Grafana integration
- Cost-effective log storage
- Promtail/Alloy for collection

**Pattern**:
```yaml
# Loki configuration
auth_enabled: false

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: loki_index_

# Application logging
import logging
from loki_logger_handler import LokiHandler

handler = LokiHandler(
    url="http://loki:3100/loki/api/v1/push",
    labels={"app": "shadow-ai", "environment": "production"}
)
logger = logging.getLogger()
logger.addHandler(handler)
```

**For Shadow AI**:
- Centralized log collection
- Correlation between agents
- Debug slow operations
- Historical log analysis

---

## Reverse Proxy

### Traefik (62.5k stars)
**GitHub**: https://github.com/traefik/traefik

**Key Concepts**:
- Dynamic configuration
- Service discovery
- Let's Encrypt automation
- Load balancing
- Middleware support

**Pattern**:
```yaml
# traefik.yml
api:
  dashboard: true

entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"

providers:
  docker:
    endpoint: "unix:///var/run/docker.sock"
    exposedByDefault: false

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@example.com
      storage: acme.json
      httpChallenge:
        entryPoint: web

# Docker labels for service
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.shadow.rule=Host(`api.shadow.local`)"
  - "traefik.http.routers.shadow.tls=true"
  - "traefik.http.routers.shadow.tls.certresolver=letsencrypt"
```

**For Shadow AI**:
- Automatic HTTPS with Let's Encrypt
- Load balancing across instances
- Rate limiting middleware
- WebSocket support for real-time features

---

### NGINX (29.8k stars)
**GitHub**: https://github.com/nginx/nginx

**Key Concepts**:
- High-performance reverse proxy
- Load balancing (round-robin, least_conn, ip_hash)
- Caching and compression
- SSL/TLS termination
- Rate limiting

**Pattern**:
```nginx
upstream shadow_api {
    least_conn;
    server shadow-api-1:8000 weight=5;
    server shadow-api-2:8000 weight=5;
    server shadow-api-3:8000 backup;
}

server {
    listen 80;
    listen 443 ssl http2;
    
    server_name api.shadow.local;
    
    ssl_certificate /etc/nginx/ssl/shadow.crt;
    ssl_certificate_key /etc/nginx/ssl/shadow.key;
    
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://shadow_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /metrics {
        proxy_pass http://shadow_api/metrics;
        allow 10.0.0.0/8;
        deny all;
    }
}
```

**For Shadow AI**:
- High-performance proxy layer
- WebSocket for agent streaming
- Rate limiting for API protection
- Static file serving for frontend

---

## Key Patterns for Shadow AI

### 1. Docker Compose Deployment
```yaml
# docker-compose.yml
version: '3.8'

services:
  shadow-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://shadow:secret@db:5432/shadow
      - REDIS_URL=redis://cache:6379
      - LLM_PROVIDER=ollama_cloud
      - LLM_API_KEY=${OLLAMA_API_KEY}
    volumes:
      - shadow-data:/app/data
    depends_on:
      - db
      - cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: shadow
      POSTGRES_USER: shadow
      POSTGRES_PASSWORD: secret
    volumes:
      - db-data:/var/lib/postgresql/data
  
  cache:
    image: redis:7-alpine
    volumes:
      - cache-data:/data
  
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"

volumes:
  shadow-data:
  db-data:
  cache-data:
```

### 2. Production Checklist
- [ ] Health checks for all services
- [ ] Resource limits (memory, CPU)
- [ ] Persistent volume configuration
- [ ] Environment-based secrets
- [ ] Log aggregation (Loki)
- [ ] Metrics collection (Prometheus)
- [ ] Rate limiting (Traefik/NGINX)
- [ ] SSL/TLS termination
- [ ] Backup strategy for data

### 3. Deployment Pipeline
```yaml
# .github/workflows/deploy.yml
name: Shadow AI Deploy

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: docker build -t shadow-ai:${{ github.ref_name }} .
    - name: Push to registry
      run: docker push shadow-ai:${{ github.ref_name }}
  
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
    - name: Deploy to staging
      run: |
        ssh staging-server "cd /opt/shadow-ai && \
          docker-compose pull && \
          docker-compose up -d"
  
  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
    - name: Deploy to production
      run: |
        ssh production-server "cd /opt/shadow-ai && \
          docker-compose pull && \
          docker-compose up -d && \
          docker system prune -f"
```

---

## Summary
- **Docker Compose**: Simple multi-container deployment
- **Kubernetes**: Production-grade orchestration with scaling
- **Podman**: Secure rootless containers with systemd
- **GitHub Actions**: CI/CD automation
- **Prometheus**: Metrics and alerting
- **Loki**: Log aggregation
- **Traefik/NGINX**: Reverse proxy and load balancing
- **Key patterns**: Health checks, resource limits, persistent storage, secrets management