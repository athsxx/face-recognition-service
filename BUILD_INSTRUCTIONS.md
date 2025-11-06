# Build and Deployment Instructions

## Prerequisites

- Docker (20.10+)
- Docker Compose (1.29+)
- OR Python 3.9+ with pip

## Option 1: Docker Deployment (Recommended)

### Build Docker Image

```bash
cd /Users/a91788/Desktop/FRS

# Build the image
docker build -t frs:latest .

# Verify build
docker images | grep frs
```

### Run with Docker Compose

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f frs-api

# Stop services
docker-compose down
```

### Run Single Container

```bash
# Run container
docker run -d \
  --name frs-service \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e OMP_NUM_THREADS=4 \
  frs:latest

# Check logs
docker logs -f frs-service

# Stop container
docker stop frs-service
docker rm frs-service
```

## Option 2: Local Development Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create Directories

```bash
mkdir -p data/{raw,processed,gallery} models/weights logs
```

### 4. Start Service

```bash
# Development mode (auto-reload)
uvicorn frs.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn frs.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Verification

### Health Check

```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "detector": "loaded",
    "embedder": "loaded",
    "matcher": "0 identities",
    "database": "connected"
  }
}
```

### API Documentation

Open browser: http://localhost:8000/docs

### Test Detection

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg"
```

## Configuration

Edit `configs/config.yaml` to customize:
- Detection thresholds
- Embedding model paths
- Matching parameters
- Database settings

## Troubleshooting

### Port Already in Use

```bash
# Change port in config or use different port
uvicorn frs.api.main:app --host 0.0.0.0 --port 8080
```

### Models Not Loading

Models will auto-download on first run using InsightFace. If this fails:

```bash
# Manually download models
mkdir -p models/weights
cd models/weights

# Download RetinaFace
wget https://github.com/deepinsight/insightface/releases/download/v0.7/retinaface_r50_v1.onnx

# Download ArcFace
wget https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100_v1.onnx
```

### Docker Build Fails

Ensure sufficient disk space and memory:

```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t frs:latest .
```

## Performance Tuning

### CPU Optimization

Set environment variables:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

### Batch Processing

For better throughput, process multiple images in batch:

```python
from frs.core.detector import FaceDetector
detector = FaceDetector()
# Process batch of images
```

### Faiss Optimization

For large galleries (>10K identities):

```yaml
# In configs/config.yaml
matching:
  use_faiss: true
  faiss_index_type: "IndexIVFFlat"  # Use IVF for speed
```

## Monitoring

### Prometheus Metrics

Enable in `configs/config.yaml`:

```yaml
monitoring:
  enable_metrics: true
  metrics_port: 9090
```

Access metrics: http://localhost:9090/metrics

### Logs

```bash
# Docker logs
docker-compose logs -f

# Local logs
tail -f logs/frs.log
```

## Scaling

### Horizontal Scaling

```bash
# Scale to 4 replicas
docker-compose up -d --scale frs-api=4

# Use nginx/HAProxy for load balancing
```

### Database Migration

For production, use PostgreSQL:

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: frs_user
      POSTGRES_PASSWORD: frs_password
      POSTGRES_DB: frs_db
```

Update `configs/config.yaml`:

```yaml
database:
  type: "postgresql"
  postgresql_url: "postgresql://frs_user:frs_password@postgres:5432/frs_db"
```

## Production Checklist

- [ ] Set strong database passwords
- [ ] Configure HTTPS/TLS
- [ ] Enable authentication (OAuth2/JWT)
- [ ] Set up monitoring and alerting
- [ ] Configure log rotation
- [ ] Set resource limits (CPU/Memory)
- [ ] Enable auto-restart policies
- [ ] Set up backup for database
- [ ] Configure firewall rules
- [ ] Review security headers

## Support

For issues, check:
1. README.md for general usage
2. TECHNICAL_REPORT.md for implementation details
3. QUICKSTART.md for quick setup
4. API documentation at /docs endpoint
