# Face Recognition Service

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready Face Recognition microservice with face detection, embedding extraction, and identity matching. Optimized for CPU inference with ONNX runtime and Faiss indexing.

## Quick Start

### Docker Deployment (Recommended)
```bash
git clone https://github.com/athsxx/face-recognition-service.git
cd face-recognition-service
docker-compose up -d
```

### Local Setup
```bash
git clone https://github.com/athsxx/face-recognition-service.git
cd face-recognition-service
./scripts/setup.sh
uvicorn frs.api.main:app --host 0.0.0.0 --port 8000
```

Access the API documentation at: **http://localhost:8000/docs**

## Performance

| Component | Latency | Throughput | Status |
|-----------|---------|------------|---------|
| Face Detection | 257ms | 3.9 FPS | Real-time |
| Face Recognition | 187ms | 5.4 FPS | Excellent |
| Health Check | 2.2ms | 452 RPS | Fast |

## Features

- **Face Detection** - RetinaFace with quality filtering
- **Face Alignment** - 5-point landmark normalization  
- **Face Recognition** - ArcFace embeddings (512-dim)
- **Fast Search** - Faiss-accelerated similarity matching
- **REST API** - FastAPI with Swagger documentation
- **Docker Ready** - Containerized deployment
- **Benchmarking** - Performance analysis tools
- **Visualization** - Detection result overlays

## Architecture

```
CCTV Frame → Detection → Alignment → Embedding → Matching → Identity
                ↓            ↓           ↓           ↓
           RetinaFace   5-point     ArcFace      Faiss
           + Quality    Landmarks   (ONNX)       Index
           Filtering
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/detect` | Detect faces in image |
| `POST` | `/recognize` | Recognize faces |
| `POST` | `/add_identity` | Add new identity |
| `GET` | `/list_identities` | List all identities |
| `DELETE` | `/identity/{id}` | Remove identity |

## Demo Examples

### Face Detection
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg"
```

### Add Identity
```bash
curl -X POST "http://localhost:8000/add_identity" \
  -F "file=@person.jpg" \
  -F "name=John Doe" \
  -F "identity_id=john_001"
```

### Recognition
```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "file=@test.jpg" \
  -F "return_top_k=5"
```

## Development Tools

### Visualization
```bash
# Visualize detection results
python visualize_detection.py image.jpg

# Run performance benchmarks
python run_benchmark.py

# Interactive demo
jupyter notebook demo_notebook.ipynb
```

### Configuration
Edit `configs/config.yaml` to customize:
- Detection thresholds and parameters
- Face quality filters (blur, brightness, size)
- Matching confidence levels
- Database settings

## Project Structure

```
face-recognition-service/
├── frs/                    # Main package
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Detection, alignment, embedding, matching
│   ├── database/          # Database models
│   └── utils/             # Configuration utilities
├── scripts/               # Setup and benchmarking
├── configs/               # Configuration files
├── tests/                 # Test suite
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Service orchestration
└── requirements.txt       # Python dependencies
```

## Technology Stack

- **Language**: Python 3.9+
- **Framework**: FastAPI
- **Models**: RetinaFace, ArcFace (InsightFace)
- **Optimization**: ONNX Runtime (CPU)
- **Search**: Faiss (CPU)
- **Database**: SQLAlchemy (SQLite/PostgreSQL)
- **Container**: Docker

## Benchmarking

Run comprehensive performance analysis:
```bash
python run_benchmark.py
```

Expected performance on Intel i7:
- **Detection**: 40-60ms (~20 FPS)
- **Embedding**: 15-25ms (~50 FPS)  
- **End-to-End**: 60-100ms (~12 FPS)

## Production Deployment

### Scaling Options
- Use `--workers 4` with uvicorn for multiple processes
- Deploy behind load balancer (nginx, HAProxy)
- Use Redis for shared session state
- Consider GPU acceleration for higher throughput

### Monitoring
- Health checks at `/health` endpoint
- Prometheus metrics support
- Configurable logging levels
- Performance monitoring built-in

## Configuration

Key configuration options in `configs/config.yaml`:

```yaml
detection:
  confidence_threshold: 0.3    # Lower = more sensitive
  min_face_size: 20           # Minimum face size in pixels
  max_face_size: 2000         # Maximum face size in pixels
  blur_threshold: 50          # Blur detection sensitivity

matching:
  threshold: 0.55             # Recognition confidence threshold
  top_k: 5                    # Number of top matches to return
  min_confidence: 0.6         # Minimum match confidence
```

## Testing

```bash
# Run test suite
pytest tests/ -v

# Test with sample image
python visualize_detection.py /path/to/image.jpg

# Benchmark performance
python run_benchmark.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face recognition models
- [Faiss](https://github.com/facebookresearch/faiss) for similarity search
- [FastAPI](https://fastapi.tiangolo.com) for the web framework

## Support

- 📧 Issues: [GitHub Issues](https://github.com/athsxx/face-recognition-service/issues)
- 📖 Documentation: Available in the repository
- 💬 Discussions: [GitHub Discussions](https://github.com/athsxx/face-recognition-service/discussions)

---

**Star this repository if you find it useful!**
