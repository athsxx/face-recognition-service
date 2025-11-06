# 🎯 Face Recognition Service

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready Face Recognition microservice with face detection, embedding extraction, and identity matching. Optimized for CPU inference with ONNX runtime and Faiss indexing.

## 🚀 Quick Start

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

## 📊 Performance

| Component | Latency | Throughput | Status |
|-----------|---------|------------|---------|
| Face Detection | 257ms | 3.9 FPS | ✅ Real-time |
| Face Recognition | 187ms | 5.4 FPS | ✅ Excellent |
| Health Check | 2.2ms | 452 RPS | ✅ Fast |

## 🎯 Features

- **🔍 Face Detection** - RetinaFace with quality filtering
- **📐 Face Alignment** - 5-point landmark normalization  
- **🧠 Face Recognition** - ArcFace embeddings (512-dim)
- **⚡ Fast Search** - Faiss-accelerated similarity matching
- **🌐 REST API** - FastAPI with Swagger documentation
- **🐳 Docker Ready** - Containerized deployment
- **📊 Benchmarking** - Performance analysis tools
- **🎨 Visualization** - Detection result overlays

## 🏗️ Architecture

```
CCTV Frame → Detection → Alignment → Embedding → Matching → Identity
                ↓            ↓           ↓           ↓
           RetinaFace   5-point     ArcFace      Faiss
           + Quality    Landmarks   (ONNX)       Index
           Filtering
```

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/detect` | Detect faces in image |
| `POST` | `/recognize` | Recognize faces |
| `POST` | `/add_identity` | Add new identity |
| `GET` | `/list_identities` | List all identities |
| `DELETE` | `/identity/{id}` | Remove identity |

## 🎬 Demo

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

## 🛠️ Development

### Visualization Tools
```bash
# Visualize detection results
python visualize_detection.py image.jpg

# Run benchmarks
python run_benchmark.py

# Interactive demo
jupyter notebook demo_notebook.ipynb
```

### Configuration
Edit `configs/config.yaml` to customize:
- Detection thresholds
- Face quality filters  
- Matching confidence
- Database settings

## 📁 Project Structure

```
face-recognition-service/
├── frs/                    # Main package
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Detection, alignment, embedding, matching
│   ├── database/          # Database models
│   └── utils/             # Configuration utilities
├── scripts/               # Data preparation & benchmarking
├── configs/               # Configuration files
├── tests/                 # Test suite
├── docs/                  # Documentation
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Service orchestration
└── requirements.txt       # Python dependencies
```

## 🔧 Technology Stack

- **Language**: Python 3.9+
- **Framework**: FastAPI
- **Models**: RetinaFace, ArcFace (InsightFace)
- **Optimization**: ONNX Runtime (CPU)
- **Search**: Faiss (CPU)
- **Database**: SQLAlchemy (SQLite/PostgreSQL)
- **Container**: Docker
- **Testing**: Pytest

## 📊 Benchmarks

Run comprehensive benchmarks:
```bash
python run_benchmark.py
```

Expected performance on Intel i7:
- **Detection**: 40-60ms (~20 FPS)
- **Embedding**: 15-25ms (~50 FPS)  
- **End-to-End**: 60-100ms (~12 FPS)

## 🚀 Production Deployment

### Scaling
- Use `--workers 4` with uvicorn
- Deploy behind load balancer
- Use Redis for session state
- Consider GPU acceleration for higher throughput

### Monitoring
- Prometheus metrics at `:9090`
- Health checks and alerts
- Performance monitoring

## 📚 Documentation

- **📖 [User Guide](README.md)** - Complete setup and usage
- **🔧 [Technical Report](TECHNICAL_REPORT.md)** - Architecture details
- **🚀 [Quick Start](QUICKSTART.md)** - 5-minute setup
- **🐳 [Docker Guide](BUILD_INSTRUCTIONS.md)** - Container deployment
- **📊 [Benchmarks](benchmark_results.json)** - Performance analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face recognition models
- [Faiss](https://github.com/facebookresearch/faiss) for similarity search
- [FastAPI](https://fastapi.tiangolo.com) for the web framework

## 📞 Support

- 📧 Issues: [GitHub Issues](https://github.com/athsxx/face-recognition-service/issues)
- 📖 Documentation: [Project Wiki](https://github.com/athsxx/face-recognition-service/wiki)
- 💬 Discussions: [GitHub Discussions](https://github.com/athsxx/face-recognition-service/discussions)

---

**⭐ Star this repository if you find it useful!**
