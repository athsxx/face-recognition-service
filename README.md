# Face Recognition Service (FRS)

Production-ready Face Recognition microservice with face detection, embedding extraction, and identity matching. Optimized for CPU inference with ONNX runtime and Faiss indexing.

## Features

- **Face Detection**: RetinaFace with quality filtering (blur, brightness, size)
- **Face Alignment**: 5-point landmark alignment for normalized face crops
- **Embedding Extraction**: ArcFace/AdaFace with ONNX optimization for CPU
- **Fast Matching**: Faiss-accelerated cosine similarity search
- **Gallery Management**: SQLite/PostgreSQL storage for identities
- **REST API**: FastAPI-based microservice with Swagger docs
- **Docker Support**: Containerized deployment
- **Benchmarking**: Performance evaluation and metrics

## Architecture

```
CCTV Frame → Detection → Alignment → Embedding → Matching → Identity
                ↓            ↓           ↓           ↓
           RetinaFace   5-point     ArcFace/     Faiss
           + Quality    Landmarks   AdaFace      Index
           Filtering                (ONNX)
```

## Project Structure

```
FRS/
├── frs/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── core/
│   │   ├── detector.py          # Face detection
│   │   ├── alignment.py         # Face alignment
│   │   ├── embedding.py         # Embedding extraction
│   │   └── matcher.py           # Face matching
│   ├── database/
│   │   └── models.py            # Database models
│   └── utils/
│       └── config.py            # Configuration management
├── scripts/
│   ├── prepare_data.py          # Data preparation
│   └── benchmark.py             # Benchmarking
├── configs/
│   └── config.yaml              # Configuration file
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Local Setup

```bash
# Clone repository
cd /Users/a91788/Desktop/FRS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,processed,gallery} models/weights logs
```

### Docker Setup

```bash
# Build Docker image
docker-compose build

# Run service
docker-compose up -d
```

## Configuration

Edit `configs/config.yaml` to customize:

- Detection thresholds and parameters
- Embedding model settings
- Matching thresholds (cosine/L2)
- Database configuration
- API settings

## Data Preparation

### 1. Prepare Dataset

Organize your dataset:
```
data/raw/
  person1/
    img1.jpg
    img2.jpg
  person2/
    img1.jpg
```

### 2. Process Data

```bash
# Run data preparation (detection, alignment, normalization)
python scripts/prepare_data.py \
    --raw_dir data/raw \
    --output_dir data/processed \
    --train_split 0.8
```

This will:
- Detect faces in all images
- Perform 5-point alignment
- Create train/val splits
- Save processed data to `data/processed/`

## Usage

### Start API Server

```bash
# Local
uvicorn frs.api.main:app --host 0.0.0.0 --port 8000

# Docker
docker-compose up -d
```

### API Documentation

Access Swagger UI at: `http://localhost:8000/docs`

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

#### 2. Detect Faces

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

Response:
```json
{
  "num_faces": 1,
  "faces": [
    {
      "bbox": [100, 150, 300, 400],
      "confidence": 0.98,
      "landmarks": [[120, 180], [180, 180], ...],
      "quality_score": 0.85
    }
  ],
  "processing_time_ms": 45.2
}
```

#### 3. Recognize Faces

```bash
curl -X POST "http://localhost:8000/recognize" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "return_top_k=5" \
  -F "min_confidence=0.6"
```

Response:
```json
{
  "num_faces": 1,
  "faces": [
    {
      "bbox": [100, 150, 300, 400],
      "detection_confidence": 0.98,
      "identity_id": "id_abc123",
      "name": "John Doe",
      "match_confidence": 0.87,
      "top_matches": [
        {
          "identity_id": "id_abc123",
          "name": "John Doe",
          "confidence": 0.87
        }
      ],
      "quality_score": 0.85
    }
  ],
  "processing_time_ms": 120.5
}
```

#### 4. Add Identity

```bash
curl -X POST "http://localhost:8000/add_identity" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@person.jpg" \
  -F "name=John Doe" \
  -F "identity_id=johndoe_001" \
  -F 'metadata={"department":"engineering"}'
```

#### 5. List Identities

```bash
curl http://localhost:8000/list_identities
```

#### 6. Delete Identity

```bash
curl -X DELETE "http://localhost:8000/identity/johndoe_001"
```

## Benchmarking

### Run Benchmarks

```bash
# Benchmark all components
python scripts/benchmark.py --component all --output results.json

# Benchmark specific component
python scripts/benchmark.py --component detection
python scripts/benchmark.py --component embedding
python scripts/benchmark.py --component matching
python scripts/benchmark.py --component e2e
```

### Expected Performance (CPU: Intel i7)

| Component | Latency (ms) | Throughput |
|-----------|-------------|------------|
| Detection | 40-60 | ~20 FPS |
| Alignment | 2-5 | ~200 FPS |
| Embedding | 15-25 | ~50 FPS |
| Matching (1K gallery) | 1-3 | ~500 QPS |
| **End-to-End** | **60-100** | **~12 FPS** |

## Optimization

### ONNX Conversion

Convert PyTorch models to ONNX for CPU optimization:

```python
from frs.core.embedding import FaceEmbedding

embedder = FaceEmbedding()
embedder.convert_to_onnx(
    pytorch_model_path="models/arcface.pth",
    onnx_output_path="models/weights/arcface_r100.onnx",
    input_size=(112, 112),
    opset_version=14
)
```

### CPU Optimization Flags

Set environment variables for better CPU performance:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

### Faiss Optimization

For larger galleries (>10K identities), consider:
- IVF (Inverted File) indexing: `faiss.IndexIVFFlat`
- PQ (Product Quantization): `faiss.IndexIVFPQ`
- GPU acceleration: `faiss-gpu`

## Evaluation & Metrics

### Detection Metrics

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Recognition Metrics

- **Top-1 Accuracy**: % of queries where top match is correct
- **Top-5 Accuracy**: % of queries where correct identity is in top 5
- **TAR @ FAR**: True Accept Rate at specific False Accept Rate

### Failure Modes & Mitigations

| Failure Mode | Cause | Mitigation |
|-------------|-------|------------|
| **Low-light** | Poor illumination | Brightness normalization, denoising |
| **Occlusion** | Masks, glasses, hair | Landmark-aware detection, ensemble |
| **Pose variation** | Side profiles | Multi-view templates, 3D alignment |
| **Image blur** | Motion, defocus | Quality filtering, deblurring |
| **Age variation** | Temporal changes | Regular re-enrollment, age-invariant models |

## Robustness Strategies

1. **Quality Filtering**:
   - Reject blurry faces (Laplacian variance < 100)
   - Check brightness range [20, 235]
   - Enforce minimum face size (40x40 pixels)

2. **Multi-Template Enrollment**:
   - Store 3-5 embeddings per identity
   - Average embeddings for robustness

3. **Threshold Tuning**:
   - Adjust `matching.threshold` based on FAR/TAR requirements
   - Lower threshold = higher recall, lower precision

4. **Ensemble Models**:
   - Combine multiple face recognition models
   - Vote-based or confidence-weighted fusion

## Model Weights

Download pretrained models:

```bash
# RetinaFace (detection)
wget https://github.com/deepinsight/insightface/releases/download/v0.7/retinaface_r50_v1.onnx \
  -O models/weights/retinaface_resnet50.onnx

# ArcFace (embedding)
wget https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_r100_v1.onnx \
  -O models/weights/arcface_r100.onnx
```

Or use InsightFace models (fallback in code):
```python
# Models will be auto-downloaded on first run
```

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=frs --cov-report=html
```

## Production Deployment

### Scaling

- Use `--workers 4` with uvicorn for multiple processes
- Deploy behind load balancer (nginx, HAProxy)
- Use Redis for shared session state
- Consider async processing for batch inference

### Monitoring

- Enable Prometheus metrics at `:9090`
- Monitor latency, throughput, error rates
- Set up alerts for degraded performance

### Security

- Add authentication (OAuth2, JWT)
- Rate limiting (per IP, per user)
- Input validation and sanitization
- HTTPS/TLS encryption

## Troubleshooting

### Issue: ONNX model not loading

```bash
# Check ONNX runtime version
pip install onnxruntime==1.16.3

# Verify model compatibility
python -c "import onnx; model = onnx.load('models/weights/model.onnx'); onnx.checker.check_model(model)"
```

### Issue: Low recognition accuracy

- Verify quality of enrollment images
- Check alignment correctness
- Tune matching threshold
- Ensure sufficient gallery diversity

### Issue: Slow inference

- Enable ONNX optimizations
- Reduce input image size
- Batch processing for multiple faces
- Profile with `cProfile` to identify bottlenecks

## References

- **RetinaFace**: [Paper](https://arxiv.org/abs/1905.00641)
- **ArcFace**: [Paper](https://arxiv.org/abs/1801.07698)
- **AdaFace**: [Paper](https://arxiv.org/abs/2204.00964)
- **Faiss**: [Documentation](https://github.com/facebookresearch/faiss)
- **InsightFace**: [GitHub](https://github.com/deepinsight/insightface)

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact the development team.
