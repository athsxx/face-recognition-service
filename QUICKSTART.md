# Face Recognition Service - Quick Start Guide

Get the FRS up and running in 5 minutes!

## Prerequisites

- Python 3.9+
- pip
- Docker (optional, for containerized deployment)

## Option 1: Local Setup (Recommended for Development)

### Step 1: Install Dependencies

```bash
cd /Users/a91788/Desktop/FRS

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Create directories
mkdir -p data/{raw,processed,gallery} models/weights logs
```

### Step 2: Start the Service

```bash
# Activate environment
source venv/bin/activate

# Start API server
uvicorn frs.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Test the API

Open your browser to http://localhost:8000/docs

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Detect faces (replace with your image)
curl -X POST "http://localhost:8000/detect" \
  -F "file=@/path/to/image.jpg"
```

## Option 2: Docker Deployment (Recommended for Production)

### Step 1: Build and Run

```bash
cd /Users/a91788/Desktop/FRS

# Build and start
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Step 2: Test

```bash
# Health check
curl http://localhost:8000/health
```

## Add Your First Identity

```bash
# Upload a face image
curl -X POST "http://localhost:8000/add_identity" \
  -F "file=@person.jpg" \
  -F "name=John Doe" \
  -F "identity_id=john_001"
```

## Recognize Faces

```bash
# Recognize faces in image
curl -X POST "http://localhost:8000/recognize" \
  -F "file=@test.jpg" \
  -F "return_top_k=5" \
  -F "min_confidence=0.6"
```

## What's Next?

1. **Prepare Your Dataset**: See `scripts/prepare_data.py`
   ```bash
   python scripts/prepare_data.py --raw_dir data/raw --output_dir data/processed
   ```

2. **Benchmark Performance**: See `scripts/benchmark.py`
   ```bash
   python scripts/benchmark.py --component all
   ```

3. **Read Full Documentation**: See `README.md`

4. **Review Technical Details**: See `TECHNICAL_REPORT.md`

## Common Issues

### Issue: Models not loading

**Solution**: The system uses InsightFace as fallback. Models will be downloaded automatically on first run.

### Issue: Port 8000 already in use

**Solution**: Change port in `configs/config.yaml` or use:
```bash
uvicorn frs.api.main:app --host 0.0.0.0 --port 8080
```

### Issue: Import errors

**Solution**: Ensure you're in the virtual environment:
```bash
source venv/bin/activate
```

## Project Structure

```
FRS/
├── frs/                    # Main package
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Detection, alignment, embedding, matching
│   ├── database/          # Database models
│   └── utils/             # Configuration, utilities
├── scripts/               # Data prep, benchmarking
├── configs/               # Configuration files
├── data/                  # Data storage
├── models/                # Model weights
└── tests/                 # Tests
```

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/detect` | POST | Detect faces in image |
| `/recognize` | POST | Recognize faces |
| `/add_identity` | POST | Add new identity to gallery |
| `/list_identities` | GET | List all identities |
| `/identity/{id}` | GET | Get specific identity |
| `/identity/{id}` | DELETE | Remove identity |

## Performance Expectations

On Intel i7 CPU:
- **Detection**: 40-60ms
- **Embedding**: 15-25ms  
- **Matching**: 1-3ms (1K gallery)
- **End-to-End**: 60-100ms (~12 FPS)

## Need Help?

- Check `README.md` for detailed documentation
- Review `TECHNICAL_REPORT.md` for implementation details
- See example tests in `tests/`
- Open an issue on GitHub

## Configuration

Edit `configs/config.yaml` to customize:
- Detection thresholds
- Face quality filters
- Matching confidence
- Database settings
- API settings

## Development Workflow

```bash
# 1. Make changes to code
vim frs/core/detector.py

# 2. Test changes
pytest tests/

# 3. Run service in reload mode
uvicorn frs.api.main:app --reload

# 4. Benchmark changes
python scripts/benchmark.py
```

Enjoy using the Face Recognition Service! 🎉
