#!/bin/bash

# Face Recognition Service Setup Script

set -e

echo "=== Face Recognition Service Setup ==="
echo ""

# Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "[3/5] Creating directories..."
mkdir -p data/{raw,processed,gallery} models/weights logs tests

# Download sample models (optional)
echo "[4/5] Setting up models..."
echo "Note: Pre-trained models will be downloaded automatically on first run."
echo "Or manually download:"
echo "  - RetinaFace: https://github.com/deepinsight/insightface/releases"
echo "  - ArcFace: https://github.com/deepinsight/insightface/releases"

# Initialize database
echo "[5/5] Initializing database..."
python -c "
from frs.database.models import Database
db = Database('sqlite:///data/frs.db')
db.create_tables()
print('Database initialized successfully')
"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Start API server: uvicorn frs.api.main:app --reload"
echo "  3. Access docs: http://localhost:8000/docs"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo ""
