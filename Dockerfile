FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY frs/ ./frs/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p data/gallery data/raw data/processed models/weights logs

# Set environment variables for CPU optimization
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV VECLIB_MAXIMUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "frs.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
