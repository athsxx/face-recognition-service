"""FastAPI microservice for Face Recognition Service."""

import io
import time
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from frs.core.alignment import FaceAligner
from frs.core.detector import FaceDetector
from frs.core.embedding import FaceEmbedding
from frs.core.matcher import FaceMatcher
from frs.database.models import Database
from frs.utils.config import config

# Initialize FastAPI app
app = FastAPI(
    title=config.service.name,
    version=config.service.version,
    description="Production-ready Face Recognition Service with detection, embedding, and matching",
)

# CORS middleware
if config.get("api.enable_cors"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api.cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize components
db = Database(f"sqlite:///{config.database.sqlite_path}")
db.create_tables()

detector = FaceDetector()
aligner = FaceAligner()
embedder = FaceEmbedding()
matcher = FaceMatcher(db)


# Pydantic models
class DetectionResponse(BaseModel):
    num_faces: int
    faces: List[dict]
    processing_time_ms: float


class RecognitionResponse(BaseModel):
    num_faces: int
    faces: List[dict]
    processing_time_ms: float


class IdentityResponse(BaseModel):
    success: bool
    message: str
    identity_id: Optional[str] = None


class IdentityListResponse(BaseModel):
    num_identities: int
    identities: List[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict


# Helper functions
def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Load image from uploaded file.

    Args:
        file: Uploaded file

    Returns:
        BGR image as numpy array
    """
    try:
        # Read file content
        contents = file.file.read()
        
        # Convert to PIL Image
        pil_img = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Convert to numpy array (RGB)
        img_array = np.array(pil_img)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": config.service.version,
        "components": {
            "detector": "loaded",
            "embedder": "loaded",
            "matcher": "loaded",
            "database": "connected"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "version": config.service.version,
        "components": {
            "detector": "loaded",
            "embedder": "loaded",
            "matcher": f"{matcher.index.ntotal if matcher.index else 0} identities" if matcher.use_faiss else "database mode",
            "database": "connected"
        }
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in an image.

    Args:
        file: Image file (JPEG, PNG, BMP)

    Returns:
        Detection results with bounding boxes and landmarks
    """
    start_time = time.time()
    
    try:
        # Load image
        image = load_image_from_upload(file)
        
        # Detect faces
        faces = detector.detect(image)
        
        # Format response
        response_faces = []
        for face in faces:
            response_faces.append({
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'landmarks': face.get('landmarks', []),
                'quality_score': face.get('quality_score', 0.0)
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'num_faces': len(faces),
            'faces': response_faces,
            'processing_time_ms': processing_time
        }
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_faces(
    file: UploadFile = File(...),
    return_top_k: int = Form(5),
    min_confidence: float = Form(0.6)
):
    """Recognize faces in an image.

    Args:
        file: Image file (JPEG, PNG, BMP)
        return_top_k: Number of top matches to return
        min_confidence: Minimum confidence threshold

    Returns:
        Recognition results with identity matches
    """
    start_time = time.time()
    
    try:
        # Load image
        image = load_image_from_upload(file)
        
        # Detect faces
        faces = detector.detect(image)
        
        if len(faces) == 0:
            return {
                'num_faces': 0,
                'faces': [],
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        
        # Align faces
        aligned_faces = aligner.align_batch(image, faces)
        
        # Extract embeddings
        embeddings = embedder.extract(aligned_faces)
        
        # Match against gallery
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        matches_list = matcher.batch_match(embeddings)
        
        # Format response
        response_faces = []
        for face, matches in zip(faces, matches_list):
            # Filter by confidence
            filtered_matches = [m for m in matches if m['confidence'] >= min_confidence]
            filtered_matches = filtered_matches[:return_top_k]
            
            # Determine best match
            if filtered_matches:
                best_match = filtered_matches[0]
                identity_id = best_match['identity_id']
                name = best_match['name']
                match_confidence = best_match['confidence']
            else:
                identity_id = None
                name = "Unknown"
                match_confidence = 0.0
            
            response_faces.append({
                'bbox': face['bbox'],
                'detection_confidence': face['confidence'],
                'identity_id': identity_id,
                'name': name,
                'match_confidence': match_confidence,
                'top_matches': filtered_matches,
                'quality_score': face.get('quality_score', 0.0)
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'num_faces': len(faces),
            'faces': response_faces,
            'processing_time_ms': processing_time
        }
        
    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


@app.post("/add_identity", response_model=IdentityResponse)
async def add_identity(
    file: UploadFile = File(...),
    name: str = Form(...),
    identity_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Add a new identity to the gallery.

    Args:
        file: Face image file
        name: Person name
        identity_id: Optional unique identifier (auto-generated if not provided)
        metadata: Optional JSON metadata

    Returns:
        Success status and identity information
    """
    try:
        # Generate identity_id if not provided
        if not identity_id:
            identity_id = f"id_{uuid.uuid4().hex[:12]}"
        
        # Load image
        image = load_image_from_upload(file)
        
        # Detect faces
        faces = detector.detect(image)
        
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        if len(faces) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Multiple faces detected ({len(faces)}). Please provide an image with a single face."
            )
        
        # Align and extract embedding
        face = faces[0]
        aligned_face = aligner.align(image, face['landmarks'])
        embedding = embedder.extract(aligned_face)
        
        # Save image (optional)
        image_dir = Path("data/gallery")
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{identity_id}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Add to database
        metadata_dict = None
        if metadata:
            import json
            metadata_dict = json.loads(metadata)
        
        success = matcher.add_identity(
            identity_id=identity_id,
            name=name,
            embedding=embedding,
            image_path=str(image_path),
            metadata=metadata_dict
        )
        
        if success:
            return {
                'success': True,
                'message': f"Successfully added identity: {name}",
                'identity_id': identity_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add identity")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add identity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add identity: {str(e)}")


@app.delete("/identity/{identity_id}", response_model=IdentityResponse)
async def delete_identity(identity_id: str):
    """Delete an identity from the gallery.

    Args:
        identity_id: Identity identifier to delete

    Returns:
        Success status
    """
    try:
        success = matcher.remove_identity(identity_id)
        
        if success:
            # Also remove image file if exists
            image_path = Path(f"data/gallery/{identity_id}.jpg")
            if image_path.exists():
                image_path.unlink()
            
            return {
                'success': True,
                'message': f"Successfully deleted identity: {identity_id}",
                'identity_id': identity_id
            }
        else:
            raise HTTPException(status_code=404, detail=f"Identity not found: {identity_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete identity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete identity: {str(e)}")


@app.get("/list_identities", response_model=IdentityListResponse)
async def list_identities():
    """List all identities in the gallery.

    Returns:
        List of all registered identities
    """
    try:
        identities = matcher.get_all_identities()
        
        return {
            'num_identities': len(identities),
            'identities': identities
        }
        
    except Exception as e:
        logger.error(f"Failed to list identities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list identities: {str(e)}")


@app.get("/identity/{identity_id}")
async def get_identity(identity_id: str):
    """Get details for a specific identity.

    Args:
        identity_id: Identity identifier

    Returns:
        Identity details
    """
    try:
        identities = matcher.get_all_identities()
        identity = next((i for i in identities if i['identity_id'] == identity_id), None)
        
        if identity:
            return identity
        else:
            raise HTTPException(status_code=404, detail=f"Identity not found: {identity_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get identity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get identity: {str(e)}")


# Run with: uvicorn frs.api.main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.service.host,
        port=config.service.port,
        log_level=config.service.log_level
    )
