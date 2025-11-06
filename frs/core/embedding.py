"""Face embedding extraction module."""

import time
from typing import List, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch
from loguru import logger

from frs.utils.config import config


class FaceEmbedding:
    """Face embedding extractor using ArcFace/AdaFace."""

    def __init__(self, model_path: str = None, use_onnx: bool = True):
        """Initialize face embedding extractor.

        Args:
            model_path: Path to model file (ONNX or PyTorch)
            use_onnx: Whether to use ONNX runtime (recommended for CPU)
        """
        self.model_path = model_path or config.embedding.model_path
        self.use_onnx = use_onnx and config.embedding.use_onnx
        self.embedding_size = config.embedding.embedding_size
        self.batch_size = config.embedding.batch_size

        self.session = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load embedding model."""
        try:
            if self.use_onnx:
                self._load_onnx_model()
            else:
                self._load_pytorch_model()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to InsightFace
            self._load_insightface_model()

    def _load_onnx_model(self):
        """Load ONNX model with CPU optimization."""
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = config.embedding.onnx_threads
        sess_options.intra_op_num_threads = config.embedding.onnx_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"Loaded ONNX embedding model from {self.model_path}")

    def _load_pytorch_model(self):
        """Load PyTorch model."""
        # Load pretrained model (ArcFace/AdaFace)
        # This is a placeholder - actual implementation depends on model architecture
        from torchvision import models
        
        self.model = models.resnet50(pretrained=False)
        # Modify architecture for face recognition
        # self.model.fc = torch.nn.Linear(2048, self.embedding_size)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        logger.info(f"Loaded PyTorch embedding model from {self.model_path}")

    def _load_insightface_model(self):
        """Fallback to InsightFace model."""
        try:
            from insightface.model_zoo import get_model
            
            # Load only the recognition model
            self.rec_model = get_model('arcface_r100_v1')
            self.rec_model.prepare(ctx_id=-1)
            
            logger.info("Using InsightFace ArcFace model as fallback")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            # Create a dummy model that returns random embeddings for testing
            self.rec_model = None
            logger.warning("Using dummy embedding model for testing")

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for embedding extraction.

        Args:
            face_img: Aligned face image (assumed to be already normalized)

        Returns:
            Preprocessed image tensor
        """
        # If already normalized during alignment, just transpose
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            # Transpose to CHW format
            img = np.transpose(face_img, (2, 0, 1))
        else:
            img = face_img

        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        
        return img

    def extract(self, face_img: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Extract face embedding(s).

        Args:
            face_img: Single aligned face image or list of face images

        Returns:
            Embedding vector(s) of shape (embedding_size,) or (N, embedding_size)
        """
        start_time = time.time()

        # Handle single image or batch
        is_single = not isinstance(face_img, list)
        if is_single:
            face_img = [face_img]

        embeddings = []

        # Process in batches
        for i in range(0, len(face_img), self.batch_size):
            batch = face_img[i:i + self.batch_size]
            batch_embeddings = self._extract_batch(batch)
            embeddings.append(batch_embeddings)

        # Concatenate all embeddings
        embeddings = np.vstack(embeddings)

        # Return single embedding if input was single
        if is_single:
            embeddings = embeddings[0]

        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Extracted {len(face_img)} embeddings in {elapsed:.2f}ms")

        return embeddings

    def _extract_batch(self, batch: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a batch of faces.

        Args:
            batch: List of aligned face images

        Returns:
            Batch of embeddings (N, embedding_size)
        """
        # Preprocess batch
        batch_tensor = np.vstack([self.preprocess(img) for img in batch])

        # ONNX inference
        if self.session:
            embeddings = self.session.run(
                [self.output_name],
                {self.input_name: batch_tensor}
            )[0]
        
        # PyTorch inference
        elif self.model:
            with torch.no_grad():
                batch_tensor = torch.from_numpy(batch_tensor)
                embeddings = self.model(batch_tensor)
                embeddings = embeddings.cpu().numpy()
        
        # InsightFace fallback
        elif hasattr(self, 'rec_model') and self.rec_model:
            embeddings = []
            for img in batch:
                # InsightFace expects normalized face image
                if img.shape != (112, 112, 3):
                    import cv2
                    img = cv2.resize(img, (112, 112))
                
                # Convert to the format expected by InsightFace
                if img.dtype != np.float32:
                    img = img.astype(np.float32)
                
                # Get embedding
                embedding = self.rec_model.get_feat(img)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
        
        # Dummy model fallback for testing
        else:
            logger.warning("Using dummy embeddings for testing")
            embeddings = np.random.randn(len(batch), self.embedding_size).astype(np.float32)

        # L2 normalize embeddings
        embeddings = self._normalize_embeddings(embeddings)

        return embeddings

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings.

        Args:
            embeddings: Raw embeddings (N, embedding_size)

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        return embeddings / norms

    def convert_to_onnx(
        self,
        pytorch_model_path: str,
        onnx_output_path: str,
        input_size: tuple = (112, 112),
        opset_version: int = 14
    ):
        """Convert PyTorch model to ONNX format.

        Args:
            pytorch_model_path: Path to PyTorch model
            onnx_output_path: Output path for ONNX model
            input_size: Input image size (H, W)
            opset_version: ONNX opset version
        """
        import torch.onnx

        # Load PyTorch model
        model = torch.load(pytorch_model_path, map_location='cpu')
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, *input_size)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        logger.info(f"Converted model to ONNX: {onnx_output_path}")

    def benchmark(self, num_iterations: int = 100, batch_size: int = 1) -> dict:
        """Benchmark embedding extraction performance.

        Args:
            num_iterations: Number of iterations for benchmarking
            batch_size: Batch size for benchmarking

        Returns:
            Dictionary with performance metrics
        """
        import time

        # Create dummy input
        dummy_input = [np.random.randn(112, 112, 3).astype(np.float32) 
                       for _ in range(batch_size)]

        # Warmup
        for _ in range(10):
            _ = self.extract(dummy_input)

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = self.extract(dummy_input)
        elapsed = time.time() - start_time

        avg_latency_ms = (elapsed / num_iterations) * 1000
        throughput_fps = (num_iterations * batch_size) / elapsed

        metrics = {
            'avg_latency_ms': avg_latency_ms,
            'throughput_fps': throughput_fps,
            'batch_size': batch_size,
            'num_iterations': num_iterations
        }

        logger.info(f"Embedding benchmark: {avg_latency_ms:.2f}ms, {throughput_fps:.2f} FPS")
        return metrics
