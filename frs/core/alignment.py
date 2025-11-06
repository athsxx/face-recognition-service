"""Face alignment module using 5-point landmarks."""

import cv2
import numpy as np
from skimage import transform as trans

from frs.utils.config import config


# Standard 5-point face landmarks (eyes, nose, mouth corners)
# Coordinates normalized to 112x112 face template
ARCFACE_SRC = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)


class FaceAligner:
    """Face alignment using 5-point landmarks."""

    def __init__(self, output_size: tuple = None):
        """Initialize face aligner.

        Args:
            output_size: Output face size (width, height)
        """
        self.output_size = tuple(output_size) if output_size else tuple(config.alignment.output_size)
        self.normalize = config.alignment.normalize
        self.mean = np.array(config.alignment.mean, dtype=np.float32)
        self.std = np.array(config.alignment.std, dtype=np.float32)

        # Scale reference landmarks to output size
        if self.output_size != (112, 112):
            scale_w = self.output_size[0] / 112.0
            scale_h = self.output_size[1] / 112.0
            self.src_pts = ARCFACE_SRC.copy()
            self.src_pts[:, 0] *= scale_w
            self.src_pts[:, 1] *= scale_h
        else:
            self.src_pts = ARCFACE_SRC

    def estimate_transform(self, landmarks: np.ndarray) -> trans.SimilarityTransform:
        """Estimate similarity transform from landmarks to canonical face.

        Args:
            landmarks: 5x2 array of facial landmarks

        Returns:
            Similarity transform object
        """
        if landmarks.shape != (5, 2):
            raise ValueError(f"Expected 5 landmarks, got {landmarks.shape}")

        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, self.src_pts)
        return tform

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face using 5-point landmarks.

        Args:
            image: Input image (BGR)
            landmarks: 5x2 array of facial landmarks in image coordinates

        Returns:
            Aligned face image
        """
        landmarks = np.array(landmarks, dtype=np.float32)
        
        if landmarks.shape[0] != 5:
            raise ValueError(f"Expected 5 landmarks, got {landmarks.shape[0]}")

        # Estimate transformation
        tform = self.estimate_transform(landmarks)

        # Apply transformation
        aligned = cv2.warpAffine(
            image,
            tform.params[:2],
            self.output_size,
            borderValue=0.0
        )

        # Normalize if required
        if self.normalize:
            aligned = self._normalize(aligned)

        return aligned

    def align_batch(self, image: np.ndarray, faces: list) -> list:
        """Align multiple faces from the same image.

        Args:
            image: Input image (BGR)
            faces: List of face detection dicts with 'landmarks' key

        Returns:
            List of aligned face images
        """
        aligned_faces = []
        
        for face in faces:
            if 'landmarks' not in face:
                raise ValueError("Face detection must include landmarks for alignment")
            
            landmarks = np.array(face['landmarks'], dtype=np.float32)
            aligned = self.align(image, landmarks)
            aligned_faces.append(aligned)

        return aligned_faces

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for model input.

        Args:
            image: Input image (BGR)

        Returns:
            Normalized image
        """
        # Convert to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        
        return img

    def align_crop(
        self,
        image: np.ndarray,
        bbox: list,
        landmarks: np.ndarray = None,
        margin: float = 0.2
    ) -> np.ndarray:
        """Align face using bbox and optional landmarks.

        Args:
            image: Input image (BGR)
            bbox: Bounding box [x1, y1, x2, y2]
            landmarks: Optional 5x2 landmarks
            margin: Margin to add around bbox (as fraction of bbox size)

        Returns:
            Aligned and cropped face
        """
        if landmarks is not None and len(landmarks) == 5:
            # Use landmark-based alignment (preferred)
            return self.align(image, landmarks)
        
        # Fallback to simple crop with margin
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # Add margin
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(image.shape[1], x2 + margin_w)
        y2 = min(image.shape[0], y2 + margin_h)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        # Resize to output size
        face_resized = cv2.resize(face_crop, self.output_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize if required
        if self.normalize:
            face_resized = self._normalize(face_resized)
        
        return face_resized
