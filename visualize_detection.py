#!/usr/bin/env python3
"""
Visualize face detection results by drawing bounding boxes and landmarks on the image.
"""

import cv2
import numpy as np
import requests
import json
import sys
from pathlib import Path

def visualize_detection(image_path: str, output_path: str = None):
    """
    Visualize face detection results on an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image (optional)
    """
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Loaded image: {image.shape} (H x W x C)")
    
    # Call the detection API
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/detect', files=files)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return
        
        result = response.json()
        print(f"API Response: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"Error calling API: {e}")
        return
    
    # Draw detection results
    img_draw = image.copy()
    
    for i, face in enumerate(result['faces']):
        # Extract face info
        bbox = face['bbox']
        confidence = face['confidence']
        landmarks = face['landmarks']
        quality_score = face.get('quality_score', 0)
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw confidence and quality scores
        label = f"Face {i+1}: {confidence:.3f} (Q: {quality_score:.3f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Background for text
        cv2.rectangle(img_draw, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     (0, 255, 0), -1)
        
        # Text
        cv2.putText(img_draw, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw landmarks (facial keypoints)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        landmark_labels = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
        
        for j, (lx, ly) in enumerate(landmarks):
            color = colors[j % len(colors)]
            cv2.circle(img_draw, (int(lx), int(ly)), 8, color, -1)
            cv2.circle(img_draw, (int(lx), int(ly)), 10, (255, 255, 255), 2)
            
            # Label landmarks
            cv2.putText(img_draw, landmark_labels[j], 
                       (int(lx) + 15, int(ly) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw face dimensions
        face_w = x2 - x1
        face_h = y2 - y1
        dim_label = f"{face_w}x{face_h}px"
        cv2.putText(img_draw, dim_label, (x1, y2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add summary info
    summary = f"Detected: {result['num_faces']} faces | Processing: {result['processing_time_ms']:.1f}ms"
    cv2.putText(img_draw, summary, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_draw, summary, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, img_draw)
        print(f"Saved visualization to: {output_path}")
    else:
        # Auto-generate output filename
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_detection{input_path.suffix}"
        cv2.imwrite(str(output_path), img_draw)
        print(f"Saved visualization to: {output_path}")
    
    # Also create a side-by-side comparison
    if image.shape[1] > 2000:  # If image is very wide, stack vertically
        comparison = np.vstack([image, img_draw])
    else:  # Stack horizontally
        comparison = np.hstack([image, img_draw])
    
    comparison_path = Path(image_path).parent / f"{Path(image_path).stem}_comparison{Path(image_path).suffix}"
    cv2.imwrite(str(comparison_path), comparison)
    print(f"Saved comparison to: {comparison_path}")
    
    return str(output_path), str(comparison_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_detection.py <image_path> [output_path]")
        print("Example: python visualize_detection.py /Users/a91788/Downloads/IMG_1869.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    visualize_detection(image_path, output_path)