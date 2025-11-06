#!/usr/bin/env python3
"""
Create a demo GIF showing face recognition in action.
This script creates a sequence of images showing the detection process.
"""

import cv2
import numpy as np
import requests
import json
from pathlib import Path
import time

def create_demo_sequence(image_path: str, output_dir: str = "demo_frames"):
    """Create a sequence of images showing the detection process."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # Frame 1: Original image
    cv2.putText(image, "Original Image", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(image, "Original Image", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.imwrite(str(output_path / "frame_01_original.jpg"), image)
    
    # Frame 2: Processing indicator
    img_processing = image.copy()
    cv2.putText(img_processing, "Processing...", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img_processing, "Processing...", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
    
    # Add loading animation
    for i in range(3):
        dots = "." * (i + 1)
        img_frame = img_processing.copy()
        cv2.putText(img_frame, f"Detecting faces{dots}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(img_frame, f"Detecting faces{dots}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(str(output_path / f"frame_02_{i+1}_processing.jpg"), img_frame)
    
    # Get detection results
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/detect', files=files)
        
        if response.status_code == 200:
            result = response.json()
            faces = result['faces']
        else:
            print(f"API Error: {response.status_code}")
            return
    except Exception as e:
        print(f"Error calling API: {e}")
        return
    
    # Frame 3: Detection results
    img_detected = image.copy()
    
    if faces:
        for i, face in enumerate(faces):
            bbox = face['bbox']
            confidence = face['confidence']
            landmarks = face['landmarks']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box with animation effect
            cv2.rectangle(img_detected, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # Draw confidence
            label = f"Face: {confidence:.1%}"
            cv2.putText(img_detected, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(img_detected, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Draw landmarks
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for j, (lx, ly) in enumerate(landmarks):
                color = colors[j % len(colors)]
                cv2.circle(img_detected, (int(lx), int(ly)), 8, color, -1)
                cv2.circle(img_detected, (int(lx), int(ly)), 10, (255, 255, 255), 2)
    
    # Add title
    cv2.putText(img_detected, "Face Detected!", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img_detected, "Face Detected!", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path / "frame_03_detected.jpg"), img_detected)
    
    # Frame 4: Analysis results
    img_analysis = img_detected.copy()
    
    if faces:
        face = faces[0]  # Use first face
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox
        face_w, face_h = x2 - x1, y2 - y1
        
        # Add analysis text
        analysis_text = [
            f"Confidence: {face['confidence']:.1%}",
            f"Size: {face_w}x{face_h}px",
            f"Quality: {face['quality_score']:.3f}",
            f"Processing: {result['processing_time_ms']:.0f}ms"
        ]
        
        for i, text in enumerate(analysis_text):
            y_pos = 100 + i * 40
            cv2.putText(img_analysis, text, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(img_analysis, text, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.putText(img_analysis, "Analysis Complete", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img_analysis, "Analysis Complete", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path / "frame_04_analysis.jpg"), img_analysis)
    
    print(f"✅ Demo frames created in: {output_path}")
    print("📁 Files created:")
    for frame_file in sorted(output_path.glob("frame_*.jpg")):
        print(f"   - {frame_file.name}")
    
    print("\n🎬 To create GIF (requires ImageMagick):")
    print(f"   convert -delay 100 {output_path}/frame_*.jpg demo_face_recognition.gif")
    
    return str(output_path)

def create_comparison_image(image_path: str):
    """Create a before/after comparison image."""
    
    # Load original
    original = cv2.imread(image_path)
    if original is None:
        return
    
    # Get detection results
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/detect', files=files)
        
        result = response.json()
        faces = result['faces']
    except:
        return
    
    # Create detected version
    detected = original.copy()
    
    for face in faces:
        bbox = face['bbox']
        landmarks = face['landmarks']
        confidence = face['confidence']
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(detected, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw landmarks
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for j, (lx, ly) in enumerate(landmarks):
            color = colors[j % len(colors)]
            cv2.circle(detected, (int(lx), int(ly)), 6, color, -1)
        
        # Add confidence
        cv2.putText(detected, f"{confidence:.1%}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Create side-by-side comparison
    h, w = original.shape[:2]
    comparison = np.hstack([original, detected])
    
    # Add labels
    cv2.putText(comparison, "ORIGINAL", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(comparison, "ORIGINAL", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    cv2.putText(comparison, "DETECTED", (w + 50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(comparison, "DETECTED", (w + 50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    cv2.imwrite("demo_comparison.jpg", comparison)
    print("✅ Comparison image saved: demo_comparison.jpg")

if __name__ == "__main__":
    image_path = "/Users/a91788/Downloads/IMG_1869.jpg"
    
    if Path(image_path).exists():
        print("🎬 Creating demo sequence...")
        create_demo_sequence(image_path)
        
        print("\n📸 Creating comparison image...")
        create_comparison_image(image_path)
        
        print("\n🎯 Demo materials ready!")
        print("   - Demo frames in demo_frames/ directory")
        print("   - Comparison image: demo_comparison.jpg")
    else:
        print(f"❌ Image not found: {image_path}")
        print("Please update the image_path variable in the script.")