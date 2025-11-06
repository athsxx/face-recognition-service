#!/usr/bin/env python3
"""
Show detailed face detection information in a readable format.
"""

import requests
import json
import sys

def show_detection_info(image_path: str):
    """Show detailed detection information."""
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/detect', files=files)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return
        
        result = response.json()
        
        print("🔍 FACE DETECTION RESULTS")
        print("=" * 50)
        print(f"📊 Total faces detected: {result['num_faces']}")
        print(f"⏱️  Processing time: {result['processing_time_ms']:.1f}ms")
        print()
        
        for i, face in enumerate(result['faces']):
            print(f"👤 FACE #{i+1}")
            print("-" * 30)
            
            # Bounding box
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            print(f"📦 Bounding Box:")
            print(f"   Top-left: ({x1}, {y1})")
            print(f"   Bottom-right: ({x2}, {y2})")
            print(f"   Size: {width} x {height} pixels")
            print()
            
            # Confidence and quality
            print(f"🎯 Confidence: {face['confidence']:.3f} ({face['confidence']*100:.1f}%)")
            print(f"✨ Quality Score: {face['quality_score']:.3f}")
            print()
            
            # Landmarks
            landmarks = face['landmarks']
            landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth Corner', 'Right Mouth Corner']
            
            print("📍 Facial Landmarks:")
            for j, (name, (lx, ly)) in enumerate(zip(landmark_names, landmarks)):
                print(f"   {name}: ({lx}, {ly})")
            print()
            
            # Face analysis
            print("🔬 Face Analysis:")
            print(f"   Face covers {width/1737*100:.1f}% of image width")
            print(f"   Face covers {height/3088*100:.1f}% of image height")
            
            # Eye distance (for scale reference)
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            eye_distance = ((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)**0.5
            print(f"   Eye distance: {eye_distance:.0f} pixels")
            print()
        
        if result['num_faces'] == 0:
            print("❌ No faces detected")
            print("💡 Try adjusting detection thresholds in config.yaml")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_detection_info.py <image_path>")
        print("Example: python show_detection_info.py /Users/a91788/Downloads/IMG_1869.jpg")
        sys.exit(1)
    
    show_detection_info(sys.argv[1])