#!/usr/bin/env python3
"""
Download and setup YOLO models for enhanced detection
"""

import os
import subprocess
import sys
from ultralytics import YOLO

def download_yolo_models():
    """Download YOLO models for ensemble detection"""
    models = [
        'yolov8n.pt',  # Nano - fastest
        'yolov8s.pt',  # Small - balanced
        'yolov8m.pt',  # Medium - good accuracy
        'yolov8l.pt',  # Large - high accuracy
        'yolov8x.pt',  # Extra large - highest accuracy
    ]
    
    print("Downloading YOLO models...")
    for model in models:
        try:
            print(f"Downloading {model}...")
            yolo = YOLO(model)
            print(f"✓ {model} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {model}: {e}")

def download_custom_datasets():
    """Download additional datasets for better detection"""
    datasets = {
        'COCO': 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip',
        'Open Images': 'https://storage.googleapis.com/openimages/web/download.html',
    }
    
    print("\nCustom datasets available:")
    for name, url in datasets.items():
        print(f"- {name}: {url}")
    
    print("\nFor custom training, consider:")
    print("1. COCO dataset for general objects")
    print("2. Open Images for diverse scenarios")
    print("3. Custom dataset with blind navigation scenarios")

def setup_directories():
    """Create necessary directories"""
    dirs = ['models', 'datasets', 'logs', 'recordings']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def main():
    print("Setting up enhanced YOLO detection...")
    setup_directories()
    download_yolo_models()
    download_custom_datasets()
    
    print("\n✓ Setup complete!")
    print("Models downloaded to current directory")
    print("Run 'python test.py' to start with enhanced detection")

if __name__ == "__main__":
    main()




