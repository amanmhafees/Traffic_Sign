#!/usr/bin/env python3
"""
Quick Start Script for Traffic Sign Recognition
This script demonstrates the basic workflow for training and testing the model
"""

import os
import sys
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("Checking prerequisites...")
    
    # Check if Dataset directory exists
    if not Path("Dataset").exists():
        print("❌ Dataset directory not found!")
        print("Please ensure you have the Dataset folder with traffic sign images.")
        return False
    
    # Check if required packages are installed
    try:
        import ultralytics
        import cv2
        import numpy as np
        print("✓ Required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def run_workflow():
    """Run the complete workflow"""
    print("\n" + "="*60)
    print("TRAFFIC SIGN RECOGNITION - QUICK START")
    print("="*60)
    
    if not check_prerequisites():
        return
    
    try:
        from traffic_sign_recognition import TrafficSignRecognition
        
        # Initialize the system
        print("\n1. Initializing Traffic Sign Recognition system...")
        tsr = TrafficSignRecognition("Dataset", "output")
        
        # Step 1: Prepare dataset
        print("\n2. Preparing dataset for YOLO training...")
        tsr.prepare_dataset()
        print("✓ Dataset preparation completed!")
        
        # Step 2: Train model (with reduced epochs for quick demo)
        print("\n3. Training YOLO model...")
        print("Note: This will take some time. Using 10 epochs for quick demo.")
        
        # Check for GPU
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            device = "0"
        else:
            print("⚠️ No GPU detected, using CPU (training will be slower)")
            device = "cpu"
            
        model_path = tsr.train_model(epochs=10, batch=8, imgsz=416, device=device)
        print(f"✓ Training completed! Model saved at: {model_path}")
        
        # Step 3: Validate model
        print("\n4. Validating model...")
        metrics = tsr.validate_model(model_path)
        print("✓ Validation completed!")
        
        # Step 4: Test with webcam (optional)
        print("\n5. Testing real-time detection...")
        print("Press 'q' to quit the detection window")
        
        response = input("Do you want to test real-time detection? (y/n): ").lower().strip()
        if response == 'y':
            tsr.load_model(model_path)
            tsr.run_realtime_detection(conf_threshold=0.3)
        
        # Step 5: Export model
        print("\n6. Exporting model for deployment...")
        exported_path = tsr.export_model(model_path, "onnx")
        print(f"✓ Model exported to: {exported_path}")
        
        print("\n" + "="*60)
        print("🎉 QUICK START COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nWhat you can do next:")
        print("1. Train with more epochs for better accuracy:")
        print("   python traffic_sign_recognition.py --mode train --epochs 100")
        print("\n2. Test with your own video:")
        print("   python traffic_sign_recognition.py --mode detect --model output/traffic_sign_model/weights/best.pt --source your_video.mp4")
        print("\n3. Export to other formats:")
        print("   python traffic_sign_recognition.py --mode export --model output/traffic_sign_model/weights/best.pt --format tensorrt")
        
    except Exception as e:
        print(f"\n❌ Error during workflow: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have sufficient disk space")
        print("2. Check if you have GPU support for faster training")
        print("3. Try reducing batch size if you get memory errors")
        print("4. Run test_installation.py to verify your setup")

def main():
    """Main function"""
    print("Traffic Sign Recognition - Quick Start")
    print("This script will guide you through the complete workflow.")
    
    response = input("\nDo you want to proceed? (y/n): ").lower().strip()
    if response == 'y':
        run_workflow()
    else:
        print("Quick start cancelled.")

if __name__ == "__main__":
    main() 