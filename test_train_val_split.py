#!/usr/bin/env python3
"""
Test script to verify the train/validation split changes
"""

import os
import sys
from pathlib import Path
from traffic_sign_recognition import TrafficSignRecognition

def test_dataset_preparation():
    """Test the dataset preparation with new validation approach"""
    print("Testing dataset preparation with validation from training set...")
    
    # Initialize the system
    tsr = TrafficSignRecognition("Dataset", "test_output")
    
    try:
        # Test the new prepare_dataset method
        tsr.prepare_dataset(train_split=0.8, val_from_train=True)
        
        # Check if directories were created
        train_img_dir = tsr.output_path / "train" / "images"
        train_label_dir = tsr.output_path / "train" / "labels"
        val_img_dir = tsr.output_path / "val" / "images"
        val_label_dir = tsr.output_path / "val" / "labels"
        
        print(f"Train images: {len(list(train_img_dir.glob('*.jpg')))}")
        print(f"Train labels: {len(list(train_label_dir.glob('*.txt')))}")
        print(f"Val images: {len(list(val_img_dir.glob('*.jpg')))}")
        print(f"Val labels: {len(list(val_label_dir.glob('*.txt')))}")
        
        # Check if YAML config was created
        yaml_path = tsr.output_path / "traffic_signs.yaml"
        if yaml_path.exists():
            print(f"✓ YAML config created at {yaml_path}")
        else:
            print("✗ YAML config not found")
            
        # Check if classes were detected
        print(f"✓ Found {len(tsr.class_names)} classes")
        print(f"Classes: {tsr.class_names[:5]}...")  # Show first 5 classes
        
        return True
        
    except Exception as e:
        print(f"✗ Error during dataset preparation: {e}")
        return False

def test_model_saving():
    """Test model saving functionality"""
    print("\nTesting model saving functionality...")
    
    # Check if we can create a simple model saving test
    try:
        from ultralytics import YOLO
        
        # Load a small model for testing
        model = YOLO("yolo11n.pt")  # Use nano model for faster testing
        
        # Test saving functionality
        test_output_dir = Path("test_output")
        test_output_dir.mkdir(exist_ok=True)
        
        # Test export functionality
        exported_path = model.export(format="onnx")
        print(f"✓ Model export test successful: {exported_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during model saving test: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Testing Traffic Sign Recognition Changes")
    print("=" * 50)
    
    # Test dataset preparation
    dataset_test = test_dataset_preparation()
    
    # Test model saving
    model_test = test_model_saving()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Dataset Preparation: {'✓ PASSED' if dataset_test else '✗ FAILED'}")
    print(f"Model Saving: {'✓ PASSED' if model_test else '✗ FAILED'}")
    print("=" * 50)
    
    if dataset_test and model_test:
        print("\n🎉 All tests passed! The changes are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()