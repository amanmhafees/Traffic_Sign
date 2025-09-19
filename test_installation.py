#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'ultralytics',
        'cv2',
        'numpy',
        'yaml',
        'torch',
        'torchvision',
        'PIL',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {failed_imports}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    else:
        print("\nAll packages imported successfully!")
        return True

def test_yolo():
    """Test YOLO model loading"""
    try:
        from ultralytics import YOLO
        print("\nTesting YOLO model loading...")
        
        # Try to load a small YOLO model
        model = YOLO("yolo11s.pt")
        print("✓ YOLO model loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ YOLO model loading failed: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    try:
        import cv2
        import numpy as np
        
        print("\nTesting OpenCV functionality...")
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:] = (255, 0, 0)  # Blue color
        
        # Test basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(test_image, (5, 5), 0)
        
        print("✓ OpenCV basic operations working!")
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def test_dataset_structure():
    """Test if dataset structure is correct"""
    import os
    from pathlib import Path
    
    print("\nTesting dataset structure...")
    
    dataset_path = Path("Dataset")
    if not dataset_path.exists():
        print("✗ Dataset directory not found!")
        return False
    
    # Check for main categories
    categories = ["Mandatory_Traffic_Signs", "Cautionary_Traffic_Signs", "Informatory_Traffic_Signs"]
    found_categories = []
    
    for category in categories:
        category_path = dataset_path / category
        if category_path.exists():
            found_categories.append(category)
            print(f"✓ Found {category}")
        else:
            print(f"✗ Missing {category}")
    
    if len(found_categories) > 0:
        print(f"\nFound {len(found_categories)} categories: {found_categories}")
        
        # Check for image files in first category
        first_category = dataset_path / found_categories[0]
        class_dirs = [d for d in first_category.iterdir() if d.is_dir()]
        
        if class_dirs:
            first_class = class_dirs[0]
            image_files = list(first_class.glob("*.jpg"))
            print(f"✓ Found {len(image_files)} images in {first_class.name}")
            return True
        else:
            print("✗ No class directories found")
            return False
    else:
        print("✗ No valid categories found")
        return False

def test_traffic_sign_recognition():
    """Test the main TrafficSignRecognition class"""
    try:
        from traffic_sign_recognition import TrafficSignRecognition
        
        print("\nTesting TrafficSignRecognition class...")
        
        # Initialize the class
        tsr = TrafficSignRecognition("Dataset", "test_output")
        print("✓ TrafficSignRecognition initialized successfully!")
        
        # Test dataset preparation (without actually running it)
        print("✓ Class methods accessible!")
        return True
    except Exception as e:
        print(f"✗ TrafficSignRecognition test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Traffic Sign Recognition - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("YOLO Model", test_yolo),
        ("OpenCV", test_opencv),
        ("Dataset Structure", test_dataset_structure),
        ("TrafficSignRecognition Class", test_traffic_sign_recognition)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("1. Prepare dataset: python traffic_sign_recognition.py --mode prepare")
        print("2. Train model: python traffic_sign_recognition.py --mode train")
        print("3. Run detection: python traffic_sign_recognition.py --mode detect --model <model_path>")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check dataset structure matches the expected format")
        print("3. Ensure you have sufficient disk space and permissions")

if __name__ == "__main__":
    main() 