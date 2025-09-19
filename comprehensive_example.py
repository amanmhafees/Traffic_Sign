#!/usr/bin/env python3
"""
Comprehensive Example Script for Traffic Sign Recognition
Demonstrates all features including analysis, augmentation, and visualization
"""

import os
import sys
from pathlib import Path
from traffic_sign_recognition import TrafficSignRecognition
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_comprehensive_analysis():
    """Run comprehensive dataset analysis"""
    print("=" * 60)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 60)
    
    # Initialize system
    tsr = TrafficSignRecognition("Dataset", "output")
    
    # Run analysis
    analysis_results = tsr.analyze_dataset()
    
    print("\nAnalysis Results:")
    print(f"Total Images: {analysis_results['analysis']['total_images']}")
    print(f"Total Classes: {analysis_results['analysis']['total_classes']}")
    print(f"Mean Images per Class: {analysis_results['analysis']['statistics']['mean_images_per_class']:.1f}")
    print(f"Imbalanced Classes: {len(analysis_results['analysis']['imbalanced_classes'])}")
    
    # Show imbalanced classes
    if analysis_results['analysis']['imbalanced_classes']:
        print("\nImbalanced Classes (need more data):")
        for cls in analysis_results['analysis']['imbalanced_classes'][:5]:  # Show top 5
            print(f"  - {cls['class']}: {cls['count']} images (deficit: {cls['deficit']})")
    
    print(f"\nAnalysis plots saved to: {tsr.output_path / 'analysis'}")
    return analysis_results

def run_dataset_balancing():
    """Run dataset balancing with augmentation"""
    print("\n" + "=" * 60)
    print("DATASET BALANCING WITH AUGMENTATION")
    print("=" * 60)
    
    # Initialize system
    tsr = TrafficSignRecognition("Dataset", "output")
    
    # Balance dataset
    balanced_dataset = tsr.balance_dataset("mean")
    
    print("\nBalancing Results:")
    total_original = sum(data['original_count'] for data in balanced_dataset.values())
    total_augmented = sum(data['augmented_count'] for data in balanced_dataset.values())
    
    print(f"Total Original Images: {total_original}")
    print(f"Total Augmented Images: {total_augmented}")
    print(f"Increase: {total_augmented - total_original} images")
    
    # Show classes that were augmented
    augmented_classes = [name for name, data in balanced_dataset.items() 
                        if data['augmentation_type'] != 'none']
    print(f"\nClasses Augmented: {len(augmented_classes)}")
    
    print(f"\nAugmented data saved to: {tsr.output_path / 'augmented_data'}")
    return balanced_dataset

def run_training_with_visualization():
    """Run training with comprehensive visualization"""
    print("\n" + "=" * 60)
    print("TRAINING WITH COMPREHENSIVE VISUALIZATION")
    print("=" * 60)
    
    # Initialize system
    tsr = TrafficSignRecognition("Dataset", "output")
    
    # Prepare dataset
    print("Preparing dataset...")
    tsr.prepare_dataset(val_from_train=True)
    
    # Train model
    print("Starting training...")
    model_path = tsr.train_model(epochs=5, imgsz=640, batch=8, device="auto")
    
    print(f"\nTraining completed!")
    print(f"Model saved at: {model_path}")
    print(f"Best model: {tsr.output_path / 'best_model.pt'}")
    print(f"Visualizations: {tsr.output_path / 'visualizations'}")
    
    return model_path

def run_validation_and_testing(model_path):
    """Run validation and testing"""
    print("\n" + "=" * 60)
    print("MODEL VALIDATION AND TESTING")
    print("=" * 60)
    
    # Initialize system
    tsr = TrafficSignRecognition("Dataset", "output")
    
    # Load model
    tsr.load_model(model_path)
    
    # Run validation
    print("Running validation...")
    metrics = tsr.validate_model(model_path)
    
    print(f"Validation completed!")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    
    return metrics

def demonstrate_augmentation_visualization():
    """Demonstrate augmentation visualization"""
    print("\n" + "=" * 60)
    print("AUGMENTATION VISUALIZATION DEMO")
    print("=" * 60)
    
    # Initialize system
    tsr = TrafficSignRecognition("Dataset", "output")
    
    # Find a sample image
    sample_paths = [
        "Dataset/Mandatory_Traffic_Signs/STOP/02000.jpg",
        "Dataset/Mandatory_Traffic_Signs/SPEED_LIMIT_30/02000.jpg",
        "Dataset/Cautionary_Traffic_Signs/SCHOOL_AHEAD/02000.jpg"
    ]
    
    for sample_path in sample_paths:
        if Path(sample_path).exists():
            print(f"Creating augmentation visualization for {sample_path}")
            tsr.image_augmenter.visualize_augmentations(
                Path(sample_path), 
                Path(sample_path).parent.name
            )
            break
    
    print(f"Augmentation examples saved to: {tsr.output_path / 'augmented_data'}")

def main():
    """Main function to run comprehensive example"""
    print("Traffic Sign Recognition - Comprehensive Example")
    print("This example demonstrates all features including:")
    print("- Dataset analysis and visualization")
    print("- Class imbalance detection")
    print("- Image augmentation")
    print("- Training with comprehensive visualization")
    print("- Model validation and testing")
    
    try:
        # Step 1: Run comprehensive analysis
        analysis_results = run_comprehensive_analysis()
        
        # Step 2: Run dataset balancing
        balanced_dataset = run_dataset_balancing()
        
        # Step 3: Demonstrate augmentation visualization
        demonstrate_augmentation_visualization()
        
        # Step 4: Run training with visualization
        model_path = run_training_with_visualization()
        
        # Step 5: Run validation
        validation_metrics = run_validation_and_testing(model_path)
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EXAMPLE COMPLETED!")
        print("=" * 60)
        print("\nGenerated outputs:")
        print(f"📊 Analysis plots: output/analysis/")
        print(f"🔄 Augmented data: output/augmented_data/")
        print(f"📈 Training visualizations: output/visualizations/")
        print(f"🤖 Trained model: {model_path}")
        print(f"📋 Training summary: output/*_training_summary.txt")
        
        print("\nTo view the results:")
        print("1. Check the analysis/ directory for class distribution plots")
        print("2. Check the augmented_data/ directory for augmentation examples")
        print("3. Check the visualizations/ directory for training curves")
        print("4. Use the trained model for inference")
        
    except Exception as e:
        logger.error(f"Error in comprehensive example: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()