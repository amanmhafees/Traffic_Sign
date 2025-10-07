#!/usr/bin/env python3
"""
Traffic Sign Recognition & Driver Alert System (India)
Using YOLO v11 for Indian traffic sign detection with real-time alerts
"""

import os
import cv2
import numpy as np
import yaml
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import time
import argparse
from typing import List, Tuple, Dict, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import DataAnalyzer
from image_augmentation import ImageAugmenter
from training_visualization import TrainingVisualizer
from tqdm import tqdm  # Import tqdm for progress bar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficSignRecognition:
    def __init__(self, dataset_path: str = "Dataset", output_path: str = "output"):
        """
        Initialize the Traffic Sign Recognition system
        
        Args:
            dataset_path: Path to the dataset directory
            output_path: Path for output files
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_path / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_path / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_path / "val" / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_path / "test" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_path / "test" / "labels").mkdir(parents=True, exist_ok=True)
        
        # Class priority mapping (higher = more urgent)
        self.class_priority_map = {
            "STOP": 10,
            "NO_ENTRY": 10,
            "GIVE_WAY": 9,
            "SPEED_LIMIT_5": 8,
            "SPEED_LIMIT_15": 8,
            "SPEED_LIMIT_20": 8,
            "SPEED_LIMIT_30": 8,
            "SPEED_LIMIT_40": 8,
            "SPEED_LIMIT_50": 8,
            "SPEED_LIMIT_60": 8,
            "SPEED_LIMIT_70": 8,
            "SPEED_LIMIT_80": 8,
            "SCHOOL_AHEAD": 7,
            "PEDESTRIAN_CROSSING": 7,
            "NO_PARKING": 6,
            "NO_STOPPING_OR_STANDING": 6,
            "LEFT_TURN_PROHIBITED": 5,
            "RIGHT_TURN_PROHIBITED": 5,
            "U_TURN_PROHIBITED": 5,
            "OVERTAKING_PROHIBITED": 5,
            "HORN_PROHIBITED": 4,
            "CYCLE_PROHIBITED": 4,
            "PEDESTRIAN_PROHIBITED": 4,
            "COMPULSARY_AHEAD": 3,
            "COMPULSARY_TURN_LEFT": 3,
            "COMPULSARY_TURN_RIGHT": 3,
            "COMPULSARY_KEEP_LEFT": 3,
            "COMPULSARY_KEEP_RIGHT": 3,
            "BARRIER_AHEAD": 2,
            "CATTLE": 2,
            "CROSS_ROAD": 2,
            "DANGEROUS_DIP": 2,
            "FALLING_ROCKS": 2,
            "HUMP_OR_ROUGH_ROAD": 2,
            "LEFT_HAIR_PIN_BEND": 2,
            "LEFT_HAND_CURVE": 2,
            "LEFT_REVERSE_BEND": 2,
            "LOOSE_GRAVEL": 2,
            "MEN_AT_WORK": 2,
            "NARROW_BRIDGE": 2,
            "NARROW_ROAD_AHEAD": 2,
            "RIGHT_HAIR_PIN_BEND": 2,
            "RIGHT_HAND_CURVE": 2,
            "RIGHT_REVERSE_BEND": 2,
            "ROAD_WIDENS_AHEAD": 2,
            "ROUNDABOUT": 2,
            "SIDE_ROAD_LEFT": 2,
            "SIDE_ROAD_RIGHT": 2,
            "SLIPPERY_ROAD": 2,
            "STAGGERED_INTERSECTION": 2,
            "STEEP_ASCENT": 2,
            "STEEP_DESCENT": 2,
            "T_INTERSECTION": 2,
            "UNGUARDED_LEVEL_CROSSING": 2,
            "Y_INTERSECTION": 2
        }
        
        self.model = None
        self.class_names = []
        
        # Initialize analysis and augmentation components
        self.data_analyzer = DataAnalyzer(str(self.dataset_path), str(self.output_path))
        self.image_augmenter = ImageAugmenter(str(self.output_path))
        self.training_visualizer = TrainingVisualizer(str(self.output_path))
        
    def prepare_dataset(self, train_split: float = 0.8, val_from_train: bool = True, test_split: float = 0.1) -> None:
        """
        Prepare the dataset for YOLO training by converting to YOLO format
        
        Args:
            train_split: Fraction of data to use for training (e.g., 0.8 = 80% train, 20% val)
            val_from_train: If True, validation data is taken from training set; if False, uses separate validation
            test_split: Fraction of data to use for testing (e.g., 0.1 = 10% test)
        """
        logger.info("Preparing dataset for YOLO training...")
        
        # Get all class directories
        class_dirs = []
        for category in ["Mandatory_Traffic_Signs", "Cautionary_Traffic_Signs", "Informatory_Traffic_Signs"]:
            category_path = self.dataset_path / category
            if category_path.exists():
                class_dirs.extend([d for d in category_path.iterdir() if d.is_dir()])
        
        # Create class mapping
        self.class_names = [d.name for d in class_dirs]
        class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Process each class
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_id = class_to_id[class_name]
            
            # Get all image files
            image_files = list(class_dir.glob("*.jpg"))
            random.shuffle(image_files)
            
            # Split data into train, val, and test
            test_idx = int(len(image_files) * test_split)
            val_idx = int(len(image_files) * (1 - train_split))
            
            test_files = image_files[:test_idx]
            val_files = image_files[test_idx:val_idx]
            train_files = image_files[val_idx:]
            
            logger.info(f"Processing {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
            
            # Process training files
            for img_file in train_files:
                self._process_image(img_file, class_id, "train")
            
            # Process validation files
            for img_file in val_files:
                self._process_image(img_file, class_id, "val")
            
            # Process test files
            for img_file in test_files:
                self._process_image(img_file, class_id, "test")
        
        # Create YAML configuration file
        self._create_yaml_config()
        
        logger.info("Dataset preparation completed!")
    
    def analyze_dataset(self) -> Dict:
        """
        Perform comprehensive dataset analysis including class distribution, 
        image statistics, and imbalance detection
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting comprehensive dataset analysis...")
        
        # Run data analysis
        analysis_results = self.data_analyzer.analyze_dataset()
        
        # Get augmentation recommendations
        augmentation_recommendations = self.data_analyzer.get_augmentation_recommendations()
        
        # Print summary
        logger.info(f"Dataset Analysis Summary:")
        logger.info(f"Total Images: {analysis_results['total_images']}")
        logger.info(f"Total Classes: {analysis_results['total_classes']}")
        logger.info(f"Mean Images per Class: {analysis_results['statistics']['mean_images_per_class']:.1f}")
        logger.info(f"Imbalanced Classes: {len(analysis_results['imbalanced_classes'])}")
        
        return {
            'analysis': analysis_results,
            'augmentation_recommendations': augmentation_recommendations
        }
    
    def balance_dataset(self, target_balance: str = "mean") -> Dict:
        """
        Balance the dataset using image augmentation
        
        Args:
            target_balance: Target balance strategy ('mean', 'max', 'custom')
            
        Returns:
            Dictionary with balanced dataset information
        """
        logger.info("Starting dataset balancing with augmentation...")
        
        # First analyze the dataset
        analysis_results = self.analyze_dataset()
        # Use the analyzer's in-memory distribution with full image paths (not the JSON-serialized summary)
        class_distribution = self.data_analyzer.class_distribution
        
        # Create augmented dataset
        augmented_dataset = self.image_augmenter.create_augmented_dataset(
            class_distribution, target_balance
        )
        
        logger.info("Dataset balancing completed!")
        return augmented_dataset
    
    def _process_image(self, img_path: Path, class_id: int, split: str) -> None:
        """
        Process a single image for YOLO training
        
        Args:
            img_path: Path to the image file
            class_id: Class ID for the image
            split: 'train' or 'val'
        """
        # Copy image
        dest_img_path = self.output_path / split / "images" / img_path.name
        shutil.copy2(img_path, dest_img_path)
        
        # Create YOLO annotation (assuming full image is the sign)
        # In a real scenario, you'd have bounding box annotations
        # For now, we'll create a simple annotation assuming the sign is centered
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            
            # Create a simple bounding box (center of image, 80% of image size)
            x_center = 0.5
            y_center = 0.5
            width = 0.8
            height = 0.8
            
            # Write YOLO annotation
            label_path = self.output_path / split / "labels" / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def _create_yaml_config(self) -> None:
        """Create YAML configuration file for YOLO training"""
        config = {
            'path': str(self.output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',  # Add test dataset path
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.output_path / "traffic_signs.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Created YAML config at {yaml_path}")
    
    def train_model(self, epochs: int = 100, imgsz: int = 640, batch: int = 16, device: str = "auto", test_split: float = 0.1) -> str:
        """
        Train the YOLO model with optional test dataset evaluation.

        Args:
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            device: Device to use ('auto', 'cpu', '0', '1', etc.)
            test_split: Fraction of data to use for testing (0.1 = 10% test)

        Returns:
            Path to the best model
        """
        logger.info("Starting YOLO model training...")
        
        # Prepare dataset with test split
        self.prepare_dataset(train_split=1 - test_split, val_from_train=False)

        # Check GPU availability and limit thread usage to reduce RAM pressure
        import torch
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            if device == "auto":
                device = "0"  # Use first GPU
        else:
            logger.warning("No GPU detected, using CPU")
            device = "cpu"
        
        # Load YOLO model
        self.model = YOLO("yolo11s.pt")  # Using YOLO11s for faster training
        
        # Clear possible corrupted label cache files to avoid EOFError
        try:
            train_cache = self.output_path / "train" / "labels.cache"
            val_cache = self.output_path / "val" / "labels.cache"
            if train_cache.exists():
                train_cache.unlink(missing_ok=True)
            if val_cache.exists():
                val_cache.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not clear label cache files: {e}")

        # Set the fixed folder name for the model
        model_name = "traffic_sign_model"
        model_path = self.output_path / model_name

        # Remove the existing folder if it exists
        if model_path.exists() and model_path.is_dir():
            logger.info(f"Removing existing folder: {model_path}")
            shutil.rmtree(model_path)

        # Train the model with GPU optimization. If CUDA/memory error occurs, retry with safer settings.
        try:
            results = self.model.train(
                data=str(self.output_path / "traffic_signs.yaml"),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                workers=4 if device != "cpu" else 0,
                augment=True,
                patience=20,
                save=True,
                save_period=10,
                project=str(self.output_path),
                name=model_name,
                cache=False,
                amp=True if device != "cpu" else False,
                val=True,
                plots=False,  # disable heavy plotting to reduce RAM
                verbose=True,
            )
        except RuntimeError as e:
            err_msg = str(e).lower()
            if ("cuda" in err_msg) or ("out of memory" in err_msg) or ("memory allocation" in err_msg):
                logger.warning("CUDA/memory error detected. Retrying training with safer CPU settings...")
                # Retry with safer defaults
                safe_imgsz = 320 if imgsz > 320 else imgsz
                safe_batch = 2 if batch > 2 else batch
                results = self.model.train(
                    data=str(self.output_path / "traffic_signs.yaml"),
                    epochs=max(min(epochs, 50), 1),
                    imgsz=safe_imgsz,
                    batch=safe_batch,
                    device="cpu",
                    workers=0,
                    augment=True,
                    patience=10,
                    save=True,
                    save_period=10,
                    project=str(self.output_path),
                    name=model_name,
                    cache=False,
                    amp=False,
                    val=True,
                    plots=False,
                    verbose=True,
                    # memory-friendly options
                    rect=True,
                    mosaic=0.0,
                    erasing=0.0,
                    fraction=0.25,
                )
            else:
                raise
        
        # Get the best model path
        best_model_path = model_path / "weights" / "best.pt"
        last_model_path = model_path / "weights" / "last.pt"
        
        # Also save a copy with a generic name for easy access
        generic_best_path = self.output_path / "best_model.pt"
        generic_last_path = self.output_path / "last_model.pt"
        
        if best_model_path.exists():
            # Copy best model to generic location
            shutil.copy2(str(best_model_path), str(generic_best_path))
            logger.info(f"Training completed! Best model saved at {best_model_path}")
            logger.info(f"Best model also copied to {generic_best_path}")
        else:
            logger.warning("Best model not found, checking for last model...")
            if last_model_path.exists():
                shutil.copy2(str(last_model_path), str(generic_last_path))
                logger.info(f"Last model saved at {last_model_path}")
                logger.info(f"Last model also copied to {generic_last_path}")
                return str(last_model_path)
            else:
                raise FileNotFoundError("No trained model found after training")
        
        # Save training results summary
        self._save_training_summary(results, model_name)
        
        # Generate comprehensive visualizations
        self._generate_training_visualizations(model_name)
        
        # Evaluate on the test dataset
        logger.info("Evaluating model on the test dataset...")
        test_metrics = self.model.val(split='test')
        logger.info(f"Test dataset evaluation completed! mAP50: {test_metrics.box.map50:.3f}, mAP50-95: {test_metrics.box.map:.3f}")
        
        return str(best_model_path)
    
    def _save_training_summary(self, results, model_name: str) -> None:
        """
        Save training results summary
        
        Args:
            results: Training results from YOLO
            model_name: Name of the trained model
        """
        try:
            summary_path = self.output_path / f"{model_name}_training_summary.txt"
            
            with open(summary_path, 'w') as f:
                f.write(f"Traffic Sign Recognition Model Training Summary\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Training Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of Classes: {len(self.class_names)}\n")
                f.write(f"Classes: {', '.join(self.class_names)}\n")
                f.write(f"\nTraining Results:\n")
                
                if hasattr(results, 'results_dict'):
                    for key, value in results.results_dict.items():
                        f.write(f"{key}: {value}\n")
                
                f.write(f"\nModel saved at: {self.output_path / model_name}\n")
                f.write(f"Best model: {self.output_path / model_name / 'weights' / 'best.pt'}\n")
                f.write(f"Last model: {self.output_path / model_name / 'weights' / 'last.pt'}\n")
            
            logger.info(f"Training summary saved to {summary_path}")
            
        except Exception as e:
            logger.warning(f"Could not save training summary: {e}")
    
    def _generate_training_visualizations(self, model_name: str) -> None:
        """
        Generate comprehensive training visualizations
        
        Args:
            model_name: Name of the trained model
        """
        try:
            logger.info("Generating comprehensive training visualizations...")
            
            # Generate training curves and interactive dashboard
            self.training_visualizer.generate_comprehensive_report(model_name)
            
            # Generate additional plots if results file exists
            results_file = self.output_path / model_name / "results.csv"
            if results_file.exists():
                self.training_visualizer.plot_training_curves(str(results_file), model_name)
                self.training_visualizer.create_interactive_dashboard(str(results_file), model_name)
            
            logger.info("Training visualizations generated successfully!")
            
        except Exception as e:
            logger.warning(f"Could not generate training visualizations: {e}")
    
    def validate_model(self, model_path: str) -> Dict:
        """
        Validate the trained model
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Validation metrics
        """
        logger.info("Validating model...")
        
        if self.model is None:
            self.model = YOLO(model_path)
        
        # Run validation
        metrics = self.model.val()
        
        logger.info(f"Validation completed! mAP50: {metrics.box.map50:.3f}, mAP50-95: {metrics.box.map:.3f}")
        return metrics
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model
        
        Args:
            model_path: Path to the trained model
        """
        model_path = Path(model_path)  # Ensure model_path is a Path object
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please ensure the model file exists.")
        
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(str(model_path))
    
    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple]:
        """
        Process a single frame for traffic sign detection
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            
        Returns:
            List of detected signs with (class_name, confidence, bbox, area)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess frame for better detection in low visibility
        processed_frame = self._preprocess_frame(frame)
        
        # Run inference
        results = self.model.predict(processed_frame, conf=conf_threshold, verbose=False)
        
        alerts = []
        for res in results:
            if res.boxes is not None:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = res.names[cls_id]
                    conf_score = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    alerts.append((cls_name, conf_score, (x1, y1, x2, y2), area))
        
        # Sort by priority and size
        alerts.sort(key=lambda x: (self.class_priority_map.get(x[0], 0), x[3]), reverse=True)
        
        return alerts
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better detection in low visibility conditions
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        """
        Draw detection results on the frame
        
        Args:
            frame: Input frame
            detections: List of detections from process_frame()
            
        Returns:
            Frame with detections drawn
        """
        result_frame = frame.copy()
        
        for i, (cls_name, conf_score, bbox, area) in enumerate(detections):
            x1, y1, x2, y2 = bbox
            
            # Choose color based on priority
            priority = self.class_priority_map.get(cls_name, 0)
            if priority >= 8:
                color = (0, 0, 255)  # Red for high priority
            elif priority >= 5:
                color = (0, 165, 255)  # Orange for medium priority
            else:
                color = (0, 255, 0)  # Green for low priority
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name} {conf_score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw priority alert for high priority signs
            if priority >= 7:
                alert_text = f"⚠️ {cls_name.upper()} AHEAD"
                cv2.putText(result_frame, alert_text, (50, 50 + i * 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return result_frame
    
    def run_realtime_detection(self, video_source: str = "0", conf_threshold: float = 0.4) -> None:
        """
        Run real-time traffic sign detection
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
            conf_threshold: Confidence threshold
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting real-time detection...")
        
        # Open video capture
        cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections = self.process_frame(frame, conf_threshold)
                
                # Draw detections
                result_frame = self.draw_detections(frame, detections)
                
                # Add info text
                cv2.putText(result_frame, f"Detections: {len(detections)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result_frame, "Press 'q' to quit", (10, result_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow("Traffic Sign Recognition", result_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def export_model(self, model_path: str, format: str = "onnx") -> str:
        """
        Export the model to different formats
        
        Args:
            model_path: Path to the model
            format: Export format ('onnx', 'tensorrt', 'coreml', etc.)
            
        Returns:
            Path to exported model
        """
        logger.info(f"Exporting model to {format} format...")
        
        if self.model is None:
            self.model = YOLO(model_path)
        
        # Export model
        exported_path = self.model.export(format=format)
        
        logger.info(f"Model exported to: {exported_path}")
        return exported_path

def main():
    """Main function to run the traffic sign recognition system"""
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition with YOLO v11")
    parser.add_argument("--mode", choices=["prepare", "analyze", "balance", "train", "validate", "detect", "export"], 
                       default="train", help="Operation mode")
    parser.add_argument("--dataset", default="Dataset", help="Dataset path")
    parser.add_argument("--output", default="output", help="Output path")
    parser.add_argument("--model", help="Path to trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs") #running epoch seting number
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, 0, 1, etc.)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--source", default="0", help="Video source for detection")
    parser.add_argument("--format", default="onnx", help="Export format")
    parser.add_argument("--balance", choices=["mean", "max", "custom"], default="mean", 
                       help="Dataset balancing strategy")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis without training")
    
    args = parser.parse_args()
    
    # Initialize system
    tsr = TrafficSignRecognition(args.dataset, args.output)
    
    try:
        if args.mode == "prepare":
            tsr.prepare_dataset(val_from_train=True)
            
        elif args.mode == "analyze":
            print("Running comprehensive dataset analysis...")
            analysis_results = tsr.analyze_dataset()
            print(f"Analysis completed! Results saved to {tsr.output_path / 'analysis'}")
            
        elif args.mode == "balance":
            print("Balancing dataset with augmentation...")
            balanced_dataset = tsr.balance_dataset(args.balance)
            print(f"Dataset balancing completed! Augmented data saved to {tsr.output_path / 'augmented_data'}")
            
        elif args.mode == "train":
            # Run analysis first if requested
            if args.analyze_only:
                print("Running dataset analysis...")
                tsr.analyze_dataset()
                print("Analysis completed!")
                return
            
            # Prepare dataset
            tsr.prepare_dataset(val_from_train=True)
            
            # Optionally balance dataset
            if args.balance != "none":
                print(f"Balancing dataset using {args.balance} strategy...")
                tsr.balance_dataset(args.balance)
            
            # Train model
            model_path = tsr.train_model(args.epochs, args.imgsz, args.batch, args.device)
            print(f"Training completed! Model saved at: {model_path}")
            print(f"Best model also available at: {tsr.output_path / 'best_model.pt'}")
            print(f"Visualizations saved to: {tsr.output_path / 'visualizations'}")
            
        elif args.mode == "validate":
            if not args.model:
                raise ValueError("Model path required for validation")
            tsr.load_model(args.model)
            metrics = tsr.validate_model(args.model)
            print(f"Validation metrics: {metrics}")
            
        elif args.mode == "detect":
            if not args.model:
                raise ValueError("Model path required for detection")
            tsr.load_model(args.model)
            tsr.run_realtime_detection(args.source, args.conf)
            
        elif args.mode == "export":
            if not args.model:
                raise ValueError("Model path required for export")
            exported_path = tsr.export_model(args.model, args.format)
            print(f"Model exported to: {exported_path}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()