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
from metrics_logger import MetricsLogger, generate_plots_from_results
# from ultralytics import YOLO # Lazy loaded
import time
import argparse
from typing import List, Tuple, Dict, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import DataAnalyzer
# from image_augmentation import ImageAugmenter
from training_visualization import TrainingVisualizer
from tqdm import tqdm
import datetime

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
        print("DEBUG: Initializing TrafficSignRecognition...", flush=True)
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
        
        self.model = None
        self.class_names = []
        # Lazy-init heavy components
        self.data_analyzer = None
        self.image_augmenter = None
        self.training_visualizer = None
        self.runtime_device: str = "cpu"
        
        # Mapping from Original Class ID (string/int in .txt) -> New Continuous Index
        self.id_mapping = {} 

    def _newest_mtime(self, root: Path) -> float:
        newest = 0.0
        try:
            for p in root.rglob("*.jpg"):
                try:
                    newest = max(newest, p.stat().st_mtime)
                except Exception:
                    pass
            for p in root.rglob("*.txt"):
                try:
                    newest = max(newest, p.stat().st_mtime)
                except Exception:
                    pass
        except Exception:
            return newest
        return newest

    def _output_is_stale(self) -> bool:
        """Return True if Dataset has newer files than prepared output splits."""
        if not (self.output_path / "train").exists():
            return True
        ds_mtime = self._newest_mtime(self.dataset_path)
        out_mtime = self._newest_mtime(self.output_path)
        return ds_mtime > out_mtime
        
    def _ensure_min_images_per_class(self, min_count: int = 300) -> None:
        """
        Top-up each class to have at least `min_count` images using augmentation.
        Updates Dataset folder with new images AND corresponding .txt labels.
        """
        categories = ["Mandatory_Traffic_Signs", "Cautionary_Traffic_Signs"] # Removed Informatory
        # Lazy init augmenter
        if self.image_augmenter is None:
            print("DEBUG: Importing ImageAugmenter...", flush=True)
            from image_augmentation import ImageAugmenter
            print("DEBUG: ImageAugmenter imported. Initializing...", flush=True)
            self.image_augmenter = ImageAugmenter(str(self.output_path))

        for category in categories:
            category_path = self.dataset_path / category
            if not category_path.exists():
                logger.warning(f"Category not found: {category}")
                continue
                
            for class_dir in [d for d in category_path.iterdir() if d.is_dir()]:
                # Find all images
                image_files = list(class_dir.glob("*.jpg"))
                curr = len(image_files)
                if curr >= min_count:
                    continue

                if curr == 0:
                    logger.warning(f"Class '{class_dir.name}' has 0 images. Skipping augmentation.")
                    continue

                needed = min_count - curr
                logger.info(f"Class '{class_dir.name}': {curr} images. Augmenting to {min_count}.")

                # Determine intensity
                deficit_ratio = needed / max(curr, 1)
                aug_type = "aggressive" if deficit_ratio > 2.0 else "balanced"

                # Run augmentation (Now returns list of Path to augmented images)
                # The Augmenter creates files in output/augmented_data/ClassName/...
                try:
                    augmented_paths = self.image_augmenter.augment_class(
                        class_name=class_dir.name,
                        image_paths=image_files,
                        target_count=min_count,
                        augmentation_type=aug_type
                    )
                except Exception as e:
                    logger.warning(f"Augmentation failed for class '{class_dir.name}': {e}")
                    continue

                # Copy augmented files (images AND txts) back to Source Dataset
                # to sustain the pipeline
                copied = 0
                idx = 0
                while copied < needed and augmented_paths:
                    src_img = Path(augmented_paths[idx % len(augmented_paths)])
                    src_txt = src_img.with_suffix('.txt')
                    
                    if not src_img.exists():
                        idx += 1
                        continue
                        
                    # Destination names
                    dest_img_name = f"aug_{(curr + copied):05d}.jpg"
                    dest_txt_name = f"aug_{(curr + copied):05d}.txt"
                    
                    dest_img_path = class_dir / dest_img_name
                    dest_txt_path = class_dir / dest_txt_name
                    
                    # Ensure uniqueness
                    suffix_count = 0
                    while dest_img_path.exists():
                        dest_img_name = f"aug_{(curr + copied):05d}_{suffix_count}.jpg"
                        dest_txt_name = f"aug_{(curr + copied):05d}_{suffix_count}.txt"
                        dest_img_path = class_dir / dest_img_name
                        dest_txt_path = class_dir / dest_txt_name
                        suffix_count += 1
                        
                    try:
                        shutil.copyfile(src_img, dest_img_path)
                        # Copy associated label if it exists
                        if src_txt.exists():
                            shutil.copyfile(src_txt, dest_txt_path)
                        copied += 1
                    except Exception as ce:
                        logger.warning(f"Failed copying {src_img} -> {dest_img_path}: {ce}")
                    idx += 1

                final_count = len(list(class_dir.glob("*.jpg")))
                logger.info(f"Class '{class_dir.name}' now has {final_count} images.")

    def _build_class_index(self, class_dirs: List[Path]) -> None:
        """
        Scan all class directories to find used Class IDs and map them to 0..N-1
        """
        logger.info("Building Class ID Map...")
        found_ids = set()
        
        # First pass: find all unique IDs used in .txt files
        # To save time, just read one file per class if we assume consistency,
        # but to be safe we scan.
        # However, typically the folder name defines the class conceptually, 
        # but the .txt file contains the numeric ID.
        # We need to map {Original_ID} -> {New_Index}
        
        # Optimization: Scan the first .txt of each class folder to determine its ID.
        # Store Mapping: {Old_ID: Class_Name}
        class_id_map = {} # Old_ID -> ClassName
        
        for d in class_dirs:
            txts = list(d.glob("*.txt"))
            if not txts:
                logger.warning(f"No labels found in {d.name}, skipping.")
                continue
            
            # Read first file to get ID
            try:
                with open(txts[0], 'r') as f:
                    content = f.read().strip()
                    if content:
                        parts = content.split()
                        old_id = int(parts[0])
                        class_id_map[old_id] = d.name
            except Exception as e:
                logger.warning(f"Error reading label in {d.name}: {e}")
        
        # Sort by Old ID to ensure deterministic order
        sorted_old_ids = sorted(class_id_map.keys())
        
        self.class_names = []
        self.id_mapping = {}
        
        for new_idx, old_id in enumerate(sorted_old_ids):
            name = class_id_map[old_id]
            self.class_names.append(name)
            self.id_mapping[old_id] = new_idx
            # logger.info(f"Mapped Old ID {old_id} ({name}) -> New ID {new_idx}")
            
        logger.info(f"Total Unique Classes: {len(self.class_names)}")

    def prepare_dataset(self, train_split: float = 0.8, val_from_train: bool = True, test_split: float = 0.1) -> None:
        """
        Prepare dataset: Augment -> Index -> Split -> Copy & Re-label
        """
        logger.info("Preparing dataset for YOLO training...")
        
        # 1. Augment
        self._ensure_min_images_per_class(min_count=300)

        # 2. Collect Valid Directories
        class_dirs = []
        for category in ["Mandatory_Traffic_Signs", "Cautionary_Traffic_Signs"]:
            category_path = self.dataset_path / category
            if category_path.exists():
                class_dirs.extend([d for d in category_path.iterdir() if d.is_dir()])
        
        if not class_dirs:
            raise FileNotFoundError("No class directories found!")

        # 3. Build Index Map
        self._build_class_index(class_dirs)
        
        # 4. Process
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class_names:
                continue # Skip if it wasn't mapped (e.g. no txt files)
            
            image_files = list(class_dir.glob("*.jpg"))
            random.shuffle(image_files)
            
            # Split
            test_idx = int(len(image_files) * test_split)
            val_idx = int(len(image_files) * (1 - train_split))
            
            test_files = image_files[:test_idx]
            val_files = image_files[test_idx:val_idx]
            train_files = image_files[val_idx:]
            
            logger.info(f"Processing {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
            
            for img in train_files:
                self._process_image(img, "train")
            for img in val_files:
                self._process_image(img, "val")
            for img in test_files:
                self._process_image(img, "test")
        
        # 5. Create YAML
        self._create_yaml_config()
        logger.info("Dataset preparation completed!")

    def _process_image(self, img_path: Path, split: str) -> None:
        """
        Copy image and transform/copy label
        """
        # Paths
        dest_img_dir = self.output_path / split / "images"
        dest_lbl_dir = self.output_path / split / "labels"
        
        dest_img = dest_img_dir / img_path.name
        dest_lbl = dest_lbl_dir / img_path.with_suffix(".txt").name
        
        src_lbl = img_path.with_suffix(".txt")
        
        # Copy Image with Retry Logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.copy2(img_path, dest_img)
                break
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.warning(f"Failed to copy {img_path} after retries. Skipping.")
                    return
            except Exception as e:
                logger.warning(f"Error copying {img_path}: {e}")
                return
        
        # Process Label
        if src_lbl.exists():
            with open(src_lbl, 'r') as f_in, open(dest_lbl, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_id = int(parts[0])
                        if old_id in self.id_mapping:
                            new_id = self.id_mapping[old_id]
                            # Write new line
                            f_out.write(f"{new_id} {' '.join(parts[1:])}\n")

    def _create_yaml_config(self) -> None:
        config = {
            'path': str(self.output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        yaml_path = self.output_path / "traffic_signs.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def train_model(self, epochs: int = 100, imgsz: int = 640, batch: int = 16, device: str = "auto", test_split: float = 0.1, workers: int = 2) -> str:
        """
        Train the YOLO model 
        """
        logger.info("Starting YOLO model training...")
        
        # Check if we need to prep dataset
        yaml_cfg = self.output_path / "traffic_signs.yaml"
        if not yaml_cfg.exists() or self._output_is_stale():
            logger.info("Preparing dataset (missing or stale output detected)...")
            # Ensure we have a validation split.
            self.prepare_dataset(train_split=0.8, val_from_train=False, test_split=test_split)
        else:
            # If exists and not stale, we still need to load class names for later usage
            with open(yaml_cfg, 'r') as f:
                data = yaml.safe_load(f)
                self.class_names = data.get('names', [])
            logger.info("Dataset verified and up-to-date.")

        dev = self.configure_device(device)
        from ultralytics import YOLO
        self.model = YOLO("yolo11s.pt")
        
        # Train
        model_name = "traffic_sign_model"
        try:
            results = self.model.train(
                data=str(yaml_cfg),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=dev,
                project=str(self.output_path),
                name=model_name,
                exist_ok=True, # Allow overwrite existing project/name
                plots=True,
                workers=workers
            )
            
            # Print Training Results
            logger.info("Training Completed. Metrics:")
            if hasattr(results, 'box'):
                print(f"Validation mAP50: {results.box.map50:.4f}")
                print(f"Validation mAP50-95: {results.box.map:.4f}")
            
            # Evaluate on Test Split
            logger.info("Evaluating on Test Split...")
            test_metrics = self.model.val(split='test', device=dev)
            print("-" * 30)
            print("TEST SET RESULTS:")
            print(f"Precision: {test_metrics.box.mp:.4f}")
            print(f"Recall:    {test_metrics.box.mr:.4f}")
            print(f"mAP50:     {test_metrics.box.map50:.4f}")
            print(f"mAP50-95:  {test_metrics.box.map:.4f}")
            print("-" * 30)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        return str(self.output_path / model_name / "weights" / "best.pt")

    def configure_device(self, device_arg: str = "auto") -> str:
        import torch
        if device_arg == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return device_arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, 0, etc)")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loading workers")
    
    args = parser.parse_args()
    
    tsr = TrafficSignRecognition()
    tsr.train_model(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers)