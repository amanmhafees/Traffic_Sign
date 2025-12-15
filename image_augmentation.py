#!/usr/bin/env python3
"""
Advanced Image Augmentation Module for Traffic Sign Recognition
Includes various augmentation techniques to handle class imbalance
Updates: Supports Bounding Boxes (YOLO format)
"""

import cv2
import numpy as np
from pathlib import Path
import random
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Optional
import logging
from PIL import Image, ImageEnhance, ImageFilter
import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageAugmenter:
    def __init__(self, output_path: str = "output"):
        """
        Initialize the Image Augmenter
        
        Args:
            output_path: Path for output files
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Create augmentation output directory
        self.aug_dir = self.output_path / "augmented_data"
        self.aug_dir.mkdir(exist_ok=True)
        
        # Initialize augmentation pipelines
        self._setup_augmentation_pipelines()
        
        # Create directory and log file for preprocessing/augmentation records
        self._init_logging()
        
    def _init_logging(self):
        """Create directory and log file for preprocessing/augmentation records."""
        self.log_dir = self.output_path / "augmentation_logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.log_dir / "preprocessing_log.txt"
        if not self.log_file.exists():
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("Preprocessing / Augmentation Log\n")
                f.write("=" * 60 + "\n")
                f.write("timestamp\tmode\tclass\tpipeline\toriginal_path\toutput_path\tapplied_transforms\n")

    def _log_preprocess_event(self,
                              mode: str,
                              class_name: str,
                              pipeline_name: str,
                              original_path: Path,
                              output_path: Path,
                              applied_transforms: List[str]):
        """Append a single preprocessing record line."""
        ts = datetime.datetime.utcnow().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{mode}\t{class_name}\t{pipeline_name}\t"
                    f"{original_path}\t{output_path}\t{';'.join(applied_transforms)}\n")

    def _extract_applied_transforms(self, replay_dict: dict) -> List[str]:
        """Extract only transforms that were actually applied from a ReplayCompose replay."""
        applied = []
        try:
            for t in replay_dict.get("transforms", []):
                if t.get("applied"):
                    name = t.get("__class_fullname__", t.get("transform", "Unknown"))
                    # Shorten class path if present
                    applied.append(name.split(".")[-1])
        except Exception:
            pass
        return applied or ["None"]

    def _setup_augmentation_pipelines(self):
        """Setup different augmentation pipelines for different scenarios with BBox support"""
        import albumentations as A
        
        # Common Bbox params: YOLO format, ensure labels are passed
        bbox_params = A.BboxParams(
            format='yolo', 
            label_fields=['class_labels'], 
            min_visibility=0.1, 
            min_area=10.0
        )

        # Basic augmentation pipeline
        self.basic_pipeline = A.Compose([
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussNoise(p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ], bbox_params=bbox_params)

        # Aggressive augmentation for minority classes
        self.aggressive_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.7),
            A.GaussNoise(p=0.5),
            A.Blur(blur_limit=5, p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            # GripDistortion/ElasticTransform can be risky for small bboxes, but keeping with conservative limits
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        ], bbox_params=bbox_params)

        # Weather and lighting conditions
        self.weather_pipeline = A.Compose([
            A.RandomRain(p=0.3),
            A.RandomShadow(p=0.3),
            A.RandomSunFlare(p=0.2),
            A.RandomFog(p=0.2),
        ], bbox_params=bbox_params)

        # Perspective and geometric transformations
        self.geometric_pipeline = A.Compose([
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-15, 15), shear=(-5, 5), p=0.5),
        ], bbox_params=bbox_params)

        # Color and lighting variations
        self.color_pipeline = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=50, val_shift_limit=40, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.ToGray(p=0.2),
            A.ChannelShuffle(p=0.2),
        ], bbox_params=bbox_params)

        # Noise and quality degradation
        self.noise_pipeline = A.Compose([
            A.GaussNoise(p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
            A.Blur(blur_limit=7, p=0.4),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),
        ], bbox_params=bbox_params)
    
    def _read_bboxes(self, img_path: Path) -> List[List[float]]:
        """
        Read YOLO format bounding boxes from corresponding .txt file
        Returns: List of [x_center, y_center, width, height, class_id]
        """
        txt_path = img_path.with_suffix('.txt')
        bboxes = []
        if txt_path.exists():
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO format: class_id x_center y_center width height
                            class_id = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                            bboxes.append([xc, yc, w, h, class_id])
            except Exception as e:
                logger.warning(f"Error reading labels for {img_path}: {e}")
        return bboxes

    def augment_class(self, class_name: str, image_paths: List[Path], 
                     target_count: int, augmentation_type: str = "balanced") -> List[Path]:
        """
        Augment images for a specific class to reach target count
        HANDLES BOUNDING BOXES (YOLO FORMAT)
        """
        logger.info(f"Augmenting class {class_name} from {len(image_paths)} to {target_count} images")
        
        # Create class directory
        class_dir = self.aug_dir / class_name
        class_dir.mkdir(exist_ok=True, parents=True)
        
        augmented_paths = []
        
        # Cache original data
        original_data = [] # List of tuples (img, bboxes, original_path)
        
        # Copy original images and labels with progress
        print(f"[{class_name}] Copying original images...")
        for i, img_path in enumerate(tqdm(image_paths, desc=f"Copying {class_name}", unit="img")):
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Read bboxes
                bboxes_with_cls = self._read_bboxes(img_path)
                
                # Prepare new paths
                new_img_name = f"original_{i:04d}.jpg"
                new_img_path = class_dir / new_img_name
                new_txt_path = class_dir / f"original_{i:04d}.txt"
                
                # Save image
                cv2.imwrite(str(new_img_path), img)
                
                # Save labels
                if bboxes_with_cls:
                    with open(new_txt_path, 'w') as f:
                        for box in bboxes_with_cls:
                            # box is [xc, yc, w, h, class_id]
                            # write as: class_id xc yc w h
                            f.write(f"{int(box[4])} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                
                augmented_paths.append(new_img_path)
                original_data.append((img, bboxes_with_cls, img_path))
                
                self._log_preprocess_event(
                    mode="copy",
                    class_name=class_name,
                    pipeline_name="original",
                    original_path=img_path,
                    output_path=new_img_path,
                    applied_transforms=["None"]
                )
        
        current_count = len(augmented_paths)
        needed_augmentations = target_count - current_count
        
        if needed_augmentations <= 0:
            return augmented_paths
        
        # Select augmentation pipeline
        if augmentation_type == "aggressive":
            pipeline = self.aggressive_pipeline
            pipeline_name = "aggressive"
        elif augmentation_type == "balanced":
            # For simplicity in 'balanced', just pick random sub-pipelines
            pipeline_name = "balanced"
        else:
            pipeline = self.basic_pipeline
            pipeline_name = "basic"
        
        # Generate augmentations
        aug_count = 0
        if not original_data:
            logger.warning(f"No valid original data for {class_name}, cannot augment.")
            return augmented_paths

        print(f"[{class_name}] Augmenting {needed_augmentations} images...")
        pbar = tqdm(total=needed_augmentations, desc=f"Augmenting {class_name}", unit="aug")
        while aug_count < needed_augmentations:
                # Randomly select sample
                img, bboxes_with_cls, original_path = random.choice(original_data)
                
                # Separate bboxes and class labels for Albumentations
                # Albumentations expects [x_center, y_center, width, height, class_label] 
                # OR seperate labels input if label_fields is used. 
                # We configured `label_fields=['class_labels']`.
                # So we pass bboxes as [xc, yc, w, h] and class_labels as [class_id]
                
                yolo_bboxes = [b[:4] for b in bboxes_with_cls]
                class_labels = [b[4] for b in bboxes_with_cls]
                
                # Choose pipeline (if balanced, rotate through some)
                if pipeline_name == "balanced":
                    current_pipeline = random.choice([
                        self.basic_pipeline, 
                        self.weather_pipeline, 
                        self.geometric_pipeline, 
                        self.color_pipeline
                    ])
                else:
                    current_pipeline = pipeline

                try:
                    # AUGMENT
                    # Pass bboxes in YOLO format. 
                    # Note: Albumentations will filter boxes that fall outside or become too small.
                    result = current_pipeline(image=img, bboxes=yolo_bboxes, class_labels=class_labels)
                    
                    aug_img = result['image']
                    aug_bboxes = result['bboxes']
                    aug_labels = result['class_labels']
                    
                    # Save
                    aug_img_name = f"aug_{aug_count:04d}.jpg"
                    aug_img_path = class_dir / aug_img_name
                    aug_txt_path = class_dir / f"aug_{aug_count:04d}.txt"
                    
                    cv2.imwrite(str(aug_img_path), aug_img)
                    
                    with open(aug_txt_path, 'w') as f:
                        for bbox, cls_id in zip(aug_bboxes, aug_labels):
                            # bbox is (xc, yc, w, h)
                            f.write(f"{int(cls_id)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                            
                    augmented_paths.append(aug_img_path)
                    
                    # Log
                    replay = result.get('replay', {})
                    applied = self._extract_applied_transforms(replay)
                    self._log_preprocess_event(
                        mode="augment",
                        class_name=class_name,
                        pipeline_name=pipeline_name,
                        original_path=original_path,
                        output_path=aug_img_path,
                        applied_transforms=applied
                    )
                    
                    aug_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    # logger.warning(f"Augmentation failed for {original_path.name}: {e}")
                    # Often fails if bboxes are invalid or image is empty
                    continue
        pbar.close()

        logger.info(f"Generated {aug_count} augmented images for class {class_name}")
        return augmented_paths

    def main(self):
        """Test augmentation with visualization"""
        # (Simplified main execution for testing)
        pass

if __name__ == "__main__":
    ImageAugmenter().main()