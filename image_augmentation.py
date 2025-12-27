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
        """Setup SAFE augmentation pipeline (strict) with YOLO bbox support"""
        import albumentations as A
        
        # Strict bbox params: keep boxes even if slightly altered; we'll validate manually
        self.bbox_params = A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.0,
            min_area=0.0
        )

        # SAFE pipeline per rules: ±5° rotation, mild brightness/contrast, light noise/blur
        self.safe_pipeline = A.Compose([
            A.Rotate(limit=5, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.4),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
        ], bbox_params=self.bbox_params)

    def _bboxes_valid(self, bboxes: List[List[float]]) -> bool:
        """Validate YOLO bboxes: >=5% size and fully within image bounds."""
        if not bboxes:
            return False
        for (xc, yc, w, h) in [b[:4] for b in bboxes]:
            # size threshold
            if w < 0.05 or h < 0.05:
                return False
            # fully inside image (normalized coords)
            if (xc - w / 2) < 0.0 or (xc + w / 2) > 1.0:
                return False
            if (yc - h / 2) < 0.0 or (yc + h / 2) > 1.0:
                return False
        return True
    
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
                      target_count: Optional[int] = None) -> List[Path]:
        """
        Augment images for a specific class using SAFE transforms.
        Limits augmentation to at most 1.5x the original class size.
        Preserves YOLO bbox integrity and rejects invalid augmentations.
        """
        orig_size = len(image_paths)
        max_allowed = int(orig_size * 1.5)
        target_count = max_allowed if (target_count is None or target_count > max_allowed) else target_count
        logger.info(f"Augmenting class {class_name}: original={orig_size}, target<= {target_count} (max 1.5x)")
        
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
        
        pipeline = self.safe_pipeline
        pipeline_name = "safe"
        
        # Generate augmentations
        aug_count = 0
        if not original_data:
            logger.warning(f"No valid original data for {class_name}, cannot augment.")
            return augmented_paths

        print(f"[{class_name}] Augmenting {needed_augmentations} images (SAFE)...")
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
                
                current_pipeline = pipeline

                try:
                    # AUGMENT
                    # Pass bboxes in YOLO format. 
                    # Note: Albumentations will filter boxes that fall outside or become too small.
                    result = current_pipeline(image=img, bboxes=yolo_bboxes, class_labels=class_labels)
                    
                    aug_img = result['image']
                    aug_bboxes = result['bboxes']
                    aug_labels = result['class_labels']

                    # Integrity checks: same count, valid boxes
                    if len(aug_bboxes) != len(yolo_bboxes) or len(aug_bboxes) != len(aug_labels):
                        continue
                    # Validate bbox sizes and bounds (normalized)
                    if not self._bboxes_valid([[*b, aug_labels[i]] for i, b in enumerate(aug_bboxes)]):
                        continue
                    
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
                    
                    # Log (static list of safe transforms)
                    applied = ["Rotate(±5)", "RandomBrightnessContrast(≤10%)", "GaussNoise(light)", "Blur(≤3)"]
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