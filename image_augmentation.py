#!/usr/bin/env python3
"""
Advanced Image Augmentation Module for Traffic Sign Recognition
Includes various augmentation techniques to handle class imbalance
"""

import cv2
import numpy as np
from pathlib import Path
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Optional
import logging
from PIL import Image, ImageEnhance, ImageFilter
import datetime

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
        """Setup different augmentation pipelines for different scenarios (ReplayCompose enabled)."""
        # Basic augmentation pipeline
        self.basic_pipeline = A.ReplayCompose([
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ])
        
        # Aggressive augmentation for minority classes
        self.aggressive_pipeline = A.ReplayCompose([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=30,
                sat_shift_limit=40,
                val_shift_limit=30,
                p=0.7
            ),
            A.GaussNoise(var_limit=(10.0, 80.0), p=0.5),
            A.Blur(blur_limit=5, p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        ])
        
        # Weather and lighting conditions
        self.weather_pipeline = A.ReplayCompose([
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=0.3),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
        ])
        
        # Perspective and geometric transformations
        self.geometric_pipeline = A.ReplayCompose([
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.5
            ),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=0.1,
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.5
            ),
        ])
        
        # Color and lighting variations
        self.color_pipeline = A.ReplayCompose([
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=40,
                sat_shift_limit=50,
                val_shift_limit=40,
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.ToGray(p=0.2),
            A.ChannelShuffle(p=0.2),
        ])
        
        # Noise and quality degradation
        self.noise_pipeline = A.ReplayCompose([
            A.GaussNoise(var_limit=(10.0, 100.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
            A.Blur(blur_limit=7, p=0.4),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),
        ])
    
    def augment_class(self, class_name: str, image_paths: List[Path], 
                     target_count: int, augmentation_type: str = "balanced") -> List[Path]:
        """
        Augment images for a specific class to reach target count
        
        Args:
            class_name: Name of the class
            image_paths: List of original image paths
            target_count: Target number of images
            augmentation_type: Type of augmentation ('basic', 'aggressive', 'balanced')
            
        Returns:
            List of augmented image paths
        """
        logger.info(f"Augmenting class {class_name} from {len(image_paths)} to {target_count} images")
        
        # Create class directory
        class_dir = self.aug_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        augmented_paths = []
        current_count = len(image_paths)
        
        # Copy original images
        for i, img_path in enumerate(image_paths):
            new_path = class_dir / f"original_{i:04d}.jpg"
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(new_path), img)
                augmented_paths.append(new_path)
                self._log_preprocess_event(
                    mode="copy",
                    class_name=class_name,
                    pipeline_name="original",
                    original_path=img_path,
                    output_path=new_path,
                    applied_transforms=["None"]
                )
        
        # Calculate how many augmentations we need
        needed_augmentations = target_count - current_count
        
        if needed_augmentations <= 0:
            return augmented_paths
        
        # Select augmentation pipeline
        if augmentation_type == "aggressive":
            pipeline = self.aggressive_pipeline
            pipeline_name = "aggressive"
        elif augmentation_type == "balanced":
            pipeline = self._get_balanced_pipeline()
            pipeline_name = "balanced"
        else:
            pipeline = self.basic_pipeline
            pipeline_name = "basic"
        
        # Generate augmentations
        aug_count = 0
        while aug_count < needed_augmentations:
            # Randomly select an original image
            original_img_path = random.choice(image_paths)
            img = cv2.imread(str(original_img_path))
            
            if img is None:
                continue
            
            # Apply augmentation
            try:
                result = pipeline(image=img)
                augmented = result['image']
                replay = result.get('replay', {})
                applied = self._extract_applied_transforms(replay)
                aug_path = class_dir / f"aug_{aug_count:04d}.jpg"
                cv2.imwrite(str(aug_path), augmented)
                augmented_paths.append(aug_path)
                aug_count += 1
                self._log_preprocess_event(
                    mode="augment",
                    class_name=class_name,
                    pipeline_name=pipeline_name,
                    original_path=original_img_path,
                    output_path=aug_path,
                    applied_transforms=applied
                )
                
            except Exception as e:
                logger.warning(f"Error augmenting image {original_img_path}: {e}")
                continue
        
        logger.info(f"Generated {aug_count} augmented images for class {class_name}")
        return augmented_paths
    
    def _get_balanced_pipeline(self) -> A.Compose:
        """Get a balanced augmentation pipeline that combines multiple techniques"""
        # Recreate on demand so ReplayCompose replays are independent
        return A.ReplayCompose([
            A.OneOf([
                A.Compose(self.basic_pipeline.transforms),
                A.Compose(self.weather_pipeline.transforms),
                A.Compose(self.geometric_pipeline.transforms),
                A.Compose(self.color_pipeline.transforms),
                A.Compose(self.noise_pipeline.transforms),
            ], p=1.0)
        ])
    
    def create_augmented_dataset(self, class_distribution: Dict, 
                                target_balance: str = "mean") -> Dict:
        """
        Create an augmented dataset to balance class distribution
        
        Args:
            class_distribution: Dictionary with class distribution data
            target_balance: Target balance strategy ('mean', 'max', 'custom')
            
        Returns:
            Dictionary with augmented dataset information
        """
        logger.info("Creating augmented dataset for class balance...")
        
        # Calculate target count
        counts = [data['count'] for data in class_distribution.values()]
        
        if target_balance == "mean":
            target_count = int(np.mean(counts))
        elif target_balance == "max":
            target_count = max(counts)
        else:
            target_count = int(target_balance)
        
        logger.info(f"Target count per class: {target_count}")
        
        augmented_dataset = {}
        
        for class_name, class_data in class_distribution.items():
            current_count = class_data['count']
            image_paths = class_data['images']
            
            if current_count < target_count:
                # Determine augmentation strategy based on deficit
                deficit_ratio = (target_count - current_count) / current_count
                
                if deficit_ratio > 2.0:  # Need more than 2x augmentation
                    aug_type = "aggressive"
                else:
                    aug_type = "balanced"
                
                # Augment the class
                augmented_paths = self.augment_class(
                    class_name, image_paths, target_count, aug_type
                )
                
                augmented_dataset[class_name] = {
                    'original_count': current_count,
                    'augmented_count': len(augmented_paths),
                    'augmentation_type': aug_type,
                    'paths': augmented_paths
                }
            else:
                # Class already has enough samples
                augmented_dataset[class_name] = {
                    'original_count': current_count,
                    'augmented_count': current_count,
                    'augmentation_type': 'none',
                    'paths': image_paths
                }
        
        # Save augmentation report
        self._save_augmentation_report(augmented_dataset, target_count)
        
        return augmented_dataset
    
    def _save_augmentation_report(self, augmented_dataset: Dict, target_count: int):
        """Save augmentation report"""
        report_path = self.aug_dir / "augmentation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Image Augmentation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Target count per class: {target_count}\n\n")
            
            for class_name, data in augmented_dataset.items():
                f.write(f"Class: {class_name}\n")
                f.write(f"  Original count: {data['original_count']}\n")
                f.write(f"  Augmented count: {data['augmented_count']}\n")
                f.write(f"  Augmentation type: {data['augmentation_type']}\n")
                f.write(f"  Increase: {data['augmented_count'] - data['original_count']}\n\n")
        
        logger.info(f"Augmentation report saved to {report_path}")
    
    def visualize_augmentations(self, original_img_path: Path, class_name: str, 
                              num_samples: int = 8) -> None:
        """
        Visualize different augmentation techniques on a sample image
        
        Args:
            original_img_path: Path to original image
            class_name: Name of the class
            num_samples: Number of augmented samples to generate
        """
        import matplotlib.pyplot as plt
        
        img = cv2.imread(str(original_img_path))
        if img is None:
            logger.error(f"Could not load image: {original_img_path}")
            return
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create subplot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Show original image
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Generate and show augmented images
        pipelines = [
            ("Basic", self.basic_pipeline),
            ("Aggressive", self.aggressive_pipeline),
            ("Weather", self.weather_pipeline),
            ("Geometric", self.geometric_pipeline),
            ("Color", self.color_pipeline),
            ("Noise", self.noise_pipeline),
            ("Balanced", self._get_balanced_pipeline()),
        ]
        
        for i, (name, pipeline) in enumerate(pipelines[:7]):
            try:
                result = pipeline(image=img)
                augmented = result['image']
                replay = result.get('replay', {})
                applied = self._extract_applied_transforms(replay)
                augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
                axes[i+1].imshow(augmented_rgb)
                axes[i+1].set_title(f"{name} Augmentation")
                axes[i+1].axis('off')
                # Log visualization augmentation
                vis_out = self.aug_dir / f"vis_{class_name}_{name.lower()}_{i}.jpg"
                cv2.imwrite(str(vis_out), augmented)
                self._log_preprocess_event(
                    mode="visualize",
                    class_name=class_name,
                    pipeline_name=name.lower(),
                    original_path=original_img_path,
                    output_path=vis_out,
                    applied_transforms=applied
                )
            except Exception as e:
                logger.warning(f"Error in {name} augmentation: {e}")
                axes[i+1].text(0.5, 0.5, f"Error in {name}",
                               ha='center', va='center', transform=axes[i+1].transAxes)
                axes[i+1].axis('off')
        
        plt.suptitle(f"Augmentation Examples for {class_name}", fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        vis_path = self.aug_dir / f"augmentation_examples_{class_name}.png"
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Augmentation visualization saved to {vis_path}")

def main():
    """Main function to test augmentation"""
    augmenter = ImageAugmenter()
    
    # Test with a sample image
    sample_path = Path("Dataset/Mandatory_Traffic_Signs/STOP/02000.jpg")
    if sample_path.exists():
        augmenter.visualize_augmentations(sample_path, "STOP")
        print("Augmentation visualization created!")
    else:
        print("Sample image not found for visualization")

if __name__ == "__main__":
    main()