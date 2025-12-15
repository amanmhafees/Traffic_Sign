import os
import cv2
import glob
from pathlib import Path
import shutil
from image_augmentation import ImageAugmenter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def repair_dataset(dataset_path: str = "Dataset", target_count: int = 300):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error("Dataset not found!")
        return

    # Initialize Augmenter
    # We pass 'output' as it expects it, though we will copy augmented files back to dataset
    augmenter = ImageAugmenter(output_path="output") 
    
    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    for category in categories:
        logger.info(f"Scanning category: {category.name}")
        class_dirs = [d for d in category.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            # 1. CLEANUP
            images = list(class_dir.glob("*.jpg"))
            valid_images = []
            
            for img_path in images:
                is_corrupt = False
                try:
                    if os.path.getsize(img_path) == 0:
                        is_corrupt = True
                    else:
                        # Deep verify with cv2
                        img = cv2.imread(str(img_path))
                        if img is None:
                            is_corrupt = True
                except Exception:
                    is_corrupt = True
                
                if is_corrupt:
                    logger.warning(f"Deleting corrupt file: {img_path.name}")
                    try:
                        os.remove(img_path)
                        txt_path = img_path.with_suffix('.txt')
                        if txt_path.exists():
                            os.remove(txt_path)
                    except Exception as e:
                        logger.error(f"Failed to delete {img_path}: {e}")
                else:
                    valid_images.append(img_path)
            
            # 2. RE-AUGMENT
            current_count = len(valid_images)
            if current_count < target_count:
                needed = target_count - current_count
                logger.info(f"Class '{class_dir.name}': {current_count} valid images. Augmenting {needed} more...")
                
                if current_count == 0:
                    logger.error(f"Class '{class_dir.name}' has 0 valid images! Cannot augment.")
                    continue

                # Run augmentation
                # Note: valid_images are paths in the SOURCE dataset
                # Augmenter creates files in output/augmented_data/...
                try:
                    augmented_paths = augmenter.augment_class(
                        class_name=class_dir.name,
                        image_paths=valid_images,
                        target_count=target_count,
                        augmentation_type="balanced"
                    )
                    
                    # Copy back to source
                    copied = 0
                    idx = 0
                    while copied < needed and augmented_paths:
                        src_img = augmented_paths[idx % len(augmented_paths)]
                        src_txt = src_img.with_suffix('.txt')
                        
                        if not src_img.exists():
                            idx += 1
                            continue
                            
                        # Unique name definition
                        # Try to avoid naming collisions if we are re-augmenting
                        dest_img_name = f"repaired_aug_{copied:05d}_{idx}.jpg"
                        dest_txt_name = f"repaired_aug_{copied:05d}_{idx}.txt"
                        
                        dest_img_path = class_dir / dest_img_name
                        dest_txt_path = class_dir / dest_txt_name
                        
                        try:
                            shutil.copyfile(src_img, dest_img_path)
                            if src_txt.exists():
                                shutil.copyfile(src_txt, dest_txt_path)
                            copied += 1
                        except Exception as ce:
                            logger.warning(f"Copy failed: {ce}")
                        idx += 1
                        
                    logger.info(f"Refilled '{class_dir.name}' to {target_count}.")
                    
                except Exception as e:
                    logger.error(f"Augmentation failed for {class_dir.name}: {e}")
            else:
                logger.info(f"Class '{class_dir.name}' OK ({current_count} images).")

if __name__ == "__main__":
    repair_dataset()
