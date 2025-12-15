
import logging
from image_augmentation import ImageAugmenter
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def test_repair():
    with open("log.txt", "w") as log:
        log.write("Script started\n")
    
    augmenter = ImageAugmenter(output_path="output")
    class_name = "CROSS_ROAD"
    dataset_path = Path(f"Dataset/Camel/Cautionary_Traffic_Signs/{class_name}") 
    # Wait, need correct path
    
    # helper to find path
    root = Path("Dataset")
    found = list(root.rglob(class_name))
    if not found:
        print(f"Could not find {class_name}")
        return
    
    class_dir = found[0]
    print(f"Testing repair on: {class_dir}")
    
    images = list(class_dir.glob("*.jpg"))
    print(f"Initial count: {len(images)}")
    
    try:
        # augment
        new_paths = augmenter.augment_class(class_name, images, 300, "balanced")
        print(f"Augmenter returned {len(new_paths)} paths")
        
        # Copy back
        import shutil
        for src in new_paths:
            try:
                dest = class_dir / src.name
                if not dest.exists():
                    print(f"Copying {src.name} to {dest}")
                    shutil.copy2(src, dest)
                    # Copy txt
                    src_txt = src.with_suffix(".txt")
                    if src_txt.exists():
                        shutil.copy2(src_txt, dest.with_suffix(".txt"))
            except Exception as e:
                print(f"Failed to copy {src.name}: {e}")

        count = len(list(class_dir.glob('*.jpg')))
        with open("debug_result.txt", "w") as f:
            f.write(f"Final valid count: {count}")
        print(f"Final valid count: {count}")
    except Exception:
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        traceback.print_exc()
    
if __name__ == "__main__":
    test_repair()
