import pathlib
import sys
import cv2
import os

OUTPUT_FILE = "dataset_report.txt"
root = pathlib.Path('Dataset')

with open(OUTPUT_FILE, 'w') as log:
    def log_print(msg):
        print(msg)
        log.write(msg + "\n")

    if not root.exists():
        log_print("Dataset folder not found!")
        sys.exit(1)

    log_print(f"Scanning {root.absolute()}...")
    categories = [d for d in root.iterdir() if d.is_dir()]

    total_classes = 0
    classes_under_300 = []
    corrupt_images = []

    for cat in categories:
        log_print(f"\n--- {cat.name} ---")
        classes = [d for d in cat.iterdir() if d.is_dir()]
        for cls in classes:
            images = list(cls.glob("*.jpg"))
            count = len(images)
            log_print(f"{cls.name}: {count}")
            total_classes += 1
            
            if count < 300:
                classes_under_300.append(f"{cat.name}/{cls.name} ({count})")
            
            # Check for corruption (sample check or full check?)
            # Let's check all since validaton failed on so many
            for img_path in images:
                try:
                    # Check file size
                    if os.path.getsize(img_path) == 0:
                        corrupt_images.append(str(img_path))
                        continue
                    
                    # Try reading with PIL (fast header check) or cv2
                    # cv2 is what YOLO uses
                    img = cv2.imread(str(img_path))
                    if img is None:
                       corrupt_images.append(str(img_path))
                except Exception:
                    corrupt_images.append(str(img_path))

    log_print("\n=== SUMMARY ===")
    log_print(f"Total Classes: {total_classes}")
    
    if classes_under_300:
        log_print("\n[WARNING] Classes UNDER 300 images:")
        for c in classes_under_300:
            log_print(f"  - {c}")
    else:
        log_print("\n[OK] ALL CLASSES HAVE >= 300 IMAGES.")

    if corrupt_images:
        log_print(f"\n[CRITICAL] Found {len(corrupt_images)} CORRUPT images!")
        log_print("First 10 corrupt files:")
        for c in corrupt_images[:10]:
            log_print(f"  - {c}")
        # Save full list
        with open("corrupt_images.txt", "w") as cel:
            for c in corrupt_images:
                cel.write(c + "\n")
    else:
        log_print("\n[OK] No corrupt images found.")
