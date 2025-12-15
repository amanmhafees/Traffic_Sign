
import os
from pathlib import Path

def scan_dataset():
    dataset_path = Path("Dataset")
    categories = ["Cautionary_Traffic_Signs", "Mandatory_Traffic_Signs"]
    
    class_map = {}
    
    for category in categories:
        cat_path = dataset_path / category
        if not cat_path.exists():
            print(f"Skipping {category} (not found)")
            continue
            
        print(f"Scanning {category}...")
        for class_dir in cat_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            # Find a txt file
            txt_files = list(class_dir.glob("*.txt"))
            if not txt_files:
                print(f"  [WARNING] No txt files in {class_dir.name}")
                continue
                
            # Read first file
            with open(txt_files[0], 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"  [WARNING] Empty file {txt_files[0].name}")
                    continue
                parts = content.split()
                if not parts:
                    continue
                class_id = parts[0]
                
            if class_dir.name in class_map:
                if class_map[class_dir.name] != class_id:
                     print(f"  [ERROR] Conflict for {class_dir.name}: {class_map[class_dir.name]} vs {class_id}")
            else:
                class_map[class_dir.name] = class_id
                # print(f"  {class_dir.name} -> {class_id}")

    print("\n--- Summary ---")
    sorted_map = sorted(class_map.items(), key=lambda x: int(x[1]))
    for name, cid in sorted_map:
        print(f"{name}: {cid}")
        
    print(f"\nTotal Classes: {len(class_map)}")

if __name__ == "__main__":
    scan_dataset()
