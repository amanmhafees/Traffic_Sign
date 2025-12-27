import argparse
import json
from pathlib import Path
from typing import List, Dict
import random


def load_yaml_names(yaml_path: Path) -> List[str]:
    import yaml
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        # sort by int key
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(n) for n in names]
    return []


def validate(cnn_root: Path, yaml_path: Path, sample_per_class: int = 5) -> Dict:
    expected = set(load_yaml_names(yaml_path))
    found = set([d.name for d in cnn_root.iterdir() if d.is_dir()]) if cnn_root.exists() else set()

    missing = sorted(list(expected - found))
    extra = sorted(list(found - expected))

    samples = {}
    for cls in sorted(list(found & expected)):
        cls_dir = cnn_root / cls
        imgs = sorted([p for p in cls_dir.glob("*.jpg")])
        take = min(sample_per_class, len(imgs))
        if take > 0:
            samples[cls] = [str(p) for p in random.sample(imgs, take)]
        else:
            samples[cls] = []

    summary = {
        "expected_classes": sorted(list(expected)),
        "found_classes": sorted(list(found)),
        "missing_classes": missing,
        "extra_classes": extra,
        "samples": samples,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Validate CNN dataset folder integrity vs YOLO YAML names")
    ap.add_argument("--cnn-dataset", type=str, default="cnn_dataset")
    ap.add_argument("--yaml", type=str, default="output/traffic_signs.yaml")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    cnn_root = Path(args.cnn_dataset)
    yaml_path = Path(args.yaml)
    result = validate(cnn_root, yaml_path, args.samples)
    out_path = cnn_root / "_validation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Validation saved to {out_path}")
    if result["missing_classes"]:
        print(f"Missing classes: {result['missing_classes']}")
    if result["extra_classes"]:
        print(f"Extra classes: {result['extra_classes']}")


if __name__ == "__main__":
    main()
