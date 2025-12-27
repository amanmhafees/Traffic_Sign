import argparse
import shutil
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import json


def yolo_to_xyxy(
    cx: float, cy: float, bw: float, bh: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized bbox to integer pixel xyxy.

    Returns clipped coordinates, not guaranteed valid.
    """
    x1 = int((cx - bw / 2.0) * img_w)
    y1 = int((cy - bh / 2.0) * img_h)
    x2 = int((cx + bw / 2.0) * img_w)
    y2 = int((cy + bh / 2.0) * img_h)
    return x1, y1, x2, y2


def load_id_to_name_from_yaml(yaml_path: Path) -> Optional[dict]:
    try:
        import yaml  # lazy
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict):
            # keys may be strings
            return {int(k): str(v) for k, v in names.items()}
    except Exception:
        return None
    return None


def process_image(
    img_path: Path,
    lbl_path: Path,
    out_root: Path,
    id_to_name: Optional[dict],
    min_side: int,
    min_area_ratio: float,
) -> int:
    """Process one image: read labels, crop qualifying ROIs, save to class folders.

    Returns number of crops saved.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]

    saved = 0
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except Exception:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)

            # Quality filters: reject if out of bounds partially
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue
            rw, rh = x2 - x1, y2 - y1
            if rw < min_side or rh < min_side:
                continue
            if (rw * rh) < (min_area_ratio * w * h):
                continue

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Resolve class name (MANDATORY). Abort if mapping missing.
            if not id_to_name or cid not in id_to_name:
                # Skip if we cannot resolve class name from YAML.
                # This enforces dataset integrity: no guessing or fallback.
                continue
            cname = id_to_name[cid]

            class_dir = out_root / cname
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save 224x224 resize
            roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
            out_name = f"{img_path.stem}_x{x1}_y{y1}_w{rw}_h{rh}.jpg"
            cv2.imwrite(str(class_dir / out_name), roi_resized)
            saved += 1

    return saved


def _iterate_yolo_output_images(yolo_output_root: Path) -> List[Path]:
    imgs: List[Path] = []
    for split in ["train", "val", "test"]:
        split_img_dir = yolo_output_root / split / "images"
        if split_img_dir.exists():
            imgs.extend(list(split_img_dir.glob("*.jpg")))
    return imgs


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO-format dataset (images + .txt) into cropped CNN dataset."
    )
    parser.add_argument(
        "--yolo-dataset",
        type=str,
        default="Dataset",
        help="Root of original YOLO dataset (used when --source=raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cnn_dataset",
        help="Output root for cropped CNN dataset",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default="output/traffic_signs.yaml",
        help="Optional YAML with class names (id->name)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["yolo-output", "raw"],
        default="yolo-output",
        help="Source of images/labels: 'yolo-output' uses output/train|val|test, 'raw' uses original Dataset tree",
    )
    parser.add_argument(
        "--yolo-output-root",
        type=str,
        default="output",
        help="Root directory containing YOLO output splits (train/val/test) when --source=yolo-output",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing cnn_dataset before regenerating",
    )
    parser.add_argument(
        "--strict-raw-check",
        action="store_true",
        help="When --source=raw, only accept samples where parent folder name equals YAML-mapped class name",
    )
    parser.add_argument("--min-side", type=int, default=32, help="Min width/height in pixels")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.01,
        help="Min ROI area as fraction of image area",
    )

    args = parser.parse_args()

    ds_root = Path(args["yolo_dataset"]) if isinstance(args, dict) else Path(args.yolo_dataset)
    out_root = Path(args["output"]) if isinstance(args, dict) else Path(args.output)
    yaml_path = Path(args["yaml"]) if isinstance(args, dict) else Path(args.yaml)
    source = args["source"] if isinstance(args, dict) else args.source
    yolo_out_root = Path(args["yolo_output_root"]) if isinstance(args, dict) else Path(args.yolo_output_root)
    if (args["clean"] if isinstance(args, dict) else args.clean) and out_root.exists():
        shutil.rmtree(out_root)

    id_to_name = load_id_to_name_from_yaml(yaml_path) if yaml_path.exists() else None

    total_images = 0
    total_crops = 0

    if source == "yolo-output":
        image_paths = _iterate_yolo_output_images(yolo_out_root)
        for img_path in image_paths:
            lbl_path = (img_path.parent.parent / "labels" / img_path.name).with_suffix(".txt")
            if not lbl_path.exists():
                continue
            total_images += 1
            saved = process_image(
                img_path,
                lbl_path,
                out_root,
                id_to_name,
                args.min_side if not isinstance(args, dict) else args["min_side"],
                args.min_area_ratio if not isinstance(args, dict) else args["min_area_ratio"],
            )
            total_crops += saved
    else:
        mismatches = []
        for category in ["Mandatory_Traffic_Signs", "Cautionary_Traffic_Signs"]:
            cat_dir = ds_root / category
            if not cat_dir.exists():
                continue
            for class_dir in [d for d in cat_dir.iterdir() if d.is_dir()]:
                for img_path in class_dir.glob("*.jpg"):
                    lbl_path = img_path.with_suffix(".txt")
                    if not lbl_path.exists():
                        continue
                    total_images += 1
                    # Optional strict check: ensure parent folder name matches YAML mapping
                    strict = args["strict_raw_check"] if isinstance(args, dict) else args.strict_raw_check
                    if strict:
                        try:
                            with open(lbl_path, "r", encoding="utf-8") as f:
                                line = f.readline().strip()
                            parts = line.split()
                            cid = int(float(parts[0])) if parts else None
                            mapped = id_to_name.get(cid) if (id_to_name and cid is not None) else None
                            if mapped and mapped != class_dir.name:
                                mismatches.append({
                                    "image": str(img_path),
                                    "label_id": cid,
                                    "yaml_class": mapped,
                                    "folder": class_dir.name,
                                })
                                # Skip this sample to avoid contaminating dataset
                                continue
                        except Exception:
                            # If label parse fails under strict, skip
                            continue

                    saved = process_image(
                        img_path,
                        lbl_path,
                        out_root,
                        id_to_name,
                        args.min_side if not isinstance(args, dict) else args["min_side"],
                        args.min_area_ratio if not isinstance(args, dict) else args["min_area_ratio"],
                    )
                    total_crops += saved
        if mismatches:
            (out_root / "_raw_mismatches.json").write_text(json.dumps(mismatches, indent=2), encoding="utf-8")

    meta = {
        "source_dataset": str(ds_root.resolve()),
        "output_dataset": str(out_root.resolve()),
        "total_images_scanned": total_images,
        "total_crops_saved": total_crops,
        "min_side": args.min_side if not isinstance(args, dict) else args["min_side"],
        "min_area_ratio": args.min_area_ratio if not isinstance(args, dict) else args["min_area_ratio"],
    }
    (out_root / "_meta.json").write_text(json.dumps(meta, indent=2))

    # Coverage report against YAML names
    if id_to_name and out_root.exists():
        expected = set(id_to_name.values())
        present = set([d.name for d in out_root.iterdir() if d.is_dir()])
        missing = sorted(list(expected - present))
        extra = sorted(list(present - expected))
        # Per-class counts
        counts = {}
        for cls in present:
            counts[cls] = len(list((out_root / cls).glob("*.jpg")))
        coverage = {
            "expected_total": len(expected),
            "present_total": len(present),
            "missing_classes": missing,
            "extra_classes": extra,
            "counts": counts,
        }
        (out_root / "_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
        if missing:
            print(f"WARNING: {len(missing)} classes have zero crops: {missing}")
            print("Hint: Use --source yolo-output to align IDs with YAML; avoid raw mode unless --strict-raw-check is enabled.")

    print(
        f"Done. Images scanned: {total_images}, crops saved: {total_crops}. Output: {out_root}"
    )


if __name__ == "__main__":
    main()
