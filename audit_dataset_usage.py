#!/usr/bin/env python3
"""
Audit that the prepared YOLO dataset (output/train|val|test) correctly
uses all available images, includes augmented images, and has valid labels.

Checks performed:
- Validate that output/traffic_signs.yaml exists and is consistent
- Count images and labels in each split; report missing labels or images
- Verify label indices are within [0, nc-1]
- Compute per-class label counts to spot skew (e.g., GAP_IN_MEDIAN dominance)
- Estimate augmentation usage via filename prefix (aug_*)
- Detect if Dataset has newer images than the prepared output (stale split)

Usage:
  python audit_dataset_usage.py
Optional args:
  --yaml output/traffic_signs.yaml
  --dataset Dataset
  --output output
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import time
import yaml


@dataclass
class SplitReport:
    name: str
    images: int
    labels: int
    images_missing_labels: int
    labels_missing_images: int
    aug_images: int
    class_counts: Dict[int, int]


def load_yaml(yaml_path: Path) -> dict:
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    required = ["path", "train", "val", "names", "nc"]
    for k in required:
        if k not in data:
            raise ValueError(f"YAML missing required key: {k}")
    return data


def index_split(img_dir: Path, lbl_dir: Path) -> Tuple[List[Path], List[Path]]:
    imgs = sorted([p for p in img_dir.glob("*.jpg")])
    lbls = sorted([p for p in lbl_dir.glob("*.txt")])
    return imgs, lbls


def parse_labels(lbl_path: Path) -> List[int]:
    cls_ids = []
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_ids.append(int(parts[0]))
                except ValueError:
                    # malformed line; skip
                    continue
    except Exception:
        pass
    return cls_ids


def audit_split(name: str, split_root: Path, nc: int) -> SplitReport:
    img_dir = split_root / "images"
    lbl_dir = split_root / "labels"
    imgs, lbls = index_split(img_dir, lbl_dir)

    img_stems = {p.stem for p in imgs}
    lbl_stems = {p.stem for p in lbls}
    missing_lbl = img_stems - lbl_stems
    missing_img = lbl_stems - img_stems

    class_counts: Dict[int, int] = {}
    bad_ids: Dict[int, int] = {}
    for lp in lbls:
        for cid in parse_labels(lp):
            if 0 <= cid < nc:
                class_counts[cid] = class_counts.get(cid, 0) + 1
            else:
                bad_ids[cid] = bad_ids.get(cid, 0) + 1

    aug_images = sum(1 for p in imgs if p.name.lower().startswith("aug_"))

    if bad_ids:
        print(f"[WARN] {name}: Found label IDs out of range: {sorted(bad_ids.items())[:5]} ...")

    return SplitReport(
        name=name,
        images=len(imgs),
        labels=len(lbls),
        images_missing_labels=len(missing_lbl),
        labels_missing_images=len(missing_img),
        aug_images=aug_images,
        class_counts=class_counts,
    )


def dataset_newer_than_output(dataset_dir: Path, output_dir: Path) -> bool:
    def newest_mtime(root: Path) -> float:
        newest = 0.0
        for p in root.rglob("*.jpg"):
            try:
                newest = max(newest, p.stat().st_mtime)
            except Exception:
                continue
        for p in root.rglob("*.txt"):
            try:
                newest = max(newest, p.stat().st_mtime)
            except Exception:
                continue
        return newest

    ds_mtime = newest_mtime(dataset_dir)
    out_mtime = newest_mtime(output_dir)
    return ds_mtime > out_mtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=str(Path("output") / "traffic_signs.yaml"))
    ap.add_argument("--dataset", default="Dataset")
    ap.add_argument("--output", default="output")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    data = load_yaml(yaml_path)

    root = Path(data["path"]) if "path" in data else Path(args.output)
    nc = int(data["nc"])
    names: List[str] = list(data["names"]) if isinstance(data["names"], list) else []

    train_root = root / Path(data["train"]).parent
    val_root = root / Path(data["val"]).parent
    test_root = root / Path(data.get("test", "test/images")).parent

    print(f"Using dataset root: {root}")
    print(f"Classes (nc): {nc}")
    if names:
        print(f"Sample class names: {names[:5]} ...")

    reports: List[SplitReport] = []
    for name, split_root in [("train", train_root), ("val", val_root), ("test", test_root)]:
        if not split_root.exists():
            print(f"[WARN] Split not found: {split_root}")
            continue
        rep = audit_split(name, split_root, nc)
        reports.append(rep)

    # Aggregate
    total_images = sum(r.images for r in reports)
    total_labels = sum(r.labels for r in reports)
    total_missing_lbl = sum(r.images_missing_labels for r in reports)
    total_missing_img = sum(r.labels_missing_images for r in reports)
    total_aug = sum(r.aug_images for r in reports)

    print("\n=== SPLIT SUMMARY ===")
    for r in reports:
        aug_pct = (r.aug_images / r.images * 100.0) if r.images else 0.0
        print(
            f"{r.name:5} | images: {r.images:6} | labels: {r.labels:6} | "
            f"img->no_lbl: {r.images_missing_labels:5} | lbl->no_img: {r.labels_missing_images:5} | "
            f"aug: {r.aug_images:6} ({aug_pct:.1f}%)"
        )

    print("\n=== GLOBAL SUMMARY ===")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Images missing labels: {total_missing_lbl}")
    print(f"Labels missing images: {total_missing_img}")
    print(f"Augmented images (by name prefix): {total_aug}")

    # Class distribution across all splits
    agg_class_counts: Dict[int, int] = {}
    for r in reports:
        for k, v in r.class_counts.items():
            agg_class_counts[k] = agg_class_counts.get(k, 0) + v

    if agg_class_counts:
        # Save as CSV for easy review
        out_csv = Path(args.output) / "analysis" / "class_counts_across_splits.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class_id", "class_name", "label_instances"])
            for cid in sorted(agg_class_counts):
                cname = names[cid] if 0 <= cid < len(names) else "(unknown)"
                w.writerow([cid, cname, agg_class_counts[cid]])
        print(f"Per-class label counts saved to: {out_csv}")

        # Top-5 classes by label instances
        top5 = sorted(agg_class_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("Top-5 classes by label instances:")
        for cid, cnt in top5:
            cname = names[cid] if 0 <= cid < len(names) else str(cid)
            print(f"  {cid:3} {cname:30} {cnt}")

    # Staleness check: Dataset newer than output -> must re-prepare
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    try:
        if dataset_dir.exists() and output_dir.exists():
            if dataset_newer_than_output(dataset_dir, output_dir):
                print("\n[STALE] Dataset has files newer than output splits.")
                print("        Re-run prepare step to include new/augmented data:")
                print("        python traffic_sign_recognition.py --mode prepare")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
