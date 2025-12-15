
import sys
import time

print("Starting imports...", flush=True)

start = time.time()
import os
print(f"Imported os in {time.time()-start:.4f}s", flush=True)

start = time.time()
import cv2
print(f"Imported cv2 in {time.time()-start:.4f}s", flush=True)

start = time.time()
import numpy as np
print(f"Imported numpy in {time.time()-start:.4f}s", flush=True)

start = time.time()
from pathlib import Path
print(f"Imported pathlib in {time.time()-start:.4f}s", flush=True)

try:
    start = time.time()
    from ultralytics import YOLO
    print(f"Imported ultralytics in {time.time()-start:.4f}s", flush=True)
except ImportError as e:
    print(f"Failed to import ultralytics: {e}", flush=True)

try:
    start = time.time()
    import albumentations as A
    print(f"Imported albumentations in {time.time()-start:.4f}s", flush=True)
except ImportError as e:
    print(f"Failed to import albumentations: {e}", flush=True)

print("All imports done.", flush=True)
