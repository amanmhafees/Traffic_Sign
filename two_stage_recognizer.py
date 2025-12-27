from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str
    conf: float
    category: str


class TwoStageSignRecognizer:
    """YOLOv11 for detection, MobileNetV3-Large for classification.

    - YOLO detects boxes only; ignore YOLO classes.
    - CNN classifies cropped ROI resized to 224x224.
    - If CNN confidence < threshold, fallback to 'Generic Prohibitory Sign'.
    """

    def __init__(
        self,
        yolo_weights: str,
        cnn_weights: str = "output/cnn_classifier.pt",
        classes_path: str = "output/cnn_classes.json",
        device: str | None = None,
        classifier_threshold: float = 0.75,
    ) -> None:
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.yolo = YOLO(yolo_weights)

        self.classifier_threshold = float(classifier_threshold)
        self.idx_to_class: Dict[int, str] = {}
        self._build_classifier(cnn_weights, classes_path)
        # Build Mandatory/Cautionary mapping from Dataset folder names
        self.class_category = self._build_class_category_map()

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _build_classifier(self, weights_path: str, classes_path: str) -> None:
        weights_file = Path(weights_path)
        if not weights_file.exists():
            raise FileNotFoundError(
                f"CNN weights not found at {weights_file}. Train it via train_cnn_classifier.py"
            )

        # Load class mapping
        cp = Path(classes_path)
        if cp.exists():
            raw = json.loads(cp.read_text())
            if isinstance(raw, list):
                # List of class names in index order
                self.idx_to_class = {i: str(name) for i, name in enumerate(raw)}
            elif isinstance(raw, dict):
                # Convert string keys to ints
                self.idx_to_class = {int(k): str(v) for k, v in raw.items()}
            else:
                self.idx_to_class = {}
        else:
            # Try to read from model checkpoint if present
            ckpt = torch.load(str(weights_file), map_location="cpu")
            self.idx_to_class = ckpt.get("idx_to_class", {})
            if isinstance(self.idx_to_class, dict):
                # Ensure keys are ints, values are strings
                self.idx_to_class = {int(k): str(v) for k, v in self.idx_to_class.items()}

        num_classes = len(self.idx_to_class) if self.idx_to_class else None

        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        if num_classes is None:
            ckpt = torch.load(str(weights_file), map_location="cpu")
            num_classes = int(ckpt.get("num_classes", model.fc.out_features))
        model.fc = torch.nn.Linear(in_features, num_classes)

        state = torch.load(str(weights_file), map_location="cpu")
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        self.cnn = model.to(self.device).eval()

    def _build_class_category_map(self) -> Dict[str, str]:
        """Scan Dataset folders to map class name -> category (Mandatory/Cautionary)."""
        mapping: Dict[str, str] = {}
        base = Path("Dataset")
        mand = base / "Mandatory_Traffic_Signs"
        caut = base / "Cautionary_Traffic_Signs"
        try:
            if mand.exists():
                for d in mand.iterdir():
                    if d.is_dir():
                        mapping[d.name] = "Mandatory"
            if caut.exists():
                for d in caut.iterdir():
                    if d.is_dir():
                        mapping[d.name] = "Cautionary"
        except Exception:
            pass
        return mapping

    @torch.inference_mode()
    def predict_frame(
        self, frame_bgr: np.ndarray, yolo_conf: float = 0.5, yolo_iou: float = 0.45
    ) -> List[Detection]:
        """Run YOLO detection then CNN classification on each ROI.

        Returns list of Detection with CNN-derived labels and confidences.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        # Use high-resolution inference to improve small sign detection
        infer_size = max(1280, max(h, w))
        results = self.yolo.predict(
            frame_bgr, imgsz=infer_size, conf=yolo_conf, iou=yolo_iou, verbose=False
        )
        dets: List[Detection] = []
        for box in results[0].boxes:
            xyxy = list(map(int, box.xyxy[0].tolist()))
            x1, y1, x2, y2 = xyxy

            # Clip to image bounds (for safety); skip if invalid
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame_bgr[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            inp = self.tf(roi_resized).unsqueeze(0).to(self.device)

            logits = self.cnn(inp)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            conf_val = float(conf.item())
            idx = int(pred_idx.item())
            label = self.idx_to_class.get(idx, str(idx))
            category = self.class_category.get(label, "Unknown")

            # Debug: show raw CNN prediction before any fallback override
            print(f"CNN raw: {label} | confidence: {conf_val:.3f}")

            if conf_val < self.classifier_threshold:
                label = "Generic Prohibitory Sign"

            dets.append(Detection(bbox=(x1, y1, x2, y2), label=label, conf=conf_val, category=category))

        return dets

    @staticmethod
    def draw(frame_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        img = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{d.category}: {d.label} ({d.conf:.2f})"
            cv2.putText(
                img,
                text,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return img
