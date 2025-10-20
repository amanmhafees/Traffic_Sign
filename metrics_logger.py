import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class MetricsLogger:
	"""
	Lightweight metrics/plots logger.
	Call update_epoch(...) after each epoch and call save_epoch_plots() to persist curves.
	Call save_confusion_matrix(...) and save_per_class_metrics(...) after validation/testing.
	"""

	def __init__(self, out_root: Union[str, Path], class_names: List[str]):
		self.out_root = Path(out_root)
		self.plots_dir = self.out_root / "visualizations"
		self.plots_dir.mkdir(parents=True, exist_ok=True)
		self.class_names = class_names

		# time-series storage
		self.epochs: List[int] = []
		self.loss_total: List[float] = []
		self.loss_components: Dict[str, List[float]] = {}  # e.g. {"box": [], "obj": [], "cls": []}
		self.precision: List[float] = []
		self.recall: List[float] = []
		self.f1: List[float] = []
		self.map50: List[float] = []
		self.map50_95: List[float] = []

	def update_epoch(
		self,
		epoch: int,
		loss_total: Optional[float] = None,
		loss_parts: Optional[Dict[str, float]] = None,
		precision: Optional[float] = None,
		recall: Optional[float] = None,
		f1: Optional[float] = None,
		map50: Optional[float] = None,
		map50_95: Optional[float] = None,
	) -> None:
		self.epochs.append(epoch)
		self.loss_total.append(loss_total if loss_total is not None else np.nan)
		if loss_parts:
			for k in loss_parts.keys():
				self.loss_components.setdefault(k, [])
			# align component lengths
			for k in self.loss_components.keys():
				self.loss_components[k].append(loss_parts.get(k, np.nan))
		self.precision.append(precision if precision is not None else np.nan)
		self.recall.append(recall if recall is not None else np.nan)
		self.f1.append(f1 if f1 is not None else np.nan)
		self.map50.append(map50 if map50 is not None else np.nan)
		self.map50_95.append(map50_95 if map50_95 is not None else np.nan)

	def save_epoch_plots(self, prefix: str = "") -> None:
		pfx = f"{prefix}_" if prefix else ""

		# Loss vs Epoch
		try:
			plt.figure(figsize=(8, 5))
			if any(~np.isnan(self.loss_total)):
				plt.plot(self.epochs, self.loss_total, label="total", linewidth=2)
			for name, series in self.loss_components.items():
				if any(~np.isnan(series)):
					plt.plot(self.epochs, series, label=f"loss_{name}", alpha=0.85)
			plt.xlabel("Epoch")
			plt.ylabel("Loss")
			plt.title("Loss vs Epoch")
			plt.grid(True, alpha=0.3)
			plt.legend()
			plt.tight_layout()
			plt.savefig(self.plots_dir / f"{pfx}loss_vs_epoch.png", dpi=220)
			plt.close()
		except Exception:
			plt.close()

		# Precision/Recall/F1 vs Epoch
		try:
			plt.figure(figsize=(8, 5))
			if any(~np.isnan(self.precision)):
				plt.plot(self.epochs, self.precision, label="Precision", linewidth=2)
			if any(~np.isnan(self.recall)):
				plt.plot(self.epochs, self.recall, label="Recall", linewidth=2)
			if any(~np.isnan(self.f1)):
				plt.plot(self.epochs, self.f1, label="F1", linewidth=2)
			plt.xlabel("Epoch")
			plt.ylabel("Score")
			plt.title("Precision / Recall / F1 vs Epoch")
			plt.grid(True, alpha=0.3)
			plt.legend()
			plt.tight_layout()
			plt.savefig(self.plots_dir / f"{pfx}prf1_vs_epoch.png", dpi=220)
			plt.close()
		except Exception:
			plt.close()

		# mAP curves
		try:
			plt.figure(figsize=(8, 5))
			if any(~np.isnan(self.map50)):
				plt.plot(self.epochs, self.map50, label="mAP@50", linewidth=2)
			if any(~np.isnan(self.map50_95)):
				plt.plot(self.epochs, self.map50_95, label="mAP@50-95", linewidth=2)
			plt.xlabel("Epoch")
			plt.ylabel("mAP")
			plt.title("mAP vs Epoch")
			plt.grid(True, alpha=0.3)
			plt.legend()
			plt.tight_layout()
			plt.savefig(self.plots_dir / f"{pfx}map_vs_epoch.png", dpi=220)
			plt.close()
		except Exception:
			plt.close()

	def save_confusion_matrix(
		self,
		cm: np.ndarray,
		normalize: bool = False,
		prefix: str = "",
	) -> None:
		"""
		Save confusion matrix heatmap + CSV.
		cm: shape [n_classes, n_classes] with rows=true, cols=pred.
		"""
		if cm is None or cm.size == 0:
			return
		if normalize:
			with np.errstate(invalid="ignore", divide="ignore"):
				cm = cm.astype(np.float64)
				row_sums = cm.sum(axis=1, keepdims=True)
				cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

		pfx = f"{prefix}_" if prefix else ""
		# CSV
		np.savetxt(self.plots_dir / f"{pfx}confusion_matrix.csv", cm, delimiter=",", fmt="%.6f")
		# Heatmap
		plt.figure(figsize=(max(6, len(self.class_names) * 0.4), max(5, len(self.class_names) * 0.4)))
		sns.heatmap(
			cm,
			annot=False,
			cmap="Blues",
			xticklabels=self.class_names,
			yticklabels=self.class_names,
			cbar_kws={'shrink': 0.6}
		)
		plt.xlabel("Predicted")
		plt.ylabel("True")
		plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
		plt.tight_layout()
		plt.savefig(self.plots_dir / f"{pfx}confusion_matrix.png", dpi=220)
		plt.close()

	def save_per_class_metrics(
		self,
		per_class: Dict[str, Dict[str, float]],
		prefix: str = "",
	) -> None:
		"""
		per_class: { class_name: {"precision": float, "recall": float, "f1": float, "map50": float, "map50_95": float } }
		Saves CSV and bar plots for each metric.
		"""
		if not per_class:
			return
		pfx = f"{prefix}_" if prefix else ""
		# CSV
		headers = ["class", "precision", "recall", "f1", "map50", "map50_95"]
		rows = []
		for cname in self.class_names:
			m = per_class.get(cname, {})
			rows.append([
				cname,
				m.get("precision", np.nan),
				m.get("recall", np.nan),
				m.get("f1", np.nan),
				m.get("map50", np.nan),
				m.get("map50_95", np.nan),
			])
		np.savetxt(self.plots_dir / f"{pfx}per_class_metrics.csv", rows, delimiter=",", fmt="%s", header=",".join(headers), comments="")

		# Bar plots per metric
		def _bar(metric_key: str, title: str, fname: str):
			vals = [float(per_class.get(c, {}).get(metric_key, np.nan)) for c in self.class_names]
			x = np.arange(len(self.class_names))
			plt.figure(figsize=(max(8, len(self.class_names) * 0.45), 5))
			plt.bar(x, vals)
			plt.xticks(x, self.class_names, rotation=45, ha="right")
			plt.ylabel(metric_key)
			plt.title(title)
			plt.tight_layout()
			plt.savefig(self.plots_dir / f"{pfx}{fname}.png", dpi=220)
			plt.close()

		_bar("precision", "Per-class Precision", "per_class_precision")
		_bar("recall", "Per-class Recall", "per_class_recall")
		_bar("f1", "Per-class F1", "per_class_f1")
		_bar("map50", "Per-class mAP@50", "per_class_map50")
		_bar("map50_95", "Per-class mAP@50-95", "per_class_map50_95")

	def save_pr_curve(
		self,
		pr_points: Dict[str, Tuple[Sequence[float], Sequence[float]]],
		prefix: str = "",
	) -> None:
		"""
		pr_points: { class_name: (precision[], recall[]) }
		Generates a multi-class PR curve panel and individual curves per class.
		"""
		if not pr_points:
			return
		pfx = f"{prefix}_" if prefix else ""

		# Multi-class grid up to 16 per page
		classes = [c for c in self.class_names if c in pr_points]
		n = len(classes)
		if n == 0:
			return

		cols = 4
		rows = int(np.ceil(n / cols))
		plt.figure(figsize=(cols * 4, rows * 3.2))
		for i, cname in enumerate(classes, start=1):
			p, r = pr_points[cname]
			plt.subplot(rows, cols, i)
			plt.plot(r, p, label=cname)
			plt.xlabel("Recall")
			plt.ylabel("Precision")
			plt.title(cname)
			plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(self.plots_dir / f"{pfx}pr_curves_panel.png", dpi=220)
		plt.close()

		# Individual per-class PR curves
		for cname in classes:
			p, r = pr_points[cname]
			plt.figure(figsize=(5, 4))
			plt.plot(r, p)
			plt.xlabel("Recall")
			plt.ylabel("Precision")
			plt.title(f"PR Curve - {cname}")
			plt.grid(True, alpha=0.3)
			plt.tight_layout()
			plt.savefig(self.plots_dir / f"{pfx}pr_curve_{cname}.png", dpi=200)
			plt.close()

	@staticmethod
	def from_ultralytics(
		out_root: Union[str, Path],
		class_names: List[str],
		metrics_obj,
	) -> "MetricsLogger":
		"""
		Convenience: build confusion matrix and per-class arrays from a Ultralytics metrics object.
		Then call save_confusion_matrix(...) and save_per_class_metrics(...).
		"""
		logger = MetricsLogger(out_root, class_names)

		# Confusion matrix
		try:
			cm = getattr(getattr(metrics_obj, "confusion_matrix", None), "matrix", None)
			if cm is not None:
				logger.save_confusion_matrix(np.array(cm), normalize=False, prefix="val")
		except Exception:
			pass

		# Per-class metrics if available on metrics object
		try:
			# The below fields can vary between versions; handle gracefully.
			prec = np.array(getattr(getattr(metrics_obj, "box", None), "p", []))  # per-class precision
			rec = np.array(getattr(getattr(metrics_obj, "box", None), "r", []))  # per-class recall
			map50 = np.array(getattr(getattr(metrics_obj, "box", None), "mp", []))  # alias fallbacks common
			map50_95 = np.array(getattr(getattr(metrics_obj, "box", None), "map", []))
			# Build dict
			per_class = {}
			for i, cname in enumerate(class_names):
				per_class[cname] = {
					"precision": float(prec[i]) if i < len(prec) else np.nan,
					"recall": float(rec[i]) if i < len(rec) else np.nan,
					"f1": float(
						(2 * prec[i] * rec[i]) / (prec[i] + rec[i])
					) if i < len(prec) and i < len(rec) and (prec[i] + rec[i]) > 0 else np.nan,
					"map50": float(map50[i]) if i < len(map50) else np.nan,
					"map50_95": float(map50_95[i]) if i < len(map50_95) else np.nan,
				}
			logger.save_per_class_metrics(per_class, prefix="val")
		except Exception:
			pass

		return logger

	def load_ultralytics_results_csv(self, results_csv: Union[str, Path]) -> None:
		"""
		Read Ultralytics results.csv and fill epoch arrays for plotting.
		Supports common headers like:                  epoch,     train/val metrics...
		Example headers include:               epoch,   train/box_loss, train/cls_loss, train/dfl_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
		"""
		p = Path(results_csv)
		if not p.exists():
			return
		import csv
		with p.open("r", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			for row in reader:
				ep = int(float(row.get("epoch", row.get("                 epoch", "0"))))
				loss_parts = {}
				# Try common loss keys
				for k in row.keys():
					lk = k.strip().lower()
					if lk.endswith("loss") and "/" in lk:
						loss_parts[lk.split("/")[-1]] = float(row[k]) if row[k] else np.nan
				# metrics (validation)
				prec = float(row.get("metrics/precision(B)", row.get("metrics/precision", np.nan)) or np.nan)
				rec = float(row.get("metrics/recall(B)", row.get("metrics/recall", np.nan)) or np.nan)
				map50 = float(row.get("metrics/mAP50(B)", row.get("metrics/mAP50", np.nan)) or np.nan)
				map5095 = float(row.get("metrics/mAP50-95(B)", row.get("metrics/mAP50-95", np.nan)) or np.nan)
				# aggregate total loss if parts exist
				total_loss = np.nan
				if loss_parts:
					total_loss = sum(v for v in loss_parts.values() if not np.isnan(v))
				self.update_epoch(ep, total_loss, loss_parts, prec, rec, (2*prec*rec/(prec+rec) if prec+rec>0 else np.nan), map50, map5095)

# Convenience: generate plots from an existing results.csv without retraining
def generate_plots_from_results(out_root: Union[str, Path],
                                class_names: List[str],
                                results_csv: Optional[Union[str, Path]] = None,
                                prefix: str = "train") -> bool:
	"""
	Read results.csv and render Loss, PR/Recall/F1, and mAP curves.
	out_root: project output root (e.g., 'output')
	class_names: list of class names for labeling
	results_csv: optional explicit path; defaults to <out_root>/traffic_sign_model/results.csv
	Returns True if plots were saved, else False.
	"""
	out_root = Path(out_root)
	csv_path = Path(results_csv) if results_csv else (out_root / "traffic_sign_model" / "results.csv")
	if not csv_path.exists():
		logging.warning(f"results.csv not found at: {csv_path}")
		return False

	logger = MetricsLogger(out_root, class_names)
	logger.load_ultralytics_results_csv(csv_path)
	logger.save_epoch_plots(prefix=prefix)
	logging.info(f"Saved training plots to: {(out_root / 'visualizations').resolve()}")
	return True
