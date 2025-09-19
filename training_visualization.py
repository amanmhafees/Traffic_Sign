#!/usr/bin/env python3
"""
Training Visualization Module for Traffic Sign Recognition
Includes comprehensive plotting for training metrics, validation results, and model performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingVisualizer:
    def __init__(self, output_path: str = "output"):
        """
        Initialize the Training Visualizer
        
        Args:
            output_path: Path for output files
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Create visualization output directory
        self.viz_dir = self.output_path / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_training_curves(self, results_csv_path: str, model_name: str = "traffic_sign_model"):
        """
        Plot comprehensive training curves
        
        Args:
            results_csv_path: Path to results.csv file
            model_name: Name of the model
        """
        try:
            # Read results
            df = pd.read_csv(results_csv_path)
            
            # Create comprehensive training plots
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'Training Progress - {model_name}', fontsize=16, fontweight='bold')
            
            # Loss curves
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue', linewidth=2)
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red', linewidth=2)
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Objectness loss
            axes[0, 1].plot(df['epoch'], df['train/obj_loss'], label='Train Obj Loss', color='green', linewidth=2)
            axes[0, 1].plot(df['epoch'], df['val/obj_loss'], label='Val Obj Loss', color='orange', linewidth=2)
            axes[0, 1].set_title('Objectness Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Classification loss
            axes[0, 2].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', color='purple', linewidth=2)
            axes[0, 2].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='brown', linewidth=2)
            axes[0, 2].set_title('Classification Loss')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # mAP curves
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='red', linewidth=2)
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='blue', linewidth=2)
            axes[1, 0].set_title('Mean Average Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Precision and Recall
            axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='green', linewidth=2)
            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='orange', linewidth=2)
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Learning rate (if available)
            if 'lr/pg0' in df.columns:
                axes[1, 2].plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='purple', linewidth=2)
                axes[1, 2].set_title('Learning Rate Schedule')
                axes[1, 2].set_xlabel('Epoch')
                axes[1, 2].set_ylabel('Learning Rate')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            else:
                # Show F1 score instead
                if 'metrics/f1(B)' in df.columns:
                    axes[1, 2].plot(df['epoch'], df['metrics/f1(B)'], label='F1 Score', color='purple', linewidth=2)
                    axes[1, 2].set_title('F1 Score')
                    axes[1, 2].set_xlabel('Epoch')
                    axes[1, 2].set_ylabel('F1 Score')
                    axes[1, 2].legend()
                    axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.viz_dir / f"{model_name}_training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting training curves: {e}")
    
    def plot_confusion_matrix(self, y_true: List, y_pred: List, class_names: List[str], 
                            model_name: str = "traffic_sign_model"):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            model_name: Name of the model
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Raw confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                       xticklabels=class_names, yticklabels=class_names)
            ax1.set_title('Confusion Matrix (Raw Counts)')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # Normalized confusion matrix
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                       xticklabels=class_names, yticklabels=class_names)
            ax2.set_title('Confusion Matrix (Normalized)')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.viz_dir / f"{model_name}_confusion_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def plot_class_performance(self, class_metrics: Dict, model_name: str = "traffic_sign_model"):
        """
        Plot per-class performance metrics
        
        Args:
            class_metrics: Dictionary with per-class metrics
            model_name: Name of the model
        """
        try:
            # Extract data
            classes = list(class_metrics.keys())
            precision = [class_metrics[cls].get('precision', 0) for cls in classes]
            recall = [class_metrics[cls].get('recall', 0) for cls in classes]
            f1 = [class_metrics[cls].get('f1', 0) for cls in classes]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # Precision by class
            axes[0, 0].bar(range(len(classes)), precision, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Precision by Class')
            axes[0, 0].set_xlabel('Classes')
            axes[0, 0].set_ylabel('Precision')
            axes[0, 0].set_xticks(range(len(classes)))
            axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Recall by class
            axes[0, 1].bar(range(len(classes)), recall, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Recall by Class')
            axes[0, 1].set_xlabel('Classes')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].set_xticks(range(len(classes)))
            axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            
            # F1 score by class
            axes[1, 0].bar(range(len(classes)), f1, color='salmon', alpha=0.7)
            axes[1, 0].set_title('F1 Score by Class')
            axes[1, 0].set_xlabel('Classes')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_xticks(range(len(classes)))
            axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Combined metrics
            x = np.arange(len(classes))
            width = 0.25
            
            axes[1, 1].bar(x - width, precision, width, label='Precision', alpha=0.7)
            axes[1, 1].bar(x, recall, width, label='Recall', alpha=0.7)
            axes[1, 1].bar(x + width, f1, width, label='F1 Score', alpha=0.7)
            
            axes[1, 1].set_title('Combined Metrics by Class')
            axes[1, 1].set_xlabel('Classes')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(classes, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.viz_dir / f"{model_name}_class_performance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Class performance plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting class performance: {e}")
    
    def plot_detection_examples(self, model, test_images: List[Path], 
                              class_names: List[str], model_name: str = "traffic_sign_model"):
        """
        Plot detection examples with bounding boxes
        
        Args:
            model: Trained YOLO model
            test_images: List of test image paths
            class_names: List of class names
            model_name: Name of the model
        """
        try:
            # Select sample images
            sample_images = test_images[:8]  # Show 8 examples
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, img_path in enumerate(sample_images):
                if i >= 8:
                    break
                    
                # Load and process image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Run detection
                results = model.predict(img, conf=0.4, verbose=False)
                
                # Draw detections
                result_img = img.copy()
                for res in results:
                    if res.boxes is not None:
                        for box in res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
                            
                            # Draw bounding box
                            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"{cls_name} {conf:.2f}"
                            cv2.putText(result_img, label, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Convert BGR to RGB for matplotlib
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # Show image
                axes[i].imshow(result_img_rgb)
                axes[i].set_title(f"Detection Example {i+1}")
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(sample_images), 8):
                axes[i].axis('off')
            
            plt.suptitle(f'Detection Examples - {model_name}', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plot_path = self.viz_dir / f"{model_name}_detection_examples.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Detection examples saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting detection examples: {e}")
    
    def create_interactive_dashboard(self, results_csv_path: str, model_name: str = "traffic_sign_model"):
        """
        Create an interactive dashboard using Plotly
        
        Args:
            results_csv_path: Path to results.csv file
            model_name: Name of the model
        """
        try:
            # Read results
            df = pd.read_csv(results_csv_path)
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Box Loss', 'Objectness Loss', 'Classification Loss', 
                              'mAP@0.5', 'Precision & Recall', 'Learning Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Box Loss
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train/box_loss'], name='Train Box Loss', 
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['val/box_loss'], name='Val Box Loss', 
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
            
            # Objectness Loss
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train/obj_loss'], name='Train Obj Loss', 
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['val/obj_loss'], name='Val Obj Loss', 
                          line=dict(color='orange', width=2)),
                row=1, col=2
            )
            
            # Classification Loss
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['train/cls_loss'], name='Train Cls Loss', 
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['val/cls_loss'], name='Val Cls Loss', 
                          line=dict(color='brown', width=2)),
                row=2, col=1
            )
            
            # mAP
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['metrics/mAP50(B)'], name='mAP@0.5', 
                          line=dict(color='red', width=2)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['metrics/mAP50-95(B)'], name='mAP@0.5:0.95', 
                          line=dict(color='blue', width=2)),
                row=2, col=2
            )
            
            # Precision and Recall
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['metrics/precision(B)'], name='Precision', 
                          line=dict(color='green', width=2)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['metrics/recall(B)'], name='Recall', 
                          line=dict(color='orange', width=2)),
                row=3, col=1
            )
            
            # Learning Rate
            if 'lr/pg0' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['epoch'], y=df['lr/pg0'], name='Learning Rate', 
                              line=dict(color='purple', width=2)),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f'Interactive Training Dashboard - {model_name}',
                height=1200,
                showlegend=True
            )
            
            # Save interactive dashboard
            dashboard_path = self.viz_dir / f"{model_name}_interactive_dashboard.html"
            fig.write_html(str(dashboard_path))
            
            logger.info(f"Interactive dashboard saved to {dashboard_path}")
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
    
    def generate_comprehensive_report(self, model_name: str = "traffic_sign_model"):
        """
        Generate a comprehensive training report with all visualizations
        
        Args:
            model_name: Name of the model
        """
        try:
            # Find results file
            results_files = list(self.output_path.glob(f"**/{model_name}/results.csv"))
            if not results_files:
                logger.warning(f"No results file found for model {model_name}")
                return
            
            results_csv_path = results_files[0]
            
            # Generate all plots
            self.plot_training_curves(str(results_csv_path), model_name)
            self.create_interactive_dashboard(str(results_csv_path), model_name)
            
            logger.info(f"Comprehensive report generated for {model_name}")
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")

def main():
    """Main function to test visualization"""
    visualizer = TrainingVisualizer()
    
    # Test with sample data
    print("Training visualizer initialized!")
    print(f"Visualization directory: {visualizer.viz_dir}")

if __name__ == "__main__":
    main()