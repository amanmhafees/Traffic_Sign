#!/usr/bin/env python3
"""
Data Analysis Module for Traffic Sign Recognition
Includes class distribution analysis, image statistics, and visualization
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional
import logging
import pathlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_paths_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_str(i) for i in obj]
    elif isinstance(obj, pathlib.Path):
        return str(obj)
    else:
        return obj

def convert_numpy_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class DataAnalyzer:
    def __init__(self, dataset_path: str = "Dataset", output_path: str = "output"):
        """
        Initialize the Data Analyzer
        
        Args:
            dataset_path: Path to the dataset directory
            output_path: Path for output files
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Create analysis output directory
        self.analysis_dir = self.output_path / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.class_stats = {}
        self.image_stats = {}
        self.class_distribution = {}
        
    def analyze_dataset(self) -> Dict:
        """
        Perform comprehensive dataset analysis
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting comprehensive dataset analysis...")
        
        # Get all class directories
        class_dirs = []
        for category in ["Mandatory_Traffic_Signs", "Cautionary_Traffic_Signs", "Informatory_Traffic_Signs"]:
            category_path = self.dataset_path / category
            if category_path.exists():
                class_dirs.extend([d for d in category_path.iterdir() if d.is_dir()])
        
        # Analyze each class
        total_images = 0
        class_data = {}
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            image_files = list(class_dir.glob("*.jpg"))
            
            if not image_files:
                continue
                
            class_data[class_name] = {
                'count': len(image_files),
                'images': image_files,
                'category': class_dir.parent.name
            }
            total_images += len(image_files)
        
        # Store results
        self.class_distribution = class_data
        
        # Generate analysis plots
        self._plot_class_distribution()
        self._plot_class_imbalance()
        self._plot_image_statistics()
        self._plot_category_distribution()
        
        # Generate summary report
        summary = self._generate_summary_report(total_images, class_data)
        
        logger.info("Dataset analysis completed!")
        return summary
    

    #class distribution
    def _plot_class_distribution(self):
        """Plot class distribution bar chart"""
        plt.figure(figsize=(20, 10))
        
        classes = list(self.class_distribution.keys())
        counts = [self.class_distribution[cls]['count'] for cls in classes]
        
        # Sort by count for better visualization
        sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_data)
        
        plt.bar(range(len(classes)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Traffic Sign Classes')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution - Traffic Sign Dataset')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + max(counts) * 0.01, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Class distribution plot saved")
    
    def _plot_class_imbalance(self):
        """Plot class imbalance analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        classes = list(self.class_distribution.keys())
        counts = [self.class_distribution[cls]['count'] for cls in classes]
        
        # Sort by count
        sorted_data = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_data)
        
        # Bar chart
        bars = ax1.bar(range(len(classes)), counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax1.set_xlabel('Traffic Sign Classes')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Class Imbalance Analysis')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add imbalance indicators
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        # Highlight severely imbalanced classes
        for i, count in enumerate(counts):
            if count < mean_count - std_count:
                bars[i].set_color('red')
            elif count > mean_count + std_count:
                bars[i].set_color('green')
        
        # Pie chart for top 10 classes
        top_10_data = sorted_data[:10]
        top_10_classes, top_10_counts = zip(*top_10_data)
        other_count = sum(counts[10:]) if len(counts) > 10 else 0
        
        if other_count > 0:
            pie_labels = list(top_10_classes) + ['Others']
            pie_counts = list(top_10_counts) + [other_count]
        else:
            pie_labels = list(top_10_classes)
            pie_counts = list(top_10_counts)
        
        ax2.pie(pie_counts, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Top 10 Classes Distribution')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'class_imbalance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Class imbalance plot saved")
    
    def _plot_image_statistics(self):
        """Plot image statistics (size, aspect ratio, etc.)"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        all_widths = []
        all_heights = []
        all_aspect_ratios = []
        all_areas = []
        
        # Sample images from each class for statistics
        sample_size = min(50, min([self.class_distribution[cls]['count'] for cls in self.class_distribution.keys()]))
        
        for class_name, class_data in self.class_distribution.items():
            images = class_data['images'][:sample_size]  # Sample for performance
            
            for img_path in images:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        all_widths.append(w)
                        all_heights.append(h)
                        all_aspect_ratios.append(w / h)
                        all_areas.append(w * h)
                except Exception as e:
                    logger.warning(f"Error reading image {img_path}: {e}")
        
        # Width distribution
        ax1.hist(all_widths, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Image Width (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Image Width Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Height distribution
        ax2.hist(all_heights, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Image Height (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Image Height Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Aspect ratio distribution
        ax3.hist(all_aspect_ratios, bins=50, color='salmon', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Aspect Ratio (Width/Height)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Aspect Ratio Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Area distribution
        ax4.hist(all_areas, bins=50, color='gold', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Image Area (pixels²)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Image Area Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'image_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Image statistics plot saved")
    
    def _plot_category_distribution(self):
        """Plot distribution by traffic sign categories"""
        plt.figure(figsize=(12, 8))
        
        categories = defaultdict(int)
        for class_data in self.class_distribution.values():
            categories[class_data['category']] += class_data['count']
        
        # Create pie chart
        labels = list(categories.keys())
        sizes = list(categories.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Distribution by Traffic Sign Categories')
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Category distribution plot saved")
    
    def _generate_summary_report(self, total_images: int, class_data: Dict) -> Dict:
        """Generate comprehensive summary report"""
        summary = {
            'total_images': total_images,
            'total_classes': len(class_data),
            'class_distribution': class_data,
            'statistics': {}
        }
        
        # Calculate statistics
        counts = [data['count'] for data in class_data.values()]
        summary['statistics'] = {
            'mean_images_per_class': np.mean(counts),
            'std_images_per_class': np.std(counts),
            'min_images_per_class': np.min(counts),
            'max_images_per_class': np.max(counts),
            'median_images_per_class': np.median(counts)
        }
        
        # Identify imbalanced classes
        mean_count = summary['statistics']['mean_images_per_class']
        std_count = summary['statistics']['std_images_per_class']
        
        imbalanced_classes = []
        for class_name, data in class_data.items():
            if data['count'] < mean_count - std_count:
                imbalanced_classes.append({
                    'class': class_name,
                    'count': data['count'],
                    'deficit': int(mean_count - data['count'])
                })
        
        summary['imbalanced_classes'] = imbalanced_classes
        
        # Save summary to JSON
        with open(self.analysis_dir / 'dataset_summary.json', 'w') as f:
            json.dump(convert_numpy_to_native(convert_paths_to_str(summary)), f, indent=2)
        
        logger.info(f"Summary report saved with {len(imbalanced_classes)} imbalanced classes identified")
        return summary
    
    def get_augmentation_recommendations(self) -> Dict:
        """Get recommendations for handling class imbalance"""
        if not self.class_distribution:
            self.analyze_dataset()
        
        counts = [data['count'] for data in self.class_distribution.values()]
        mean_count = np.mean(counts)
        
        recommendations = {
            'oversampling_needed': [],
            'undersampling_needed': [],
            'augmentation_strategy': {}
        }
        
        for class_name, data in self.class_distribution.items():
            count = data['count']
            if count < mean_count * 0.5:  # Less than 50% of mean
                recommendations['oversampling_needed'].append({
                    'class': class_name,
                    'current_count': count,
                    'target_count': int(mean_count),
                    'augment_factor': int(mean_count / count)
                })
            elif count > mean_count * 2:  # More than 200% of mean
                recommendations['undersampling_needed'].append({
                    'class': class_name,
                    'current_count': count,
                    'target_count': int(mean_count)
                })
        
        # Save recommendations
        with open(self.analysis_dir / 'augmentation_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        return recommendations

def main():
    """Main function to run data analysis"""
    analyzer = DataAnalyzer()
    
    print("Starting dataset analysis...")
    summary = analyzer.analyze_dataset()
    
    print(f"\nDataset Analysis Summary:")
    print(f"Total Images: {summary['total_images']}")
    print(f"Total Classes: {summary['total_classes']}")
    print(f"Mean Images per Class: {summary['statistics']['mean_images_per_class']:.1f}")
    print(f"Imbalanced Classes: {len(summary['imbalanced_classes'])}")
    
    # Get augmentation recommendations
    recommendations = analyzer.get_augmentation_recommendations()
    print(f"\nOversampling needed for {len(recommendations['oversampling_needed'])} classes")
    print(f"Undersampling needed for {len(recommendations['undersampling_needed'])} classes")
    
    print(f"\nAnalysis plots saved to: {analyzer.analysis_dir}")

if __name__ == "__main__":
    main()