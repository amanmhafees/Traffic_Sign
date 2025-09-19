# Traffic Sign Recognition - Enhanced Version

A comprehensive traffic sign recognition system using YOLO v11 with advanced data analysis, image augmentation, and visualization capabilities.

## 🚀 New Features

### 📊 Comprehensive Data Analysis
- **Class Distribution Analysis**: Visualize the distribution of traffic sign classes
- **Image Statistics**: Analyze image dimensions, aspect ratios, and quality metrics
- **Class Imbalance Detection**: Identify classes with insufficient data
- **Category Distribution**: Analyze distribution across traffic sign categories
- **Augmentation Recommendations**: Get suggestions for handling class imbalance

### 🔄 Advanced Image Augmentation
- **Multiple Augmentation Pipelines**: Basic, aggressive, weather, geometric, color, and noise augmentations
- **Class-Specific Balancing**: Automatically balance dataset by augmenting minority classes
- **Augmentation Visualization**: Visualize different augmentation techniques
- **Smart Augmentation Strategy**: Choose augmentation intensity based on class deficit

### 📈 Comprehensive Training Visualization
- **Training Curves**: Plot loss curves, mAP, precision, recall, and F1 scores
- **Interactive Dashboards**: Interactive Plotly dashboards for training metrics
- **Confusion Matrix**: Visualize model performance per class
- **Detection Examples**: Show detection results on sample images
- **Class Performance Analysis**: Detailed per-class performance metrics

### 🎯 Enhanced Training Pipeline
- **Automatic Dataset Analysis**: Run analysis before training
- **Smart Data Balancing**: Balance dataset using augmentation
- **Comprehensive Logging**: Detailed training logs and summaries
- **Model Versioning**: Automatic model versioning with timestamps
- **Multiple Model Formats**: Save models in various formats (PyTorch, ONNX)

## 📁 Project Structure

```
traffic_sign_recognition/
├── traffic_sign_recognition.py      # Main training and inference script
├── data_analysis.py                 # Comprehensive data analysis module
├── image_augmentation.py            # Advanced image augmentation module
├── training_visualization.py        # Training visualization module
├── streamlit_app.py                 # Web interface
├── comprehensive_example.py         # Complete example script
├── requirements_enhanced.txt        # Enhanced dependencies
├── config.yaml                      # Configuration file
├── Dataset/                         # Traffic sign dataset
│   ├── Mandatory_Traffic_Signs/
│   ├── Cautionary_Traffic_Signs/
│   └── Informatory_Traffic_Signs/
└── output/                          # Output directory
    ├── analysis/                    # Data analysis plots
    ├── augmented_data/              # Augmented images
    ├── visualizations/              # Training visualizations
    ├── train/                       # Training data
    ├── val/                         # Validation data
    └── traffic_sign_model_*/        # Trained models
```

## 🛠️ Installation

1. **Install Dependencies**:
```bash
pip install -r requirements_enhanced.txt
```

2. **Verify Installation**:
```bash
python test_installation.py
```

## 🚀 Quick Start

### 1. Run Comprehensive Analysis
```bash
python traffic_sign_recognition.py --mode analyze
```

### 2. Balance Dataset with Augmentation
```bash
python traffic_sign_recognition.py --mode balance --balance mean
```

### 3. Train with Full Visualization
```bash
python traffic_sign_recognition.py --mode train --epochs 50 --balance mean
```

### 4. Run Complete Example
```bash
python comprehensive_example.py
```

## 📊 Usage Examples

### Data Analysis
```python
from traffic_sign_recognition import TrafficSignRecognition

# Initialize system
tsr = TrafficSignRecognition("Dataset", "output")

# Run comprehensive analysis
analysis_results = tsr.analyze_dataset()

# Print summary
print(f"Total Images: {analysis_results['analysis']['total_images']}")
print(f"Imbalanced Classes: {len(analysis_results['analysis']['imbalanced_classes'])}")
```

### Dataset Balancing
```python
# Balance dataset using augmentation
balanced_dataset = tsr.balance_dataset("mean")

# Check results
for class_name, data in balanced_dataset.items():
    print(f"{class_name}: {data['original_count']} -> {data['augmented_count']}")
```

### Training with Visualization
```python
# Prepare dataset
tsr.prepare_dataset(val_from_train=True)

# Train with comprehensive visualization
model_path = tsr.train_model(epochs=50, imgsz=640, batch=16)

# Visualizations are automatically generated
```

## 📈 Generated Visualizations

### Data Analysis Plots
- `class_distribution.png`: Bar chart of class distribution
- `class_imbalance.png`: Class imbalance analysis
- `image_statistics.png`: Image dimension and quality statistics
- `category_distribution.png`: Distribution by traffic sign categories

### Training Visualizations
- `training_curves.png`: Comprehensive training curves
- `confusion_matrix.png`: Confusion matrix (raw and normalized)
- `class_performance.png`: Per-class performance metrics
- `detection_examples.png`: Sample detection results
- `interactive_dashboard.html`: Interactive Plotly dashboard

### Augmentation Examples
- `augmentation_examples_*.png`: Visualization of different augmentation techniques

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Dataset Configuration
dataset:
  train_split: 0.8
  val_from_train: true
  image_formats: [".jpg", ".jpeg", ".png"]

# Training Configuration
training:
  epochs: 100
  batch_size: 16
  image_size: 640
  save_period: 10
  save_best: true
  save_generic_names: true

# Augmentation Configuration
augmentation:
  basic_pipeline: true
  aggressive_pipeline: true
  weather_pipeline: true
  geometric_pipeline: true
  color_pipeline: true
  noise_pipeline: true
```

## 🎯 Advanced Features

### Class Imbalance Handling
The system automatically detects and handles class imbalance:

1. **Analysis**: Identifies classes with insufficient data
2. **Recommendations**: Suggests augmentation strategies
3. **Balancing**: Automatically augments minority classes
4. **Visualization**: Shows before/after class distribution

### Smart Augmentation
Different augmentation strategies based on class deficit:

- **Basic**: For classes with minor deficit
- **Aggressive**: For classes with major deficit
- **Weather**: For robustness to weather conditions
- **Geometric**: For perspective variations
- **Color**: For lighting variations
- **Noise**: For quality degradation

### Comprehensive Monitoring
- **Real-time Metrics**: Track training progress
- **Interactive Dashboards**: Explore training data
- **Detection Examples**: Visualize model performance
- **Class Analysis**: Detailed per-class metrics

## 🔧 Command Line Options

```bash
python traffic_sign_recognition.py --help
```

### Available Modes:
- `analyze`: Run comprehensive dataset analysis
- `balance`: Balance dataset with augmentation
- `prepare`: Prepare dataset for training
- `train`: Train model with visualization
- `validate`: Validate trained model
- `detect`: Run real-time detection
- `export`: Export model to different formats

### Key Parameters:
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--balance`: Balancing strategy (mean, max, custom)
- `--analyze-only`: Run analysis without training
- `--device`: Device to use (auto, cpu, 0, 1, etc.)

## 📊 Output Structure

```
output/
├── analysis/                        # Data analysis results
│   ├── class_distribution.png
│   ├── class_imbalance.png
│   ├── image_statistics.png
│   ├── category_distribution.png
│   ├── dataset_summary.json
│   └── augmentation_recommendations.json
├── augmented_data/                  # Augmented images
│   ├── CLASS_NAME/
│   │   ├── original_*.jpg
│   │   ├── aug_*.jpg
│   │   └── augmentation_examples_*.png
│   └── augmentation_report.txt
├── visualizations/                  # Training visualizations
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── class_performance.png
│   ├── detection_examples.png
│   └── interactive_dashboard.html
├── train/                          # Training data
├── val/                            # Validation data
└── traffic_sign_model_*/           # Trained models
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    └── training_summary.txt
```

## 🎨 Web Interface

Launch the Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

Features:
- Upload images for detection
- Adjust confidence threshold
- View detection results
- Download results

## 🔍 Troubleshooting

### Common Issues:

1. **Memory Issues**: Reduce batch size or image size
2. **CUDA Out of Memory**: Use CPU or reduce batch size
3. **Missing Dependencies**: Install requirements_enhanced.txt
4. **Dataset Issues**: Check dataset structure and file formats

### Debug Mode:
```bash
python traffic_sign_recognition.py --mode analyze --verbose
```

## 📚 Examples

### Complete Workflow:
```bash
# 1. Analyze dataset
python traffic_sign_recognition.py --mode analyze

# 2. Balance dataset
python traffic_sign_recognition.py --mode balance --balance mean

# 3. Train model
python traffic_sign_recognition.py --mode train --epochs 50 --balance mean

# 4. Validate model
python traffic_sign_recognition.py --mode validate --model output/best_model.pt

# 5. Run detection
python traffic_sign_recognition.py --mode detect --model output/best_model.pt
```

### Custom Analysis:
```python
from traffic_sign_recognition import TrafficSignRecognition

tsr = TrafficSignRecognition("Dataset", "output")

# Custom analysis
analysis = tsr.analyze_dataset()
print(f"Found {analysis['analysis']['total_classes']} classes")

# Custom balancing
balanced = tsr.balance_dataset("max")
print(f"Balanced {len(balanced)} classes")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO v11 by Ultralytics
- Albumentations for image augmentation
- Streamlit for web interface
- Matplotlib and Plotly for visualizations