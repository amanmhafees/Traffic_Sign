# Traffic Sign Recognition & Driver Alert System (India)

A comprehensive traffic sign recognition system using YOLO v11 for Indian traffic signs with real-time driver alerts.

## Features

- **YOLO v11 Integration**: Latest YOLO model for superior detection accuracy
- **Indian Traffic Signs**: Support for Mandatory, Cautionary, and Informatory traffic signs
- **Real-time Detection**: Live video processing with webcam or video file support
- **Priority-based Alerts**: Intelligent alert system based on sign importance
- **Low Visibility Enhancement**: CLAHE preprocessing for better detection in poor conditions
- **Model Export**: Support for ONNX, TensorRT, and other deployment formats
- **Modular Design**: Clean, well-documented code structure

## Dataset Structure

The system expects the following dataset structure:
```
Dataset/
├── Mandatory_Traffic_Signs/
│   ├── STOP/
│   ├── NO_ENTRY/
│   ├── SPEED_LIMIT_30/
│   └── ...
├── Cautionary_Traffic_Signs/
│   ├── BARRIER_AHEAD/
│   ├── CATTLE/
│   └── ...
└── Informatory_Traffic_Signs/
    ├── Destination_Sign/
    ├── Hospital/
    └── ...
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd traffic_sign_recognition
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import ultralytics; print('YOLO v11 installed successfully')"
```

## Usage

### 1. Dataset Preparation

Prepare your dataset for YOLO training:
```bash
python traffic_sign_recognition.py --mode prepare --dataset Dataset --output output
```

### 2. Model Training

Train the YOLO model on your dataset:
```bash
python traffic_sign_recognition.py --mode train --dataset Dataset --output output --epochs 100 --batch 16
```

**Training Parameters**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Input image size (default: 640)

### 3. Model Validation

Validate the trained model:
```bash
python traffic_sign_recognition.py --mode validate --model output/traffic_sign_model/weights/best.pt
```

### 4. Real-time Detection

Run real-time traffic sign detection:
```bash
# Using webcam
python traffic_sign_recognition.py --mode detect --model output/traffic_sign_model/weights/best.pt --source 0

# Using video file
python traffic_sign_recognition.py --mode detect --model output/traffic_sign_model/weights/best.pt --source video.mp4
```

**Detection Parameters**:
- `--conf`: Confidence threshold (default: 0.4)
- `--source`: Video source (0 for webcam, or path to video file)

### 5. Model Export

Export the model for deployment:
```bash
python traffic_sign_recognition.py --mode export --model output/traffic_sign_model/weights/best.pt --format onnx
```

**Export Formats**:
- `onnx`: ONNX format for cross-platform deployment
- `tensorrt`: TensorRT for NVIDIA GPUs
- `coreml`: CoreML for Apple devices
- `engine`: TensorRT engine file

## Priority System

The system implements a priority-based alert system:

- **Priority 10**: Critical signs (STOP, NO_ENTRY)
- **Priority 9**: High importance (GIVE_WAY)
- **Priority 8**: Speed limits
- **Priority 7**: Safety signs (SCHOOL_AHEAD, PEDESTRIAN_CROSSING)
- **Priority 6**: Parking restrictions
- **Priority 5**: Turn restrictions
- **Priority 4**: Vehicle restrictions
- **Priority 3**: Directional signs
- **Priority 2**: Warning signs

## Advanced Usage

### Custom Training Configuration

You can modify training parameters in the script:
```python
# In traffic_sign_recognition.py
results = self.model.train(
    data=str(self.output_path / "traffic_signs.yaml"),
    epochs=100,
    imgsz=640,
    batch=16,
    augment=True,
    patience=20,
    save=True,
    project=str(self.output_path),
    name="traffic_sign_model"
)
```

### Custom Priority Mapping

Modify the priority system in the `TrafficSignRecognition` class:
```python
self.class_priority_map = {
    "STOP": 10,
    "NO_ENTRY": 10,
    # Add your custom priorities
}
```

### Low Visibility Enhancement

The system includes automatic preprocessing for low visibility conditions:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- LAB color space conversion for better contrast
- Automatic brightness adjustment

## Performance Optimization

### For Training:
- Use GPU acceleration if available
- Adjust batch size based on your GPU memory
- Use smaller models (yolo11s) for faster training

### For Inference:
- Lower confidence threshold for more detections
- Use smaller input image sizes for faster processing
- Export to optimized formats (TensorRT, ONNX) for deployment

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**:
   - Reduce batch size: `--batch 8`
   - Use smaller image size: `--imgsz 416`

2. **Poor detection accuracy**:
   - Increase training epochs: `--epochs 200`
   - Lower confidence threshold: `--conf 0.3`
   - Check dataset quality and annotations

3. **Slow inference**:
   - Use smaller model (yolo11s instead of yolo11x)
   - Export to optimized format
   - Reduce input image size

### System Requirements:

- **Minimum**: 8GB RAM, CPU-only
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **OS**: Windows 10/11, Linux, macOS

## File Structure

```
traffic_sign_recognition/
├── traffic_sign_recognition.py    # Main script
├── requirements.txt               # Dependencies
├── README.md                     # This file
├── Dataset/                      # Your dataset
└── output/                       # Training outputs
    ├── train/
    ├── val/
    ├── traffic_signs.yaml
    └── traffic_sign_model/
        └── weights/
            └── best.pt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO v11
- Indian traffic sign dataset contributors
- OpenCV community for computer vision tools

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub
4. Contact the maintainers

---

**Note**: This system is designed for educational and research purposes. Always follow local traffic laws and regulations when using in real-world applications.

# Traffic Sign Recognition & Driver Alert System (India)

YOLO-based Indian traffic sign detection with real-time visual and multilingual audio alerts (Streamlit UI).

## 1) Environment

- Python 3.10–3.12
- Recommended: a GPU with CUDA for faster training (optional)

Install dependencies:
```
pip install -U pip
pip install ultralytics opencv-python albumentations pillow tqdm seaborn matplotlib pyyaml
pip install streamlit gTTS pydub playsound langcodes
```
Optional (for some augmentation warnings):
```
pip install scikit-image
```
Windows audio backend tip:
- For pydub autoplay, install ffmpeg and add it to PATH.

## 2) Dataset Layout

Place your dataset under `Dataset/`:
```
Dataset/
  Mandatory_Traffic_Signs/
    STOP/
      00001.jpg
      ...
  Cautionary_Traffic_Signs/
  Informatory_Traffic_Signs/
```

## 3) Quick Start (Train with defaults)

From the repo root:
```
python traffic_sign_recognition.py
```
This runs training with defaults:
- Prepares dataset into `output/{train,val,test}/`
- Trains YOLO (model folder: `output/traffic_sign_model/`)
- Saves best weights to `output/traffic_sign_model/weights/best.pt` and copies to `output/best_model.pt`
- Generates visualizations and confusion matrix under `output/visualizations/`

You can customize:
```
python traffic_sign_recognition.py --mode train --epochs 50 --batch 16 --imgsz 640 --device auto
```

## 4) Validate

```
python traffic_sign_recognition.py --mode validate --model output/traffic_sign_model/weights/best.pt
```
Outputs mAP and saves validation confusion matrix to `output/visualizations/`.

## 5) Real-time Detection

Webcam (0) or a video file:
```
python traffic_sign_recognition.py --mode detect --model output/best_model.pt --source 0 --conf 0.4
```

## 6) Export

Export to ONNX (example):
```
python traffic_sign_recognition.py --mode export --model output/best_model.pt --format onnx
```

## 7) Streamlit App (optional)

If you use the Streamlit UI (with audio notifications):
```
streamlit run streamlit_app.py
```
- Pre-generated audio files should be placed at `output/audio_alerts/<RAW_CLASS>_<lang>.mp3`
  e.g., `output/audio_alerts/GAP_IN_MEDIAN_en.mp3`, `..._hi.mp3`

## 8) Augmentation

Augmented images and reports are saved in:
- `output/augmented_data/`
- `output/augmentation_logs/preprocessing_log.txt` (records transforms applied)

## 9) Outputs and Visualizations

- Training results CSV: `output/traffic_sign_model/results.csv`
- Visual graphs and confusion matrix: `output/visualizations/`
- Best model: `output/traffic_sign_model/weights/best.pt` and `output/best_model.pt`

## 10) Common Issues

- ffmpeg not found: install ffmpeg and ensure it is in PATH for audio autoplay.
- Slow training on CPU: reduce `--imgsz 320` and `--batch 2`, or use a GPU (`--device 0`).