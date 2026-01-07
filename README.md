# AI Driver Monitoring System (DMS)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.51%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Introduction

An AI-powered real-time drowsiness detection system using computer vision and machine learning to monitor driver alertness and prevent accidents. This system achieves 99.51% accuracy using a Random Forest classifier trained on 478 facial landmarks extracted via MediaPipe Face Mesh.

---

## Key Features

- **Real-Time Detection**: Processes video at 30+ FPS with &lt;50ms latency on standard CPU
- **High Accuracy**: 99.51% classification accuracy with Random Forest model
- **Multi-State Detection**: Distinguishes between Awake, Drowsy, Looking Down (Phone), and Microsleep states
- **Smart Alert System**: Voice warnings (TTS) + visual overlay with intelligent timing
- **Auto-Calibration**: Adapts to individual drivers automatically in 5 seconds
- **Production-Ready**: No GPU required, runs on standard webcams

---

## System Architecture

### Pipeline Overview

```
Webcam Input (720p @ 30 FPS)
    ↓
MediaPipe Face Mesh (468 landmarks extraction)
    ↓
Feature Engineering (936-dimensional vector)
    ↓
Random Forest Classifier (100 trees, 99.51% accuracy)
    ↓
Temporal Smoothing (8-frame rolling average)
    ↓
Multi-Modal Alerts (Voice + Visual)
```

### Performance Specifications

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 99.51% | Tested on 611 samples (20% holdout) |
| **FPS** | 25-30 | On Intel i5/i7 CPU |
| **Latency** | &lt; 50ms | End-to-end pipeline |
| **Model Size** | &lt; 10MB | Lightweight deployment |
| **Hardware** | CPU-only | No GPU required |

---

## Installation

### Prerequisites

- Python 3.8 - 3.11
- Webcam (built-in or USB)
- Windows 10/11, macOS, or Linux

### Quick Setup

```bash
# Clone repository
git clone https://github.com/moPPingg/sleeping-detect.git
cd sleeping-detect

# Install dependencies
pip install -r requirements.txt

# Run the system
python drowsiness_detection_system.py
```

---

## Usage

### 1. Run Detection System

```bash
python drowsiness_detection_system.py
```

**Steps:**
1. **Calibration (5 seconds)**: Look straight at camera, eyes open
2. **Detection**: System monitors drowsiness in real-time
3. **Alerts**: Receive voice + visual warnings when drowsiness detected
4. **Exit**: Press `q` to quit

### 2. Collect Custom Data

```bash
python data_collector.py
```

**Instructions:**
- Hold keys while maintaining pose:
  - `0` = Awake (sit straight, eyes open)
  - `1` = Drowsy (close eyes)
  - `2` = Looking Down (head down)
  - `3` = Microsleep (head tilted, eyes closed)
- Collect 600-800 samples per class
- Press `q` to save and exit

### 3. Train New Model

```bash
python model_trainer.py
```

Trains Random Forest model on collected data and saves to `drowsiness_model.pkl`.

### 4. Generate Performance Charts

```bash
python charts.py
```

Creates professional visualizations:
- Confusion Matrix
- Feature Importance (Top 20 landmarks)
- Dataset Class Distribution

---

## Project Structure

```
sleeping-detect/
│
├── assets/
│   ├── Project_Report.html          # Comprehensive technical documentation
│   ├── confusion_matrix.png          # Model performance visualization
│   ├── data_balance.png              # Dataset distribution chart
│   └── feature_importance.png        # Feature importance analysis
│
├── Core Application
│   ├── drowsiness_detection_system.py   # Main application entry point
│   ├── face_mesh_detector.py            # MediaPipe wrapper
│   ├── drowsiness_detector.py           # Feature extraction & analysis
│   ├── glasses_detector.py              # Glasses detection module
│   ├── drowsiness_model.pkl             # Trained Random Forest model
│   ├── scaler.pkl                       # StandardScaler for normalization
│   └── face_landmarker.task             # MediaPipe model file
│
├── Development Tools
│   ├── data_collector.py                # Data collection utility
│   ├── model_trainer.py                 # Model training script
│   ├── charts.py                        # Visualization generator
│   └── face_data.csv                    # Training dataset
│
└── Documentation
    ├── README.md                        # This file
    ├── QUICK_START.md                   # 3-minute quick start guide
    └── requirements.txt                 # Python dependencies
```

---

## Model Performance

### Algorithm Comparison

We compared 6 machine learning algorithms and selected Random Forest for optimal performance:

| Model | Accuracy | F1-Score | Training Time | Inference Time |
|-------|----------|----------|---------------|----------------|
| **Random Forest** | **99.51%** | **0.9951** | 12.3s | ~5ms |
| XGBoost | 99.18% | 0.9918 | 45.7s | ~6ms |
| SVM RBF | 98.85% | 0.9885 | 189.2s | ~10ms |
| SVM Linear | 97.54% | 0.9754 | 23.1s | ~8ms |
| Logistic Regression | 96.89% | 0.9689 | 3.2s | ~3ms |
| KNN | 95.74% | 0.9574 | 0.5s | ~15ms |

**Winner:** Random Forest - Best balance of accuracy, speed, and robustness.

### Dataset Statistics

- **Total Samples**: 3,052
- **Features**: 936 (468 landmarks × 2 coordinates)
- **Classes**: 4 (balanced distribution: ~750 samples each)
- **Train/Test Split**: 80/20 with stratification

---

## Full Documentation

For a comprehensive deep dive into the system architecture, mathematical methodology, feature engineering techniques, and future roadmap, please refer to the:

**[📖 View Detailed Project Report](https://htmlpreview.github.io/?https://github.com/moPPingg/sleeping-detect/blob/master/assets/Project_Report.html)**

*(HTML report rendered via htmlpreview.github.io)*

This HTML report includes:
- Complete ML workflow and methodology
- 6 algorithm comparison with detailed metrics
- Confusion matrix and feature importance analysis
- System architecture and data pipeline visualization
- Deployment guide and troubleshooting
- Future improvements roadmap

---

## Troubleshooting

### Camera Not Opening

- Check camera permissions in OS settings
- Try different `WEBCAM_ID` (0, 1, 2) in code
- Close other applications using camera

### Low FPS Performance

- Reduce video resolution in code
- Increase `SKIP_FRAMES` parameter
- Close background applications

### No Voice Alerts

- Check audio output device
- Verify speakers/headphones connected
- Reinstall: `pip install --upgrade pyttsx3`

### Model Not Found

```bash
python model_trainer.py
```

---

## Technologies

### Core Stack

- **Computer Vision**: OpenCV 4.8, MediaPipe 0.10.9
- **Machine Learning**: scikit-learn 1.3, XGBoost 2.0
- **Audio**: pyttsx3 (Text-to-Speech)
- **Data Analysis**: pandas 2.0, numpy 1.24
- **Visualization**: matplotlib 3.7, seaborn 0.12

### Key Algorithms

- **Face Detection**: MediaPipe Face Mesh (Google)
- **Feature Extraction**: EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), Head Pose
- **Classification**: Random Forest Classifier (100 trees)
- **Normalization**: StandardScaler
- **Post-Processing**: Temporal smoothing (8-frame window)

---

## Future Enhancements

### Short Term (1-3 months)
- Glasses/Sunglasses detection support
- Multi-face tracking for carpooling
- Session recording & analytics dashboard

### Medium Term (3-6 months)
- Deep learning models (CNN + LSTM)
- Low-light/night mode enhancement
- Multi-language voice alerts

### Long Term (6-12 months)
- Steering wheel grip sensor integration
- Lane departure correlation analysis
- Cloud-based fleet management platform
- iOS/Android native mobile apps

---

## Contributing

Contributions welcome! Areas to contribute:

- Bug fixes and performance optimizations
- New features and enhancements
- Documentation improvements
- Testing and validation
- Multi-language support

### How to Contribute

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

**moPPingg**

- GitHub: [@moPPingg](https://github.com/moPPingg)
- Project Repository: [sleeping-detect](https://github.com/moPPingg/sleeping-detect)
- Documentation: [Project Report](assets/Project_Report.html)

---

## Acknowledgments

- [OpenCV](https://opencv.org/) - Computer vision library
- [MediaPipe](https://google.github.io/mediapipe/) - Face mesh by Google
- [scikit-learn](https://scikit-learn.org/) - Machine learning toolkit
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting library

---

## Project Stats

- **Lines of Code**: 2,000+
- **Training Samples**: 3,052
- **Model Accuracy**: 99.51%
- **Processing Speed**: 30+ FPS
- **Models Compared**: 6
- **Technologies Used**: 10+

---

<div align="center">

**Stay Safe on the Roads! Drive Smart.**

**Made with Python • OpenCV • MediaPipe**

*Last Updated: January 2026*

[⬆ Back to Top](#ai-driver-monitoring-system-dms)

</div>
