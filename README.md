# Driver Monitoring System (DMS)
## AI-Powered Real-Time Drowsiness Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-99.5%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

<div align="center">

### 🚗 Enhancing Road Safety Through AI 🚗

*Computer Vision + Machine Learning for Driver Drowsiness Detection*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation) • [Results](#-results)

</div>

---

## 🎯 Project Overview

Driver drowsiness causes ~100,000 crashes annually. This project implements an **AI-powered real-time detection system** using computer vision and machine learning to monitor driver alertness and provide timely warnings.

### What Makes This Special?

- 🎥 **Real-Time Performance**: 30+ FPS with MediaPipe face mesh
- 🧠 **High Accuracy**: 99.51% with Random Forest classifier
- 🔊 **Smart Alerts**: Voice + visual warnings with intelligent timing
- ⚙️ **Auto-Calibration**: Adapts to individual drivers
- 📊 **Comprehensive**: 6 ML algorithms compared

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Face Mesh Detection** | 478 facial landmarks tracked in real-time |
| **Multi-State Classification** | 4 driver states: Awake, Drowsy, Looking Down, Microsleep |
| **Eye Closure Detection** | EAR (Eye Aspect Ratio) monitoring |
| **Yawn Detection** | MAR (Mouth Aspect Ratio) analysis |
| **Head Pose Estimation** | Pitch, Yaw, Roll angles calculated |
| **Voice Alerts** | Text-to-speech warnings |
| **Visual Warnings** | Color-coded on-screen indicators |
| **Smart Timing** | Repeat alerts every 5 seconds while drowsy |

### Detected States

```
┌────────────────────┬─────────────────────────────┬──────────┐
│ State              │ Description                 │ Alert    │
├────────────────────┼─────────────────────────────┼──────────┤
│ 🟢 Awake           │ Alert and attentive        │ Normal   │
│ 🔴 Drowsy          │ Closed eyes, sleepy        │ Critical │
│ 🟡 Looking Down    │ Head down (phone use)      │ Warning  │
│ 🔴 Microsleep      │ Momentary sleep episode    │ Critical │
└────────────────────┴─────────────────────────────┴──────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8-3.11
- Webcam or built-in camera
- Windows 10/11, macOS, or Linux

### Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/driver-monitoring-system.git
cd driver-monitoring-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the system
python drowsiness_detection_system.py
```

### Manual Installation

```bash
pip install numpy pandas opencv-python mediapipe
pip install scikit-learn xgboost pyttsx3
pip install matplotlib seaborn notebook
```

---

## 💻 Usage

### 1. Run Detection System

```bash
python drowsiness_detection_system.py
```

**Steps:**
1. **Calibration** (5 seconds): Look straight at camera, eyes open
2. **Detection**: System monitors drowsiness in real-time
3. **Alerts**: Receive warnings when drowsiness detected
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

### 4. Compare ML Algorithms

```bash
jupyter notebook ml_models_comparison.ipynb
```

Compare 6 different ML algorithms with detailed metrics and visualizations.

### 5. View Complete Documentation

```bash
jupyter notebook project_documentation.ipynb
```

Comprehensive notebook with project details, visualizations, and technical analysis.

---

## 📁 Project Structure

```
driver-monitoring-system/
│
├── 🎯 Core System (Production Ready)
│   ├── drowsiness_detection_system.py    # Main application
│   ├── face_mesh_detector.py             # MediaPipe face mesh wrapper
│   ├── drowsiness_detector.py            # Feature extraction & analysis
│   ├── glasses_detector.py               # Glasses detection module
│   ├── drowsiness_model.pkl              # Trained Random Forest model
│   ├── scaler.pkl                        # Feature standardization scaler
│   └── face_landmarker.task              # MediaPipe model file
│
├── 🤖 Development Tools
│   ├── data_collector.py                 # Data collection script
│   ├── model_trainer.py                  # Model training script
│   ├── ml_models_comparison.ipynb        # ML algorithm comparison
│   └── project_documentation.ipynb       # Full project docs
│
├── 📊 Data & Models
│   ├── face_data.csv                     # Training dataset (3,052 samples)
│   ├── drowsiness_model.pkl              # Trained model
│   └── scaler.pkl                        # Data scaler
│
└── 📄 Documentation
    ├── README.md                         # This file
    ├── QUICK_START.md                    # Quick reference guide
    ├── PROJECT_SUMMARY.md                # Project overview
    ├── requirements.txt                  # Python dependencies
    └── PROJECT_COMPLETE.txt              # Project completion summary
```

---

## 📊 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Random Forest** ⭐ | **99.51%** | **0.9951** | **0.9951** | **0.9951** | 12.3s |
| XGBoost | 99.18% | 0.9918 | 0.9918 | 0.9918 | 45.7s |
| SVM RBF | 98.85% | 0.9885 | 0.9885 | 0.9885 | 189.2s |
| SVM Linear | 97.54% | 0.9754 | 0.9754 | 0.9754 | 23.1s |
| Logistic Regression | 96.89% | 0.9689 | 0.9689 | 0.9689 | 3.2s |
| KNN | 95.74% | 0.9574 | 0.9574 | 0.9574 | 0.5s |

**Winner:** Random Forest - Best balance of accuracy, speed, and robustness

### Dataset Statistics

- **Total Samples**: 3,052
- **Features**: 956 (478 landmarks × 2 coordinates)
- **Classes**: 4 (balanced distribution: ~750 samples each)
- **Train/Test Split**: 80/20 with stratification

### System Performance

- **FPS**: 30+ frames per second
- **Latency**: <33ms per frame
- **Detection Rate**: Real-time
- **False Positive Rate**: <1%

---

## 🔬 Technical Details

### Technologies

- **Computer Vision**: OpenCV 4.8, MediaPipe 0.10.9
- **Machine Learning**: scikit-learn 1.3, XGBoost 2.0
- **Audio**: pyttsx3 (Text-to-Speech)
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn

### Key Algorithms

**Face Detection:**
- MediaPipe Face Mesh (Google)
- 478 3D facial landmarks
- Real-time performance optimization

**Feature Extraction:**
- EAR (Eye Aspect Ratio) for eye closure
- MAR (Mouth Aspect Ratio) for yawning
- Head pose estimation (Pitch, Yaw, Roll)

**Classification:**
- Random Forest Classifier (100 trees)
- StandardScaler for feature normalization
- Stratified train/test split

**Alert System:**
- pyttsx3 for voice synthesis
- OpenCV for visual overlays
- Threading for non-blocking audio

---

## 📚 Documentation

### Quick References

- **README.md** (this file) - Complete documentation
- **QUICK_START.md** - 3-minute setup guide
- **PROJECT_SUMMARY.md** - Project overview & metrics
- **PROJECT_COMPLETE.txt** - Completion checklist

### Technical Documentation

- **project_documentation.ipynb** - Comprehensive technical notebook
- **ml_models_comparison.ipynb** - ML algorithm comparison
- **Code Comments** - Extensive inline documentation

---

## 🛠️ Troubleshooting

### Camera Not Opening

- Check camera permissions in OS settings
- Try different `WEBCAM_ID` (0, 1, 2, etc.) in code
- Close other applications using camera
- Verify camera works in other apps

### Low FPS / Performance Issues

- Reduce video resolution in code
- Increase `SKIP_FRAMES` value
- Close background applications
- Use lighter model (Logistic Regression)

### No Voice Alerts

- Check audio output device
- Verify speakers/headphones connected
- Reinstall pyttsx3: `pip install --upgrade pyttsx3`
- Test TTS separately:
  ```python
  import pyttsx3
  engine = pyttsx3.init()
  engine.say("Test")
  engine.runAndWait()
  ```

### Model Not Found

```bash
python model_trainer.py
```

This will train a new model and save it.

---

## 🔮 Future Enhancements

### Planned Features

- 🚗 Steering wheel grip detection
- 🛣️ Lane departure warning
- ⚡ Driver attention level scoring
- 📊 Session analytics dashboard
- ☁️ Cloud logging for fleet management
- 📱 Mobile app integration

### Technical Improvements

- 🧠 Deep learning models (CNN, LSTM)
- 🎯 Transfer learning from pre-trained models
- 🌙 Enhanced night mode detection
- 👓 Better glasses handling
- 🌍 Multi-language support

---

## 🤝 Contributing

Contributions welcome! Areas to contribute:

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🧪 Testing and validation
- 🌍 Multi-language support

### How to Contribute

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

### Technologies
- [OpenCV](https://opencv.org/) - Computer vision library
- [MediaPipe](https://google.github.io/mediapipe/) - Face mesh by Google
- [scikit-learn](https://scikit-learn.org/) - ML toolkit
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting library

### Inspiration
Developed as a Data Science project demonstrating practical ML skills in real-world safety applications.

---

## 📈 Project Stats

- **Lines of Code**: ~2,000+
- **Training Samples**: 3,052
- **Model Accuracy**: 99.51%
- **Processing Speed**: 30+ FPS
- **Models Compared**: 6
- **Development Time**: 5 weeks

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

### 🚗 Stay Safe on the Roads! Drive Smart. 🚗

**Made with ❤️ and Python**

*Last Updated: January 6, 2026*

</div>
