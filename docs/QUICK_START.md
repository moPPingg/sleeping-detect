# Quick Start Guide

**Get the Driver Monitoring System running in 3 minutes**

---

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the System

```bash
python drowsiness_detection_system.py
```

### Step 3: Use the System

1. **Calibration Phase (5 seconds)**: Look straight at the camera with eyes open
2. **Detection Active**: System monitors drowsiness in real-time
3. **Exit**: Press 'q' to quit the application

---

## Core Files Overview

### Production Files

| File | Purpose |
|------|---------|
| `drowsiness_detection_system.py` | Main application entry point |
| `face_mesh_detector.py` | MediaPipe face detection wrapper |
| `drowsiness_detector.py` | Feature extraction and analysis |
| `drowsiness_model.pkl` | Trained Random Forest model (99.51% accuracy) |
| `scaler.pkl` | StandardScaler for feature normalization |

### Development Tools

| File | Purpose |
|------|---------|
| `data_collector.py` | Custom data collection utility |
| `model_trainer.py` | Model training script |
| `charts.py` | Performance visualization generator |

---

## Essential Commands

### Run Detection System
```bash
python drowsiness_detection_system.py
```

### Collect Custom Data
```bash
python data_collector.py
# Hold keys 0-3 for different states
# Press 'q' to save and exit
```

### Train New Model
```bash
python model_trainer.py
```

### Generate Performance Charts
```bash
python charts.py
```

---

## Troubleshooting

### Camera Not Opening

**Solution:**
- Verify camera permissions in OS settings
- Try different `WEBCAM_ID` values (0, 1, 2) in the code
- Close other applications using the camera
- Test camera in other applications first

### Model File Not Found

**Solution:**
```bash
python model_trainer.py
```
This will train a new model and save it to `drowsiness_model.pkl`.

### Low FPS Performance

**Solution:**
- Reduce video resolution in code settings
- Increase `SKIP_FRAMES` parameter
- Close background applications
- Consider using a lighter model

### No Voice Alerts

**Solution:**
- Check audio output device settings
- Verify speakers/headphones are connected
- Reinstall pyttsx3: `pip install --upgrade pyttsx3`

---

## System Features

| Feature | Specification |
|---------|--------------|
| **Accuracy** | 99.51% on test set |
| **Performance** | 30+ FPS real-time processing |
| **Latency** | < 50ms per frame |
| **States Detected** | Awake, Drowsy, Looking Down, Microsleep |
| **Alert System** | Voice warnings + visual overlay |
| **Hardware** | Standard CPU (no GPU required) |

---

## Next Steps

For detailed documentation:
- **README.md** - Complete project documentation
- **Project_Report.html** - Comprehensive technical report with visualizations
- Review source code for customization options

---

**Project Repository:** [https://github.com/moPPingg/sleeping-detect](https://github.com/moPPingg/sleeping-detect)
