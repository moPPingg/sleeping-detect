# 🚀 Quick Start Guide

## Get Started in 3 Minutes!

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the System

```bash
python drowsiness_detection_system.py
```

### Step 3: Use the System

1. **Calibration** (5 seconds): Look straight at camera
2. **Detection**: System monitors your drowsiness
3. **Press 'q'**: To quit

---

## 📚 Core Files

### Essential Files
- `drowsiness_detection_system.py` - Main application
- `face_mesh_detector.py` - Face detection module
- `drowsiness_detector.py` - Analysis module
- `drowsiness_model.pkl` - Trained model (99.51%)
- `scaler.pkl` - Data scaler

### Development Tools
- `data_collector.py` - Collect your own data
- `model_trainer.py` - Train new model
- `ml_models_comparison.ipynb` - Compare 6 algorithms
- `project_documentation.ipynb` - Full documentation

---

## ⚡ Quick Commands

```bash
# Run detection system
python drowsiness_detection_system.py

# Collect custom data
python data_collector.py

# Train new model
python model_trainer.py

# View ML comparison
jupyter notebook ml_models_comparison.ipynb

# View full documentation
jupyter notebook project_documentation.ipynb
```

---

## 🆘 Common Issues

### Camera Not Opening?
- Check camera permissions
- Try changing `WEBCAM_ID` in code (0, 1, 2, etc.)

### Model Not Found?
```bash
python model_trainer.py
```

### Need More Help?
- Read **README.md** for detailed troubleshooting
- Open **project_documentation.ipynb** for technical details

---

## 📊 What You Get

✅ Real-time drowsiness detection  
✅ 99.5% accuracy  
✅ Voice & visual alerts  
✅ Automatic calibration  
✅ 30+ FPS performance  

---

**Happy Coding! Stay Safe! 🚗**
