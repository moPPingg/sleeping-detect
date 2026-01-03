# Drowsiness Detection System

A computer vision project that detects drowsiness using face mesh detection and machine learning.

## Setup Instructions

### 1. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Project

#### Option A: Quick Run (if model exists)
```bash
python Runsystem.py
```

Or simply double-click `run.bat`

#### Option B: Train Model First (if model doesn't exist)

If you don't have `drowsiness_model.pkl`, you need to train it first:

1. **Prepare dataset** (if needed):
   ```bash
   python collect_data.py  # Processes video to create dataset_full.csv
   ```

2. **Train the model**:
   ```bash
   python trainmodel.py   # Trains model and saves to drowsiness_model.pkl
   ```
   
   OR compare models:
   ```bash
   python Comparemodel.py  # Compares different ML models
   ```

3. **Run the system**:
   ```bash
   python Runsystem.py
   ```

## Project Files

- **Runsystem.py** - Main application (runs drowsiness detection with webcam)
- **trainmodel.py** - Trains ML model from dataset.csv
- **Comparemodel.py** - Compares different ML models using dataset_full.csv
- **chongngugat.py** - Processes video to extract features and create dataset
- **FaceMeshModule.py** - Face mesh detection module
- **FaceMeshBasics.py** - Basic face mesh example

## Usage

1. Make sure your webcam is connected
2. Run `Runsystem.py`
3. The system will detect your face and monitor for drowsiness
4. Press 'q' to quit

## Requirements

- Python 3.8+
- Webcam
- All dependencies listed in `requirements.txt`

