# Sleep State Detection & Driver Drowsiness Alert System

## Project Overview
This is a personal computer vision learning project focused on classifying facial states such as `Awake`, `Sleep/Drowsy`, and `Microsleep`. The workflow combines deep learning-based feature extraction with traditional machine learning classifiers: EfficientNet-B2 is used as a pre-trained feature extractor, and the extracted feature vectors are used to train classifiers such as SVM, Random Forest, KNN, and an ensemble voting model.

## Problem Statement
Detecting drowsiness and microsleep can be useful for driver safety systems because short lapses in attention can lead to delayed reactions and dangerous driving behavior. This project explores a simple learning pipeline for recognizing fatigue-related facial states and testing alert behavior in a webcam demo.

## Approach
1. Prepare facial image sequence data from train/test folders.
2. Use EfficientNet-B2 as a pre-trained feature extractor.
3. Convert image sequences into feature vectors.
4. Train classifiers such as SVM, Random Forest, and KNN.
5. Evaluate models using confusion matrices and classification metrics.
6. Optionally test a webcam-based demo for alert behavior.

## Model Pipeline
```text
Input Facial Image
-> Image Preprocessing
-> EfficientNet-B2 Feature Extractor
-> Feature Vector
-> SVM / Random Forest / KNN Classifier
-> Predicted State: Awake / Sleep-Drowsy / Microsleep
-> Optional Alert Demo
```

## Repository Structure
```text
Demo1/
├── README.md
├── requirements.txt
├── .gitignore
├── bone_final.py
├── check_gpu.py
├── check_result.py
├── count_data.py
├── extract_features.py
├── test_model.py
├── train.py
├── voting.py
├── data_summary_report.csv
├── CHANGES.md
├── features/              # generated feature arrays (.npy)
├── train_output/          # training reports, confusion matrix, saved models
├── voting_output/         # ensemble model and confusion matrix
├── test_output/           # independent test report, confusion matrix, sorted frames
└── __pycache__/
```

## How to Run
### 1. Create a virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Prepare the expected dataset layout
The scripts expect data under:

- `data/train`
- `data/test`

`extract_features.py` can also detect nested variants inside those folders, such as augmented or cropped subsets.

### 4. Run feature extraction
```powershell
python extract_features.py
```

This generates feature arrays in `features/`.

### 5. Train the base classifiers
```powershell
python train.py
```

This writes reports and saved models to `train_output/`.

### 6. Build and evaluate the voting ensemble
```powershell
python voting.py
```

This writes the final ensemble model and confusion matrix to `voting_output/`.

### 7. Run independent test evaluation
```powershell
python test_model.py
```

This writes the independent test report and confusion matrix to `test_output/`.

### 8. Run the webcam alert demo
```powershell
python bone_final.py
```

Notes:
- The webcam demo expects `voting_output/final_ensemble_model.joblib` to exist first.
- `bone_final.py` uses `winsound`, so the audible alert portion is Windows-specific.

## Results
Real result files are included in this repository and are summarized below.

- `train_output/gridsearch_results.csv` shows 5-fold cross-validation `f1_macro` scores for the base models:
  - SVM: `0.9225`
  - Random Forest: `0.8793`
  - KNN: `0.8430`
- `train_output/test_report.csv` reports the internal holdout evaluation used after training:
  - Accuracy: `0.9362`
  - Macro F1: `0.9359`
- `test_output/independent_test_report.csv` reports the separate independent test evaluation:
  - Accuracy: `0.6569`
  - Macro F1: `0.6465`
- `data_summary_report.csv` reports:
  - Train total in the summary file: `3636`
  - Test total in the summary file: `277`

Reference artifacts:
- `train_output/confusion_matrix.png`
- `voting_output/ensemble_confusion_matrix.png`
- `test_output/independent_test_confusion_matrix.png`

## Learning Outcomes
- Transfer learning with a pre-trained EfficientNet-B2 backbone
- Feature extraction from image sequences
- Using traditional ML classifiers with deep feature vectors
- Comparing models with confusion matrices and classification metrics
- Building a basic real-time computer vision alert demo

## Limitations
- This is a personal learning project and is not production-ready.
- Performance may vary depending on dataset quality, class balance, and label consistency.
- Real-world lighting, pose, occlusion, and camera placement can reduce reliability.
- The webcam demo is intended for testing and experimentation only.

## Future Improvements
- Improve dataset quality and class balance
- Test under more real-world lighting and camera conditions
- Add stronger temporal smoothing for video predictions
- Compare against end-to-end CNN fine-tuning
- Improve alert logic and false-positive handling

## CV Summary
Sleep State Detection & Driver Drowsiness Alert System
- Developed a computer vision project for classifying Awake, Sleep/Drowsy, and Microsleep states from facial images.
- Used EfficientNet-B2 as a pre-trained feature extractor and trained SVM, Random Forest, and KNN models for classification.
- Compared model performance using confusion matrices and classification metrics.
- Built or prepared a simple webcam-based demo to test alert behavior.
