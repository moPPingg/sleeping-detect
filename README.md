# Sleep State Detection & Driver Drowsiness Alert System

## Project Overview
This is a personal computer vision learning project for classifying facial states such as `Awake`, `Sleep/Drowsy`, and `Microsleep`. The project combines deep learning-based feature extraction with traditional machine learning classifiers: EfficientNet-B2 is used as a pre-trained feature extractor, and the extracted feature vectors are used to train SVM, Random Forest, KNN, and a voting ensemble.

## Problem Statement
Detecting drowsiness and microsleep can be useful for driver safety and alert systems because even short lapses in attention can increase accident risk. This repository explores a beginner-friendly computer vision workflow for recognizing fatigue-related facial states and testing alert behavior with a webcam demo.

## Approach
1. Prepare facial image sequence data from `data/train` and `data/test`.
2. Use EfficientNet-B2 as a pre-trained feature extractor.
3. Convert image sequences into feature vectors.
4. Train SVM, Random Forest, and KNN classifiers.
5. Build a voting ensemble from the trained base models.
6. Evaluate performance using cross-validation, internal holdout testing, independent testing, and confusion matrices.
7. Optionally run a webcam-based alert demo.

## Model Pipeline
```text
Input Facial Image
-> Image Preprocessing
-> EfficientNet-B2 Feature Extractor
-> Feature Vector
-> SVM / Random Forest / KNN / Voting Ensemble
-> Predicted State: Awake / Sleep-Drowsy / Microsleep
-> Optional Alert Demo
```

## Repository Structure
```text
sleep-state-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── CHANGES.md
├── docs/
│   └── limitations.md
├── src/
│   ├── extract_features.py
│   ├── train_models.py
│   ├── evaluate_model.py
│   ├── build_voting_ensemble.py
│   ├── webcam_alert_demo.py
│   ├── check_gpu.py
│   ├── data_summary.py
│   └── check_results.py
├── reports/
│   ├── data_summary_report.csv
│   ├── train_output/
│   ├── voting_output/
│   └── test_output/
├── features/
│   └── .gitkeep
└── screenshots/
    ├── train_confusion_matrix.png
    ├── ensemble_confusion_matrix.png
    ├── independent_test_confusion_matrix.png
    └── .gitkeep
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

### 3. Prepare the dataset layout
The scripts expect local data under:

- `data/train`
- `data/test`

The `data/` folder is intentionally ignored by Git. `src/extract_features.py` can also detect nested variants inside those folders, such as augmented or cropped subsets.

### 4. Generate a data summary report
```powershell
python src/data_summary.py
```

### 5. Run feature extraction
```powershell
python src/extract_features.py
```

This generates feature arrays in `features/`.

### 6. Train the base classifiers
```powershell
python src/train_models.py
```

This writes training reports and saved model artifacts to `reports/train_output/`.

### 7. Build the voting ensemble
```powershell
python src/build_voting_ensemble.py
```

This writes the ensemble confusion matrix and final ensemble artifact to `reports/voting_output/`.

### 8. Run independent test evaluation
```powershell
python src/evaluate_model.py
```

This writes the independent test report and confusion matrix to `reports/test_output/`.

### 9. Run the webcam alert demo
```powershell
python src/webcam_alert_demo.py
```

Notes:
- The webcam demo expects `reports/voting_output/final_ensemble_model.joblib` to exist locally first.
- `src/webcam_alert_demo.py` uses `winsound`, so the audible alert portion is Windows-specific.

## Results
Real metrics from the current repository are summarized below.

Cross-validation results:
- SVM `f1_macro`: `0.9225`
- Random Forest `f1_macro`: `0.8793`
- KNN `f1_macro`: `0.8430`

Internal holdout evaluation:
- Accuracy: `0.9362`
- Macro F1: `0.9359`

Independent test evaluation:
- Accuracy: `0.6569`
- Macro F1: `0.6465`

Data summary:
- Train total: `3636`
- Test total: `277`

The independent test result is lower than the internal holdout result, which suggests a generalization gap between the training distribution and separate test data. This is an important limitation and future improvement area.

Reference artifacts:
- `reports/train_output/gridsearch_results.csv`
- `reports/train_output/test_report.csv`
- `reports/test_output/independent_test_report.csv`
- `screenshots/train_confusion_matrix.png`
- `screenshots/ensemble_confusion_matrix.png`
- `screenshots/independent_test_confusion_matrix.png`

## Learning Outcomes
- Transfer learning with a pre-trained EfficientNet-B2 backbone
- Feature extraction with EfficientNet-B2 for sequence-based facial state classification
- Using traditional ML classifiers with deep feature vectors
- Comparing models using confusion matrices and classification metrics
- Building a basic real-time webcam alert demo for drowsiness detection

## Limitations
- This is a personal learning project and is not production-ready.
- Performance may vary depending on dataset quality, class balance, and label consistency.
- Real-world lighting, pose, occlusion, and camera placement can reduce reliability.
- The lower independent test score suggests limited generalization to separate data distributions.
- The webcam demo is intended for testing and experimentation only.

More detailed notes are available in `docs/limitations.md`.

## Future Improvements
- Improve dataset quality and class balance
- Test under more real-world lighting and camera conditions
- Add stronger temporal smoothing for video predictions
- Compare against end-to-end CNN fine-tuning
- Improve alert logic and false-positive handling

## CV Summary
Sleep State Detection & Driver Drowsiness Alert System
- Built a computer vision classification workflow for Awake, Sleep/Drowsy, and Microsleep facial states.
- Used EfficientNet-B2 as a pre-trained feature extractor and trained SVM, Random Forest, KNN, and voting ensemble classifiers.
- Evaluated performance with confusion matrices, cross-validation, internal holdout testing, and independent testing.
- Achieved 93.62% internal holdout accuracy and 65.69% independent test accuracy, highlighting real-world generalization challenges.
- Built a webcam-based alert demo for testing driver drowsiness detection behavior.
