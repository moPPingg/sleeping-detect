"""
TRAIN MODEL - TIME-SERIES DRIVER DROWSINESS DETECTION
Trains and compares 6 different ML algorithms on time-series features
to find the best model for Driver Drowsiness Detection.

Dataset: sequence_data.csv (Time-Series Statistical Features)
Features: mean_ear, std_ear, mean_mar, max_mar, mean_pitch, std_pitch, 
          mean_yaw, std_yaw, mean_roll, std_roll
"""

import pathlib
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "sequence_data.csv"
MODEL_PATH = ROOT / "models" / "drowsiness_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

# Label mapping (English)
LABEL_NAMES = {
    0: 'Awake',
    1: 'Drowsy', 
    2: 'Phone',
    3: 'Microsleep'
}

# Feature columns (time-series statistical features)
FEATURE_COLUMNS = [
    'mean_ear', 'std_ear',
    'mean_mar', 'max_mar',
    'mean_pitch', 'std_pitch',
    'mean_yaw', 'std_yaw',
    'mean_roll', 'std_roll'
]

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
def get_models():
    """
    Returns a dictionary of models to train and evaluate.
    """
    return {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            random_state=42,
            probability=True
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    }

# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================
def load_and_clean_data():
    """
    Load sequence_data.csv and clean it.
    
    Returns:
        X: Feature matrix
        y: Target labels
    """
    print("\n[1] Loading data...")
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"    [OK] Loaded: {len(df)} samples, {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"    [ERROR] Data file not found: {DATA_FILE}")
        raise
    
    # Check for required columns
    required_cols = FEATURE_COLUMNS + ['label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"    [ERROR] Missing required columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Extract features and labels
    print("\n[2] Extracting features and labels...")
    X = df[FEATURE_COLUMNS].copy()
    y = df['label'].copy()
    
    print(f"    Features: {len(FEATURE_COLUMNS)} columns")
    print(f"    Labels: {len(y)} samples")
    
    # CRITICAL: Check for NaN and Infinity values
    print("\n[3] Checking for NaN and Infinity values...")
    
    # Check for NaN
    nan_mask = X.isnull().any(axis=1) | y.isnull()
    nan_count = nan_mask.sum()
    if nan_count > 0:
        print(f"    [WARN] Found {nan_count} rows with NaN values, dropping...")
        X = X[~nan_mask]
        y = y[~nan_mask]
    
    # Check for Infinity
    inf_mask = np.isinf(X).any(axis=1)
    inf_count = inf_mask.sum()
    if inf_count > 0:
        print(f"    [WARN] Found {inf_count} rows with Infinity values, dropping...")
        X = X[~inf_mask]
        y = y[~inf_mask]
    
    # Final check
    if X.isnull().any().any() or np.isinf(X).any().any():
        print(f"    [ERROR] Still found NaN/Inf values after cleaning!")
        raise ValueError("Data cleaning failed")
    
    print(f"    [OK] Clean data: {len(X)} samples remaining")
    
    # Check data distribution
    print("\n[4] Checking data distribution:")
    for label in sorted(y.unique()):
        count = (y == label).sum()
        percentage = count / len(y) * 100
        print(f"    Label {label} ({LABEL_NAMES.get(label, 'Unknown')}): "
              f"{count} samples ({percentage:.1f}%)")
    
    return X.values, y.values

# ============================================================================
# MODEL EVALUATION
# ============================================================================
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    
    Returns:
        Dictionary containing accuracy and f1_score (macro)
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'F1-Score (Macro)': f1_macro
    }

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main():
    print("="*80)
    print("TRAIN MODEL - TIME-SERIES DRIVER DROWSINESS DETECTION")
    print("Multi-Algorithm Comparison & Selection")
    print("="*80)
    
    # ------------------------------------------------------------------------
    # Step 1-4: Load and Clean Data
    # ------------------------------------------------------------------------
    X, y = load_and_clean_data()
    
    # ------------------------------------------------------------------------
    # Step 5: Train/Test Split (80/20, Stratified)
    # ------------------------------------------------------------------------
    print("\n[5] Splitting data (80% train / 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"    [OK] Training set: {len(X_train)} samples")
    print(f"    [OK] Test set: {len(X_test)} samples")
    
    # ------------------------------------------------------------------------
    # Step 6: Feature Scaling
    # ------------------------------------------------------------------------
    print("\n[6] Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("    [OK] Features normalized (mean=0, std=1)")
    print("    [INFO] Scaler will be saved for real-time inference")
    
    # ------------------------------------------------------------------------
    # Step 7: Train and Evaluate All Models
    # ------------------------------------------------------------------------
    print("\n[7] Training and evaluating 6 models...")
    print("="*80)
    
    models = get_models()
    results = []
    trained_models = {}
    
    # Models that require scaling: SVM, KNN
    models_requiring_scaling = ['SVM', 'KNN']
    
    for model_name, model in models.items():
        print(f"\n    Training {model_name}...", end=" ")
        
        try:
            # Use scaled data for distance-based models
            if model_name in models_requiring_scaling:
                model.fit(X_train_scaled, y_train)
                metrics = evaluate_model(model, X_test_scaled, y_test, model_name)
                trained_models[model_name] = {
                    'model': model,
                    'X_test': X_test_scaled,
                    'requires_scaling': True
                }
            else:
                # Use original data for tree-based and linear models
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test, model_name)
                trained_models[model_name] = {
                    'model': model,
                    'X_test': X_test,
                    'requires_scaling': False
                }
            
            results.append(metrics)
            print(f"✓ Accuracy: {metrics['Accuracy']:.4f}, F1-Score: {metrics['F1-Score (Macro)']:.4f}")
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue
    
    # ------------------------------------------------------------------------
    # Step 8: Create Comparison Table
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("MODEL COMPARISON LEADERBOARD")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    
    # Sort by F1-Score (Macro) - this is our primary metric
    results_df = results_df.sort_values('F1-Score (Macro)', ascending=False)
    results_df = results_df.reset_index(drop=True)
    
    # Display formatted table
    print("\n" + results_df.to_string(index=False))
    print("\n" + "="*80)
    
    # Find winner based on F1-Score (Macro)
    winner_name = results_df.iloc[0]['Model']
    winner_f1 = results_df.iloc[0]['F1-Score (Macro)']
    winner_accuracy = results_df.iloc[0]['Accuracy']
    
    print(f"\n[WINNER] {winner_name}")
    print(f"         F1-Score (Macro): {winner_f1:.4f} ({winner_f1*100:.2f}%)")
    print(f"         Accuracy: {winner_accuracy:.4f} ({winner_accuracy*100:.2f}%)")
    
    # ------------------------------------------------------------------------
    # Step 9: Save Winner Model and Scaler
    # ------------------------------------------------------------------------
    print("\n[8] Saving winner model and scaler...")
    
    winner_data = trained_models[winner_name]
    winner_model = winner_data['model']
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(winner_model, f)
    print(f"    [OK] Model saved: {MODEL_PATH}")
    
    # Save scaler (ALWAYS save it, even if model doesn't require it)
    # This ensures consistency and allows switching models later
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"    [OK] Scaler saved: {SCALER_PATH}")
    
    if winner_data['requires_scaling']:
        print(f"    [INFO] This model REQUIRES StandardScaler for inference")
    else:
        print(f"    [INFO] This model does not require scaling, but scaler is saved for consistency")
    
    # ------------------------------------------------------------------------
    # Step 10: Detailed Report for Winner
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print(f"DETAILED REPORT: {winner_name}")
    print("="*80)
    
    # Get predictions
    if winner_data['requires_scaling']:
        y_pred = winner_model.predict(X_test_scaled)
    else:
        y_pred = winner_model.predict(X_test)
    
    # Classification Report
    print("\n[Classification Report]")
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())],
        zero_division=0
    ))
    
    # Confusion Matrix
    print("\n[Confusion Matrix]")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())],
        columns=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
    )
    print(cm_df)
    
    # Per-class metrics
    print("\n[Per-Class Metrics]")
    for i, label_name in enumerate([LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {label_name:15} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # ------------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Model: {winner_name}")
    print(f"F1-Score (Macro): {winner_f1:.4f} ({winner_f1*100:.2f}%)")
    print(f"Accuracy: {winner_accuracy:.4f} ({winner_accuracy*100:.2f}%)")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print("="*80)

if __name__ == "__main__":
    main()
