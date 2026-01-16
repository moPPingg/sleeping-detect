"""
Professional Data Visualization Script for Driver Drowsiness Detection System
Generates publication-quality charts for model evaluation and reporting.

Charts Generated:
1. Algorithm Comparison Bar Chart (F1-Score Macro Average)
2. Normalized Confusion Matrix Heatmap (Recall-based)
3. Feature Importance Horizontal Bar Chart (Top 10 Features)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "sequence_data.csv"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# Class labels
CLASS_LABELS = ['Awake', 'Drowsy', 'Phone', 'Microsleep']

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
sns.set_style("whitegrid")
sns.set_palette("husl")

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_models():
    """Returns dictionary of models to train and evaluate."""
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

# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_and_prepare_data():
    """Load data, handle missing values, and prepare features."""
    print("[1] Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"    Loaded: {len(df)} samples, {len(df.columns)} columns")
    
    # Drop non-feature columns
    feature_columns = [
        'mean_ear', 'std_ear', 'mean_mar', 'max_mar',
        'mean_pitch', 'std_pitch', 'mean_yaw', 'std_yaw',
        'mean_roll', 'std_roll'
    ]
    
    # Check for missing values
    missing = df[feature_columns + ['label']].isnull().sum()
    if missing.sum() > 0:
        print(f"    [WARN] Found {missing.sum()} missing values, dropping...")
        df = df.dropna(subset=feature_columns + ['label'])
        print(f"    Remaining: {len(df)} samples")
    
    X = df[feature_columns].values
    y = df['label'].values
    
    # Split data (80% train, 20% test)
    print("[2] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"    Training set: {len(X_train)} samples")
    print(f"    Test set: {len(X_test)} samples")
    
    # Scale features
    print("[3] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("    [OK] Features normalized (mean=0, std=1)")
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """Train all models and evaluate F1-Score (Macro Average)."""
    print("\n[4] Training and evaluating 6 models...")
    
    models = get_models()
    results = []
    trained_models = {}
    
    # Models that require scaling
    models_requiring_scaling = ['SVM', 'KNN']
    
    for model_name, model in models.items():
        print(f"    Training {model_name}...", end=" ")
        
        try:
            # Use scaled data for distance-based models
            if model_name in models_requiring_scaling:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate F1-Score (Macro Average)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            results.append({
                'Model': model_name,
                'F1-Score (Macro)': f1_macro
            })
            
            trained_models[model_name] = {
                'model': model,
                'requires_scaling': model_name in models_requiring_scaling,
                'f1_score': f1_macro,
                'predictions': y_pred
            }
            
            print(f"F1-Score: {f1_macro:.4f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score (Macro)', ascending=False)
    
    # Find winner
    winner_name = results_df.iloc[0]['Model']
    winner_f1 = results_df.iloc[0]['F1-Score (Macro)']
    
    print(f"\n    [WINNER] {winner_name}: F1-Score = {winner_f1:.4f}")
    
    return results_df, trained_models, winner_name, y_test

# =============================================================================
# CHART GENERATION
# =============================================================================

def create_chart_a_algorithm_comparison(results_df):
    """Chart A: Algorithm Comparison Bar Chart (F1-Score Macro Average)."""
    print("\n[5] Creating Chart A: Algorithm Comparison...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by F1-Score for better visualization
    results_sorted = results_df.sort_values('F1-Score (Macro)', ascending=True)
    
    # Create color list (green for winner, blue for others)
    colors = ['#2ECC71' if model == results_df.iloc[0]['Model'] 
              else '#3498DB' for model in results_sorted['Model']]
    
    # Create bar chart
    bars = ax.barh(range(len(results_sorted)), 
                   results_sorted['F1-Score (Macro)'],
                   color=colors,
                   edgecolor='black',
                   linewidth=1.5,
                   alpha=0.85)
    
    # Add data labels on bars
    for i, (bar, f1) in enumerate(zip(bars, results_sorted['F1-Score (Macro)'])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{f1:.3f}',
                ha='left', va='center',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         edgecolor='black', 
                         alpha=0.8))
    
    # Styling
    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels(results_sorted['Model'], fontsize=11)
    ax.set_xlabel('F1-Score (Macro Average)', fontsize=13, fontweight='bold')
    ax.set_title('Algorithm Comparison: F1-Score Performance\nDriver Drowsiness Detection System',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set x-axis limits for better visualization
    ax.set_xlim(0, min(1.0, results_df['F1-Score (Macro)'].max() * 1.15))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC71', edgecolor='black', label='Winner (Best Model)'),
        Patch(facecolor='#3498DB', edgecolor='black', label='Other Models')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    output_path = ASSETS_DIR / "chart_a_algorithm_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"    [OK] Saved: {output_path}")
    return output_path


def create_chart_b_confusion_matrix(trained_models, winner_name, y_test):
    """Chart B: Normalized Confusion Matrix Heatmap (Recall-based)."""
    print("\n[6] Creating Chart B: Normalized Confusion Matrix...")
    
    # Get winner model predictions
    winner_data = trained_models[winner_name]
    y_pred = winner_data['predictions']
    
    # Create normalized confusion matrix (shows Recall for each class)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cm_percent = cm * 100  # Convert to percentages
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Create heatmap with percentages
    sns.heatmap(cm_percent, 
                annot=True, 
                fmt='.1f',
                cmap='Blues',
                xticklabels=CLASS_LABELS,
                yticklabels=CLASS_LABELS,
                cbar_kws={'label': 'Recall (%)'},
                linewidths=2,
                linecolor='white',
                annot_kws={'size': 13, 'weight': 'bold', 'color': 'white'},
                vmin=0,
                vmax=100,
                ax=ax)
    
    # Styling
    ax.set_title(f'Normalized Confusion Matrix (Recall)\nBest Model: {winner_name}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label (Ground Truth)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
    
    # Add interpretation text
    info_text = ('Values show Recall: "Out of 100 true cases, how many were correctly identified?"')
    ax.text(0.5, -0.12, info_text,
            transform=ax.transAxes,
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    output_path = ASSETS_DIR / "chart_b_confusion_matrix.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"    [OK] Saved: {output_path}")
    return output_path


def create_chart_d_all_models_recall_matrix(trained_models, y_test):
    """Chart D: 6-Model Recall Confusion Matrix Comparison (2x3 Grid)."""
    print("\n[7] Creating Chart D: All Models Recall Matrix Comparison...")
    
    # Define model order for grid (2 rows x 3 columns)
    model_order = [
        ['Logistic Regression', 'SVM', 'KNN'],
        ['Decision Tree', 'Random Forest', 'XGBoost']
    ]
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Recall Confusion Matrix Comparison: All 6 Models\nNormalized by True Label (Row-wise)',
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Process each model
    for row_idx, row_models in enumerate(model_order):
        for col_idx, model_name in enumerate(row_models):
            ax = axes[row_idx, col_idx]
            
            # Check if model exists in trained_models
            if model_name not in trained_models:
                ax.text(0.5, 0.5, f'{model_name}\nNot Available',
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.set_title(model_name, fontsize=13, fontweight='bold', pad=10)
                ax.axis('off')
                continue
            
            # Get predictions for this model
            y_pred = trained_models[model_name]['predictions']
            
            # Create normalized confusion matrix (Recall: normalize='true')
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            cm_percent = cm * 100  # Convert to percentages
            
            # Create heatmap
            sns.heatmap(cm_percent,
                       annot=True,
                       fmt='.0f',  # Show as whole percentages (e.g., 95%)
                       cmap='YlGnBu',  # Yellow-Green-Blue colormap
                       xticklabels=CLASS_LABELS,
                       yticklabels=CLASS_LABELS,
                       cbar_kws={'label': 'Recall (%)', 'shrink': 0.8},
                       linewidths=1.5,
                       linecolor='white',
                       annot_kws={'size': 10, 'weight': 'bold', 'color': 'white'},
                       vmin=0,
                       vmax=100,
                       ax=ax)
            
            # Styling
            ax.set_title(model_name, fontsize=13, fontweight='bold', pad=10)
            
            # Set axis labels (show on all subplots but smaller to reduce clutter)
            ax.set_xlabel('Predicted', fontsize=9, fontweight='bold')
            ax.set_ylabel('True', fontsize=9, fontweight='bold')
            
            # Rotate labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle
    output_path = ASSETS_DIR / "all_models_recall_matrix.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"    [OK] Saved: {output_path}")
    return output_path


def create_chart_c_feature_importance(trained_models, feature_columns, X_train, y_train):
    """Chart C: Feature Importance Horizontal Bar Chart (Top 10)."""
    print("\n[8] Creating Chart C: Feature Importance...")
    
    # Try to get feature importance from Random Forest or XGBoost
    importance_model = None
    model_name = None
    
    # Prefer Random Forest, fallback to XGBoost
    if 'Random Forest' in trained_models:
        importance_model = trained_models['Random Forest']['model']
        model_name = 'Random Forest'
    elif 'XGBoost' in trained_models:
        importance_model = trained_models['XGBoost']['model']
        model_name = 'XGBoost'
    else:
        # Find first available tree-based model
        for name, data in trained_models.items():
            if hasattr(data['model'], 'feature_importances_'):
                importance_model = data['model']
                model_name = name
                break
        
        # If still None, train a Random Forest just for importance
        if importance_model is None:
            print("    [INFO] Training Random Forest for feature importance extraction...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            importance_model = rf
            model_name = 'Random Forest (for importance)'
    
    if importance_model is None or not hasattr(importance_model, 'feature_importances_'):
        print("    [ERROR] Cannot extract feature importance. Skipping Chart C.")
        return None
    
    # Get feature importances
    importances = importance_model.feature_importances_
    
    # Get top 10 features
    top_k = 10
    indices = np.argsort(importances)[::-1][:top_k]
    top_importances = importances[indices]
    top_features = [feature_columns[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    bars = ax.barh(range(top_k), top_importances,
                   color='#9B59B6',
                   edgecolor='darkviolet',
                   linewidth=1.5,
                   alpha=0.85)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}',
                ha='left', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         edgecolor='darkviolet',
                         alpha=0.8))
    
    # Styling
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_features, fontsize=11)
    ax.invert_yaxis()  # Highest importance at top
    
    ax.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Features', fontsize=13, fontweight='bold')
    ax.set_title(f'Top {top_k} Most Important Features\nFeature Importance Analysis ({model_name})',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add info text
    info_text = f'Model: {model_name} | Total Features: {len(feature_columns)} | Shown: Top {top_k}'
    ax.text(0.5, -0.08, info_text,
            transform=ax.transAxes,
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    plt.tight_layout()
    output_path = ASSETS_DIR / "chart_c_feature_importance.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"    [OK] Saved: {output_path}")
    return output_path

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*70)
    print("PROFESSIONAL DATA VISUALIZATION FOR DRIVER DROWSINESS DETECTION")
    print("="*70)
    
    # Load and prepare data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_columns = load_and_prepare_data()
    
    # Train and evaluate models
    results_df, trained_models, winner_name, y_test = train_and_evaluate_models(
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Generate charts
    chart_a_path = create_chart_a_algorithm_comparison(results_df)
    chart_b_path = create_chart_b_confusion_matrix(trained_models, winner_name, y_test)
    chart_d_path = create_chart_d_all_models_recall_matrix(trained_models, y_test)
    chart_c_path = create_chart_c_feature_importance(trained_models, feature_columns, X_train, y_train)
    
    # Print summary
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n[WINNER] Best Model: {winner_name}")
    winner_f1 = results_df[results_df['Model'] == winner_name]['F1-Score (Macro)'].values[0]
    print(f"         F1-Score (Macro Average): {winner_f1:.4f} ({winner_f1*100:.2f}%)")
    print("\n[CHARTS GENERATED]")
    print(f"  ✓ Chart A: Algorithm Comparison")
    print(f"    → {chart_a_path}")
    print(f"  ✓ Chart B: Normalized Confusion Matrix (Recall)")
    print(f"    → {chart_b_path}")
    print(f"  ✓ Chart D: All Models Recall Matrix Comparison (2x3 Grid)")
    print(f"    → {chart_d_path}")
    if chart_c_path:
        print(f"  ✓ Chart C: Feature Importance (Top 10)")
        print(f"    → {chart_c_path}")
    print("\n[INFO] All charts saved at 300 DPI (publication quality)")
    print(f"[INFO] Output directory: {ASSETS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
