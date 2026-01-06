"""
Professional Chart Generation for Driver Monitoring System
Creates publication-quality visualizations with English labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import os

# Configure matplotlib for better output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12  # Balanced size
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# --- CONFIGURATION ---
DATA_FILE = 'face_data.csv'
MODEL_FILE = 'drowsiness_model.pkl'

# Class labels (English)
CLASS_LABELS = ['Awake', 'Drowsy', 'Looking Down\n(Phone)', 'Microsleep']
CLASS_LABELS_SHORT = ['Awake', 'Drowsy', 'Phone', 'Microsleep']

# Professional color palette
COLORS = {
    'primary': '#2E86AB',      # Blue
    'success': '#06A77D',      # Green
    'warning': '#F77F00',      # Orange
    'danger': '#D62828',       # Red
    'purple': '#9B59B6',       # Purple
    'teal': '#16A085',         # Teal
}

PALETTE_QUALITATIVE = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']  # Blue, Red, Green, Orange

def main():
    print("="*70)
    print("GENERATING PROFESSIONAL CHARTS FOR REPORT")
    print("="*70)
    
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        return
    
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] Model file not found: {MODEL_FILE}")
        return

    # Load Data
    print("\n[1/3] Loading dataset and model...")
    df = pd.read_csv(DATA_FILE)
    
    # Reconstruct labels (same logic as training)
    chunk_size = len(df) // 4
    labels = np.zeros(len(df), dtype=int)
    labels[chunk_size : 2*chunk_size] = 1
    labels[2*chunk_size : 3*chunk_size] = 2
    labels[3*chunk_size :] = 3
    df['label'] = labels
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load trained model
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    
    # Get predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   Model loaded: {MODEL_FILE}")
    print(f"   Test set accuracy: {accuracy*100:.2f}%")
    
    # ---------------------------------------------------------
    # CHART 1: CONFUSION MATRIX (Professional Design)
    # ---------------------------------------------------------
    print("\n[2/3] Creating Chart 1: Confusion Matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap with custom colormap
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', 
                xticklabels=CLASS_LABELS_SHORT, 
                yticklabels=CLASS_LABELS_SHORT,
                cbar_kws={'label': 'Number of Samples'},
                linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'weight': 'bold'},
                vmin=0, vmax=cm.max(),
                ax=ax)
    
    # Styling
    ax.set_title('Confusion Matrix - Random Forest Classifier\nTest Set Performance', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('True Label (Ground Truth)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    # Add accuracy text
    accuracy_text = f'Overall Accuracy: {accuracy*100:.2f}%'
    ax.text(0.5, -0.15, accuracy_text, 
            transform=ax.transAxes, 
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("   [OK] Saved: confusion_matrix.png (10x8 inches, 300 DPI)")

    # ---------------------------------------------------------
    # CHART 2: DATA DISTRIBUTION (Professional Pie Chart)
    # ---------------------------------------------------------
    print("\n[2/3] Creating Chart 2: Dataset Distribution...")
    
    counts = df['label'].value_counts().sort_index()
    total_samples = len(df)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart with explosion effect
    explode = (0.05, 0.05, 0.05, 0.05)  # Slightly separate all slices
    
    wedges, texts, autotexts = ax.pie(
        counts, 
        labels=CLASS_LABELS_SHORT,
        autopct='%1.1f%%',
        startangle=90,
        colors=PALETTE_QUALITATIVE,
        explode=explode,
        shadow=True,
        textprops={'fontsize': 13, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Style percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(15)
        autotext.set_weight('bold')
    
    # Add title
    ax.set_title('Dataset Class Distribution\nBalanced Training Data', 
                 fontsize=20, fontweight='bold', pad=20)
    
    # Add legend with sample counts
    legend_labels = [f'{label}: {count} samples' 
                     for label, count in zip(CLASS_LABELS_SHORT, counts)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=14, frameon=True, shadow=True)
    
    # Add total count text
    total_text = f'Total Samples: {total_samples:,}'
    ax.text(0.5, -0.1, total_text, 
            transform=ax.transAxes, 
            ha='center', fontsize=15, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data_balance.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("   [OK] Saved: data_balance.png (10x8 inches, 300 DPI)")

    # ---------------------------------------------------------
    # CHART 3: FEATURE IMPORTANCE (Top 20 - Professional Bar Chart)
    # ---------------------------------------------------------
    print("\n[3/3] Creating Chart 3: Feature Importance...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort descending
    
    # Get top 20 features
    top_k = 20
    top_indices = indices[:top_k]
    top_importances = importances[top_indices]
    
    # Create feature names (landmark indices)
    feature_names = [f'Landmark {idx}' for idx in top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(range(top_k), top_importances, 
                   color=COLORS['purple'], 
                   edgecolor='darkviolet', 
                   linewidth=1.5,
                   alpha=0.85)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}',
                ha='left', va='center', fontsize=11, 
                fontweight='bold', color='darkviolet',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='darkviolet', alpha=0.7))
    
    # Styling
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(feature_names, fontsize=13)
    ax.invert_yaxis()  # Highest importance at top
    
    ax.set_xlabel('Importance Score (Gini Importance)', 
                  fontsize=16, fontweight='bold')
    ax.set_ylabel('Facial Landmark Features', 
                  fontsize=16, fontweight='bold')
    ax.set_title(f'Top {top_k} Most Important Features\nRandom Forest Feature Importance Analysis', 
                 fontsize=20, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add info text
    info_text = (f'Total Features: {len(importances)} | '
                 f'Shown: Top {top_k} | '
                 f'Model: Random Forest (100 trees)')
    ax.text(0.5, -0.08, info_text, 
            transform=ax.transAxes, 
            ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("   [OK] Saved: feature_importance.png (12x8 inches, 300 DPI)")
    
    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("CHART GENERATION COMPLETE!")
    print("="*70)
    print(f"[OK] confusion_matrix.png    - {accuracy*100:.2f}% accuracy visualization")
    print(f"[OK] data_balance.png         - {total_samples:,} samples across 4 classes")
    print(f"[OK] feature_importance.png   - Top {top_k} features from {len(importances)} total")
    print("="*70)
    print("\n[INFO] All charts are ready for inclusion in Project_Report.html")
    print("[INFO] Resolution: 300 DPI (publication quality)")
    print("[INFO] Language: English (professional)")
    print("="*70)

if __name__ == "__main__":
    main()
