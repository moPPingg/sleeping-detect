import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
# ==========================================
TRAIN_OUT_DIR = PROJECT_ROOT / "reports" / "train_output"
VOTING_OUT_DIR = PROJECT_ROOT / "reports" / "voting_output"
os.makedirs(VOTING_OUT_DIR, exist_ok=True) 

RANDOM_STATE = 42
TEST_SIZE = 0.2

logger.info("Loading data (Anti-Leakage Version)...")
X_train_list, y_train_list = [], []

# ĐỒNG BỘ 1: Chặn file test giống y hệt train_models.py
for file in os.listdir(FEATURES_DIR):
    if file.startswith("X_flat_") and file != "X_flat.npy" and "test" not in file.lower():
        suffix = file.replace("X_flat_", "").replace(".npy", "")
        y_path = FEATURES_DIR / f"y_labels_{suffix}.npy"
        if not os.path.exists(y_path):
            y_path = FEATURES_DIR / f"y_{suffix}.npy"
            
        if os.path.exists(y_path):
            X_train_list.append(np.load(FEATURES_DIR / file))
            y_train_list.append(np.load(y_path))

if not X_train_list:
    logger.error("Data files not found in features directory!")
    exit(1)

X = np.vstack(X_train_list)
y = np.concatenate(y_train_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

logger.info("Loading individual models from reports/train_output...")
model_names = ["svm", "random_forest", "knn"]
loaded_models = {}

for model_name in tqdm(model_names, desc="Loading models"):
    model_path = TRAIN_OUT_DIR / f"best_{model_name}.joblib"
    if not os.path.exists(model_path):
        logger.error(f"Không tìm thấy model: {model_path}. Vui lòng chạy file src/train_models.py trước!")
        exit(1)
        
    model = joblib.load(model_path)
    loaded_models[model_name] = model

logger.info("Creating Voting Ensembles...")
voting_clf_hard = VotingClassifier(
    estimators=[('svm', loaded_models['svm']), ('rf', loaded_models['random_forest']), ('knn', loaded_models['knn'])],
    voting='hard', weights=[1, 1, 1]
)
voting_clf_soft = VotingClassifier(
    estimators=[('svm', loaded_models['svm']), ('rf', loaded_models['random_forest']), ('knn', loaded_models['knn'])],
    voting='soft', weights=[1, 1, 1]
)

voting_clf_hard.fit(X_train, y_train)
voting_clf_soft.fit(X_train, y_train)

# ĐỒNG BỘ 2: Chấm điểm bằng F1-Macro giống y hệt train_models.py
individual_scores = {}
for model_name, model in loaded_models.items():
    y_pred_ind = model.predict(X_test)
    individual_scores[model_name] = {'f1_macro': f1_score(y_test, y_pred_ind, average='macro', zero_division=0)}

f1_values = np.array([individual_scores[name]['f1_macro'] for name in model_names])
normalized_weights = f1_values / f1_values.sum()

voting_clf_hard_weighted = VotingClassifier(
    estimators=[('svm', loaded_models['svm']), ('rf', loaded_models['random_forest']), ('knn', loaded_models['knn'])],
    voting='hard', weights=normalized_weights
)
voting_clf_soft_weighted = VotingClassifier(
    estimators=[('svm', loaded_models['svm']), ('rf', loaded_models['random_forest']), ('knn', loaded_models['knn'])],
    voting='soft', weights=normalized_weights
)

voting_clf_hard_weighted.fit(X_train, y_train)
voting_clf_soft_weighted.fit(X_train, y_train)

voting_strategies = [
    ('Hard (Equal)', voting_clf_hard),
    ('Hard (Weighted)', voting_clf_hard_weighted),
    ('Soft (Equal)', voting_clf_soft),
    ('Soft (Weighted)', voting_clf_soft_weighted)
]

strategy_results = {}
for strategy_name, clf in voting_strategies:
    y_pred_temp = clf.predict(X_test)
    f1_mac = f1_score(y_test, y_pred_temp, average='macro', zero_division=0)
    strategy_results[strategy_name] = {'clf': clf, 'y_pred': y_pred_temp, 'f1_macro': f1_mac}

best_strategy_name = max(strategy_results.keys(), key=lambda k: strategy_results[k]['f1_macro'])
logger.info(f"\n✓ Best strategy: {best_strategy_name}")

voting_clf_final = strategy_results[best_strategy_name]['clf']
y_pred = strategy_results[best_strategy_name]['y_pred']

print(classification_report(y_test, y_pred))

# ==========================================
# XUẤT KẾT QUẢ VÀO voting_output
# ==========================================
final_model_path = VOTING_OUT_DIR / "final_ensemble_model.joblib"
joblib.dump(voting_clf_final, final_model_path)
logger.info(f"Đã lưu mô hình Voting vô địch vào: {final_model_path}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
ax.set_title(f'Ensemble Voting ({best_strategy_name}) - Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()

cm_path = VOTING_OUT_DIR / "ensemble_confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
logger.info(f"Đã lưu ảnh Confusion Matrix vào: {cm_path}")
logger.info("Hoàn tất quy trình Voting!")
