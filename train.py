import os
import warnings
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
OUTPUT_DIR = os.path.join(BASE_DIR, "train_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration Constants (synced with voting.py)
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 2
SCORING = "f1_macro"

#

# 1. Chuẩn bị dữ liệu (Phiên bản CHỐNG RÒ RỈ DATA - Đã Fix tên file)
X_train_list, y_train_list = [], []

print("[INFO] Đang nạp dữ liệu Train, chặn hoàn toàn dữ liệu Test...")
for file in os.listdir(FEATURES_DIR):
    # Lấy các file đặc trưng X_flat_..., NHƯNG loại trừ file tổng (X_flat.npy) và tập test
    if file.startswith("X_flat_") and file != "X_flat.npy" and "test" not in file.lower():
        suffix = file.replace("X_flat_", "").replace(".npy", "")
        
        # Tìm file y tương ứng
        y_path = os.path.join(FEATURES_DIR, f"y_labels_{suffix}.npy")
        if not os.path.exists(y_path): # Cú pháp dự phòng
            y_path = os.path.join(FEATURES_DIR, f"y_{suffix}.npy")
            
        if os.path.exists(y_path):
            X_train_list.append(np.load(os.path.join(FEATURES_DIR, file)))
            y_train_list.append(np.load(y_path))
            print(f"   + Nạp thành công tập Train: {file}")

if not X_train_list:
    print("[❌] LỖI: Không tìm thấy dữ liệu Train. Hãy kiểm tra lại thư mục features.")
    exit()

# Gộp tất cả các tập Train con lại thành 1 tập X thuần khiết (100% là ảnh học)
X = np.vstack(X_train_list)
y = np.concatenate(y_train_list)

# Chia 80/20 NỘI BỘ tập Train để đánh giá trong quá trình học (Không đụng tới folder test gốc)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# Chia 80/20 NỘI BỘ tập Train để đánh giá trong quá trình học (Không đụng tới folder test gốc)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# 2. Định nghĩa danh sách các model và các tham số cần test (Parameter Grid)
USE_GPU = os.getenv("USE_GPU", "1") == "1"
GPU_BACKEND = "cpu"

if USE_GPU:
    try:
        from cuml.svm import SVC as cuSVC
        from cuml.ensemble import RandomForestClassifier as cuRF
        from cuml.neighbors import KNeighborsClassifier as cuKNN

        SVC_CLASS = cuSVC
        RF_CLASS = cuRF
        KNN_CLASS = cuKNN
        GPU_BACKEND = "cuml"
    except Exception as exc:
        warnings.warn(f"GPU not available ({exc}). Falling back to CPU.")
        USE_GPU = False

if not USE_GPU:
    SVC_CLASS = SVC
    RF_CLASS = RandomForestClassifier
    KNN_CLASS = KNeighborsClassifier

print(f"Using backend: {GPU_BACKEND}")

if USE_GPU:
    # cuML may not support all sklearn parameters, keep a safe grid
    model_params = {
        "svm": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC_CLASS())
            ]),
            "params": {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["rbf", "linear"],
                "clf__gamma": ["scale", "auto"],
            }
        },
        "random_forest": {
            "model": Pipeline([
                ("scaler", "passthrough"),
                ("clf", RF_CLASS(random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [None, 10, 20],
            }
        },
        "knn": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNN_CLASS())
            ]),
            "params": {
                "clf__n_neighbors": [3, 5, 11],
            }
        }
    }
else:
    model_params = {
        "svm": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC_CLASS(probability=True))  # Enable probability for soft voting
            ]),
            "params": {
                "clf__C": [0.1, 1, 10],
                "clf__kernel": ["rbf", "linear"],
                "clf__gamma": ["scale", "auto"],
                "clf__class_weight": [None, "balanced"]
            }
        },
        "random_forest": {
            "model": Pipeline([
                ("scaler", "passthrough"),
                ("clf", RF_CLASS(random_state=RANDOM_STATE))
            ]),
            "params": {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
                "clf__class_weight": [None, "balanced", "balanced_subsample"]
            }
        },
        "knn": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNN_CLASS())
            ]),
            "params": {
                "clf__n_neighbors": [3, 5, 11],
                "clf__weights": ["uniform", "distance"]
            }
        }
    }

# 3. Chạy vòng lặp GridSearchCV
results = []
best_estimators = {}

cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
nested_outer_cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
nested_inner_cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
repeated_cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)

for model_name, mp in tqdm(model_params.items(), desc="Training models", total=len(model_params)):
    print(f"\n{'='*60}")
    print(f"Đang tìm tham số tốt nhất cho {model_name.upper()}...")
    print(f"{'='*60}")
    clf = GridSearchCV(
        mp["model"],
        mp["params"],
        cv=cv,
        return_train_score=False,
        n_jobs=-1,
        scoring=SCORING,
        refit=True,
        verbose=2,  # Show detailed progress for each fit
    )
    clf.fit(X_train, y_train)

    best_idx = clf.best_index_
    mean_score = clf.cv_results_["mean_test_score"][best_idx]
    std_score = clf.cv_results_["std_test_score"][best_idx]

    cv_results_df = pd.DataFrame(clf.cv_results_)
    cv_results_path = os.path.join(OUTPUT_DIR, f"cv_results_{model_name}.csv")
    cv_results_df.to_csv(cv_results_path, index=False)
    print(f"Da luu cv_results_ cho {model_name} vao {cv_results_path}")
    
    # Nested CV on training split for an unbiased training-set estimate
    print(f"\nRunning nested cross-validation for {model_name}...")
    nested_clf = GridSearchCV(
        mp["model"],
        mp["params"],
        cv=nested_inner_cv,
        return_train_score=False,
        n_jobs=-1,
        scoring=SCORING,
        refit=True,
        verbose=1,  # Less verbose for nested CV
    )
    nested_scores = cross_val_score(
        nested_clf,
        X_train,
        y_train,
        cv=nested_outer_cv,
        scoring=SCORING,
        n_jobs=-1,
    )

    # Repeated CV on training split using the tuned estimator
    print(f"Running repeated cross-validation for {model_name}...")
    repeated_scores = cross_val_score(
        clf.best_estimator_,
        X_train,
        y_train,
        cv=repeated_cv,
        scoring=SCORING,
        n_jobs=-1,
    )

    results.append({
        "model": model_name,
        "scoring": SCORING,
        "cv_splits": CV_SPLITS,
        "cv_repeats": CV_REPEATS,
        "mean_cv_score": mean_score,
        "std_cv_score": std_score,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_,
        "nested_cv_mean": float(np.mean(nested_scores)),
        "nested_cv_std": float(np.std(nested_scores)),
        "repeated_cv_mean": float(np.mean(repeated_scores)),
        "repeated_cv_std": float(np.std(repeated_scores)),
    })
    best_estimators[model_name] = clf.best_estimator_
    
    # Lưu individual model
    model_path = os.path.join(OUTPUT_DIR, f"best_{model_name}.joblib")
    joblib.dump(clf.best_estimator_, model_path)
    print(f"   -> Đã lưu {model_name} vào {model_path}")

# 4. Hiển thị bảng so sánh
df_results = pd.DataFrame(results)
print(df_results)

results_path = os.path.join(OUTPUT_DIR, "gridsearch_results.csv")
df_results.to_csv(results_path, index=False)
print(f"\nĐã lưu kết quả GridSearch vào {results_path}")

# 5. Lưu lại Model tốt nhất tuyệt đối và Scaler
best_model_name = df_results.loc[df_results["best_score"].idxmax()]["model"]
best_model = best_estimators[best_model_name]
best_model_path = os.path.join(OUTPUT_DIR, "best_model_overall.joblib")
joblib.dump(best_model, best_model_path)

# 6. Danh gia tren tap test
y_pred = best_model.predict(X_test)
print("\nTest classification report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

report_df = pd.DataFrame(report).T
report_path = os.path.join(OUTPUT_DIR, "test_report.csv")
report_df.to_csv(report_path, index=True)
print(f"Da luu test report vao {report_path}")

cm_fig_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
class_labels = np.unique(np.concatenate([y_test, y_pred]))
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, [str(label) for label in class_labels])
plt.yticks(tick_marks, [str(label) for label in class_labels])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(cm_fig_path, dpi=150)
plt.close()
print(f"Da luu confusion matrix image vao {cm_fig_path}")

print(f"\nDa luu model {best_model_name} thanh file '{best_model_path}'")