from sklearnex import patch_sklearn
patch_sklearn()

import os
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import shutil

# Import Config to reproduce the same scanning order used during feature extraction
from extract_features import Config

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"

# Thư mục LẤY model vô địch (Do src/build_voting_ensemble.py xuất ra)
MODEL_PATH = PROJECT_ROOT / "reports" / "voting_output" / "final_ensemble_model.joblib"

# Thư mục LƯU toàn bộ kết quả test
TEST_OUT_DIR = PROJECT_ROOT / "reports" / "test_output"
os.makedirs(TEST_OUT_DIR, exist_ok=True)

print("\n" + "="*60)
print("🚀 HỆ THỐNG ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ĐỘC LẬP")
print("="*60)

# ==========================================
# 2. NẠP MÔ HÌNH
# ==========================================
if not os.path.exists(MODEL_PATH):
    print(f"[❌] LỖI: Không tìm thấy file model: {MODEL_PATH}")
    print("Vui lòng đảm bảo bạn đã chạy src/build_voting_ensemble.py thành công!")
    exit()

print("[INFO] Đang nạp mô hình vô địch (Best Model)...")
model = joblib.load(MODEL_PATH)
print(f"[OK] Đã nạp mô hình thành công: {type(model).__name__}")

# ==========================================
# 3. NẠP DỮ LIỆU TEST
# ==========================================
print("\n[INFO] Đang quét tìm dữ liệu Test trong thư mục 'features'...")
X_test_list, y_test_list = [], []

if not os.path.exists(FEATURES_DIR):
    print(f"[❌] LỖI: Không tìm thấy thư mục {FEATURES_DIR}")
    exit()

test_suffixes = []
for file in os.listdir(FEATURES_DIR):
    if file.startswith("X_flat_") and "test" in file.lower() and file.endswith(".npy"):
        suffix = file.replace("X_flat_", "").replace(".npy", "")
        test_suffixes.append(suffix)
        y_path = FEATURES_DIR / f"y_{suffix}.npy"
        
        if not os.path.exists(y_path):
            y_path = FEATURES_DIR / f"y_labels_{suffix}.npy"
            
        if os.path.exists(y_path):
            X_test_list.append(np.load(FEATURES_DIR / file))
            y_test_list.append(np.load(y_path))
            print(f"   + Đã nạp file Test ({suffix}): {len(np.load(y_path))} mẫu")

if not X_test_list:
    print("[❌] LỖI: Không tìm thấy file X_flat_test...npy nào trong thư mục features.")
    print("Gợi ý: Đảm bảo bạn đã chạy src/extract_features.py cho thư mục data/test.")
    exit()

X_test = np.vstack(X_test_list)
y_test = np.concatenate(y_test_list)

# ==========================================
# 4. DỰ ĐOÁN VÀ ĐÁNH GIÁ (NGUYÊN BẢN 100% CỦA EM)
# ==========================================
print("\n" + "="*60)
print("PHẦN 2: DỰ ĐOÁN VÀ XUẤT BÁO CÁO")
print("="*60)

print("[INFO] Đang chạy dự đoán...")
y_pred = model.predict(X_test)


def build_test_set_paths(test_root, required_frames=30, valid_exts=('.jpg', '.jpeg', '.png', '.bmp')):
    cfg = Config()
    set_paths = []
    root = test_root

    if not os.path.isdir(root):
        return set_paths

    subfolders_at_root = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    has_direct_labels = any(label in [s.lower() for s in subfolders_at_root] for label in cfg.labels)

    if has_direct_labels:
        for label in sorted(os.listdir(root)):
            l_path = os.path.join(root, label)
            if not os.path.isdir(l_path):
                continue
            sets = sorted([s for s in os.listdir(l_path) if os.path.isdir(os.path.join(l_path, s))])
            for set_folder in sets:
                s_path = os.path.join(l_path, set_folder)
                files = [f for f in os.listdir(s_path) if os.path.isfile(os.path.join(s_path, f)) and f.lower().endswith(valid_exts)]
                if len(files) >= required_frames:
                    set_paths.append(s_path)
    else:
        persons = sorted([p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))])
        for person in persons:
            p_path = os.path.join(root, person)
            for label in sorted(os.listdir(p_path)):
                l_path = os.path.join(p_path, label)
                if not os.path.isdir(l_path):
                    continue
                sets = sorted([s for s in os.listdir(l_path) if os.path.isdir(os.path.join(l_path, s))])
                for set_folder in sets:
                    s_path = os.path.join(l_path, set_folder)
                    files = [f for f in os.listdir(s_path) if os.path.isfile(os.path.join(s_path, f)) and f.lower().endswith(valid_exts)]
                    if len(files) >= required_frames:
                        set_paths.append(s_path)

    return set_paths


def extract_true_label_from_path(set_path, cfg):
    path_parts = set_path.replace("\\", "/").split("/")
    for label_key, label_idx in cfg.labels.items():
        if label_key in [p.lower() for p in path_parts]:
            return label_idx
    return -1


def save_frames_sorted_by_prediction(set_paths, predictions, out_base, label_map=None):
    if not set_paths:
        print("[WARN] No set paths found to save frames.")
        return

    n_src = len(set_paths)
    n_pred = len(predictions)
    if n_src != n_pred:
        print(f"[WARN] Number of discovered set folders ({n_src}) != number of predictions ({n_pred}). Truncating to smaller length.")

    length = min(n_src, n_pred)
    os.makedirs(out_base, exist_ok=True)

    for i in range(length):
        src = set_paths[i]
        pred = predictions[i]
        label_name = str(pred)
        if label_map and pred in label_map:
            label_name = label_map[pred]

        dest_dir = os.path.join(out_base, label_name, os.path.basename(src))
        os.makedirs(dest_dir, exist_ok=True)

        for fname in sorted(os.listdir(src)):
            fpath = os.path.join(src, fname)
            if os.path.isfile(fpath):
                try:
                    shutil.copy2(fpath, os.path.join(dest_dir, fname))
                except Exception as e:
                    pass

    print(f"[INFO] Copied {length} set folders into '{out_base}', grouped by predicted label.")


def save_misclassified_report(set_paths, predictions, true_labels, out_csv, label_map=None):
    if not set_paths or len(set_paths) != len(predictions) or len(predictions) != len(true_labels):
        print("[WARN] Cannot create misclassified report: length mismatch")
        return
    
    misclassified = []
    for i, (path, pred, true) in enumerate(zip(set_paths, predictions, true_labels)):
        if pred != true:
            set_name = os.path.basename(path)
            true_label = label_map.get(true, str(true)) if label_map else str(true)
            pred_label = label_map.get(pred, str(pred)) if label_map else str(pred)
            misclassified.append({
                'Set_Name': set_name,
                'True_Layer': true_label,
                'Predicted_Layer': pred_label
            })
    
    if misclassified:
        df = pd.DataFrame(misclassified)
        try:
            os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
            df.to_csv(out_csv, index=False)
            print(f"✅ Đã xuất báo cáo các mẫu sai: {out_csv} ({len(misclassified)} mẫu)")
        except Exception as e:
            print(f"[WARN] Lỗi khi lưu báo cáo sai: {e}")
    else:
        print("[INFO] Không có mẫu nào bị phân loại sai.")


# ==========================================
# 5. XUẤT FILE LƯU TRỮ VÀO TEST_OUT_DIR
# ==========================================
try:
    cfg = Config()
    
    suffix_to_dir = {
        "test": PROJECT_ROOT / "data" / "test",
        "test_haar_cropped": PROJECT_ROOT / "data" / "test" / "haar_cropped",
        "test_augmented": PROJECT_ROOT / "data" / "test" / "train_augmented",
        "test_augmented_cropped": PROJECT_ROOT / "data" / "test" / "train_augmented_cropped",
    }
    
    if test_suffixes:
        primary_suffix = test_suffixes[0]
        test_data_dir = suffix_to_dir.get(primary_suffix, PROJECT_ROOT / "data" / "test")
    else:
        test_data_dir = PROJECT_ROOT / "data" / "test"
    
    required_frames = getattr(cfg, 'batch_size', 30)
    set_paths = build_test_set_paths(test_data_dir, required_frames=required_frames, valid_exts=cfg.valid_exts)
    label_map = {v: k for k, v in cfg.labels.items()}
    
    true_labels_for_sets = np.array([extract_true_label_from_path(p, cfg) for p in set_paths])
    
    # XUẤT ẢNH DỰ ĐOÁN VÀO test_output
    out_base = TEST_OUT_DIR / "sorted_frames_by_prediction"
    save_frames_sorted_by_prediction(set_paths, y_pred, out_base, label_map=label_map)
    
    # XUẤT CSV MẪU SAI VÀO test_output
    misclassified_csv = TEST_OUT_DIR / "misclassified_samples.csv"
    save_misclassified_report(set_paths, y_pred, true_labels_for_sets, misclassified_csv, label_map=label_map)
    
except Exception as e:
    print(f"[WARN] Could not save sorted frames: {e}")

total = len(y_test)
correct = np.sum(y_pred == y_test)
accuracy = (correct / total) * 100

print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC:")
print(f"   - Tổng số mẫu Test: {total}")
print(f"   - Đoán đúng:        {correct}")
print(f"   - Đoán sai:         {total - correct}")
print(f"   - Độ chính xác:     {accuracy:.2f}%")

print("\nBÁO CÁO CHI TIẾT (Classification Report):")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

try:
    df_report = pd.DataFrame(report).T
    # XUẤT CSV REPORT VÀO test_output
    report_csv_path = TEST_OUT_DIR / "independent_test_report.csv"
    df_report.to_csv(report_csv_path, index=True)
    print(f"\n✅ Đã xuất báo cáo CSV: {report_csv_path}")
except Exception as e:
    print(f"\n[WARN] Lỗi khi lưu CSV: {e}")

try:
    cfg = Config()
    idx_to_name = {v: k for k, v in cfg.labels.items()}
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix - Layers (Independent Test)", fontsize=14, fontweight='bold')
    plt.colorbar()

    class_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    tick_marks = np.arange(len(class_labels))
    label_names = [idx_to_name.get(i, str(i)) for i in class_labels]
    
    plt.xticks(tick_marks, label_names, fontsize=11)
    plt.yticks(tick_marks, label_names, fontsize=11)
    plt.xlabel("Predicted Layer", fontsize=12, fontweight='bold')
    plt.ylabel("True Layer", fontsize=12, fontweight='bold')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center", fontsize=12,
                     color="white" if cm[i, j] > thresh else "black", fontweight='bold')

    plt.tight_layout()
    # XUẤT ẢNH MA TRẬN VÀO test_output
    cm_fig_path = TEST_OUT_DIR / "independent_test_confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=150)
    plt.close()
    print(f"✅ Đã xuất ảnh Ma trận nhầm lẫn: {cm_fig_path}")
except Exception as e:
    print(f"[WARN] Lỗi khi lưu confusion matrix: {e}")

print("\nHoàn tất quy trình kiểm thử!")
