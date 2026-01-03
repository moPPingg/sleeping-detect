import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, precision_score, 
                            recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. LOAD VÀ KIỂM TRA DỮ LIỆU
# ==========================================
print("⏳ Đang đọc dữ liệu từ 'dataset_full.csv'...")
try:
    data = pd.read_csv('dataset_full.csv')
except FileNotFoundError:
    print("❌ LỖI: Không tìm thấy file dataset_full.csv.")
    print("💡 Hãy chạy chongngugat.py hoặc collect_data.py trước để tạo dataset.")
    exit()

print(f"✅ Tổng số dòng dữ liệu: {len(data)}")
print("\n📊 Phân bố nhãn:")
print(data['Label'].value_counts())
print(f"Tỷ lệ: {data['Label'].value_counts(normalize=True)}")

# Kiểm tra cân bằng dữ liệu
label_counts = data['Label'].value_counts()
if len(label_counts) == 2:
    imbalance_ratio = abs(label_counts[0] - label_counts[1]) / len(data)
    if imbalance_ratio > 0.2:
        print(f"⚠️ CẢNH BÁO: Dữ liệu không cân bằng! (Chênh lệch: {imbalance_ratio*100:.1f}%)")
        print("   Có thể ảnh hưởng đến độ chính xác. Nên thu thập thêm dữ liệu.")

# Kiểm tra missing values
if data.isnull().sum().sum() > 0:
    print("⚠️ CẢNH BÁO: Có dữ liệu bị thiếu!")
    print(data.isnull().sum())

# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU
# ==========================================
# Dùng tất cả features: EAR, MAR, Pitch, Yaw, Roll
X = data[['EAR', 'MAR', 'Pitch', 'Yaw', 'Roll']]
y = data['Label']

# Chia train/test với stratify để giữ tỷ lệ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa dữ liệu (quan trọng vì Pitch, Yaw, Roll có giá trị lớn)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lưu scaler để dùng khi predict
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\n✅ Đã lưu scaler vào 'scaler.pkl'")

print(f"\n--- SẴN SÀNG HUẤN LUYỆN ---")
print(f"Dữ liệu học: {len(X_train)} dòng")
print(f"Dữ liệu thi: {len(X_test)} dòng")

# ==========================================
# 3. ĐỊNH NGHĨA CÁC MODEL
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(
        class_weight='balanced', 
        max_iter=1000,
        random_state=42
    ),
    "SVM": SVC(
        kernel='rbf', 
        probability=True, 
        class_weight='balanced',
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42,
        max_depth=10
    )
}

# ==========================================
# 4. TRAIN VÀ ĐÁNH GIÁ TỪNG MODEL
# ==========================================
results = {}
best_model = None
best_score = 0
best_name = ""

print("\n" + "="*60)
print("🚀 BẮT ĐẦU HUẤN LUYỆN VÀ SO SÁNH")
print("="*60)

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"🤖 MODEL: {name}")
    print(f"{'='*60}")
    
    # Train model (dùng dữ liệu đã chuẩn hóa)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Tính các metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC AUC (quan trọng cho bài toán imbalanced)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = 0
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Lưu kết quả
    results[name] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # In kết quả
    print(f"\n📊 KẾT QUẢ:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc*100:.2f}%")
    
    print(f"\n📉 Confusion Matrix:")
    print(f"  [{cm[0][0]:4d}  {cm[0][1]:4d}]  <- True Negative | False Positive")
    print(f"  [{cm[1][0]:4d}  {cm[1][1]:4d}]  <- False Negative | True Positive")
    print(f"\n  Giải thích:")
    print(f"  - Đoán đúng Tỉnh táo: {cm[0][0]} dòng")
    print(f"  - Đoán đúng Buồn ngủ: {cm[1][1]} dòng")
    print(f"  - Báo ĐỘNG GIẢ (Thức → Ngủ): {cm[0][1]} dòng")
    print(f"  - BỎ SÓT (Ngủ → Thức): {cm[1][0]} dòng ⚠️ NGUY HIỂM!")
    
    # Tìm model tốt nhất (dựa trên F1-score vì quan trọng cả precision và recall)
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name

# ==========================================
# 5. SO SÁNH TỔNG QUAN
# ==========================================
print("\n" + "="*60)
print("📊 BẢNG SO SÁNH TỔNG QUAN")
print("="*60)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
print("-"*60)

for name, result in results.items():
    print(f"{name:<20} {result['accuracy']*100:>10.2f}% {result['precision']*100:>10.2f}% "
          f"{result['recall']*100:>10.2f}% {result['f1']*100:>10.2f}% {result['roc_auc']*100:>10.2f}%")

# ==========================================
# 6. PHÂN TÍCH ĐIỂM MẠNH/ĐIỂM YẾU
# ==========================================
print("\n" + "="*60)
print("🔍 PHÂN TÍCH ĐIỂM MẠNH VÀ ĐIỂM YẾU")
print("="*60)

for name, result in results.items():
    print(f"\n📌 {name}:")
    
    # Điểm mạnh
    strengths = []
    if result['accuracy'] == max(r['accuracy'] for r in results.values()):
        strengths.append("Accuracy cao nhất")
    if result['precision'] == max(r['precision'] for r in results.values()):
        strengths.append("Precision cao nhất (ít báo động giả)")
    if result['recall'] == max(r['recall'] for r in results.values()):
        strengths.append("Recall cao nhất (ít bỏ sót)")
    if result['f1'] == max(r['f1'] for r in results.values()):
        strengths.append("F1-Score cao nhất (cân bằng tốt)")
    
    if strengths:
        print(f"  ✅ Điểm mạnh: {', '.join(strengths)}")
    else:
        print(f"  ✅ Điểm mạnh: Không có điểm nổi trội")
    
    # Điểm yếu
    weaknesses = []
    if result['recall'] < 0.8:
        weaknesses.append("Recall thấp → Dễ bỏ sót trường hợp nguy hiểm")
    if result['precision'] < 0.8:
        weaknesses.append("Precision thấp → Nhiều báo động giả")
    if name == "Logistic Regression":
        weaknesses.append("Model đơn giản, có thể không bắt được pattern phức tạp")
    elif name == "SVM":
        weaknesses.append("Chậm hơn khi dữ liệu lớn, khó tune hyperparameters")
    elif name == "Random Forest":
        weaknesses.append("Có thể overfit nếu dữ liệu ít, tốn bộ nhớ")
    
    if weaknesses:
        print(f"  ⚠️ Điểm yếu: {'; '.join(weaknesses)}")

# ==========================================
# 7. LƯU MODEL TỐT NHẤT
# ==========================================
print("\n" + "="*60)
print(f"🏆 MODEL TỐT NHẤT: {best_name}")
print(f"🥇 F1-Score: {best_score*100:.2f}%")
print("="*60)

model_filename = "drowsiness_model.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ Đã lưu model vào '{model_filename}'")
print("✅ Đã lưu scaler vào 'scaler.pkl'")
print("\n👉 Bây giờ bạn có thể dùng file này để chạy thực tế!")

# ==========================================
# 8. VẼ BIỂU ĐỒ SO SÁNH (Tùy chọn)
# ==========================================
try:
    # So sánh metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        values = [results[model][metric] for model in models.keys()]
        bars = ax.bar(models.keys(), values, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel('Score')
        ax.set_title(f'{name} Comparison')
        ax.set_ylim([0, 1])
        
        # Thêm giá trị lên cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    print("\n✅ Đã lưu biểu đồ so sánh vào 'model_comparison.png'")
    plt.close()
except Exception as e:
    print(f"\n⚠️ Không thể vẽ biểu đồ: {e}")
    print("   (Có thể do thiếu matplotlib hoặc seaborn)")