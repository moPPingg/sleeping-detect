import pandas as pd
import numpy as np
import pickle # Để lưu model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. ĐỌC DỮ LIỆU
print("⏳ Đang đọc dữ liệu từ 'dataset.csv'...")
try:
    data = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("❌ LỖI: Không tìm thấy file dataset.csv. Hãy chắc chắn bạn đã chạy ProcessVideo.py xong.")
    exit()

# Kiểm tra sơ bộ
print(f"✅ Tổng số dòng dữ liệu: {len(data)}")
print("📊 Phân bố nhãn (0=Tỉnh, 1=Ngủ):")
print(data['Label'].value_counts())

# 2. CHIA DỮ LIỆU
# X = Dữ liệu đầu vào (EAR, MAR)
# y = Đáp án (Label)
X = data[['EAR', 'MAR']]
y = data['Label']

# Chia: 80% để Học (Train), 20% để Thi (Test)
# stratify=y giúp đảm bảo tỷ lệ Tỉnh/Ngủ trong tập Train và Test giống nhau
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n--- SẴN SÀNG HUẤN LUYỆN ---")
print(f"Dữ liệu học: {len(X_train)} dòng")
print(f"Dữ liệu thi: {len(X_test)} dòng")

# 3. KHỞI TẠO CÁC MODEL
# class_weight='balanced': Giúp model chú ý hơn đến nhãn ít dữ liệu (để không bị thiên vị)
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

best_model = None
best_accuracy = 0
best_name = ""

# 4. CHO TỪNG MODEL ĐI THI
for name, model in models.items():
    print(f"\n==================================")
    print(f"🤖 Đang train: {name}...")
    
    # Dạy học
    model.fit(X_train, y_train)
    
    # Đi thi
    y_pred = model.predict(X_test)
    
    # Chấm điểm
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"🎯 ĐỘ CHÍNH XÁC (Accuracy): {acc*100:.2f}%")
    print("📉 Confusion Matrix (Ma trận nhầm lẫn):")
    print(cm)
    print("\nGiải thích nhanh:")
    print(f"- Đoán đúng Tỉnh: {cm[0][0]} dòng")
    print(f"- Đoán đúng Ngủ : {cm[1][1]} dòng")
    print(f"- Báo ĐỘNG GIẢ (Thức mà bảo Ngủ): {cm[0][1]} dòng")
    print(f"- BỎ SÓT (Ngủ mà bảo Thức): {cm[1][0]} dòng (Cái này NGUY HIỂM nhất)")
    
    # So sánh tìm quán quân
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

# 5. LƯU QUÁN QUÂN
print(f"\n==================================")
print(f"🏆 MODEL VÔ ĐỊCH: {best_name}")
print(f"🥇 Độ chính xác: {best_accuracy*100:.2f}%")

model_filename = "drowsiness_model.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ Đã lưu model vào file '{model_filename}'")
print("👉 Bây giờ bạn có thể dùng file này để chạy thực tế!")