import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- CÁC MODEL SẼ SO SÁNH ---
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.svm import SVC                         # SVM
from sklearn.ensemble import RandomForestClassifier # Random Forest

# ==========================================
# 1. LOAD DỮ LIỆU
# ==========================================
# Thay 'driver_data.csv' bằng tên file thực tế của bạn
# Giả sử file CSV có các cột: EAR, MAR, Pitch, Yaw, Roll, Label (0=Tỉnh, 1=Ngủ)
filename = 'dataset_full.csv' 
try:
    data = pd.read_csv(filename)
    print(f"Đã load thành công: {len(data)} dòng dữ liệu.")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{filename}'. Hãy kiểm tra lại tên file.")
    exit()

# Tách features (X) và nhãn (y)
# Giả sử cột cuối cùng là nhãn (Label)
X = data.iloc[:, :-1] # Lấy tất cả các cột trừ cột cuối
y = data.iloc[:, -1]  # Chỉ lấy cột cuối cùng

# Chia tập train (80%) và tập test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu (SVM và KNN rất cần bước này để chạy tốt)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. ĐỊNH NGHĨA CÁC MODEL
# ==========================================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# ==========================================
# 3. CHẠY VÒNG LẶP SO SÁNH
# ==========================================
print("\n--- BẮT ĐẦU SO SÁNH ---")
results = {}

for name, model in models.items():
    print(f"\nĐang train model: {name}...")
    # Dùng dữ liệu đã chuẩn hóa cho KNN và SVM
    if name in ["KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    # Random Forest không bắt buộc phải chuẩn hóa, nhưng dùng luôn cũng được
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Tính độ chính xác
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"-> Độ chính xác của {name}: {acc:.4f} (hay {acc*100:.2f}%)")
    # Nếu muốn xem chi tiết thì bỏ comment dòng dưới:
    # print(classification_report(y_test, y_pred))

# ==========================================
# 4. TỔNG KẾT
# ==========================================
print("\n\n=== KẾT QUẢ CUỐI CÙNG ===")
# Sắp xếp model từ tốt nhất đến tệ nhất
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, acc in sorted_results:
    print(f"{name:<15}: {acc*100:.2f}%")

best_model_name = sorted_results[0][0]
print(f"\n=> Model tốt nhất là: {best_model_name.upper()}. Hãy dùng model này cho dự án!")