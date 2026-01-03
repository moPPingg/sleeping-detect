import cv2
import math
import pickle
import numpy as np
import FaceMeshModule as fm
import winsound # Thư viện để phát tiếng bíp (chỉ chạy trên Windows)

# --- CẤU HÌNH ---
MODEL_PATH = 'drowsiness_model.pkl'
SCORE_THRESHOLD = 10  # Nếu điểm ngủ vượt quá 10 thì báo động

# --- KHỞI TẠO ---
cap = cv2.VideoCapture(0) # 0 là webcam, hoặc điền đường dẫn video test
detector = fm.FaceMeshDetector(maxFaces=1)

# Load bộ não AI đã train
print("⏳ Đang load Model AI...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print("✅ Đã load xong! Hệ thống bắt đầu chạy...")

# Biến đếm điểm (Score)
# - Nếu AI bảo Ngủ -> Cộng điểm
# - Nếu AI bảo Tỉnh -> Trừ điểm
score = 0

# --- HÀM TÍNH TOÁN (Giữ nguyên) ---
def findDistance(p1, p2):
    x1, y1 = p1[1], p1[2]
    x2, y2 = p2[1], p2[2]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_EAR(face, id_mat):
    d_doc_1 = findDistance(face[id_mat[1]], face[id_mat[5]])
    d_doc_2 = findDistance(face[id_mat[2]], face[id_mat[4]])
    d_ngang = findDistance(face[id_mat[0]], face[id_mat[3]])
    ear = (d_doc_1 + d_doc_2) / (2.0 * d_ngang + 0.0001)
    return ear

def calculate_MAR(face, mouth_indices):
    chieu_ngang = findDistance(face[mouth_indices[0]], face[mouth_indices[1]])
    chieu_doc = findDistance(face[mouth_indices[2]], face[mouth_indices[3]])
    mar = chieu_doc / (chieu_ngang + 0.0001)
    return mar

id_mat_trai = [33, 160, 158, 133, 153, 144]
id_mieng = [78, 308, 13, 14]

while True:
    success, img = cap.read()
    img, face = detector.findFaceMesh(img, draw=False)

    if face:
        # 1. Tính toán EAR, MAR
        ear = calculate_EAR(face, id_mat_trai)
        mar = calculate_MAR(face, id_mieng)

        # 2. Đưa vào Model AI để phán đoán
        # Model cần input là dạng 2 chiều [[ear, mar]]
        input_data = np.array([[ear, mar]])
        prediction = model.predict(input_data)[0] # 0 là Tỉnh, 1 là Ngủ
        
        # Lấy xác suất (Độ tự tin của AI)
        prob = model.predict_proba(input_data)[0][1] # Xác suất là Ngủ (0.0 -> 1.0)

        # 3. Logic chấm điểm (Score)
        if prediction == 1: # AI bảo là Ngủ
            score += 1
            cv2.putText(img, "AI: Sleepy", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else: # AI bảo là Tỉnh
            score -= 1
            if score < 0: score = 0 # Không cho điểm âm
            cv2.putText(img, "AI: Alert", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 4. Kiểm tra báo động
        cv2.putText(img, f"Score: {score}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(img, f"Prob: {prob:.2f}", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if score > SCORE_THRESHOLD:
            # BÁO ĐỘNG ĐỎ!
            cv2.putText(img, "WAKE UP!!!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            try:
                winsound.Beep(2500, 100) # Kêu Bíp (Chỉ Windows)
            except:
                pass # Mac/Linux thì bỏ qua dòng này

    cv2.imshow("AI Drowsiness Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()