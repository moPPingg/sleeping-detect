import cv2
import csv
import math
import os
import numpy as np
import FaceMeshModule as fm

# ==========================================
# ⚙️ CẤU HÌNH
# ==========================================
OUTPUT_FILE = "dataset_full.csv"
TARGET_COUNT = None  # None = Không giới hạn, lưu TẤT CẢ frame hợp lệ

# ==========================================
# 🚀 KHỞI TẠO
# ==========================================
cap = cv2.VideoCapture("videos/buon_ngu.MOV")
detector = fm.FaceMeshDetector(maxFaces=1)

if not cap.isOpened():
    print(f"❌ LỖI: Không thể mở video")
    exit()

# Chuẩn bị file CSV
file_exists = os.path.isfile(OUTPUT_FILE)
file = open(OUTPUT_FILE, 'a', newline='')
writer = csv.writer(file)

if not file_exists:
    writer.writerow(['EAR', 'MAR', 'Pitch', 'Yaw', 'Roll', 'Label'])

# Đọc dữ liệu cũ nếu có
LABEL_0_COUNT = 0
LABEL_1_COUNT = 0
if file_exists:
    try:
        import pandas as pd
        existing_data = pd.read_csv(OUTPUT_FILE)
        LABEL_0_COUNT = len(existing_data[existing_data['Label'] == 0])
        LABEL_1_COUNT = len(existing_data[existing_data['Label'] == 1])
        print(f"📊 Dữ liệu hiện có: Label 0 = {LABEL_0_COUNT}, Label 1 = {LABEL_1_COUNT}")
    except:
        pass

# Các hàm tính toán
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

def get_head_pose(face, img_w, img_h):
    face_2d = []
    face_3d = []
    key_points = [1, 199, 33, 263, 61, 291]
    
    for idx in key_points:
        x, y = face[idx][1], face[idx][2]
        face_2d.append([x, y])
        face_3d.append([x, y, 0])
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                          [0, focal_length, img_w / 2],
                          [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360
    
    return pitch, yaw, roll

def is_valid_face_data(ear, mar, pitch, yaw, roll):
    if not (0.08 <= ear <= 0.6):
        return False
    if not (0.03 <= mar <= 1.2):
        return False
    if abs(pitch) > 70 or abs(yaw) > 70 or abs(roll) > 50:
        return False
    return True

id_mat_trai = [33, 160, 158, 133, 153, 144]
id_mieng = [78, 308, 13, 14]

print("=" * 50)
print("📹 BẮT ĐẦU THU THẬP DỮ LIỆU")
print("=" * 50)
print("Nhấn '0' để gán label TỈNH TÁO cho TOÀN BỘ VIDEO")
print("Nhấn '1' để gán label BUỒN NGỦ cho TOÀN BỘ VIDEO")
print("(Chỉ cần nhấn 1 lần, label sẽ áp dụng cho tất cả frame)")
print("Nhấn SPACE để tạm dừng/tiếp tục")
print("Nhấn 'q' để thoát")
print("=" * 50)

current_label = None
frame_skip = 1  # Lưu mỗi frame (không bỏ sót)
frame_counter = 0
paused = False

# Thống kê
total_frames = 0
face_detected = 0
valid_data = 0
invalid_data = 0

while True:
    if not paused:
        success, img = cap.read()
        if not success:
            print("✅ Đã chạy hết video!")
            break
        total_frames += 1
    else:
        success = True
    
    if success:
        img_h, img_w, _ = img.shape
        img, face = detector.findFaceMesh(img, draw=False)
        
        # Hiển thị trạng thái
        if TARGET_COUNT is None:
            status_text = f"Label 0: {LABEL_0_COUNT} | Label 1: {LABEL_1_COUNT} (Không giới hạn)"
        else:
            status_text = f"Label 0: {LABEL_0_COUNT}/{TARGET_COUNT} | Label 1: {LABEL_1_COUNT}/{TARGET_COUNT}"
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if paused:
            cv2.putText(img, "⏸️ PAUSED - Press SPACE to continue", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset = 90
        else:
            y_offset = 60
        
        if current_label is not None:
            label_text = f"Current Label: {current_label} ({'Tỉnh táo' if current_label == 0 else 'Buồn ngủ'})"
            color = (0, 255, 0) if current_label == 0 else (0, 0, 255)
            cv2.putText(img, label_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if face:
            face_detected += 1
            ear = calculate_EAR(face, id_mat_trai)
            mar = calculate_MAR(face, id_mieng)
            pitch, yaw, roll = get_head_pose(face, img_w, img_h)
            
            cv2.putText(img, f"EAR: {ear:.3f} | MAR: {mar:.3f}", (10, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            is_valid = is_valid_face_data(ear, mar, pitch, yaw, roll)
            if is_valid:
                valid_data += 1
                status_color = (0, 255, 0)
                status_text = "Data: VALID"
            else:
                invalid_data += 1
                status_color = (0, 0, 255)
                status_text = "Data: INVALID"
            
            cv2.putText(img, status_text, (10, y_offset + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # ⭐ SỬA: Tự động lưu nếu có label và dữ liệu hợp lệ (bỏ điều kiện TARGET_COUNT)
            if not paused and current_label is not None and frame_counter % frame_skip == 0:
                if is_valid:
                    # Kiểm tra TARGET_COUNT chỉ khi nó không phải None
                    should_save = True
                    if TARGET_COUNT is not None:
                        if (current_label == 0 and LABEL_0_COUNT >= TARGET_COUNT) or \
                           (current_label == 1 and LABEL_1_COUNT >= TARGET_COUNT):
                            should_save = False
                    
                    if should_save:
                        writer.writerow([ear, mar, pitch, yaw, roll, current_label])
                        if current_label == 0:
                            LABEL_0_COUNT += 1
                        else:
                            LABEL_1_COUNT += 1
                        if (LABEL_0_COUNT + LABEL_1_COUNT) % 10 == 0:
                            print(f"✅ Đã lưu: Label {current_label} | Tổng: {LABEL_0_COUNT + LABEL_1_COUNT}")
            
            if not paused:
                frame_counter += 1
        else:
            cv2.putText(img, "No Face Detected", (10, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Resize video xuống 0.3 lần kích thước gốc để hiển thị
    img_display = cv2.resize(img, None, fx=0.3, fy=0.3)
    cv2.imshow("Thu thập dữ liệu - Nhấn 0/1 để chọn label, SPACE để pause, q để thoát", img_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('0'):
        current_label = 0
        print("📌 Đã chọn Label 0 (Tỉnh táo) - Áp dụng cho TOÀN BỘ VIDEO")
    elif key == ord('1'):
        current_label = 1
        print("📌 Đã chọn Label 1 (Buồn ngủ) - Áp dụng cho TOÀN BỘ VIDEO")
    elif key == ord(' '):  # Phím SPACE
        paused = not paused
        if paused:
            print("⏸️ Đã tạm dừng - Nhấn SPACE để tiếp tục")
        else:
            print("▶️ Đã tiếp tục")
    elif key == ord('q'):
        break
    
    # ⭐ SỬA: Chỉ kiểm tra TARGET_COUNT nếu không phải None
    if TARGET_COUNT is not None:
        if LABEL_0_COUNT >= TARGET_COUNT and LABEL_1_COUNT >= TARGET_COUNT:
            print("🎉 Đã thu thập đủ dữ liệu!")
            break

file.close()
cap.release()
cv2.destroyAllWindows()

# In thống kê
print("\n" + "="*50)
print("📊 THỐNG KÊ")
print("="*50)
print(f"Tổng số frame: {total_frames}")
print(f"Frame có face: {face_detected} ({face_detected/total_frames*100:.1f}%)")
print(f"Dữ liệu hợp lệ: {valid_data} ({valid_data/face_detected*100:.1f}% nếu có face)")
print(f"Dữ liệu không hợp lệ: {invalid_data}")
print(f"\n✅ Hoàn thành! Tổng: {LABEL_0_COUNT + LABEL_1_COUNT} dòng dữ liệu")
print(f"   - Label 0: {LABEL_0_COUNT} dòng")
print(f"   - Label 1: {LABEL_1_COUNT} dòng")