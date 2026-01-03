import cv2
import csv
import math
import os
import numpy as np  # <--- CẦN THÊM CÁI NÀY
import FaceMeshModule as fm

# ==========================================
# ⚙️ CẤU HÌNH
# ==========================================
VIDEO_PATH = "videos/tinh_tao.MOV"   
LABEL_TO_SAVE = 0             
OUTPUT_FILE = "dataset_full.csv" # Đổi tên file để phân biệt

# ==========================================
# 🚀 BẮT ĐẦU XỬ LÝ
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
detector = fm.FaceMeshDetector(maxFaces=1)

if not cap.isOpened():
    print(f"❌ LỖI: Không tìm thấy video: {VIDEO_PATH}")
    exit()

# Chuẩn bị file CSV (Thêm Pitch, Yaw, Roll)
file_exists = os.path.isfile(OUTPUT_FILE)
file = open(OUTPUT_FILE, 'a', newline='')
writer = csv.writer(file)

# --- SỬA HEADER CSV ---
if not file_exists:
    # Thêm cột Pitch, Yaw, Roll
    writer.writerow(['EAR', 'MAR', 'Pitch', 'Yaw', 'Roll', 'Label'])

# --- CÁC HÀM TÍNH TOÁN ---
def findDistance(p1, p2):
    # p1, p2 format: [id, x, y]
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

# --- HÀM MỚI: TÍNH HEAD POSE (GÓC ĐẦU) ---
def get_head_pose(face, img_w, img_h):
    # Các điểm mốc 2D quan trọng trên khuôn mặt (Pixel)
    # 1: Mũi, 199: Cằm, 33: Mắt trái, 263: Mắt phải, 61: Miệng trái, 291: Miệng phải
    # Lưu ý: face[id] trả về [id, x, y]
    
    face_2d = []
    face_3d = []
    
    # Danh sách các điểm mốc để tính toán tư thế đầu
    key_points = [1, 199, 33, 263, 61, 291]
    
    for idx in key_points:
        # Lấy toạ độ x, y từ face list
        x, y = face[idx][1], face[idx][2]
        face_2d.append([x, y])
        face_3d.append([x, y, 0]) # Giả định z=0 cho điểm 3D tương ứng ban đầu

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Ma trận Camera giả lập
    focal_length = 1 * img_w
    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])

    # Ma trận biến dạng (giả sử bằng 0)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Giải bài toán PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Chuyển vector xoay thành ma trận
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Lấy các góc Euler
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Đổi sang độ và gán tên cho dễ hiểu
    pitch = angles[0] * 360  # Gật lên/xuống (X)
    yaw = angles[1] * 360    # Quay trái/phải (Y)
    roll = angles[2] * 360   # Nghiêng đầu (Z)

    return pitch, yaw, roll

# ID Landmarks
id_mat_trai = [33, 160, 158, 133, 153, 144]
id_mieng = [78, 308, 13, 14]

print(f"--- ĐANG CHẠY ---")
count = 0

while True:
    success, img = cap.read()
    if not success:
        print("✅ Đã chạy hết video!")
        break 

    img_h, img_w, _ = img.shape # Lấy kích thước ảnh
    img, face = detector.findFaceMesh(img, draw=False)

    if face:
        # 1. Tính EAR & MAR
        ear = calculate_EAR(face, id_mat_trai)
        mar = calculate_MAR(face, id_mieng)
        
        # 2. Tính HEAD POSE (MỚI)
        pitch, yaw, roll = get_head_pose(face, img_w, img_h)

        # 3. GHI FULL DỮ LIỆU VÀO CSV
        writer.writerow([ear, mar, pitch, yaw, roll, LABEL_TO_SAVE])
        count += 1
        
        # Hiện thông số (để check)
        cv2.putText(img, f"P: {pitch:.1f}, Y: {yaw:.1f}", (30, 110), 
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    # Resize để xem cho dễ
    img_nho = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) 
    cv2.imshow("Xu ly Video Full", img_nho)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
print(f"🎉 XONG! Dữ liệu (gồm cả Pitch/Yaw) đã lưu vào '{OUTPUT_FILE}'.")