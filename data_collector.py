"""
THU THAP DU LIEU - SCRIPT CUOI CUNG (FIXED FOR 478 LANDMARKS)
=================================================================
Nhan phim:
- '0' = TINH TAO (Awake) 
- '1' = BUON NGU (Drowsy - nham mat)
- '2' = CUI XUONG (Phone - dau cui)
- '3' = NGU GAT (Microsleep - dau nghieng nham mat)
- 'q' = Thoat

GIU PHIM de ghi du lieu lien tuc!
"""

import cv2
import csv
import os
from face_mesh_detector import FaceMeshDetector

# Cau hinh
OUTPUT_FILE = "face_data.csv"
NUM_LANDMARKS = 478  # MediaPipe Face Mesh tra ve 478 landmarks!

CLASS_INFO = {
    0: {"name": "TINH TAO", "desc": "Awake - Ngoi thang, mat mo", "color": (0, 255, 0)},
    1: {"name": "BUON NGU", "desc": "Drowsy - Nham mat, buon ngu", "color": (0, 0, 255)},
    2: {"name": "CUI XUONG", "desc": "Phone - Dau cui xuong", "color": (0, 255, 255)},
    3: {"name": "NGU GAT", "desc": "Microsleep - Dau nghieng, ngu gat", "color": (255, 0, 255)}
}

def init_csv():
    """Tao file CSV moi voi header dung"""
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(NUM_LANDMARKS):
            header.extend([f'x{i}', f'y{i}'])
        writer.writerow(header)
    print(f"[OK] Da tao file: {OUTPUT_FILE}")
    print(f"[OK] Header: 1 label + {NUM_LANDMARKS} landmarks * 2 = {1 + NUM_LANDMARKS * 2} cot")

def save_data(landmarks, label):
    """Luu du lieu vao CSV"""
    if len(landmarks) != NUM_LANDMARKS:
        print(f"[WARNING] Expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}!")
        return False
    
    with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        row = [label]
        for point in landmarks:
            row.extend([point[0], point[1]])
        writer.writerow(row)
    return True

def main():
    # Khoi tao
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = FaceMeshDetector(maxFaces=1)
    
    # Kiem tra file
    if os.path.exists(OUTPUT_FILE):
        print(f"\n[WARNING] File {OUTPUT_FILE} da ton tai!")
        response = input("Ban co muon XOA va bat dau lai? (y/n): ")
        if response.lower() == 'y':
            os.remove(OUTPUT_FILE)
            init_csv()
        else:
            print("[OK] Se them du lieu vao file hien tai...")
    else:
        init_csv()
    
    # Dem so mau cho moi class
    count = {0: 0, 1: 0, 2: 0, 3: 0}
    current_label = None
    
    print("\n" + "="*70)
    print("  THU THAP DU LIEU - DRIVER MONITORING SYSTEM")
    print("="*70)
    print("\nPHIM BAM:")
    for label, info in CLASS_INFO.items():
        print(f"  '{label}' = {info['name']:12} - {info['desc']}")
    print("  'q' = Thoat")
    print("\n[CHU Y] GIU PHIM de ghi du lieu lien tuc!")
    print("="*70 + "\n")
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("[ERROR] Khong doc duoc camera!")
            break
        
        # Detect face landmarks
        img, face_landmarks = detector.findFaceMesh(img, draw=True)
        
        # Xu ly phim bam
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('0'):
            current_label = 0
        elif key == ord('1'):
            current_label = 1
        elif key == ord('2'):
            current_label = 2
        elif key == ord('3'):
            current_label = 3
        else:
            current_label = None
        
        # Hien thi thong tin
        y_pos = 30
        cv2.rectangle(img, (0, 0), (700, 200), (0, 0, 0), -1)
        
        # Hien thi so landmarks
        if len(face_landmarks) > 0:
            cv2.putText(img, f"Landmarks: {len(face_landmarks)}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
        else:
            cv2.putText(img, "No face detected!", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
        
        # Hien thi trang thai
        if current_label is not None and len(face_landmarks) == NUM_LANDMARKS:
            info = CLASS_INFO[current_label]
            cv2.putText(img, f"RECORDING: {info['name']}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, info['color'], 2)
            y_pos += 35
            
            # Luu du lieu
            if save_data(face_landmarks, current_label):
                count[current_label] += 1
        else:
            cv2.putText(img, "Press 0/1/2/3 to record", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 35
        
        # Hien thi so mau
        for label, info in CLASS_INFO.items():
            cv2.putText(img, f"{label}-{info['name']}: {count[label]}", (10, y_pos), 
                       cv2.FONT_HERSHEY_PLAIN, 1.2, info['color'], 2)
            y_pos += 25
        
        cv2.imshow("Data Collection - Press Q to Quit", img)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Thong ke cuoi cung
    print("\n" + "="*70)
    print("  KET THUC THU THAP DU LIEU")
    print("="*70)
    total = sum(count.values())
    print(f"\nTONG CONG: {total} mau")
    for label, info in CLASS_INFO.items():
        print(f"  {info['name']:12}: {count[label]:4} mau")
    print(f"\nDu lieu da luu vao: {OUTPUT_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()

