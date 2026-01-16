import cv2
import numpy as np
import os
import urllib.request

class GlassesDetector:
    def __init__(self):
        self.cascade_path = 'haarcascade_eye_tree_eyeglasses.xml'
        self.cascade = None
        self._load_cascade()
    
    def _load_cascade(self):
        # Tự động tải file xml chuẩn nếu chưa có
        if not os.path.exists(self.cascade_path):
            print(f"⏳ Đang tải {self.cascade_path}...")
            url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
            try:
                urllib.request.urlretrieve(url, self.cascade_path)
            except:
                print("❌ Không tải được file XML kính!")
                return

        self.cascade = cv2.CascadeClassifier(self.cascade_path)
    
    def detect_glasses(self, img, face_landmarks):
        if self.cascade is None or self.cascade.empty():
            return False
            
        h, w, c = img.shape
        
        # 1. Cắt vùng mắt rộng hơn một chút để lấy hết gọng kính
        # Dùng các điểm landmark mắt để định vị
        try:
            x_min = int(face_landmarks[33][1] * w) # Mắt trái
            x_max = int(face_landmarks[263][1] * w) # Mắt phải
            y_min = int(face_landmarks[10][2] * h)  # Đỉnh trán
            y_max = int(face_landmarks[152][2] * h) # Cằm
            
            # Padding an toàn
            pad = 20
            x_min = max(0, x_min - pad)
            x_max = min(w, x_max + pad)
            y_min = max(0, y_min)
            # Lấy nửa trên khuôn mặt thôi (đỡ bị nhầm với miệng)
            y_max = min(h, y_min + int((y_max - y_min) * 0.6)) 
            
            roi = img[y_min:y_max, x_min:x_max]
            if roi.size == 0: return False

            # 2. Xử lý ảnh: Tăng độ tương phản tối đa (Histogram Equalization)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray) 
            
            # Làm mờ nhẹ để loại bỏ nhiễu da
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # 3. Detect với tham số NHẠY HƠN
            # scaleFactor=1.01 (Quét cực kỹ), minNeighbors=3 (Dễ tính hơn)
            glasses = self.cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=3, minSize=(30, 30))
            
            # Vẽ khung xanh quanh cái kính nó tìm thấy (DEBUG)
            for (gx, gy, gw, gh) in glasses:
                cv2.rectangle(img, (x_min+gx, y_min+gy), (x_min+gx+gw, y_min+gy+gh), (255, 255, 0), 1)

            return len(glasses) > 0
        except:
            return False