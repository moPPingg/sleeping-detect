import cv2                # Thư viện xử lý hình ảnh (OpenCV)
import mediapipe as mp    # Thư viện AI của Google (xử lý tay, mặt, dáng...)
import time               # Thư viện đo thời gian (để tính FPS)

# Mở cam
cap = cv2.VideoCapture(0)
pTime = 0  # Để tính FPS


mpDraw = mp.solutions.drawing_utils   # Chuẩn bị bút
mpFaceMesh = mp.solutions.face_mesh   # Lấy module Face Mesh từ kho
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1) # tạo bot chỉ tìm 1 mặt 

# cài đặt nét vẽ (độ dày nét, bán kính chấm tròn)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)


while True:
    # 1. Đọc ảnh từ Camera
    success, img = cap.read()

    if not success:
        print("End of video or cannot read frame")
        break
    
    # Chuyển hệ màu vì opencv và mediapipe dùng hệ màu khác nhau
    # OpenCV dùng màu BGR (Xanh-Lục-Đỏ), còn AI của Google cần RGB (Đỏ-Lục-Xanh)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Cho AI quét khuôn mặt
    # Kết quả trả về (tọa độ các điểm) sẽ lưu vào biến 'results'
    results = faceMesh.process(imgRGB)
    
    # 4. Xử lý kết quả (Nếu tìm thấy mặt)
    if results.multi_face_landmarks:
        # Duyệt qua từng khuôn mặt tìm được
        for faceLms in results.multi_face_landmarks:
            
            # vẽ lưới lên mặt
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                  drawSpec, drawSpec) 
            # FACEMESH_TESSELATION: dùng để vẽ lưới
            
            # lấy tọa độ
            # faceLms.landmark: Chứa danh sách 468 điểm
            # id: Số thứ tự (0 là môi, 1 là mũi, 33 là mắt...)
            # lm: Chứa tọa độ x, y dạng % (0.5, 0.7...)
            for id, lm in enumerate(faceLms.landmark):
                
                # Lấy kích thước ảnh thật (cao, ngang, màu)
                ih, iw, ic = img.shape
                
                # Đổi từ số % (0.5) sang số Pixel (ví dụ 640px) 
# Nếu không đổi sang pixel thì không dùng được vì thực tế để biết mắt nhắm hay mở là tính khoảng cách mí trên và mí dưới nếu không đổi thì số rất nhỏ không hợp lý
                # Công thức: Tọa độ % * Kích thước thật
                x, y = int(lm.x * iw), int(lm.y * ih)
                #Mai mốt mình sẽ lọc lấy ID mắt ở đây
                print(id, x, y) 
                
    
    #Tính FPS
    cTime = time.time()       # Thời gian hiện tại
    fps = 1 / (cTime - pTime) # Công thức: 1 giây / thời gian xử lý 1 ảnh
    pTime = cTime             # Cập nhật lại thời gian cũ
    
    # Vẽ số FPS lên góc màn hình
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), 
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break