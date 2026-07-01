import os
import cv2
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(BASE_DIR, "sorted_frames_by_prediction")

def view_predictions():
    if not os.path.exists(TARGET_DIR):
        print(f"[❌] LỖI: Không tìm thấy thư mục: {TARGET_DIR}")
        return

    print("=========================================")
    print("👁️ CÔNG CỤ XEM ẢNH DỰ ĐOÁN (CÓ BACK/NEXT) 👁️")
    print("=========================================\n")
    print("Phím tắt khi đang xem ảnh:")
    print("  [f] hoặc [y] : TIẾN LÊN (Next Set)")
    print("  [b] : QUAY LẠI (Back Set)")
    print("  [n] : XÓA BỎ (Delete Set - xóa luôn folder)")
    print("  [l] : BỎ QUA NHÃN (Chuyển sang Nhãn dự đoán tiếp theo)")
    print("  [q] : THOÁT (Quit)")
    print("=========================================\n")

    # Lấy danh sách các nhãn dự đoán (thường là awake, sleep, microsleep)
    labels = sorted([d for d in os.listdir(TARGET_DIR) if os.path.isdir(os.path.join(TARGET_DIR, d))])
    
    if not labels:
        print(f"Thư mục {TARGET_DIR} đang trống!")
        return

    for label in labels:
        print(f"\n🚀 ĐANG XEM NHÃN DỰ ĐOÁN: {label.upper()}")
        label_dir = os.path.join(TARGET_DIR, label)
        skip_label = False

        # Lấy danh sách các set hợp lệ
        valid_sets = sorted([s for s in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, s))])
        
        current_idx = 0
        while current_idx < len(valid_sets):
            if skip_label:
                break
                
            set_folder = valid_sets[current_idx]
            set_path = os.path.join(label_dir, set_folder)

            # Lỡ set này bị em bấm [n] xóa từ trước đó rồi thì bỏ qua
            if not os.path.exists(set_path):
                current_idx += 1
                continue

            images = sorted([img for img in os.listdir(set_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if not images:
                current_idx += 1
                continue

            decision_made = False
            while not decision_made:
                for img_name in images:
                    img_path = os.path.join(set_path, img_name)
                    frame = cv2.imread(img_path)
                    if frame is None: continue

                    # Phóng to ảnh
                    frame = cv2.resize(frame, (0, 0), fx=2.0, fy=2.0)
                    
                    # Ghi chữ lên màn hình
                    cv2.putText(frame, f"Predict: {label.upper()} | Set: {set_folder}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, "Keys: [f]Next | [b]Back | [n]Del | [l]Skip Lbl | [q]Quit", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    cv2.imshow("Prediction Viewer", frame)
                    
                    # Chờ 100ms để tạo hiệu ứng video lặp
                    key = cv2.waitKey(100) & 0xFF
                    
                    if key == ord('y') or key == ord('f'):
                        print(f"⏭️ [NEXT] Chuyển tới set sau: {set_folder}")
                        current_idx += 1      # Tăng index để NEXT
                        decision_made = True
                        break
                    elif key == ord('b'):
                        print(f"⏮️ [BACK] Quay lại set trước")
                        current_idx = max(0, current_idx - 1) # Giảm index để LÙI (không cho âm)
                        decision_made = True
                        break
                    elif key == ord('n'):
                        print(f"🗑️ [ĐÃ XÓA] Set dự đoán sai: {set_folder}")
                        shutil.rmtree(set_path)
                        current_idx += 1      # Xóa xong tự động NEXT
                        decision_made = True
                        break
                    elif key == ord('l'):
                        print(f"⏭️ [BỎ QUA NHÃN] Nhảy sang nhãn tiếp theo...")
                        skip_label = True
                        decision_made = True
                        break
                    elif key == ord('q'):
                        print("\n🛑 Đã dừng chương trình xem ảnh dự đoán.")
                        cv2.destroyAllWindows()
                        return
                        
                if decision_made:
                    break # Phá vòng lặp for frame để vòng while tiếp tục duyệt set mới

    cv2.destroyAllWindows()
    print("\n🎉 HOÀN TẤT DUYỆT ẢNH DỰ ĐOÁN!")

if __name__ == "__main__":
    view_predictions()