import cv2
import time
import numpy as np
import winsound
import pyttsx3
import threading
from face_mesh_detector import FaceMeshDetector
from drowsiness_detector import DrowsinessDetector

# --- CẤU HÌNH ---
WEBCAM_ID = 0
SHOW_DETAILS = True
ALARM_FREQUENCY = 2500 
ALARM_DURATION = 500    

# Performance tuning
PROCESS_WIDTH = 640        # Lower resolution for processing to speed up detection
SKIP_FRAMES = 2            # Process detection every N frames (set 1 for every frame)

# Số frame dùng để căn chỉnh lúc đầu
CALIBRATION_FRAMES = 50 

def draw_pose_axis(img, pitch, yaw, roll, tdx, tdy, size=50):
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    x1 = size * (np.cos(yaw_rad) * np.cos(roll_rad))
    y1 = size * (np.cos(pitch_rad) * np.sin(roll_rad) + np.cos(roll_rad) * np.sin(pitch_rad) * np.sin(yaw_rad))
    x2 = size * (-np.cos(yaw_rad) * np.sin(roll_rad))
    y2 = size * (-np.cos(pitch_rad) * np.cos(roll_rad) + np.sin(pitch_rad) * np.sin(yaw_rad) * np.sin(roll_rad))
    x3 = size * (np.sin(yaw_rad))
    y3 = size * (-np.cos(yaw_rad) * np.sin(pitch_rad))

    tdx, tdy = int(tdx), int(tdy)
    cv2.arrowedLine(img, (tdx, tdy), (int(tdx+x1), int(tdy+y1)), (0,0,255), 2)
    cv2.arrowedLine(img, (tdx, tdy), (int(tdx+x2), int(tdy+y2)), (0,255,0), 2)
    cv2.arrowedLine(img, (tdx, tdy), (int(tdx+x3), int(tdy+y3)), (255,0,0), 2)

def main():
    # Try opening camera with DirectShow backend on Windows to avoid driver hangs
    try:
        cap = cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(WEBCAM_ID)

    # Fallback if DirectShow failed
    if not cap.isOpened():
        cap = cv2.VideoCapture(WEBCAM_ID)

    # Wait briefly for camera initialization
    t0 = time.time()
    while not cap.isOpened() and time.time() - t0 < 5:
        time.sleep(0.2)

    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open camera {WEBCAM_ID}")
        return

    # Set camera properties safely (some drivers may hang here)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(PROCESS_WIDTH * 9 / 16))
    except KeyboardInterrupt:
        print("Interrupted while setting camera properties")
        cap.release()
        return
    except Exception as e:
        print(f"Warning: failed to set camera properties: {e}")

    detector = FaceMeshDetector(maxFaces=1)
    drowsiness = DrowsinessDetector()

    # --- TTS for YAWNING ---
    engine = pyttsx3.init()
    try:
        engine.setProperty('rate', 150)
    except Exception:
        pass
    last_voice_time = 0
    VOICE_COOLDOWN = 10  # seconds
    def speak(text):
        def _s():
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=_s, daemon=True).start()

    # Threaded non-blocking beep to avoid blocking the main loop
    def play_beep(frequency, duration):
        def _b():
            try:
                winsound.Beep(frequency, duration)
            except Exception:
                pass
        threading.Thread(target=_b, daemon=True).start()

    pTime = 0
    alarm_last_time = 0 
    
    # Biến cho quá trình căn chỉnh
    calib_frame_count = 0
    pitch_list = []
    yaw_list = []
    is_calibrated = False

    # Frame skipping cache
    frame_idx = 0
    last_landmarks = []
    last_result = None
    
    # Biến theo dõi trạng thái ngáp - nhắc lại mỗi 5 giây
    is_yawning = False
    last_yawn_reminder_time = 0
    YAWN_REMINDER_INTERVAL = 5.0  # giây

    print("--- SYSTEM STARTED ---")
    
    while True:
        success, img = cap.read()
        if not success:
            print("Cannot read frame")
            break
        
        height, width, _ = img.shape

        # Detection only every SKIP_FRAMES frames
        do_detect = (frame_idx % SKIP_FRAMES == 0) or (not last_landmarks)
        if do_detect:
            _, face_landmarks = detector.findFaceMesh(img, draw=False)
            last_landmarks = face_landmarks
            if len(face_landmarks) != 0:
                # Compute fresh result
                result = drowsiness.detect(face_landmarks, img.shape)
                last_result = result
            else:
                result = last_result
        else:
            face_landmarks = last_landmarks
            result = last_result

        # Xử lý logic
        if result is not None and len(face_landmarks) != 0:
            
            # --- GIAI ĐOẠN 1: CĂN CHỈNH (CALIBRATION) ---
            if not is_calibrated:
                # Lấy dữ liệu thô để tính trung bình
                # Gọi hàm detect nhưng chỉ để lấy raw pitch/yaw
                pitch_list.append(result["pitch"])
                yaw_list.append(result["yaw"])
                # Collect EAR for baseline if available
                ear_val = result.get("ear")
                if ear_val is not None and not np.isnan(ear_val):
                    if 'ear_list' not in locals():
                        ear_list = []
                    ear_list.append(ear_val)
                calib_frame_count += 1
                
                # Hiển thị thanh loading
                cv2.rectangle(img, (0, 0), (width, height), (50, 50, 50), -1)
                cv2.putText(img, f"CALIBRATING... {calib_frame_count}/{CALIBRATION_FRAMES}", 
                            (width//2 - 200, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, "LOOK STRAIGHT AHEAD", (width//2 - 180, height//2 + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if calib_frame_count >= CALIBRATION_FRAMES:
                    # Tính trung bình và set làm mốc
                    avg_pitch = sum(pitch_list) / len(pitch_list)
                    avg_yaw = sum(yaw_list) / len(yaw_list)
                    avg_ear = None
                    if 'ear_list' in locals() and len(ear_list) > 0:
                        avg_ear = sum(ear_list) / len(ear_list)
                    drowsiness.set_calibration(avg_pitch, avg_yaw, avg_ear)
                    is_calibrated = True
                    play_beep(1000, 200) # Bíp nhẹ báo xong (non-blocking)

            # --- GIAI ĐOẠN 2: HOẠT ĐỘNG CHÍNH ---
            else:
                # Use cached result when skipping frames to improve performance
                if last_result is None:
                    result = drowsiness.detect(face_landmarks, img.shape)
                    last_result = result
                else:
                    result = last_result

                ear = result["ear"]
                mar = result["mar"]
                pitch = result["pitch"]
                yaw = result["yaw"]
                status = result["status"]
                color = result["color"]
                alarm = result["alarm"]
                nose = result["nose_point"]

                if SHOW_DETAILS:
                    draw_pose_axis(img, pitch, yaw, 0, nose[0], nose[1])

                    # Compact semi-transparent HUD (top-left)
                    hud_x, hud_y, hud_w, hud_h = 10, 10, 260, 90
                    overlay = img.copy()
                    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
                    alpha = 0.45
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                    # Nicely formatted values (avoid printing NaN)
                    ear_text = f"{ear:.2f}" if not (ear is None or (isinstance(ear, float) and np.isnan(ear))) else "N/A"
                    mar_text = f"{mar:.2f}"
                    pitch_text = f"{int(pitch)}"

                    cv2.putText(img, f'{status}', (hud_x + 8, hud_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(img, f'EAR: {ear_text}   MAR: {mar_text}', (hud_x + 8, hud_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(img, f'Pitch: {pitch_text}', (hud_x + 8, hud_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # Yawn detection: Nhắc nhở mỗi 5 giây khi đang ngáp
                    if status == "YAWNING":
                        current_time = time.time()
                        if not is_yawning:
                            # Bắt đầu ngáp lần này -> Nhắc nhở ngay
                            is_yawning = True
                            last_yawn_reminder_time = current_time
                            speak("You seem tired. Please focus.")
                        elif current_time - last_yawn_reminder_time >= YAWN_REMINDER_INTERVAL:
                            # Vẫn đang ngáp sau 5 giây -> Nhắc nhở lại
                            last_yawn_reminder_time = current_time
                            speak("You seem tired. Please focus.")
                    else:
                        # Không còn ngáp nữa -> reset để sẵn sàng cho lần ngáp tiếp theo
                        is_yawning = False

                    # Only play alarm beep for true alarms (not yawning)
                    if alarm and status != "YAWNING":
                         cv2.putText(img, "WARNING!", (width//2 - 200, height//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
                         current_time = time.time()
                         if current_time - alarm_last_time > 1.0: 
                             play_beep(ALARM_FREQUENCY, ALARM_DURATION)
                             alarm_last_time = current_time

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        cv2.putText(img, f'FPS: {int(fps)}', (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Driver Monitoring System", img)
        
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()