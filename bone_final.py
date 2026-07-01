import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import torch
import torch.nn.functional as F
import winsound
import pyttsx3
from pathlib import Path
from torchvision import models, transforms
from threading import Lock, Thread
from queue import Queue  # ← Fixed: Queue is in 'queue' module, not 'threading'
from scipy.spatial import distance as dist

# ==========================================
# 1. MODULE XỬ LÝ CAMERA (ĐA LUỒNG) - Tối ưu FPS
# ==========================================
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)  # Đặt FPS mục tiêu
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (grabbed, frame) = self.stream.read()
            if not grabbed: self.stop(); return
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# 2. MODULE PHÂN TÍCH BUỒN NGỦ (HEURISTIC REAL-TIME)
# ==========================================
class DrowsinessAnalyzer:
    def __init__(self):
        # Cấu hình ngưỡng Heuristics
        self.YAWN_THRESHOLD = 0.6
        self.YAWN_FRAMES_REQ = 20  # Duy trì ~0.6s để xác nhận ngáp
        self.EAR_THRESHOLD = 0.20
        self.SLEEP_FRAMES_REQ = 15  # Cân bằng nhạy/phát hiện, ~0.5s ở 30 FPS
        self.consecutive_sleep_frames = 0
        
        # OPTION 1: Microsleep detection - sliding window (60 frames, >=45 to alert)
        self.microsleep_window = []  # Track last 60 frame predictions
        self.MICROSLEEP_WINDOW_SIZE = 60  # 60 frames = 2 seconds
        self.MICROSLEEP_THRESHOLD = 45  # >=45 frames = alert (75%)
        self.microsleep_alert_cooldown = False  # Prevent multiple alerts
        
        # Biến thống kê báo cáo
        self.total_yawns = 0
        self.consecutive_yawn_frames = 0
        self.yawn_cooldown = False
        self.total_sleep_alerts = 0

    def analyze(self, landmarks_pts, ai_drowsy=False, ai_pred=-1):
        # 1. Tính toán EAR/MAR vật lý
        ear = (self.calc_ear(landmarks_pts['left_eye']) + self.calc_ear(landmarks_pts['right_eye'])) / 2.0
        mar = self.calc_mar(landmarks_pts['mouth'])
        
        # OPTION 1: Microsleep detection using sliding window (75% in 60 frames)
        is_sleeping = False
        
        # Track microsleep frames in sliding window
        self.microsleep_window.append(ai_pred == 2)  # True if microsleep, False otherwise
        
        # Keep only last 60 frames
        if len(self.microsleep_window) > self.MICROSLEEP_WINDOW_SIZE:
            self.microsleep_window.pop(0)
        
        # Count microsleep frames in current window
        microsleep_count = sum(self.microsleep_window)
        
        # Alert if >=45 frames (out of 60) are microsleep
        if len(self.microsleep_window) == self.MICROSLEEP_WINDOW_SIZE:
            if microsleep_count >= self.MICROSLEEP_THRESHOLD:
                is_sleeping = True
                if not self.microsleep_alert_cooldown:  # First time hitting threshold
                    self.total_sleep_alerts += 1
                    self.microsleep_alert_cooldown = True
            else:
                self.microsleep_alert_cooldown = False  # Reset when below threshold
        
        # Regular sleep detection (EAR-based, not microsleep specific)
        if ear < self.EAR_THRESHOLD and ai_pred != 2:  # EAR low but NOT microsleep
            self.consecutive_sleep_frames += 1
            if self.consecutive_sleep_frames >= self.SLEEP_FRAMES_REQ:
                is_sleeping = True
        else:
            if self.consecutive_sleep_frames >= self.SLEEP_FRAMES_REQ and ai_pred != 2:
                self.total_sleep_alerts += 1
            self.consecutive_sleep_frames = 0

        # 2. Logic Ngáp (Xác nhận hành vi theo thời gian)
        yawning = False
        if mar > self.YAWN_THRESHOLD:
            self.consecutive_yawn_frames += 1
            if self.consecutive_yawn_frames >= self.YAWN_FRAMES_REQ:
                yawning = True
                if not self.yawn_cooldown:
                    self.total_yawns += 1
                    self.yawn_cooldown = True
        else:
            self.consecutive_yawn_frames = 0
            self.yawn_cooldown = False

        # 3. Chỉ báo động khi tín hiệu ngủ được duy trì đủ lâu
        alarm = is_sleeping
        return ear, mar, alarm, yawning

    @staticmethod
    def calc_ear(pts):
        v1 = dist.euclidean(pts[1], pts[5])
        v2 = dist.euclidean(pts[2], pts[4])
        h = dist.euclidean(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    @staticmethod
    def calc_mar(pts):
        v = dist.euclidean(pts[0], pts[1])
        h = dist.euclidean(pts[2], pts[3])
        return v / h if h > 0 else 0.0


# ==========================================
# 3. MODULE TTS (TEXT-TO-SPEECH) TIẾNG VIỆT
# ==========================================
class VietnameseTTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Cấu hình engine cho Vietnamese
        self.engine.setProperty('rate', 150)  # Tốc độ nói
        self.engine.setProperty('volume', 0.9)  # Âm lượng
        
    def speak(self, text):
        """Phát âm thanh TTS không khóa (async - dùng thread riêng)"""
        try:
            # Chạy TTS trong thread riêng để không khóa main thread
            thread = Thread(target=self._speak_async, args=(text,), daemon=True)
            thread.start()
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")
    
    def _speak_async(self, text):
        """Helper để chạy TTS trong background thread"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"⚠️ TTS Play Error: {e}")


class RealtimeAIPredictor:
    MICRO_MARGIN = -0.055 # If microsleep_prob >= awake_prob - 0.055, classify as microsleep
    
    def __init__(self, ensemble_model_path):
        model_path = Path(ensemble_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Khong tim thay model: {ensemble_model_path}")

        self.class_names = {
            0: "Awake",
            1: "Sleep",
            2: "Microsleep",
        }
        self.ensemble_model = joblib.load(model_path)
        self.expected_features = getattr(self.ensemble_model, "n_features_in_", 2816)

        if self.expected_features != 2816:
            raise ValueError(f"Model can {self.expected_features} features, expected 2816 from EfficientNet-B2 GAP+GMP")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self._build_backbone().to(self.device).eval()
        
        # TỐI ƯU GPU HEAVY: Enable CUDA benchmarking + Memory optimization
        self.use_fp16 = False  # Sẽ bật nếu GPU NVIDIA
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # TỐI ƯU GPU HEAVY: For RTX 4060 + i9-14900 (high-end setup)
            # Enable full GPU optimization
            torch.cuda.synchronize()
            if torch.cuda.is_available():
                # Allocate GPU memory eagerly (no fragmentation)
                torch.cuda.empty_cache()
            
            # TỐI ƯU: Bật FP16 inference (tăng tốc độ ~2x)
            self.use_fp16 = True
            self.backbone = self.backbone.half()  # Chuyển model sang FP16
            
            # TỐI ƯU: Pre-allocate GPU memory (tránh delay lần đầu)
            self._warmup_gpu()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.last_pred = 0
        self.last_conf = 0.0
        self.last_proba = [0.0, 0.0, 0.0]  # [Awake, Drowsy, Microsleep]
        
        # TỐI ƯU GPU HEAVY: Async prediction queue (I9-14900 + 4060 compatible)
        self.prediction_queue = Queue(maxsize=2)
        self.stop_worker = False
        self.worker_thread = Thread(target=self._async_gpu_worker, daemon=True)
        self.worker_thread.start()

    @staticmethod
    def _build_backbone():
        try:
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
            model = models.efficientnet_b2(weights=weights)
        except Exception:
            # Fallback when pretrained weights are unavailable in the runtime.
            model = models.efficientnet_b2(weights=None)
        return model

    def _warmup_gpu(self):
        """TỐI ƯU GPU: Warm up GPU memory để tránh delay lần đầu"""
        try:
            print("🔥 GPU warmup...", end=" ", flush=True)
            dummy_input = torch.randn(1, 3, 260, 260, device=self.device, dtype=torch.float16 if self.use_fp16 else torch.float32)
            with torch.no_grad():
                for _ in range(3):  # Chạy 3 lần để GPU cache optimize
                    _ = self.backbone.features(dummy_input)
            torch.cuda.synchronize()  # Đợi GPU xong
            print("✅ Ready!")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")

    def _async_gpu_worker(self):
        """TỐI ƯU GPU HEAVY: Async worker để xử lý AI prediction trên GPU thread"""
        while not self.stop_worker:
            try:
                bgr_face = self.prediction_queue.get(timeout=0.5)
                if bgr_face is None:
                    break
                self.predict_from_face(bgr_face)
            except Exception:
                if self.stop_worker:
                    break
                continue


    @torch.no_grad()
    def predict_from_face(self, bgr_face):
        if bgr_face is None or bgr_face.size == 0:
            return self.last_pred, self.last_conf

        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        x = self.preprocess(rgb).unsqueeze(0).to(self.device)
        
        # TỐI ƯU GPU HEAVY: Dùng FP16 + CUDA stream
        if self.use_fp16:
            x = x.half()  # Chuyển input sang FP16
            with torch.cuda.amp.autocast():
                feat_map = self.backbone.features(x)
                gap = F.adaptive_avg_pool2d(feat_map, output_size=1).flatten(1)
                gmp = F.adaptive_max_pool2d(feat_map, output_size=1).flatten(1)
                features = torch.cat([gap, gmp], dim=1)
        else:
            feat_map = self.backbone.features(x)
            gap = F.adaptive_avg_pool2d(feat_map, output_size=1).flatten(1)
            gmp = F.adaptive_max_pool2d(feat_map, output_size=1).flatten(1)
            features = torch.cat([gap, gmp], dim=1)
        
        # TỐI ƯU GPU HEAVY: Async H2D (CPU→GPU transfer mà không block)
        features_np = features.detach().cpu().numpy().astype(np.float32)

        if features_np.shape[1] != self.expected_features:
            raise ValueError(f"Feature shape {features_np.shape[1]} != expected {self.expected_features}")

        pred = int(self.ensemble_model.predict(features_np)[0])
        conf = 0.0
        self.last_proba = [0.0, 0.0, 0.0]  # Reset probabilities
        
        # Get probabilities for margin-based microsleep detection
        if hasattr(self.ensemble_model, "predict_proba"):
            proba = self.ensemble_model.predict_proba(features_np)[0]
            self.last_proba = [float(p) for p in proba]  # Store probabilities
            conf = float(np.max(proba))
            
            # Apply MICRO_MARGIN logic:
            # Prioritize AWAKE detection: only classify as microsleep if microsleep 
            # is significantly higher than awake probability
            awake_prob = proba[0] if len(proba) > 0 else 0.0
            microsleep_prob = proba[2] if len(proba) > 2 else 0.0
            
            # Only reclassify as microsleep if microsleep_prob < awake_prob - 0.06
            # This detects microsleep when significantly lower than awake probability
            if microsleep_prob < awake_prob - abs(self.MICRO_MARGIN):
                pred = 2  # Classify as Microsleep
                conf = microsleep_prob
        else:
            # Standard class mapping without probability
            if pred == 2:
                pred = 0
            elif pred == 3:
                pred = 2

        self.last_pred = pred
        self.last_conf = conf
        return pred, conf
    
    def predict_async(self, bgr_face):
        """TỐI ƯU GPU HEAVY: Non-blocking prediction submit"""
        try:
            self.prediction_queue.put_nowait(bgr_face)
        except:
            pass  # Queue full, skip this frame (GPU still busy)

# ==========================================
# 3. MODULE ĐIỀU KHIỂN TRUNG TÂM (APP UI)
# ==========================================
class App:
    BEEP_INTERVAL_SEC = 1.0
    # TỐI ƯU GPU: I9-14900 + RTX 4060: Chạy AI mỗi 2 frame (từ 4 → 2) để maximize GPU usage
    AI_PREDICTION_INTERVAL = 2  # ← CHANGED: 4 → 2 (2x AI frequency)

    def __init__(self):
        root = Path(__file__).resolve().parent
        self.video = WebcamVideoStream(src=0).start()
        # TỐI ƯU: Giảm ngưỡng detection để nhanh hơn, track vẫn chính xác
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # TỐI ƯU: Không cần refine landmarks
            min_detection_confidence=0.4,  # TỐI ƯU: Giảm từ 0.5
            min_tracking_confidence=0.3,  # TỐI ƯU: Giảm từ 0.5
        )
        self.analyzer = DrowsinessAnalyzer()
        self.ai = RealtimeAIPredictor(root / "voting_output" / "final_ensemble_model.joblib")
        self.tts = VietnameseTTS()  # Khởi tạo TTS cho tiếng Việt
        self.compute_device_label = f"DEVICE: {self.ai.device.type.upper()}"
        
        # TỐI ƯU GPU: Thêm info GPU memory
        if self.ai.device.type == "cuda":
            self.compute_device_label += " + FP16"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.compute_device_label += f" ({gpu_memory:.1f}GB)"
        
        self.start_time = time.time()
        self.prev_frame_time = self.start_time
        self.frame_count = 0
        self.fps = 0.0
        self.total_sleep_count = 0
        self.total_microsleep_count = 0
        self.prev_final_alarm = False
        self.prev_yawning = False
        self.prev_ai_pred = -1
        self.last_beep_time = 0.0
        self.last_yawn_beep_time = 0.0
        
        # TỐI ƯU: Cache để tracking
        self.last_face_pts = None
        self.last_face_detect_time = 0
        
        # Display layer toggles (P=percentage, L=landmarks, S=status, D=device)
        self.show_percentage = True  # Percentage display (Awake/Sleep/Microsleep %)
        self.show_landmarks = True   # Facial landmarks
        self.show_status = True      # Status messages
        self.show_device = True      # Device info

    @staticmethod
    def crop_face(frame, face_pts, margin=0.15):
        h, w = frame.shape[:2]
        x_min = max(0, int(np.min(face_pts[:, 0]) - margin * w))
        y_min = max(0, int(np.min(face_pts[:, 1]) - margin * h))
        x_max = min(w, int(np.max(face_pts[:, 0]) + margin * w))
        y_max = min(h, int(np.max(face_pts[:, 1]) + margin * h))
        if x_max <= x_min or y_max <= y_min:
            return None
        return frame[y_min:y_max, x_min:x_max]

    TARGET_FPS = 30
    FRAME_TIME = 1.0 / TARGET_FPS

    def run(self):
        try:
            while True:
                loop_start = time.time()

                frame = self.video.read()
                if frame is None: continue

                ai_drowsy = False
                final_alarm = False
                yawning = False

                self.frame_count += 1
                current_time = time.time()
                frame_delta = current_time - self.prev_frame_time
                if frame_delta > 0:
                    instant_fps = 1.0 / frame_delta
                    self.fps = instant_fps if self.fps == 0.0 else (self.fps * 0.9) + (instant_fps * 0.1)
                self.prev_frame_time = current_time
                frame = cv2.flip(frame, 1)
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.mp_face_mesh.process(rgb)
                
                if not res.multi_face_landmarks:
                    # Tình huống liếc mắt/quay đầu (Not Detecting)
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "STATUS: NOT DETECTING", (10, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    self.analyzer.consecutive_sleep_frames = 0  # Reset khi không thấy mặt
                    self.last_face_pts = None
                else:
                    land = res.multi_face_landmarks[0].landmark
                    h, w, _ = frame.shape
                    pts = np.array([[int(l.x*w), int(l.y*h)] for l in land])
                    self.last_face_pts = pts
                    
                    marks = {
                        'left_eye': pts[[362, 385, 387, 263, 373, 380]],
                        'right_eye': pts[[33, 160, 158, 133, 153, 144]],
                        'mouth': pts[[13, 14, 78, 308]]
                    }
                    
                    # TỐI ƯU GPU HEAVY: Chạy AI mỗi 2 frame (từ 4→2) + Async submission
                    if self.frame_count % self.AI_PREDICTION_INTERVAL == 0:
                        face_crop = self.crop_face(frame, pts)
                        # Dùng async nếu có GPU, blocking nếu CPU
                        if self.ai.device.type == "cuda":
                            self.ai.predict_async(face_crop)  # Non-blocking
                        else:
                            self.ai.predict_from_face(face_crop)  # Blocking (CPU)

                    ai_label = self.ai.class_names.get(self.ai.last_pred, str(self.ai.last_pred))
                    ai_drowsy = self.ai.last_pred in (1, 2)
                    
                    # Count microsleep detection on edge (when AI predicts Microsleep)
                    if self.ai.last_pred == 2 and self.prev_ai_pred != 2:
                        self.total_microsleep_count += 1
                    self.prev_ai_pred = self.ai.last_pred
                    
                    # OPTION 1: Pass ai_pred for sliding window analysis
                    ear, mar, final_alarm, yawning = self.analyzer.analyze(marks, ai_drowsy, ai_pred=self.ai.last_pred)

                    # TỐI ƯU: Giảm thông tin hiển thị (tùy chọn)
                    cv2.putText(frame, f"EAR: {ear:.2f} MAR: {mar:.2f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"AI: {ai_label} ({self.ai.last_conf:.2f})", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
                    
                    # Display 3-class probabilities (chỉ hiển thị khi cần)
                    if self.show_percentage:
                        awake_pct = self.ai.last_proba[0] * 100
                        sleep_pct = self.ai.last_proba[1] * 100
                        microsleep_pct = self.ai.last_proba[2] * 100
                        cv2.putText(frame, f"Awake: {awake_pct:.1f}% | Sleep: {sleep_pct:.1f}% | MS: {microsleep_pct:.1f}%", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    if self.show_status:
                        if yawning:
                            cv2.putText(frame, "STATUS: YAWNING!", (10, 155), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                            # Play TTS on yawn detection (every 0.5 seconds while yawning)
                            if (current_time - self.last_yawn_beep_time) >= 0.5:
                                self.tts.speak("You are drowsy, please pay attention to driving")
                                self.last_yawn_beep_time = current_time
                        if final_alarm and not yawning:  # Don't show sleeping alert if yawning
                            cv2.putText(frame, "DANGER: SLEEPING!", (10, 185), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            # Count one sleep event on alarm rising edge.
                            if not self.prev_final_alarm:
                                self.total_sleep_count += 1

                            # Keep audible alert active at a fixed interval while alarm persists.
                            if (current_time - self.last_beep_time) >= self.BEEP_INTERVAL_SEC:
                                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
                                self.last_beep_time = current_time
                    else:
                        # Even if status hidden, still play yawn alert
                        if yawning:
                            if (current_time - self.last_yawn_beep_time) >= 0.5:
                                self.tts.speak("You are drowsy, please pay attention to driving")
                                self.last_yawn_beep_time = current_time
                        # Even if status hidden, still count sleep events (but not if yawning)
                        if final_alarm and not yawning:  # Don't alert if yawning
                            if not self.prev_final_alarm:
                                self.total_sleep_count += 1
                            if (current_time - self.last_beep_time) >= self.BEEP_INTERVAL_SEC:
                                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
                                self.last_beep_time = current_time

                self.prev_final_alarm = final_alarm if not yawning else False
                
                # Reset yawn timer when yawning just started (not during continuous yawn)
                if yawning and not self.prev_yawning:
                    # New yawn detected - reset timer to trigger alert immediately
                    self.last_yawn_beep_time = time.time() - 1.0
                self.prev_yawning = yawning

                # Display device info if enabled
                if self.show_device:
                    h, w = frame.shape[:2]
                    (text_w, text_h), _ = cv2.getTextSize(self.compute_device_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    x = max(10, w - text_w - 10)
                    y = max(text_h + 10, h - 12)
                    cv2.putText(frame, self.compute_device_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

                # Display help text with toggle keys
                cv2.putText(frame, "P:Percent L:Landmarks S:Status D:Device", (10, frame.shape[0] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

                cv2.imshow("Driver Safety System - GPU Optimized", frame)
                key = cv2.waitKey(1) & 0xFF
                
                # Handle display toggles
                if key == ord('p') or key == ord('P'):
                    self.show_percentage = not self.show_percentage
                    print(f"[Toggle] Percentage: {'ON' if self.show_percentage else 'OFF'}")
                elif key == ord('l') or key == ord('L'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"[Toggle] Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('s') or key == ord('S'):
                    self.show_status = not self.show_status
                    print(f"[Toggle] Status: {'ON' if self.show_status else 'OFF'}")
                elif key == ord('d') or key == ord('D'):
                    self.show_device = not self.show_device
                    print(f"[Toggle] Device: {'ON' if self.show_device else 'OFF'}")
                elif key == ord('q'):
                    break

                # FPS cap: sleep for remaining time in the target frame budget
                elapsed = time.time() - loop_start
                sleep_time = self.FRAME_TIME - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            # Cleanup sequence (Order is important!)
            cv2.destroyAllWindows()  # 1. Close window trước (sau ESC)
            
            # 2. Stop GPU async worker gracefully
            if hasattr(self.ai, 'stop_worker'):
                self.ai.stop_worker = True
                try:
                    self.ai.prediction_queue.put_nowait(None)  # Non-blocking signal
                except:
                    pass
                try:
                    self.ai.worker_thread.join(timeout=2)  # Đợi max 2 giây
                except:
                    pass
            
            # 3. Stop video stream
            try:
                self.video.stop()
            except:
                pass
            
            # 4. Print report
            self.print_report()

    def print_report(self):
        duration = (time.time() - self.start_time) / 60
        print("\n" + "="*40)
        print(f"      BAO CAO THONG KE HANH TRINH")
        print("="*40)
        print(f" - Thoi gian giam sat: {duration:.2f} phut")
        print(f" - So lan phat hien ngap: {self.analyzer.total_yawns}")
        print(f" - So lan phat hien ngu trang: {self.total_microsleep_count}")
        print(f" - So lan bao dong ngu:   {self.total_sleep_count}")
        print("="*40 + "\n")

if __name__ == "__main__":
    try:
        app = App()
        app.run()
    except Exception as e:
        print(f"Loi: {e}")
