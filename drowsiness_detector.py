from scipy.spatial import distance as dist
import numpy as np
import cv2
import time
from collections import deque

class DrowsinessDetector:
    def __init__(self):
        # --- 1. CẤU HÌNH ĐỘ NHẠY (Đã đưa về mức chuẩn bạn thấy oke) ---
        self.EAR_THRESHOLD = 0.25    # Mắt: < 0.25 là nhắm (Nếu đeo kính dày có thể chỉnh lên 0.22)
        self.EAR_SLEEP = 0.15        # EAR thấp để xác định 'thực sự nhắm' (tránh false-positive khi nhìn xuống)
        self.MAR_THRESHOLD = 0.5     # Miệng: > 0.5 là ngáp
        self.CLOSED_EYE_FRAME = 15   # Số frame mắt nhắm để báo động
        
        # --- 2. CẤU HÌNH GIẤC NGỦ TRẮNG (HEAD DROP) ---
        # Thay vì số cứng, ta dùng độ lệch so với lúc căn chỉnh
        self.HEAD_DROP_ANGLE = 10    # Cúi xuống quá 10 độ so với lúc bình thường -> Bắt lỗi
        self.HEAD_DROP_TIME = 20     # Số frame cúi đầu liên tục để báo động (Giấc ngủ trắng)

        # Biến nội bộ
        self.counter = 0
        self.head_drop_counter = 0
        self.alarm_on = False
        self.base_ear = None

        # Closure relative factors for baseline and look-down adjustments
        self.CLOSURE_RELATIVE_FACTOR = 0.50
        self.CLOSURE_RELATIVE_FACTOR_LOOKDOWN = 0.45

        # Pre-drowsy (sit still / drowsy) requirement before head-drop alarm
        self.pre_drowsy_counter = 0
        self.PRE_DROWSY_FRAMES = 15  # number of frames indicating drowsy state before head drop can alarm

        # Blink frequency low threshold used to classify pre-drowsy
        self.BLINK_FREQ_LOW = 8.0  # blinks per minute considered low

        # Eye motion history to detect stationary gaze (possible microsleep)
        self.eye_center_history = deque()
        self.EYE_MOTION_WINDOW = 1.5  # seconds window to measure eye motion
        self.EYE_MOTION_THRESHOLD = 0.0025  # relative to image width (fraction/sec) below which gaze is considered stationary
        
        # Biến căn chỉnh (Sẽ được set lúc khởi động)
        self.base_pitch = 0 
        self.base_yaw = 0
        self.is_calibrated = False

        # Smoothing cho pitch để ổn định head-pose
        self.smoothed_pitch = None
        self.smoothed_yaw = None
        self.smoothed_roll = None
        self.smoothed_nose = None
        self.smooth_alpha = 0.35  # 0..1, lớn hơn -> nhạy nhưng kém mượt

        # Ngưỡng yaw để bỏ qua detection head-drop khi đang nhìn sang hai bên
        self.YAW_IGNORE_THRESHOLD = 18.0  # degree

        # Blink tracking to distinguish frequent short blinks from prolonged eye closure
        self.blink_times = deque()
        self.BLINK_MIN_FRAMES = 3        # max frames considered a blink (short closure)
        self.BLINK_WINDOW = 60           # window (seconds) to compute blink frequency
        self.BLINK_FREQ_IGNORE = 30.0    # if blinks/min >= this, suppress sleep alarm

        # Yawn suppression: avoid beep/alarm for a short time after a yawn
        self.last_yawn_time = 0.0
        self.YAWN_SUPPRESSION_TIME = 3.0  # seconds

        # Pitch history for velocity detection (for nod detection)
        self.pitch_history = deque(maxlen=6)  # store (timestamp, pitch)
        self.PITCH_VEL_THRESHOLD = 12.0  # deg/sec, downward velocity threshold for nod
        self.PITCH_VEL_WINDOW = 0.5      # seconds over which to compute velocity


        # Chỉ số MediaPipe
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [13, 14, 61, 291] 
        
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])

    def set_calibration(self, pitch, yaw, base_ear=None):
        """Hàm lưu lại tư thế ngồi chuẩn
        Optionally accept a baseline EAR measured during calibration.
        """
        self.base_pitch = pitch
        self.base_yaw = yaw
        if base_ear is not None and np.isfinite(base_ear):
            self.base_ear = base_ear
        else:
            self.base_ear = None
        self.is_calibrated = True
        print(f"--- ĐÃ CĂN CHỈNH: Pitch={pitch:.2f}, Yaw={yaw:.2f}, BaseEAR={self.base_ear if self.base_ear is not None else 'N/A'} ---")

    def eye_aspect_ratio(self, eye_points):
        """Return EAR or None if calculation is invalid or unreliable.
        Avoid returning 0 which may be misinterpreted as closed eye when landmarks are missing/occluded.
        """
        try:
            A = dist.euclidean(eye_points[1], eye_points[5])
            B = dist.euclidean(eye_points[2], eye_points[4])
            C = dist.euclidean(eye_points[0], eye_points[3])
        except Exception:
            return None
        # If denominator is zero or extremely small, landmark geometry is invalid
        if C == 0 or C < 1e-6:
            return None
        ear = (A + B) / (2.0 * C)
        # If EAR is NaN or non-finite, treat as invalid
        if not np.isfinite(ear) or ear <= 0:
            return None
        return ear

    def mouth_aspect_ratio(self, mouth_points):
        A = dist.euclidean(mouth_points[0], mouth_points[1])
        C = dist.euclidean(mouth_points[2], mouth_points[3])
        if C == 0: return 0
        return A / C

    def get_head_pose(self, landmarks, img_w, img_h):
        image_points = np.array([
            landmarks[1], landmarks[152], landmarks[33], 
            landmarks[263], landmarks[61], landmarks[291]
        ], dtype="double")

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1)) 

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Dùng decomposeProjectionMatrix để ổn định
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)

        return eulerAngles[0][0], eulerAngles[1][0], eulerAngles[2][0], list(image_points[0])

    def detect(self, landmarks, img_shape):
        h, w, c = img_shape
        
        # Tính toán EAR, MAR (tính EAR một cách an toàn trước khi dùng trong logic head-drop)
        left_eye_pts = np.array([landmarks[i] for i in self.LEFT_EYE])
        right_eye_pts = np.array([landmarks[i] for i in self.RIGHT_EYE])
        leftEAR = self.eye_aspect_ratio(left_eye_pts)
        rightEAR = self.eye_aspect_ratio(right_eye_pts)
        valid_ears = [e for e in (leftEAR, rightEAR) if e is not None and np.isfinite(e)]
        avgEAR = (sum(valid_ears) / len(valid_ears)) if len(valid_ears) > 0 else None

        # Compute eye center (pixel) and update eye motion history
        eye_center = None
        try:
            le_center = np.mean(left_eye_pts, axis=0)
            re_center = np.mean(right_eye_pts, axis=0)
            eye_center = ((le_center + re_center) / 2.0).tolist()
        except Exception:
            eye_center = None

        # Normalize eye center and append to history
        now = time.time()
        if eye_center is not None:
            # store normalized by image width
            img_w = img_shape[1]
            self.eye_center_history.append((now, eye_center[0] / img_w, eye_center[1] / img_w))
            # prune old entries
            while self.eye_center_history and self.eye_center_history[0][0] < now - self.EYE_MOTION_WINDOW:
                self.eye_center_history.popleft()

        mouth_pts = np.array([landmarks[i] for i in self.MOUTH])
        mar = self.mouth_aspect_ratio(mouth_pts)

        # Tính Head Pose hiện tại
        curr_pitch, curr_yaw, curr_roll, nose_point = self.get_head_pose(landmarks, w, h)

        # Smoothing cho pitch/yaw/roll/nose nhằm giảm nhiễu và tín hiệu nhảy
        now = time.time()
        if self.smoothed_pitch is None:
            self.smoothed_pitch = curr_pitch
            self.smoothed_yaw = curr_yaw
            self.smoothed_roll = curr_roll
            self.smoothed_nose = nose_point
        else:
            self.smoothed_pitch = (1.0 - self.smooth_alpha) * self.smoothed_pitch + self.smooth_alpha * curr_pitch
            self.smoothed_yaw = (1.0 - self.smooth_alpha) * self.smoothed_yaw + self.smooth_alpha * curr_yaw
            self.smoothed_roll = (1.0 - self.smooth_alpha) * self.smoothed_roll + self.smooth_alpha * curr_roll
            self.smoothed_nose = [(1.0 - self.smooth_alpha) * self.smoothed_nose[0] + self.smooth_alpha * nose_point[0],
                                  (1.0 - self.smooth_alpha) * self.smoothed_nose[1] + self.smooth_alpha * nose_point[1]]

        # Update pitch history for velocity calculation
        self.pitch_history.append((now, self.smoothed_pitch))
        # Remove too-old entries (safety)
        while self.pitch_history and self.pitch_history[0][0] < now - max(self.PITCH_VEL_WINDOW, 1.0):
            self.pitch_history.popleft()

        # Nếu chưa căn chỉnh thì chưa cảnh báo
        if not self.is_calibrated:
            return {"ear": avgEAR, "mar": mar, "pitch": curr_pitch, "yaw": curr_yaw, 
                    "status": "CALIBRATING...", "color": (128, 128, 128), "alarm": False, "nose_point": nose_point}

        # --- TÍNH TOÁN ĐỘ LỆCH (Relative) ---
        # Delta Pitch: dựa trên giá trị đã được lọc
        # Thông thường: Ngước lên là +, Cúi xuống là - (hoặc ngược lại tuỳ cam)
        delta_pitch = self.smoothed_pitch - self.base_pitch
        delta_yaw = curr_yaw - self.base_yaw
        self.alarm_on = False

        # Default status (ensure variable always defined)
        status = "AWAKE"
        color = (0, 255, 0)

        # --- PRE-DROWSY CHECK ---
        is_pre_drowsy = False
        # a) Yawning is a pre-drowsy sign
        if mar > self.MAR_THRESHOLD:
            is_pre_drowsy = True
        # b) Mild eye closure (less than sleep threshold) also signals drowsiness
        elif avgEAR is not None and avgEAR < (self.EAR_THRESHOLD * 1.05):
            is_pre_drowsy = True
        # c) Very low blink rate indicates drowsiness
        if len(self.blink_times) > 0:
            blinks_per_min = len(self.blink_times) * (60.0 / self.BLINK_WINDOW)
            if blinks_per_min < self.BLINK_FREQ_LOW:
                is_pre_drowsy = True

        # d) Low eye motion (static gaze) is also a pre-drowsy sign
        eye_motion = None
        if len(self.eye_center_history) >= 2:
            t0, x0, y0 = self.eye_center_history[0]
            t1, x1, y1 = self.eye_center_history[-1]
            dt = max(1e-6, t1 - t0)
            dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            eye_motion = dist / dt  # fraction of image width per second
            if eye_motion < self.EYE_MOTION_THRESHOLD:
                is_pre_drowsy = True

        # Update pre-drowsy counter (decay when not pre-drowsy)
        if is_pre_drowsy:
            self.pre_drowsy_counter += 1
        else:
            self.pre_drowsy_counter = max(0, self.pre_drowsy_counter - 1)

        # --- LOGIC ƯU TIÊN 1: GIẤC NGỦ TRẮNG (HEAD DROP) ---
        # Nếu nhìn sang hai bên (yaw lệch lớn) thì bỏ qua head-drop
        if abs(delta_yaw) > self.YAW_IGNORE_THRESHOLD:
            # Reset counter để tránh chuyển thành alarm khi vừa quay lại
            self.head_drop_counter = 0
            status = "LOOKING SIDEWAYS"
            color = (200, 200, 0)
        # Kiểm tra nếu đầu gục xuống quá ngưỡng (Giả sử gục là giảm giá trị Pitch)
        # Bạn chỉnh dấu < thành > nếu camera bạn bị ngược chiều
        elif delta_pitch < -self.HEAD_DROP_ANGLE:
            # Calculate pitch velocity (deg/sec) over the available window
            pitch_vel = 0.0
            if len(self.pitch_history) >= 2:
                t0, p0 = self.pitch_history[0]
                t1, p1 = self.pitch_history[-1]
                dt = max(1e-6, t1 - t0)
                pitch_vel = (p1 - p0) / dt

            # If a fast downward nod is detected, increase the counter faster (more likely micro-sleep)
            if pitch_vel < -self.PITCH_VEL_THRESHOLD:
                self.head_drop_counter += 2
            else:
                self.head_drop_counter += 1

            # Nếu gục đầu đủ lâu, yêu cầu 'thực sự nhắm' để báo động (tránh false positive khi nhìn điện thoại)
            if self.head_drop_counter > self.HEAD_DROP_TIME:
                # Require that the user was already in a pre-drowsy state for several frames
                if self.pre_drowsy_counter >= self.PRE_DROWSY_FRAMES:
                    eyes_closed = (self.counter >= self.CLOSED_EYE_FRAME) or (avgEAR is not None and avgEAR < self.EAR_SLEEP)
                    # compute eye motion low condition if available
                    eye_motion_low = (eye_motion is not None and eye_motion < self.EYE_MOTION_THRESHOLD)
                    # Suppress alarm if a yawn happened recently
                    recent_yawn = (time.time() - self.last_yawn_time) < self.YAWN_SUPPRESSION_TIME
                    # If eyes are closed OR (eyes open but gaze is motionless and pre-drowsy), trigger
                    if (eyes_closed or eye_motion_low) and not recent_yawn:
                        status = "WHITE SLEEP (HEAD DROP)!"
                        color = (0, 0, 255)
                        self.alarm_on = True
                    elif (eyes_closed or eye_motion_low) and recent_yawn:
                        status = "YAWN RECENTLY - NO ALARM"
                        color = (0, 165, 255)
                        self.alarm_on = False
                    else:
                        status = "LOOKING DOWN..."
                        color = (0, 255, 255)
                else:
                    status = "DROWSY (PRE) - WAITING"
                    color = (0, 165, 255)
                    self.alarm_on = False
            else:
                status = "LOOKING DOWN..."
                color = (0, 255, 255)
        else:
            self.head_drop_counter = 0

        # --- LOGIC ƯU TIÊN 2: NGÁP (Giữ nguyên độ nhạy bạn thích) ---
        if mar > self.MAR_THRESHOLD:
            status = "YAWNING"
            color = (0, 165, 255)
            # Ngáp chỉ cảnh báo bằng hình ảnh và giọng nói — KHÔNG bật alarm/beep
            self.alarm_on = False
            self.last_yawn_time = time.time()  # record time to suppress immediate alarms after a yawn

        # --- LOGIC ƯU TIÊN 3: NHẮM MẮT (Drowsiness) ---
        # If EAR is unavailable, do not increment closed-eye counter or raise alarm
        if avgEAR is None:
            # Reset closed-eye counter (we cannot assume eyes are closed if landmarks are occluded)
            self.counter = 0
            if status == "AWAKE":
                status = "EYES NOT DETECTED"
                color = (100, 100, 100)
            blinks_per_min = 0.0
        else:
            # Determine dynamic closure threshold based on baseline EAR and pitch
            if self.base_ear is not None:
                factor = self.CLOSURE_RELATIVE_FACTOR
                if delta_pitch < - (self.HEAD_DROP_ANGLE / 2.0):
                    factor = self.CLOSURE_RELATIVE_FACTOR_LOOKDOWN
                closure_threshold = max(self.EAR_SLEEP, self.base_ear * factor)
            else:
                closure_threshold = self.EAR_THRESHOLD

            # Eye closed duration counting using dynamic threshold
            if avgEAR < closure_threshold:
                self.counter += 1
            else:
                # Eye opened: if previous closure was short, count as a blink event
                if 0 < self.counter < self.BLINK_MIN_FRAMES:
                    now = time.time()
                    self.blink_times.append(now)
                    # Prune old blinks
                    while self.blink_times and self.blink_times[0] < now - self.BLINK_WINDOW:
                        self.blink_times.popleft()
                self.counter = 0
                if status == "AWAKE": self.alarm_on = False

            # Compute blink frequency in blinks per minute
            now = time.time()
            while self.blink_times and self.blink_times[0] < now - self.BLINK_WINDOW:
                self.blink_times.popleft()
            blinks_per_min = len(self.blink_times) * (60.0 / self.BLINK_WINDOW)

            # If eyes closed long enough, only trigger alarm if blink frequency is NOT abnormally high
            if self.counter >= self.CLOSED_EYE_FRAME:
                # Suppress alarm if a yawn happened recently
                recent_yawn = (time.time() - self.last_yawn_time) < self.YAWN_SUPPRESSION_TIME
                if blinks_per_min >= self.BLINK_FREQ_IGNORE:
                    status = "FREQUENT BLINKS"
                    color = (150, 150, 0)
                    self.alarm_on = False
                elif recent_yawn:
                    status = "YAWN RECENTLY - NO ALARM"
                    color = (0, 165, 255)
                    self.alarm_on = False
                else:
                    self.alarm_on = True
                    status = "SLEEPING (EYES)!"
                    color = (0, 0, 255)

        # For UI, return a numeric EAR (NaN if unavailable) to avoid formatting errors downstream
        display_ear = avgEAR if avgEAR is not None else float('nan')

        return {
            "ear": display_ear, "mar": mar,
            "pitch": self.smoothed_pitch, "yaw": self.smoothed_yaw, "roll": self.smoothed_roll,
            "status": status, "color": color, "alarm": self.alarm_on,
            "nose_point": self.smoothed_nose,
            "eye_motion": eye_motion
        }


# -----------------------------------------------------------------------------
# Compatibility layer for the advanced runner (run_advanced_dms.py)
# Provides: Thresholds, DriverState, Metrics, DriverMonitoringSystem
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from enum import Enum


@dataclass
class Thresholds:
    EAR_CLOSED: float = 0.20
    EAR_SLEEP: float = 0.15
    PITCH_DROWSY: float = -15.0
    PITCH_ALERT: float = -25.0
    EYES_CLOSED_DURATION_WARNING: float = 1.0
    EYES_CLOSED_DURATION_ALERT: float = 2.0
    YAWN_DURATION: float = 1.5
    BLINK_FREQ_LOW: float = 8.0


@dataclass
class Metrics:
    ear_avg: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


class DriverState(Enum):
    AWAKE = "AWAKE"
    YAWNING = "YAWNING"
    DROWSY = "DROWSY"
    SLEEPING = "SLEEPING"


class DriverMonitoringSystem:
    """A lightweight wrapper to match the interface expected by run_advanced_dms.py

    It uses the existing DrowsinessDetector internally and adapts thresholds
    provided by the Thresholds dataclass.
    """
    def __init__(self, face_detector, thresholds: Thresholds = None, fps: float = 30.0,
                 use_model: bool = False, model_path: str = None):
        self.face_detector = face_detector
        self.thresholds = thresholds or Thresholds()
        self.fps = fps
        self.use_model = use_model
        self.model_path = model_path

        # Internal detector
        self.detector = DrowsinessDetector()
        # Apply thresholds to the detector
        self.detector.EAR_THRESHOLD = self.thresholds.EAR_CLOSED
        self.detector.MAR_THRESHOLD = max(0.4, 0.5)  # keep reasonable default for yawning
        # Convert seconds -> frames for closed eye durations
        self.detector.CLOSED_EYE_FRAME = max(1, int(self.thresholds.EYES_CLOSED_DURATION_ALERT * self.fps))
        # Head drop sensitivity
        self.detector.HEAD_DROP_ANGLE = abs(self.thresholds.PITCH_ALERT)
        self.detector.HEAD_DROP_TIME = max(1, int(self.thresholds.EYES_CLOSED_DURATION_ALERT * self.fps))

        # Runtime state
        self.current_state = DriverState.AWAKE
        self._last_result = None

    def process_frame(self, frame):
        """Process a single frame and return (state, metrics, output_frame)"""
        out = frame.copy()
        out, landmarks = self.face_detector.findFaceMesh(out, draw=True)

        if len(landmarks) == 0:
            # No face detected -> keep previous state but return basic metrics
            metrics = Metrics()
            return self.current_state, metrics, out

        result = self.detector.detect(landmarks, out.shape)
        self._last_result = result

        status = result.get("status", "AWAKE")
        # Map textual status to DriverState enum
        if "YAWN" in status.upper():
            state = DriverState.YAWNING
        elif "SLEEP" in status.upper() or result.get("alarm", False):
            state = DriverState.SLEEPING
        elif "LOOKING DOWN" in status.upper() or result.upper().find("DROWSY") != -1:
            state = DriverState.DROWSY
        else:
            state = DriverState.AWAKE

        self.current_state = state

        # Build metrics object
        metrics = Metrics(
            ear_avg=result.get("ear", 0.0),
            mar=result.get("mar", 0.0),
            pitch=result.get("pitch", 0.0),
            yaw=result.get("yaw", 0.0),
            roll=result.get("roll", 0.0)
        )

        # Overlay simple HUD
        cv2.rectangle(out, (0, 0), (360, 120), (0, 0, 0), -1)
        cv2.putText(out, f'State: {self.current_state.value}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, f'EAR: {metrics.ear_avg:.2f}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(out, f'MAR: {metrics.mar:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if result.get("alarm", False):
            cv2.putText(out, 'ALERT!', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return self.current_state, metrics, out

    def trigger_alert(self):
        """Return True if current situation should trigger an alert sound"""
        if self._last_result is None:
            return False
        return bool(self._last_result.get("alarm", False))

    def get_performance_stats(self):
        return {"fps": float(self.fps), "current_state": self.current_state.value}
