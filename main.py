#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REAL-TIME DRIVER DROWSINESS DETECTION SYSTEM
Auto-calibration with correct priority logic.

Priority Order (Highest Danger First):
1. Yawning (MAR > 0.6)
2. Sleeping (Closed Eyes) - EAR < base_ear * 0.7
3. Staring (Open-Eye Microsleep) - Low EAR variance
4. Phone (Looking Down) - Pitch > base_pitch + 15
5. Safe
"""

import cv2
import math
import pickle
import numpy as np
import time
import threading
import queue
import winsound
from pathlib import Path
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat

# Import MediaPipe drawing utilities
try:
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import face_mesh_connections as mp_face_mesh
    DRAWING_AVAILABLE = True
except ImportError:
    DRAWING_AVAILABLE = False
    print("[WARNING] MediaPipe drawing utilities not available. Using simplified visualization.")

# Try to import pyttsx3 for TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("[WARNING] pyttsx3 not installed. Voice alerts will be disabled.")
    print("[INFO] Install with: pip install pyttsx3")
    TTS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "drowsiness_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
MODEL_FILE = ROOT / "models" / "face_landmarker.task"

WEBCAM_ID = 0
WINDOW_SIZE = 30  # Must match training window size
CALIBRATION_FRAMES = 60  # Frames for auto-calibration

# Label names (must match training)
LABEL_NAMES = {
    0: 'Awake',
    1: 'Drowsy',
    2: 'Phone',
    3: 'Microsleep'
}

# MediaPipe Face Mesh landmark indices
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Head pose estimation landmarks
LANDMARK_NOSE_TIP = 1
LANDMARK_CHIN = 152
LANDMARK_LEFT_EYE_CORNER = 33
LANDMARK_RIGHT_EYE_CORNER = 263
LANDMARK_LEFT_MOUTH = 61
LANDMARK_RIGHT_MOUTH = 291

# Thresholds
YAWN_THRESHOLD = 0.6  # MAR > 0.6 indicates yawning
EAR_DELTA_THRESHOLD = 0.08  # EAR < base_ear - 0.08 for sleep detection
EAR_RATIO_THRESHOLD = 0.7  # EAR < base_ear * 0.7 for closed eyes
PITCH_DELTA_THRESHOLD = 15  # Pitch > base_pitch + 15 for phone
PITCH_DELTA_WARNING = 10  # Pitch > base_pitch + 10 for warning
STARING_EAR_STD_THRESHOLD = 0.005  # Low variance indicates staring
STARING_DURATION_SECONDS = 5.0  # Duration for staring detection

# False-positive protection thresholds
CONSECUTIVE_SLEEP_FRAMES_REQUIRED = 10  # Frames required for sleep alarm (Layer 1)
MOTION_YAW_THRESHOLD = 25  # abs(Yaw) > 25 indicates extreme head turn (Layer 2)
MOTION_ROLL_THRESHOLD = 20  # abs(Roll) > 20 indicates extreme head tilt (Layer 2)
EAR_YAW_COMPENSATION = 0.002  # EAR threshold adjustment per degree of Yaw (Layer 3)

# Alarm settings
ALARM_FREQUENCY = 2500  # Hz
ALARM_DURATION = 500    # ms

# Voice alert cooldowns (in seconds)
YAWN_COOLDOWN = 10.0
WAKE_UP_COOLDOWN = 2.0
STARING_COOLDOWN = 5.0
PHONE_COOLDOWN = 5.0

# =============================================================================
# AUDIO SYSTEM (Robust Queue-Based Handler)
# =============================================================================

class AudioHandler:
    """
    Robust audio handler using queue system to prevent "Event Loop is already running" errors.
    Uses a worker thread that processes messages from a queue.
    """
    
    def __init__(self):
        self.message_queue = queue.Queue()
        self.last_spoken = {}  # Dictionary to track when messages were last spoken
        self.running = True
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        """Worker thread that processes messages from the queue."""
        while self.running:
            try:
                # Get message from queue (blocking with timeout)
                message_type, text, play_beep = self.message_queue.get(timeout=1.0)
                
                # Check cooldown
                current_time = time.time()
                cooldown_map = {
                    "yawning": YAWN_COOLDOWN,
                    "wake_up": WAKE_UP_COOLDOWN,
                    "staring": STARING_COOLDOWN,
                    "phone": PHONE_COOLDOWN
                }
                cooldown = cooldown_map.get(message_type, 2.0)
                
                if message_type in self.last_spoken:
                    if current_time - self.last_spoken[message_type] < cooldown:
                        continue  # Skip if within cooldown
                
                # Update last spoken time
                self.last_spoken[message_type] = current_time
                
                # Initialize engine for this message (prevents "Event Loop" error)
                if TTS_AVAILABLE:
                    try:
                        engine = pyttsx3.init()
                        voices = engine.getProperty('voices')
                        if voices:
                            engine.setProperty('voice', voices[0].id)
                        engine.setProperty('rate', 150)
                        engine.setProperty('volume', 0.9)
                        
                        # Speak the message
                        engine.say(text)
                        engine.runAndWait()
                        
                        # Close engine (important to prevent errors)
                        engine.stop()
                    except Exception as e:
                        print(f"[VOICE ERROR] {e}")
                        print(f"[VOICE] {text}")  # Fallback
                else:
                    print(f"[VOICE] {text}")
                
                # Play beep if requested
                if play_beep:
                    try:
                        winsound.Beep(ALARM_FREQUENCY, ALARM_DURATION)
                    except Exception as e:
                        print(f"[BEEP ERROR] {e}")
                
            except queue.Empty:
                continue  # Timeout - check again
            except Exception as e:
                print(f"[AUDIO WORKER ERROR] {e}")
    
    def speak(self, message_type, text, play_beep=False):
        """
        Add a message to the queue for processing.
        
        Args:
            message_type: "yawning", "wake_up", "staring", or "phone"
            text: Text to speak
            play_beep: Whether to play beep sound after speaking
        """
        try:
            self.message_queue.put((message_type, text, play_beep), block=False)
        except queue.Full:
            print(f"[WARNING] Audio queue full, skipping message: {text}")
    
    def shutdown(self):
        """Shutdown the audio handler."""
        self.running = False
        self.worker_thread.join(timeout=2.0)

# =============================================================================
# VISUALIZATION (Minimalist HUD)
# =============================================================================

def draw_minimalist_mesh(frame, landmarks, frame_w, frame_h):
    """
    Draw minimalist face mesh:
    - FACEMESH_CONTOURS: Green lines (Eyes, Lips, Face Oval)
    - FACEMESH_IRISES: Red circles
    """
    if DRAWING_AVAILABLE:
        try:
            # Create landmark list
            landmark_list = []
            for landmark in landmarks:
                landmark_list.append(mp.framework.formats.landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z if hasattr(landmark, 'z') else 0.0
                ))
            
            # Draw contours (Green)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark_list,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # Green
                    thickness=2,
                    circle_radius=0
                )
            )
            
            # Draw irises (Red circles)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark_list,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 0, 255),  # Red
                    thickness=1,
                    circle_radius=2
                )
            )
        except Exception as e:
            # Fallback: simple drawing
            draw_simple_mesh(frame, landmarks, frame_w, frame_h)
    else:
        draw_simple_mesh(frame, landmarks, frame_w, frame_h)


def draw_simple_mesh(frame, landmarks, frame_w, frame_h):
    """Fallback simple mesh drawing."""
    points = []
    for landmark in landmarks:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        points.append((x, y))
    
    # Draw face oval (green)
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    for i in range(len(face_oval) - 1):
        idx1, idx2 = face_oval[i], face_oval[i + 1]
        if idx1 < len(points) and idx2 < len(points):
            cv2.line(frame, points[idx1], points[idx2], (0, 255, 0), 2)
    
    # Draw eyes (green)
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    for eye_indices in [left_eye, right_eye]:
        for i in range(len(eye_indices) - 1):
            idx1, idx2 = eye_indices[i], eye_indices[i + 1]
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(frame, points[idx1], points[idx2], (0, 255, 0), 2)
    
    # Draw mouth (green)
    mouth = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318]
    for i in range(len(mouth) - 1):
        idx1, idx2 = mouth[i], mouth[i + 1]
        if idx1 < len(points) and idx2 < len(points):
            cv2.line(frame, points[idx1], points[idx2], (0, 255, 0), 2)
    
    # Draw iris centers (red circles)
    iris_centers = [468, 473]  # Left and right iris centers
    for idx in iris_centers:
        if idx < len(points):
            cv2.circle(frame, points[idx], 3, (0, 0, 255), -1)


def draw_dashboard(frame, ear, mar, pitch, base_pitch=None, x=10, y=480):
    """Draw debug dashboard at bottom-left showing EAR, MAR, Pitch, and Delta Pitch."""
    box_height = 100
    box_width = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - box_height), (x + box_width, y), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x, y - box_height), (x + box_width, y), (255, 255, 255), 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    start_y = y - box_height + 20
    
    cv2.putText(frame, f"EAR: {ear:.3f}", (x + 5, start_y), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"MAR: {mar:.3f}", (x + 5, start_y + line_height), font, font_scale, (255, 255, 255), thickness)
    
    if pitch is not None:
        pitch_text = f"Pitch: {pitch:.1f}"
        if base_pitch is not None:
            delta_pitch = pitch - base_pitch
            pitch_text += f" (Δ{delta_pitch:+.1f})"
        cv2.putText(frame, pitch_text, (x + 5, start_y + line_height * 2), font, font_scale, (255, 255, 255), thickness)
    else:
        cv2.putText(frame, "Pitch: N/A", (x + 5, start_y + line_height * 2), font, font_scale, (255, 255, 255), thickness)
    
    if base_pitch is not None:
        cv2.putText(frame, f"Base Pitch: {base_pitch:.1f}", (x + 5, start_y + line_height * 3), font, font_scale, (200, 200, 200), thickness)

# =============================================================================
# MEDIAPIPE SETUP
# =============================================================================

def init_face_landmarker():
    """Initialize MediaPipe Face Landmarker."""
    if not MODEL_FILE.exists():
        print(f"[ERROR] Face landmarker model not found: {MODEL_FILE}")
        return None
    
    base_options = python.BaseOptions(model_asset_path=str(MODEL_FILE))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return vision.FaceLandmarker.create_from_options(options)

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR)."""
    try:
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices], dtype=np.float32)
        A = euclidean_distance(points[1], points[5])
        B = euclidean_distance(points[2], points[4])
        C = euclidean_distance(points[0], points[3])
        if C == 0:
            return 0.0
        return float((A + B) / (2.0 * C))
    except (IndexError, KeyError):
        return 0.0


def calculate_mar(landmarks, frame_w, frame_h):
    """Calculate Mouth Aspect Ratio (MAR)."""
    try:
        left_mouth = landmarks[LANDMARK_LEFT_MOUTH]
        right_mouth = landmarks[LANDMARK_RIGHT_MOUTH]
        upper_lip_idx = 13
        lower_lip_idx = 14
        
        if upper_lip_idx >= len(landmarks) or lower_lip_idx >= len(landmarks):
            return 0.0
        
        upper_lip = landmarks[upper_lip_idx]
        lower_lip = landmarks[lower_lip_idx]
        
        left_pt = (left_mouth.x * frame_w, left_mouth.y * frame_h)
        right_pt = (right_mouth.x * frame_w, right_mouth.y * frame_h)
        upper_pt = (upper_lip.x * frame_w, upper_lip.y * frame_h)
        lower_pt = (lower_lip.x * frame_w, lower_lip.y * frame_h)
        
        vertical = euclidean_distance(upper_pt, lower_pt)
        horizontal = euclidean_distance(left_pt, right_pt)
        
        if horizontal == 0:
            return 0.0
        
        return float(vertical / horizontal)
    except (IndexError, KeyError):
        return 0.0


def calculate_head_pose(landmarks, frame_w, frame_h):
    """Estimate head pose (Pitch, Yaw, Roll) using solvePnP."""
    try:
        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -63.6, -12.5),    # Chin
            (-43.3, 32.7, -26.0),   # Left eye left corner
            (43.3, 32.7, -26.0),    # Right eye right corner
            (-28.9, -28.9, -24.1),  # Left mouth corner
            (28.9, -28.9, -24.1),   # Right mouth corner
        ], dtype=np.float32)
        
        nose = landmarks[LANDMARK_NOSE_TIP]
        chin = landmarks[LANDMARK_CHIN]
        left_eye = landmarks[LANDMARK_LEFT_EYE_CORNER]
        right_eye = landmarks[LANDMARK_RIGHT_EYE_CORNER]
        left_mouth = landmarks[LANDMARK_LEFT_MOUTH]
        right_mouth = landmarks[LANDMARK_RIGHT_MOUTH]
        
        image_points = np.array([
            (nose.x * frame_w, nose.y * frame_h),
            (chin.x * frame_w, chin.y * frame_h),
            (left_eye.x * frame_w, left_eye.y * frame_h),
            (right_eye.x * frame_w, right_eye.y * frame_h),
            (left_mouth.x * frame_w, left_mouth.y * frame_h),
            (right_mouth.x * frame_w, right_mouth.y * frame_h),
        ], dtype=np.float32)
        
        focal_length = frame_w
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        if not success:
            return None, None, None
        
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            pitch = math.degrees(math.atan2(-rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(rmat[2, 0], sy))
            roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(-rmat[1, 0], rmat[0, 0]))
            roll = 0.0
        
        return float(pitch), float(yaw), float(roll)
    except (IndexError, KeyError, cv2.error):
        return None, None, None


def extract_features(frame, landmarks):
    """Extract all features from a frame."""
    h, w = frame.shape[:2]
    
    left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
    avg_ear = (left_ear + right_ear) / 2.0
    
    mar = calculate_mar(landmarks, w, h)
    pitch, yaw, roll = calculate_head_pose(landmarks, w, h)
    
    return {
        'ear': avg_ear,
        'mar': mar,
        'pitch': pitch,
        'yaw': yaw,
        'roll': roll,
    }

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    print("="*70)
    print("REAL-TIME DRIVER DROWSINESS DETECTION SYSTEM")
    print("="*70)
    
    # Initialize audio handler
    audio = AudioHandler()
    
    # Load model and scaler
    print("\n[1] Loading model and scaler...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"    [OK] Model loaded: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"    [ERROR] Model not found: {MODEL_PATH}")
        print("    [INFO] Please train the model first using: python tools/model_trainer.py")
        return
    
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"    [OK] Scaler loaded: {SCALER_PATH}")
    except FileNotFoundError:
        print(f"    [ERROR] Scaler not found: {SCALER_PATH}")
        return
    
    # Initialize MediaPipe
    print("\n[2] Initializing MediaPipe Face Landmarker...")
    face_landmarker = init_face_landmarker()
    if face_landmarker is None:
        return
    print("    [OK] Face landmarker ready")
    
    # Open webcam
    print("\n[3] Opening webcam...")
    try:
        cap = cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(WEBCAM_ID)
    
    if not cap.isOpened():
        print(f"    [ERROR] Cannot open camera {WEBCAM_ID}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("    [OK] Camera opened")
    
    # Initialize buffers
    print(f"\n[4] Initializing sliding window (size={WINDOW_SIZE})...")
    ear_buffer = deque(maxlen=WINDOW_SIZE)
    mar_buffer = deque(maxlen=WINDOW_SIZE)
    pitch_buffer = deque(maxlen=WINDOW_SIZE)
    yaw_buffer = deque(maxlen=WINDOW_SIZE)
    roll_buffer = deque(maxlen=WINDOW_SIZE)
    print("    [OK] Buffers ready")
    
    print("\n" + "="*70)
    print("SYSTEM READY - Press 'q' to quit")
    print("="*70 + "\n")
    
    # Main loop variables
    pTime = time.time()
    frame_count = 0
    
    # Auto-calibration variables
    calibration_pitch_samples = []
    calibration_yaw_samples = []
    calibration_ear_samples = []
    base_pitch = None
    base_yaw = None
    base_ear = None
    is_calibrating = True
    
    # Staring detection variables
    staring_start_time = None
    staring_ear_buffer = deque(maxlen=150)  # ~5 seconds at 30 FPS
    
    # Layer 1: Debouncing (consecutive sleep frames)
    consecutive_sleep_frames = 0
    last_prediction = None
    
    # Current status
    current_status = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame")
            break
        
        frame_count += 1
        
        # Resize frame
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face
        try:
            detection_result = face_landmarker.detect(mp_image)
        except Exception as e:
            if len(ear_buffer) > 0:
                ear_buffer.clear()
                mar_buffer.clear()
                pitch_buffer.clear()
                yaw_buffer.clear()
                roll_buffer.clear()
            frame_count = 0
            is_calibrating = True
            calibration_pitch_samples.clear()
            calibration_yaw_samples.clear()
            calibration_ear_samples.clear()
            consecutive_sleep_frames = 0
            last_prediction = None
            continue
        
        # Check if face detected
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            if len(ear_buffer) > 0:
                ear_buffer.clear()
                mar_buffer.clear()
                pitch_buffer.clear()
                yaw_buffer.clear()
                roll_buffer.clear()
            frame_count = 0
            is_calibrating = True
            calibration_pitch_samples.clear()
            calibration_yaw_samples.clear()
            calibration_ear_samples.clear()
            consecutive_sleep_frames = 0
            last_prediction = None
            cv2.putText(frame, "No Face Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            landmarks = detection_result.face_landmarks[0]
            
            # Draw minimalist mesh
            draw_minimalist_mesh(frame, landmarks, w, h)
            
            # Extract features
            features = extract_features(frame, landmarks)
            
            # AUTO-CALIBRATION PHASE (First 60 frames)
            if is_calibrating:
                if frame_count <= CALIBRATION_FRAMES:
                    # Collect calibration samples
                    if features['pitch'] is not None:
                        calibration_pitch_samples.append(features['pitch'])
                        calibration_yaw_samples.append(features['yaw'])
                        calibration_ear_samples.append(features['ear'])
                    
                    # Display calibration message
                    calib_text = f"CALIBRATING... SIT STILL LOOKING FORWARD [{frame_count}/{CALIBRATION_FRAMES}]"
                    text_size = cv2.getTextSize(calib_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = h // 2
                    
                    cv2.rectangle(frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10),
                                 (0, 0, 0), -1)
                    cv2.putText(frame, calib_text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    current_status = "CALIBRATING"
                else:
                    # Calibration complete - calculate base values
                    if len(calibration_pitch_samples) > 0:
                        base_pitch = float(np.mean(calibration_pitch_samples))
                        base_yaw = float(np.mean(calibration_yaw_samples))
                        base_ear = float(np.mean(calibration_ear_samples))
                        is_calibrating = False
                        print(f"\n[INFO] Calibration complete:")
                        print(f"    Base Pitch: {base_pitch:.2f}°")
                        print(f"    Base Yaw: {base_yaw:.2f}°")
                        print(f"    Base EAR: {base_ear:.3f}\n")
                    else:
                        # Retry calibration
                        frame_count = 0
                        calibration_pitch_samples.clear()
                        calibration_yaw_samples.clear()
                        calibration_ear_samples.clear()
            
            # FEATURE EXTRACTION (Add to buffers)
            if not is_calibrating and features['pitch'] is not None:
                ear_buffer.append(features['ear'])
                mar_buffer.append(features['mar'])
                pitch_buffer.append(features['pitch'])
                yaw_buffer.append(features['yaw'])
                roll_buffer.append(features['roll'])
                
                # Add to staring detection buffer
                staring_ear_buffer.append(features['ear'])
            
            # PRIORITY LOGIC CHAIN (Only if past calibration and buffer full)
            if not is_calibrating and len(ear_buffer) == WINDOW_SIZE and base_pitch is not None:
                # Calculate window statistics
                mean_ear = float(np.mean(ear_buffer))
                std_ear = float(np.std(ear_buffer))
                mean_mar = float(np.mean(mar_buffer))
                mean_pitch = float(np.mean(pitch_buffer))
                
                # Calculate all features for model
                std_ear_full = float(np.std(ear_buffer))
                max_mar = float(np.max(mar_buffer))
                std_pitch = float(np.std(pitch_buffer))
                mean_yaw = float(np.mean(yaw_buffer))
                std_yaw = float(np.std(yaw_buffer))
                mean_roll = float(np.mean(roll_buffer))
                std_roll = float(np.std(roll_buffer))
                
                # Calculate relative values
                delta_pitch = mean_pitch - base_pitch
                
                # LAYER 3: Dynamic EAR Threshold (adjust based on Yaw angle)
                yaw_compensation = abs(mean_yaw) * EAR_YAW_COMPENSATION
                dynamic_ear_threshold = base_ear - EAR_DELTA_THRESHOLD - yaw_compensation
                ear_ratio_threshold = (base_ear * EAR_RATIO_THRESHOLD) - yaw_compensation
                
                # LAYER 2: Motion Filtering (Ignore Extreme Angles)
                abs_yaw = abs(mean_yaw)
                abs_roll = abs(mean_roll)
                is_extreme_motion = (abs_yaw > MOTION_YAW_THRESHOLD or abs_roll > MOTION_ROLL_THRESHOLD)
                
                if is_extreme_motion:
                    current_status = "ACTIVE (MOVING)"
                    cv2.putText(frame, "STATUS: ACTIVE (MOVING)", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # Yellow
                    # Reset sleep counters (Layer 1)
                    consecutive_sleep_frames = 0
                    last_prediction = None
                    # Skip sleep prediction for this frame
                
                # PRIORITY 1: YAWNING (MAR > 0.6)
                elif mean_mar > YAWN_THRESHOLD:
                    current_status = "YAWNING (TIRED)"
                    cv2.putText(frame, "STATUS: YAWNING (TIRED)", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # Yellow
                    audio.speak("yawning", "Stay focused", play_beep=False)
                    # Reset sleep counters
                    consecutive_sleep_frames = 0
                    last_prediction = None
                
                # Run model prediction (always, for debouncing)
                feature_vector = np.array([[
                    mean_ear, std_ear_full,
                    mean_mar, max_mar,
                    mean_pitch, std_pitch,
                    mean_yaw, std_yaw,
                    mean_roll, std_roll
                ]], dtype=np.float32)
                
                feature_vector_scaled = scaler.transform(feature_vector)
                prediction = model.predict(feature_vector_scaled)[0]
                
                # LAYER 1: Stricter Debouncing (10 consecutive frames)
                # Track consecutive sleep predictions from model
                if prediction in [1, 3]:  # Drowsy or Microsleep
                    if last_prediction in [1, 3]:
                        consecutive_sleep_frames += 1
                    else:
                        consecutive_sleep_frames = 1
                    last_prediction = prediction
                else:
                    # Reset counter if model predicts Awake or Phone
                    consecutive_sleep_frames = 0
                    last_prediction = prediction
                
                # PRIORITY 2: SLEEPING (Closed Eyes) - Check BEFORE Phone
                # Even if looking down, if eyes are closed, it is SLEEP
                # Only trigger if EAR indicates closed eyes AND model predicts sleep for 10+ frames
                if mean_ear < ear_ratio_threshold or mean_ear < dynamic_ear_threshold:
                    if consecutive_sleep_frames >= CONSECUTIVE_SLEEP_FRAMES_REQUIRED:
                        current_status = "DANGER: SLEEPING"
                        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (0, 0, 255), 3)  # Red border
                        cv2.putText(frame, "DANGER: SLEEPING", (20, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)  # Red text
                        audio.speak("wake_up", "DANGER! WAKE UP!", play_beep=True)
                    else:
                        # Show warning but don't alarm yet (waiting for debouncing)
                        current_status = f"WARNING: {LABEL_NAMES.get(prediction, 'DROWSY')} ({consecutive_sleep_frames}/{CONSECUTIVE_SLEEP_FRAMES_REQUIRED})"
                        cv2.putText(frame, current_status, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)  # Orange
                
                # PRIORITY 3: STARING (Open-Eye Microsleep)
                # Low EAR variance indicates staring (not blinking/moving)
                elif len(staring_ear_buffer) >= 150:  # ~5 seconds of data
                    staring_std = float(np.std(staring_ear_buffer))
                    if staring_std < STARING_EAR_STD_THRESHOLD:
                        if staring_start_time is None:
                            staring_start_time = time.time()
                        elif time.time() - staring_start_time >= STARING_DURATION_SECONDS:
                            current_status = "STARING (BLINK YOUR EYES)"
                            cv2.putText(frame, "STATUS: STARING (BLINK YOUR EYES)", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)  # Orange
                            audio.speak("staring", "Blink your eyes", play_beep=False)
                    else:
                        staring_start_time = None  # Reset if variance increases
                
                # PRIORITY 4: PHONE (Looking Down) - Only if eyes are NOT closed
                elif delta_pitch > PITCH_DELTA_THRESHOLD:
                    current_status = "DISTRACTED (PHONE)"
                    cv2.putText(frame, "STATUS: DISTRACTED (PHONE)", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # Yellow
                    audio.speak("phone", "Eyes on road", play_beep=False)
                    # Reset sleep counters
                    consecutive_sleep_frames = 0
                    last_prediction = None
                
                # PRIORITY 5: SAFE
                else:
                    # Model prediction already done above
                    # Show status based on prediction
                    current_status = f"STATUS: {LABEL_NAMES.get(prediction, 'SAFE')}"
                    color = (0, 255, 0) if prediction == 0 else (0, 165, 255)
                    cv2.putText(frame, current_status, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw dashboard
            draw_dashboard(frame, features['ear'], features['mar'], features['pitch'], base_pitch)
        
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        cv2.putText(frame, f'FPS: {int(fps)}', (w - 100, h - 20),
                   cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Driver Drowsiness Detection System", frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    audio.shutdown()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] System shutdown complete")

if __name__ == "__main__":
    main()
