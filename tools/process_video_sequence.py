#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIDEO SEQUENCE PROCESSOR - Time-Series Analysis for Driver Drowsiness Detection

This script processes video files organized in folders and extracts temporal features
using sliding window analysis. Designed for detecting microsleep sequences which
require time-series analysis rather than static image classification.

Features:
- Recursive folder scanning with auto-labeling
- Frame-by-frame feature extraction (EAR, MAR, Head Pose)
- Sliding window with statistical aggregation
- Output to CSV for machine learning training
"""

import cv2
import math
import csv
import numpy as np
from pathlib import Path
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat
import os
import urllib.request
import time
import gc

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DATASET_DIR = Path("raw_dataset")
OUTPUT_CSV = Path("sequence_data.csv")

# Sliding Window Configuration
WINDOW_SIZE = 30      # Number of frames per window (~1 second at 30fps)
STEP_SIZE = 10        # Step size for sliding window (overlap = WINDOW_SIZE - STEP_SIZE)

# Debug and Performance Settings
DEBUG_VERBOSE = False  # Set to True to see all debug messages about face detection failures
FRAME_SKIP = 1         # Process every Nth frame (1 = every frame, 2 = every 2nd frame, etc.)
MAX_FRAME_WIDTH = 640  # Resize frames to max width (maintains aspect ratio) - speeds up MediaPipe significantly

# Video file extensions
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".flv", ".wmv")

# Folder to Label Mapping (Auto-labeling based on folder name)
FOLDER_LABEL_MAP = {
    "0_tinh_tao": 0,      # Awake
    "1_buon_ngu": 1,      # Drowsy
    "2_dien_thoai": 2,    # Phone Distraction
    "3_ngu_trang": 3,     # Microsleep
}

# =============================================================================
# MEDIAPIPE SETUP (New Tasks-based API)
# =============================================================================

# Model path - try multiple locations
def find_model_path():
    """Find the face landmarker model file."""
    possible_paths = [
        Path("models/face_landmarker.task"),
        Path("../models/face_landmarker.task"),  # If running from _archive
        Path(__file__).parent.parent / "models" / "face_landmarker.task",  # Absolute from script
    ]
    for path in possible_paths:
        if path.exists():
            return path
    # Return default if not found (will trigger download)
    return Path("models/face_landmarker.task")

MODEL_PATH = find_model_path()

# Initialize MediaPipe Face Landmarker
def init_face_landmarker():
    """Initialize MediaPipe Face Landmarker with the new tasks API."""
    global MODEL_PATH
    
    # Re-check model path
    MODEL_PATH = find_model_path()
    
    # Check if model exists, download if not
    if not MODEL_PATH.exists():
        print(f"[INFO] Model not found at {MODEL_PATH}, downloading...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        try:
            urllib.request.urlretrieve(url, str(MODEL_PATH))
            print(f"[INFO] Model downloaded to {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to download model: {e}")
            raise
    
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH.resolve()))
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

# Initialize the detector (will be created per video or reused)
face_landmarker = None

# MediaPipe Face Mesh landmark indices for feature extraction
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Head pose estimation landmarks
LANDMARK_NOSE_TIP = 1
LANDMARK_CHIN = 152
LANDMARK_LEFT_EYE_CORNER = 33
LANDMARK_RIGHT_EYE_CORNER = 263
LANDMARK_LEFT_MOUTH = 61
LANDMARK_RIGHT_MOUTH = 291

# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def calculate_ear(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR).
    
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    where p1-p6 are eye landmark points.
    
    Returns:
        float: EAR value (lower = more closed)
    """
    try:
        points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices], dtype=np.float32)
        
        # Vertical distances (eye opening)
        A = euclidean_distance(points[1], points[5])  # Top-bottom distance 1
        B = euclidean_distance(points[2], points[4])  # Top-bottom distance 2
        
        # Horizontal distance (eye width)
        C = euclidean_distance(points[0], points[3])  # Left-right distance
        
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        return float(ear)
    except (IndexError, KeyError):
        return 0.0


def calculate_mar(landmarks, frame_w, frame_h):
    """
    Calculate Mouth Aspect Ratio (MAR).
    
    MAR = vertical_distance / horizontal_distance
    Higher MAR indicates mouth opening (yawning).
    
    Returns:
        float: MAR value
    """
    try:
        # Mouth corner points
        left_mouth = landmarks[LANDMARK_LEFT_MOUTH]
        right_mouth = landmarks[LANDMARK_RIGHT_MOUTH]
        
        # Upper and lower lip points (approximate indices)
        # Using landmarks near the mouth center
        upper_lip_idx = 13   # Upper inner lip
        lower_lip_idx = 14   # Lower inner lip
        
        if upper_lip_idx >= len(landmarks) or lower_lip_idx >= len(landmarks):
            # Fallback: use mouth corners only
            left_pt = (left_mouth.x * frame_w, left_mouth.y * frame_h)
            right_pt = (right_mouth.x * frame_w, right_mouth.y * frame_h)
            horizontal = euclidean_distance(left_pt, right_pt)
            if horizontal == 0:
                return 0.0
            # Approximate vertical as small value
            return 0.0
        
        upper_lip = landmarks[upper_lip_idx]
        lower_lip = landmarks[lower_lip_idx]
        
        # Convert to pixel coordinates
        left_pt = (left_mouth.x * frame_w, left_mouth.y * frame_h)
        right_pt = (right_mouth.x * frame_w, right_mouth.y * frame_h)
        upper_pt = (upper_lip.x * frame_w, upper_lip.y * frame_h)
        lower_pt = (lower_lip.x * frame_w, lower_lip.y * frame_h)
        
        # Calculate distances
        vertical = euclidean_distance(upper_pt, lower_pt)
        horizontal = euclidean_distance(left_pt, right_pt)
        
        if horizontal == 0:
            return 0.0
        
        mar = vertical / horizontal
        return float(mar)
    except (IndexError, KeyError):
        return 0.0


def calculate_head_pose(landmarks, frame_w, frame_h):
    """
    Estimate head pose (Pitch, Yaw, Roll) using solvePnP.
    
    Pitch: Up/Down rotation (nodding)
    Yaw: Left/Right rotation (turning head)
    Roll: Tilting left/right
    
    Returns:
        tuple: (pitch, yaw, roll) in degrees, or (None, None, None) if failed
    """
    try:
        # 3D model points of a generic head (in arbitrary units)
        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -63.6, -12.5),    # Chin
            (-43.3, 32.7, -26.0),   # Left eye left corner
            (43.3, 32.7, -26.0),    # Right eye right corner
            (-28.9, -28.9, -24.1),  # Left mouth corner
            (28.9, -28.9, -24.1),   # Right mouth corner
        ], dtype=np.float32)
        
        # Extract 2D image points from MediaPipe landmarks
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
        
        # Camera intrinsic parameters (approximate pinhole model)
        focal_length = frame_w
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # No lens distortion
        
        # Solve PnP to get rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        
        if not success:
            return None, None, None
        
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Extract Euler angles from rotation matrix
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = math.degrees(math.atan2(-rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(rmat[2, 0], sy))
            roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            # Fallback for singular case
            pitch = math.degrees(math.atan2(-rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(-rmat[1, 0], rmat[0, 0]))
            roll = 0.0
        
        return float(pitch), float(yaw), float(roll)
    except (IndexError, KeyError, cv2.error):
        return None, None, None


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def is_video_file(path: Path) -> bool:
    """Check if file is a video file based on extension."""
    return path.suffix.lower() in VIDEO_EXTS


def extract_frame_features(frame, landmarks):
    """
    Extract all features from a single frame.
    
    Returns:
        dict: Features dictionary with keys: ear, mar, pitch, yaw, roll
              Values are None if extraction failed
    """
    h, w = frame.shape[:2]
    
    # Calculate EAR for both eyes
    left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Calculate MAR
    mar = calculate_mar(landmarks, w, h)
    
    # Calculate head pose
    pitch, yaw, roll = calculate_head_pose(landmarks, w, h)
    
    return {
        'ear': avg_ear,
        'mar': mar,
        'pitch': pitch,
        'yaw': yaw,
        'roll': roll,
    }


def process_video(video_path: Path, label: int, all_windows: list):
    """
    Process a single video file and extract sliding window features.
    
    CRITICAL DATA INTEGRITY RULE:
    - Sliding windows must contain ONLY continuous face sequences.
    - If face is not detected in ANY frame, ALL buffers are cleared.
    - This ensures no windows contain gaps or missing frames.
    - Only valid, continuous sequences are included in the dataset.
    
    Args:
        video_path: Path to video file
        label: Label for this video (from folder name)
        all_windows: List to append window feature dictionaries
    """
    global face_landmarker
    
    # Initialize face landmarker if not already done
    if face_landmarker is None:
        face_landmarker = init_face_landmarker()
    
    print(f"  Processing: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [WARN] Cannot open video: {video_path}")
        return
    
    # Buffers to store features for sliding window
    # Note: No maxlen - we manually manage window size for proper sliding
    ear_buffer = deque()
    mar_buffer = deque()
    pitch_buffer = deque()
    yaw_buffer = deque()
    roll_buffer = deque()
    
    frame_count = 0
    valid_frame_count = 0
    processed_frame_count = 0  # Count of frames actually processed (after skipping)
    
    # Get video FPS and total frames for progress reporting
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    
    # Calculate resize factor if needed
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize_needed = original_width > MAX_FRAME_WIDTH
    if resize_needed:
        scale_factor = MAX_FRAME_WIDTH / original_width
        new_width = MAX_FRAME_WIDTH
        new_height = int(original_height * scale_factor)
        print(f"    [INFO] Resizing frames: {original_width}x{original_height} -> {new_width}x{new_height} (speedup: ~{(original_width/new_width)**2:.1f}x)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Frame skipping for faster processing
        # FRAME_SKIP=1: process every frame (frame_count % 1 == 0 always)
        # FRAME_SKIP=2: process every 2nd frame (frame_count % 2 == 0)
        if frame_count % FRAME_SKIP != 0:
            continue
        
        processed_frame_count += 1
        
        # Resize frame for faster MediaPipe processing (maintain aspect ratio)
        if resize_needed:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Landmarker (new API)
        # Use context manager or explicit cleanup to prevent memory leaks
        mp_image = None
        detection_result = None
        
        try:
            # Create mp.Image - MediaPipe will handle cleanup internally
            mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            
            # Add timeout protection and better error handling
            detection_result = face_landmarker.detect(mp_image)
            
        except KeyboardInterrupt:
            # Allow graceful shutdown on Ctrl+C
            print(f"\n    [INFO] Interrupted by user at frame {frame_count}")
            break
        except Exception as e:
            # If detection fails, treat as no face detected
            if len(ear_buffer) > 0:
                ear_buffer.clear()
                mar_buffer.clear()
                pitch_buffer.clear()
                yaw_buffer.clear()
                roll_buffer.clear()
                print(f"    [DEBUG] Frame {frame_count}: Detection error ({type(e).__name__}) - Resetting window")
            # Explicitly clear mp_image reference to help GC
            mp_image = None
            continue
        finally:
            # Explicit cleanup (though MediaPipe should handle this)
            mp_image = None
        
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            # CRITICAL: No face detected - clear all buffers to maintain continuity
            # A sliding window must contain continuous face sequences only.
            # Missing even 1 frame breaks the temporal continuity.
            if len(ear_buffer) > 0:
                ear_buffer.clear()
                mar_buffer.clear()
                pitch_buffer.clear()
                yaw_buffer.clear()
                roll_buffer.clear()
                print(f"    [DEBUG] Frame {frame_count}: No face - Resetting window")
            continue
        
        # Get landmarks from first detected face
        landmarks = detection_result.face_landmarks[0]
        
        # Extract features from this frame
        features = extract_frame_features(frame, landmarks)
        
        # Skip if head pose extraction failed (also breaks continuity)
        if features['pitch'] is None:
            # CRITICAL: Head pose extraction failed - clear buffers to maintain continuity
            if len(ear_buffer) > 0:
                ear_buffer.clear()
                mar_buffer.clear()
                pitch_buffer.clear()
                yaw_buffer.clear()
                roll_buffer.clear()
                print(f"    [DEBUG] Frame {frame_count}: Head pose extraction failed - Resetting window")
            continue
        
        # Add features to buffers (only if face detected and features extracted successfully)
        ear_buffer.append(features['ear'])
        mar_buffer.append(features['mar'])
        pitch_buffer.append(features['pitch'])
        yaw_buffer.append(features['yaw'])
        roll_buffer.append(features['roll'])
        
        valid_frame_count += 1
        
        # Progress reporting (every 100 processed frames)
        if processed_frame_count % 100 == 0:
            if total_frames > 0:
                # Calculate progress based on original frame count
                progress_pct = (frame_count / total_frames) * 100
                print(f"    Processing frame {frame_count}... ({progress_pct:.1f}% | Valid: {valid_frame_count} | Windows: {len([w for w in all_windows if w['video_file'] == video_path.name])})")
            else:
                print(f"    Processing frame {frame_count}... (Valid: {valid_frame_count} | Windows: {len([w for w in all_windows if w['video_file'] == video_path.name])})")
        
        # Force garbage collection periodically to prevent memory buildup
        if processed_frame_count % 1000 == 0:
            gc.collect()
        
        # When buffer reaches WINDOW_SIZE, compute window statistics
        if len(ear_buffer) >= WINDOW_SIZE:
            # Use the last WINDOW_SIZE elements (in case buffer grew larger)
            ear_window = list(ear_buffer)[-WINDOW_SIZE:]
            mar_window = list(mar_buffer)[-WINDOW_SIZE:]
            pitch_window = list(pitch_buffer)[-WINDOW_SIZE:]
            yaw_window = list(yaw_buffer)[-WINDOW_SIZE:]
            roll_window = list(roll_buffer)[-WINDOW_SIZE:]
            
            # Calculate statistical features for this window
            mean_ear = float(np.mean(ear_window))
            std_ear = float(np.std(ear_window))
            
            mean_mar = float(np.mean(mar_window))
            max_mar = float(np.max(mar_window))
            
            mean_pitch = float(np.mean(pitch_window))
            std_pitch = float(np.std(pitch_window))
            
            mean_yaw = float(np.mean(yaw_window))
            std_yaw = float(np.std(yaw_window))
            
            mean_roll = float(np.mean(roll_window))
            std_roll = float(np.std(roll_window))
            
            # Create window feature dictionary
            window_features = {
                'label': label,
                'video_file': video_path.name,
                'start_frame': valid_frame_count - WINDOW_SIZE,
                'end_frame': valid_frame_count,
                'mean_ear': mean_ear,
                'std_ear': std_ear,
                'mean_mar': mean_mar,
                'max_mar': max_mar,
                'mean_pitch': mean_pitch,
                'std_pitch': std_pitch,
                'mean_yaw': mean_yaw,
                'std_yaw': std_yaw,
                'mean_roll': mean_roll,
                'std_roll': std_roll,
            }
            
            all_windows.append(window_features)
            
            # Slide window by STEP_SIZE: remove STEP_SIZE oldest frames
            for _ in range(min(STEP_SIZE, len(ear_buffer))):
                ear_buffer.popleft()
                mar_buffer.popleft()
                pitch_buffer.popleft()
                yaw_buffer.popleft()
                roll_buffer.popleft()
    
    cap.release()
    
    # Force cleanup after video processing
    gc.collect()
    
    if valid_frame_count == 0:
        print(f"    [WARN] No valid frames extracted from: {video_path.name}")
    else:
        print(f"    [OK] Extracted {len([w for w in all_windows if w['video_file'] == video_path.name])} windows from {valid_frame_count} valid frames")


def scan_and_process_dataset():
    """
    Recursively scan raw_dataset/ folder and process all videos.
    Auto-label based on folder name.
    """
    if not BASE_DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {BASE_DATASET_DIR.resolve()}")
        print(f"Please create the directory structure:")
        for folder_name in FOLDER_LABEL_MAP.keys():
            print(f"  - {BASE_DATASET_DIR / folder_name}/")
        print(f"\nNote: The script expects videos in: {BASE_DATASET_DIR.resolve()}")
        return
    
    all_windows = []
    
    # Recursively scan for folders matching our label map
    for folder_path in BASE_DATASET_DIR.iterdir():
        if not folder_path.is_dir():
            continue
        
        folder_name = folder_path.name
        
        # Check if folder name matches our label map
        if folder_name not in FOLDER_LABEL_MAP:
            print(f"[WARN] Unknown folder name '{folder_name}', skipping...")
            print(f"  Expected one of: {list(FOLDER_LABEL_MAP.keys())}")
            continue
        
        label = FOLDER_LABEL_MAP[folder_name]
        label_name = ["Awake", "Drowsy", "Phone", "Microsleep"][label]
        
        print(f"\n[INFO] Processing class '{label_name}' (Label {label}) from folder: {folder_path.name}")
        
        # Find all video files in this folder (recursively)
        video_files = []
        for ext in VIDEO_EXTS:
            video_files.extend(folder_path.rglob(f"*{ext}"))
            video_files.extend(folder_path.rglob(f"*{ext.upper()}"))
        
        if not video_files:
            print(f"  [WARN] No video files found in: {folder_path}")
            continue
        
        print(f"  Found {len(video_files)} video file(s)")
        
        # Process each video
        for idx, video_path in enumerate(sorted(video_files), 1):
            try:
                print(f"\n  [{idx}/{len(video_files)}] Processing video...")
                process_video(video_path, label, all_windows)
                
                # Save progress after each video to prevent data loss
                if all_windows:
                    temp_output = OUTPUT_CSV.with_suffix('.tmp.csv')
                    fieldnames = [
                        'label', 'video_file', 'start_frame', 'end_frame',
                        'mean_ear', 'std_ear', 'mean_mar', 'max_mar',
                        'mean_pitch', 'std_pitch', 'mean_yaw', 'std_yaw',
                        'mean_roll', 'std_roll',
                    ]
                    with temp_output.open('w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(all_windows)
                    print(f"    [SAVED] Progress saved: {len(all_windows)} windows so far")
                    
            except KeyboardInterrupt:
                print(f"\n[INFO] Processing interrupted by user")
                print(f"[INFO] Saving {len(all_windows)} windows collected so far...")
                # Save what we have before exiting
                if all_windows:
                    fieldnames = [
                        'label', 'video_file', 'start_frame', 'end_frame',
                        'mean_ear', 'std_ear', 'mean_mar', 'max_mar',
                        'mean_pitch', 'std_pitch', 'mean_yaw', 'std_yaw',
                        'mean_roll', 'std_roll',
                    ]
                    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(all_windows)
                    print(f"[SAVED] Partial results saved to {OUTPUT_CSV}")
                raise  # Re-raise to exit
            except Exception as e:
                print(f"    [ERROR] Failed to process {video_path.name}: {e}")
                print(f"    [INFO] Continuing with next video...")
                continue
    
    if not all_windows:
        print("\n[WARN] No windows were extracted from any videos.")
        return
    
    # Save to CSV
    fieldnames = [
        'label',
        'video_file',
        'start_frame',
        'end_frame',
        'mean_ear',
        'std_ear',
        'mean_mar',
        'max_mar',
        'mean_pitch',
        'std_pitch',
        'mean_yaw',
        'std_yaw',
        'mean_roll',
        'std_roll',
    ]
    
    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_windows)
    
    print(f"\n[DONE] Saved {len(all_windows)} windows to {OUTPUT_CSV.resolve()}")
    
    # Print statistics
    print("\n" + "="*70)
    print("PROCESSING STATISTICS")
    print("="*70)
    label_counts = {}
    for window in all_windows:
        label = window['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    label_names = ["Awake", "Drowsy", "Phone", "Microsleep"]
    for label, count in sorted(label_counts.items()):
        print(f"  {label_names[label]:15} (Label {label}): {count:5} windows")
    print(f"  {'TOTAL':15}: {len(all_windows):5} windows")
    print("="*70)


def main():
    print("="*70)
    print("VIDEO SEQUENCE PROCESSOR - Time-Series Analysis")
    print("="*70)
    print(f"Dataset directory: {BASE_DATASET_DIR.resolve()}")
    print(f"Output file: {OUTPUT_CSV.resolve()}")
    print(f"Window size: {WINDOW_SIZE} frames (~{WINDOW_SIZE/30:.1f} seconds at 30fps)")
    print(f"Step size: {STEP_SIZE} frames (overlap: {WINDOW_SIZE - STEP_SIZE} frames)")
    print(f"Frame skip: {FRAME_SKIP} (processing every {FRAME_SKIP} frame(s))")
    print(f"Max frame width: {MAX_FRAME_WIDTH}px (for speed optimization)")
    print(f"Debug verbose: {DEBUG_VERBOSE}")
    print("="*70)
    print("Press Ctrl+C to stop and save progress")
    print("="*70)
    print()
    
    scan_and_process_dataset()


if __name__ == "__main__":
    main()
