# Technical Breakdown: Video Sequence Processing for Time-Series Analysis

## Overview

This document explains the technical implementation of `process_video_sequence.py`, which transforms raw video data into time-series features suitable for detecting microsleep sequences in driver drowsiness detection.

---

## 1. Sliding Window Logic

### How It Works

The sliding window technique divides a continuous video stream into overlapping segments (windows) for temporal analysis. Here's the step-by-step process:

#### **Window Initialization**
```python
ear_buffer = deque(maxlen=WINDOW_SIZE)  # WINDOW_SIZE = 30 frames
```

- Uses Python's `deque` (double-ended queue) with a maximum length of 30 frames
- Automatically discards oldest elements when new ones are added
- Each buffer stores one feature type (EAR, MAR, Pitch, Yaw, Roll)

#### **Frame-by-Frame Collection**
1. **Read Frame**: Extract a single frame from the video
2. **Detect Face**: Use MediaPipe Face Mesh to detect facial landmarks
3. **Extract Features**: Calculate EAR, MAR, and head pose for that frame
4. **Append to Buffer**: Add features to respective buffers

```python
ear_buffer.append(features['ear'])
mar_buffer.append(features['mar'])
pitch_buffer.append(features['pitch'])
# ... etc
```

#### **Window Completion & Statistics**
When the buffer reaches `WINDOW_SIZE` (30 frames):

1. **Calculate Statistics**: Compute mean and standard deviation for each feature
   ```python
   mean_ear = np.mean(ear_buffer)
   std_ear = np.std(ear_buffer)
   ```

2. **Create Window Feature Vector**: Combine all statistics into one row
   ```python
   window_features = {
       'mean_ear': mean_ear,
       'std_ear': std_ear,
       'mean_mar': mean_mar,
       'max_mar': max_mar,
       # ... etc
   }
   ```

3. **Slide Window**: Remove `STEP_SIZE` (10) oldest frames from buffers
   ```python
   for _ in range(STEP_SIZE):
       ear_buffer.popleft()  # Remove oldest frame
   ```

#### **Visual Example**

```
Frame Sequence: [1, 2, 3, ..., 30, 31, 32, ..., 40, 41, 42, ...]
                 └───────── Window 1 ─────────┘
                              └───────── Window 2 ─────────┘
                                           └───────── Window 3 ─────────┘

Window 1: Frames 1-30   → Extract statistics → Save row
Window 2: Frames 11-40  → Extract statistics → Save row  (overlapped by 20 frames)
Window 3: Frames 21-50  → Extract statistics → Save row  (overlapped by 20 frames)
```

**Why Overlap?**
- **Overlap = WINDOW_SIZE - STEP_SIZE = 30 - 10 = 20 frames**
- Ensures no temporal information is lost at boundaries
- Captures transitions between states (e.g., awake → drowsy)
- Provides more training samples from limited video data

---

## 2. Why Standard Deviation (std) is Critical for Microsleep Detection

### The Problem with Mean Alone

**Static Classification Limitation:**
- A single frame showing "eyes closed" could be:
  - A normal blink (0.1 seconds)
  - Drowsiness (2-3 seconds)
  - Microsleep (5+ seconds with head drop)

**Mean EAR alone cannot distinguish these!**

### Standard Deviation Reveals Motion Patterns

#### **Scenario 1: Normal Blink (Low std)**
```
Frames: [0.28, 0.27, 0.12, 0.15, 0.26, 0.28, ...]
        └───── Open ────┘ └─ Blink ┘ └───── Open ────┘

Mean EAR: ~0.25
Std EAR:  ~0.05  (Low variation - quick recovery)
```
**Interpretation**: Eyes briefly closed, quickly reopened → Normal blink

#### **Scenario 2: Drowsiness (Medium std)**
```
Frames: [0.28, 0.25, 0.20, 0.18, 0.19, 0.20, 0.22, ...]
        └───── Open ────┘ └─────── Closed ────────┘

Mean EAR: ~0.21
Std EAR:  ~0.04  (Moderate variation - gradual change)
```
**Interpretation**: Gradual eye closure, some fluctuation → Drowsy state

#### **Scenario 3: Microsleep (High std + Head Movement)**
```
Frames: [0.28, 0.15, 0.10, 0.08, 0.12, 0.25, 0.28, ...]
        └───── Open ────┘ └── Closed ───┘ └── Jerk Awake ───┘

Mean EAR: ~0.18
Std EAR:  ~0.08  (High variation - sudden changes)
Mean Pitch: +15° (Head dropped forward)
Std Pitch:  ~8°  (Head movement during microsleep)
```
**Interpretation**: 
- **High std_EAR**: Rapid eye closure followed by sudden reopening (jerk awake)
- **High std_Pitch**: Head nodding motion (drop → recovery)
- **Combined**: Classic microsleep pattern!

### Mathematical Insight

**Standard Deviation Formula:**
```
std = sqrt(Σ(xi - mean)² / N)
```

**What it Measures:**
- **Low std**: Values cluster tightly around mean → **Stable state**
- **High std**: Values spread widely → **Unstable/transitional state**

**For Microsleep Detection:**
- **std_EAR**: Detects eye flutter, rapid blinks, or sudden reopening
- **std_Pitch**: Detects head nodding motion (critical for microsleep)
- **std_Yaw**: Detects head turning (distraction vs. drowsiness)

### Real-World Example

**Microsleep Sequence:**
1. **Frame 1-10**: Eyes open (EAR = 0.28)
2. **Frame 11-20**: Eyes gradually close (EAR = 0.28 → 0.10)
3. **Frame 21-25**: Eyes fully closed, head drops (EAR = 0.08, Pitch = +20°)
4. **Frame 26-30**: Jerk awake (EAR = 0.28, Pitch = -5°)

**Window Statistics:**
- `mean_ear = 0.20` (could be confused with drowsiness)
- `std_ear = 0.09` (HIGH - indicates rapid changes)
- `mean_pitch = +8°` (head dropped)
- `std_pitch = 12°` (HIGH - indicates head movement)

**Machine Learning Model Can Learn:**
```
IF std_ear > 0.07 AND std_pitch > 10° AND mean_pitch > 5°:
    → MICROSLEEP
ELSE IF std_ear < 0.05 AND mean_ear < 0.20:
    → DROWSY (stable closed eyes)
ELSE:
    → AWAKE
```

---

## 3. Folder Traversal and Auto-Labeling Implementation

### Directory Structure

```
raw_dataset/
├── 0_tinh_tao/          → Label 0 (Awake)
│   ├── video1.mp4
│   ├── video2.avi
│   └── subfolder/
│       └── video3.mov
├── 1_buon_ngu/          → Label 1 (Drowsy)
│   └── drowsy_video.mp4
├── 2_dien_thoai/        → Label 2 (Phone Distraction)
│   └── phone_video.mp4
└── 3_ngu_trang/         → Label 3 (Microsleep)
    └── microsleep_video.mp4
```

### Implementation Logic

#### **Step 1: Folder Name Mapping**
```python
FOLDER_LABEL_MAP = {
    "0_tinh_tao": 0,      # Awake
    "1_buon_ngu": 1,      # Drowsy
    "2_dien_thoai": 2,    # Phone Distraction
    "3_ngu_trang": 3,     # Microsleep
}
```

**Key Design Decision:**
- Folder name **starts with label number** (e.g., `0_tinh_tao`)
- Enables **automatic label extraction** without manual configuration
- Prevents labeling errors (human can't mislabel if folder name is correct)

#### **Step 2: Recursive Scanning**
```python
for folder_path in BASE_DATASET_DIR.iterdir():
    if not folder_path.is_dir():
        continue
    
    folder_name = folder_path.name
    
    # Check if folder matches our label map
    if folder_name not in FOLDER_LABEL_MAP:
        print(f"[WARN] Unknown folder, skipping...")
        continue
    
    label = FOLDER_LABEL_MAP[folder_name]  # Auto-extract label
```

**Process:**
1. Iterate through all items in `raw_dataset/`
2. Filter for directories only
3. Extract folder name (e.g., `"0_tinh_tao"`)
4. Look up label in `FOLDER_LABEL_MAP`
5. Skip unknown folders (with warning)

#### **Step 3: Recursive Video Discovery**
```python
video_files = []
for ext in VIDEO_EXTS:
    video_files.extend(folder_path.rglob(f"*{ext}"))
    video_files.extend(folder_path.rglob(f"*{ext.upper()}"))
```

**`rglob()` Function:**
- **Recursive glob**: Searches folder and all subfolders
- Finds videos at any depth: `0_tinh_tao/video1.mp4` or `0_tinh_tao/subfolder/video2.avi`
- Case-insensitive: Matches both `.mp4` and `.MP4`

**Example:**
```
0_tinh_tao/
├── video1.mp4          ← Found
├── video2.AVI          ← Found (case-insensitive)
└── session1/
    └── video3.mov      ← Found (recursive)
```

#### **Step 4: Label Propagation**
```python
for video_path in sorted(video_files):
    process_video(video_path, label, all_windows)
    #                              ^^^^^
    #                              Label from folder name
```

**Every window extracted from videos in `0_tinh_tao/` gets `label = 0`**

### Benefits of This Approach

1. **No Manual Labeling**: Labels automatically assigned based on folder structure
2. **Scalable**: Add new videos by dropping them into correct folder
3. **Organized**: Folder structure serves as both organization and labeling
4. **Error Prevention**: Impossible to mislabel if folder name is correct
5. **Flexible**: Supports nested subfolders for organization

### Error Handling

```python
if folder_name not in FOLDER_LABEL_MAP:
    print(f"[WARN] Unknown folder name '{folder_name}', skipping...")
    print(f"  Expected one of: {list(FOLDER_LABEL_MAP.keys())}")
    continue
```

**What Happens:**
- Unknown folders (e.g., `test_videos/`) are **skipped with warning**
- Script continues processing other folders
- User gets clear feedback about what went wrong

---

## Summary

### Key Technical Concepts

1. **Sliding Window**: Divides video into overlapping 30-frame segments
2. **Statistical Aggregation**: Mean captures average state, std captures variability
3. **Standard Deviation**: Critical for detecting microsleep's characteristic instability
4. **Auto-Labeling**: Folder name → Label mapping eliminates manual errors
5. **Recursive Processing**: Handles nested folder structures automatically

### Why This Approach Works for Microsleep

**Microsleep is NOT a static state** - it's a **sequence of events**:
1. Eyes close (EAR drops)
2. Head drops (Pitch increases)
3. Jerk awake (EAR spikes, Pitch recovers)

**Static images miss the temporal pattern. Time-series analysis with std captures it!**

---

## References

- **EAR Calculation**: Soukupová & Čech (2016) - "Real-time Eye Blink Detection using Facial Landmarks"
- **Head Pose Estimation**: solvePnP algorithm from OpenCV
- **Sliding Window**: Standard technique in time-series analysis and signal processing
