# Changes Log

---

## [2026-03-16] Sleep Warning Sensitivity

File changed: sleeping-detect/bone.py

## What was changed

1. EAR threshold (stricter eye-closure requirement)
- Before: `EAR_THRESHOLD = 0.18`
- After: `EAR_THRESHOLD = 0.16`
- Effect: Eyes must be more closed before counting toward sleep.

2. Consecutive frame requirement (longer confirmation window)
- Before: `SLEEP_FRAMES_REQ = 30`
- After: `SLEEP_FRAMES_REQ = 45`
- Effect: Sleep alarm needs longer continuous evidence before triggering.

3. Final alarm logic (major sensitivity fix)
- Before: `alarm = ear < self.EAR_THRESHOLD`
- After: `alarm = is_sleeping`
- Effect: Alarm no longer triggers immediately on a single low-EAR frame.
  It now triggers only after sustained detection (`consecutive_sleep_frames >= SLEEP_FRAMES_REQ`).

## Why this reduces false positives

- Single-frame blinks or brief eye narrowing are less likely to trigger alarms.
- Sleep state now requires persistence over time, matching the intended logic already present in the analyzer.

---

## [2026-03-16] Microsleep Detection Fixes

File changed: sleeping-detect/bone.py

1. Removed broken class remapping (bone.py line ~168)
- Before: `pred == 2` was remapped to `0` (Awake), silently killing all microsleep predictions
- After: `pred == 2` is kept as `2` (Microsleep); only `pred == 3` is remapped to `2` as a 4-class fallback
- Effect: AI microsleep predictions now actually reach the rest of the pipeline

2. Added dedicated microsleep flag (bone.py line ~252)
- Before: no separate microsleep flag existed
- After: `ai_microsleep = self.ai.last_pred == 2`
- Effect: Microsleep can be handled independently from general drowsiness

3. Added microsleep alarm display (bone.py line ~263)
- Before: no UI warning for microsleep
- After: `WARNING: MICROSLEEP!` shown in orange immediately when AI detects class 2
- Effect: Short microsleep events are surfaced in real time without needing sustained frames

4. Increased AI inference frequency (bone.py line ~248)
- Before: inference run every 3rd frame
- After: inference run every 2nd frame
- Effect: Tighter detection window for brief microsleep events (~0.5–2s duration)

---

## [2026-03-16] Decouple Microsleep from Sleep Counter

File changed: sleeping-detect/bone.py

1. Changed `ai_drowsy` flag assignment (bone.py line ~254)
- Before: `ai_drowsy = self.ai.last_pred in (1, 2)` — Microsleep predictions incremented `consecutive_sleep_frames`, making the SLEEPING alarm more sensitive
- After: `ai_drowsy = self.ai.last_pred == 1` — only Drowsy (class 1) feeds the sustained sleep counter
- Effect: Microsleep detections no longer bleed into the sleep alarm; each signal is fully independent

---

## [2026-03-16] Restore Microsleep Contribution to Sleep Counter

File changed: sleeping-detect/bone.py

1. Reverted `ai_drowsy` flag to include class 2 (bone.py line ~254)
- Before: `ai_drowsy = self.ai.last_pred == 1` — decoupled microsleep fully, but also removed class 3 (remapped to 2) from sleep counter, making sleep less sensitive than before the remapping fix
- After: `ai_drowsy = self.ai.last_pred in (1, 2)`
- Effect: Drowsy (1) and Microsleep (2) both feed `consecutive_sleep_frames`; sustained microsleep events correctly escalate to the SLEEPING alarm. Immediate MICROSLEEP warning display is unchanged.

---

## [2026-03-16] Restore Microsleep to Default + Sleep Beep

File changed: sleeping-detect/bone.py

1. Restored original microsleep remap (bone.py predict_from_face)
- Reverted: `pred == 2` is remapped back to `0` (Awake), `pred == 3` → `2`
- Removed: `ai_microsleep` flag and `WARNING: MICROSLEEP!` display
- Effect: Microsleep behaviour is back to original baseline

2. Added beep sound on sleep alarm (bone.py run loop)
- Added `import winsound`
- On `final_alarm`, every 30 frames: `winsound.Beep(1000, 500)` — 1000 Hz tone for 500ms
- Effect: Audible alert fires alongside the on-screen DANGER: SLEEPING! warning

---

## [2026-03-16] FPS Optimization (Observed ~15 FPS)

File changed: sleeping-detect/bone.py

1. Reduced MediaPipe FaceMesh workload
- Before: `FaceMesh(refine_landmarks=True)`
- After: `FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)`
- Effect: lower CPU cost per frame

2. Reduced AI inference frequency
- Before: run every 2nd frame
- After: run every 3rd frame
- Effect: less backbone inference overhead, higher overall FPS

3. Replaced blocking beep with async system sound
- Before: `winsound.Beep(1000, 500)` (blocks loop for 500ms)
- After: `winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)`
- Effect: avoids frame stalls when sleep alarm is active

---

## [2026-03-16] Sleep Detection Reliability Re-Tune

File changed: sleeping-detect/bone.py

1. Relaxed sleep thresholds
- Before: `EAR_THRESHOLD = 0.16`, `SLEEP_FRAMES_REQ = 45`
- After: `EAR_THRESHOLD = 0.20`, `SLEEP_FRAMES_REQ = 30`
- Effect: easier to trigger sustained sleep when eyes are actually closed

2. Restored precise eye landmarks
- Before: `refine_landmarks=False`
- After: `refine_landmarks=True`
- Effect: EAR values are more stable/accurate for eyelid closure detection

3. Increased AI update rate
- Before: AI inference every 3rd frame
- After: AI inference every 2nd frame
- Effect: faster AI state refresh, less lag in drowsy transitions

---

## [2026-03-16] Fix Sleep Count + Beep Trigger Reliability

File changed: sleeping-detect/bone.py

1. Replaced frame-modulo trigger for count/beep
- Before: count/beep only when `final_alarm` and `frame_count % 30 == 0`
- Problem: alarm could display but miss exact modulo frames, causing no count and no beep

2. Added alarm edge detection
- New state: `prev_final_alarm`
- Count increments once on rising edge (`False -> True`) of sleep alarm
- Effect: every displayed sleep episode is counted reliably

3. Added time-based beep interval
- New state: `last_beep_time`, `BEEP_INTERVAL_SEC = 1.0`
- While alarm is active, async sound plays at fixed 1-second intervals
- Effect: audible warning is consistent and no longer dependent on frame modulo alignment

---

## [2026-03-16] Add bone_micro Variant With Previous Microsleep Settings

Files changed: sleeping-detect/bone_micro.py

1. Created new file from `bone.py`
- Added `bone_micro.py` as an alternate runtime profile

2. Restored previous microsleep behavior in `bone_micro.py`
- Preserved class 2 (Microsleep) prediction by removing `2 -> 0` remap
- Kept class `3 -> 2` normalization for compatibility with 4-class outputs
- Added immediate `WARNING: MICROSLEEP!` overlay
- Kept `ai_drowsy = self.ai.last_pred in (1, 2)` so sustained microsleep can still escalate to sleep warning

3. Result
- `bone.py` remains default behavior with beep
- `bone_micro.py` provides the prior microsleep-focused behavior

---

## [2026-03-16] FPS Cap

File changed: sleeping-detect/bone.py

1. Added class-level constants to App
- `TARGET_FPS = 30`
- `FRAME_TIME = 1.0 / TARGET_FPS`

2. Added loop throttle at end of each frame
- Before: loop ran as fast as CPU/GPU allowed (unbounded)
- After: `time.sleep()` consumes remaining budget so each frame takes at least `1/30 s`
- Effect: Processing is capped at 30 FPS; reduces CPU usage and makes timing-dependent logic (SLEEP_FRAMES_REQ, YAWN_FRAMES_REQ) more predictable

---

## [2026-03-16] Pipeline Documentation

Files changed: sleeping-detect/PIPELINE.md

1. Added end-to-end pipeline document for `bone.py`
- Includes initialization, per-frame loop, analyzer logic, UI/alerts, FPS cap, and shutdown/report flow

2. Added Mermaid flowchart
- Visualizes control flow from capture to detection, alerting, and cleanup

3. Purpose
- Make troubleshooting and tuning easier by mapping each runtime stage and decision path
