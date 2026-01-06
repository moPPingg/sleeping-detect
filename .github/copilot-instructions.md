# Copilot instructions for this repository

Purpose: Help an AI coding assistant be productive immediately when editing or extending the Drowsiness Detection System.

## Big-picture architecture ✅
- Face detection & landmarks: `FaceMeshModule.py` (MediaPipe Face Landmarker). The module returns face landmark lists used downstream.
- Feature extraction: `drowsiness_detector.py` computes EAR, MAR, and head pose (pitch/yaw/roll).
- Decision & UI: `run_advanced_dms.py` (production entrypoint) and `run_advanced_dmsNEW.py` (simpler runner) implement multi-state logic (AWAKE, YAWNING, DROWSY, SLEEPING, MICROSLEEP, DISTRACTED), temporal smoothing via counters and thresholds, and alarms/voice.
- Glasses detection: `glasses_detector.py` (Haar cascade) used to adapt behavior for eyeglass wearers.
- Training & experiments: `Complete_Drowsiness_Detection.ipynb` contains training/analysis cells used to create `drowsiness_model.pkl` and `scaler.pkl`.

## Quick dev workflows & commands ⚙️
- Setup virtualenv and deps (Windows):
  - `python -m venv .venv` && `.\.venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`
- Run the system (production script):
  - `python run_advanced_dms.py --camera 0 --fps 30`
  - For the simpler runner: `python run_advanced_dmsNEW.py`
- Training (not a script in repo): use the notebook `Complete_Drowsiness_Detection.ipynb` to train & produce `drowsiness_model.pkl` and `scaler.pkl`.

## Project-specific conventions & patterns 🧭
- Mixed Vietnamese/English comments and UI strings — preserve or copy style when editing user-facing messages.
- Thresholds and tuning are done inline as constants or via a `Thresholds`-style object (see `run_advanced_dms.py` and `QUICK_START.md`). Prefer explicit constants for clarity when adding new checks.
- Temporal smoothing via counters (e.g., `counters['drowsy']`) used extensively. To add a new state, create counter + threshold logic in the runner or in DMS class.
- Auto-download behavior: files are automatically fetched at runtime if missing:
  - `face_landmarker.task` (FaceMeshModule)
  - `haarcascade_eye_tree_eyeglasses.xml` (GlassesDetector)
  Keep network I/O in mind for tests and CI.

## Integration & runtime notes 🔌
- Model & scaler files: `drowsiness_model.pkl`, `scaler.pkl` are optional; code checks for their presence and gracefully falls back to rule-based logic if absent.
- Platform specifics:
  - `winsound` is used for Windows beep; fallback is a console bell. `pyttsx3` is used for voice output.
  - MediaPipe tasks API is used; code expects a local `face_landmarker.task` model file.
- Camera properties and performance tuning: lowering resolution and `minDetectionCon`/`minTrackCon` in `FaceMeshDetector` helps increase detection rate and FPS.

## Notable bugs / gotchas to call out 🔍
- Head-pose coordinate mismatch: `FaceMeshModule.findFaceMesh` returns pixel coordinates (`[x, y]`), but `drowsiness_detector.get_head_pose` multiplies landmarks by `img_w` and `img_h` as if they were normalized. This likely causes incorrect head-pose results — verify coordinate conventions before changing logic.
- Some docs reference files that are not present (e.g., `trainmodel.py`, `test_dms.py`) — prefer using the notebook or search the repo before adding commands to docs.

## Helpful examples for code edits ✍️
- To add a new alert state, follow the pattern in `run_advanced_dmsNEW.py`:
  - Extract metrics from `DrowsinessDetector.analyze_frame`
  - Use counters and a threshold constant
  - Set `state` and trigger `start_alarm()` / `trigger_voice()` where necessary
- For detection tuning, change `FaceMeshDetector(maxFaces=1, minDetectionCon=0.1, minTrackCon=0.1)` — lower numbers increase sensitivity.

## Testing & debugging tips 🧪
- Run notebooks cells that generate sample frames or use the `main()` function in `FaceMeshModule.py` to visually verify landmark positions and labels.
- Verify model behavior by loading `drowsiness_model.pkl` and running `scaler.transform([[ear, mar, pitch, yaw, roll]])` in a Python REPL.
- For CI or headless testing, stub out network downloads and the camera with recorded video from `/videos/` and monkeypatch `FaceMeshDetector` to return deterministic landmarks.

---
If anything here is unclear or you'd like me to include more examples (e.g., code snippets for unit tests or a short checklist for PR reviewers), tell me which area to expand. 
