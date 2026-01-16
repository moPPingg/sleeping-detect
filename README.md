# AI Driver Monitoring System (DMS)

## Description
AI-powered real-time drowsiness detection system using computer vision and machine learning. Monitors driver alertness and provides multi-modal alerts to prevent accidents. High accuracy with lightweight, hardware-agnostic deployment.

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the detection system with:

```bash
python main.py
```

- On first launch, calibrate for 5 seconds (look straight at camera).
- Receive voice + visual alerts when drowsiness is detected.
- Press `q` to quit at any time.

## Project Structure

```
sleeping-detect/
├── assets/
│   ├── Project_Report.html
│   ├── confusion_matrix.png
│   ├── data_balance.png
│   ├── feature_importance.png
│   └── ...
├── tools/
│   ├── auto_collect_drowsy.py
│   ├── convert_to_bw.py
│   ├── data_collector.py
│   ├── fix_colors.py
│   ├── fix_final.py
│   ├── glasses_detector.py
│   └── ...
├── main.py  # Entry point
├── FaceMeshModule.py
├── drowsiness_detector.py
├── drowsiness_model.pkl
├── face_data.csv
├── train_model.py
├── requirements.txt
├── .gitignore
├── README.md
└── ...
```

---

## Documentation
For full technical report, see: [📖 Project_Report.html](assets/Project_Report.html)

---

## License
MIT License
