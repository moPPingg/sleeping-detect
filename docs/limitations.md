# Limitations

This project is a personal learning project, so it still has several practical limitations:

- The dataset is relatively limited, and microsleep is especially difficult to represent clearly. Because of that, the model can still confuse microsleep with awake samples.
- The current webcam setup uses a fixed `640 x 480` resolution, which is not very flexible under different lighting conditions.
- Runtime performance is not heavily optimized yet. In the current setup, FPS typically stays around `20-30 FPS`.
- The current system relies mainly on AI-based detection, so it still has room for improvement with additional handcrafted behavioral signals.

## Future Improvement Ideas

- Collect video data from a wider range of people, ideally `50+` participants.
- Explore HDR-style image processing or tone mapping to improve robustness under difficult lighting.
- Add a face-detection confidence filter so low-confidence frames can be skipped.
- Add extra signals such as eye open/close ratio and blink frequency to improve detection accuracy.
