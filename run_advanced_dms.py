"""
Advanced Driver Monitoring System - Main Entry Point
====================================================

Production-ready driver drowsiness detection with:
- Multi-state detection (Awake, Yawning, Drowsy, Sleeping)
- Temporal smoothing
- Glasses-aware detection
- Real-time performance optimization

Usage:
    python run_advanced_dms.py [--camera 0] [--fps 30] [--thresholds custom_thresholds.json]
"""

import cv2
import argparse
import time
import sys
from drowsiness_detector import (
    DriverMonitoringSystem, Thresholds, DriverState
)
import FaceMeshModule as fm

# Try to import winsound for Windows alert sound
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


def play_alert_sound(frequency: int = 2500, duration: int = 200):
    """Play alert sound (cross-platform)"""
    if HAS_WINSOUND:
        try:
            winsound.Beep(frequency, duration)
        except:
            pass
    else:
        # For Linux/Mac, could use system beep or audio file
        print("\a")  # ASCII bell


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Advanced Driver Monitoring System')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--fps', type=float, default=30.0, help='Expected FPS (default: 30)')
    parser.add_argument('--window-width', type=int, default=1280, help='Window width (default: 1280)')
    parser.add_argument('--window-height', type=int, default=720, help='Window height (default: 720)')
    parser.add_argument('--model', type=str, default=None, help='Path to ML model (optional)')
    parser.add_argument('--no-sound', action='store_true', help='Disable alert sounds')
    
    args = parser.parse_args()
    
    # Initialize face detector
    print("⏳ Initializing face detector...")
    # Giảm confidence thresholds để nhận diện tốt hơn
    # Nếu vẫn không nhận diện được, thử giảm xuống 0.1 hoặc 0.05
    face_detector = fm.FaceMeshDetector(maxFaces=1, minDetectionCon=0.1, minTrackCon=0.1)
    
    # Initialize thresholds với giá trị đã được điều chỉnh để nhạy hơn
    thresholds = Thresholds(
        # EAR thresholds (nhạy hơn)
        EAR_CLOSED=0.18,  # Giảm từ 0.20
        EAR_SLEEP=0.12,   # Giảm từ 0.15
        # Head pose (nhạy hơn với gật đầu)
        PITCH_DROWSY=-10.0,  # Giảm từ -15.0
        PITCH_ALERT=-20.0,   # Giảm từ -25.0
        # Temporal (phản ứng nhanh hơn)
        EYES_CLOSED_DURATION_WARNING=0.8,  # Giảm từ 1.0
        EYES_CLOSED_DURATION_ALERT=1.5,    # Giảm từ 2.0
        YAWN_DURATION=1.0,  # Giảm từ 1.5
        # Blink frequency
        BLINK_FREQ_LOW=10.0,  # Tăng từ 8.0
    )
    
    # Initialize DMS
    print("⏳ Initializing Driver Monitoring System...")
    dms = DriverMonitoringSystem(
        face_detector=face_detector,
        thresholds=thresholds,
        fps=args.fps,
        use_model=(args.model is not None),
        model_path=args.model
    )
    
    # Initialize camera
    print(f"📹 Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open camera {args.camera}")
        return 1
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create window
    window_name = "Advanced DMS - Press 'q' to quit, 'f' for fullscreen, 'r' to reset"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)
    
    print("✅ System ready!")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  'f' - Toggle fullscreen")
    print("  'r' - Reset window size")
    print("=" * 60)
    
    # Alert state tracking
    last_alert_time = 0
    alert_cooldown = 0.5  # Don't play sound more than once per 0.5s
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Cannot read frame from camera")
                break
            
            # Flip frame horizontally (mirror mode)
            frame = cv2.flip(frame, 1)
            
            # Process frame
            state, metrics, output_frame = dms.process_frame(frame)
            
            # Trigger alert sound (with cooldown)
            if dms.trigger_alert() and not args.no_sound:
                current_time = time.time()
                if current_time - last_alert_time > alert_cooldown:
                    play_alert_sound()
                    last_alert_time = current_time
            
            # Display frame
            cv2.imshow(window_name, output_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✅ Shutting down...")
                break
            elif key == ord('f'):
                # Toggle fullscreen
                current_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if current_prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, args.window_width, args.window_height)
                    print("📺 Fullscreen off")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("📺 Fullscreen on")
            elif key == ord('r'):
                # Reset window size
                cv2.resizeWindow(window_name, args.window_width, args.window_height)
                print(f"📐 Window reset to {args.window_width}x{args.window_height}")
            
            # Print state to console (optional, can be disabled for performance)
            if state != DriverState.AWAKE:
                print(f"[{time.strftime('%H:%M:%S')}] State: {state.value} | "
                      f"EAR: {metrics.ear_avg:.3f} | MAR: {metrics.mar:.3f}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = dms.get_performance_stats()
        print("\n" + "=" * 60)
        print("📊 Final Statistics:")
        print(f"  Average FPS: {stats['fps']:.2f}")
        print(f"  Final State: {stats['current_state']}")
        print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

