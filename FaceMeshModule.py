import cv2                
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat
import numpy as np
import time              

class FaceMeshDetector:
    def __init__(self, staticMode = False, maxFaces = 1, refine_landmarks = False, minDetectionCon = 0.3, minTrackCon = 0.3):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine_landmarks = refine_landmarks    
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        # Initialize MediaPipe Face Landmarker
        # Download model if not exists
        import os
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=maxFaces,
            min_face_detection_confidence=minDetectionCon,
            min_face_presence_confidence=minTrackCon,
            min_tracking_confidence=minTrackCon
        )
        
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # For drawing (using OpenCV directly since drawing_utils is not available)
        self.draw_spec = None
        
        # Eye landmark indices used in drowsiness detection
        # Left eye (from driver's perspective = right eye on screen)
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # [outer, top1, top2, inner, bottom1, bottom2]
        # Right eye
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def findFaceMesh(self, img, draw=True, draw_eyes=True, debug=False):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=img_rgb)
        
        # Detect face landmarks
        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            if debug:
                print(f"❌ Error in MediaPipe detection: {e}")
            return img, []
        
        face = []
        
        # Check if face_landmarks exists and is not empty
        if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
            # MediaPipe provides landmarks in order (0-467 for 468 landmarks)
            for face_landmarks in detection_result.face_landmarks:
                if not face_landmarks or len(face_landmarks) == 0:
                    continue
                    
                ih, iw, ic = img.shape
                
                # Create face list indexed by landmark ID (0-467)
                # face_landmarks is a list of 468 NormalizedLandmark objects
                for landmark_id, landmark in enumerate(face_landmarks):
                    try:
                        x = int(landmark.x * iw)
                        y = int(landmark.y * ih)
                        face.append([landmark_id, x, y])
                    except Exception as e:
                        if debug:
                            print(f"❌ Error processing landmark {landmark_id}: {e}")
                        continue
                
                # Draw face mesh if requested
                if draw and len(face_landmarks) > 0:
                    # Draw connections (simplified - you can add more connections)
                    self._draw_face_mesh(img, face_landmarks, iw, ih)
                
                # Draw eye points specifically if requested
                if draw_eyes and len(face_landmarks) > 0:
                    self._draw_eye_points(img, face_landmarks, iw, ih)
                
                # Only process first face if maxFaces=1
                break
        else:
            # No face detected - show message
            h, w = img.shape[:2]
            cv2.putText(img, "NO FACE DETECTED", (w//2 - 150, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if debug:
                print(f"⚠️ No face detected. Detection result: face_landmarks={detection_result.face_landmarks}")
        
        return img, face
    
    def _draw_eye_points(self, img, landmarks, img_w, img_h):
        """Draw eye landmark points with labels for verification"""
        # Left eye points (from driver's perspective)
        left_eye_colors = {
            33: (255, 0, 0),    # Outer corner - RED
            160: (0, 255, 0),   # Top 1 - GREEN
            158: (0, 255, 255), # Top 2 - YELLOW
            133: (255, 0, 255), # Inner corner - MAGENTA
            153: (255, 255, 0), # Bottom 1 - CYAN
            144: (0, 165, 255)  # Bottom 2 - ORANGE
        }
        
        # Right eye points
        right_eye_colors = {
            362: (255, 0, 0),   # Outer corner - RED
            385: (0, 255, 0),   # Top 1 - GREEN
            387: (0, 255, 255), # Top 2 - YELLOW
            263: (255, 0, 255), # Inner corner - MAGENTA
            373: (255, 255, 0), # Bottom 1 - CYAN
            380: (0, 165, 255)  # Bottom 2 - ORANGE
        }
        
        # Draw left eye points
        for idx in self.LEFT_EYE_INDICES:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                x = int(landmark.x * img_w)
                y = int(landmark.y * img_h)
                color = left_eye_colors.get(idx, (255, 255, 255))
                # Draw circle
                cv2.circle(img, (x, y), 5, color, -1)
                # Draw label
                cv2.putText(img, str(idx), (x + 8, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw right eye points
        for idx in self.RIGHT_EYE_INDICES:
            if idx < len(landmarks):
                landmark = landmarks[idx]
                x = int(landmark.x * img_w)
                y = int(landmark.y * img_h)
                color = right_eye_colors.get(idx, (255, 255, 255))
                # Draw circle
                cv2.circle(img, (x, y), 5, color, -1)
                # Draw label
                cv2.putText(img, str(idx), (x + 8, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw connections between eye points for better visualization
        # Left eye connections
        left_eye_connections = [
            (self.LEFT_EYE_INDICES[0], self.LEFT_EYE_INDICES[3]),  # Outer to inner
            (self.LEFT_EYE_INDICES[1], self.LEFT_EYE_INDICES[4]),  # Top1 to Bottom1
            (self.LEFT_EYE_INDICES[2], self.LEFT_EYE_INDICES[5])   # Top2 to Bottom2
        ]
        
        for start_idx, end_idx in left_eye_connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = (int(landmarks[start_idx].x * img_w), int(landmarks[start_idx].y * img_h))
                pt2 = (int(landmarks[end_idx].x * img_w), int(landmarks[end_idx].y * img_h))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        
        # Right eye connections
        right_eye_connections = [
            (self.RIGHT_EYE_INDICES[0], self.RIGHT_EYE_INDICES[3]),  # Outer to inner
            (self.RIGHT_EYE_INDICES[1], self.RIGHT_EYE_INDICES[4]),  # Top1 to Bottom1
            (self.RIGHT_EYE_INDICES[2], self.RIGHT_EYE_INDICES[5])   # Top2 to Bottom2
        ]
        
        for start_idx, end_idx in right_eye_connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = (int(landmarks[start_idx].x * img_w), int(landmarks[start_idx].y * img_h))
                pt2 = (int(landmarks[end_idx].x * img_w), int(landmarks[end_idx].y * img_h))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    def _draw_face_mesh(self, img, landmarks, img_w, img_h):
        # Draw basic face mesh connections
        # This is a simplified version - you can expand with more connections
        connections = [
            # Face outline
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
            (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
            # Left eyebrow
            (17, 18), (18, 19), (19, 20), (20, 21),
            # Right eyebrow
            (22, 23), (23, 24), (24, 25), (25, 26),
            # Nose
            (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
            # Left eye
            (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
            # Right eye
            (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
            # Mouth
            (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
            (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = (int(landmarks[start_idx].x * img_w), int(landmarks[start_idx].y * img_h))
                pt2 = (int(landmarks[end_idx].x * img_w), int(landmarks[end_idx].y * img_h))
                cv2.line(img, pt1, pt2, (0, 255, 0), 1)

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0  
    # Lower confidence for better detection
    detector = FaceMeshDetector(maxFaces=1, minDetectionCon=0.3, minTrackCon=0.3)
    while True:
        success, img = cap.read()
        if not success:
            print("End of video or cannot read frame")
            break
        
        # Enable drawing to see eye points
        img, face = detector.findFaceMesh(img, draw=True, draw_eyes=True)
        
        # Show face detection status
        if face:
            cv2.putText(img, f'Face Detected: {len(face)} landmarks', (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'NO FACE DETECTED', (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cTime = time.time()      
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime            
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), 
                   cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()