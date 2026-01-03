import cv2                
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat
import numpy as np
import time              

class FaceMeshDetector:
    def __init__(self, staticMode = False, maxFaces = 1, refine_landmarks = False, minDetectionCon = 0.5, minTrackCon = 0.5):
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

    def findFaceMesh(self, img, draw=True):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=img_rgb)
        
        # Detect face landmarks
        detection_result = self.detector.detect(mp_image)
        
        face = []
        
        if detection_result.face_landmarks:
            # MediaPipe provides landmarks in order (0-467 for 468 landmarks)
            for face_landmarks in detection_result.face_landmarks:
                ih, iw, ic = img.shape
                
                # Create face list indexed by landmark ID (0-467)
                # face_landmarks is a list of 468 NormalizedLandmark objects
                for landmark_id, landmark in enumerate(face_landmarks):
                    x = int(landmark.x * iw)
                    y = int(landmark.y * ih)
                    
                    if draw:
                        cv2.putText(img, str(landmark_id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                    
                    face.append([landmark_id, x, y])
                
                # Draw face mesh if requested
                if draw and len(face_landmarks) > 0:
                    # Draw connections (simplified - you can add more connections)
                    self._draw_face_mesh(img, face_landmarks, iw, ih)
        
        return img, face
    
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
    detector = FaceMeshDetector(maxFaces=1)
    while True:
        success, img = cap.read()
        if not success:
            print("End of video or cannot read frame")
            break
            
        img, face = detector.findFaceMesh(img, draw=False)
        
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
