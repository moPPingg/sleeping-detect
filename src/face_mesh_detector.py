import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat
import os
import urllib.request

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        # Tự động tải model
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading face landmarker model...")
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

    def findFaceMesh(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=img_rgb)
        
        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            return img, []

        face = []
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                ih, iw, ic = img.shape
                for landmark in face_landmarks:
                    x = int(landmark.x * iw)
                    y = int(landmark.y * ih)
                    # Chỉ lấy [x, y]
                    face.append([x, y])
                
                if draw:
                    self._draw_face_mesh(img, face_landmarks, iw, ih)
                
                break # Chỉ lấy 1 mặt
                
        return img, face

    def _draw_face_mesh(self, img, landmarks, iw, ih):
        # Vẽ các điểm neo quan trọng
        indices = [33, 133, 362, 263, 1, 61, 291] 
        for idx in indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * iw)
                y = int(landmarks[idx].y * ih)
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        if not success: break
        img, face = detector.findFaceMesh(img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()