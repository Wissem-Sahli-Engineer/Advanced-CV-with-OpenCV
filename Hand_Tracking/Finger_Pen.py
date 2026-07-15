# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
from pathlib import Path
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python.vision import drawing_utils
import time
# pyrefly: ignore [missing-import]
import numpy as np

cap = cv2.VideoCapture(0)

baseOptions = mp.tasks.BaseOptions
handLandmarker = mp.tasks.vision.HandLandmarker
handLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configuration
options = handLandmarkerOptions(
    base_options = baseOptions(model_asset_path='hand_landmarker.task'),
    running_mode = VisionRunningMode.VIDEO,
    num_hands=5,
    min_hand_detection_confidence=0.5
)

# 1. Initialize drawing canvas and points BEFORE the loop
canvas = None
px, py = 0, 0

with handLandmarker.create_from_options(options) as landmarker:
    test , img = cap.read()
    frame_count = 0

    while test:
        test , img = cap.read()
        if not test or img is None:
            break

        # Flip the image horizontally (Y-axis) for a mirror effect
        img = cv2.flip(img, 1)

        # fps 
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        
        timestamp_ms = int((frame_count / fps) * 1000)
        frame_count += 1

         # preprocessing
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        res = landmarker.detect_for_video(mp_img,timestamp_ms)
        
        # 2. Create the canvas once when the first image is read
        if canvas is None:
            canvas = np.zeros_like(img)

        if res.hand_landmarks: 
            for hand in res.hand_landmarks:
                index = hand[8] 

                h, w, c = img.shape
                cx, cy = int(index.x*w), int(index.y*h)

                if px == 0 and py == 0:
                    px, py = cx, cy

                cv2.line(canvas, (px, py), (cx, cy), (255, 0, 255), 10)
                px, py = cx, cy
        else:
            px, py = 0, 0

        combined_img = cv2.add(img, canvas)


        cv2.imshow('live', cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()
