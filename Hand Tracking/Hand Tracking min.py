# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
from pathlib import Path
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python.vision import drawing_utils

cap = cv2.VideoCapture(0)

# API Calling
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

with handLandmarker.create_from_options(options) as landmarker:

    frame_count = 0
    while True:
        test , img = cap.read()

        if not test or img is None:
            break

        # fps 
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        
        timestamp_ms = int((frame_count / fps) * 1000)
        frame_count += 1

        # preprocessing
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        res = landmarker.detect_for_video(mp_img,timestamp_ms)

        if res.hand_landmarks:
            for hand in res.hand_landmarks:
                drawing_utils.draw_landmarks(
                    img,
                    hand,
                )
            

        # print(res.hand_landmarks, "\n")

        # display
        img = cv2.flip(img,1)
        cv2.imshow('live',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()