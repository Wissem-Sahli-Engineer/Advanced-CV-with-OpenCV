# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
from pathlib import Path
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python.vision import drawing_utils
import time

# custom styles
custom_dots = drawing_utils.DrawingSpec(color=(255, 0, 0),
                                        thickness=5, 
                                        circle_radius=4
                                        )

custom_lines = drawing_utils.DrawingSpec(color=(0, 255, 0), 
                                        thickness=5
                                        )

class handDetector():
    def __init__(self, 
                model_path='hand_landmarker.task', 
                num_hands = 4,
                confidence = 0.5 ) :
        
        # Arguments
        self.model_path = model_path
        self.num_hands = num_hands
        self.confidence = confidence
        # APIs
        self.baseOptions = mp.tasks.BaseOptions
        self.handLandmarker = mp.tasks.vision.HandLandmarker
        self.handLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        # configuration options
        self.options = self.handLandmarkerOptions(
            base_options=self.baseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.VIDEO,  
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.confidence
        )
        # Build 
        self.landmarker = self.handLandmarker.create_from_options(self.options)


def get_fps(cap, pTime,type='default'):
    if type == "default":
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        return fps, pTime

    elif type =="cap":
        fps= cap.get(cv2.CAP_PROP_FPS)
        if fps<= 0:
            return 30, pTime
        return fps, pTime
    else:
        return 30, pTime
