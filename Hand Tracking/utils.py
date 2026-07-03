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



def main(landmarker):
    cap = cv2.VideoCapture(0)

    frame_count = 0
    pTime = time.time()

    while True:
        test , img = cap.read()
        if not test or img is None:
            break
        
        fps , pTime = get_fps(cap,pTime,"default")
        
        timestamp_ms = int((frame_count / get_fps(cap,0,type='cap')[0]) * 1000)
        frame_count += 1

        # preprocessing
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        res = landmarker.landmarker.detect_for_video(mp_img,timestamp_ms)

        if res.hand_landmarks:
            for hand in res.hand_landmarks:
                for id , lm in enumerate(hand):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w) , int(lm.y*h)

                    drawing_utils.draw_landmarks(
                        img,
                        hand,
                        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                        landmark_drawing_spec=custom_dots,
                        connection_drawing_spec=custom_lines,
                    )

        cv2.putText(img,str(int(fps)),
                    (10,70),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,(255,0,255),3)

        # display
        cv2.imshow('live',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF ==ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    landmarker = handDetector()
    main(landmarker) 