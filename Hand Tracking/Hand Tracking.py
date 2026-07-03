# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
from pathlib import Path
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python.vision import drawing_utils
import time

cap = cv2.VideoCapture(0)

# API Calling
baseOptions = mp.tasks.BaseOptions
handLandmarker = mp.tasks.vision.HandLandmarker
handLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# custom styles

custom_dots = drawing_utils.DrawingSpec(color=(255, 0, 0),
                                        thickness=5, 
                                        circle_radius=4
                                        )

custom_lines = drawing_utils.DrawingSpec(color=(0, 255, 0), 
                                        thickness=5
                                        )

# Configuration
options = handLandmarkerOptions(
    base_options = baseOptions(model_asset_path='hand_landmarker.task'),
    running_mode = VisionRunningMode.VIDEO,
    num_hands=5,
    min_hand_detection_confidence=0.5
)

# calculation the fps 1
pTime = 0
cTime = 0


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

        # flipping the image Y-AXIS : 
        img = cv2.flip(img,1)

        res = landmarker.detect_for_video(mp_img,timestamp_ms)

        if res.hand_landmarks:
            for hand in res.hand_landmarks:
                for id , lm in enumerate(hand):
                    # print(id,lm)

                    h, w, c = img.shape
                    cx, cy = int(lm.x*w) , int(lm.y*h)
                    print(id," : ", "x : ",cx, "y : ",cy)

                    if id == 0 :
                        cv2.circle(img, (cx,cy),25,(255,0,255),cv2.FILLED)

                    cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)

                    if id % 4 == 0 and id>0:
                        cv2.circle(img, (cx,cy),25,(201,97,48),cv2.FILLED)

                    if id == 8 :
                        img = cv2.line(img, (cx,cy),(cx+1,cy+1),(0,0,0), 10)
                
                drawing_utils.draw_landmarks(
                    img,
                    hand,
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=custom_dots,
                    connection_drawing_spec=custom_lines,
                )
        
        # calculation the fps 2
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),
                    (10,70),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,(255,0,255),3)

        # print(res.hand_landmarks, "\n")

        # display
        cv2.imshow('live',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF ==ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()