# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
import time
from utils import get_fps

cap = cv2.VideoCapture('us.MOV')

pTime = time.time()

while True:

    test , img = cap.read()
    if not test or img is None :
        break

    # preprocessing


    fps , pTime = get_fps(cap,pTime)



    # display
    cv2.putText(img,str(int(fps)),
            (10,100),
            cv2.FONT_HERSHEY_PLAIN,
            7,(255,0,255),5)

    cv2.imshow('Video',img)
    if cv2.waitKey(1) & 0xFF==ord(' '):
        break

cap.release()
cv2.destroyAllWindows()