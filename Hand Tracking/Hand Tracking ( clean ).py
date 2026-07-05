# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python.vision import drawing_utils
import time

from utils import get_fps, handDetector, custom_dots , custom_lines


def main(landmarker,stop =" ",):

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

        # flipping the image Y-AXIS : 
        img = cv2.flip(img,1)

        cv2.putText(img,str(int(fps)),
                    (10,70),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,(255,0,255),3 )

        # preprocessing
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        res = landmarker.landmarker.detect_for_video(mp_img,timestamp_ms)

        lmList = landmarker.findHands(img,res, draw =True,draw_finger=12)


        # display
        cv2.imshow('live',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF ==ord(stop):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    landmarker = handDetector()
    main(landmarker," ")