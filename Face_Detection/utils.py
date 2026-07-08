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

class FaceDetector():
    def __init__(self,
                model_path = "face_detection_full_range.tflite",
                confidence = 0.5):
        
        # Arguments
        self.model_path = model_path
        self.confidence = confidence

        # APIs
        self.Baseoptions = mp.tasks.BaseOptions
        self.faceDetector = mp.tasks.vision.FaceDetector
        self.faceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        #  Options configuration
        self.Options = self.faceDetectorOptions(
            base_options = self.Baseoptions(model_asset_path=self.model_path),
            running_mode = self.VisionRunningMode.VIDEO,
            min_detection_confidence = self.confidence,
            min_suppression_threshold = self.confidence,
        )

        # Build
        self.detector = self.faceDetector.create_from_options(self.Options)

    def findFace(self,img,res, draw = True, show_conf = False, fancy_draw = True):

        h, w, c = img.shape

        if res.detections :
            for face in res.detections:
                
                bbox = face.bounding_box

                x1 , y1 , w , h = bbox.origin_x , int(bbox.origin_y * 0.8) , bbox.width , bbox.height + int(bbox.origin_y * 0.2)

                if draw:
                    cv2.rectangle(img , (x1, y1) , (x1+w , y1+h) , (0,255,0),2)

                if show_conf:
                    cv2.putText(img,f'{int(face.categories[0].score * 100)}%',
                                (x1,y1-20), cv2.FONT_HERSHEY_PLAIN,
                                8 , (0,255,0), 4)

                if fancy_draw:

                    t = 10  # Line thickness
                    length = 50
                    color = (0, 255, 0)

                    # --- TOP LEFT CORNER ---
                    cv2.line(img, (x1, y1), (x1 + length, y1), color, t)
                    cv2.line(img, (x1, y1), (x1, y1 + length), color, t)

                    # --- TOP RIGHT CORNER ---
                    cv2.line(img, (x1 + w, y1), (x1 + w - length, y1), color, t)
                    cv2.line(img, (x1 + w, y1), (x1 + w, y1 + length), color, t)

                    # --- BOTTOM LEFT CORNER ---
                    cv2.line(img, (x1, y1 + h), (x1 + length, y1 + h), color, t)
                    cv2.line(img, (x1, y1 + h), (x1, y1 + h - length), color, t)

                    # --- BOTTOM RIGHT CORNER ---
                    cv2.line(img, (x1 + w, y1 + h), (x1 + w - length, y1 + h), color, t)
                    cv2.line(img, (x1 + w, y1 + h), (x1 + w, y1 + h - length), color, t)



# init " pTime = time.time() " before the While loop
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