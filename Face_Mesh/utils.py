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
class FaceMesh():
    def __init__(self,
                model_path = "face_landmarker.task",
                num_faces = 4,
                confidence = 0.5):
        
        # Arguments
        self.model_path = model_path
        self.num_faces = num_faces
        self.confidence = confidence

        # APIs
        self.Baseoptions = mp.tasks.BaseOptions
        self.faceLandmarker = mp.tasks.vision.FaceLandmarker
        self.faceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        #  Options configuration
        self.Options = self.faceLandmarkerOptions(
            base_options = self.Baseoptions(model_asset_path=self.model_path),
            running_mode = self.VisionRunningMode.VIDEO,
            num_faces = self.num_faces,
            min_face_presence_confidence = self.confidence,
            min_face_detection_confidence = self.confidence,
            min_tracking_confidence = self.confidence
        )

        # Build
        self.detector = self.faceLandmarker.create_from_options(self.Options)

    def findFace(self,img,res, draw = True):

        h, w, c = img.shape

        all_faces = []
        if res.face_landmarks :
            for face in res.face_landmarks:

                if draw :

                    if draw == 1 : 
                        drawing_utils.draw_landmarks(
                            img,
                            face,
                            mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                            landmark_drawing_spec=custom_dots,
                            connection_drawing_spec=custom_lines,
                            )

                    if draw == 2 :
                        drawing_utils.draw_landmarks(
                            img,
                            face,
                            mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS, 
                            landmark_drawing_spec=None, 
                            connection_drawing_spec=custom_lines
                            )

                lmList = []
                for id , lm in enumerate(face):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w) , int(lm.y*h)
                    lmList.append([id,cx,cy])

                all_faces.append(lmList)

        return all_faces





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