# Face Detection Project

A real-time face detection application built using OpenCV and the Google MediaPipe Face Detector API.

## Project Structure

- **`Face_Detection.py`**: The main execution script. It loads the video file (`us.MOV`), processes each frame, runs the face detector, and displays the output with face bounding boxes.
- **`utils.py`**: A helper module containing:
  - `FaceDetector` class: Handles the setup and instantiation of the MediaPipe Face Detector API.
  - `findFace()`: Processes detections to draw standard and stylized (fancy corner lines) bounding boxes with confidence scores.
  - `get_fps()`: Utility to calculate the frame rate.
- **`readme.md`**: Project documentation.

---

## Setup & Requirements

### 1. Install Dependencies
Make sure you have python, opencv, and mediapipe installed:
```bash
pip install opencv-python mediapipe numpy
```

### 2. Download the Model File
Download the pre-trained MediaPipe Face Detector model and place it in the `Face_Detection` directory:
- [blaze_face_short_range.tflite](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite)

---

## How to Run

Execute the main detection script from the folder:
```bash
python Face_Detection.py
```
* **Controls:** Press the **Spacebar** key while the video window is focused to exit.
