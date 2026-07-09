# Face Mesh Project

A real-time face landmarking and face mesh application built using OpenCV and the modern Google MediaPipe Face Landmarker Tasks API.

## Project Structure

- **`Face_Mesh.py`**: The main execution script. It captures video frames from a source (`us.MOV`), processes them, runs the face landmarker model, and outputs/displays the facial landmarks.
- **`utils.py`**: A helper module containing:
  - `FaceMesh` class: Configures the MediaPipe Face Landmarker API and builds the landmarker pipeline.
  - `findFace()`: Processes results and draws the face landmarks on the screen.
    - `draw = 1`: Draws full face tesselation (all mesh points).
    - `draw = 2`: Draws contours only (outlines of eyes, lips, face shape) without showing individual points.
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
Download the pre-trained MediaPipe Face Landmarker task model file and place it in the `Face_Mesh` directory:
- [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)

---

## How to Run

Execute the face mesh script:
```bash
python Face_Mesh.py
```
* **Controls:** Press the **Spacebar** key to close the window and exit the program.
