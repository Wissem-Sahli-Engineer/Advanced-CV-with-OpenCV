# Hand Tracking Project

A real-time hand tracking and interaction project built using OpenCV and the modern Google MediaPipe Tasks API.

## Project Structure

- **`Hand Tracking.py`**: A standalone script that runs hand tracking, extracts hand landmarks, and prints coordinates.
- **`Hand Tracking ( clean ).py`**: A modularized implementation of the hand tracking script that imports helper utilities and uses the `handDetector` class.
- **`Finger_Pen.py`**: A virtual drawing application where you can draw on the screen using your index finger.
- **`utils.py`**: A utility library containing:
  - `handDetector` class: Wraps the MediaPipe Hand Landmarker Tasks API configuration and landmark extraction.
  - `get_fps()` helper function: Calculates the frame rate.
  - Custom drawing configurations for dots and lines.
- **`solutions/Hand Tracking ( solutions ).py`**: A legacy version using the older, deprecated `mp.solutions.hands` API (requires older Python and MediaPipe versions).

---

## Setup & Requirements

### 1. Install Dependencies
Make sure you have python, opencv, and mediapipe installed:
```bash
pip install opencv-python mediapipe numpy
```

### 2. Download the Model File
The modern MediaPipe Tasks API requires a pre-trained model file. Download the hand landmarker task file and place it in the project root:
```bash
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## How to Run

### Real-Time Hand Tracking (Modular)
To run the clean hand tracking demonstration:
```bash
python "Hand Tracking ( clean ).py"
```

### Virtual Finger Pen
To run the virtual drawing application:
```bash
python "Finger_Pen.py"
```
* **Controls:** 
  - Raise your index finger to draw purple lines on the canvas.
  - Press the **Spacebar** key to exit the program.
