# 🚀 Advanced Computer Vision with OpenCV & MediaPipe

Welcome to the **Advanced Computer Vision** repository! This project contains a collection of modular, real-time Computer Vision pipelines implemented using **OpenCV** and the modern **Google MediaPipe Tasks API** in Python.

---

## 📂 Project Structure & Modules

The project is structured into four main visual tracking sub-modules:

```text
Advanced CV with OpenCV/
│
├── 🖐️ Hand Tracking/          # Hand tracking, landmark detection, and Virtual Pen
│   ├── Finger_Pen.py         # Draw on the screen using your index finger
│   ├── Hand Tracking.py      # Playground script for hand landmarking
│   ├── utils.py              # Modular HandDetector class & FPS helpers
│   └── readme.md             
│
├── 🏃 Pose Estimation/       # Human body pose tracking (33 landmarks)
│   ├── Pose Estimation.py    # Main script running pose landmarker on video
│   ├── utils.py              # Modular PoseDetector class
│   └── readme.md             
│
├── 👤 Face Detection/        # Fast bounding-box face detection
│   ├── Face_Detection.py     # Runs face detection on a video stream
│   ├── utils.py              # Modular FaceDetector with custom layouts
│   └── readme.md             
│
└── 🎭 Face Mesh/             # 468-point 3D facial landmark mesh
    ├── Face_Mesh.py          # Runs face mesh landmarking (tesselation/contours)
    ├── utils.py              # Modular FaceMesh class with drawing modes
    └── readme.md             
```

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository & Setup Environment
Ensure you have Python 3.10+ installed, then install the dependencies:
```bash
pip install opencv-python mediapipe numpy
```

### 2️⃣ Download the Pre-trained Task Models
The modern MediaPipe Tasks API requires `.task` and `.tflite` model files. Download them and place them in their respective module folders or project root:

| Module | Model Name | Download Link |
| :--- | :--- | :--- |
| **🖐️ Hand Tracking** | `hand_landmarker.task` | [📥 Download](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) |
| **🏃 Pose Estimation** | `pose_landmarker_full.task` | [📥 Download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task) |
| **👤 Face Detection** | `face_detection_full_range.tflite` | [📥 Download](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite) |
| **🎭 Face Mesh** | `face_landmarker.task` | [📥 Download](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) |

*For more details on models, visit the [MediaPipe Solutions Developer Site](https://developers.google.com/edge/mediapipe/solutions/vision).*

---

## 💻 How to Run the Modules

Navigate into any project directory and run the main Python scripts:

### 🖐️ Hand Tracking & Virtual Pen
```bash
cd "Hand Tracking"
python "Hand Tracking ( clean ).py"  # Standard Hand Tracking
python "Finger_Pen.py"              # Draw with your finger!
```

### 🏃 Pose Estimation
```bash
cd "Pose Estimation"
python "Pose Estimation.py"
```

### 👤 Face Detection
```bash
cd "Face_Detection"
python "Face_Detection.py"
```

### 🎭 Face Mesh
```bash
cd "Face_Mesh"
python "Face_Mesh.py"
```

---

## ⌨️ Controls
* **Spacebar (` `)**: Press Spacebar to safely exit any running camera or video pipeline.
