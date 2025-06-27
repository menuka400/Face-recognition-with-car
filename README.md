# 🚗 Face Recognition with Car

A robust Python project for real-time face recognition, detection, and management, designed for security and automation applications. This project leverages state-of-the-art deep learning models (YOLOv11m, InsightFace) to detect and recognize faces, manage a face database, and handle unknown faces efficiently.

---

## ✨ Features
- Real-time face detection and recognition
- Database management for known faces
- Handles unknown faces with logging
- Uses YOLOv11m and InsightFace models
- Modular and extensible codebase

---

## 📁 Folder Structure
```
Face-recognition with car/
├── database_manager.py         # Database operations for faces
├── face_detection.py          # Face detection logic (YOLO, InsightFace)
├── face_recognition.py        # Face recognition and matching
├── main.py                    # Main entry point
├── face_embeddings.json       # Stored face embeddings
├── yolov11m-face.pt           # YOLOv11m model weights
├── insightface_models/        # InsightFace ONNX models
├── Face_imagers/              # (Your face image dataset)
├── unknown_faces/             # Storage for unknown face images
├── move/                      # Utilities and setup scripts
│   ├── database_utils.py
│   ├── inspect_json.py
│   └── setup_insightface.py
└── ...
```

---

## 🚀 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/face-recognition-with-car.git
   cd "Face-recognition with car"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Create a `requirements.txt` if not present. Typical dependencies: `opencv-python`, `insightface`, `torch`, `numpy`, etc.)*

3. **Download models:**
   - Place YOLOv11m weights (`yolov11m-face.pt`) and InsightFace ONNX models in the provided directories if not already present.

4. **Run the main application:**
   ```bash
   python main.py
   ```

---

## 🛠️ Usage Example

- To start face recognition and detection:
  ```bash
  python main.py
  ```
- To manage the face database, use `database_manager.py`:
  ```bash
  python database_manager.py
  ```

---

## 📦 Dependencies
- Python 3.7+
- OpenCV (`opencv-python`)
- InsightFace
- PyTorch
- NumPy
- (See `requirements.txt` for full list)

---

## 🙏 Credits
- [InsightFace](https://github.com/deepinsight/insightface) for face recognition models
- [YOLO](https://github.com/ultralytics/yolov5) for object detection
- Your contributions and feedback are welcome!

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. 
