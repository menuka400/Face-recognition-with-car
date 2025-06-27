import threading
import time
import os
from face_detection import FaceDetector
from face_recognition import FaceRecognizer

def main():
    # Configuration
    YOLO_MODEL_PATH = "yolov11m-face.pt"
    SAVE_FOLDER = r"C:\Users\menuk\Desktop\FYP\New folder\Face_imagers"
    INSIGHTFACE_MODEL_PATH = r"C:\Users\menuk\Desktop\FYP\New folder\insightface_models\models\buffalo_l.zip"
    JSON_DATABASE_PATH = "face_embeddings.json"
    
    print("🚀 Starting Face Recognition System...")
    
    if not os.path.exists(JSON_DATABASE_PATH):
        print(f"❌ JSON database not found: {JSON_DATABASE_PATH}")
        return
    
    try:
        # Initialize components
        face_detector = FaceDetector(YOLO_MODEL_PATH, SAVE_FOLDER)
        
        # Get queues
        face_queue = face_detector.get_face_queue()
        results_queue = face_detector.get_recognition_results_queue()
        
        # Initialize face recognizer
        face_recognizer = FaceRecognizer(INSIGHTFACE_MODEL_PATH, JSON_DATABASE_PATH, results_queue)
        
        # Create threads
        detection_thread = threading.Thread(target=face_detector.detect_faces, name="FaceDetectionThread")
        recognition_thread = threading.Thread(target=face_recognizer.recognize_faces, args=(face_queue,), name="FaceRecognitionThread")
        
        # Start threads
        detection_thread.start()
        recognition_thread.start()
        
        print("✅ System running - Press 'q' in video window to quit")
        
        # Keep main thread alive
        while detection_thread.is_alive() and recognition_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Stop components
        if 'face_detector' in locals():
            face_detector.stop()
        if 'face_recognizer' in locals():
            face_recognizer.stop()
        
        # Wait for threads to finish
        if 'detection_thread' in locals() and detection_thread.is_alive():
            detection_thread.join(timeout=5)
        if 'recognition_thread' in locals() and recognition_thread.is_alive():
            recognition_thread.join(timeout=5)
        
        print("✅ System shutdown complete.")

if __name__ == "__main__":
    main()