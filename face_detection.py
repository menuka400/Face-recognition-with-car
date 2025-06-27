import cv2
import numpy as np
import os
import threading
import time
from ultralytics import YOLO
from queue import Queue
import pickle

class FaceDetector:
    def __init__(self, model_path, save_folder):
        self.model = YOLO(model_path)
        self.save_folder = save_folder
        self.unknown_folder = "unknown_faces"
        self.face_queue = Queue()
        self.recognition_results = Queue()
        self.cleanup_queue = Queue()
        self.face_arrays = []
        self.running = False
        self.face_counter = 0
        self.saved_faces = {}
        self.saved_unknown_faces = {}
        self.current_recognitions = {}
        
        # Create save folders if they don't exist
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(self.unknown_folder, exist_ok=True)
        
    def detect_faces(self):
        """Main face detection loop with smart cleanup (silent mode)"""
        self.running = True
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return
            
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=self.cleanup_worker, daemon=True)
        cleanup_thread.start()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Check for new recognition results and cleanup requests
            self.update_recognition_results()
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Draw face boxes and names
            display_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        
                        # Only process if confidence is high enough
                        if conf > 0.6:
                            # Calculate face dimensions
                            face_width = x2 - x1
                            face_height = y2 - y1
                            
                            # Add generous padding (50% of face size)
                            padding_x = int(face_width * 0.5)
                            padding_y = int(face_height * 0.5)
                            
                            # Apply padding with bounds checking
                            x1_padded = max(0, x1 - padding_x)
                            y1_padded = max(0, y1 - padding_y)
                            x2_padded = min(frame.shape[1], x2 + padding_x)
                            y2_padded = min(frame.shape[0], y2 + padding_y)
                            
                            # Extract face region with padding
                            face_img = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                            
                            # Check if face is large enough
                            if (face_img.size > 0 and 
                                face_img.shape[0] > 100 and 
                                face_img.shape[1] > 100):
                                
                                # Resize face to a standard size for better recognition
                                face_resized = cv2.resize(face_img, (224, 224))
                                
                                # Save original and resized face
                                face_id = self.save_face(face_img, face_resized)
                                
                                # Add resized face to queue for recognition
                                face_data = {
                                    'image': face_resized.copy(),
                                    'face_id': face_id,
                                    'bbox': (x1, y1, x2, y2),
                                    'timestamp': time.time()
                                }
                                self.face_queue.put(face_data)
                                
                                # Draw face rectangle
                                color = (0, 255, 0)  # Green for detected face
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Check if we have recognition result for this area
                                person_name, confidence = self.get_recognition_for_area(x1, y1, x2, y2)
                                
                                if person_name:
                                    # Display person name and confidence
                                    label = f"{person_name}"
                                    confidence_label = f"{confidence:.1f}%" if confidence > 0 else ""
                                    
                                    # Calculate text size for background
                                    (text_width, text_height), baseline = cv2.getTextSize(
                                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                                    (conf_width, conf_height), _ = cv2.getTextSize(
                                        confidence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                    
                                    # Draw background rectangles for text
                                    bg_color = (0, 255, 0) if person_name != "UNKNOWN" else (0, 0, 255)
                                    cv2.rectangle(display_frame, 
                                                (x1, y1 - text_height - 35), 
                                                (x1 + max(text_width, conf_width) + 10, y1), 
                                                bg_color, -1)
                                    
                                    # Draw person name
                                    cv2.putText(display_frame, label, 
                                              (x1 + 5, y1 - 20), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                                    
                                    # Draw confidence
                                    if confidence_label:
                                        cv2.putText(display_frame, confidence_label, 
                                                  (x1 + 5, y1 - 5), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add minimal system info
            queue_text = f"Queue: {self.face_queue.qsize()}"
            cv2.putText(display_frame, queue_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Count images in both folders
            known_count = len([f for f in os.listdir(self.save_folder) if f.endswith('.jpg')])
            unknown_count = len([f for f in os.listdir(self.unknown_folder) if f.endswith('.jpg')]) if os.path.exists(self.unknown_folder) else 0
            
            folder_text = f"Known: {known_count} | Unknown: {unknown_count}"
            cv2.putText(display_frame, folder_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition System', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def cleanup_worker(self):
        """Worker thread to handle smart file cleanup (silent mode)"""
        while self.running:
            try:
                cleanup_request = self.cleanup_queue.get(timeout=1)
                face_id = cleanup_request['face_id']
                person_name = cleanup_request.get('person_name', 'UNKNOWN')
                
                if person_name != "UNKNOWN":
                    self.cleanup_known_faces(face_id, person_name)
                else:
                    self.cleanup_unknown_faces(face_id)
                        
            except:
                continue
                
        if self.running:
            self.periodic_cleanup()
    
    def cleanup_known_faces(self, face_id, person_name):
        """Smart cleanup for known/recognized faces (silent)"""
        current_images = [f for f in os.listdir(self.save_folder) if f.endswith('.jpg')]
        image_count = len(current_images)
        
        if image_count > 10:
            if face_id in self.saved_faces:
                files_to_delete = self.saved_faces[face_id]
                self.delete_face_files(files_to_delete, face_id, "known")
    
    def cleanup_unknown_faces(self, face_id):
        """Smart cleanup for unknown faces (silent)"""
        if not os.path.exists(self.unknown_folder):
            return
            
        current_unknown = [f for f in os.listdir(self.unknown_folder) if f.endswith('.jpg')]
        unknown_count = len(current_unknown)
        
        if unknown_count > 10:
            if face_id in self.saved_unknown_faces:
                files_to_delete = self.saved_unknown_faces[face_id]
                self.delete_face_files(files_to_delete, face_id, "unknown")
    
    def delete_face_files(self, file_paths, face_id, face_type):
        """Delete face image files (silent)"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        
        # Remove from tracking
        if face_type == "known" and face_id in self.saved_faces:
            del self.saved_faces[face_id]
        elif face_type == "unknown" and face_id in self.saved_unknown_faces:
            del self.saved_unknown_faces[face_id]
    
    def update_recognition_results(self):
        """Update recognition results and handle cleanup (silent)"""
        while not self.recognition_results.empty():
            try:
                result = self.recognition_results.get_nowait()
                face_id = result['face_id']
                person_name = result['person_name']
                confidence = result['confidence']
                timestamp = result['timestamp']
                bbox = result['bbox']
                
                # Store recognition result
                self.current_recognitions[face_id] = {
                    'person_name': person_name,
                    'confidence': confidence,
                    'timestamp': timestamp,
                    'bbox': bbox
                }
                
                # Schedule cleanup based on recognition result
                cleanup_request = {
                    'face_id': face_id,
                    'person_name': person_name
                }
                self.cleanup_queue.put(cleanup_request)
                
                # Clean old recognition results (older than 3 seconds)
                current_time = time.time()
                self.current_recognitions = {
                    k: v for k, v in self.current_recognitions.items()
                    if current_time - v['timestamp'] < 3.0
                }
                
            except:
                break
    
    def get_recognition_for_area(self, x1, y1, x2, y2):
        """Get recognition result for a face area"""
        best_match = None
        best_overlap = 0
        
        for face_id, result in self.current_recognitions.items():
            stored_bbox = result['bbox']
            if stored_bbox:
                overlap = self.calculate_bbox_overlap((x1, y1, x2, y2), stored_bbox)
                if overlap > best_overlap and overlap > 0.3:
                    best_overlap = overlap
                    best_match = result
        
        if best_match:
            return best_match['person_name'], best_match['confidence']
        return None, None
    
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
        
    def save_face(self, face_img, face_resized):
        """Save detected face to folder (silent)"""
        self.face_counter += 1
        face_id = f"face_{self.face_counter}_{int(time.time())}"
        
        # Save original face image
        filename = f"{face_id}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        cv2.imwrite(filepath, face_img)
        
        # Save resized face image
        filename_resized = f"{face_id}_resized.jpg"
        filepath_resized = os.path.join(self.save_folder, filename_resized)
        cv2.imwrite(filepath_resized, face_resized)
        
        # Track saved files
        self.saved_faces[face_id] = [filepath, filepath_resized]
        
        # Store as numpy array (keep limited amount)
        face_array = {
            'image': face_img,
            'image_resized': face_resized,
            'filename': filename,
            'face_id': face_id,
            'timestamp': time.time()
        }
        self.face_arrays.append(face_array)
        
        # Keep only last 10 numpy arrays in memory
        if len(self.face_arrays) > 10:
            self.face_arrays = self.face_arrays[-10:]
        
        # Save numpy arrays less frequently
        if len(self.face_arrays) % 10 == 0:
            self.save_arrays_to_file()
            
        return face_id
    
    def save_arrays_to_file(self):
        """Save face arrays to pickle file (silent)"""
        arrays_file = os.path.join(self.save_folder, "face_arrays.pkl")
        with open(arrays_file, 'wb') as f:
            pickle.dump(self.face_arrays, f)
    
    def get_face_queue(self):
        return self.face_queue
    
    def get_recognition_results_queue(self):
        return self.recognition_results
    
    def periodic_cleanup(self):
        """Periodic cleanup (silent)"""
        try:
            current_time = time.time()
            self.cleanup_old_files_in_folder(self.save_folder, "known", current_time)
            if os.path.exists(self.unknown_folder):
                self.cleanup_old_files_in_folder(self.unknown_folder, "unknown", current_time)
        except:
            pass
    
    def cleanup_old_files_in_folder(self, folder_path, folder_type, current_time):
        """Cleanup old files in folder (silent)"""
        try:
            files_in_folder = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            
            files_with_time = []
            for filename in files_in_folder:
                filepath = os.path.join(folder_path, filename)
                try:
                    file_time = os.path.getctime(filepath)
                    files_with_time.append((filename, filepath, file_time))
                except:
                    continue
            
            files_with_time.sort(key=lambda x: x[2])
            
            if len(files_with_time) > 10:
                files_to_delete = files_with_time[:-10]
                
                for filename, filepath, file_time in files_to_delete:
                    file_age = current_time - file_time
                    if file_age > 600:  # 10 minutes
                        try:
                            os.remove(filepath)
                        except:
                            pass
        except:
            pass
    
    def stop(self):
        """Stop face detection"""
        self.running = False
        self.save_arrays_to_file()
        self.periodic_cleanup()