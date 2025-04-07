import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_training_data(root_dir):
    faces = []
    labels = []
    label_map = {}  # Maps numeric labels to person names
    current_label = 0

    for person_name in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_map[current_label] = person_name

        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                image_path = os.path.join(person_dir, filename)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))
                for (x, y, w, h) in detected_faces:
                    face_roi = img[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (200, 200))
                    face_roi = cv2.equalizeHist(face_roi)
                    faces.append(face_roi)
                    labels.append(current_label)
                    break  

        current_label += 1
    return faces, labels, label_map

# Directory containing training images
training_data_dir = "training_data" 
faces, labels, label_map = get_training_data(training_data_dir)

if not faces:
    raise Exception("No faces found in the training data. Check your images and detection parameters.")

# Create or load the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_file = "face_recognizer.yml"

if os.path.exists(model_file):
    recognizer.read(model_file)
    print("Loaded existing trained model.")
else:
    recognizer.train(faces, np.array(labels))
    recognizer.save(model_file)
    print("Trained and saved new model.")

# Recognition threshold
threshold = 50  

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fps_start = time.time()
frame_count = 0
fps = 0.0

# Prepare CSV logging
csv_filename = "results/recognition_log.csv"
run_number = 1

# Create and write header once
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Run #', 'Timestamp', 'Confidence', 'Label'])

# Begin main loop
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

        for (x, y, w, h) in detected_faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)

            try:
                predicted_label, confidence = recognizer.predict(face_roi)
            except Exception:
                predicted_label, confidence = -1, 1000  

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if confidence < threshold:
                person_name = label_map.get(predicted_label, "Unknown")
                box_color = (0, 255, 0)  
                text = f"{person_name} {confidence:.2f}"
            else:
                person_name = "Unknown"
                box_color = (255, 0, 0)  
                text = f"Unknown {confidence:.2f}"

            # Log to CSV
            writer.writerow([run_number, timestamp, round(confidence, 2), person_name])
            run_number += 1

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        frame_count += 1
        if frame_count >= 10:
            fps_end = time.time()
            fps = frame_count / (fps_end - fps_start)
            fps_start = time.time()
            frame_count = 0

        fps_text = f"FPS: {fps:.2f}" if fps > 0 else "FPS: Calculating..."
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
