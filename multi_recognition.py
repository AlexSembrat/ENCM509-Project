import cv2
import numpy as np
import os

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_training_data(root_dir):
    faces = []
    labels = []
    label_map = {}  # Maps numeric labels to person names
    current_label = 0

    # Each subdirectory in root_dir corresponds to a different person
    for person_name in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_map[current_label] = person_name

        # Process each image in the person's directory
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                image_path = os.path.join(person_dir, filename)
                # Load image: load PGM directly in grayscale, or convert others to grayscale
                if filename.lower().endswith('.pgm'):
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect face in the image
                detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in detected_faces:
                    face_roi = img[y:y+h, x:x+w]
                    # Resize and equalize to standardize the input
                    face_roi = cv2.resize(face_roi, (200, 200))
                    face_roi = cv2.equalizeHist(face_roi)
                    faces.append(face_roi)
                    labels.append(current_label)
                    break  # Use only the first detected face per image
        current_label += 1
    return faces, labels, label_map

# Directory containing subdirectories for each person
training_data_dir = "training_sets"  # For example, "training_data/PersonName/..."
faces, labels, label_map = get_training_data(training_data_dir)

if not faces:
    raise Exception("No faces found in the training data. Check your images and detection parameters.")

# Create and train the LBPH face recognizer (requires opencv-contrib-python)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Set a threshold for recognition (lower values mean a better match)
threshold = 50  # Adjust based on your data

# Start video capture (using DirectShow backend on Windows if necessary)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in detected_faces:
        face_roi = gray_frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))
        face_roi = cv2.equalizeHist(face_roi)
        
        try:
            predicted_label, confidence = recognizer.predict(face_roi)
        except Exception:
            predicted_label, confidence = -1, 1000  # Fallback for prediction errors
        
        # Decide if the face matches any known person based on the confidence
        if confidence < threshold:
            person_name = label_map.get(predicted_label, "Unknown")
            box_color = (0, 255, 0)  # Green for a recognized face
            text = f"{person_name} {confidence:.2f}"
        else:
            box_color = (255, 0, 0)  # Blue for an unknown face
            text = f"Unknown {confidence:.2f}"
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
