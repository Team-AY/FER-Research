import cv2
from mtcnn import MTCNN
import numpy as np
from deepface import DeepFace as df

# Initialize MTCNN detector
detector = MTCNN()

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame from webcam")
        break

    # Convert frame to RGB format (MTCNN expects RGB)
    rgb_frame = frame[:, :, ::-1]  # BGR -> RGB

    # Detect faces using MTCNN
    faces = detector.detect_faces(rgb_frame)

    # Process each detected face
    for face in faces:
        # Extract bounding box coordinates
        x, y, w, h = face['box']

        # Draw bounding box on the frame (optional)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face ROI (region of interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Emotion prediction using DeepFace
        emotions = df.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        
        
        # Access the predicted emotion (modify based on deepface output format)
        emotion = emotions[0]["dominant_emotion"]  # Assuming "dominant_emotion" is the key
        confidence = emotions[0]['emotion'][emotion]#emotions[0]["dominant_emotion_probability"] * 100  # Convert to percentage

        # Draw emotion text on the frame
        cv2.putText(frame, f"{emotion} ({confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Extract facial landmarks (optional)
        landmarks = face['keypoints']

        # Draw facial landmarks (optional)
        for landmark in landmarks.values():
            cv2.circle(frame, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)

    # Display the resulting frame
    cv2.imshow('Facial Recognition with Emotion Classification', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
