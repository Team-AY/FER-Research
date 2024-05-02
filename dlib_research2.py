import cv2
import dlib

# Load the pre-trained face detector and facial landmark predictor from dlib
detector = dlib.get_frontal_face_detector()

# from https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("models/dlib/shape_predictor_68_face_landmarks.dat")  # Provide the path to your shape predictor model

# Function to detect faces and facial landmarks
def detect_faces_and_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    faces_landmarks = []

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        faces_landmarks.append(landmarks_points)

    return faces, faces_landmarks

# Example usage:
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces, faces_landmarks = detect_faces_and_landmarks(frame)

    for face, landmarks in zip(faces, faces_landmarks):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for landmark in landmarks:
            cv2.circle(frame, landmark, 2, (0, 0, 255), -1)

    cv2.imshow('Face Detection with Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
