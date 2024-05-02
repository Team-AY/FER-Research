import cv2
import face_recognition

# Initialize the webcam
cap = cv2.VideoCapture(0)

# with mp4
#cap = cv2.VideoCapture('WhatsApp Video 2024-04-25 at 17.58.22.mp4')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame, model='hog')

    # Find all face landmarks in the frame
    face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)

    # Draw the face landmarks on the frame
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow('Face Landmarks - face_recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()