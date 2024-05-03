import cv2
from fer import FER
import matplotlib.pyplot as plt

# Initialize the FER object
detector = FER()

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Convert frame to grayscale (FER expects grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and emotions in the frame
    results = detector.detect_emotions(frame)

    # Display the frame with bounding boxes and emotion labels
    for result in results:
        x, y, w, h = result["box"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        dom_emotion = max(result["emotions"], key=result["emotions"].get)
        dom_score = result["emotions"][dom_emotion]
        cv2.putText(frame, f'{dom_emotion}: {dom_score}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()