import cv2
from deepface import DeepFace

img_path='faces_db\woman-6328478_1280.webp';
frame = cv2.imread(img_path)

faces = DeepFace.extract_faces(img_path=img_path)

for face in faces:
    x = face['facial_area']['x']
    y = face['facial_area']['y']
    w = face['facial_area']['w']
    h = face['facial_area']['h']
    # Extract the face ROI (Region of Interest)
    face_roi = frame[y:y + h, x:x + w]

    
    # Perform emotion analysis on the face ROI
    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)    

    # Determine the dominant emotion
    emotion = result[0]['dominant_emotion']    

    # Draw rectangle around face and label with predicted emotion
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

while(True):
    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()