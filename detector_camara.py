import cv2
import numpy as np
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('modelo1.h5')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (48, 48))
        resized_face = np.expand_dims(resized_face, axis=-1)
        resized_face = np.expand_dims(resized_face, axis=0)
        
        prediction = model.predict(resized_face)
        emotion = "Emoción: " + str(prediction)  # Ejemplo de cómo obtener la emoción predicha
        predicted_class =   prediction.argmax(axis=1)[0]
    # Obtener el nombre de la clase
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        predicted_emotion = class_names[predicted_class]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()