import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Cargar el modelo entrenado
model = tf.keras.models.load_model(f'{BASE_DIR}/hand_gesture_number_model.h5')

# Inicializar captura de video
cap = cv2.VideoCapture(0)

# Clase a número (etiquetas)
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Unknown']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen para mejor experiencia de usuario
    frame = cv2.flip(frame, 1)

    # Definir una región de interés (ROI) donde se espera la mano
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Preprocesar la ROI para hacer la predicción
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0  # Normalizar la imagen
    img = np.expand_dims(img, axis=0)

    # Realizar la predicción
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    label = labels[class_index]

    # Mostrar la predicción en la ventana
    cv2.putText(frame, f'Numero: {label}', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar el frame
    cv2.imshow('Reconocimiento de Numeros', frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
