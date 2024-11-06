import cv2
import mediapipe as mp
import tkinter as tk
import threading
from tkinter import Label
from PIL import Image, ImageTk

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("Detección de Manos en Tiempo Real")
root.geometry("800x600")

camera_label = Label(root)
camera_label.pack()

def actualizar_frame():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)
    
    camera_label.after(10, actualizar_frame)

def iniciar_video():
    actualizar_frame()

thread = threading.Thread(target=iniciar_video)
thread.daemon = True
thread.start()

root.mainloop()

cap.release()