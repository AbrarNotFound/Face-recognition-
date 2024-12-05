import gradio as gr 
from PIL import Image
import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def face_detection(img):
    img = img.convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
   
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        message = "No faces found."
        processed_img = img
    else:
        eyes_detected = False
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)

            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img_cv[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) > 0:
                eyes_detected = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        if eyes_detected:
            message = "Faces and eyes detected."
        else:
            message = "Faces detected, but no eyes found."

        processed_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    return processed_img, message

iface = gr.Interface(
    fn=face_detection,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Processed Image with Face Detection"),
        gr.Textbox(label="Detection Message")
    ],
    title="Face Detection with Python",
    description="Upload an image to detect faces and eyes. See the processed image and get a detection message."
)

iface.launch()