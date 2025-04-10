import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np

# --- Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_haar(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []
    for (x, y, w, h) in faces:
        results.append({"box": [x, y, w, h]})
    return results

# --- MTCNN
detector = MTCNN()

def detect_faces_mtcnn(img_array):
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    return detector.detect_faces(img_array)
