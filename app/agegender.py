import cv2
import numpy as np
from deepface import DeepFace
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

def detect_faces_mtcnn(img):
    """
    Mendeteksi wajah menggunakan MTCNN.
    Mengembalikan daftar koordinat wajah (x, y, w, h).
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.dtype == np.uint8 else img
    else:
        img_rgb = img

    faces = detector.detect_faces(img_rgb)
    face_coords = []

    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']

        if confidence > 0.9:
            face_coords.append((x, y, width, height))

    return face_coords

def predict_age_gender(face_img):
    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        results = DeepFace.analyze(
            face_rgb,
            actions=['age', 'gender'],
            enforce_detection=False,
            silent=True
        )

        print("DeepFace result:", results)

        age = -1
        dominant_gender = "Unknown"
        gender_prob = {}

        if isinstance(results, dict):
            age = int(results.get('age', -1))
            dominant_gender = results.get('dominant_gender', 'Unknown')
            gender_prob = results.get('gender', {})
        elif isinstance(results, list) and len(results) > 0:
            age = int(results[0].get('age', -1))
            dominant_gender = results[0].get('dominant_gender', 'Unknown')
            gender_prob = results[0].get('gender', {})

        return age, dominant_gender, gender_prob

    except Exception as e:
        print(f"Error in prediction: {e}")
        return -1, "Unknown", {}
