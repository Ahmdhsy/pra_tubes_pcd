import cv2
from mtcnn import MTCNN
from deepface import DeepFace

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

# --- RetinaFace
def detect_faces_retinaface(img_array):
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    faces = DeepFace.extract_faces(img_array, 
                                 detector_backend='retinaface',
                                 enforce_detection=False,
                                 align=False)
    
    results = []
    for face in faces:
        region = face['facial_area']
        x = region['x']
        y = region['y']
        w = region['w']
        h = region['h']
        
        box = [int(x), int(y), int(w), int(h)]
        results.append({"box": box, "confidence": face.get('confidence', 0.9)})
    
    return results