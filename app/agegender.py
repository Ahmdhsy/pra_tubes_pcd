import cv2
import numpy as np

# Load age & gender models (tetap pakai OpenCV DNN)
age_proto = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_proto = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
face_proto = "models/opencv_face_detector.pbtxt"
face_model = "models/opencv_face_detector_uint8.pb"

face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Parameters
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Gunakan MTCNN untuk deteksi wajah
def detect_faces_dnn(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * \
                np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
    MODEL_MEAN_VALUES, swapRB=False)
    
    
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    return age, gender
