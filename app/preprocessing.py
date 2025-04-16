import os
import cv2
import numpy as np
from PIL import Image
from detection import detect_faces_mtcnn
from utils import resize_image
import albumentations as A
from collections import defaultdict 

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
IMAGE_SIZE = (224, 224)

augmentations = {
    "rotasi": A.Rotate(limit=15, p=1.0),
    "flip": A.HorizontalFlip(p=1.0),
    "contrast": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    "noise": A.GaussNoise(p=1.0),
}

def crop_face(img, box):
    x, y, w, h = box
    return img[y:y+h, x:x+w]

def process_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detect_faces_mtcnn(img)

    if faces:
        face = crop_face(img, faces[0]['box'])  # Ambil wajah pertama
        face = Image.fromarray(face)
        face = resize_image(face, IMAGE_SIZE)
        return np.array(face)
    return None

def save_image(img_array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def preprocess_dataset():
    all_data = []

    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)
                filename = os.path.basename(file)
                parts = filename.split('_')
                if len(parts) < 3:
                    continue

                nama = parts[0]
                suku = parts[1].lower()

                face_img = process_image(file_path)
                if face_img is not None:
                    all_data.append((face_img, nama, suku))

    name_counters = defaultdict(int)
    
    print("Preprocessing selesai!")
    print("Menyimpan hasil preprocessing ke folder train...")
    print(f"Jumlah total data: {len(all_data)}")

    for img, nama, suku in all_data:
        key = f"{nama}_{suku}"
        name_counters[key] += 1
        idx = name_counters[key]

        base_path = f"{OUTPUT_DIR}/train/{suku}"

        # Simpan gambar original
        original_filename = f"{nama}_{suku}_original_image{idx}.jpg"
        save_image(img, os.path.join(base_path, original_filename))

        # Simpan hasil augmentasi satu per satu
        for aug_name, aug in augmentations.items():
            aug_img = aug(image=img)["image"]
            aug_filename = f"{nama}_{suku}_{aug_name}_image{idx}.jpg"
            save_image(aug_img, os.path.join(base_path, aug_filename))

if __name__ == "__main__":
    preprocess_dataset()
