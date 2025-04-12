import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from detection import detect_faces_mtcnn
from utils import resize_image
import albumentations as A
from collections import defaultdict

# Konfigurasi
RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
IMAGE_SIZE = (224, 224)

# Augmentasi
augment = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.3),
])

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

    # Split data
    train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Counter agar file tidak tertimpa
    name_counters = defaultdict(int)
    
    
    print("Preprocessing selesai!")
    print("Menyimpan hasil preprocessing...")
    print(f"Jumlah data train: {len(train_data)}")
    print(f"Jumlah data val: {len(val_data)}")
    print(f"Jumlah data test: {len(test_data)}")
    print("Menyimpan gambar ke folder processed...")
    print("Hasil gambar berhasil disimpan di folder data/processed")
    
    # Simpan hasil split
    for split_name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for img, nama, suku in split:
            key = f"{nama}_{suku}"
            name_counters[key] += 1
            idx = name_counters[key]

            base_path = f"{OUTPUT_DIR}/{split_name}/{suku}"

            # Original
            original_filename = f"{nama}_{suku}_original_image{idx}.jpg"
            save_image(img, os.path.join(base_path, original_filename))

            # Augmented
            aug_img = augment(image=img)["image"]
            aug_filename = f"{nama}_{suku}_augmented_image{idx}.jpg"
            save_image(aug_img, os.path.join(base_path, aug_filename))

if __name__ == "__main__":
    preprocess_dataset()