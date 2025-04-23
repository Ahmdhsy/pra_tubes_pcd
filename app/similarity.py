import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from deepface.detectors import FaceDetector

# =================== SETUP =================== #
input_shape = (224, 224, 3)
embedding_dim = 128
base_model_path = "models/base_model.keras"
embedding_csv_path = "face_embeddings.csv"

base_model = None
retina_face_detector = FaceDetector.build_model("retinaface")

# =================== DATA =================== #
def detect_and_crop_face(img):
    img_array = np.array(img)
    faces = FaceDetector.detect_face(retina_face_detector, "retinaface", img_array, align=False)

    if isinstance(faces, list) and len(faces) > 0:
        face, region, _ = faces[0]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        return img.crop((x, y, x + w, y + h))
    return img

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    return img

def load_image(path):
    img = Image.open(path).convert("RGB")
    face = detect_and_crop_face(img)
    return preprocess_image(face)

def load_dataset(meta_folder="meta"):
    data = {}
    for person in os.listdir(meta_folder):
        folder = os.path.join(meta_folder, person)
        if os.path.isdir(folder):
            data[person] = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    return data

def make_pairs(data_dict):
    persons = list(data_dict.keys())
    same_pairs = []
    diff_pairs = []

    for person in persons:
        imgs = data_dict[person]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                same_pairs.append((imgs[i], imgs[j], 1))

    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            img1 = np.random.choice(data_dict[persons[i]])
            img2 = np.random.choice(data_dict[persons[j]])
            diff_pairs.append((img1, img2, 0))

    pairs = same_pairs + diff_pairs
    np.random.shuffle(pairs)
    return pairs

# =================== MODEL =================== #
def create_base_network(input_shape):
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    x = Flatten()(base.output)
    x = Dense(embedding_dim, activation="relu")(x)
    return Model(inputs=base.input, outputs=x)

def euclidean_distance(vectors):
    x, y = vectors
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

def create_siamese_model(input_shape):
    base = create_base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    emb_a = base(input_a)
    emb_b = base(input_b)
    distance = Lambda(euclidean_distance)([emb_a, emb_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model, base

# =================== TRAINING =================== #
class SiameseDataGenerator(Sequence):
    def __init__(self, pairs, batch_size=16):
        self.pairs = pairs
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        batch_pairs = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        x1, x2, y = [], [], []
        for p1, p2, label in batch_pairs:
            x1.append(load_image(p1))
            x2.append(load_image(p2))
            y.append([label])
        return [np.array(x1), np.array(x2)], np.array(y)

def train_siamese():
    print("[INFO] Loading dataset...")
    dataset = load_dataset("meta")
    pairs = make_pairs(dataset)
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

    train_gen = SiameseDataGenerator(train_pairs)
    val_gen = SiameseDataGenerator(val_pairs)

    model, base = create_siamese_model(input_shape)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

    print("[INFO] Training model...")
    model.fit(train_gen, validation_data=val_gen, epochs=10)

    print(f"[INFO] Saving base model to {base_model_path}")
    base.save(base_model_path)

# =================== MODEL LOADING =================== #
def init_base_model():
    global base_model
    if base_model is None:
        if os.path.exists(base_model_path):
            base_model = load_model(base_model_path, compile=False)
            print("[INFO] Base model loaded.")
        else:
            raise FileNotFoundError(f"Base model not found at {base_model_path}. Run training first.")

# =================== SIMILARITY CHECK =================== #
def get_embedding(img):
    init_base_model()
    img = np.expand_dims(preprocess_image(detect_and_crop_face(img)), axis=0)
    return base_model.predict(img, verbose=0)[0]

def compute_similarity_score(embedding1, embedding2):
    dot = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return np.nan
    return dot / (norm1 * norm2)

def evaluate_model_from_csv(csv_path, threshold=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)
    y_true = []
    y_scores = []

    # ⏱️ Caching embeddings supaya tidak hitung ulang
    embedding_cache = {}

    def get_cached_embedding(path):
        if path not in embedding_cache:
            img = Image.open(path).convert("RGB")
            embedding_cache[path] = get_embedding(img)
        return embedding_cache[path]

    for idx, row in df.iterrows():
        p1 = row["photo1"].replace("\\", "/")
        p2 = row["photo2"].replace("\\", "/")
        label = row["label"]

        emb1 = get_cached_embedding(p1)
        emb2 = get_cached_embedding(p2)
        score = compute_similarity_score(emb1, emb2)

        y_true.append(label)
        y_scores.append(score)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # ROC + Metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    best_threshold = thresholds[eer_idx] if threshold is None else threshold

    y_pred = (y_scores >= best_threshold).astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    TAR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 0

    auc = roc_auc_score(y_true, y_scores)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f"TAR (True Acceptance Rate): {TAR:.4f}")
    print(f"FAR (False Acceptance Rate): {FAR:.4f}")
    print(f"FRR (False Rejection Rate): {FRR:.4f}")
    print(f"EER (Equal Error Rate): {eer:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")


# =================== MAIN =================== #
if __name__ == "__main__":
    if not os.path.exists(base_model_path):
        print("[INFO] Base model not found. Training model...")
        train_siamese()
    else:
        print("[INFO] Base model already exists. Skipping training.")
        print("[INFO] Evaluating model using CSV pairing...")
        evaluate_model_from_csv("pairs.csv")