import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from deepface.detectors import FaceDetector

# =================== SETUP =================== #
input_shape = (224, 224, 3)
embedding_dim = 128
base_model_path = "models/base_model.keras"

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

# =================== MAIN =================== #
if __name__ == "__main__":
    if not os.path.exists(base_model_path):
        print("[INFO] Base model not found. Training model...")
        train_siamese()
    else:
        print("[INFO] Base model already exists. Skipping training.")
