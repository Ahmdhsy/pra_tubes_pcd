import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Import dari file preprocessing kamu
from app.detection import detect_faces_mtcnn
from app.utils import resize_image  # Opsional, jika kamu pakai resize custom

# Label suku Indonesia
ethnicity_labels = ['batak', 'jawa', 'minang', 'sunda']
num_ethnicities = len(ethnicity_labels)

@st.experimental_singleton
def load_models():
    # ResNet50
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x1 = GlobalAveragePooling2D()(resnet_base.output)
    x1 = Dense(1024, activation='relu')(x1)
    pred1 = Dense(num_ethnicities, activation='softmax')(x1)
    model_resnet = Model(inputs=resnet_base.input, outputs=pred1)
    for layer in resnet_base.layers:
        layer.trainable = False

    # VGG16
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x2 = GlobalAveragePooling2D()(vgg_base.output)
    x2 = Dense(1024, activation='relu')(x2)
    pred2 = Dense(num_ethnicities, activation='softmax')(x2)
    model_vgg = Model(inputs=vgg_base.input, outputs=pred2)
    for layer in vgg_base.layers:
        layer.trainable = False

    return model_resnet, model_vgg

def preprocess_image(image, model_type='resnet'):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    if model_type == 'resnet':
        image = preprocess_resnet(image)
    else:
        image = preprocess_vgg(image)
    image = np.expand_dims(image, axis=0)
    return image

def ensemble_predict(models, image):
    resnet_input = preprocess_image(image, 'resnet')
    vgg_input = preprocess_image(image, 'vgg')

    pred1 = models[0].predict(resnet_input)[0]
    pred2 = models[1].predict(vgg_input)[0]
    final_pred = (pred1 + pred2) / 2

    top_idx = np.argmax(final_pred)
    return ethnicity_labels[top_idx], final_pred[top_idx], final_pred

def preprocess_face_for_prediction(img):
    faces = detect_faces_mtcnn(img)
    if not faces:
        return None

    face_box = faces[0]['box']
    x, y, w, h = face_box
    face = img[y:y+h, x:x+w]

    # Jika kamu ingin pakai resize custom dari utils:
    # face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    # resized = resize_image(face_pil, (224, 224))
    # return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)

    # Atau langsung resize pakai OpenCV
    face = cv2.resize(face, (224, 224))
    return face

def run():
    st.title("Klasifikasi Suku di Indonesia")
    st.write("Deteksi suku dari wajah dengan model CNN (ResNet50 + VGG16 Ensemble)")

    tab1, tab2 = st.tabs(["Live Kamera", "Upload Gambar"])

    # Tab Kamera
    with tab1:
        st.write("Ambil foto dari kamera dan deteksi suku:")
        camera_image = st.camera_input("Ambil Foto Wajah")

        if camera_image is not None:
            image = Image.open(camera_image)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            st.image(image, channels="BGR", caption="Gambar dari Kamera")

            with st.spinner("Melakukan deteksi wajah dan prediksi..."):
                face = preprocess_face_for_prediction(image)
                if face is None:
                    st.warning("Wajah tidak terdeteksi. Pastikan wajah terlihat jelas.")
                else:
                    model_resnet, model_vgg = load_models()
                    label, confidence, all_conf = ensemble_predict([model_resnet, model_vgg], face)

                    st.success(f"Hasil Prediksi: **{label.capitalize()}** (Confidence: {confidence*100:.2f}%)")
                    st.subheader("Confidence untuk semua suku:")
                    for suku, prob in zip(ethnicity_labels, all_conf):
                        st.write(f"- {suku.capitalize()}: {prob*100:.2f}%")

    # Tab Upload Gambar
    with tab2:
        uploaded = st.file_uploader("Upload gambar wajah...", type=['jpg', 'jpeg', 'png'])

        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            st.image(image, channels="BGR", caption="Gambar yang diproses")

            with st.spinner("Melakukan deteksi wajah dan prediksi..."):
                face = preprocess_face_for_prediction(image)
                if face is None:
                    st.warning("Wajah tidak terdeteksi. Coba upload gambar dengan wajah yang lebih jelas.")
                else:
                    model_resnet, model_vgg = load_models()
                    label, confidence, all_conf = ensemble_predict([model_resnet, model_vgg], face)

                    st.success(f"Hasil Prediksi: **{label.capitalize()}** (Confidence: {confidence*100:.2f}%)")
                    st.subheader("Confidence untuk semua suku:")
                    for suku, prob in zip(ethnicity_labels, all_conf):
                        st.write(f"- {suku.capitalize()}: {prob*100:.2f}%")

if __name__ == '__main__':
    run()
