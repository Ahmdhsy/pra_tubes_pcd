import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from PIL import Image

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model_resnet, self.model_vgg = load_models()

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        label, confidence, _ = ensemble_predict([self.model_resnet, self.model_vgg], image)
        text = f"{label.capitalize()} ({confidence*100:.2f}%)"
        cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image

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

def video_frame_callback(frame):
    # Convert frame to image
    image = frame.to_ndarray(format="bgr24")

    # Process image and predict
    model_resnet, model_vgg = load_models()
    label, confidence, all_conf = ensemble_predict([model_resnet, model_vgg], image)

    # Add prediction text on the frame
    text = f"{label.capitalize()} ({confidence*100:.2f}%)"
    cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def run():
    st.title("Klasifikasi Suku di Indonesia")
    st.write("Deteksi suku dari wajah dengan model CNN (ResNet50 + VGG16 Ensemble)")

    tab1, tab2 = st.tabs(["Live Kamera", "Upload Gambar"])

    # Tab untuk live kamera
        # Tab untuk kamera (ambil foto 1x)
    with tab1:
        st.write("Ambil foto dari kamera dan deteksi suku:")
        camera_image = st.camera_input("Ambil Foto Wajah")

        if camera_image is not None:
            image = Image.open(camera_image)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            st.image(image, channels="BGR", caption="Gambar dari Kamera")
            with st.spinner("Melakukan prediksi..."):
                model_resnet, model_vgg = load_models()
                label, confidence, all_conf = ensemble_predict([model_resnet, model_vgg], image)

            st.success(f"Hasil Prediksi: **{label.capitalize()}** (Confidence: {confidence*100:.2f}%)")
            st.subheader("Confidence untuk semua suku:")
            for suku, prob in zip(ethnicity_labels, all_conf):
                st.write(f"- {suku.capitalize()}: {prob*100:.2f}%")

    # Tab untuk upload gambar
        # Tab untuk upload gambar
    with tab2:
        uploaded = st.file_uploader("Upload gambar wajah...", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            st.image(image, channels="BGR", caption="Gambar yang diproses")
            with st.spinner("Melakukan prediksi..."):
                model_resnet, model_vgg = load_models()
                label, confidence, all_conf = ensemble_predict([model_resnet, model_vgg], image)

            st.success(f"Hasil Prediksi: **{label.capitalize()}** (Confidence: {confidence*100:.2f}%)")
            st.subheader("Confidence untuk semua suku:")
            for suku, prob in zip(ethnicity_labels, all_conf):
                st.write(f"- {suku.capitalize()}: {prob*100:.2f}%")

