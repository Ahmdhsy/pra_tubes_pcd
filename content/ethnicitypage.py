import streamlit as st
import numpy as np
import cv2
from PIL import Image
from app.ethnicity import preprocess_face_for_prediction, ethnicity_labels, load_models, ensemble_predict
from streamlit_option_menu import option_menu

def run():
    st.title("Ethnicity Recognition")
    st.write("Klasifikasi etnisitas suku berdasarkan menggunakan model deep learning.")

    selected = option_menu(
            menu_title="Choose Recognition Method",
            options=["Upload Image", "Take Photo"],
            icons=["file-earmark-arrow-up-fill", "camera-fill"],
            menu_icon="list-task",
            default_index=0,
            orientation="horizontal",
            key="upload_menu"
        )

    if selected == "Upload Image":
        uploaded = st.file_uploader("Upload File", type=['jpg', 'jpeg', 'png'])

        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            col1, spacer, col2 = st.columns([1, 0.05, 1])

            with col1:
                st.image(image, channels="BGR", use_column_width=True)
            
            with col2:
                with st.spinner("Melakukan deteksi wajah dan prediksi..."):
                    face = preprocess_face_for_prediction(image)
                    if face is None:
                        st.warning("Tidak ditemukan wajah pada gambar. Pastikan kualitas foto baik dan wajah terlihat jelas.")
                    else:
                        model_resnet, model_vgg = load_models()
                        label, confidence, all_conf = ensemble_predict([model_resnet, model_vgg], face)

                        st.success(f"Hasil Prediksi: **{label.capitalize()}** (Confidence: {confidence*100:.2f}%)")
                        st.subheader("Confidence untuk semua suku:")
                        for suku, prob in zip(ethnicity_labels, all_conf):
                            st.write(f"- {suku.capitalize()}: {prob*100:.2f}%")
    
    elif selected == "Take Photo":
        camera_image = st.camera_input("Take a Picture")

        if camera_image is not None:
            image = Image.open(camera_image)
            image = np.array(image)
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            cameracols1, spacer, cameracols2 = st.columns([1, 0.05, 1])

            with cameracols1:
                st.image(image, channels="BGR", use_column_width=True)

            with cameracols2: 
                with st.spinner("Melakukan deteksi wajah dan prediksi..."):
                    face = preprocess_face_for_prediction(image)
                    if face is None:
                        st.warning("Tidak ditemukan wajah pada gambar. Pastikan kualitas foto baik dan wajah terlihat jelas.")
                    else:
                        model_resnet, model_vgg = load_models()
                        label, confidence, all_conf = ensemble_predict([model_resnet, model_vgg], face)

                        st.success(f"Hasil Prediksi: **{label.capitalize()}** (Confidence: {confidence*100:.2f}%)")
                        st.subheader("Confidence untuk semua suku:")
                        for suku, prob in zip(ethnicity_labels, all_conf):
                            st.write(f"- {suku.capitalize()}: {prob*100:.2f}%")