import streamlit as st
import cv2
import numpy as np
from PIL import Image
from app.utils import load_image, convert_image_to_array
from app.agegender import detect_faces_dnn, predict_age_gender
from streamlit_option_menu import option_menu

def run():
    st.title("Deteksi Umur dan Gender")
    st.write("Pilih metode input untuk mendeteksi umur dan gender dari wajah.")

    tab1, tab2 = st.tabs(["Input Gambar", "Kamera Live"])
    
    with tab1:

        selected = option_menu(
            menu_title="Pilih Sumber Gambar",
            options=["Upload Gambar", "Gunakan Kamera"],
            icons=["file-earmark-arrow-up", "camera"],
            menu_icon="image",
            default_index=0,
            orientation="horizontal",
        )

        image = None

        if selected == "Upload Gambar":
            uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = load_image(uploaded_file)
                img_array = convert_image_to_array(image)
                faces = detect_faces_dnn(img_array)

                if not faces:
                    st.warning("Tidak ada wajah terdeteksi.")
                else:
                    for face in faces:
                        x, y, w, h = face
                        face_img = img_array[y:y+h, x:x+w]
                        age, gender = predict_age_gender(face_img)
                        label = f"{gender}, {age}"

                        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img_array, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    st.image(img_array, caption="Hasil Deteksi")
        
        elif selected == "Gunakan Kamera":
            st.title("Deteksi Umur dan Gender dari Kamera")

            camera_image = st.camera_input("Pengambilan Gambar")

            if camera_image:
                image = load_image(camera_image)
                img_array = convert_image_to_array(image)

                faces = detect_faces_dnn(img_array)

                if not faces:
                    st.warning("Tidak ada wajah terdeteksi.")
                else:
                    for face in faces:
                        x, y, w, h = face
                        face_img = img_array[y:y+h, x:x+w]
                        age, gender = predict_age_gender(face_img)
                        label = f"{gender}, {age}"
                        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img_array, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    st.image(img_array, caption="Hasil Deteksi")

    with tab2:
        st.title("Deteksi Umur dan Gender")
        st.write("Buka kamera dan deteksi umur serta gender secara langsung.")

        run_camera = st.button("Mulai Kamera")
        stop_camera = st.button("Berhenti Kamera")

        if run_camera:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            stframe = st.empty()
            camera_running = True

            try:
                while camera_running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Gagal membaca frame dari kamera.")
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = detect_faces_dnn(rgb_frame)

                    for face in faces:
                        x, y, w, h = face
                        face_img = rgb_frame[y:y+h, x:x+w]
                        age, gender = predict_age_gender(face_img)
                        label = f"{gender}, {age}"

                        cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(rgb_frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    resized_frame = cv2.resize(rgb_frame, (800, 600))
                    stframe.image(resized_frame, channels="RGB", use_column_width=True)
                    
                    if stop_camera:
                        camera_running = False
                        cap.release()
                        st.write("Kamera dihentikan.")
                        break

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
            finally:
                if camera_running:
                    cap.release()
