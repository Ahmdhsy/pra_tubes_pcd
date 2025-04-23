import streamlit as st
import cv2
import numpy as np
from app.utils import load_image, convert_image_to_array
from app.agegender import detect_faces_mtcnn, predict_age_gender_emotion
from streamlit_option_menu import option_menu

def run():
    st.title("Gender and Age Identifier")
    st.write("Identifikasi jenis kelamin, usia individu, dan deteksi emosi berdasarkan gambar wajah.")
    
    selected = option_menu(
        menu_title="Choose Identifier Method",
        options=["Image Upload", "Take Photo"],
        icons=["file-earmark-arrow-up-fill", "camera-fill"],
        menu_icon="image",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Image Upload":
        uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = load_image(uploaded_file).convert("RGB")
            img_array = convert_image_to_array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            faces = detect_faces_mtcnn(img_array)

            if not faces:
                st.warning("Tidak ada wajah terdeteksi.")
                return

            results_info = []
            man_count = 0
            woman_count = 0

            for idx, face in enumerate(faces):
                x, y, w, h = face
                y_max = min(y + h, img_array.shape[0])
                x_max = min(x + w, img_array.shape[1])
                face_img = img_array[max(0, y):y_max, max(0, x):x_max]

                if face_img.size == 0:
                    continue

                age, gender, gender_prob, emotion, emotion_prob = predict_age_gender_emotion(face_img)
                label = f"{idx+1}: {gender}, {age}yr, {emotion}"

                cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font_scale = max(min(w, h) / 300, 0.4)
                thickness = max(int(min(w, h) / 200), 2)
                cv2.putText(img_array, label, (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                results_info.append((idx + 1, gender, age, gender_prob, emotion, emotion_prob))
                if gender.lower() == "man":
                    man_count += 1
                elif gender.lower() == "woman":
                    woman_count += 1

            col1, spacer, col2 = st.columns([1, 0.05, 1])

            with col1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                st.image(img_array, caption="Hasil Deteksi", use_column_width=True)

            with col2:
                st.markdown(f"### Hasil Ringkasan")
                st.write(f"Total Wajah Terdeteksi: {len(results_info)}")
                st.write(f"Jumlah Man: {man_count}")
                st.write(f"Jumlah Woman: {woman_count}")
                st.markdown("### Data Tiap Wajah:")
                for idx, gender, age, gender_prob, emotion, emotion_prob in results_info:
                    woman_prob = gender_prob.get('Woman', 0)
                    man_prob = gender_prob.get('Man', 0)
                    st.markdown(f"""
                    **No. {idx}:**
                    - Jenis Kelamin: {gender}
                    - Usia: {age} tahun
                    - Emosi Dominan: {emotion}
                    - Probabilitas Gender:
                        - Perempuan: {woman_prob:.2f}%
                        - Laki-laki: {man_prob:.2f}%
                    - Probabilitas Emosi:
                        - Happy: {emotion_prob.get('happy', 0):.2f}%
                        - Sad: {emotion_prob.get('sad', 0):.2f}%
                        - Angry: {emotion_prob.get('angry', 0):.2f}%
                        - Surprise: {emotion_prob.get('surprise', 0):.2f}%
                        - Neutral: {emotion_prob.get('neutral', 0):.2f}%
                    """)

    elif selected == "Take Photo":
        camera_image = st.camera_input("Pengambilan Gambar")

        if camera_image:
            image = load_image(camera_image).convert("RGB")
            img_array = convert_image_to_array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            faces = detect_faces_mtcnn(img_array)

            if not faces:
                st.warning("Tidak ada wajah terdeteksi.")
                return

            results_info = []
            man_count = 0
            woman_count = 0

            for idx, face in enumerate(faces):
                x, y, w, h = face
                y_max = min(y + h, img_array.shape[0])
                x_max = min(x + w, img_array.shape[1])
                face_img = img_array[max(0, y):y_max, max(0, x):x_max]

                if face_img.size == 0:
                    continue

                age, gender, gender_prob, emotion, emotion_prob = predict_age_gender_emotion(face_img)
                label = f"{idx+1}: {gender}, {age}yr, {emotion}"

                cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font_scale = max(min(w, h) / 300, 0.4)
                thickness = max(int(min(w, h) / 200), 2)
                cv2.putText(img_array, label, (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                results_info.append((idx + 1, gender, age, gender_prob, emotion, emotion_prob))
                if gender.lower() == "man":
                    man_count += 1
                elif gender.lower() == "woman":
                    woman_count += 1

            cameracols1, spacer, cameracols2 = st.columns([1, 0.05, 1])

            with cameracols1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                st.image(img_array, caption="Hasil Deteksi", use_column_width=True)

            with cameracols2:
                st.markdown(f"### Hasil Ringkasan")
                st.write(f"Total Wajah Terdeteksi: {len(results_info)}")
                st.write(f"Jumlah Man: {man_count}")
                st.write(f"Jumlah Woman: {woman_count}")
                st.markdown("### Data Tiap Wajah:")
                for idx, gender, age, gender_prob, emotion, emotion_prob in results_info:
                    woman_prob = gender_prob.get('Woman', 0)
                    man_prob = gender_prob.get('Man', 0)
                    st.markdown(f"""
                    **No. {idx}:**
                    - Jenis Kelamin: {gender}
                    - Usia: {age} tahun
                    - Emosi Dominan: {emotion}
                    - Probabilitas Gender:
                        - Perempuan: {woman_prob:.2f}%
                        - Laki-laki: {man_prob:.2f}%
                    - Probabilitas Emosi:
                        - Happy: {emotion_prob.get('happy', 0):.2f}%
                        - Sad: {emotion_prob.get('sad', 0):.2f}%
                        - Angry: {emotion_prob.get('angry', 0):.2f}%
                        - Surprise: {emotion_prob.get('surprise', 0):.2f}%
                        - Neutral: {emotion_prob.get('neutral', 0):.2f}%
                    """)





