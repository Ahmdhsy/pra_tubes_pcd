import streamlit as st
import cv2
from app.utils import load_image, convert_image_to_array
from app.agegender import detect_faces_dnn, predict_age_gender
from streamlit_option_menu import option_menu

def run():
    st.title("Gender and Age Identifier")
    st.write("Identifikasi jenis kelamin dan usia individu berdasarkan gambar wajah.")
    
    selected = option_menu(
        menu_title="Choose Identifier Method",
        options=["Upload an Image", "Take a Photo"],
        icons=["file-earmark-arrow-up-fill", "camera-fill"],
        menu_icon="list-ul",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Upload an Image":
        uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = load_image(uploaded_file).convert("RGB")
            img_array = convert_image_to_array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
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
                    font_scale = max(min(w, h) / 300, 0.4   )
                    thickness = max(int(min(w, h) / 200), 2)
                    cv2.putText(img_array, label, (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                st.image(img_array, caption="Hasil Deteksi", use_column_width=True)
    
    elif selected == "Take a Photo":
        camera_image = st.camera_input("Take a Photo")

        if camera_image:
            image = load_image(camera_image).convert("RGB")
            img_array = convert_image_to_array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

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
                    
                    font_scale = max(min(w, h) / 300, 0.4)
                    thickness = max(int(min(w, h) / 200), 2)
                    cv2.putText(img_array, label, (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                st.image(img_array, caption="Hasil Deteksi", use_column_width=True)