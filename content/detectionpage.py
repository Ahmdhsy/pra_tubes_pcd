import streamlit as st
import os
from PIL import ImageFont, ImageDraw
from app.utils import load_image, convert_image_to_array
from app.detection import detect_faces_haar, detect_faces_mtcnn, detect_faces_retinaface
from streamlit_option_menu import option_menu

def run():
    st.title("Face Detection")
    st.write("Halaman untuk melihat perbandingan metode deteksi wajah menggunakan Haar Cascade Classifier, MTCNN, dan RetinaFace.")

    tab1, tab2 = st.tabs(["Face Detection Using Image Upload", "Face Detection Using Live Camera"])

    with tab1:
        selected = option_menu(
            menu_title="Choose Detection Method",
            options=["Haar Cascade Classifier", "MTCNN", "RetinaFace"],
            icons=["image", "image-fill", 'images'],
            menu_icon="list-task",
            default_index=0,
            orientation="horizontal",
            key="upload_menu"
        )

        uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = load_image(uploaded_file)
            img_array = convert_image_to_array(image)

            if selected == "Haar Cascade Classifier":
                faces = detect_faces_haar(img_array)

            elif selected == "MTCNN":
                faces = detect_faces_mtcnn(img_array)
                faces = [f for f in faces if f.get("confidence", 1.0) >= 0.9]

            elif selected == "RetinaFace":
                faces = detect_faces_retinaface(img_array)
                faces = [f for f in faces if f.get("confidence", 1.0) >= 0.9]

            else:
                faces = []

            st.write(f"Jumlah wajah terdeteksi: {len(faces)}")

            font_path = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "jakartasans-bold.ttf")
            font = ImageFont.truetype(font_path, size=16)
            draw = ImageDraw.Draw(image)
            for i, face in enumerate(faces):
                box = face['box']
                x, y, w, h = box
                draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
                label = f"Wajah {i+1}"
                draw.text((x, y - 30), label, fill="red", font=font)

            st.image(image, caption="Hasil Deteksi", use_column_width=True)

    with tab2:
        method = option_menu(
            menu_title="Choose Detection Method",
            options=["Haar Cascade Classifier", "MTCNN", "RetinaFace"],
            icons=["image", "image-fill", 'images'],
            menu_icon="list-task",
            default_index=0,
            orientation="horizontal",
            key="camera_menu"
        )
        camera_file = st.camera_input("Take a picture")
        if camera_file is not None:
            image = load_image(camera_file)
            img_array = convert_image_to_array(image)
            
            if method == "Haar Cascade Classifier":
                faces = detect_faces_haar(img_array)
            elif method == "MTCNN":
                faces = detect_faces_mtcnn(img_array)
                faces = [f for f in faces if f.get("confidence", 1.0) >= 0.9]
            elif method == "RetinaFace":
                faces = detect_faces_retinaface(img_array)
                faces = [f for f in faces if f.get("confidence", 1.0) >= 0.9]

            st.write(f"Jumlah wajah terdeteksi: {len(faces)}")

            font_path = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "jakartasans-bold.ttf")
            font = ImageFont.truetype(font_path, size=20)
            draw = ImageDraw.Draw(image)
            for i, face in enumerate(faces):
                box = face['box']
                x, y, w, h = box
                draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
                label = f"Wajah {i+1}"
                draw.text((x, y - 30), label, fill="red", font=font)

            st.image(image, caption="Hasil Deteksi", use_column_width=True)