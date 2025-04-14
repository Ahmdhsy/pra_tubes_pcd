import streamlit as st
from PIL import Image, ImageDraw
from app.utils import load_image, convert_image_to_array
from app.detection import detect_faces_haar, detect_faces_mtcnn
from streamlit_option_menu import option_menu

def run():
    st.title("Halaman Deteksi Wajah")

    selected = option_menu(
        menu_title="Opsi Deteksi Wajah",
        options=["Haar Cascade", "MTCNN", "Camera"],
        icons=["filetype-jpg", "filetype-jpg", 'camera'],
        menu_icon="list-task",
        default_index=0,
        orientation="horizontal",
    )
    
    if selected == "Haar Cascade":
        uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = load_image(uploaded_file)
            img_array = convert_image_to_array(image)

            faces = detect_faces_haar(img_array)
        
            st.write(f"Jumlah wajah terdeteksi: {len(faces)}")

            draw = ImageDraw.Draw(image)
            for face in faces:
                box = face['box']
                if selected == "MTCNN" and 'confidence' in face and face['confidence'] < 0.9:
                    continue
                draw.rectangle([box[0], box[1], box[0]+box[2], box[1]+box[3]], outline="red", width=2)

            st.image(image, caption="Hasil Deteksi", use_container_width=True)

    elif selected == "MTCNN":
        uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = load_image(uploaded_file)
            img_array = convert_image_to_array(image)

            faces = detect_faces_mtcnn(img_array)

            st.write(f"Jumlah wajah terdeteksi: {len(faces)}")

            draw = ImageDraw.Draw(image)
            for face in faces:
                box = face['box']
                if selected == "MTCNN" and 'confidence' in face and face['confidence'] < 0.9:
                    continue
                draw.rectangle([box[0], box[1], box[0]+box[2], box[1]+box[3]], outline="red", width=2)

            st.image(image, caption="Hasil Deteksi", use_container_width=True)
    
    elif selected == "Camera":
        st.write("Not implemented yet")

    
