import streamlit as st
import os
import io
import zipfile
from app.utils import load_image, convert_image_to_array, image_to_bytes, draw_faces
from app.detection import detect_faces_haar, detect_faces_mtcnn, detect_faces_retinaface
from streamlit_option_menu import option_menu

def run():
    font_path = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "jakartasans-bold.ttf")

    def show_results(image):
        """Menampilkan hasil deteksi wajah dari ketiga algoritma."""
        img_array = convert_image_to_array(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Haar Cascade")
            faces_haar = detect_faces_haar(img_array)
            st.write(f"Jumlah wajah terdeteksi: {len(faces_haar)}")
            result_haar = draw_faces(image, faces_haar, font_path)
            st.image(result_haar, caption="Hasil Haar Cascade", use_column_width=True)

        with col2:
            st.subheader("MTCNN")
            faces_mtcnn = [f for f in detect_faces_mtcnn(img_array) if f.get("confidence", 1.0) >= 0.9]
            st.write(f"Jumlah wajah terdeteksi: {len(faces_mtcnn)}")
            result_mtcnn = draw_faces(image, faces_mtcnn, font_path)
            st.image(result_mtcnn, caption="Hasil MTCNN", use_column_width=True)

        with col3:
            st.subheader("RetinaFace")
            faces_retina = [f for f in detect_faces_retinaface(img_array) if f.get("confidence", 1.0) >= 0.9]
            st.write(f"Jumlah wajah terdeteksi: {len(faces_retina)}")
            result_retina = draw_faces(image, faces_retina, font_path)
            st.image(result_retina, caption="Hasil RetinaFace", use_column_width=True)

        return result_haar, result_mtcnn, result_retina    

    def process_and_download(image):
        result_haar, result_mtcnn, result_retina = show_results(image)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("haar_result.png", image_to_bytes(result_haar))
            zip_file.writestr("mtcnn_result.png", image_to_bytes(result_mtcnn))
            zip_file.writestr("retinaface_result.png", image_to_bytes(result_retina))

        zip_buffer.seek(0)
        st.download_button(
            label="Download ZIP",
            data=zip_buffer,
            file_name="face_detection_compare_results.zip",
            mime="application/zip",
        )

    st.title("Face Detection")
    st.write("Bandingkan performa berbagai metode deteksi wajah: Haar Cascade, MTCNN, dan RetinaFace.")

    selected = option_menu(
        menu_title="Choose Detection Method",
        options=["Upload an Image", "Take a Photo"],
        icons=["file-earmark-arrow-up-fill", "camera"],
        menu_icon="list-task",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Upload an Image":
        uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = load_image(uploaded_file)
            process_and_download(image)

    elif selected == "Take a Photo":
        camera_file = st.camera_input("Ambil Foto")
        if camera_file:
            image = load_image(camera_file)
            process_and_download(image)
