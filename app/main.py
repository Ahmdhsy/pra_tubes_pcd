import streamlit as st
from PIL import Image, ImageDraw
from utils import load_image, convert_image_to_array
from detection import detect_faces_haar, detect_faces_mtcnn

st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("Deteksi Wajah: Haar vs MTCNN")

method = st.radio("Pilih Metode Deteksi:", ["Haar Cascade", "MTCNN"])

uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = load_image(uploaded_file)
    img_array = convert_image_to_array(image)

    if method == "Haar Cascade":
        faces = detect_faces_haar(img_array)
    else:
        faces = detect_faces_mtcnn(img_array)

    st.write(f"Jumlah wajah terdeteksi: {len(faces)}")

    draw = ImageDraw.Draw(image)
    for face in faces:
        box = face['box']
        if method == "MTCNN" and 'confidence' in face and face['confidence'] < 0.9:
            continue
        draw.rectangle([box[0], box[1], box[0]+box[2], box[1]+box[3]], outline="red", width=2)

    st.image(image, caption="Hasil Deteksi", use_column_width=True)
