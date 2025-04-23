import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
from app.similarity import get_embedding, compute_similarity_score

def run():
    st.title("Face Similarity")
    st.write("Bandingkan dua gambar wajah untuk mengetahui tingkat kemiripan.")

    selected = option_menu(
        menu_title="Choose Similarity Method",
        options=["Upload an Image", "Take a Photo"],
        icons=["file-earmark-arrow-up-fill", "camera"],
        menu_icon="list-ul",
        default_index=0,
        orientation="horizontal",
    )

    image1, image2 = None, None

    if selected == "Upload an Image":
        uploaded_file1 = st.file_uploader("Upload Gambar 1", type=["jpg", "jpeg", "png"], key="upload1")
        uploaded_file2 = st.file_uploader("Upload Gambar 2", type=["jpg", "jpeg", "png"], key="upload2")

        if uploaded_file1 and uploaded_file2:
            image1 = Image.open(uploaded_file1).convert("RGB")
            image2 = Image.open(uploaded_file2).convert("RGB")

    elif selected == "Take a Photo":
        camera_image1 = st.camera_input("Ambil Gambar 1", key="cam1")
        camera_image2 = st.camera_input("Ambil Gambar 2", key="cam2")

        if camera_image1 and camera_image2:
            image1 = Image.open(camera_image1).convert("RGB")
            image2 = Image.open(camera_image2).convert("RGB")

    if image1 and image2:
        st.subheader("📷 Gambar yang Dimasukkan")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, caption="Gambar 1", use_column_width=True)
        with col2:
            st.image(image2, caption="Gambar 2", use_column_width=True)

        st.subheader("📊 Hasil Kemiripan Wajah")
        with st.spinner("Menghitung kemiripan wajah..."):
            emb1 = get_embedding(image1)
            emb2 = get_embedding(image2)
            similarity = compute_similarity_score(emb1, emb2)

            if np.isnan(similarity):
                st.error("❌ Gagal menghitung kemiripan. Coba ulangi dengan gambar lain.")
                return

            similarity_percent = round(similarity * 100, 2)

            st.metric(label="Similarity Score", value=f"{similarity:.4f}")
            st.metric(label="Kemiripan", value=f"{similarity_percent:.2f}%")
            if similarity >= 0.60:
                st.success("✅ Wajah Mirip")
            else:
                st.error("❌ Wajah Tidak Mirip")
