import streamlit as st
from streamlit_option_menu import option_menu
from app.utils import load_image

def run():
    st.title("Face Similarity")
    st.write("Bandingkan dua gambar wajah untuk mengetahui tingkat kemiripan secara visual.")

    selected = option_menu(
        menu_title="Choose Similarity Method",
        options=["Upload an Image", "Take a Photo"],
        icons=["file-earmark-arrow-up-fill", "camera-fill"],
        menu_icon="list-ul",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Upload an Image":
        uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = load_image(uploaded_file)

    elif selected == "Take a Photo":
        camera_image = st.camera_input("Take a Photo")
        if camera_image:
            image = load_image(camera_image)

