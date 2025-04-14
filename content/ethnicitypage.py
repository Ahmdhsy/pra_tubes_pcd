import streamlit as st
from PIL import Image, ImageDraw
from app.utils import load_image, convert_image_to_array
from app.detection import detect_faces_haar, detect_faces_mtcnn

def run():
    st.title("Halaman Deteksi Suku")