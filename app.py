import streamlit as st
import os
from streamlit_option_menu import option_menu # type: ignore

st.set_page_config(page_title="look[A]like", layout='wide')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/style/style.css")

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation Menu",
        options=["Home", "Face Detection", "Face Similarity", "Face Recognition", "Ethnicity Recognition", "Extra Feature"],
        icons=["house-fill",  "person-rolodex", "people-fill", "person-bounding-box", "person-lines-fill", "emoji-wink-fill"],
        menu_icon="cast",
        default_index=0
    )

if selected == "Home":
    image_path = os.path.join("assets", "img", "Banner.jpg")
    st.image(image_path, use_column_width=True)
    
elif selected == "Face Detection":
    import content.detectionpage as detection
    detection.run()

elif selected == "Face Similarity":
    import content.similaritypage as similarity
    similarity.run()

elif selected == "Face Recognition":
    import content.recognitionpage as recognition
    recognition.run()
    
elif selected == "Ethnicity Recognition":
    import content.ethnicitypage as ethnic
    ethnic.run()
    
elif selected == "Extra Feature":
    import content.extrafeaturepage as extra
    extra.run()