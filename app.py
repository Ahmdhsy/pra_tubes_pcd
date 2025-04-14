import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="look[A]like", layout='wide')

with st.sidebar:
    selected = option_menu(
        menu_title="Navigasi",
        options=["Beranda", "Deteksi Wajah", "Deteksi Suku"],
        icons=["house", "people-fill", "person-bounding-box"],
        menu_icon="cast",
        default_index=0
    )

if selected == "Beranda":
    image_path = r'C:\Users\basga\Politeknik Negeri Bandung\Semester 4\Pengolahan Citra Digital\Praktek\ETS\pra_tubes_pcd\assets\img\banner.jpg'
    st.image(image_path, use_container_width=True)

elif selected == "Deteksi Wajah":
    import content.detectionpage as detectionpage
    detectionpage.run()

elif selected == "Deteksi Suku":
    import content.ethnicitypage as ethnic
    ethnic.run()