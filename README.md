![look[A]like](assets/img/Banner.jpg)
# look[A]like - Face Similarity & Ethnicity Recognition App

## About Application
**look[A]like** adalah sebuah Web Application yang memungkinkan pengguna untuk mendeteksi kemiripan wajah (*Face Similarity*) dan mengenali kemiripan suku atau etnis (*Ethnicity Recognition*), yang dibangun menggunakan bahasa pemrograman [Python](https://www.python.org/) dengan framework [Streamlit](https://streamlit.io/).

## App Features
### Face Detection (Completed)
- **Face Detection Using Image Upload**: Mendeteksi keberadaan wajah dalam gambar yang diunggah (format PNG/JPG/JPEG), tanpa melakukan identifikasi. Menggunakan metode Haar Cascade atau MTCNN.
- **Face Detection Using Live Camera**: Mendeteksi posisi wajah secara real-time menggunakan kamera perangkat. Cocok untuk aplikasi pemantauan atau pelacakan wajah.

### Face Recognition (Not Completed)
- **Face Recognition Using Image Upload**: Mengenali identitas wajah dari gambar yang diunggah dengan mencocokkannya ke database wajah yang sudah dikenal. Proses diawali dengan deteksi wajah menggunakan Haar Cascade atau MTCNN, dilanjutkan dengan ekstraksi fitur.
- **Face Recognition Using Live Camera**: Mengenali wajah secara langsung melalui kamera perangkat, dengan proses pencocokan ke identitas yang tersimpan dalam sistem.

### Face Similarity (Completed)
- **Face Similarity Using Image Upload**: Mengukur tingkat kemiripan antara dua wajah dengan mengunggah dua gambar (PNG/JPG/JPEG). Aplikasi akan menghitung dan menampilkan skor kemiripan di antara kedua wajah tersebut.
- **Face Similarity Using Live Camera**: Mengukur kemiripan dua wajah secara real-time melalui kamera perangkat, dengan perhitungan skor kemiripan berdasarkan hasil ekstraksi fitur.

### Ethnicity Recognition (Completed)
- **Ethnicity Recognition Using Image Upload**: Memprediksi kemiripan wajah pengguna dengan beberapa kelompok etnis berdasarkan model yang telah dilatih. Aplikasi menampilkan persentase kemiripan terhadap tiap kategori etnis.
- **Ethnicity Recognition Using Live Camera**: Melakukan prediksi etnis secara real-time menggunakan kamera perangkat, dengan hasil berupa persentase kemiripan terhadap beberapa kategori etnis.

### Age and Gender Identifier (Completed)
- **Age and Gender Identifier Using Image Upload**: Memprediksi usia dan jenis kelamin dari gambar yang diunggah menggunakan model yang telah dilatih pada dataset relevan. Hasil ditampilkan dalam bentuk estimasi usia dan gender.
- **Age and Gender Identifier Using Live Camera**: Memprediksi usia dan jenis kelamin pengguna secara real-time dari kamera perangkat, berdasarkan model yang telah dilatih.

## Developer Guidelines

### Project Structure
```
./
│
├── app/                          # Algoritma Program
│   ├── agegender.py              # Logic Fitur Age & Gender Identifier
│   ├── detection.py              # Logic Fitur Face Detection
│   ├── ethnicity.py              # Logic Fitur Ethnicity Recognition
│   ├── generate_csv.py           # Logic Fitur Generate CSV Dataset
│   ├── preprocessing.py          # Program untuk Training Dataset
│   ├── similarity.py             # Logic Fitur Similarity
│   └── utils.py                  # Utility Logic
│
├── assets/                       # Aset Aplikasi
│   ├── fonts                     # Font untuk Antarmuka Aplikasi
│   ├── img                       # Image untuk Antarmuka Aplikasi
│   └── style                     # Custom style untuk Antarmuka Aplikasi
│
├── content/                      # Antarmuka Aplikasi
│   ├── agegenderpage.py          # Antarmuka Halaman Age & Gender Identifier
│   ├── detectionpage.py          # Antarmuka Halaman Face Detection
│   ├── ethnicitypage.py          # Antarmuka Halaman Ethnicity Recognition
│   └── similaritypage.py         # Antarmuka Halaman Face Similarity
│
├── data/                         # Dataset Image untuk Training Ethnic
│   ├── processed                 # Pretrained Data
│   └── raw                       # Raw Data
│
├── meta/                         # Dataset Image untuk Training Similarity
├── models/                       # Hasil Model untuk Training Similarity
├── notebooks/                    # Embedding dan Classifier
├── venv/                         # Konfigurasi Virtual Environment
├── face_embedding.csv            # Hasil Generate CSV
├── packages.txt                  # Packages libgl1
├── pairs.csv                     # Path dan Label Pairing Data Positif dan Negatif
├── photo_paths.csv               # Path Raw Dataset Photo
├── .gitignore
├── app.py                        # Main Program dan Navigasi Aplikasi
├── README.md
└── requirements.txt              # Dependencies List
```
### Prerequisite
1. **Python** - Version [3.9](https://www.python.org/downloads/release/python-390/) or [3.10](https://www.python.org/downloads/release/python-3100/) (Versi terbaru mungkin dapat menyebabkan conflict pada beberapa dependencies)
2. **Visual Studio Code** - [Download](https://code.visualstudio.com/download) untuk pengembangan project
3. **Git** - [Download](https://git-scm.com/downloads) untuk cloning repository project

### Installation
Buka terminal di Visual Studio Code. 
```python
# Clone repository
git clone https://github.com/Ahmdhsy/pra_tubes_pcd.git

# Pindah direktori project
cd pra_tubes_pcd

# Instalasi Virtual Environment (Linux Only)
# -- Ubuntu/Debian
sudo apt install python3-venv
# -- Fedora
sudo dnf install python3-venv

# Buat Virtual Environment
python -m venv venv

# Aktivasi Virtual Environment
# -- Windows
source venv/Scripts/activate
# -- Mac/Linux
source venv/bin/activate

# Install Dependecies dari Requirements.txt
pip install -r requirements.txt

# Enjoy!
```


### How to Run
```python
streamlit run app.py
```

## User Guidelines
Pengguna yang ingin menggunakan aplikasi ini dapat mengujunginya pada tautan berikut https://py-lookalike.streamlit.app/

## Technology & Dependencies
- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Numpy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [Albumentations](https://albumentations.ai/)
- [Matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Deepface](https://pypi.org/project/deepface/)
- [Scikit-learn](https://scikit-learn.org/)
- [Imutils](https://pypi.org/project/imutils/)
- [Pillow](https://pypi.org/project/pillow/)
- [ImgAug](https://pypi.org/project/imgaug/)
- [MTCNN](https://pypi.org/project/mtcnn/)
- [Seaborn](https://seaborn.pydata.org/)

## Reference
- [Face Recognition with Python/OpenCV](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) - PyImageSearch
- [Build a Face Recognition System with FaceNet](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) - Machine Learning Mastery
- [Facial Recognition with Transfer Learning in Keras](https://keras.io/examples/vision/siamese_network/) - Keras Examples
- [Data Augmentation for Face Recognition](https://github.com/aleju/imgaug) - ImgAug Library
- [Ethnic Classification dengan CNN dan Deep Learning](https://github.com/aleju/imgaug) - DeepFace Library GitHub

## Contributor
1. **Ahmad F. Naji** - [https://github.com/Ahmdhsy](https://github.com/Ahmdhsy)
   - Leader and Developed the Ethnicity Recognition module.
2. **Bandyaga A. Sugandi** - [https://github.com/basganajaah](https://github.com/basganajaah)
   - Set up the project environment, developed the Face Detection and Face Similarity modules.
3. **Dwika A. R. Ibrahim** - [https://github.com/DAliRIJTK](https://github.com/DAliRIJTK)
   - Handled dataset preparation for training and developed the Age & Gender Identifier.

## Contact
- [Github](https://github.com/Ahmdhsy/pra_tubes_pcd)
- [Gmail](mailto:bandyagaadiansyah@gmail.com)

Pra Tugas Besar Pengolahan Citra Digital - D3 Teknik Informatika - Politeknik Negeri Bandung
