![look[A]like](assets/img/Banner.jpg)
# look[A]like - Face Similarity & Ethnicity Recognition App

## About Application
**look[A]like** adalah sebuah Web Application yang memungkinkan pengguna untuk mendeteksi kemiripan wajah (*Face Similarity*) dan mengenali kemiripan suku atau etnis (*Ethnicity Recognition*), yang dibangun menggunakan bahasa pemrograman [Python](https://www.python.org/) dengan framework [Streamlit](https://streamlit.io/).

## App Features
### Face Detection
- **Face Detection Using Image Upload**: Mendeteksi keberadaan wajah pada gambar yang diunggah (format PNG/JPG/JPEG) tanpa mengenali identitas. Pendekatan menggunakan metode HaarCascade atau MTCNN.
- **Face Detection Using Live Camera**: Mendeteksi posisi wajah secara real-time menggunakan kamera perangkat, cocok untuk aplikasi pemantauan atau pelacakan wajah.

### Face Recognition
- **Face Recognition Using Image Upload**: Mengenali identitas wajah dari gambar yang diunggah dengan mencocokkannya ke database wajah yang sudah dikenal. Deteksi awal dilakukan menggunakan HaarCascade atau MTCNN sebelum proses ekstraksi fitur.
- **Face Recognition Using Live Camera**: Mengenali wajah secara langsung melalui kamera perangkat, dengan proses pencocokan ke identitas yang tersimpan dalam sistem.

### Face Similarity
- **Face Similarity Using Image Upload**: Mengukur tingkat kemiripan antara dua wajah dengan mengunggah dua gambar dalam format PNG/JPG/JPEG. Aplikasi akan menghitung dan menampilkan skor kemiripan antara kedua wajah tersebut.

### Ethnicity Recognition
- **Ethnicity Recognition Using Image Upload**: Memprediksi kemiripan wajah pengguna dengan suku atau etnis tertentu berdasarkan model yang telah dilatih pada dataset etnis. Aplikasi akan menampilkan persentase kemiripan terhadap beberapa kategori etnis.

## Developer Guidelines

### Project Structure
```
./
│
├── app/                          # Algoritma Program
│   ├── detection.py              # Fitur Face Detection
│   └── preprocessing.py          # Program untuk Training Dataset
│
├── assets/                       # Aset Aplikasi
│   └── img                       # Image untuk Antarmuka Aplikasi
│
├── content/                      # Antarmuka Aplikasi
│   ├── detectionpages.py         # Antarmuka Halaman Face Detection
│   └── ethnicitypages.py         # Antarmuka Halaman Ethnicity Recognition
│
├── utils/                        # Utility Functions
│   └── imageutils.py             # Utility untuk Upload Image
│
├── data/                         # Dataset Image untuk Training
│   ├── processed                 # Pretrained Data
│   └── raw                       # Raw Data
│ 
├── notebooks/                    # Embedding dan Classifier
│
├── venv/                         # Konfigurasi Virtual Environment
│
├── .gitignore
├── app.py                        # Main Program dan Navigasi Aplikasi
├── README.md
└── requirements.txt              # Daftar Dependensi
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
Pengguna yang ingin menggunakan aplikasi ini dapat mengujunginya pada tautan berikut.

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

## Reference
- [Face Recognition with Python/OpenCV](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) - PyImageSearch
- [Build a Face Recognition System with FaceNet](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) - Machine Learning Mastery
- [Facial Recognition with Transfer Learning in Keras](https://keras.io/examples/vision/siamese_network/) - Keras Examples
- [Data Augmentation for Face Recognition](https://github.com/aleju/imgaug) - ImgAug Library
- [Ethnic Classification dengan CNN dan Deep Learning](https://github.com/aleju/imgaug) - DeepFace Library GitHub

## Contributor
1. **Ahmad F. Naji** (231511033) - [https://github.com/Ahmdhsy](https://github.com/Ahmdhsy)
   - Leader and Developed the Ethnicity Recognition module.
2. **Bandyaga A. Sugandi** (231511037) - [https://github.com/basganajaah](https://github.com/basganajaah)
   - Set up the project environment, developed the Face Detection and Face Recognition modules.
3. **Dwika A. R. Ibrahim** (231511042) - [https://github.com/DAliRIJTK](https://github.com/DAliRIJTK)
   - Handled dataset preparation for training and developed the Face Similarity module.

## Contact
- [Github](https://github.com/Ahmdhsy/pra_tubes_pcd)
- [Gmail](mailto:bandyagaadiansyah@gmail.com)

Pra Tugas Besar Pengolahan Citra Digital - D3 Teknik Informatika - Politeknik Negeri Bandung
