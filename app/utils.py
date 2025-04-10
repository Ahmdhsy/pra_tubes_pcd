from PIL import Image
import numpy as np

def load_image(image_file):
    img = Image.open(image_file)
    return img

def convert_image_to_array(image: Image.Image):
    return np.array(image)

def resize_image(image, size=(224, 224)):
    return image.resize(size)
