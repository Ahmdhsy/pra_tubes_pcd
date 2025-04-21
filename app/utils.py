from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

def load_image(image_file):
    img = Image.open(image_file)
    return img

def convert_image_to_array(image: Image.Image):
    return np.array(image)

def resize_image(image, size=(224, 224)):
    return image.resize(size)

def image_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def draw_faces(image, faces, font_path):
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    font = ImageFont.truetype(font_path, size=16)
    for i, face in enumerate(faces):
        box = face['box']
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline="green", width=4)
        label = f"Wajah {i+1}"
        draw.text((x, y - 30), label, fill="green", font=font)
    return image_copy
