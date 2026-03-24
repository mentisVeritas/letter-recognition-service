import numpy as np
from PIL import Image
import io


def preprocess_image(contents: bytes):
    img = Image.open(io.BytesIO(contents)).convert("L")
    img = img.resize((28, 28))
    return np.array(img, dtype=np.float32) / 255.0