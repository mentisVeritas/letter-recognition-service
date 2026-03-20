import base64
import numpy as np
from PIL import Image
from io import BytesIO

def preprocess(base64_str):
    # убрать prefix
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    # padding fix
    base64_str += "=" * (-len(base64_str) % 4)

    image_bytes = base64.b64decode(base64_str)

    img = Image.open(BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28))

    arr = np.array(img, dtype=np.uint8)

    # инверсия
    arr = 255 - arr

    # убрать шум
    arr[arr < 30] = 0

    # reshape
    arr = arr.reshape(1, -1)

    return arr