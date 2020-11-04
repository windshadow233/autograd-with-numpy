import struct
import numpy as np
from nptorch.transforms import ToPILImage, Normalize, RandomMask, ToTensor
import nptorch


def load_mnist(img_path):
    with open(img_path, 'rb') as img:
        _, num, rows, cols = struct.unpack('>IIII', img.read(16))
        images = np.fromfile(img, dtype=np.uint8).reshape(num, rows, cols)
    return images

