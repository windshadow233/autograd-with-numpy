import numpy as np
from PIL import Image
from ..tensor import Tensor


def to_tensor(img):
    if isinstance(img, np.ndarray):
        return Tensor(img / 255)
    if isinstance(img, Image.Image):
        return Tensor(np.array(img))