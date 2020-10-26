import numpy as np
from PIL import Image
from ..tensor import Tensor


def to_tensor(img):
    if not isinstance(img, (np.ndarray, Image.Image)):
        raise TypeError(f'img should be PIL Image or ndarray. Got {type(img)}')
    if isinstance(img, np.ndarray):
        if img.ndim not in {2, 3}:
            raise ValueError(f'img should be 2 or 3 dimensional. Got {img.ndim} dimensions.')
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.transpose((2, 0, 1))
        if np.max(img) <= 1.:
            return Tensor(img, dtype=np.float32)
        else:
            return Tensor(img / 255, dtype=np.float32)
    if img.mode == '1':  # 二值图
        img = np.array(img)[:, :, None]
    elif img.mode in {'L', 'I', 'F', 'P'}:
        img = np.array(img)[:, :, None] / 255.
    elif img.mode in {'RGB', 'RGBA', 'CMYK', 'YCbCr'}:
        img = np.array(img) / 255.
    img = img.transpose((2, 0, 1))
    return Tensor(img, dtype=np.float32)
