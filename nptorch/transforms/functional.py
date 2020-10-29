import numpy as np
from PIL import Image
from ..tensor import Tensor


def _is_pil_img(img):
    return isinstance(img, Image.Image)


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_tensor(img):
    return isinstance(img, Tensor)


def to_tensor(img):
    if not (_is_tensor(img) or _is_numpy(img) or _is_pil_img(img)):
        raise TypeError(f'img should be Tensor, PIL Image or ndarray. Got {type(img)}')
    if isinstance(img, (np.ndarray, Tensor)):
        assert img.ndim in {2, 3}, f'img should be 2 or 3 dimensional. Got {img.ndim} dimensions.'
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.transpose((2, 0, 1))
        if np.max(img) <= 1.:
            return Tensor(img, dtype=np.float32)
        else:
            return Tensor(img / 255., dtype=np.float32)
    if img.mode == '1':
        img = np.array(img)[:, :, None]
    elif img.mode in {'L', 'I', 'F', 'P'}:
        img = np.array(img)[:, :, None] / 255.
    elif img.mode in {'RGB', 'RGBA', 'CMYK', 'YCbCr'}:
        img = np.array(img) / 255.
    img = img.transpose((2, 0, 1))
    return Tensor(img, dtype=np.float32)


def resize(img, size):
    if not _is_pil_img(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    img = img.resize(size)
    return img


def gray_scale(img, out_channels):
    if not _is_pil_img(img):
        raise TypeError(f'img should be PIL Image. Got {type(img)}')
    assert out_channels in {1, 3}, f'output_channels should be either 1 or 3. Got {out_channels}'
    img = img.convert('L')
    if out_channels == 1:
        return img
    img = Image.merge('RGB', (img,) * 3)
    return img


def to_pil_image(img: np.ndarray or Tensor, mode=None):
    if not (_is_numpy(img) or _is_tensor(img)):
        raise TypeError(f'img should be ndarray or Tensor. Got {type(img)}.')
    assert img.ndim in {2, 3}, f'img should be 2 or 3 dimensional. Got {img.ndim} dimensions.'
    if isinstance(img, Tensor):
        img = img.data
    if img.ndim == 3:
        if img.shape[0] == 3:
            img = img.transpose((1, 2, 0))
        elif img.shape[0] == 1:
            img = img.squeeze()
    if np.max(img) <= 1.0:
        img = img * 255
    if mode == '1':
        img = Image.fromarray(img, 'L').convert('1')
        return img
    if mode in {'L', 'P'}:
        img = img.astype(np.uint8)
    elif mode == 'I':
        img = img.astype(np.uint32)
    img = Image.fromarray(img, mode)
    return img


def normalize(img, mean, std):
    if not (_is_numpy(img) or _is_tensor(img)):
        raise TypeError(f'img should be ndarray or Tensor. Got {type(img)}')
    assert img.ndim == 3, 'img should be 2 or 3 dimensional. Got {img.ndim} dimensions.'
    mean = Tensor(mean)
    std = Tensor(std)
    return (img - mean[:, None, None]) / std[:, None, None]
