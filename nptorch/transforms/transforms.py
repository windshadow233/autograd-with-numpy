import numpy as np
from numbers import Number
from . import functional as F


class Compose(object):
    def __init__(self, transforms):
        """

        @param transforms: a list of transform
        """
        self.transforms = transforms

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img

    def __repr__(self):
        s = self.__class__.__name__ + '('
        for transform in self.transforms:
            s += '\n'
            s += f'    {transform}'
        s += '\n)'
        return s


class Transform(object):
    def __repr__(self):
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self):
        return ''


class ToTensor(Transform):
    """
    将PIL.Image.Image类型或numpy.ndarray类型转换为Tensor,数据范围在0-1,若为多通道,则放在最后一维
    """
    def __call__(self, img):
        return F.to_tensor(img)


class Resize(Transform):
    """
    将PIL.Image.Image类型的图片修改尺寸,
    """
    def __init__(self, size):
        self.size = size

    def extra_repr(self):
        return f'size={self.size}'

    def __call__(self, img):
        return F.resize(img, self.size)


class Grayscale(Transform):
    """
    将PIL.Image.Image类型转为灰度
    """
    def __init__(self, out_channels):
        if out_channels not in {1, 3}:
            raise ValueError(f'output_channels should be either 1 or 3. Got {out_channels}')
        self.out_channels = out_channels

    def extra_repr(self):
        return f'out_channels={self.out_channels}'

    def __call__(self, img):
        return F.gray_scale(img, self.out_channels)


class ToPILImage(Transform):
    """
    将Tensor或np.ndarray类型转为PIL.Image.Image,若为多通道图像,需要指定mode
    """
    def __init__(self, mode=None):
        self.mode = mode

    def extra_repr(self):
        return f'mode={self.mode}'

    def __call__(self, img):
        return F.to_pil_image(img, self.mode)


class Normalize(Transform):
    """
    将Tensor或np.ndarray类型的三维图片进行标准化
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def extra_repr(self):
        return f'mean={self.mean}, std={self.std}'

    def __call__(self, img):
        return F.normalize(img, self.mean, self.std)


class RandomHorizontalFlip(Transform):
    """
    以概率p左右翻转
    """
    def __init__(self, p=0.5):
        self.p = p

    def extra_repr(self):
        return f'p={self.p}'

    def __call__(self, img):
        return F.random_horizontal_flip(img, self.p)


class RandomVerticalFlip(Transform):
    """
    以概率p上下翻转
    """
    def __init__(self, p=0.5):
        self.p = p

    def extra_repr(self):
        return f'p={self.p}'

    def __call__(self, img):
        return F.random_vertical_flip(img, self.p)


class CenterCrop(Transform):
    """
    在图片中央截取给定大小的区域
    """
    def __init__(self, size):
        if not isinstance(size, tuple):
            size = (size, size)
        self.size = size

    def extra_repr(self):
        return f'size={self.size}'

    def __call__(self, img):
        return F.center_crop(img, self.size)


class RandomCrop(Transform):
    """
    在图片中随机截取给定大小的区域
    """
    def __init__(self, size):
        if not isinstance(size, tuple):
            size = (size, size)
        self.size = size

    def extra_repr(self):
        return f'size={self.size}'

    def _get_random_field(self, img):
        w, h = img.size
        out_h, out_w = self.size
        if w == out_w and h == out_h:
            return 0, 0, h, w

        x = np.random.randint(0, w - out_w)
        y = np.random.randint(0, h - out_h)
        return x, y, out_h, out_w

    def __call__(self, img):
        x, y, h, w = self._get_random_field(img)
        return F.crop(img, x, y, h, w)


class RandomMask(Transform):
    """
    以p的概率在张量形式的图片中随机擦除给定大小的区域
    """

    def __init__(self, size, p=0.5, value=0.):
        """

        @param size: 区域大小
        @param p: 擦除概率
        @param value: 用此值来替换被擦除的部分,若value是列表形式,则要求图像通道数与其长度相同
        """
        if not isinstance(size, tuple):
            size = (size, size)
        self.size = size
        assert 0. <= p <= 1., 'probability p must be in interval: [0, 1]'
        self.p = p
        assert isinstance(value, (Number, list))
        if isinstance(value, Number):
            value = [value]
        self.value = value

    def extra_repr(self):
        return f'size={self.size}, p={self.p}, value={self.value}'

    def _get_random_field(self, img):
        w, h = img.shape[-2:]
        out_h, out_w = self.size
        if w == out_w and h == out_h:
            return 0, 0, h, w

        x = np.random.randint(0, w - out_w)
        y = np.random.randint(0, h - out_h)
        return x, y, out_h, out_w

    def __call__(self, img):
        if np.random.rand() >= self.p:
            return img
        x, y, h, w = self._get_random_field(img)
        return F.mask(img, x, y, h, w, self.value)


