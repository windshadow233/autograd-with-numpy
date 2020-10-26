import numpy as np
from PIL import Image
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
        return f'{self.__class__.__name__}()'


class ToTensor(Transform):
    """
    将PIL.Image.Image类型或numpy.ndarray类型转换为Tensor,数据范围在0-1
    """
    def __call__(self, img):
        return F.to_tensor(img)

