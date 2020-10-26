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
    将PIL.Image.Image类型或numpy.ndarray类型转换为Tensor,数据范围在0-1
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

