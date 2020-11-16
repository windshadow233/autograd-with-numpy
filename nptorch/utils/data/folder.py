from PIL import Image
import os
from . import Dataset


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DatasetFolder(Dataset):
    def __init__(self, root, loader, transform=None, extensions=None, check_valid_file=None):
        super(DatasetFolder, self).__init__()
        self.root = root
        self.loader = loader
        self.transform = transform
        self.extensions = extensions
        self.check_valid_file = check_valid_file

        self.classes, self.class_to_idx = self._find_classes(root)
        self.samples = self._make_samples(root, self.class_to_idx, extensions, check_valid_file)
        self.targets = [sample[1] for sample in self.samples]

    @staticmethod
    def _find_classes(root):
        classes = sorted([p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def _make_samples(root, class_to_idx, extensions=None, check_valid_file=None):
        samples = []
        if not ((extensions is None) ^ (check_valid_file is None)):
            raise ValueError("arguments extensions and check_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def check_valid_file(filename):
                return filename.endswith(extensions)
        for cls in os.listdir(root):
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            for rt, _, files in os.walk(d):
                for file in sorted(files):
                    path = os.path.join(rt, file)
                    if check_valid_file(path):
                        item = (path, class_to_idx[cls])
                        samples.append(item)
        return samples

    def __getitem__(self, item):
        sample_path, sample_target = self.samples[item]
        sample = self.loader(sample_path)
        if self.transform is None:
            return sample, sample_target
        sample = self.transform(sample)
        return sample, sample_target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'bmp')


class ImageFolder(DatasetFolder):
    def __init__(self, root, loader=pil_loader, transform=None, check_valid_file=None):
        super(ImageFolder, self).__init__(root, loader, transform,
                                          IMG_EXTENSIONS if check_valid_file is None else None,
                                          check_valid_file)
