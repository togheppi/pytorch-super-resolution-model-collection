import torch.utils.data as data
from torchvision.transforms import *
from os import listdir
from os.path import join
from PIL import Image
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, is_gray=False, random_scale=True, crop_size=None, rotate=True, fliplr=True, fliptb=True,
                 input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        if len(image_dirs) == 1:
            self.image_filenames = [join(image_dirs[0], x) for x in listdir(image_dirs[0]) if is_image_file(x)]
        else:
            self.image_filenames = []
            for image_dir in image_dirs:
                self.image_filenames.extend(join(image_dir, x) for x in listdir(image_dir) if is_image_file(x))
        self.is_gray = is_gray
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        w = input.size[0]
        h = input.size[1]

        if self.random_scale:
            # random scaling between [0.5, 1.0]
            eps = 1e-3
            ratio = random.randint(5, 10) * 0.1
            if w*ratio < self.crop_size:
                ratio = self.crop_size / w + eps
            if h*ratio < self.crop_size:
                ratio = self.crop_size / h + eps

            scale_w = int(w * ratio)
            scale_h = int(h * ratio)
            transform = Scale((scale_w, scale_h), interpolation=Image.BICUBIC)
            input = transform(input)

        if self.crop_size is not None:
            # random crop
            transform = RandomCrop(self.crop_size)
            input = transform(input)

        if self.rotate:
            # random rotation between [90, 180, 270] degrees
            rv = random.randint(1, 3)
            input = input.rotate(90*rv, expand=True)

        if self.fliplr:
            # random flip
            transform = RandomHorizontalFlip()
            input = transform(input)

        if self.fliptb:
            # random flip
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_TOP_BOTTOM)

        if self.is_gray:
            input = input.convert('YCbCr')
            input, _, _ = input.split()

        target = input.copy()

        if self.input_transform is not None:
            input = self.input_transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)