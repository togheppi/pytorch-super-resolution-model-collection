import torch.utils.data as data
from torchvision.transforms import *
from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, is_gray=False, crop_size=None, fliplr=False, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.is_gray = is_gray
        self.crop_size = crop_size
        self.fliplr = fliplr

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])

        if self.crop_size is not None:
            # center crop
            transform = CenterCrop(self.crop_size)
            input = transform(input)
        if self.fliplr:
            # random flip
            transform = RandomHorizontalFlip()
            input = transform(input)

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