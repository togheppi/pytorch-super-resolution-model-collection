from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale
from PIL import Image
from dataset import DatasetFromFolder


def download_bsds300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        if not exists(dest):
            makedirs(dest)
        url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor, interpolation=None):
    if interpolation == 'bicubic':
        interpolation = Image.BICUBIC
    elif interpolation == 'bilinear':
        interpolation = Image.BILINEAR
    elif interpolation == 'nearest':
        interpolation = Image.NEAREST

    if interpolation is not None:
        return Compose([
            # CenterCrop(crop_size),
            Scale(crop_size // upscale_factor),
            Scale(crop_size, interpolation=interpolation),  # upscale back to high-resolution image
            ToTensor()
        ])
    else:
        return Compose([
            # CenterCrop(crop_size),
            Scale(crop_size // upscale_factor),
            ToTensor()
        ])


def target_transform():
    return Compose([
        # CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(data_dir, dataset, upscale_factor, interpolation=None, is_rgb=False):
    root_dir = ''
    input_size = 256
    if dataset == 'bsds300':
        root_dir = download_bsds300(data_dir)
        input_size = 256
    elif dataset == 'fashion-mnist':
        input_size = 28
    elif dataset == 'celebA':
        input_size = 64

    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(input_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             is_rgb=is_rgb,
                             crop_size=crop_size,
                             fliplr=True,
                             input_transform=input_transform(crop_size, upscale_factor, interpolation=interpolation),
                             target_transform=target_transform())


def get_test_set(data_dir, dataset, upscale_factor, interpolation=None, is_rgb=False):
    root_dir = ''
    input_size = 256
    if dataset == 'bsds300':
        root_dir = download_bsds300(data_dir)
        input_size = 256
    elif dataset == 'fashion-mnist':
        input_size = 28
    elif dataset == 'celebA':
        input_size = 64

    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(input_size, upscale_factor)

    return DatasetFromFolder(test_dir,
                             is_rgb=is_rgb,
                             crop_size=crop_size,
                             fliplr=False,
                             input_transform=input_transform(crop_size, upscale_factor, interpolation=interpolation),
                             target_transform=target_transform())
