from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import *
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


def input_transform(crop_size, upscale_factor, normalize=False):
    # downscale to low-resolution image
    transforms = [Scale(crop_size // upscale_factor)]

    # # upscale back to high-resolution image
    # if interpolation is not None:
    #     transforms.append(Scale(crop_size, interpolation=interpolation))

    # convert (0, 255) image to torch.tensor (0, 1)
    transforms.append(ToTensor())

    # normalize (-1, 1)
    if normalize:
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5]))

    return Compose(transforms)


def target_transform(normalize=False):
    # convert (0, 255) image to torch.tensor (0, 1)
    transforms = [ToTensor()]

    # normalize (-1, 1)
    if normalize:
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5]))

    return Compose(transforms)


def get_training_set(data_dir, dataset, image_size, upscale_factor, is_gray=False, normalize=False):
    root_dir = ''
    if dataset == 'bsds300':
        root_dir = download_bsds300(data_dir)

    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(image_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             is_gray=is_gray,
                             crop_size=crop_size,   # random crop
                             fliplr=True,           # random flip
                             input_transform=input_transform(crop_size, upscale_factor, normalize=normalize),
                             target_transform=target_transform(normalize=normalize))


def get_test_set(data_dir, dataset, image_size, upscale_factor, is_gray=False, normalize=False):
    root_dir = ''
    if dataset == 'bsds300':
        root_dir = download_bsds300(data_dir)

    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(image_size, upscale_factor)

    return DatasetFromFolder(test_dir,
                             is_gray=is_gray,
                             crop_size=crop_size,   # random crop
                             fliplr=False,          # random flip
                             input_transform=input_transform(crop_size, upscale_factor, normalize=normalize),
                             target_transform=target_transform(normalize=normalize))
