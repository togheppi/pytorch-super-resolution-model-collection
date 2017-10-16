from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from dataset import *


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


def get_training_set(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    train_dir = []
    for dataset in datasets:
        if dataset == 'bsds300':
            root_dir = download_bsds300(data_dir)
            train_dir.append(join(root_dir, "train"))
        elif dataset == 'DIV2K':
            train_dir.append(join(data_dir, dataset, 'DIV2K_train_LR_bicubic/X4'))
        else:
            train_dir.append(join(data_dir, dataset))

    return TrainDatasetFromFolder(train_dir,
                                  is_gray=is_gray,
                                  random_scale=True,    # random scaling
                                  crop_size=crop_size,  # random crop
                                  rotate=True,          # random rotate
                                  fliplr=True,          # random flip
                                  fliptb=True,
                                  scale_factor=scale_factor)


def get_test_set(data_dir, dataset, scale_factor, is_gray=False):
    if dataset == 'bsds300':
        root_dir = download_bsds300(data_dir)
        test_dir = join(root_dir, "test")
    elif dataset == 'DIV2K':
        test_dir = join(data_dir, dataset, 'DIV2K_test_LR_bicubic/X4')
    else:
        test_dir = join(data_dir, dataset)

    return TestDatasetFromFolder(test_dir,
                                 is_gray=is_gray,
                                 scale_factor=scale_factor)
