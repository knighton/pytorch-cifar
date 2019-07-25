import numpy as np
import os
import pickle
from PIL import Image
import tarfile
from time import time
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

from .downloading import download_if_missing


CIFAR10_REMOTE = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR100_REMOTE = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


def load_cifar10_samples(tar, is_train, verbose):
    if verbose == 2:
        bar = tqdm(total=5, leave=False)

    train = []
    val = None
    for info in tar.getmembers():
        if not info.isreg():
            continue

        if info.path.startswith('cifar-10-batches-py/data_batch_'):
            got_is_train = True
        elif info.path == 'cifar-10-batches-py/test_batch':
            got_is_train = False
        else:
            continue

        if got_is_train != is_train:
            continue

        data = tar.extractfile(info).read()
        obj = pickle.loads(data, encoding='bytes')
        x = obj[b'data']
        x = np.array(x, np.uint8)
        x = x.reshape(-1, 3, 32, 32)
        y = obj[b'labels']
        y = np.array(y, np.uint8)

        if is_train:
            train.append((info.path, x, y))
        else:
            val = x, y

        if verbose == 2:
            bar.update(1)

    if verbose == 2:
        bar.close()

    if is_train:
        train.sort()
        _, x, y = zip(*train)
        x = np.concatenate(x, 0)
        y = np.concatenate(y, 0)
        return x, y
    else:
        return val


def load_cifar10_class_names(tar):
    path = 'cifar-10-batches-py/batches.meta'
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    labels = obj[b'label_names']
    return list(map(lambda s: s.decode('utf-8'), labels))


def load_cifar10_split(root, is_train, verbose):
    local = os.path.join(root, os.path.basename(CIFAR10_REMOTE))
    download_if_missing(CIFAR10_REMOTE, local, verbose)
    tar = tarfile.open(local, 'r:gz')
    data, targets = load_cifar10_samples(tar, is_train, verbose)
    class_names = load_cifar10_class_names(tar)
    tar.close()
    return data, targets, class_names


def load_cifar100_samples(tar, num_classes, split):
    path = 'cifar-100-python/%s' % split
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    x = obj[b'data']
    x = np.array(x, np.uint8)
    x = x.reshape(-1, 3, 32, 32)
    if num_classes == 20:
        key = b'coarse_labels'
    elif num_classes == 100:
        key = b'fine_labels'
    else:
        assert False
    y = obj[key]
    y = np.array(y, np.uint8)
    return x, y


def load_cifar100_class_names(tar, num_classes):
    info = tar.getmember('cifar-100-python/meta')
    data = tar.extractfile(info).read()
    obj = pickle.loads(data, encoding='bytes')
    if num_classes == 20:
        key = b'coarse_label_names'
    elif num_classes == 100:
        key = b'fine_label_names'
    else:
        assert False
    labels = obj[key]
    return list(map(lambda s: s.decode('utf-8'), labels))


def load_cifar100_split(num_classes, root, is_train, verbose):
    local = os.path.join(root, os.path.basename(CIFAR100_REMOTE))
    download_if_missing(CIFAR100_REMOTE, local, verbose)
    tar = tarfile.open(local, 'r:gz')
    split = 'train' if is_train else 'test'
    data, targets = load_cifar100_samples(tar, num_classes, split)
    class_names = load_cifar100_class_names(tar, num_classes)
    tar.close()
    return data, targets, class_names


def load_split(num_classes, root, is_train, verbose):
    if num_classes == 10:
        return load_cifar10_split(root, is_train, verbose)
    elif num_classes == 20:
        return load_cifar100_split(20, root, is_train, verbose)
    elif num_classes == 100:
        return load_cifar100_split(100, root, is_train, verbose)
    else:
        assert False


class CIFARX(VisionDataset):
    def __init__(self, num_classes, root, train=True, transform=None,
                 target_transform=None, download=False, verbose=0):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.train = train  # Training or test set.
        self.data, self.targets, self.classes = \
            load_split(num_classes, root, train, verbose)
        self.data = self.data.transpose(0, 2, 3, 1)  # Convert to NHWC.
        self.class_to_idx = {c: i for (i, c) in enumerate(self.classes)}

    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        target = self.targets[index]
        target = target.astype(np.int64)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR10(CIFARX):
    def __init__(self, *args, **kwargs):
        super().__init__(10, *args, **kwargs)


class CIFAR20(CIFARX):
    def __init__(self, *args, **kwargs):
        super().__init__(20, *args, **kwargs)


class CIFAR100(CIFARX):
    def __init__(self, *args, **kwargs):
        super().__init__(100, *args, **kwargs)
