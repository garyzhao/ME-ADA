from __future__ import print_function, absolute_import, division

import os
import bz2
import scipy
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST, SVHN
from torchvision.datasets.utils import download_url
from common.utils import unfold_label, shuffle_data

_image_size = 32
_trans = transforms.Compose([
    transforms.Resize(_image_size),
    transforms.ToTensor()
])


def get_data_loaders():
    return [
        ['MNIST', 'SVHN', 'MNIST_M', 'SYN', 'USPS'],
        [load_mnist, load_svhn, load_mnist_m, load_syn, load_usps]
    ]


def load_mnist(root_dir, train=True):
    dataset = MNIST(root_dir, train=train, download=True, transform=_trans)
    images, labels = [], []

    for i in range(10000 if train else len(dataset)):
        image, label = dataset[i]
        images.append(image.expand(3, -1, -1).numpy())
        labels.append(label)
    return np.stack(images), np.array(labels)


def load_svhn(root_dir, train=True):
    split = 'train' if train else 'test'
    dataset = SVHN(os.path.join(root_dir, 'SVHN'), split=split, download=True, transform=_trans)
    images, labels = [], []

    for i in range(len(dataset)):
        image, label = dataset[i]
        images.append(image.numpy())
        labels.append(label)
    return np.stack(images), np.array(labels)


def load_usps(root_dir, train=True):
    split_list = {
        'train': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
            "usps.bz2", 'ec16c51db3855ca6c91edd34d0e9b197'
        ],
        'test': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
            "usps.t.bz2", '8ea070ee2aca1ac39742fdd1ef5ed118'
        ],
    }

    split = 'train' if train else 'test'
    url, filename, checksum = split_list[split]
    root = os.path.join(root_dir, 'USPS')
    full_path = os.path.join(root, filename)

    if not os.path.exists(full_path):
        download_url(url, root, filename, md5=checksum)

    with bz2.BZ2File(full_path) as fp:
        raw_data = [l.decode().split() for l in fp.readlines()]
        imgs = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
        imgs = np.asarray(imgs, dtype=np.float32).reshape((-1, 16, 16))
        imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
        targets = [int(d[0]) - 1 for d in raw_data]

    images, labels = [], []
    for img, target in zip(imgs, targets):
        img = Image.fromarray(img, mode='L')
        img = _trans(img)
        images.append(img.expand(3, -1, -1).numpy())
        labels.append(target)
    return np.stack(images), np.array(labels)


def load_syn(root_dir, train=True):
    split_list = {
        'train': "synth_train_32x32.mat",
        'test': "synth_test_32x32.mat"
    }

    split = 'train' if train else 'test'
    filename = split_list[split]
    full_path = os.path.join(root_dir, 'SYN', filename)

    raw_data = scipy.io.loadmat(full_path)
    imgs = np.transpose(raw_data['X'], [3, 0, 1, 2])
    images = []
    for img in imgs:
        img = Image.fromarray(img, mode='RGB')
        img = _trans(img)
        images.append(img.numpy())
    targets = raw_data['y'].reshape(-1)
    targets[np.where(targets == 10)] = 0

    return np.stack(images), targets.astype(np.int64)


def load_mnist_m(root_dir, train=True):
    split_list = {
        'train': [
            "mnist_m_train",
            "mnist_m_train_labels.txt"
        ],
        'test': [
            "mnist_m_test",
            "mnist_m_test_labels.txt"
        ],
    }

    split = 'train' if train else 'test'
    data_dir, filename = split_list[split]
    full_path = os.path.join(root_dir, 'MNIST_M', filename)
    data_dir = os.path.join(root_dir, 'MNIST_M', data_dir)
    with open(full_path) as f:
        lines = f.readlines()

    lines = [l.split('\n')[0] for l in lines]
    files = [l.split(' ')[0] for l in lines]
    labels = np.array([int(l.split(' ')[1]) for l in lines]).reshape(-1)
    images = []
    for img in files:
        img = Image.open(os.path.join(data_dir, img)).convert('RGB')
        img = _trans(img)
        images.append(img.numpy())

    return np.stack(images), labels


class BatchImageGenerator:
    def __init__(self, flags, stage, file_path, data_loader, b_unfold_label):

        if stage not in ['train', 'test']:
            assert ValueError('invalid stage!')

        self.configuration(flags, stage, file_path)
        self.load_data(data_loader, b_unfold_label)

    def configuration(self, flags, stage, file_path):
        self.batch_size = flags.batch_size
        self.current_index = 0
        self.file_path = file_path
        self.stage = stage

    def load_data(self, data_loader, b_unfold_label):
        file_path = self.file_path
        train = True if self.stage == 'train' else False
        self.images, self.labels = data_loader(file_path, train)

        if b_unfold_label:
            self.labels = unfold_label(labels=self.labels, classes=len(np.unique(self.labels)))
        assert len(self.images) == len(self.labels)

        self.file_num_train = len(self.labels)
        print('data num loaded:', self.file_num_train)

        if self.stage == 'train':
            self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)

    def get_images_labels_batch(self):
        images = []
        labels = []
        for index in range(self.batch_size):
            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.shuffle()

            images.append(self.images[self.current_index])
            labels.append(self.labels[self.current_index])

            self.current_index += 1

        images = np.stack(images)
        labels = np.stack(labels)

        return images, labels

    def shuffle(self):
        self.file_num_train = len(self.labels)
        self.current_index = 0
        self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)
