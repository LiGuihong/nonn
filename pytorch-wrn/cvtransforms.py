""" OpenCV-based transforms
    Operate on np.ndarrays only, no PIL or torch dependency
"""
from __future__ import division
import math
import random
import numpy as np
import numbers
#import cv2


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the np.ndarray, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class Scale(object):
    """Rescales the input np.ndarray to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv.INTER_CUBIC
    """
    def __init__(self, size, interpolation=None):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(float(self.size) * h / w)
        else:
            oh = self.size
            ow = int(float(self.size) * w / h)
        return cv2.resize(img, dsize=(ow, oh),
                          interpolation=self.interpolation)


class CenterCrop(object):
    """Crops the given np.ndarray at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1+th, x1:x1+tw, :]


class Pad(object):
    """Pads the given np.ndarray on all sides with the given "pad" value."""

    def __init__(self, padding, borderType=None, borderValue=0):
        assert isinstance(padding, numbers.Number)
        self.padding = padding
        self.borderType = borderType
        self.borderValue = borderValue

    def __call__(self, img):
        if self.padding == 0:
            return img
        p = self.padding
        res = cv2.copyMakeBorder(img, p, p, p, p,
                                 borderType=self.borderType,
                                 value=self.borderValue)
        return res[:, :, np.newaxis] if np.ndim(res) == 2 else res


class RandomCrop(object):
    """Crops the given np.ndarray at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img[y1:y1+th, x1:x1+tw, :]


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img):
        if random.random() < 0.5:
            return cv2.flip(img, 1).reshape(img.shape)
        return img


class RandomSizedCrop(object):
    """Random crop the given np.ndarray to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_CUBIC
    """
    def __init__(self, size, interpolation=None):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4., 4. / 3.)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                x1 = random.randint(0, img.shape[1] - w)
                y1 = random.randint(0, img.shape[0] - h)

                img = img[y1:y1+h, x1:x1+w, :]
                assert img.shape[0] == h and img.shape[1] == w

                return cv2.resize(img, (self.size, self.size),
                                  interpolation=self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))

import argparse
import os
import json
import numpy as np
#import torch
#import torch.optim
#import torch.utils.data
import cvtransforms as T
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F

import torch.backends.cudnn as cudnn

def create_val_dataset(mode):
    convert = tnt.transform.compose([
        lambda x: x.astype(np.float32),
        T.Normalize([125.3, 123.0, 113.9], [63.0, 62.1, 66.7]),
        lambda x: x.transpose(2,0,1).astype(np.float32),
        torch.from_numpy,
    ])

    train_transform = tnt.transform.compose([
        T.RandomHorizontalFlip(),
        T.Pad(randomcrop_pad, cv2.BORDER_REFLECT),
        T.RandomCrop(32),
        convert,
    ])

    ds = getattr(datasets, dataset)(data_root, train=mode, download=True)
    smode = 'train' if mode else 'test'
    ds = tnt.dataset.TensorDataset([
        getattr(ds, smode+'_data'),
        getattr(ds, smode+'_labels')])
    #KK: I think you can do split dataset here since split dataset is also tnt.SplitDtaset(...)
    partitions={'trainNotUsed': 0.9, 'validation': 0.1}
    ds = tnt.dataset.ShuffleDataset(dataset=ds)
    ds.resample(seed=0)
    #print(ds[0])
    ds = tnt.dataset.SplitDataset(dataset=ds,partitions=partitions)
    
    ds.select('validation')
    print(len(ds))
    
    return ds.transform({0: convert})


# In[3]:


epoch_step1='[60,120,160]'
batchSize=1
depth=40
width=4.0
dataset='CIFAR10'
data_root='.'
mode=False
gpu_id='0'
cuda=True
alpha=0
beta=0
nthread=4
randomcrop_pad=4.0


model_id='resnet_40-4_model_notOnVal_saveLast_correct'
loadVal = False


# In[4]:

def create_dataset(mode):
    convert = tnt.transform.compose([
        lambda x: x.astype(np.float32),
        T.Normalize([125.3, 123.0, 113.9], [63.0, 62.1, 66.7]),
        lambda x: x.transpose(2,0,1).astype(np.float32),
        torch.from_numpy,
    ])

    train_transform = tnt.transform.compose([
        T.RandomHorizontalFlip(),
        T.Pad(randomcrop_pad, cv2.BORDER_REFLECT),
        T.RandomCrop(32),
        convert,
    ])

    ds = getattr(datasets, dataset)(data_root, train=mode, download=True)
    smode = 'train' if mode else 'test'
    ds = tnt.dataset.TensorDataset([
        getattr(ds, smode+'_data'),
        getattr(ds, smode+'_labels')])
    return ds.transform({0: train_transform if mode else convert})

def create_iterator(mode, loadVal, batchSize):
    if loadVal:
        ds = create_val_dataset(mode) #Mode must be true for loadVal = True since val set must come out of train set
    else:
        ds = create_dataset(mode)

    return ds.parallel(batch_size=batchSize, shuffle=mode,
                       num_workers=nthread, pin_memory=True)


# In[7]:


