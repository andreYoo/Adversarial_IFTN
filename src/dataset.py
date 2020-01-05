import os
import numpy as np
import scipy.misc
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import cv2
import pdb





class CFDDataset(object):
    def __init__(self, img_path, gt_path):
        self.img_path = img_path
        self.get_path = gt_path

    def sync_list(self):
        sync_path1 = []
        sync_path2 = []
        for _tmp in self.img_path:
            _str = _tmp.split('.')[0]
            if (_str+'.png') in self.get_path:
                sync_path1.append(_tmp)
                sync_path2.append(_str+'.png')
            else:
                continue
        return sync_path1, sync_path2



class imgdataset(Dataset):
    def __init__(self, imgset,freqset,transform=None):
        self.imgs = imgset
        self.freqs = freqset
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
        # self._load_mnist()

    def __getitem__(self, idx):
        img = self.imgs[idx]
        frqd = self.freqs[idx]
        if self.transform:
            img = self.transform(img)
            frqd = self.transform(frqd)
        return img,frqd
