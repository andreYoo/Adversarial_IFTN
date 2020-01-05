import os
import numpy as np
import scipy.misc
import tensorflow as tf
from src import utils,dataset
import cv2
import pdb

if __name__ == '__main__':
    img_path = '/media/neumann/Warehouse/crack_DB/pavement crack datasets/CFD/cfd_image'
    gt_path = '/media/neumann/Warehouse/crack_DB/pavement crack datasets/CFD/seg_gt'
    utils.ext_trainset(img_path,gt_path)

