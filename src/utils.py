import matplotlib
matplotlib.use('qt4agg')
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
import os
import numpy as np
from numpy import inf
import scipy.misc
import cv2
import pdb
from src import dataset
import h5py




def show_img_for_dedug(img):
    cv2.imshow('temp',img)
    cv2.waitKey(0)

def extract_img_samples(p1,p2,img_path,gt_path,temp_size=[80,80],pick_size = 128):
    """
    :param p1: pathlist for synchronized path for image
    :param p2: pathlist for synchronized path for gt
    :return: None
    """
    train_img_dataset = []
    train_frq_dataset = []

    if len(p1)!=len(p2):
        print("The lengths of lists should have to be same - Check the lists")
        exit()
    _shape = min(np.shape(cv2.imread(img_path + '/' + p1[0]))[0:1])
    cnt = 0.
    num_sample = 0
    length = float(len(p1))
    for _t1, _t2 in zip(p1,p2):
        print('Generating dataset [%.2f] - Processing file - %s'%(float(cnt/length)*100.,_t1))
        img = cv2.imread(img_path+'/'+_t1)
        gt = cv2.imread(gt_path+'/'+_t2)
        random_index = np.random.randint(_shape-temp_size[0],size=pick_size)
        for _idx  in random_index:
            if [255,255,255] not in gt[_idx:_idx+temp_size[0],_idx:_idx+temp_size[0]]:
                cropped_img = img[_idx:_idx+temp_size[0],_idx:_idx+temp_size[0]]
                #show_img_for_dedug(cropped_img)
                train_img_dataset.append(cropped_img)
                num_sample+=1
                #tmp_code for visualization
                f = np.fft.fft2(cropped_img)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 0.1*np.log(np.abs(fshift))

                #thrrsholding for -inf values manuually.
                magnitude_spectrum[magnitude_spectrum==-inf] = -9.99
                train_frq_dataset.append(magnitude_spectrum)


        cnt += 1
    print('%d samples are generated'%(num_sample))
    return train_img_dataset,train_frq_dataset

def ext_trainset(img_path,gt_path,db_type='CFD',is_save=True):
    """
    :param img_path: the path for image directory  - should be provided by image file such as .jpg, .png, etc.
    :param gt_path: the path for gt directory - should be provided by image file such as .jpg, .png, etc.
    :return:
    """
    #get filepath_list
    imgfile_list = os.listdir(img_path)
    gtfile_list = os.listdir(gt_path)

    #synchronization
    if db_type=='CFD':
        db = dataset.CFDDataset(imgfile_list,gtfile_list)
        sync_imglist, sync_gtlist= db.sync_list()

        train_img_samples, train_freq_samples = extract_img_samples(sync_imglist, sync_gtlist,img_path,gt_path)
        if is_save==True:
            hf = h5py.File('CDF_Cropped_dataset.h5','w')
            hf.create_dataset('image',data=train_img_samples)
            hf.create_dataset('frequency',data=train_freq_samples)
            hf.close()


    '''    
    if db_type=='CRACK500':
        train_img_sample = dataset.CRACK500Dataset()
        
    if db_type=='Cracktree200':
        train_img_sample = dataset.CFDDataset()
        
    if db_type=='CAPS384':
        train_img_sample = dataset.CFDDataset()
    '''
    return train_img_samples,train_freq_samples



