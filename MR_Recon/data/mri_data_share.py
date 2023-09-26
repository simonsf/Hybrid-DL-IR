import os 
import time
import math
import glob
import torch
import random
import numpy as np
import sigpy as sp
import SimpleITK as sitk

from utils import mymath
from random import randint
from torch.utils.data import Dataset
from .utils import load_config, read_imlist_file


class MRIDataset(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """
    def __init__(
        self, 
        opt, transform):
        """
        opt: dataset parameters in 
        """
        super(MRIDataset, self).__init__()
        self.opt = opt
        self.transform = transform
        self.FullList = read_imlist_file(opt['data_list'])

    def __len__(self):
        """ get the number of images in this dataset """
        return len(self.FullList)

    def __getitem__(self, index: int):
        # TODO output: input_tensor, mask_tensor, target_tensor

        case_path = self.FullList[index]
        case_path = case_path.strip('\r')

        # modify this if necessary
        full_sample_path = os.path.join(case_path, "full")
        mask_x_path = os.path.join(case_path, "X250_mask")
        channelnum = len(glob.glob(os.path.join(full_sample_path, '*')))

        realDir = full_sample_path + '/channel1/' + 'real'
         # get all slice and its number
        sliceList = glob.glob(realDir + '//slice*.mhd')
        totalZSize = len(sliceList)
        if totalZSize == 0:
            print(realDir)
        sliceInd = randint(1, totalZSize)  # random select slice

        # get mask
        mask_mhd_path = glob.glob(os.path.join(mask_x_path, '*.mhd'))[0]
        mask_mhd = sitk.ReadImage(mask_mhd_path)
        # 1,h,w -> hw
        mask_arr = np.squeeze(sitk.GetArrayFromImage(mask_mhd), axis=0)
        zero_loc_h, zero_loc_w = np.nonzero(mask_arr)
        h, w = mask_arr.shape[-2:]
        if np.sum(mask_arr[zero_loc_h[0], :]) == w:
            # 横条采样
            mask_direction = 'x'
        else:
            mask_direction = 'y'

        # mask sure size of height and width are even
        h_f = int(np.floor(h / 2) * 2)  
        w_f = int(np.floor(w / 2) * 2)
        k_sampling_mask = mask_arr[:h_f, :w_f]

        target = np.zeros((channelnum, h_f, w_f, 2))
        kspace = np.zeros((channelnum, h_f, w_f, 2))

        for ichannel in range(channelnum):
            try:
                full_real_slice = sitk.GetArrayFromImage(sitk.ReadImage(full_sample_path + '//channel' + str(ichannel + 1) + '//' + 'real' + '//' + 'slice' + str(sliceInd) + '.mhd'))
                full_imag_slice = sitk.GetArrayFromImage(sitk.ReadImage(full_sample_path + '//channel' + str(ichannel + 1) + '//' + 'imag' + '//' + 'slice' + str(sliceInd) + '.mhd'))
            except:
                print('full real path--->',full_sample_path + '//channel' + str(ichannel + 1) + '//' + 'real' + '//' + 'slice' + str(sliceInd) + '.mhd')
                continue

            full_real_numpy = full_real_slice[:, :h_f, :w_f]    # (h_f, w_f)
            full_img_numpy = full_imag_slice[:, :h_f, :w_f]    # (h_f, w_f)

            if math.isnan(full_real_numpy.mean()) or math.isinf(full_real_numpy.mean()):
                # print('%s  channelID:%d  sliceID:%d'%(sub_case_path, ichannel, sliceInd))
                print('failure--->',
                      full_sample_path + '//channel' + str(ichannel + 1) + '//' + 'real' + '//' + 'slice' + str(
                          sliceInd) + '.mhd')
                continue

            # concatenate the real part and imaging part of the input
            target_s = np.concatenate((full_real_numpy, full_img_numpy), axis=0).transpose(1,2,0) # (h_f, w_f, 2)

            # im_ful = mymath.c2r(im_ful) # [1, 2, height, width]
            im_sub = target_s * k_sampling_mask[:,:,None] + 0.0
            
            kspace[ichannel, :, :, :] = im_sub
            target[ichannel, :, :, :] = target_s

        target_t = mymath.r2c(target, axis=-1)
        target_t = mymath.ifft2c(target_t)
        target_t = mymath.c2r(target_t, axis=-1)
        target_rss = mymath.rss_complex(target_t)

        kspace_t = mymath.r2c(kspace, axis=-1)
        kspace_t = mymath.ifft2c(kspace_t)
        kspace_t = mymath.c2r(kspace_t, axis=-1)
        kspace_rss = mymath.rss_complex(kspace_t)

        # todo 增加图像域的
        img_sub = mymath.kspace2im_3d(kspace)
        k_sampling_mask = np.tile(k_sampling_mask, (2,1,1))
        if self.transform is None:
            sample = (img_sub, kspace, kspace_rss, k_sampling_mask, target_rss, mask_x_path)
        else:
            sample = self.transform(img_sub, kspace, kspace_rss, k_sampling_mask, mask_direction, target_rss, mask_x_path)

        return sample

