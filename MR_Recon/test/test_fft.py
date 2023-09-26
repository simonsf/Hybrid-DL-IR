# create on 
from cgi import test
import os
import sys
import torch
sys.path.insert(0, '.')
import argparse

import numpy as np
import matplotlib.pyplot as plt
import md.image3d.python.image3d_io as cio

from utils import mymath


def test_fft_ifft(opts):
    """
    Function for testing the fft and ifft procedure
    """
    data_channel_path = opts.test_img_path
    real_path = os.path.join(data_channel_path, 'real/slice10.mhd')
    imag_path = os.path.join(data_channel_path, 'imag/slice10.mhd')
    full_real_slice = cio.read_image(real_path, dtype=np.float32)
    full_imag_slice = cio.read_image(imag_path, dtype=np.float32)
    full_real_numpy = full_real_slice.to_numpy()
    full_img_numpy = full_imag_slice.to_numpy()
    # concatenate the real part and imaging part of the input
    target = np.concatenate((full_real_numpy, full_img_numpy), axis=0) # (h, w, 2)
    target = target[None,:]     # (n, h, w, 2)

    # test numpy fft 
    target_c_np = mymath.r2c(target)    # (n, h, w)
    target_ifft_c = mymath.ifft2c(target_c_np)
    target_r_np = mymath.c2r(target_ifft_c) # (n,2,h,w)
    ref_rss = mymath.rss_complex(target_r_np.transpose(0,2,3,1)) # (nc, h, w, 2) -> (h, w)
    plt.imsave('.visualization/vis_target_rss.png', ref_rss, cmap='gray')

    # test torch.fft
    target_torch = torch.from_numpy(target.transpose(0,2,3,1)).contiguous()  # (n, h, w, 2)
    target_torch_c = torch.view_as_complex(target_torch)    # (n, h, w)
    target_torch_ifft_c = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(target_torch_c))) # (n, h, w)
    target_torch_r = torch.view_as_real(target_torch_ifft_c) # (n, h, w, 2)
    target_t2n_r = target_torch_r.numpy()
    ref_rss_t2n = mymath.rss_complex(target_t2n_r) # (nc, h, w, 2) -> (nc, h, w)
    plt.imsave('.visualization/vis_target_t2n_rss.png', ref_rss_t2n, cmap='gray')


def args_parse():
    parser = argparse.ArgumentParser(description='Test Deep JSense .')
    parser.add_argument('--test_img_path', type=str, 
        default='/data/data/xinglie01/ACS_2D_MC/ACS_2D_15T/oneToOne/lmAcc_ppa/crop1.5/head/head_t2_fse_flair_tra_fs_g1_2/mhd/UID_1563157687_r1+r4/full/channel1/',
        help='fft test data dir')
    opts = parser.parse_args()
    return opts


if __name__ == '__main__':
    opts = args_parse()
    test_fft_ifft(opts)
    print('Test Done!')