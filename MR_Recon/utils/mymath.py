import os
import torch
import numpy as np

from utils import mymath_tensor 
from numpy.lib.stride_tricks import as_strided
from md.image3d.python.image3d_io import read_image
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift


def fftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(fft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def ifftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(ifft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def fft2c(x):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    axes = (-2, -1)  # get last 2 axes
    # res = fft2(x, norm='ortho')
    res = fftshift(fft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res


def ifft2c(x):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res


def  r2c(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data

    x: ndarray
        input data (n, 2, h, w)
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    if axis < 0: axis = x.ndim + axis
    ctype = np.complex64 if x.dtype == np.float32 else np.complex128

    if axis < len(shape):
        newshape = tuple([i for i in range(0, axis)]) \
                   + tuple([i for i in range(axis+1, x.ndim)]) + (axis,)

        x = x.transpose(newshape)

    # trans x to complex, so its last dimension [1, h, w, 2] -> [1, h, w, 1]
    x = np.ascontiguousarray(x).view(dtype=ctype)
    # x = np.ascontiguousarray(x)
    # x = x.view(dtype=ctype)
    return x.reshape(x.shape[:-1])


def c2r(x,mask=False, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    if mask:  # Hacky solution
        x = x*(1+1j)

    shape = x.shape
    dtype = np.float32 if x.dtype == np.complex64 else np.float64

    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)])
        x = x.transpose(newshape)

    return x


def padding_full_sample_mask(x_full, mask, base=16):
    """
    x_full : [1, height, width], [1, 556, 320]-->[1, 560, 320]
    """
    assert mask.shape == x_full.shape
    if x_full.dtype != np.complex64:
        x_full=r2c(x_full,0)
        mask=mask[0]

    NRO, NPE =  x_full.shape[-2], x_full.shape[-1]
    NPE_f = np.ceil(NPE / base) * base # 320
    NRO_f = np.ceil(NRO / base) * base # 560
    x_full_tmp=np.zeros((x_full.shape[0], int(NRO_f), int(NPE_f)),dtype=np.complex64)
    mask_tmp=np.zeros((x_full.shape[0], int(NRO_f), int(NPE_f)),dtype=np.float32)

    PE_bc = np.floor(NPE_f / 2) # 160
    PE_sc = np.floor(NPE / 2) # 160
    PE_start = int(PE_bc - PE_sc) # 0
    PE_end = int(PE_start + NPE) # 320

    RO_bc = np.floor(NRO_f / 2) # 280
    RO_sc = np.floor(NRO / 2) # 278
    RO_start = int(RO_bc - RO_sc) # 2
    RO_end = int(RO_start + NRO) # 558

    x_full_tmp[:,RO_start:RO_end,PE_start:PE_end] = x_full
    mask_tmp[:,RO_start:RO_end,PE_start:PE_end] = mask

    # im_und = ifft2c(x_full_tmp)
    return x_full_tmp, mask_tmp


def padding_full_sample(x_full, base=16):
    """
    x_full : [1, height, width], [1, 556, 320]-->[1, 560, 320]
    """
    # assert mask.shape == x_full.shape
    if x_full.dtype != np.complex64:
        x_full=r2c(x_full,0)
        # mask=mask[0]

    NRO, NPE =  x_full.shape[-2], x_full.shape[-1]
    NPE_f = np.ceil(NPE / base) * base # 320
    NRO_f = np.ceil(NRO / base) * base # 560
    x_full_tmp=np.zeros((x_full.shape[0], int(NRO_f), int(NPE_f)),dtype=np.complex64)
    # mask_tmp=np.zeros((x_full.shape[0], int(NRO_f), int(NPE_f)),dtype=np.float32)

    PE_bc = np.floor(NPE_f / 2) # 160
    PE_sc = np.floor(NPE / 2) # 160
    PE_start = int(PE_bc - PE_sc) # 0
    PE_end = int(PE_start + NPE) # 320

    RO_bc = np.floor(NRO_f / 2) # 280
    RO_sc = np.floor(NRO / 2) # 278
    RO_start = int(RO_bc - RO_sc) # 2
    RO_end = int(RO_start + NRO) # 558

    x_full_tmp[:,RO_start:RO_end,PE_start:PE_end] = x_full
    # mask_tmp[:,RO_start:RO_end,PE_start:PE_end] = mask

    # im_und = ifft2c(x_full_tmp)
    return x_full_tmp


def padding_mask(mask, base=16):
    """
    mask : [1, height, width], [1, 556, 320]-->[1, 560, 320]
    """
    # # assert mask.shape == x_full.shape
    # if x_full.dtype != np.complex64:
    #     x_full=r2c(x_full,0)
    #     # mask=mask[0]

    NRO, NPE =  mask.shape[-2], mask.shape[-1]
    NPE_f = np.ceil(NPE / base) * base # 320
    NRO_f = np.ceil(NRO / base) * base # 560
    # mask_tmp=np.zeros((mask.shape[0], int(NRO_f), int(NPE_f)),dtype=np.complex64)
    mask_tmp=np.zeros((mask.shape[0], int(NRO_f), int(NPE_f)), dtype=np.float32)

    PE_bc = np.floor(NPE_f / 2) # 160
    PE_sc = np.floor(NPE / 2) # 160
    PE_start = int(PE_bc - PE_sc) # 0
    PE_end = int(PE_start + NPE) # 320

    RO_bc = np.floor(NRO_f / 2) # 280
    RO_sc = np.floor(NRO / 2) # 278
    RO_start = int(RO_bc - RO_sc) # 2
    RO_end = int(RO_start + NRO) # 558

    # x_full_tmp[:,RO_start:RO_end,PE_start:PE_end] = x_full
    mask_tmp[:,RO_start:RO_end,PE_start:PE_end] = mask

    # im_und = ifft2c(x_full_tmp)
    return mask_tmp


def complex_abs_sq(data):
    """
    Compute the squared absolute value of a complex numpy array.

    Args:
        data: A complex valued numpy array, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Numpy assay does not have separate complex dim.")

    return np.sum((data**2), axis=-1)


def rss_complex(data, axis=0):
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.
    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input numpy array
        axis: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return np.sqrt(np.sum(complex_abs_sq(data), axis=0) + 1e-8)


def im2kspace_3d(input):
    """
    change real image numpy to real kspace
    :param: input in image domain [2*channel,1,h,w]
    :return: kspace data [1,2,h,w]
    """
    [ch,s,w,h]=input.shape
    # [s,ch,w,h]=input.shape

    if ch ==2:
        input=input.transpose((1,0,2,3))

        input_c = r2c(input)
        k_c = fft2c(input_c)
        k_r = c2r(k_c)
        k_r=k_r.transpose((1,0,2,3))
    else:
        input=input.reshape([int(ch/2),2,w,h])
        input_c = r2c(input) # [ch//2, w, h]
        k_c = fft2c(input_c)
        k_r = c2r(k_c)
        k_r=k_r.reshape([ch,s,w,h])

    return k_r



def kspace2im_3d(input):
    """
    change real image numpy to real kspace at slice level
    :param: input in k space [channel,h,w,2]
    :return: image domain [channel,h,w,2]
    """
    ch,w,h,_=input.shape
    if ch ==2:
        input=input.transpose((0,3,1,2))

        input_c = r2c(input)
        k_c = ifft2c(input_c)
        k_r = c2r(k_c)
        k_r=k_r.transpose((1,2,3,0))
    else:
        # input=input.reshape([int(ch/2),2,w,h])
        input = input.transpose((0,3,1,2))
        input_c = r2c(input) # [ch//2, w, h]
        k_c = ifft2c(input_c)
        k_r = c2r(k_c)
        k_r=k_r.transpose((0,2,3,1))

    return k_r