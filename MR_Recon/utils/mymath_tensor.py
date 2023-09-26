#-*-coding:utf-8-*-
# @Time    : 21-6-1 下午5:51
# @Author  : Lxy
# @Site    : 
# @File    : mymath_tensor.py
# @Software: PyCharm
import os
from typing import List, Optional
import torch

import numpy as np
import torch.fft as fft
import utils.mymath as mymath
from md.image3d.python.image3d import Image3d
from  md.image3d.python.image3d_io import write_image


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.sqrt((data**2).sum(dim=-1) + 1e-8)


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data**2).sum(dim) + 1e-8)


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim) + 1e-8)


def multi_channel_rss_vis(image, save_file_path='.vis'):
    """
        Args:
            image: Input 4D/5D tensor of shape `(1, in_chans, D, H, W)/(1, in_chans, H, W)`.
            save_file_path: "savepath/x.mha"
        Returns:
            Output tensor of shape `(D, H, W)`.
    """
    # image = image.cpu().detach().numpy()
    assert (len(image.shape) in [4, 5]), "nput 4D/5D tensor of shape `(1, in_chans, D, H, W)/(1, in_chans, D, H, W)`"
    if len(image.shape) == 5:
        D, H, W = image.shape[2:5]
        rss_vol = np.zeros((D, H, W))
        for d_i in range(D):
            rss_d_i = rss(image[:, :, d_i, :, :], dim=1)
            rss_vol[d_i] = rss_d_i.cpu().detach().numpy()
    
    else:
        H, W = image.shape[2:4]
        # rss_vol = np.zeros((1, H, W))
        rss_vol = rss(image[:, :, :, :], dim=1).cpu().detach().numpy()

    if not os.path.isdir(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))

    mha_image = Image3d()
    mha_image.from_numpy(rss_vol)

    write_image(mha_image, save_file_path, dtype=np.float32, compression=True)


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


def r2c(x):
    if x.shape[-1] != 2:
        print('Wrong! the shape of r2c is not 2')
    assert(x.shape[-1] == 2)

    x_real = x[..., 0]
    x_imag = x[..., 1]
    return torch.complex(x_real, x_imag)

def c2r(x):
    # if x.shape[0] != 1:
    #     # print('Wrong! the first shape of c2r is not 1')
    x_real = x.real
    x_imag = x.imag
    x_2c = torch.cat((x_real, x_imag), 0)
    return torch.unsqueeze(x_2c, 0)

def ifft2c(x):
    axes = (-2, -1)  # get last 2 axes
    res = fft.fftshift(fft.ifft2(fft.ifftshift(x, dim=axes), norm='ortho'), dim=axes)
    return res


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fft2c(x):
    axes = (-2, -1)  # get last 2 axes
    res = fft.fftshift(fft.fft2(fft.ifftshift(x, dim=axes), norm='ortho'), dim=axes)
    return res

def im2kspace_3d(input):

    [ch,s,w,h]=input.shape
    if ch ==2:
        input=input.transpose((1,0,2,3))

        input_c = r2c(input)
        k_c = fft2c(input_c)
        k_r = c2r(k_c)
        k_r=k_r.transpose((1,0,2,3))
    else:
        #   [24,1,224,192]
        input=input.reshape([int(ch/2), 2,w,h]).permute(0,2,3,1)
        input_c = r2c(input)
        k_c = fft2c(input_c)
        k_r = c2r(k_c)
        k_r = k_r.permute(1,0,2,3)
        # k_r=k_r.reshape([ch,s,w,h])

    return k_r


def undersample_crop_init(x,subx, mask, base=16):
    '''
    Undersample x. FFT2 will be applied to the last 2 axis
    Parameters
    ----------
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain

    base: matrxi size base


    Returns
    -------
    xu: array_like
        undersampled image in image domain. Note that it is complex valued

    x_fu: array_like
        undersampled data in k-space

    '''
    assert x.shape == mask.shape

    NRO, NPE = torch.tensor(x.shape[-2]), torch.tensor(x.shape[-1])
    NPE_f = torch.floor(NPE / base) * base
    NRO_f = torch.floor(NRO / base) * base

    PE_sc = torch.floor(NPE_f / 2)
    PE_bc = torch.floor(NPE / 2)
    PE_start = int(PE_bc - PE_sc)
    PE_end = int(PE_start + NPE_f)

    RO_sc = torch.floor(NRO_f / 2)
    RO_bc = torch.floor(NRO / 2)
    RO_start = int(RO_bc - RO_sc)
    RO_end = int(RO_start + NRO_f)

    #   修改流程
    x_tmp = x[:,RO_start:RO_end,PE_start:PE_end]
    subx_tmp = subx[:,RO_start:RO_end,PE_start:PE_end]
    mask_tmp = mask[:,RO_start:RO_end,PE_start:PE_end]

    im_full = ifft2c(x_tmp)
    k_und = mask_tmp * (subx_tmp)
    im_und = ifft2c(subx_tmp)
    return im_und, k_und,im_full,mask_tmp


def cal_nonzero_mean_std(sub_tensor, mask_tensor):
    """
    Calculating the mean and standard deviation of nonzero elemants in a tensor
    :param sub_tensor: subsample tensor to calculate the mean and std of nonzero elemants
    :param mask_tensor: subsample mask tensor as the same size as sub_tensor
    return mean std
    """
    assert sub_tensor.shape == mask_tensor.shape, "not the same size"
    mask_nonzero_bool = mask_tensor.ge(0.5)
    nonzero_tensor = torch.masked_select(sub_tensor, mask_nonzero_bool)
    sub_nonzero_mean = nonzero_tensor.mean()
    sub_nonzero_std = nonzero_tensor.std()

    return sub_nonzero_mean, sub_nonzero_std


def get_sub_subk_full(full_tensor, mask_tensor, self_norm=True, in_gpu=True, return_mean_std=False):
    """
    Prepare data for training
    :param full_tensor: k space full sample image -> [2*channel, 1, h, w]
    :param mask_tensor: k space mask volume -> [2*channel, 1, h, w]
    :param self_norm: standard normalization
    :param in_gpu: compute in GPU
    :return: subsample_vulume, k_subsample_vulume, full_sample_vulume
    """
    k_subsample = full_tensor * mask_tensor
    # fft 
    subsample_vulume = torch.zeros(full_tensor.shape)
    fullsample_vulume = torch.zeros(k_subsample.shape)

    if in_gpu:
        subsample_vulume = subsample_vulume.cuda()
        fullsample_vulume = fullsample_vulume.cuda()

    for ch_i in range(full_tensor.shape[0]//2):
        # to complex format for ifft along channel dimension 
        # fullsample k space 2 image domain
        data_full_ch_i = full_tensor[2 * ch_i:2 * (ch_i + 1), 0, :, :].permute(1,2,0)
        data_full_complex = torch.view_as_complex(data_full_ch_i.contiguous())
        data_full = ifft2c(data_full_complex)
        data_full = torch.view_as_real(data_full)
        fullsample_vulume[2 * ch_i:2 * (ch_i + 1), 0, :, :] = data_full.permute(2,0,1)

        # subsample k space 2 image domain
        data_sub_ch_i = k_subsample[2 * ch_i:2 * (ch_i + 1), 0, :, :].permute(1,2,0)
        data_sub_complex = torch.view_as_complex(data_sub_ch_i.contiguous())
        data_sub = ifft2c(data_sub_complex)
        data_sub = torch.view_as_real(data_sub)
        subsample_vulume[2 * ch_i:2 * (ch_i + 1), 0, :, :] = data_sub.permute(2,0,1)

    if self_norm:
        # normalization of aubsample image and full sample image, must in image domain
        # should normalize alone channel
        # wrong way to get the mean and std value
        # sub_tensor_mean = subsample_vulume.mean()
        # sub_tensor_std = subsample_vulume.std()
        # right way, exclude those point in zero
        sub_tensor_mean, sub_tensor_std = cal_nonzero_mean_std(k_subsample, mask_tensor)
        if sub_tensor_std == 0:
            sub_tensor_std = 1
        sub_tensor_norm = (subsample_vulume - sub_tensor_mean) / sub_tensor_std
        full_tensor_norm = (fullsample_vulume - sub_tensor_mean) / sub_tensor_std
        # copy
        fullsample_vulume = full_tensor_norm
        subsample_vulume = sub_tensor_norm

    # get the k space image from normalized subsample image
    for ch_i in range(full_tensor.shape[0]//2):
        data_sub_ch_i = sub_tensor_norm[2 * ch_i:2 * (ch_i + 1), 0, :, :].permute(1,2,0)
        data_sub_complex = torch.view_as_complex(data_sub_ch_i.contiguous())
        data_sub = fft2c(data_sub_complex)
        data_sub = torch.view_as_real(data_sub)
        k_subsample[2 * ch_i:2 * (ch_i + 1), 0, :, :] = data_sub.permute(2,0,1)

    if return_mean_std:
        return subsample_vulume, k_subsample, fullsample_vulume, sub_tensor_mean, sub_tensor_std
    else:
        return subsample_vulume, k_subsample, fullsample_vulume