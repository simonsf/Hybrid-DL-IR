"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.fft
from utils import mymath
from utils import mymath_tensor


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return torch.where(mask == 0, torch.Tensor([0]), data), mask


def fft2(data):
    """
    Powerful!!!!
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    #this code is used based on torch1.8
    
    data = torch.view_as_complex(data.contiguous())
    # data = mymath_tensor.r2c(data)
    # data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = mymath_tensor.fft2c(data)
    # data = torch.fft.fftshift(data, dim=(-2, -1))
    data = mymath_tensor.c2r(data)
    if len(data.shape) == 5:
        data = data.permute((0, 2, 3, 4, 1))
    elif len(data.shape) == 4:
        data = data.permute((0, 2, 3, 1))

    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    
    temp = mymath_tensor.r2c(data)
    temp = mymath_tensor.ifft2c(temp)
    data = mymath_tensor.c2r(temp)
    if len(data.shape) == 5:
        data = data.permute((0, 2, 3, 4, 1))
    elif len(data.shape) == 4:
        data = data.permute((0, 2, 3, 1))

    return data

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return ((data ** 2).sum(dim=-1) + 1e-8).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim) + 1e-8)


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

# import torch
# import numpy as np

def mask_center(x: torch.Tensor, mask_direction: str, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    # x.shape: [batch_size, num_coils, h, w, 2]
    mask = torch.zeros_like(x)
    if mask_direction == "y":
        mask[:, :, :, mask_from:mask_to, :] = x[:, :, :, mask_from:mask_to, :]
    else:
        mask[:, :, mask_from:mask_to, :, :] = x[:, :, mask_from:mask_to, :, :]
    return mask
    

def batched_mask_center(
    x: torch.Tensor, mask_direction: str, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, mask_direction, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        sub_img: img in image space
        masked_kspace: k-space after applying sampling mask.
        masked_kspace_rss: root sum of square of mask_kaspce transformed into image space.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """
    sub_img: torch.Tensor
    masked_kspace: torch.Tensor
    masked_kspace_rss: torch.Tensor
    mask: torch.Tensor
    mask_direction: str
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    # slice_num: int
    max_value: float
    # crop_size: Tuple[int, int]


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """
    def __init__(self, use_seed: bool = False):
        """
        Args:
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.use_seed = use_seed

    def __call__(
        self,
        sub_img: np.ndarray,
        kspace: np.ndarray,
        kspace_rss: np.ndarray,
        mask: np.ndarray,
        mask_direction: str,
        target: Optional[np.ndarray],
        # attrs: Dict,
        fname: str,
        # slice_num: int,
    ) -> VarNetSample:
        """
            Args:
                kspace: Input k-space of shape (num_coils, rows, cols, 2) for
                    multi-coil data.
                mask: Mask from the test dataset. (H, W)
                target: Target image.
                attrs: Acquisition related information stored in the HDF5 object.
                fname: File name.
                slice_num: Serial number of the slice.

            Returns:
                A VarNetSample with the masked k-space, sampling mask, target
                image, the filename, the slice number, the maximum image value
                (from target), the target crop size, and the number of low
                frequency lines sampled.
            """
        if target is not None:
            target_torch = to_tensor(target)
            # data_full_ch1 = target_torch[0, :, :, :] # (h,w,2)
            # data_full_complex = torch.view_as_complex(data_full_ch1.contiguous())
            # data_full =  mymath_tensor.ifft2c(data_full_complex)
            # data_full = torch.view_as_real(data_full)   # (h,w,2)
            # data_full_rss = mymath_tensor.rss_complex(data_full)
            max_value = target_torch.max()

        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        sub_img_torch = to_tensor(sub_img)
        kspace_torch = to_tensor(kspace)
        kspace_torch_rss = to_tensor(kspace_rss)

        # TODO: apply mask
        masked_kspace = kspace_torch
        # shape = np.array(kspace_torch.shape)
        # num_rows, num_cols = shape[-3:-1]    # num_rows:h  num_cols:w
        # mask_shape = [1] * len(shape)
        # if mask_direction == 'x':
        #     # 横条采样
        #     mask_shape[-2] = 1
        #     mask_shape[-3] = num_rows
        #     mask = mask[:,0]   # 全部h
        # else:
        #     # 竖条采样
        #     mask_shape[-2] = num_cols
        #     mask_shape[-3] = 1
        #     mask = mask[0,:]

        # mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        # mask_torch = mask_torch.reshape(*mask_shape)
        mask_torch = torch.from_numpy(mask)
        sample = VarNetSample(
                sub_img=sub_img_torch,
                masked_kspace=masked_kspace,
                masked_kspace_rss=kspace_torch_rss,
                mask=mask_torch,
                mask_direction=mask_direction,
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                # slice_num=slice_num,
                max_value=max_value,
            )

        return sample


if __name__ == '__main__':

    a=torch.randn(4,3,2)
    b=fft2(a)
    c=np.transpose(a.numpy(),(2,0,1))
    cc=mymath.r2c(np.expand_dims(c, 0))
    d=mymath.fft2c(cc)
    e=mymath.c2r(d)
    print ('end')