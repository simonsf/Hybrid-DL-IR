import torch
import torch.nn as nn

from data import transform

def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
        # out = (1 - mask) * k +mask
    return out


def data_consistency_alpha(k, k0, mask, alpha):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    # out = (1 - mask) * k + mask * k0
    out = (1 - mask) * k + (1-alpha)*mask*k0 + alpha*mask*k
    return out


class DataConsistencyInKspace(nn.Module):
    """
    utilize hard data consitency
    """
    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        return: x-res - return image in image domain
        """
        if x.dim() == 4: # input is 2D
            x1 = x.permute(0, 2, 3, 1)
            k0 = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5: # input is 3D
            [b,ch,s,w,h]=x.size()
            x1 = x.reshape((b,int(ch/2),2,w,h)).permute(0,1,3,4,2)
            k0 = k0.reshape((b,int(ch/2),2,w,h)).permute(0,1,3,4,2)
            mask = mask.reshape((b,int(ch/2),2,w,h)).permute(0,1,3,4,2)

        k = transform.fft2(x1)
        # k = transform_wxl.fft2(x1)
        # k = torch.fft(x, 2, normalized=self.normalized)
        if self.noise_lvl==None:
            out = data_consistency(k, k0, mask, self.noise_lvl)
        else:
            #在这里使用self.noise_lvl代替alpha
            out = data_consistency_alpha(k, k0, mask, self.noise_lvl)
        x_res=transform.ifft2(out)   #[1,12,w,h,2]
        # x_res=transform_wxl.ifft2(out)
        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 1, 4, 2, 3).reshape((b, ch, s, w, h))
        return x_res
    

class DataConsistencyInKspaceSDC(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None, norm='ortho', dc_lmda_w=None):
        super(DataConsistencyInKspaceSDC, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl
        self.dc_lmda_w = dc_lmda_w

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def image_to_kspace(self, x):
        if x.dim() == 4:    # input is 2D
            x = x.permute(0, 2, 3, 1)
        elif x.dim() == 5:  # input is 3D
            x = x.permute(0, 4, 2, 3, 1)
        k = transform.fft2(x)
        return k

    def perform_kspace(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        [b,ch,s,w,h]=x.size()
        if x.dim() == 4: # input is 2D
            x    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5: # input is 3D
            x = x.reshape((b,int(ch/2),2,w,h)).permute(0,1,3,4,2)
            k0 = k0.reshape((b,int(ch/2),2,w,h)).permute(0,1,3,4,2)
            mask = mask.reshape((b,int(ch/2),2,w,h)).permute(0,1,3,4,2)

        k=transform.fft2(x)
        # k = torch.fft(x, 2, normalized=self.normalized)
        # TODO: 增加lambda返回的计算？
        if self.dc_lmda:
            out = data_consistency_deepCascade(k, k0, mask, self.dc_lmda)
        elif self.noise_lvl==None:
            out = data_consistency(k, k0, mask, self.noise_lvl)
        else:
            #在这里使用self.noise_lvl代替alpha
            out = data_consistency_alpha(k, k0, mask, self.noise_lvl)
        # x_res = torch.ifft(out, 2, normalized=self.normalized)
        return out

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        return: x-res - return image in image domain
        """
        if x.dim() == 4: # input is 2D
            x1    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
            # ori_x = ori_x.permute(0,2,3,1)
        elif x.dim() == 5: # input is 3D
            [b,ch,s,w,h]=x.size()
            x1 = x.reshape((b,int(ch/2),2,w,h)).permute(0,1,3,4,2)
            k0 = k0.reshape((b,int(ch/2),2,w,h)).permute(0,1,3,4,2)
            mask = mask.reshape((b,int(ch/2),2,w,h)).permute(0,1,3,4,2)
        k = transform.fft2(x1)
        # k = torch.fft(x, 2, normalized=self.normalized)
        if self.dc_lmda_w is not None:
            # print('use double soft dc!')
            out = data_consistency_lmda_softmax(k, k0, mask, self.dc_lmda_w)
        elif self.noise_lvl==None:
            # print('use hard dc!')
            out = data_consistency(k, k0, mask, self.noise_lvl)
        else:
            #在这里使用self.noise_lvl代替alpha
            out = data_consistency_alpha(k, k0, mask, self.noise_lvl)
        x_res=transform.ifft2(out)   #[1,12,w,h,2]
        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 1, 4, 2, 3).reshape((b, ch, s, w, h))
        return x_res


    def perform_3d_alpha(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        if x.dim() == 4: # input is 2D
            x1    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
            if self.noise_lvl is not None:
                noise_lvl = self.noise_lvl.permute(0, 2, 3, 1)
        k = transform.fft2(x1)
        if self.noise_lvl is None:
            out = data_consistency(k, k0, mask, self.noise_lvl)
        else:
            out = data_consistency_alpha(k, k0, mask, noise_lvl)
        x_res=transform.ifft2(out)   #[1,12,w,h,2]
        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        return x_res

    def perform_alpha_gamma(self, x, k0, mask, gamma):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
            gamma = gamma.permute(0,2,3,1)
        elif x.dim() == 5: # input is 3D
            x    = x.permute(0, 4, 2, 3, 1)
            k0   = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)
            gamma = gamma.permute(0, 4, 2, 3, 1)

        k=transform.fft2(x)
        # k = torch.fft(x, 2, normalized=self.normalized)
        if self.noise_lvl is None:
            out = data_consistency(k, k0, mask, self.noise_lvl)
        else:
            #在这里使用self.noise_lvl代替alpha
            out = data_consistency_alpha_gamma(k, k0, mask, self.noise_lvl, gamma)
        # x_res = torch.ifft(out, 2, normalized=self.normalized)
        x_res=transform.ifft2(out)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res