import torch
import torch.nn as nn

from .ssim import SSIMLoss, L1SSIM


def create_loss(train_opt):
    loss_type = train_opt['loss_type']
    if loss_type == 'l1':
        criterion_pix = nn.L1Loss()
    elif loss_type == 'l2':
        criterion_pix = nn.MSELoss()
    elif loss_type == 'ssim':
        criterion_pix = SSIMLoss(win_size=train_opt["ssim_winsize"])
    elif loss_type == 'l1ssim':
        criterion_pix = L1SSIM(win_size=train_opt["ssim_winsize"], ssim_weight=train_opt["ssim_weight"])
    else:
        raise NotImplementedError('Loss type [%s] is not implemented!'%loss_type)
    
    return criterion_pix