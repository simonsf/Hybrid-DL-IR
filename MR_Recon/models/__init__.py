# create by wangerxiao on 20221013
import functools

import torch
import torch.nn as nn
from torch.nn import init

from .rep_vgg import RepVGG, NormRepVGG


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        m.weight.data *= scale
        nn.init.zeros_(m.bias)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

####################
# define network
####################

def create_model(opt):
    if opt['mode'] == 'sr':
        net = define_net(opt['networks'])
        return net
    else:
        raise NotImplementedError("The mode [%s] of networks is not recognized." % opt['mode'])

# choose one network
def define_net(opt):
    which_model = opt['which_model'].upper()
    print('===> Building network [%s]...'%which_model)

    if which_model == 'E2E_VARNET':
        if opt['which_model'].upper() == "E2E_VARNET":
            # from .adaptive_varnet import AdaptiveVarNet
            from .varnet import VarNet 
        # net = AdaptiveVarNet(num_cascades=opt['num_cascades'], sens_chans=opt['sens_chans'], sens_pools=opt['sens_pools'],
        #                             chans=opt['num_chans'], pools=opt['num_pools'], num_sense_lines=opt['num_sense_lines'],
        #                             hard_dc=opt['hard_dc'], dc_mode=opt['dc_mode'], sparse_dc_gradients=opt['sparse_dc_gradients'])
        net = VarNet(num_cascades=opt['num_cascades'], sens_chans=opt['sens_chans'], sens_pools=opt['sens_pools'],
                            chans=opt['num_chans'], pools=opt['num_pools'], mask_center=opt['mask_center'])
        
    elif which_model == "CASCADECNN":
        from .cascadenet import CascadeCNN
        net = CascadeCNN([(NormRepVGG, opt['num_cascades'])], mid_chans=opt['num_chans'])

    else:
        raise NotImplementedError("Network [%s] is not recognized." % which_model)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net).cuda()
    else:
        net.cuda()
    return net
