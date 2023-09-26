# create by 
import torch
import torch.nn as nn

from utils import mymath_tensor
from .dc_layer import DataConsistencyInKspace, DataConsistencyInKspaceSDC


class CascadeCNN(nn.Module):
    """
    Model from paper 'A Deep Cascade of Convolutional Neural Networks for MR Image Reconstruction'
    """
    def __init__(self, net_meta, mid_chans=18, use_soft_dc=False, dc_lambda_k0=None):
        """
        :param net_meta: network for cascade [(model1, cascade1_n),....(modelm, cascadem_n),]
        :param mid_chans: number of middle channel in VGG like network setting
        :param use_soft_dc: number of filters
        :param dc_lambda_k0: kernel size
        """
        super(CascadeCNN, self).__init__()
        self.mid_chans = mid_chans
        # self.nd = nd
        # self.nf = nf
        # self.ks = ks
        # initialize net dictionary
        self.net = nn.ModuleList()

        # j = 0
        # cascade structure initialize, can cascade different net at a model
        for cascade_net, cascade_depth in net_meta:
            # Cascade layer
            for i in range(cascade_depth):
                # if isinstance(cascade_ner, UNet)
                self.net.append(cascade_net(self.mid_chans))

                # if i == (cascade_depth-1):
                # TODO: add unlearnable soft DC?
                # the last DC in cascade net as the learnable softDC
                if use_soft_dc:
                    self.soft_dc_w = torch.nn.Parameter(torch.FloatTensor(2), requires_grad=True)
                    self.soft_dc_w.data = torch.Tensor([dc_lambda_k0, 1-dc_lambda_k0])
                    self.net.append(DataConsistencyInKspaceSDC(norm='ortho', dc_lmda_w=self.soft_dc1_w))
                else:
                    self.net.append(DataConsistencyInKspace(norm='ortho'))
        
        # initialize model weight
        # self.init_weights()                

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, k, m, isdiffK=False):
        # forward pass
        temp = x
        for i, module_meta in enumerate(self.net):
            # DC layer 
            if isinstance(module_meta, (DataConsistencyInKspace, DataConsistencyInKspaceSDC)):
                # the channel of data is in_chn:24 (maybe change later)
                #  TODO: dc forward
                dc_out = torch.zeros((temp.shape)).cuda()
                # Note: dc is performed along channel dimension
                for ch_i in range(temp.shape[1]):
                    dc_temp = module_meta.perform(temp[:,ch_i,:,:,:].permute(0,3,1,2), k[:,ch_i,:,:,:].permute(0,3,1,2), m)
                    # dc_temp = module_meta.perform(temp[:,2*ch_i:2*(ch_i+1), :, :], k[:,2*ch_i:2*(ch_i+1), :, :], m[:,2*ch_i:2*(ch_i+1), :, :])
                    dc_out[:,ch_i,:,:] = dc_temp.permute(0,2,3,1)
                temp = dc_out
            else:
                # cascade net forward
                temp = module_meta(temp)

        temp = mymath_tensor.rss(mymath_tensor.complex_abs(temp), dim=1)

        return temp