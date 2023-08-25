import torch
import torch.nn as nn
import torch.nn.functional as F


class dense_layer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate):
        super(dense_layer, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.act1 = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(num_input_features, bottle_neck_size*growth_rate, kernel_size=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(bottle_neck_size*growth_rate)
        # self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_input_features, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, input):
        # bottle_neck_out = self.conv1(self.act1(self.bn1(datalists)))
        out = self.conv2(self.act1(self.bn1(input)))
        if self.drop_rate:
            out = F.dropout2d(out, p=self.drop_rate, training=self.training)

        return out


class dense_block(nn.Module):
    def __init__(self, num_input_features, num_layers, growth_rate, drop_rate, is_up=False):
        super(dense_block, self).__init__()
        self.is_up = is_up
        num_features = num_input_features
        for i in range(num_layers):
            layer = dense_layer(num_features, growth_rate, drop_rate)
            num_features += growth_rate
            self.add_module('layer{}'.format(i+1), layer)

    def forward(self, input):
        if self.is_up:
            cat_features = []
            for name, child in self.named_children():
                new_features = child(input)
                input = torch.cat([input, new_features], 1)
                cat_features.append(new_features)

            return torch.cat(cat_features, 1)                      #exlude the datalists
        else:
            features = input
            for name, child in self.named_children():
                new_features = child(features)
                features = torch.cat([features, new_features], 1)

            return features


class transition_down_layer(nn.Module):
    def __init__(self, in_ch, dropout_rate):
        super(transition_down_layer, self).__init__()

        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)
        self.trans_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.drop_rate = dropout_rate
        self.pool = nn.MaxPool2d(2)

    def forward(self, input):
        out = self.trans_conv(self.act(self.bn(input)))
        if self.drop_rate:
            out = F.dropout2d(out, self.drop_rate, training=self.training)
        out = self.pool(out)

        return out


class transition_up_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(transition_up_layer, self).__init__()

        # self.bn = nn.BatchNorm2d(in_ch)
        # self.act = nn.ReLU(inplace=True)
        self.trans_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)

    def forward(self, input, skip):
        out = self.trans_conv(input)
        out = torch.cat([out, skip], 1)

        return out


class bottleneck_layer(nn.Module):
    def __init__(self, in_ch, num_layers, growth_rate, drop_rate, is_up):
        super(bottleneck_layer, self).__init__()

        self.dense_block = dense_block(in_ch, num_layers, growth_rate, drop_rate, is_up)

    def forward(self, input):
        out = self.dense_block(input)

        return out


class ReconPETUNet(nn.Module):
    def __init__(self, in_ch, out_ch, num_init_features, dense_block_layers=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], growth_rate=16, dense_drop_rate=0, trans_drop_rate=0.2):
        super(ReconPETUNet, self).__init__()

        self.long_residual_connection = True

        self.init_conv = nn.Conv2d(in_ch, num_init_features, kernel_size=3, padding=1, bias=False)

        self.dense_block1 = dense_block(num_init_features, dense_block_layers[0], growth_rate, dense_drop_rate, False)
        in_down_ch1 = num_init_features + dense_block_layers[0]*growth_rate
        self.trans_down_block1 = transition_down_layer(in_down_ch1, trans_drop_rate)

        self.dense_block2 = dense_block(in_down_ch1, dense_block_layers[1], growth_rate, dense_drop_rate, False)
        in_down_ch2 = in_down_ch1 + dense_block_layers[1]*growth_rate
        self.trans_down_block2 = transition_down_layer(in_down_ch2, trans_drop_rate)

        self.dense_block3 = dense_block(in_down_ch2, dense_block_layers[2], growth_rate, dense_drop_rate, False)
        in_down_ch3 = in_down_ch2 + dense_block_layers[2]*growth_rate
        self.trans_down_block3 = transition_down_layer(in_down_ch3, trans_drop_rate)

        self.dense_block4 = dense_block(in_down_ch3, dense_block_layers[3], growth_rate, dense_drop_rate, False)
        in_down_ch4 = in_down_ch3 + dense_block_layers[3]*growth_rate
        self.trans_down_block4 = transition_down_layer(in_down_ch4, trans_drop_rate)

        self.dense_block5 = dense_block(in_down_ch4, dense_block_layers[4], growth_rate, dense_drop_rate, False)
        in_down_ch5 = in_down_ch4 + dense_block_layers[4]*growth_rate
        self.trans_down_block5 = transition_down_layer(in_down_ch5, trans_drop_rate)

        self.dense_bottle_block = bottleneck_layer(in_down_ch5, dense_block_layers[5], growth_rate, dense_drop_rate, True)
        in_up_ch1 = dense_block_layers[5]*growth_rate

        self.trans_up_block1 = transition_up_layer(in_up_ch1, in_up_ch1)
        in_dense_ch1 = in_up_ch1 + in_down_ch5
        self.dense_up_block1 = dense_block(in_dense_ch1, dense_block_layers[6], growth_rate, dense_drop_rate, True)

        in_up_ch2 = dense_block_layers[6]*growth_rate
        self.trans_up_block2 = transition_up_layer(in_up_ch2, in_up_ch2)
        in_dense_ch2 = in_up_ch2 + in_down_ch4
        self.dense_up_block2 = dense_block(in_dense_ch2, dense_block_layers[7], growth_rate, dense_drop_rate, True)

        in_up_ch3 = dense_block_layers[7]*growth_rate
        self.trans_up_block3 = transition_up_layer(in_up_ch3, in_up_ch3)
        in_dense_ch3 = in_up_ch3 + in_down_ch3
        self.dense_up_block3 = dense_block(in_dense_ch3, dense_block_layers[8], growth_rate, dense_drop_rate, True)

        in_up_ch4 = dense_block_layers[8] * growth_rate
        self.trans_up_block4 = transition_up_layer(in_up_ch4, in_up_ch4)
        in_dense_ch4 = in_up_ch4 + in_down_ch2
        self.dense_up_block4 = dense_block(in_dense_ch4, dense_block_layers[9], growth_rate, dense_drop_rate, True)

        in_up_ch5 = dense_block_layers[9] * growth_rate
        self.trans_up_block5 = transition_up_layer(in_up_ch5, in_up_ch5)
        in_dense_ch5 = in_up_ch5 + in_down_ch1
        self.dense_up_block5 = dense_block(in_dense_ch5, dense_block_layers[10], growth_rate, dense_drop_rate, True)

        in_conv_ch = dense_block_layers[10]*growth_rate
        self.conv = nn.Conv2d(in_conv_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x, res):
        input_block = self.init_conv(x)
        dense_down_block1 = self.dense_block1(input_block)
        trans_down_block1 = self.trans_down_block1(dense_down_block1)

        dense_down_block2 = self.dense_block2(trans_down_block1)
        trans_down_block2 = self.trans_down_block2(dense_down_block2)

        dense_down_block3 = self.dense_block3(trans_down_block2)
        trans_down_block3 = self.trans_down_block3(dense_down_block3)

        dense_down_block4 = self.dense_block4(trans_down_block3)
        trans_down_block4 = self.trans_down_block4(dense_down_block4)

        dense_down_block5 = self.dense_block5(trans_down_block4)
        trans_down_block5 = self.trans_down_block5(dense_down_block5)

        dense_bottle_block = self.dense_bottle_block(trans_down_block5)

        trans_up_block1 = self.trans_up_block1(dense_bottle_block, dense_down_block5)
        dense_up_block1 = self.dense_up_block1(trans_up_block1)

        trans_up_block2 = self.trans_up_block2(dense_up_block1, dense_down_block4)
        dense_up_block2 = self.dense_up_block2(trans_up_block2)

        trans_up_block3 = self.trans_up_block3(dense_up_block2, dense_down_block3)
        dense_up_block3 = self.dense_up_block3(trans_up_block3)

        trans_up_block4 = self.trans_up_block4(dense_up_block3, dense_down_block2)
        dense_up_block4 = self.dense_up_block4(trans_up_block4)

        trans_up_block5 = self.trans_up_block5(dense_up_block4, dense_down_block1)
        dense_up_block5 = self.dense_up_block5(trans_up_block5)

        out_block = self.conv(dense_up_block5)
        if len(list(res.shape)) == 3:
            res = torch.unsqueeze(res, 1)

        out = torch.add(out_block, res)

        return out


if __name__ == "__main__":
    from thop import profile, clever_format
    import torch

    # input = torch.randn(1, 5, 128, 128)
    # res = input[0, 2, ...].clone().unsqueeze(0).unsqueeze(0)
    # net = DenseUNet(5, 1, 32)
    # print(net)
    # device = torch.device("cuda:1")
    # input = input.to(device)
    # res = res.to(device)
    # net = net.to(device)
    #
    # flops, params = profile(net, inputs=(input, res))
    # flops, params = clever_format([flops, params], "%.3f")
    # msg = "flops:{}, params:{}".format(flops, params)
    # print(msg)
