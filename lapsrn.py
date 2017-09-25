import torch.nn as nn
from base_networks import *

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, num_convs):
        super(Net, self).__init__()

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='lrelu', norm=None, bias=False)

        self.convt_I1 = DeconvBlock(num_channels, num_channels, 4, 2, 1, activation=None, norm=None, bias=False)
        self.convt_R1 = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None, bias=False)

        conv_blocks1 = []
        for _ in range(num_convs):
            conv_blocks1.append(ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=None, bias=False))
        conv_blocks1.append(DeconvBlock(base_filter, base_filter, 4, 2, 1, activation='lrelu', norm=None, bias=False))
        self.convt_F1 = nn.Sequential(*conv_blocks1)

        self.convt_I2 = DeconvBlock(num_channels, num_channels, 4, 2, 1, activation=None, norm=None, bias=False)
        self.convt_R2 = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None, bias=False)

        conv_blocks2 = []
        for _ in range(num_convs):
            conv_blocks2.append(ConvBlock(base_filter, base_filter, 3, 1, 1, activation='lrelu', norm=None, bias=False))
        conv_blocks2.append(DeconvBlock(base_filter, base_filter, 4, 2, 1, activation='lrelu', norm=None, bias=False))
        self.convt_F2 = nn.Sequential(*conv_blocks2)

    def weight_init(self):
        pass

    def forward(self, x):
        out = self.input_conv(x)
        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)

        HR_4x = convt_I2 + convt_R2
        return HR_2x, HR_4x


class LapSRN():
    def __init__(self, args):
        self.model_name = args.model_name
        self.num_channels = args.num_channels

    def load_dataset(self, is_train=True):
        pass

    def train(self):
        self.model = Net(self.num_channels, 64, 10)
        pass

    def test(self):
        pass

    def save_model(self, model_dir):
        pass

    def load_model(self, model_dir):
        pass
