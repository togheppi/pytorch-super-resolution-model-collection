import torch
import torch.nn as nn
from base_networks import *


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, num_residuals):
        super(Net, self).__init__()

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, norm=None, bias=False)

        conv_blocks = []
        for _ in range(num_residuals):
            conv_blocks.append(ConvBlock(base_filter, base_filter, 3, 1, 1, norm=None, bias=False))
        self.residual_layers = nn.Sequential(*conv_blocks)

        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None, bias=False)

    def weight_init(self):
        pass

    def forward(self, x):
        residual = x
        out = self.input_conv(x)
        out = self.residual_layers(out)
        out = self.output_conv(out)
        out = torch.add(out, residual)

        return out


class VDSR():
    def __init__(self, args):
        self.model_name = args.model_name
        self.num_channels = args.num_channels

    def load_dataset(self, is_train=True):
        pass

    def train(self):
        self.model = Net(self.num_channels, 64, 18)
        pass

    def test(self):
        pass

    def save_model(self, model_dir):
        pass

    def load_model(self, model_dir):
        pass
