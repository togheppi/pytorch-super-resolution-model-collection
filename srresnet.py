import torch
import torch.nn as nn
from base_networks import *

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, num_residuals):
        super(Net, self).__init__()

        self.input_conv = ConvBlock(num_channels, base_filter, 9, 1, 4, activation='lrelu', norm=None, bias=False)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResNetBlock(base_filter, activation='lrelu', bias=False))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None, bias=False)

        self.upscale4x = nn.Sequential(
            Upsample2x(base_filter, activation='lrelu', norm=None, bias=False),
            Upsample2x(base_filter, activation='lrelu', norm=None, bias=False),
        )

        self.output_conv = ConvBlock(base_filter, num_channels, 9, 1, 4, activation=None, norm=None, bias=False)

    def weight_init(self):
        pass

    def forward(self, x):
        out = self.input_conv(x)
        residual = out
        out = self.residual_layers(out)
        out = self.mid_conv(out)
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.output_conv(out)
        return out


class SRResNet():
    def __init__(self, args):
        self.model_name = args.model_name
        self.num_channels = args.num_channels

    def load_dataset(self, is_train=True):
        pass

    def train(self):
        self.model = Net(self.num_channels, 64, 16)
        pass

    def test(self):
        pass

    def save_model(self, model_dir):
        pass

    def load_model(self, model_dir):
        pass
