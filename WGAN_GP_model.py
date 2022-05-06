import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.ConvNet = nn.Sequential(
                    # Input: N x channels_noise x 1 x 1
                    self.__block(100, 16 * 16, 4, 1, 0),  # img: 4x4
                    self.__block(16 * 16, 16 * 8, 4, 2, 1),  # img: 8x8
                    self.__block(16 * 8, 16 * 4, 4, 2, 1),  # img: 16x16
                    self.__block(16 * 4, 16 * 2, 4, 2, 1),  # img: 32x32
                    nn.ConvTranspose2d(
                        16 * 2, 1, kernel_size=4, stride=2, padding=1
                    ),
                    # Output: N x channels_img x 64 x 64
                    nn.Tanh(),
                    )


    def __block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    )


    def forward(self , x):
        return self.ConvNet(x)



class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator , self).__init__()

        self.ConvNet = nn.Sequential(
                        # input: N x channels_img x 64 x 64
                        nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU(0.2),
                        # _block(in_channels, out_channels, kernel_size, stride, padding)
                        self.__block(16, 16 * 2, 4, 2, 1),
                        self.__block(16 * 2, 16 * 4, 4, 2, 1),
                        self.__block(16 * 4, 16 * 8, 4, 2, 1),
                        # After all _block img output is 4x4 (Conv2d below makes into 1x1)
                        nn.Conv2d(16 * 8, 1, kernel_size=4, stride=2, padding=0),
                        )


    def __block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                    nn.InstanceNorm2d(out_channels, affine=True),
                    nn.LeakyReLU(0.2),
                    )


    def forward(self, x):
        return self.ConvNet(x)
