import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNoise(nn.Module):
    """ add gasussian noise into feature """
    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        zeros_ = torch.zeros_like(x)
        n = torch.normal(zeros_, std=self.std)
        return x + n


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
