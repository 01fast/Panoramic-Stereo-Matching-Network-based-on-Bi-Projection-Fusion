from __future__ import print_function

import unittest
import cv2
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from scipy import signal
import numba.cuda
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import TEST_NUMPY


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1], order='c'))

    def forward(self, x):
        out = torch.sum(x * self.disp.data, 1, keepdim=True)
        return out


class cost_ud:
    def __init__(self):
        super(cost_ud, self).__init__()

    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def run(self, image, stride):
        image = image.squeeze()  # ([32, 32, 32])
        if 2 * stride >= np.shape(image)[1]:
            return image
        image = image[:, stride:32-stride, stride:32-stride].cpu().detach().numpy().astype(np.float32)
        image = torch.from_numpy(cv2.resize(np.transpose(image, (1, 2, 0)), (32, 32)).astype(np.float16)).cuda()
        return image
