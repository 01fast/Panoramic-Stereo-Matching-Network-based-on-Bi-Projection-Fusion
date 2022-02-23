from __future__ import print_function
from .util import Equirec2Cube
from .util import Cube2Equirec
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
from PIL import Image
import torch.nn.functional as F
from .submodule import *
import math


class forfilter(nn.Module):
    def __init__(self, inplanes):
        super(forfilter, self).__init__()

        self.forfilter1 = nn.Conv2d(1, 1, (7, 1), 1, (0, 0), bias=False)
        self.inplanes = inplanes

    def forward(self, x):
        out = self.forfilter1.cuda()(F.pad(torch.unsqueeze(x[:, 0, :, :], 1), pad=[0, 0, 3, 3], mode='replicate'))
        # ([1, 1, 134, 256])
        for i in range(1, self.inplanes):
            out = torch.cat((out, self.forfilter1(F.pad(torch.unsqueeze(x[:, i, :, :], 1), pad=[0, 0, 3, 3],
                                                        mode='replicate'))), 1)
        return out


class disparityregression_sub3(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression_sub3, self).__init__()
        self.disp = Variable(torch.Tensor(
            np.reshape(np.array(range(maxdisp * 3)), [1, maxdisp * 3, 1, 1]) /
            3).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class cost360SD(nn.Module):
    def __init__(self, maxdisp):
        super(cost360SD, self).__init__()
        self.maxdisp = maxdisp

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.fuse0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.fuse1 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.c2e = Cube2Equirec(32, 64, 128)
        self.e2c = Equirec2Cube(64, 128, 32)
        self.forF = forfilter(32)
        self.forF1 = forfilter(32)

    def forward(self, outputs_up, outputs_down, training):
        global pred1, pred2
        cost = Variable(
            torch.zeros(outputs_up.size()[0],
                        outputs_up.size()[1] * 2, int(self.maxdisp / 4 * 3),
                        outputs_up.size()[2],
                        outputs_up.size()[3]), requires_grad=True).cuda()

        for i in range(int(self.maxdisp / 4 * 3)):
            if i > 0:
                cost[:, :outputs_up.size()[1], i, :, :] = outputs_down[:, :, :, :]
                cost[:, outputs_up.size()[1]:, i, :, :] = shift_down[:, :, :, :]
                shift_down = self.forF(shift_down)
            else:
                cost[:, :outputs_up.size()[1], i, :, :] = outputs_down
                cost[:, outputs_up.size()[1]:, i, :, :] = outputs_up
                shift_down = self.forF(outputs_up)

        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3+cost0
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.interpolate(cost1, [self.maxdisp * 3, outputs_up.size()[2] * 4, outputs_up.size()[3] * 4],
                               mode='trilinear', align_corners=True)
            cost2 = F.interpolate(cost2, [self.maxdisp * 3, outputs_up.size()[2] * 4, outputs_up.size()[3] * 4],
                               mode='trilinear', align_corners=True)
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression_sub3(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression_sub3(self.maxdisp)(pred2)

        cost3 = F.interpolate(cost3, [self.maxdisp * 3, outputs_up.size()[2] * 4, outputs_up.size()[3] * 4],
                              mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression_sub3(self.maxdisp)(pred3)


        if training:
            return pred1, pred2, pred3
        else:
            return pred3, None, None
