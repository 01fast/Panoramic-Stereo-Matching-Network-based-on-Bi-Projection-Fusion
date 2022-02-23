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


class cost_basic(nn.Module):
    def __init__(self, maxdisp):
        super(cost_basic, self).__init__()
        self.maxdisp = maxdisp

        self.dres0_0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1_0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))
        self.dres0_1 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1_1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))
        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres2_1 = hourglass(32)

        self.dres3_1 = hourglass(32)

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
        cube_feat1 = torch.from_numpy(
            self.e2c.run(outputs_up[0].cpu().detach().numpy()).astype(np.float16)[np.newaxis, :]).cuda()
        cube_feat2 = torch.from_numpy(
            self.e2c.run(outputs_down[0].cpu().detach().numpy()).astype(np.float16)[np.newaxis, :]).cuda()
        if not training:
            cube_feat1 = torch.tensor(cube_feat1, dtype=torch.float32)
            cube_feat2 = torch.tensor(cube_feat2, dtype=torch.float32)
        #print("begin my cost catch:", "\n", "cube_up:", np.shape(cube_feat1), "output_up", np.shape(outputs_up), "\n",
        #      "cube_down:", np.shape(cube_feat2), "outputs_down", np.shape(outputs_down))
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

        cost1 = Variable(
            torch.zeros(cube_feat1.size()[0],
                        cube_feat1.size()[1] * 2, 16,
                        cube_feat1.size()[2],
                        cube_feat1.size()[3]), requires_grad=True).cuda()
        siz3 = cube_feat1.size()[3]
        for i in range(16):
            if i > 0:
                cost1[:, :cube_feat1.size()[1], i, :, :int(siz3 / 3 * 2)] = cube_feat1[:, :, :, :int(siz3 / 3 * 2)]
                cost1[:, cube_feat1.size()[1]:, i, :, :int(siz3 / 3 * 2)] = shift_cube2[:, :, :, :]
                # ([1, 32, 63, 256])
                shift_cube2 = self.forF1(shift_cube2)

                cost1[:, :cube_feat1.size()[1], i, :, int(siz3 / 3 * 2):int(siz3 / 6 * 5)] \
                    = cube_feat1[:, :, :, int(siz3 / 3 * 2):int(siz3 / 6 * 5)]
                cost1[:, cube_feat1.size()[1]:, i, :, int(siz3 / 3 * 2):int(siz3 / 6 * 5)] \
                    = cost_ud().run(cube_feat2[:, :, :, int(siz3 / 3 * 2):int(siz3 / 6 * 5)], i)

                cost1[:, :cube_feat1.size()[1], i, :, -int(siz3 / 6 * 1):] \
                    = cost_ud().run(cube_feat1[:, :, :, -int(siz3 / 6 * 1):], i)

                cost1[:, cube_feat1.size()[1]:, i, :, -int(siz3 / 6 * 1):] \
                    = cube_feat2[:, :, :, -int(siz3 / 6 * 1):]

            else:
                cost1[:, :cube_feat1.size()[1], i, :, :] = cube_feat1[:, :, :, :]
                cost1[:, cube_feat1.size()[1]:, i, :, :] = cube_feat2[:, :, :, :]

                shift_cube2 = self.forF1(cube_feat2[:, :, :, :int(siz3 / 3 * 2)])

        cost = cost.contiguous()  # ([1, 64, 32, 128, 256])
        cost2 = cost1.contiguous()  # ([1, 64, 16, 64, 384])
        cost_cube = torch.zeros(cost.size()[0], cost.size()[1], cost.size()[2], cost.size()[3], cost.size()[4]).cuda()
        for p in range(cost2.size()[2]):
            cost_cube[:, :, p, :, :] = self.c2e.run(cost1[:, :, p, :, :])

        cost0 = self.dres0_0(cost)
        cost0 = self.dres1_0(cost0) + cost0  # cost0[1, 32, 12, 128, 256]
        cost0_cube = self.dres0_1(cost_cube)
        cost0_cube = self.dres1_1(cost0_cube) + cost0_cube  # cost0[1, 32, 12, 128, 256]

        out1_cube, pre1_cube, post1_cube = self.dres2_1(cost0_cube, None, None)
        out1_cube = out1_cube + cost0_cube
        out2_cube, pre2_cube, post2_cube = self.dres3_1(out1_cube, pre1_cube, post1_cube)
        out2_cube = out2_cube + cost0_cube

        out1, pre1, post1 = self.dres2(cost0, None, None)  # out1 + cost0:([1, 32, 12, 128, 256])
        out1 = self.fuse0(torch.cat((F.relu(out1 + cost0, inplace=True), out1_cube), dim=1))
        pre1 = F.relu(pre1 + pre1_cube, inplace=True)
        post1 = F.relu(post1 + post1_cube, inplace=True)

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = self.fuse1(torch.cat((F.relu(out2 + cost0, inplace=True), out2_cube), dim=1))

        post2 = F.relu(post2 + post1_cube, inplace=True)

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = F.relu(out3 + cost0, inplace=True)

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if training:
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

        cost3 = F.interpolate(cost3, [self.maxdisp*3, outputs_up.size()[2]*4, outputs_up.size()[3]*4],
                              mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparityregression_sub3(self.maxdisp)(pred3)

        if training:
            return pred1, pred2, pred3
        else:
            return pred3, None, None
