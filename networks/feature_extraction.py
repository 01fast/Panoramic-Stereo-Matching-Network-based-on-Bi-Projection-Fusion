from __future__ import print_function
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CEELayer(nn.Module):
    def __init__(self, channels):
        super(CEELayer, self).__init__()

        self.res_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False)
        self.res_bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.selayer = SELayer(channels)

    def forward(self, equi_feat_main, equi_feat):
        x = torch.cat([equi_feat_main, equi_feat], 1)
        x = self.relu(self.res_bn1(self.res_conv1(x)))
        x = self.relu(self.res_bn1(self.res_conv1(torch.cat([x, equi_feat_main], 1))))
        x = self.selayer(x)
        return x


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=dilation if dilation > 1 else pad,
                  dilation=dilation,
                  bias=False), nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  padding=pad,
                  stride=stride,
                  bias=False), nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation),
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


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        # ASPP network
        self.aspp1 = nn.Sequential(convbn(160, 16, 1, 1, 0, 1),
                                   nn.ReLU(inplace=True))
        self.aspp2 = nn.Sequential(convbn(160, 16, 3, 1, 1, 6),
                                   nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(convbn(160, 16, 3, 1, 1, 12),
                                   nn.ReLU(inplace=True))
        self.aspp4 = nn.Sequential(convbn(160, 16, 3, 1, 1, 18),
                                   nn.ReLU(inplace=True))
        self.aspp5 = nn.Sequential(convbn(160, 16, 3, 1, 1, 24),
                                   nn.ReLU(inplace=True))
        self.newlastconv = nn.Sequential(
            convbn(144, 64, 3, 1, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0, stride=1, bias=False))
        # Polar Branch
        self.firstcoord = nn.Sequential(convbn(1, 32, 3, 2, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 32, 3, 2, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True))


        self.firstcoord = nn.Sequential(convbn(1, 32, 3, 2, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 32, 3, 2, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True))
        self.CEELayer32 = CEELayer(32)
        self.CEELayer64 = CEELayer(64)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, input_img_4):
        input_img = input_img_4[:, :3, :, :]
        input_img = input_img.cuda()
        output = self.firstconv(input_img)
        output = self.layer1(output)
        output_fused_1 = self.CEELayer32(output, nn.ReLU(inplace=True)(convbn(3, 32, 3, 2, 1, 1).cuda()(input_img)))
        output_raw = self.layer2(output_fused_1)
        output_fused_2 = self.CEELayer64(output_raw, nn.ReLU(inplace=True)(convbn(32, 64, 3, 2, 1, 1).cuda()(output)))
        output = self.layer3(output_fused_2)
        output_skip = self.layer4(output)

        # coord avg pooling and concat to main feature
        out_coord = self.firstcoord(input_img_4[:, 3:, :, :])
        output_skip_c = torch.cat((output_skip, out_coord), 1)

        # ASPP and new last conv
        ASPP1 = self.aspp1(output_skip_c)
        ASPP2 = self.aspp2(output_skip_c)
        ASPP3 = self.aspp3(output_skip_c)
        ASPP4 = self.aspp4(output_skip_c)
        ASPP5 = self.aspp5(output_skip_c)
        output_feature = torch.cat(
            (output_raw, ASPP1, ASPP2, ASPP3, ASPP4, ASPP5), 1)

        output_feature = self.newlastconv(output_feature)

        return output_feature
