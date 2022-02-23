from __future__ import print_function
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .feature_extraction import feature_extraction
from .cost360SD import *


class net360SD(nn.Module):
    def __init__(self, maxdisp):
        super(net360SD, self).__init__()
        self.model_cnn = feature_extraction().cuda()
        self.model_cost = cost360SD(maxdisp).cuda()

    def forward(self, equi_inputs_up, equi_inputs_down, training):
        outputs_up = self.model_cnn(equi_inputs_up)
        outputs_down = self.model_cnn(equi_inputs_down)
        output1, output2, output3 = self.model_cost(outputs_up, outputs_down, training)
        return output1, output2, output3
