from __future__ import print_function

import torch
import torch.nn as nn


class LWF_DIST(nn.Module):
    """
    Learning without Forgetting (LwF) - Distillation Loss
    code from author: https://github.com/mmasana/FACIL/blob/master/src/approach/lwf.py
    """

    def __init__(self, temp):
        super(LWF_DIST, self).__init__()
        self.temp = temp
        self.exp = 1.0 / self.temp

    def forward(self, outputs, targets, size_average=True, eps=1e-5):
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if self.exp != 1:
            out = out.pow(self.exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(self.exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce