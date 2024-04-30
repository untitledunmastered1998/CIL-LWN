from __future__ import print_function

import torch
import torch.nn as nn


class L2_reg(nn.Module):
    def __init__(self, p=2):
        super(L2_reg, self).__init__()
        self.p = p
        self.reg = torch.nn.PairwiseDistance(p=self.p)

    def forward(self, f_s, f_t):
        return torch.sqrt(torch.sum(self.reg(f_s, f_t)))