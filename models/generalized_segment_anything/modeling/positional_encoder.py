import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=256, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x):
        B, C, H, W = x.shape
        #feat_token = x
        cnn_feat = x
        cnn_feat = self.proj(cnn_feat)
        
        return cnn_feat


