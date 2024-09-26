# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
import math



class FirstConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(FirstConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        #self.norm = nn.LayerNorm(out_channels)
        self.gelu = nn.ReLU(inplace=True)

        
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,3,1)
        x = self.gelu(x)
        return x.permute(0,3,1,2)
        
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        #self.norm = nn.LayerNorm(out_channels)
        self.gelu = nn.ReLU(inplace=True)

        
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,3,1)
        x = self.gelu(x)
        return x.permute(0,3,1,2)
        

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        #self.norm = nn.LayerNorm(out_channels)
        self.gelu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.conv(self.pool(x))
        x = x.permute(0,2,3,1)
        x = self.gelu(x)
        x = x.permute(0,3,1,2)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = in_channels
        modules = []
        modules.append(FirstConv(in_channels, out_channels))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        #self.project = nn.Sequential(
        #    nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
        #    nn.BatchNorm2d(out_channels),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(0.1),)

    def forward(self, x):
        res = 0
        for conv in self.convs:
            res += conv(x)
        #res = torch.cat(res, dim=1)
        
        return res+x


class ASPPAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True, aspp_dilate=[12, 24, 36]): #0.25
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
        self.aspp = ASPP(D_hidden_features, aspp_dilate)
        
    def forward(self, x, H, W):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        
       
        ##############################################
        #xs = xs.view(xs.shape[0], H, W, -1)
        xs = xs.permute(0,3,1,2)
        xs = self.aspp(xs)
        xs = xs.permute(0,2,3,1)
        #xs = xs.view(xs.shape[0], H*W, -1)
        ##############################################
        
        
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x




class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True): #0.25
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x




class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
