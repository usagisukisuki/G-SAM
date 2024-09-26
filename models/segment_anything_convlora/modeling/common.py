# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type
import math

class Expert(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        scale: float,
        ) -> None:
        super().__init__()
        self.scale = scale
        
        self.conv = nn.Conv2d(embedding_dim,embedding_dim,3,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        H_n, W_n = int(H*self.scale), int(W*self.scale)
        x = F.interpolate(x, (H_n, W_n))
        x = self.conv(x)
        x = F.interpolate(x, (H, W))
        return x
        

class Gate(nn.Module):
    def __init__(
        self,
        num_expert: int,
        embedding_dim: int,
        ) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(embedding_dim,num_expert,3,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        
        return F.softmax(x, dim=1)



class ConvLoRA(nn.Module):
    def __init__(self, in_dim, mlp_ratio=0.25):
        super().__init__()
        bottle_dim = int(in_dim * mlp_ratio)
        self.lora_A = nn.Parameter(torch.zeros(in_dim, bottle_dim))
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, in_dim))
        self.scaling = 1.0 / bottle_dim
        
        self.expert1 = Expert(bottle_dim, 1.0)
        self.expert2 = Expert(bottle_dim, 2.0)
        self.expert3 = Expert(bottle_dim, 3.0)
        self.expert4 = Expert(bottle_dim, 4.0)
        #self.expert5 = Expert(bottle_dim, 5.0)
        #self.expert6 = Expert(bottle_dim, 6.0)
        #self.expert7 = Expert(bottle_dim, 7.0)
        #self.expert8 = Expert(bottle_dim, 8.0)
        
        self.gate = Gate(num_expert=4, embedding_dim=bottle_dim)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def dtype(self):
        return self.lora_A.dtype

    def forward(self, x, H, W):
        x = x @ self.lora_A #[300, 196, 16]

        ##############################################
        
        x = x.view(x.shape[0], H, W, -1)
        x = x.permute(0,3,1,2)
        x1 = self.expert1(x)
        x2 = self.expert2(x)
        x3 = self.expert3(x)
        x4 = self.expert4(x)
        #x5 = self.expert5(x)
        #x6 = self.expert6(x)
        #x7 = self.expert7(x)
        #x8 = self.expert8(x)
        xg = self.gate(x)
        
        x = x1*xg[:, 0].unsqueeze(1)+x2*xg[:, 1].unsqueeze(1)+x3*xg[:, 2].unsqueeze(1)+x4*xg[:, 3].unsqueeze(1)#+x5*xg[:, 4].unsqueeze(1)+x6*xg[:, 5].unsqueeze(1)+x7*xg[:, 6].unsqueeze(1)+x8*xg[:, 7].unsqueeze(1)
        
        x = x.permute(0,2,3,1)
        x = x.view(x.shape[0], H*W, -1)
        ##############################################
        x = x @ self.lora_B
        x = self.scaling * x
        return x


class LoRA(nn.Module):
    def __init__(self, in_dim, mlp_ratio=0.25):
        super().__init__()
        bottle_dim = int(in_dim * mlp_ratio)
        self.lora_A = nn.Parameter(torch.zeros(in_dim, bottle_dim))
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, in_dim))
        self.scaling = 1.0 / bottle_dim
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def dtype(self):
        return self.lora_A.dtype

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
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
