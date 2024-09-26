# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from tkinter import X
from unittest import skip
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock, Adapter, AugAdapter, MSAdaptFormer
import math

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.cnn_embed = ResNetEmbed(embed_dim=embed_dim, network='resnet101') # new to sam
        self.patch_embed = PatchEmbed0(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=3,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            #self.pos_embed = nn.Parameter(
            #    torch.zeros(1, 1024//16, 1024//16, embed_dim) # torch.zeros(1, 1024//16, 1024//16, embed_dim)
            #)
            #self.post_pos_embed = PostPosEmbed(embed_dim=embed_dim, ori_feature_size=1024//16, new_feature_size=img_size//patch_size) # new to sam
            self.pos_embed = PosConvEmbed(embed_dim=embed_dim)

        self.blocks = nn.ModuleList()

        for i in range(depth):
            block = ParaBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                depth = i,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        self.input_Adapter = Adapter(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1) # b c h w
        cnnx = self.cnn_embed(x) # b h w c
        x = self.patch_embed(x) # b h w c
        x = self.input_Adapter(x)
        
        cnnx = F.interpolate(cnnx, (x.shape[1], x.shape[2]))
        cnnx = cnnx.permute(0, 2, 3, 1)
        
        x = x + cnnx

        for j, blk in enumerate(self.blocks):
            x, cnnx = blk(x, cnnx) # b h w c
            
            #------------------- PEG ----------------------------
            if j==0:
                pos_embed = self.pos_embed(x)
                pos_embed = pos_embed.permute(0, 2, 3, 1)
                x = x + pos_embed  
            #---------------------------------------------------
        
        x = x + 0.5*cnnx

        x = self.neck(x.permute(0, 3, 1, 2))
        
        return x


class ParaBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        depth: int=0
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size


        # ------------------ new to sam----------------------
        if self.window_size == 0:
            self.refine_Adapter = MobileConv(in_channels=dim, out_channels=dim)          
        self.MLP_Adapter = MSAdaptFormer(dim, skip_connect=False)  # new to sam, MLP-adapter, no skip connection
        self.scale = 0.5
        # ---------------------------------------------------
        
        
        self.dim = dim
        self.depth = depth

    def forward(self, x: torch.Tensor, cnnx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        if self.window_size == 0:
            # ------------------ new to sam----------------------
            
            cnnx = self.refine_Adapter(cnnx.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
            # ---------------------------------------------------

        x = self.attn(x)
        
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x

        xn = self.norm2(x)
        x = x + self.mlp(xn)


        # ------------------ new to sam----------------------
        
        Hn, Wn = x.shape[1], x.shape[2]
        x = x + self.scale * self.MLP_Adapter(xn, Hn, Wn)
        
        # ---------------------------------------------------

        return x, cnnx


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv0 = self.qkv(x)
        qkv = qkv0.reshape(B, H, W, 3, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)#[3, B, num_head, H*W, -1]
        q, k, v = qkv.reshape(3, B * self.num_heads, H, W, -1).unbind(0)

        
        # ----------------- Attention pooling ----------------------
        
        q = q.reshape(q.shape[0], q.shape[1]*q.shape[2], -1)
        
        k = k.permute(0, 3, 1, 2)
        k = nn.AdaptiveAvgPool2d(3)(k)
        Bk, Ck, Hk, Wk = k.shape
        k = k.permute(0, 2, 3, 1)
        k = k.reshape(Bk, Hk*Wk, -1)
        
        v = v.permute(0, 3, 1, 2)
        v = nn.AdaptiveAvgPool2d(3)(v)
        Bv, Cv, Hv, Wv = v.shape
        v = v.permute(0, 2, 3, 1)
        v = v.reshape(Bv, Hv*Wv, -1)
        
        # ----------------------------------------------------------
        
        
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (Hk, Wk))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class qkvAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #self.k = nn.Linear(dim, dim, bias=qkv_bias)
        #self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.k = nn.Sequential(
                 nn.AdaptiveAvgPool2d(7),
                 nn.Conv2d(dim, dim, kernel_size=1, stride=1),
                 LayerNorm2d(dim),
                 nn.GELU(),
                 )
                 
        self.v = nn.Sequential(
                 nn.AdaptiveAvgPool2d(7),
                 nn.Conv2d(dim, dim, kernel_size=1, stride=1),
                 LayerNorm2d(dim),
                 nn.GELU(),
                 )
        
        
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v:torch.Tensor) -> torch.Tensor:
        #q [8, 16, 16, 768]
        #k [8, 16, 16, 768]
        #v [8, 16, 16, 768]
        B, H, W, _ = q.shape
        q = self.q(q).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B*self.num_heads, H*W, -1)
        
        k = k.permute(0,3,1,2)
        v = v.permute(0,3,1,2)
        k = self.k(k)
        v = self.v(v)
        k = k.permute(0,2,3,1)
        v = v.permute(0,2,3,1)
        
        B, H_k, W_k, _ = k.shape
        
        k = k.reshape(B, H_k * W_k, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B*self.num_heads, H_k*W_k, -1)
        v = v.reshape(B, H_k * W_k, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B*self.num_heads, H_k*W_k, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    #print(r_q.shape)
    #print(Rh.shape)
    #print(Rw.shape)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SingleDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()     #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SingleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class MobileConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, round(out_channels//8), 1, 1, bias=False),
            LayerNorm2d(round(out_channels//8)),
            nn.GELU(),
            nn.Conv2d(round(out_channels//8), round(out_channels//8), 3, 1, 1, bias=False, groups=int(out_channels//8)),
            LayerNorm2d(round(out_channels//8)),
            nn.GELU(),            
            nn.Conv2d(round(out_channels//8), out_channels, 1, 1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        
        self.se = SE_Block(in_channels, out_channels)


    def forward(self, x):
        x_path = x
        x = self.conv(x)
        x_path  = self.se(x_path)
        
        return x+x_path
        
        
class SE_Block(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=4): #0.25
        super().__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features=in_channels, out_features=round(out_channels/reduction_ratio))
        self.fc2 = nn.Linear(in_features=round(out_channels/reduction_ratio), out_features=out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.globalAvgPool(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        return out * x



class CNNEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        patchsize: int = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (int): kernel size of the tokenization layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        downtimes = int(math.log2(patchsize))
        mid_channel = 64
        self.inc = DoubleConv(in_chans, mid_channel)
        self.downs = nn.ModuleList()
        for i in range(downtimes):
            if i == downtimes-1:
                down = Down(mid_channel, embed_dim)
            else:
                down = Down(mid_channel, mid_channel*2)
            mid_channel = mid_channel*2
            self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class SingleCNNEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        patchsize: int = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (int): kernel size of the tokenization layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        downtimes = int(math.log2(patchsize))
        #print(downtimes)
        mid_channel = 64
        self.inc = SingleConv(in_chans, mid_channel)
        self.downs = nn.ModuleList()
        for i in range(downtimes):
            if i == downtimes-1:
                down = SingleDown(mid_channel, embed_dim)
            else:
                down = SingleDown(mid_channel, mid_channel*2)
            mid_channel = mid_channel*2
            self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("Single")
        #print(x.shape)
        x = self.inc(x)
        #print(x.shape)
        for down in self.downs:
            x = down(x)
            #print(x.shape)
        #x = F.interpolate(())
        # B C H W -> B H W C
        #x = x.permute(0, 2, 3, 1)
        #print("Double")
        return x
        
        
        
class ResNetEmbed(nn.Module):
    def __init__(self, embed_dim: int = 768, network='resnet18') -> None:
        super(ResNetEmbed, self).__init__()

        ##### network ######
        if network=='resnet18':
            self.backbone = models.resnet18(pretrained=True)
            f_dim = 256

        elif network=='resnet50':
            self.backbone = models.resnet50(pretrained=True)
            f_dim = 1024

        elif network=='resnet101':
            self.backbone =models.resnet101(pretrained=True)
            f_dim = 1024

        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.res2 = self.backbone.layer1
        self.res3 = self.backbone.layer2
        self.res4 = self.backbone.layer3
        
        self.conv_f = nn.Conv2d(f_dim, embed_dim, 1, 1, bias=False)


    def __call__(self, x):
        if x.shape[1] != 3:
            x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])

        h1 = F.relu(self.bn1(self.conv1(x)))
        low_feat = self.res2(h1)
        h3 = self.res3(low_feat)
        h4 = self.res4(h3)
        #print(h4.shape)
        return self.conv_f(h4)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x        
        



class PostPosEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ori_feature_size: int = 64,
        new_feature_size: int = 32,
    ) -> None:
        """
        Args:
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        downtimes = int(math.log2(ori_feature_size//new_feature_size))
        self.downs = nn.ModuleList()
        for i in range(downtimes):
            down = SingleDown(embed_dim, embed_dim)
            #down = nn.MaxPool2d(2)
            self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("PostPosEmbed")
        #print(x.shape)
        # B H W C -> B C H W
        x = x.permute(0, 3, 1, 2) # [1, h, w, c]
        #print("size")
        #print(x.shape)
        B, C, H, W = x.shape
        for down in self.downs:
            x = down(x)
        # B C H W -> B H W C
        #print(x.shape)
        x = F.interpolate(x, (H//4, W//4))
        x = x.permute(0, 2, 3, 1)
        #print(x.shape)
        return x


class PatchEmbed0(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=16, stride=(8, 8), padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("Size")#256 --> 32 : 8
        #print(x.shape)
        B, C, H, W = x.shape
        x = F.interpolate(x, (H+8, W+8), mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = F.interpolate(x, (H//8, W//8))
        #print(x.shape)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)

        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
        
        
class PosConvEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        #downtimes = int(math.log2(ori_feature_size//new_feature_size))
        #self.downs = nn.ModuleList()
        #for i in range(3):
        #self.down = SingleDown(in_channels=3, out_channels=embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True, groups=embed_dim)
            #down = nn.MaxPool2d(2)
            #self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("PostPosEmbed")
        #print(x.shape)
        # B H W C -> B C H W
        x = x.permute(0, 3, 1, 2) # [1, h, w, c]
        #print("size")
        #print(x.shape)
        B, C, H, W = x.shape
        #for down in self.downs:
        x = self.conv(x)
        #print(x.shape)
        # B C H W -> B H W C
        #print(x.shape)
        #x = F.interpolate(x, (H//4, W//4))
        #x = x.permute(0, 2, 3, 1)
        #print(x.shape)
        return x
