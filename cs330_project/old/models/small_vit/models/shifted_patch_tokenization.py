# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 02:00:55 2022

@author: Shahir, Faraz, Pratyush
Modified from: https://github.com/aanna0701/SPT_LSA_ViT
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


class PatchShifting(nn.Module):
    CARDINAL_4_DIRECTIONS_MODE = 0
    CARDINAL_8_DIRECTIONS_MODE = 1
    DIAGONAL_4_DIRECTIONS_MODE = 2

    def __init__(
            self,
            patch_size,
            mode=0):

        super().__init__()
        self.mode = mode
        self.shift = int(patch_size * (1 / 2))

    def forward(self, x):
        x_pad = F.pad(x, (self.shift, self.shift, self.shift, self.shift))

        # 4 cardinal directions
        if self.mode == self.CARDINAL_4_DIRECTIONS_MODE:
            x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift * 2]
            x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift * 2:]
            x_t2 = x_pad[:, :, :-self.shift * 2, self.shift:-self.shift]
            x_b2 = x_pad[:, :, self.shift * 2:, self.shift:-self.shift]
            x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)

        # 4 diagonal directions
        if self.mode == self.DIAGONAL_4_DIRECTIONS_MODE:
            x_lu = x_pad[:, :, :-self.shift * 2, :-self.shift * 2]
            x_ru = x_pad[:, :, :-self.shift * 2, self.shift * 2:]
            x_lb = x_pad[:, :, self.shift * 2:, :-self.shift * 2]
            x_rb = x_pad[:, :, self.shift * 2:, self.shift * 2:]
            x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)

        # 8 cardinal directions
        if self.mode == self.CARDINAL_8_DIRECTIONS_MODE:
            x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift * 2]
            x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift * 2:]
            x_t2 = x_pad[:, :, :-self.shift * 2, self.shift:-self.shift]
            x_b2 = x_pad[:, :, self.shift * 2:, self.shift:-self.shift]
            x_lu = x_pad[:, :, :-self.shift * 2, :-self.shift * 2]
            x_ru = x_pad[:, :, :-self.shift * 2, self.shift * 2:]
            x_lb = x_pad[:, :, self.shift * 2:, :-self.shift * 2]
            x_rb = x_pad[:, :, self.shift * 2:, self.shift * 2:]
            x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2,
                              x_lu, x_ru, x_lb, x_rb], dim=1)

        out = x_cat

        return out


class ShiftedPatchEmbed2d(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            merging_size=2,
            exist_class_t=False,
            is_pe=False):

        super().__init__()

        self.exist_class_t = exist_class_t
        self.is_pe = is_pe

        self.patch_shifting = PatchShifting(merging_size)

        patch_dim = (in_dim * 5) * (merging_size**2)
        if exist_class_t:
            self.class_linear = nn.Linear(in_dim, out_dim)

        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=merging_size, p2=merging_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, out_dim)
        )

    def forward(self, x):
        if self.exist_class_t:
            visual_tokens, class_token = x[:, 1:], x[:, (0,)]
            reshaped = rearrange(
                visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out_visual = self.patch_shifting(reshaped)
            out_visual = self.merging(out_visual)
            out_class = self.class_linear(class_token)
            out = torch.cat([out_class, out_visual], dim=1)
        else:
            out = x if self.is_pe else rearrange(
                x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out = self.patch_shifting(out)
            out = self.merging(out)

        return out