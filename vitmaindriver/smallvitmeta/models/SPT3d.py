import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math
import torch.nn.functional as F

class ShiftedPatchEmbed3d(nn.Module):
    def __init__(
            self,
            in_channels,
            embed_dim,
            shift_size,
            tubelet_size,
            has_class_token=True,
            is_bcthw=True):

        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.shift_size = shift_size
        self.tubelet_size = tubelet_size
        self.has_class_token = has_class_token
        self.is_bcthw = is_bcthw
        #(math.pow(9, 3)
        self.patch_shifting = PatchShifting(shift_size)
        print (self.patch_shifting.expansion)
        shift_scale = math.pow(shift_size, 2)
        self.patch_dim = int(in_channels * self.patch_shifting.expansion * shift_scale * tubelet_size)



        if has_class_token:
            self.class_linear = nn.Linear(in_channels, embed_dim)

        self.proj = nn.Sequential(
            Rearrange(
                'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                p0=self.tubelet_size,
                p1=self.shift_size,
                p2=self.shift_size),
            nn.Linear(self.patch_dim, self.embed_dim)
        )

    def forward(self, x):
 #       if self.has_class_token:
 #           visual_tokens, class_token = x[:, 1:], x[:, (0,)]
 #           out_visual = rearrange(
 #               visual_tokens, 'b (t h w) d -> b d t h w', h=int(math.sqrt(x.size(1))))
 #       else:
 #           out_visual = x if self.is_bcthw else rearrange(
 #               x, 'b (t h w) d -> b d t h w', h=int(math.sqrt(x.size(1))))
        if self.has_class_token:
            visual_tokens, class_token = x[:, 1:], x[:, (0,)]

        out_visual=x
        b, c, t, h, w = out_visual.shape
        out_visual = out_visual.view(b, -1, h, w)
        out_visual = self.patch_shifting(out_visual)
        out_visual = out_visual.view(b, -1, t, h, w)
        out = self.proj(out_visual)

     #   if self.has_class_token:
     #       out_class = self.class_linear(class_token)
     #       out = torch.cat([out_class, out_visual], dim=1)

        return out

class PatchShifting(nn.Module):
    CARDINAL_4_DIRECTIONS_MODE = 0
    CARDINAL_8_DIRECTIONS_MODE = 1
    DIAGONAL_4_DIRECTIONS_MODE = 2

    def __init__(
            self,
            shift_size,
            mode=0):

        super().__init__()
        self.mode = mode
        self.expansion = 9 if self.mode == self.CARDINAL_8_DIRECTIONS_MODE else 5
        self.half_shift = int(shift_size * (1 / 2))

    def forward(self, x):
        x_pad = F.pad(x, (self.half_shift, self.half_shift,
                      self.half_shift, self.half_shift))

        # 4 cardinal directions
        if self.mode == self.CARDINAL_4_DIRECTIONS_MODE:
            x_l2 = x_pad[:, :, self.half_shift:-
                         self.half_shift, :-self.half_shift * 2]
            x_r2 = x_pad[:, :, self.half_shift:-
                         self.half_shift, self.half_shift * 2:]
            x_t2 = x_pad[:, :, :-self.half_shift *
                         2, self.half_shift:-self.half_shift]
            x_b2 = x_pad[:, :, self.half_shift * 2:,
                         self.half_shift:-self.half_shift]
            x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)

        # 4 diagonal directions
        if self.mode == self.DIAGONAL_4_DIRECTIONS_MODE:
            x_lu = x_pad[:, :, :-self.half_shift * 2, :-self.half_shift * 2]
            x_ru = x_pad[:, :, :-self.half_shift * 2, self.half_shift * 2:]
            x_lb = x_pad[:, :, self.half_shift * 2:, :-self.half_shift * 2]
            x_rb = x_pad[:, :, self.half_shift * 2:, self.half_shift * 2:]
            x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)

        # 8 cardinal directions
        if self.mode == self.CARDINAL_8_DIRECTIONS_MODE:
            x_l2 = x_pad[:, :, self.half_shift:-
                         self.half_shift, :-self.half_shift * 2]
            x_r2 = x_pad[:, :, self.half_shift:-
                         self.half_shift, self.half_shift * 2:]
            x_t2 = x_pad[:, :, :-self.half_shift *
                         2, self.half_shift:-self.half_shift]
            x_b2 = x_pad[:, :, self.half_shift * 2:,
                         self.half_shift:-self.half_shift]
            x_lu = x_pad[:, :, :-self.half_shift * 2, :-self.half_shift * 2]
            x_ru = x_pad[:, :, :-self.half_shift * 2, self.half_shift * 2:]
            x_lb = x_pad[:, :, self.half_shift * 2:, :-self.half_shift * 2]
            x_rb = x_pad[:, :, self.half_shift * 2:, self.half_shift * 2:]
            x_cat = torch.cat([
                x, x_l2, x_r2, x_t2, x_b2,
                x_lu, x_ru, x_lb, x_rb], dim=1)

        out = x_cat

        return out
