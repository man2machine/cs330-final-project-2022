# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 22:34:32 2022

@author: Shahir, Faraz, Pratyush
Modified from: https://github.com/aanna0701/SPT_LSA_ViT
"""

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from cs330_project.utils import make_pair_shape


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(
            self,
            drop_prob=None):

        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def drop_path(
            x,
            drop_prob=0.0,
            training=False):
        """
        Obtained from: github.com:rwightman/pytorch-image-models
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        # issuecomment-532968956 ... I've opted for
        See discussion: https://github.com/tensorflow/tpu/issues/494
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.
        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(
            self,
            x):

        return self.drop_path(x, self.drop_prob, self.training)


class PatchShifting2d(nn.Module):
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
        self.half_shift = shift_size // 2

    def forward(
            self,
            x):

        # x is shape [b, c, h, w]
        x_pad = F.pad(
            x,
            (self.half_shift, self.half_shift,
             self.half_shift, self.half_shift))

        # 4 cardinal directions
        if self.mode == self.CARDINAL_4_DIRECTIONS_MODE:
            x_l2 = x_pad[:, :, self.half_shift:-
                         self.half_shift, : -self.half_shift * 2]
            x_r2 = x_pad[:, :, self.half_shift: -
                         self.half_shift, self.half_shift * 2:]
            x_t2 = x_pad[:, :, : -self.half_shift *
                         2, self.half_shift: -self.half_shift]
            x_b2 = x_pad[:, :, self.half_shift * 2:,
                         self.half_shift: -self.half_shift]
            x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)

        # 4 diagonal directions
        if self.mode == self.DIAGONAL_4_DIRECTIONS_MODE:
            x_lu = x_pad[:, :, : -self.half_shift * 2, : -self.half_shift * 2]
            x_ru = x_pad[:, :, : -self.half_shift * 2, self.half_shift * 2:]
            x_lb = x_pad[:, :, self.half_shift * 2:, : -self.half_shift * 2]
            x_rb = x_pad[:, :, self.half_shift * 2:, self.half_shift * 2:]
            x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)

        # 8 cardinal directions
        if self.mode == self.CARDINAL_8_DIRECTIONS_MODE:
            x_l2 = x_pad[:, :, self.half_shift: -
                         self.half_shift, : -self.half_shift * 2]
            x_r2 = x_pad[:, :, self.half_shift: -
                         self.half_shift, self.half_shift * 2:]
            x_t2 = x_pad[:, :, : -self.half_shift *
                         2, self.half_shift: -self.half_shift]
            x_b2 = x_pad[:, :, self.half_shift * 2:,
                         self.half_shift: -self.half_shift]
            x_lu = x_pad[:, :, : -self.half_shift * 2, : -self.half_shift * 2]
            x_ru = x_pad[:, :, : -self.half_shift * 2, self.half_shift * 2:]
            x_lb = x_pad[:, :, self.half_shift * 2:, : -self.half_shift * 2]
            x_rb = x_pad[:, :, self.half_shift * 2:, self.half_shift * 2:]
            x_cat = torch.cat([
                x, x_l2, x_r2, x_t2, x_b2,
                x_lu, x_ru, x_lb, x_rb], dim=1)

        out = x_cat  # [b, c * self.expansion, h, w]

        return out


class PatchShifting3d(PatchShifting2d):
    def forward(
            self,
            x):

        b, c, t, h, w = x.shape
        x = x.view(b, c * t, h, w)
        x = super().forward(x)
        x = x.view(b, c * self.expansion, t, h, w)

        return x


class ShiftedPatchEmbed2d(nn.Module):
    def __init__(
            self,
            in_img_size,
            in_channels,
            embed_dim,
            patch_size):

        super().__init__()

        self.in_img_size = in_img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_shifting = PatchShifting2d(patch_size)
        self.patch_dim = (
            self.in_channels *
            self.patch_shifting.expansion *
            (self.patch_size ** 2))

        self.num_patches = (
            (self.in_img_size[0] // self.patch_size) * (self.in_img_size[1] // self.patch_size))

        self.patchify = nn.Sequential(
            self.patch_shifting,
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=self.patch_size,
                p2=self.patch_size)
        )
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(
            self,
            x):

        # x is shape [b, c, h, w]
        x = self.patchify(x)
        x = self.proj(x)

        return x


class ShiftedPatchEmbed3d(nn.Module):
    def __init__(
            self,
            in_img_size,
            in_channels,
            in_num_frames,
            embed_dim,
            patch_size,
            tubelet_size):

        super().__init__()

        self.in_img_size = in_img_size
        self.in_channels = in_channels
        self.in_num_frames = in_num_frames
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.patch_shifting = PatchShifting3d(patch_size)
        self.patch_dim = (
            self.in_channels *
            self.patch_shifting.expansion *
            (self.patch_size ** 2) *
            self.tubelet_size)

        self.num_patches = ((self.in_num_frames // self.tubelet_size) * (
            self.in_img_size[0] // self.patch_size) * (self.in_img_size[1] // self.patch_size))

        self.patchify = nn.Sequential(
            self.patch_shifting,
            Rearrange(
                'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                p0=self.tubelet_size,
                p1=self.patch_size,
                p2=self.patch_size)
        )
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

    def forward(
            self,
            x):

        # x is shape [b, c, t, h, w]
        x = self.patchify(x)
        x = self.proj(x)

        return x


class PatchEmbed2d(nn.Module):
    def __init__(
            self,
            in_img_size,
            in_channels,
            patch_height,
            patch_width,
            embed_dim):

        super().__init__()

        self.in_img_size = in_img_size
        self.in_channels = in_channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_dim = in_channels * patch_height * patch_width
        self.embed_dim = embed_dim

        self.num_patches = (
            (self.in_img_size[0] // patch_height) * (self.in_img_size[1] // patch_width))

        self.patchify = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_height,
            p2=self.patch_width)
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

    def forward(
            self,
            x):

        # x is shape [b, c, h, w]
        x = self.patchify(x)
        x = self.proj(x)

        return x


class PatchEmbed3d(nn.Module):
    def __init__(
            self,
            in_img_size,
            in_channels,
            in_num_frames,
            patch_height,
            patch_width,
            tubelet_size,
            embed_dim):

        super().__init__()

        self.in_img_size = in_img_size
        self.in_channels = in_channels
        self.in_num_frames = in_num_frames
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.tubelet_size = tubelet_size
        self.patch_dim = in_channels * patch_height * patch_width * tubelet_size
        self.embed_dim = embed_dim

        self.num_patches = ((in_num_frames // tubelet_size) * (
            self.in_img_size[0] // patch_height) * (self.in_img_size[1] // patch_width))

        self.patchify = Rearrange(
            'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
            p0=self.tubelet_size,
            p1=self.patch_height,
            p2=self.patch_width)
        self.proj = nn.Linear(self.patch_dim, self.embed_dim)

    def forward(
            self,
            x):

        # x is shape [b, c, t, h, w]
        x = self.patchify(x)
        x = self.proj(x)

        return x


class PreNorm(nn.Module):
    def __init__(
            self,
            num_tokens,
            dim,
            fn):

        super().__init__()

        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(
            self,
            x,
            **kwargs):

        return self.fn(self.norm(x), **kwargs)


class TransformerInnerMlp(nn.Module):
    def __init__(
            self,
            dim,
            num_patches,
            hidden_dim,
            dropout=0.0):

        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            x):

        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_patches,
            num_heads=8,
            head_dim=64,
            dropout=0.0,
            is_lsa=False):

        super().__init__()
        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == dim)
        self.num_patches = num_patches
        self.heads = num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.softmax = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)

        self.projection = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_lsa:
            self.scale = nn.Parameter(self.scale * torch.ones(num_heads))
            self.mask = torch.eye(self.num_patches + 1, self.num_patches + 1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(
            self,
            x):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if self.mask is None:
            dots = torch.einsum(
                'b h i d, b h j d -> b h i j', q, k) * self.scale
        else:
            scale = self.scale
            dots = torch.mul(
                torch.einsum('b h i d, b h j d -> b h i j', q, k),
                scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            mask_value = max_neg_value(dots)
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = mask_value

        attn = self.softmax(dots)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.projection(out)

    def get_num_flops(
            self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches + 1)
        else:
            flops += (self.dim + 2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3

        return flops


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            num_patches,
            depth,
            num_heads,
            head_dim,
            mlp_dim_ratio,
            dropout=0.0,
            stochastic_depth=0.0,
            is_lsa=False):

        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(
                    num_tokens=num_patches,
                    dim=dim,
                    fn=Attention(
                        dim=dim,
                        num_patches=num_patches,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        is_lsa=is_lsa)),
                PreNorm(
                    num_tokens=num_patches,
                    dim=dim,
                    fn=TransformerInnerMlp(
                        dim=dim, num_patches=num_patches,
                        hidden_dim=dim * mlp_dim_ratio,
                        dropout=dropout))
            ]))

        self.drop_path = DropPath(
            stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(
            self,
            x):

        for i, (attn, ff) in enumerate(self.layers):
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
            self.scale[str(i)] = attn.fn.scale

        return x


class ViTEncoder(nn.Module):
    def __init__(
            self,
            *,
            in_img_size,
            in_channels=3,
            patch_size,
            spatio_temporal=False,
            in_num_frames=None,
            tubelet_size=None,
            embed_dim,
            depth,
            num_heads,
            mlp_dim_ratio,
            head_dim=16,
            dropout=0.0,
            pos_embed_dropout=0.0,
            stochastic_depth=0.0,
            class_embed=False,
            is_lsa=False,
            is_spt=False):

        super().__init__()

        in_img_size = make_pair_shape(in_img_size)
        patch_height, patch_width = make_pair_shape(patch_size)
        self.embed_dim = embed_dim
        self.class_embed = bool(class_embed)
        self.spatio_temporal = spatio_temporal
        self.tublet_size = tubelet_size

        if not is_spt:
            if self.spatio_temporal:
                self.patch_embedder = PatchEmbed3d(
                    in_img_size=in_img_size,
                    in_channels=in_channels,
                    in_num_frames=in_num_frames,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    tubelet_size=tubelet_size,
                    embed_dim=embed_dim)
            else:
                self.patch_embedder = PatchEmbed2d(
                    in_img_size=in_img_size,
                    in_channels=in_channels,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    embed_dim=embed_dim)
        else:
            assert patch_height == patch_width
            patch_size = patch_width
            if self.spatio_temporal:
                self.patch_embedder = ShiftedPatchEmbed3d(
                    in_img_size=in_img_size,
                    in_channels=in_channels,
                    in_num_frames=in_num_frames,
                    embed_dim=self.embed_dim,
                    patch_size=patch_size,
                    tubelet_size=tubelet_size)
            else:
                self.patch_embedder = ShiftedPatchEmbed2d(
                    in_img_size=in_img_size,
                    in_channels=in_channels,
                    embed_dim=self.embed_dim,
                    patch_size=patch_size)
        self.num_patches = self.patch_embedder.num_patches
        self.patch_dim = self.patch_embedder.patch_dim

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + self.class_embed, self.embed_dim))
        if self.class_embed:
            self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.embed_dropout = nn.Dropout(pos_embed_dropout)

        self.transformer = Transformer(
            self.embed_dim,
            self.num_patches,
            depth,
            num_heads,
            head_dim,
            mlp_dim_ratio,
            dropout,
            stochastic_depth,
            is_lsa=is_lsa)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            x,
            masked_indices=None):

        x = self.patch_embedder(x)
        b, n, d = x.shape
        
        if masked_indices is not None:
            x = torch.gather(
                x, dim=1, index=masked_indices.unsqueeze(-1).repeat(1, 1, d)) 

        if self.class_embed:
            class_tokens = repeat(self.class_token, '() n d -> b n d', b=b)
            x = torch.cat((class_tokens, x), dim=1)

        x = x + self.pos_embedding

        x = self.embed_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)

        if self.class_embed:
            # remove cls token
            x = x[:, 1:, :]

        return x


class ViTDecoder(nn.Module):
    def __init__(
            self,
            *,
            in_latent_dim,
            in_num_patches,
        out_patch_dim,
            embed_dim,
            depth,
            num_heads,
            mlp_dim_ratio,
            head_dim=16,
            dropout=0.0,
            pos_embed_dropout=0.0,
            stochastic_depth=0.0,
            class_embed=False,
            use_masking=False,
            is_lsa=False):

        super().__init__()

        self.out_patch_dim = out_patch_dim
        self.in_latent_dim = in_latent_dim
        self.in_num_patches = in_num_patches

        self.input_dim = in_latent_dim
        self.embed_dim = embed_dim
        self.class_embed = bool(class_embed)
        self.use_masking = bool(use_masking)

        self.encoder_to_decoder = nn.Linear(
            self.input_dim, self.embed_dim, bias=True)

        if self.class_embed:
            self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        if self.use_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.in_num_patches + self.class_embed + self.use_masking, self.embed_dim))
        self.embed_dropout = nn.Dropout(pos_embed_dropout)

        self.transformer = Transformer(
            self.embed_dim,
            self.in_num_patches,
            depth,
            num_heads,
            head_dim,
            mlp_dim_ratio,
            dropout,
            stochastic_depth,
            is_lsa=is_lsa)

        self.norm = nn.LayerNorm(self.embed_dim)
        self.pred = nn.Linear(
            self.embed_dim, self.out_patch_dim, bias=True)  # decoder to patch

    def forward(
            self,
            x,
            unshuffle_indices=None):

        # [b, n, latent_dim]
        x = self.encoder_to_decoder(x)
        b, n, d = x.shape

        if self.use_masking:
            x = torch.cat((x, self.mask_token.expand(b, -1, -1)), dim=1)
            x = torch.gather(
                x, dim=1, index=unshuffle_indices.unsqueeze(-1).repeat(1, 1, d))  # unshuffle

        if self.class_embed:
            class_tokens = repeat(self.class_token, '() n d -> b n d', b=b)
            x = torch.cat((class_tokens, x), dim=1)

        x = x + self.pos_embedding
        x = self.embed_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)  # [b, n, d]
        x = self.pred(x)  # [b, n, c * p0 * p1 * p2]

        if self.class_embed:
            # remove cls token
            x = x[:, 1:, :]

        return x


class ViTAutoEncoder(nn.Module):
    def __init__(
            self,
            *,
            in_img_size,
            in_channels=3,
            patch_size,
            spatio_temporal=False,
            in_num_frames=None,
            tubelet_size=None,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            mlp_dim_ratio,
            head_dim=16,
            dropout=0.0,
            pos_embed_dropout=0.0,
            stochastic_depth=0.0,
            class_embed=False,
            is_lsa=False,
            is_spt=False,
            use_masking=False):

        super().__init__()

        self.use_masking = use_masking
        if self.use_masking:
            self.mask_token = nn.Parameter(
                torch.zeros(1, 1, decoder_embed_dim))

        self.encoder = ViTEncoder(
            in_img_size=in_img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            spatio_temporal=spatio_temporal,
            in_num_frames=in_num_frames,
            tubelet_size=tubelet_size,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_dim_ratio=mlp_dim_ratio,
            head_dim=head_dim,
            dropout=dropout,
            pos_embed_dropout=pos_embed_dropout,
            stochastic_depth=stochastic_depth,
            class_embed=class_embed,
            is_lsa=is_lsa,
            is_spt=is_spt)

        self.decoder = ViTDecoder(
            in_latent_dim=encoder_embed_dim,
            in_num_patches=self.encoder.num_patches,
            out_patch_dim=self.encoder.patch_dim,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_dim_ratio=mlp_dim_ratio,
            head_dim=head_dim,
            dropout=dropout,
            pos_embed_dropout=pos_embed_dropout,
            stochastic_depth=stochastic_depth,
            class_embed=class_embed,
            use_masking=use_masking,
            is_lsa=is_lsa)

    def forward(
            self,
            x,
            mask_info=None):

        if self.use_masking:
            _, _, unshuffle_indices, masked_indices = mask_info
        else:
            unshuffle_indices, masked_indices = None, None

        x_patched = self.encoder.patch_embedder.patchify(x)
        latent_patched = self.encoder(x, masked_indices=masked_indices)
        x_hat_patched = self.decoder(
            latent_patched, unshuffle_indices=unshuffle_indices)

        return latent_patched, x_patched, x_hat_patched


class ViTClassifierHead(nn.Module):
    def __init__(
            self,
            input_dim,
            num_classes):

        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.num_classes)
        )

    def forward(
            self,
            x):

        return self.net(x)


class ViTClassifier(nn.Module):
    def __init__(
            self,
            *,
            in_img_size,
            in_channels=3,
            patch_size,
            spatio_temporal=False,
            in_num_frames=None,
            tubelet_size=None,
            num_classes,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            mlp_dim_ratio,
            head_dim=16,
            dropout=0.0,
            pos_embed_dropout=0.0,
            head_dropout=0.0,
            stochastic_depth=0.0,
            class_embed=False,
            is_lsa=False,
            is_spt=False,
            use_masking=False):
            
        super().__init__() 
        
        self.use_masking = use_masking

        self.encoder = ViTEncoder(
            in_img_size=in_img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            spatio_temporal=spatio_temporal,
            in_num_frames=in_num_frames,
            tubelet_size=tubelet_size,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_dim_ratio=mlp_dim_ratio,
            head_dim=head_dim,
            dropout=dropout,
            pos_embed_dropout=pos_embed_dropout,
            stochastic_depth=stochastic_depth,
            class_embed=class_embed,
            is_lsa=is_lsa,
            is_spt=is_spt)
            
        self.head_norm = nn.LayerNorm(encoder_embed_dim)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = ViTClassifierHead(
            input_dim=encoder_embed_dim,
            num_classes=num_classes)

    def forward(
            self,
            x,
            mask_info):
            
        if self.use_masking:
            _, _, _, masked_indices = mask_info
        else:
            masked_indices = None

        latent = self.encoder(x, masked_indices)
        
        print(latent.shape)
        latent = latent.mean(dim=1)  # patch-wise average pooling
        print(latent.shape)
        latent = self.head_norm(latent)
        latent = self.head_dropout(latent)
        print(latent.shape)

        pred = self.head(latent)

        return pred
