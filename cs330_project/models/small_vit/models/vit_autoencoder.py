# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 22:34:32 2022

@author: Shahir, Faraz, Pratyush
Modified from: https://github.com/aanna0701/SPT_LSA_ViT
"""

import torch
from torch import nn


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from cs330_project.models.small_vit.utils.drop_path import DropPath
from cs330_project.models.small_vit.models.shifted_patch_tokenization import ShiftedPatchTokenization
from cs330_project.models.small_vit.models.utils import make_pair_shape


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

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class TransformerInnerMlp(nn.Module):
    def __init__(
            self,
            dim,
            num_patches,
            hidden_dim,
            dropout=0.):

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

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_patches,
            num_heads=8,
            head_dim=64,
            dropout=0.,
            is_lsa=False):

        super().__init__()
        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == dim)
        self.num_patches = num_patches
        self.heads = num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_lsa:
            self.scale = nn.Parameter(self.scale * torch.ones(num_heads))
            self.mask = torch.eye(self.num_patches + 1, self.num_patches + 1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if self.mask is None:
            dots = torch.einsum(
                'b h i d, b h j d -> b h i j', q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(torch.einsum('b h i d, b h j d -> b h i j', q, k),
                             scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches + 1)
        else:
            flops += (self.dim + 2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3


class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 num_patches,
                 depth,
                 num_heads,
                 head_dim,
                 mlp_dim_ratio,
                 dropout=0.,
                 stochastic_depth=0.,
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

    def forward(self, x):
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
            dim,
            depth,
            num_heads,
            mlp_dim_ratio,
            head_dim=16,
            dropout=0.,
            emb_dropout=0.,
            stochastic_depth=0.,
            is_lsa=False,
            is_spt=False):

        super().__init__()
        image_height, image_width = make_pair_shape(in_img_size)
        patch_height, patch_width = make_pair_shape(patch_size)
        self.num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dim = dim

        if not is_spt:
            self.to_patch_embedding = nn.Sequential(
                Rearrange(
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_height,
                    p2=patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )

        else:
            self.to_patch_embedding = ShiftedPatchTokenization(
                in_dim=3,
                out_dim=self.dim,
                patch_size=patch_size,
                is_pe=True)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            self.dim,
            self.num_patches,
            depth,
            num_heads,
            head_dim,
            mlp_dim_ratio,
            dropout,
            stochastic_depth,
            is_lsa=is_lsa)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])

        return x


class ViTDecoder(nn.Module):
    def __init__(
            self,
            *,
            out_img_size,
            in_channels=3,
            patch_size,
            input_dim,
            dim,
            depth,
            num_heads,
            mlp_dim_ratio,
            head_dim=16,
            dropout=0.,
            emb_dropout=0.,
            stochastic_depth=0.,
            is_lsa=False):

        super().__init__()

        image_height, image_width = make_pair_shape(out_img_size)
        patch_height, patch_width = make_pair_shape(patch_size)
        self.num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.input_dim = input_dim
        self.dim = dim
        self.patch_size = patch_size

        self.embedding = nn.Linear(self.input_dim, self.dim, bias=True)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            self.dim,
            self.num_patches,
            depth,
            num_heads,
            head_dim,
            mlp_dim_ratio,
            dropout,
            stochastic_depth,
            is_lsa=is_lsa)
        self.decoder_norm = nn.LayerNorm(self.dim)
        self.decoder_pred = nn.Linear(
            self.dim, patch_size ** 2 * in_channels, bias=True)  # decoder to patch

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token

        return x


class ViTAutoEncoder(nn.Module):
    def __init__(
            self,
            *,
            img_size,
            in_channels=3,
            patch_size,
            encoder_dim,
            encoder_depth,
            encoder_num_heads,
            decoder_depth,
            decoder_num_heads,
            decoder_dim,
            mlp_dim_ratio,
            head_dim=16,
            dropout=0.,
            emb_dropout=0.,
            stochastic_depth=0.,
            is_lsa=False,
            is_spt=False):

        super().__init__()

        self.encoder = ViTEncoder(
            in_img_size=img_size,
            patch_size=patch_size,
            dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_dim_ratio=mlp_dim_ratio,
            in_channels=in_channels,
            head_dim=head_dim,
            drouput=dropout,
            emb_dropout=emb_dropout,
            stochastic_depth=stochastic_depth,
            is_lsa=is_lsa,
            is_spt=is_spt
        )

        self.decoder = ViTDecoder(
            out_img_size=img_size,
            patch_size=patch_size,
            input_dim=encoder_dim,
            mlp_dim_ratio=mlp_dim_ratio,
            in_channels=in_channels,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            dim=decoder_dim,
            head_dim=head_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            stochastic_depth=stochastic_depth,
            is_lsa=is_lsa
        )

    def forward(self, img):
        latent = self.encoder(img)
        pred = self.decoder(latent)

        return latent, pred


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

    def forward(self, x):
        return self.net(x)


class ViTClassifier:
    def __init__(
            self,
            *,
            img_size,
            in_channels=3,
            patch_size,
            num_classes,
            dim,
            depth,
            num_heads,
            mlp_dim_ratio,
            head_dim=16,
            dropout=0.,
            emb_dropout=0.,
            stochastic_depth=0.,
            is_lsa=False,
            is_spt=False):

        self.encoder = ViTEncoder(
            in_img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim_ratio=mlp_dim_ratio,
            in_channels=in_channels,
            head_dim=head_dim,
            drouput=dropout,
            emb_dropout=emb_dropout,
            stochastic_depth=stochastic_depth,
            is_lsa=is_lsa,
            is_spt=is_spt
        )

        self.head = ViTClassifierHead(
            input_dim=dim,
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)

        return x
