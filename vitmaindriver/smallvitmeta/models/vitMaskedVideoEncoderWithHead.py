# based onhttps://github.com/aanna0701/SPT_LSA_ViT
#@author:  Faraz, Shahir, Pratyush
import torch
from torch import nn, einsum
from vitmaindriver.smallvitmeta.utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .SPT import ShiftedPatchTokenization




# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout=0.):
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
    def __init__(self, dim, num_patches, heads=8, dim_head=64, dropout=0., is_LSA=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        init_weights(self.to_qkv)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_LSA:
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = torch.eye(self.num_patches + 1, self.num_patches + 1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k),
                             scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches + 1)
        else:
            flops += (self.dim + 2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3

#  def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., attn_head_dim=None):
class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout=0., stochastic_depth=0.,
                 is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}


        #AttentionST from VideoMAE code based can be plugged-in by replacing Attention by AttentionST
        #num_heads
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim,
                        Attention(dim, num_patches, heads=heads, dim_head=dim_head, dropout=dropout, is_LSA=is_LSA)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout=dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()



    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
            self.scale[str(i)] = attn.fn.scale
        return x


class ViTMaskedVideoEncoderWithHead(nn.Module):
    def __init__(self, *, img_size, patch_size, frames_depth, num_classes, dim, depth, heads,  mlp_dim_ratio, channels=3,
                 dim_head=16, dropout=0., emb_dropout=0., stochastic_depth=0., tubelet_size=2, is_LSA=False, is_SPT=False,norm_pix_loss=False, in_chans=3):

        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.image_height,  self.image_width= pair(img_size)
        self.num_patches = int( (image_height // patch_height) * (image_width // patch_width) * (frames_depth/tubelet_size))
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.frames_depth = frames_depth
        self.tubelet_size= tubelet_size
        self.numChannels = in_chans
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        is_SPT=True

        self.to_patch_embedding = PatchEmbed(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim, tubelet_size=tubelet_size)

        self.pos_embedding =  nn.Parameter(torch.zeros(1, self.num_patches + 1, self.dim), requires_grad=False)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout,
                                       stochastic_depth, is_LSA=is_LSA)

        self.norm = nn.LayerNorm(self.dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )

        self.norm_pix_loss = norm_pix_loss

        self.apply(init_weights)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward(self, img, mask_ratio=0.85):
        # patch embedding

        #encoder

        #_, _, T, _, _ = x.shape
        #x = self.patch_embed(x)
        x = self.to_patch_embedding(img)
        # B, number of tokens, latent dim (shape)
        #B, _, C = x.shape
        b, n, _ = x.shape

        x = x + self.pos_embedding[:, 1:, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)


        x = self.dropout(x)

        x = self.transformer(x)

        x = self.norm(x)
        return self.mlp_head(x[:, 0])





    def patchify(self, imgs):
        """
        1 320 96
        imgs: (B, C, T, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[3] == imgs.shape[4] and imgs.shape[3] % p == 0
        d=imgs.shape[2] # frames depth we are conisdering tube as one unit from where we want to extract tokens.

        h = w = imgs.shape[3] // p
        frames = int  (self.frames_depth /self.tubelet_size )
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p, d))
        x = torch.einsum('nchpdwq->nhwpdqc', x)

        x = x.reshape(shape=(imgs.shape[0], h * w * frames, p**2 * 3 * self.tubelet_size ))
        return x

class PatchEmbed(nn.Module):
    """ Image to  Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = pair(img_size)
        patch_size = pair(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]),
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x