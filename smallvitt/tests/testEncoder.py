import torch
import torch.nn as nn
from  models.vitEncoder import ViTEncoder
def test_VanillaEncoder():
    img_size=32
    patch_size=4
    n_classes=10
    test_tensor = torch.randn(1, 3,32,32)
    sd=0.1 #stochastic depth
    is_SPT=False
    is_LSA=False

    #def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio,
    #             decoder_depth, decoder_heads,
    #             dim_head=16, dropout=0., emb_dropout=0., stochastic_depth=0., is_LSA=False, is_SPT=False,
    #             norm_pix_loss=False, in_chans):

    encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, num_classes=n_classes, dim=192,
                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                stochastic_depth=0.0, is_SPT=is_SPT, is_LSA=is_LSA, channels=3)

    out = encoder(test_tensor)
    print (out.shape)# shape remains same
