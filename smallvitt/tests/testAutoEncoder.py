import torch
import torch.nn as nn
from  models.vitAutoEncoder import  ViTAutoEncoder
def test_AutoEncoder():
    img_size=32
    patch_size=4
    n_classes=10
    test_tensor = torch.randn(1, 3,32,32)
    sd=0.1 #stochastic depth
    is_SPT=False
    is_LSA=False

    #def __init__(self, img_size=224, patch_size=16, in_chans=3,
    #             embed_dim=1024, depth=24, num_heads=16,
    #             decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #             mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):

    encoder = ViTAutoEncoder(img_size=32, patch_size=patch_size, num_classes=n_classes, dim=192,
                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                stochastic_depth=sd, decoder_dim=96, decoder_depth=3, decoder_heads=16, is_SPT=is_SPT, is_LSA=is_LSA)

    out = encoder.forward(test_tensor)
    images = encoder.unpatchify(out)

    assert images.shape == test_tensor.shape# shape remains same
