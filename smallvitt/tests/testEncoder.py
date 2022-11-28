import torch
import torch.nn as nn
from smallvitt.models.vitEncoder import ViTEncoder
def test_VanillaEncoder():
    img_size=32
    patch_size=4
    n_classes=10
    test_tensor = torch.randn(1, 3,32,32)
    sd=0.1 #stochastic depth
    is_SPT=False
    is_LSA=False



    encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, num_classes=n_classes, dim=192,
                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                stochastic_depth=0.0, is_SPT=is_SPT, is_LSA=is_LSA, channels=3)

    out = encoder(test_tensor)
    print (out.shape)# shape remains same
