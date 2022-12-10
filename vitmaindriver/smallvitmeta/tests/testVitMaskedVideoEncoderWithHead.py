import torch
import torch.nn as nn
from  vitmaindriver.smallvitmeta.models.vitMaskedVideoEncoderWithHead import  ViTMaskedVideoEncoderWithHead
def test_AutoEncoder():
    img_size=32
    patch_size=4
    n_classes=26
    # (B, C_{in}, T, H, W)  d=Depth means number of frames in video N is batch size C_{in} are channels in
    test_tensor = torch.randn(4, 3, 16,32,32) # we have sampled 16 frames from video with batch size of 1
    B = test_tensor.shape[0]
    sd=0.1 #stochastic depth
    is_SPT=False
    is_LSA=False
    #def __init__(self, img_size=224, patch_size=16, in_chans=3,
    #             embed_dim=1024, depth=24, num_heads=16,
    #              decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
    #             mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
    #frames depth should match dimension 1 of above tensor
    encoder = ViTMaskedVideoEncoderWithHead(img_size=32, frames_depth=16, patch_size=patch_size, num_classes=n_classes, dim=192,
                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                stochastic_depth=sd, tubelet_size=2, is_SPT=is_SPT, is_LSA=is_LSA)

    out = encoder.forward(test_tensor)
    assert out.shape[0] == B
    assert out.shape[1] == n_classes
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([1,4,5,2])
    loss = criterion(out, target)
    print(loss)

    #now calculate loss




