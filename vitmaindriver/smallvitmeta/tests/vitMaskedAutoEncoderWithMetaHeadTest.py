#@author:  Faraz, Shahir, Pratyush
import torch
import torch.nn as nn
from  smallvitt.models.vitMaskedVideoEncoderWithMetaHead import  VitMaskedEncoderWithMetaHead
from  smallvitt.models.vitMaskedVideoEncoder import  ViTMaskedVideoEncoder
from einops import rearrange, repeat

DEVICE = torch.device("cpu")
print(DEVICE)
def test_AutoEncoder():
    img_size=32
    patch_size=4
    n_classes=3
    #K are shots size of support set
    #N is n-way of number of classes
    # (B, N, K+1, C_{in}, T, H, W)  T=Depth means number of frames in video N is batch size C_{in} are channels in
    test_tensor = torch.randn(2, 2, 3, 3, 16,32,32) # we have sampled 16 frames from video with batch size of 1
    B = test_tensor.shape[0]
    sd=0.1 #stochastic depth
    is_SPT=False
    is_LSA=False

    encoder = ViTMaskedVideoEncoder(img_size=32, frames_depth=16, patch_size=patch_size, num_classes=n_classes, dim=192,
                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                stochastic_depth=sd, tubelet_size=2, is_SPT=is_SPT, is_LSA=is_LSA)


    k=1

    hidden_dim=128
    #number of labels and images per class pass are k+1
    model = VitMaskedEncoderWithMetaHead(num_classes=n_classes, vitmaskedvideoautoencoder=encoder, samples_per_class=k, hidden_dim=128, num_layers=1, rnn_type="lstm")
    # labels (B, K+1*  N) where n are n-ways and K is support set k+1 is support + query set
    labels = torch.as_tensor([[ [2,1,0], [0,1,2]], [[2,0,1],[ 1, 0,2] ]])
    print(labels.shape)
    model.to(DEVICE)


    out = model.forward(test_tensor, labels)
    print(out)
    loss = model.loss_function(out, labels)
    print(loss)

    #now calculate loss




