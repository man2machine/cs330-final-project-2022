# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 23:02:41 2022

@author: Shahir, Faraz, Pratyush
Modified from: https://github.com/aanna0701/SPT_LSA_ViT
"""

import math

def autoencoder_loss(x_patched, x_hat_patched, mask_info=None, norm_pix_loss=False):
    if norm_pix_loss:
        mean = x_patched.mean(dim=-1, keepdim=True)
        var = x_patched.var(dim=-1, keepdim=True)
        x_patched = (x_patched - mean) / math.sqrt(var + 1.e-6)

    loss = (x_hat_patched - x_patched) ** 2 # [b, n, patch_dim]
    loss = loss.mean(dim=-1) # [b, n], mean loss per patch
    if mask_info is not None:
        mask = mask_info[0]
        loss = (loss * mask).sum() / mask.sum() # mean loss on removed patches
    else:
        loss = loss.sum()
    return loss / (x_patched.size(0) * x_patched.size(1))
