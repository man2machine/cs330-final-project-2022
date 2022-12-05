# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:24:03 2022

@author: Shahir, Faraz, Pratyush
Modified from: https://github.com/aanna0701/SPT_LSA_ViT
"""

from cs330_project.models.small_vit.models.vit import ViT
from cs330_project.models.small_vit.models.vit_autoencoder import ViTAutoEncoder
from cs330_project.models.small_vit.models.vit_masked_autoencoder import ViTMaskedAutoEncoder
from cs330_project.models.small_vit.models.vit_masked_video_autoencoder import ViTMaskedVideoAutoEncoder


class ModelType:
    VIT = 'vit'
    VIT_AUTOENCODER = 'vit_autoencoder'
    VIT_MASKED_AUTOENCODER = 'vit_masked_autoencoder'
    VIT_MASKED_VIDEO_AUTOENCODER = 'vit_masked_video_autoencoder'


def create_model(
        img_size,
        n_classes,
        model_type,
        stochastic_depth=0,
        is_spt=False,
        is_las=False):

    if model_type == ModelType.VIT:
        patch_size = 4 if img_size == 32 else 8
        model = ViT(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            mlp_dim_ratio=2,
            depth=9,
            heads=12,
            dim_head=192 // 12,
            stochastic_depth=stochastic_depth,
            is_spt=is_spt,
            is_lsa=is_las)

    elif model_type == ModelType.VIT_AUTOENCODER:
        patch_size = 4 if img_size == 32 else 8
        stochastic_depth = 0.1
        is_spt = False
        is_lsa = False
        model = ViTAutoEncoder(
            img_size=32,
            patch_size=patch_size,
            num_classes=n_classes,
            encoder_dim=192,
            mlp_dim_ratio=2,
            depth=9,
            heads=12,
            head_dim=192 // 12,
            stochastic_depth=stochastic_depth,
            decoder_dim=96,
            decoder_depth=3,
            decoder_num_heads=16,
            is_spt=is_spt,
            is_lsa=is_lsa)

    elif model_type == ModelType.VIT_MASKED_AUTOENCODER:
        patch_size = 4 if img_size == 32 else 8
        stochastic_depth = 0.1
        is_spt = False
        is_lsa = False
        model = ViTMaskedAutoEncoder(
            img_size=32,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            mlp_dim_ratio=2,
            depth=9,
            heads=12,
            dim_head=192 // 12,
            stochastic_depth=stochastic_depth,
            decoder_dim=96,
            decoder_depth=3,
            decoder_heads=16,
            is_spt=is_spt,
            is_lsa=is_lsa)

    elif model_type == ModelType.VIT_MASKED_VIDEO_AUTOENCODER:
        patch_size = 4 if img_size == 32 else 8
        stochastic_depth = 0.1
        is_spt = False
        is_lsa = False
        model = ViTMaskedVideoAutoEncoder(
            img_size=32,
            frames_depth=10,
            patch_size=patch_size,
            num_classes=n_classes,
            dim=192,
            mlp_dim_ratio=2,
            depth=9,
            heads=12,
            dim_head=192 // 12,
            stochastic_depth=stochastic_depth,
            decoder_dim=96,
            decoder_depth=3,
            decoder_heads=16,
            tubelet_size=2,
            is_spt=is_spt,
            is_lsa=is_lsa)

    return model
