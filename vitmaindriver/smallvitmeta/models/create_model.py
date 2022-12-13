# based onhttps://github.com/aanna0701/SPT_LSA_ViT
#@author:  Faraz, Shahir, Pratyush
from models.vitMaskedVideoAutoEncoder import ViTMaskedVideoAutoEncoder
from models.vitMaskedVideoEncoderWithHead import ViTMaskedVideoEncoderWithHead


def create_model(img_size, n_classes, args):

    if args.model == 'vitmaskedvideoautoencoder':
        patch_size = 4 if args.input_size == 32 else 8
        sd = 0.1  # stochastic depth
        is_SPT = args.is_SPT
        is_LSA = True

        model = ViTMaskedVideoAutoEncoder(img_size=args.input_size, frames_depth=args.num_frames, patch_size=patch_size, num_classes=n_classes, dim=192,
                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                stochastic_depth=sd, decoder_dim=96, decoder_depth=3, decoder_heads=16, tubelet_size=2, is_SPT=is_SPT, is_LSA=is_LSA)
    elif args.model == 'vitmaskedvideoencoderwithhead':
        patch_size = 4 if args.input_size == 32 else 8
        sd = 0.1  # stochastic depth
        is_SPT = args.is_SPT
        is_LSA = True
        model = ViTMaskedVideoEncoderWithHead(img_size=args.input_size, frames_depth=16, patch_size=patch_size,
                                                num_classes=n_classes, dim=192,
                                                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                                                stochastic_depth=sd, tubelet_size=2, is_SPT=is_SPT, is_LSA=is_LSA)


    return model