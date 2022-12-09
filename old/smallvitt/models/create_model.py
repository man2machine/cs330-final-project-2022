
from models.vit import ViT
from models.vitAutoEncoder import  ViTAutoEncoder
from models.vitMaskedAutoEncoder import  ViTMaskedAutoEncoder
from models.vitMaskedVideoAutoEncoder import ViTMaskedVideoAutoEncoder
from models.vitMaskedVideoEncoderWithHead import ViTMaskedVideoEncoderWithHead
from models.vitEncoder import ViTEncoder

def create_model(img_size, n_classes, args):
    if args.model == 'vit':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA)
    elif args.model == 'vitautoencoder':
        patch_size = 4 if img_size == 32 else 8
        sd = 0.1  # stochastic depth
        is_SPT = False
        is_LSA = False
        model = ViTAutoEncoder(img_size=32, patch_size=patch_size, num_classes=n_classes, dim=192,
                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                stochastic_depth=sd, decoder_dim=96, decoder_depth=3, decoder_heads=16, is_SPT=is_SPT, is_LSA=is_LSA)

    elif args.model == 'vitmaskedautoencoder':
        patch_size = 4 if img_size == 32 else 8
        sd = 0.1  # stochastic depth
        is_SPT = False
        is_LSA = False
        model = ViTMaskedAutoEncoder(img_size=32, patch_size=patch_size, num_classes=n_classes, dim=192,
                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                stochastic_depth=sd, decoder_dim=96, decoder_depth=3, decoder_heads=16, is_SPT=is_SPT, is_LSA=is_LSA)
    elif args.model == 'vitmaskedvideoautoencoder':
        patch_size = 4 if img_size == 32 else 8
        sd = 0.1  # stochastic depth
        is_SPT = False
        is_LSA = False

        model = ViTMaskedVideoAutoEncoder(img_size=args.input_size, frames_depth=args.num_frames, patch_size=patch_size, num_classes=n_classes, dim=192,
                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                stochastic_depth=sd, decoder_dim=96, decoder_depth=3, decoder_heads=16, tubelet_size=2, is_SPT=is_SPT, is_LSA=is_LSA)
    elif args.model == 'vitmaskedvideoencoderwithhead':
        patch_size = 4 if img_size == 32 else 8
        sd = 0.1  # stochastic depth
        is_SPT = False
        is_LSA = False
        model = ViTMaskedVideoEncoderWithHead(img_size=32, frames_depth=16, patch_size=patch_size,
                                                num_classes=n_classes, dim=192,
                                                mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                                                stochastic_depth=sd, tubelet_size=2, is_SPT=is_SPT, is_LSA=is_LSA)
    elif args.model == 'vitencoder':
        patch_size = 4 if img_size == 32 else 8
        sd = 0.1  # stochastic depth
        is_SPT = False
        is_LSA = False
        model = ViTEncoder(img_size=img_size, patch_size=patch_size, num_classes=n_classes, dim=192,
                           mlp_dim_ratio=2, depth=9, heads=12, dim_head=192 // 12,
                           stochastic_depth=0.0, is_SPT=is_SPT, is_LSA=is_LSA, channels=3)

    return model