from .vit import ViT
from .cait import CaiT
from .pit import PiT
from .swin import SwinTransformer
from .t2t import T2T_ViT
from  models.vitAutoEncoder import  ViTAutoEncoder
from  models.vitMaskedAutoEncoder import  ViTMaskedAutoEncoder

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

    elif args.model == 'cait':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, stochastic_depth=args.sd, 
                     is_LSA=args.is_LSA, is_SPT=args.is_SPT)
        
    elif args.model == 'pit':
        patch_size = 2 if img_size == 32 else 4    
        args.channel = 96
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        dim_head = args.channel // args.heads[0]
        
        model = PiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, 
                    mlp_dim_ratio=2, depth=args.depth, heads=args.heads, dim_head=dim_head, 
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA)

    elif args.model =='t2t':
        model = T2T_ViT(img_size=img_size, num_classes=n_classes, drop_path_rate=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA)
        
    elif args.model =='swin':
        depths = [2, 6, 4]
        num_heads = [3, 6, 12]
        mlp_ratio = 2
        window_size = 4
        patch_size = 2 if img_size == 32 else 4
            
        model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=args.sd, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                is_SPT=args.is_SPT, is_LSA=args.is_LSA)
        
    return model