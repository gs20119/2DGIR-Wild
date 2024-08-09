

from argparse import ArgumentParser, Namespace
import sys
import os

def argument_init(args):

    args.map_num=args.map_num
    args.feature_maps_dim=16
    args.feature_maps_combine="cat"
    args.use_indep_box_coord=True

    args.map_generator_params={
        "features_dim":args.feature_maps_dim*args.map_num,
        "backbone":"resnet18",
        "use_features_mask":args.use_features_mask,
        "use_independent_mask_branch":args.use_indep_mask_branch
    }
                                     
    args.features_dim=args.feature_maps_dim*args.map_num
    args.features_weight_loss_coef=0.01
    args.color_net_params={
        "fin_dim":5, "pin_dim":3, "view_dim":3, # 5 = base(3)+rough(1)+metal(1)
        "pfin_dim":args.features_dim,
        "en_dims":[128,96,64],
        "de_dims":[48,48],
        "multires":[10,0],
        "cde_dims":[48],
        "use_pencoding":[True,False], #postion viewdir
        "weight_norm":False,
        "weight_xavier":True,
        "use_drop_out":True,
        "use_decode_with_pos":args.use_decode_with_pos,
    }
    
    return args
            