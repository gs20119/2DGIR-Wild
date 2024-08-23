

from argparse import ArgumentParser, Namespace
import sys
import os

def argument_init(args):

    args.map_num=args.map_num
    args.feature_maps_dim=16
    args.feature_maps_combine="cat"
    args.use_indep_box_coord=True
    args.dynamic_features_dim = args.feature_maps_dim * args.map_num

    args.map_generator_params={
        "features_dim":args.dynamic_features_dim,
        "backbone":"resnet18",
        "use_features_mask":args.use_features_mask,
        "use_independent_mask_branch":args.use_indep_mask_branch
    }
                                     
    args.features_weight_loss_coef=0.01
    args.color_net_params={
        "fin_dim":5, "pin_dim":3, "view_dim":3, 
        "pfin_dim":args.dynamic_features_dim,
        "en_dims":[64,48,32], # reduced size of MLP 
        "de_dims":[24,24],
        "multires":[10,0],
        "cde_dims":[48], 
        "use_pencoding":[True,False], # encode postion, viewdir
        "weight_norm":False,
        "weight_xavier":True,
        "use_drop_out":True,
        "use_decode_with_pos":args.use_decode_with_pos,
    }
    
    return args
            