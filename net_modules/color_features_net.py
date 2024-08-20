import torch
import torch.nn as nn
from net_modules.embedder import *
from net_modules.basic_mlp import lin_module


class Color_net(nn.Module):
    def __init__(self,
                fin_dim,
                pin_dim,
                view_dim,
                pfin_dim,
                en_dims,
                de_dims,
                multires,
                cde_dims=None,
                use_pencoding=[False,False],#postion viewdir
                weight_norm=False,
                weight_xavier=True,
                use_drop_out=False,
                use_decode_with_pos=False,
                ):
        super().__init__()
        self.use_pencoding=use_pencoding
        self.embed_fns=[]
        self.cache_outd=None
        self.use_decode_with_pos=use_decode_with_pos
        if use_pencoding[0]:
            embed_fn, input_ch = get_embedder(multires[0])
            pin_dim = input_ch
            self.embed_fns.append(embed_fn)
        else:
            self.embed_fns.append(None)
            
        if use_pencoding[1]:
            embed_fn, input_ch = get_embedder(multires[1])
            view_dim = input_ch
            self.embed_fns.append(embed_fn)
        else:
            self.embed_fns.append(None)
            
        self.encoder=lin_module(
            fin_dim+pin_dim+pfin_dim, fin_dim*3, en_dims, multires[0], # how many? TODO
            act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
        
        self.decoder=lin_module( # inf_dim*4 = dim. of (encoder output + inf)
            fin_dim*4, fin_dim, de_dims, multires[0],
            act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
        
        self.color_decoder=lin_module(
            fin_dim+view_dim, 3, cde_dims, multires[0],
            act_fun=nn.ReLU(), last_act_fun=nn.Sigmoid(), # apply sigmoid at last
            weight_norm=weight_norm, weight_xavier=weight_xavier)
        
        self.use_drop_out=use_drop_out
        if use_drop_out:
            self.drop_outs=[nn.Dropout(0.1)]

                
    def forward(self, inp, inf, inpf, direction=None, inter_weight=1.0, store_cache=False): 
        # inp = point coordinates, inf = intrinsic features, inpf = dynamic features
        p_num=inf.shape[0]
        if self.use_drop_out: inpf = self.drop_outs[0](inpf) 
        if self.use_pencoding[0]: # encode inp if wanted
            inp = self.embed_fns[0](inp)
        if self.use_pencoding[1] and (direction is not None): # encode direction if wanted
            direction = self.embed_fns[1](direction)

        inpf=inpf*inter_weight # if inter_weight = 0, removes dynamic appearance
        inx=torch.cat([inp,inpf,inf],dim=1) 

        oute = self.encoder(inx)
        outd = self.decoder(torch.cat([oute,inf],dim=1))
        self.cache_outd = None
        if store_cache: self.cache_outd=outd # to use forward_cache() later, set store_cache True
        if direction is None: return outd # do not want color for now

        outc = self.color_decoder(torch.cat([outd, direction],dim=1)) 
        return outc


    def forward_cache(self, direction, num_samples=1):
        # repeat forward using same outd (=same features, different angles)
        if self.use_pencoding[1]:
            direction = self.embed_fns[1](direction)
        outd = self.cache_outd # [N,d]
        outd_ = outd[:,None,:].repeat(1,num_samples,1).reshape(-1,outd.shape[-1]) # [NS,d]
        outc = self.color_decoder(torch.cat([outd_,direction],dim=1)) 
        return outc # [NS,3]
        

    

        