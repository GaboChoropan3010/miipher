import os

import torch
import torch.nn as nn

from .common import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class PReLU(nn.Module):
    def __init__(self, input_size):
        super(PReLU, self).__init__()
        self.act = torch.nn.PReLU(num_parameters=input_size)
    
    def forward(self, x):
        return self.act(x.transpose(1, 2)).transpose(1, 2)


class DFconformer(nn.Module):
    def __init__(self, input_size, output_size, filters, num_heads=8, dil_rate_base=2, n_per_stack=4, n_layers=12, out_act = None, attn_type=None, expansion_factors=[4,4], attn_mask_prob=0.2, **kwargs):
        #  input_size=80, 
        #  output_size=80, 
        #  filters=256, 
        #  n_layers=12, 
        #  n_per_stack=4,
        #  dil_rate_base=2,
        #  kernel_size=5,
        #  dropout=0.2, 
        #  momentum=0.9, 
        #  expand_factor=[2, 4],
        #  num_heads=8,
        #  out_act="relu",
        #  batch_norm=True,
        #  use_mask=True,
        #  attn_type=None,
        #  use_pos=True,
        #  use_pos_scale=True
        super().__init__()

        self.use_mask = False
        self.use_global_attn_mask = True
        self.cond_as_prefix = False
        self.attn_mask_prob = attn_mask_prob
        
        
        self.dense_inp = nn.Linear(input_size, filters)
        # if self.cond_input_size != 0:
        #     self.dense_cond = nn.Linear(self.cond_input_size, H.filters)
            
        self.blocks = []
        for i in range(n_layers):
            dil_rate = dil_rate_base**(i % n_per_stack)
            # block = ConformerBlock(filters, filters, filters, kernel_size=kernel_size, num_heads=num_heads, dil_rate=dil_rate, dropout=dropout, 
            #                        expansion_factors=expand_factor, momentum=momentum, 
            #                        batch_norm=bah_norm, attn_type=attn_type, use_pos=use_pos, use_pos_scale=use_pos_scale, causal=causal)
            block = ConformerBlock(input_size=input_size, 
                                   filters=filters, 
                                   num_heads = num_heads, 
                                   dil_rate = dil_rate,
                                   attn_type = attn_type,
                                   expansion_factors=expansion_factors)
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)
        
        if out_act=="softplus":
            print("Using softplus for masking.")
            self.out_act = torch.nn.Softplus()
        elif out_act=="sigmoid":
            print("Using sigmoid for masking.")
            self.out_act = torch.nn.Sigmoid()
        elif out_act=="relu":
            print("Using relu for masking.")
            self.out_act = torch.nn.ReLU()
        elif out_act=="prelu":
            print("Using prelu for masking.", output_size)
            self.out_act = PReLU(output_size)
        elif out_act==None:
            self.out_act = lambda x: x
        else:
            print("Mask activation not recognized: {}".format(out_act))
            os._exit(1)

        self.out_dense = nn.Linear(filters, output_size)

    
    def forward(self, inp, condition=None, branch_mask=None, augment_spec=None):

        out = self.dense_inp(inp)
        
        
        if condition is not None:
            condition = self.dense_cond(condition)
            
        # else:
        #     out = self.dense(mels)
         
        self_att_maps = []
        cross_att_maps = []

        N_FRAME = inp.shape[1]
        if condition is not None:
            COND_N_FRAME = condition.shape[1]

        self_attn_mask = None
        cross_attn_mask = None
        if self.use_global_attn_mask and self.training:
            if self.cond_as_prefix:
                self_attn_mask = generate_mask_with_prob((N_FRAME + COND_N_FRAME, N_FRAME + COND_N_FRAME), self.attn_mask_prob, inp.device)
            else:
                self_attn_mask = generate_mask_with_prob((N_FRAME, N_FRAME), self.attn_mask_prob, inp.device)
                if condition is not None:
                    cross_attn_mask = generate_mask_with_prob((N_FRAME, COND_N_FRAME), self.attn_mask_prob, inp.device)

        for i, module in enumerate(self.blocks):
            out_layer, self_att_map, cross_att_map = module(out, condition, self_attn_mask, cross_attn_mask, branch_mask) # batch_size mask
            out = out + out_layer
            self_att_maps.append(self_att_map)
            cross_att_maps.append(cross_att_map)

        if self.use_mask:
            mask = self.out_act(self.out_dense(out))
            out = inp * mask
        else:
            out = self.out_act(self.out_dense(out))
        
        if self.use_mask:
            return out, mask, self_att_maps, cross_att_maps
        else:
            return out, self_att_maps, cross_att_maps


if __name__ == '__main__':

    res = RelativePositionBias(dim=120, heads=4)
    x = res(200)
    print(x.shape)

        
        