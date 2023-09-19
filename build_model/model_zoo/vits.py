import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import math
from functools import partial, reduce
import operator
import timm 


class vit_tiny_rpc(nn.Module):
    def __init__(self, ispretrained, n_cls, body_partitions, init_par):
        super().__init__()
        # choise of the model
        vit_tiny = timm.create_model('vit_tiny_patch16_224', ispretrained)    

        # replace the last layer
        vit_tiny.head = nn.Linear(vit_tiny.head.weight.shape[1], n_cls)

        # freeze the last layer 
        for p in vit_tiny.head.parameters():
            p.requires_grad_(False)

        # repack the model
        self.embedding = nn.Sequential(vit_tiny.patch_embed, vit_tiny.pos_drop, vit_tiny.atch_drop, vit_tiny.norm_pre) 

        MHA_0 = nn.Sequential(vit_tiny.blocks[0].norm1, vit_tiny.blocks[0].attn, vit_tiny.blocks[0].ls1, vit_tiny.blocks[0].drop_path1, vit_tiny.blocks[0].norm2)
        MLP_0 = nn.Sequential(vit_tiny.blocks[0].mlp, vit_tiny.blocks[0].ls2, vit_tiny.blocks[0].drop_path2)

        MHA_6 = nn.Sequential(vit_tiny.blocks[6].norm1, vit_tiny.blocks[6].attn, vit_tiny.blocks[6].ls1, vit_tiny.blocks[6].drop_path1, vit_tiny.blocks[6].norm2)
        MLP_6 = nn.Sequential(vit_tiny.blocks[6].mlp, vit_tiny.blocks[6].ls2, vit_tiny.blocks[6].drop_path2)

        MHA_11 = nn.Sequential(vit_tiny.blocks[11].norm1, vit_tiny.blocks[11].attn, vit_tiny.blocks[11].ls1, vit_tiny.blocks[11].drop_path1, vit_tiny.blocks[11].norm2)
        MLP_11 = nn.Sequential(vit_tiny.blocks[11].mlp, vit_tiny.blocks[11].ls2, vit_tiny.blocks[11].drop_path2)

        block_0 = nn.Sequential(MHA_0, MLP_0)
        block_6 = nn.Sequential(MHA_6, MLP_6)
        block_11 = nn.Sequential(MHA_11, MLP_11)

        blocks_upper = nn.Sequential(block_0, vit_tiny.blocks[1:6])
        blocks_lower = nn.Sequential(block_6, vit_tiny.blocks[7:11], block_11)
        self.blocks = nn.Sequential(blocks_upper, blocks_lower)

        self.cls_head = nn.Sequential(vit_tiny.norm, vit_tiny.fc_norm, vit_tiny.head_drop, vit_tiny.head)

        # weight initalization 
        # self._init_parameters(init_par)

    def forward(self, x):
        x = self.embedding(x) 
        x = self.body(x)
        x = self.cls_head(x)
        return x

    def _init_parameters(self, p):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0, 1/math.sqrt(m.weight.shape[0]**(1+p)))
