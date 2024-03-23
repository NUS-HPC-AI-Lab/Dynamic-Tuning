import torch
import fvcore
from fvcore.nn import FlopCountAnalysis
import argparse
import models
from models.vision_transformer_IN21K import Block
from timm.models import create_model
from easydict import EasyDict

import os
from functools import partial
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Callable, Optional, Tuple, Union
from timm.layers import PatchEmbed, Mlp,  PatchDropout
from models.vision_transformer_IN21K import Block, Mlp


# tuning_config = EasyDict(
#     # AdaptFormer
#     ffn_adapt=True,
#     ffn_option="parallel",
#     ffn_adapter_layernorm_option="none",
#     ffn_adapter_init_option="lora",
#     ffn_adapter_scalar="0.1",
#     ffn_num=64,
#     d_model=768,
# )

def get_block_flops(args):


    from models.vision_transformer_IN21K import Block
    one_block = Block(dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True, tuning_config=args.tuning_config, select=True)
        

    inputs = torch.rand((1, 197, 768))
    num_tokens = 197

    flops_dict = torch.zeros(num_tokens+1)
    one_block.apply(lambda x: setattr(x, 'count_flops', True))
    # block flops
    for t in range(1, num_tokens+1): # 1, 2, 3,.....197
        one_block.apply(lambda x : setattr(x, 'token_select_num', t))
        flops = FlopCountAnalysis(one_block, inputs).total() / (1000**3) 
        # print(t, flops)
        flops_dict[t] = flops
    # flops_dict = dict(meta_info="The number of selected tokens from 1 to 197. GFlops.",
    #                 flops_dict=flops_dict)
    # torch.save(flops_dict, "vit-b16_token-select_flops_dict.pth")
    
    return flops_dict
 
def select_flops(flops_dict,  token_select, block_num, base_flops=0.33):
    t = token_select.shape[1]
    
    if token_select is None :
        token_select = [t] * block_num
    else :
        ada_t = token_select.shape[0]
        token_select = [t] * (block_num - ada_t) + token_select.sum(-1).int().tolist()
        
    token_select = [i+1 for i in token_select] # add cls token
    
    flops = base_flops
    for t in token_select:
        flops += flops_dict[t]
    return flops


def batch_select_flops(bs, flops_dict, token_select, block_num=12, base_flops=0.116):

    token_select = token_select.squeeze(-1) #[N, layer, tokens]

    
    batch_flops = []
    for t in token_select:
        batch_flops.append(select_flops(flops_dict, t, block_num, base_flops))
    
    return torch.tensor(batch_flops)




class VisionTransformerIN21K(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
            tuning_config=None
    ):

        super().__init__()
        self.tuning_config = tuning_config
        
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        x = self.norm(x)
        
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x
    
    

    
def vit_base_patch16_224_in21k(**kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, **kwargs)
    model = VisionTransformerIN21K(**model_kwargs)
    return model


## get the flops of vit without blocks class_num=100
def get_base_flops(args):

    if os.path.basename(args.finetune).startswith('VIT_BASE_IN21K'):
        model = vit_base_patch16_224_in21k(num_classes=args.nb_classes,  drop_path_rate=args.drop_path, tuning_config=args.tuning_config)
        
    inputs = torch.rand((1, 3, 224, 224))

    flops = FlopCountAnalysis(model, inputs).total() / (1000**3)

    return flops

    
if __name__ == '__main__':
    get_block_flops()
    # get_base_flops(num_classes=100) #vit-b/16 class_num=100   Flops=0.116438784 GFlops
    
    # VIT_B_IN21K 0.116438784 GFlops
    # VIT_B_MAE 0.115686144 GFlops
    # VIT_B_CLIP 0.116442624 GFlops