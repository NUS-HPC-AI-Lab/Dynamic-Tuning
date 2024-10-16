import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional
from timm.layers.format import Format, nchw_to
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv, resolve_pretrained_cfg, checkpoint_seq
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_, _assert
from timm.models.layers import to_2tuple
from timm.models.registry import register_model
from torch.jit import Final
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from timm.layers import PatchEmbed, Mlp, DropPath, PatchDropout, \
    trunc_normal_, use_fused_attn
# from .dynamic_adapter import Adapter, TokenSelect
from einops import rearrange

def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    y_soft = logits.sigmoid()

    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).masked_fill(y_soft > threshold, 1.0)


    return y_hard



class TokenSelect(nn.Module):
    def __init__(self, dim_in, num_sub_layer, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.mlp_head = nn.Linear(dim_in, num_sub_layer, bias=bias)

        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        b, l = x.shape[:2]
        logits = self.mlp_head(x[:, 1:, :])
        
        token_select = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        token_select = torch.cat([token_select.new_ones(b, 1, 1), token_select], dim=1)
        
        return token_select, logits
    
    
class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout

                
    def _init_weights(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)


    def forward(self, x, add_residual=True, residual=None):


        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale


        return up
    
    

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            tuning_config=None,
            layer_id=None,
            select=False, 
    ):
        super().__init__()
        self.tuning_config = tuning_config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.adaptmlp = Adapter(self.tuning_config, dropout=0.1, bottleneck=tuning_config.ffn_num,
                        init_option=tuning_config.ffn_adapter_init_option,
                        adapter_scalar=tuning_config.ffn_adapter_scalar,
                        adapter_layernorm_option=tuning_config.ffn_adapter_layernorm_option,
                        )
        if select:
            self.mlp_token_select = TokenSelect(dim, num_sub_layer=1)
        else:
            self.token_select = None
            
        
    def forward(self, x):
        if x.shape[0] == 1:
            out = self.single_forward(x)
        else:
            out = self.batch_forward(x)
            
        return out
    
    def single_forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        
        ################ dyanmic token router ################ 
        policy_token = x
        if self.mlp_token_select is not None :
            sub_token_select, token_logits = self.mlp_token_select(policy_token)
        else:
            sub_token_select, token_logits = None, None
        ################ dyanmic token router ################ 
    


        ################ adapter ################ 
        adapt_x = self.adaptmlp(x, add_residual=False)
        ################ adapter ################ 
        
        
        ################ non-skip token MLP  ################ 
        select_inedx = sub_token_select.nonzero()[:, 1]
        gather_x = x[:, select_inedx, :]
        mlp_x = torch.zeros(x.shape, device=sub_token_select.device, dtype=sub_token_select.dtype)
        mlp_x_noskip = self.drop_path2(self.ls2(self.mlp(self.norm2(gather_x))))
        mlp_x[:, select_inedx, :] = mlp_x_noskip
        ################ non-skip token MLP  ################ 
        
        x = x + adapt_x + mlp_x
        
        return x
    

    def batch_forward(self, x):
        

        
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        

        ################ dyanmic token router ################ 
        policy_token = x
        if self.mlp_token_select is not None :
            sub_token_select, token_logits = self.mlp_token_select(policy_token)
        else:
            sub_token_select, token_logits = None, None
        ################ dyanmic token router ################ 


        # ################ adapter ################ 
        adapt_x = self.adaptmlp(x, add_residual=False)
        # ################ adapter ################ 
        
        
        ################ non-skip token MLP  ################
        residual = x
        sub_token_select = rearrange(sub_token_select, "n token_num c -> (n token_num c)")
        original_shape = x.shape
        x = rearrange(x, "n token_num c -> (n token_num) c")
        select_inedx = sub_token_select.nonzero()[:, 0]
        gather_x = x[select_inedx, :]
        mlp_x = torch.zeros(x.shape, device=sub_token_select.device, dtype=sub_token_select.dtype)
        mlp_x_noskip = self.drop_path2(self.ls2(self.mlp(self.norm2(gather_x))))
        mlp_x[select_inedx, :] = mlp_x_noskip
        mlp_x = residual + rearrange(mlp_x, "(n token_num) c -> n token_num c", n=original_shape[0], token_num=original_shape[1])
        ################ non-skip token MLP  ################ 
        
        x = adapt_x + mlp_x
        
        return x
        




class VisionTransformer(nn.Module):
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
            tuning_config=None,
            select_config=None
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
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
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                tuning_config=tuning_config,
                layer_id=i,
                select=select_config.open and i>= select_config.keep_layers 
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif hasattr(m, '_init_weights'):
            m._init_weights()



        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}



    def forward_features(self, x):
        x = self.patch_embed(x)
        
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.patch_drop(x)
        x = self.norm_pre(x)


        for i, blk in enumerate(self.blocks):
            x = blk(x)

        
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def convert_list_to_tensor(list_convert):
    if len(list_convert):
        result = torch.stack(list_convert, dim=1)
    else :
        result = None
    return result 



def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
        
def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm
    
def vit_base_patch16_224_in21k(**kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, **kwargs)
    model = VisionTransformer(**model_kwargs)
    return model


if __name__ == '__main__':
    model = vit_base_patch16_224_in21k()
    checkpoint_model = torch.load("vit_base_patch16_224_in21k.pth")
    for k in ['head.weight', 'head.bias']:
            del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
