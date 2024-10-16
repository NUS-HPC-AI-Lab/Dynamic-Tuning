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
from models.dynamic_adapter import Adapter, TokenSelect
import numpy as np
from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

class AdaLoss(nn.Module):
    def __init__(self, 
                 
                 layer_target_ratio=0.5, 
                 layer_loss_ratio=2., 
                 layer_diverse_ratio=0.1, 
                 layer_entropy_weight=0.1, 
                 layer_minimal_weight=0., 
                 layer_minimal=0.,
                 
                token_target_ratio=0.5, 
                token_loss_ratio=2., 
                token_minimal=0.1, 
                token_minimal_weight=1.
                 ):
        super().__init__()

        
        # self.layer_target_ratio = layer_target_ratio
        # self.layer_loss_ratio = layer_loss_ratio
        # self.layer_diverse_ratio = layer_diverse_ratio
        # self.layer_entropy_weight = layer_entropy_weight
        # self.layer_minimal_weight = layer_minimal_weight
        # self.layer_minimal = layer_minimal
        
        self.token_target_ratio = token_target_ratio
        self.token_loss_ratio = token_loss_ratio
        self.token_minimal = token_minimal
        self.token_minimal_weight = token_minimal_weight


    def forward(self, outputs):

        token_select = outputs["token_select"]

        token_loss = self._get_token_loss(token_select)
        
        loss = self.token_loss_ratio * token_loss

        return loss
    
    def _get_token_loss(self, token_select):


        token_mean = token_select.mean()
        # token_flops_loss = (token_mean - self.token_target_ratio).abs().mean()
        # token_flops_loss = (token_mean - self.token_target_ratio).clamp(min=0.).mean()
        token_flops_loss = ((token_mean - self.token_target_ratio)**2).mean()

        if self.token_minimal_weight > 0 :
            token_mean = token_select.mean(-1)
            token_minimal_loss = (self.token_minimal - token_mean).clamp(min=0.).sum()
        else :
            token_minimal_loss = 0

        token_loss = token_flops_loss + self.token_minimal_weight * token_minimal_loss


        return token_loss




def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):

    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)


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
            window_size=None,
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
        
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
            
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
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
            window_size=None,
            tuning_config=None,
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
            window_size=window_size
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
            self.mlp_token_select = None
        

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))


        policy_token = x
        if self.mlp_token_select is not None :
            sub_token_select, token_logits = self.mlp_token_select(policy_token)
        else:
            sub_token_select, token_logits = None, None
            
            
        adapt_x = self.adaptmlp(x, add_residual=False)
        residual = x
        mlp_x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        if sub_token_select is not None:
            mlp_x = sub_token_select * mlp_x
        x = residual + mlp_x + adapt_x
        
        
        return x, dict(sub_token_select=sub_token_select, token_logits=token_logits)

    
        
        



@BACKBONES.register_module()
class VisionTransformer21K(nn.Module):
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
            select_config=None,
            out_indices=[3, 5, 7, 11],
            use_rel_pos_bias=False
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
        self.select_config = select_config
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
                window_size=self.patch_embed.grid_size if use_rel_pos_bias else None,
                tuning_config=tuning_config,
                # layer_id=i,
                select=select_config.open and i>= select_config.keep_layers 
            )
            
        for i in range(depth)])
        
        
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            # nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            
            
        self.out_indices = out_indices
            
        # self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        # self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        # self.head_drop = nn.Dropout(drop_rate)
        # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)
        
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self.init_weights)
        self.token_loss = AdaLoss(
                        layer_target_ratio=select_config.layer_target_ratio,  # 0.5
                        layer_loss_ratio=select_config.layer_loss_ratio, # 2.0
                        layer_diverse_ratio=select_config.layer_diverse_ratio, # 0.0
                        layer_entropy_weight=select_config.layer_entropy_weight, # 0.0
                        layer_minimal_weight=select_config.layer_minimal_weight, # 0.0
                        layer_minimal=select_config.layer_minimal,
                        
                        
                        token_target_ratio=select_config.token_target_ratio,
                        token_loss_ratio=select_config.token_ratio,
                        token_minimal=select_config.token_minimal,
                        token_minimal_weight=select_config.token_minimal_weight
                        ) # 0.0
        

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif hasattr(m, '_init_weights'):
            m._init_weights()

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        # if (not self.with_cls_token
        #         and ckpt_pos_embed_shape[1] == self.pos_embed.shape[1] + 1):
        #     # Remove cls token from state dict if it's not used.
        #     state_dict[name] = state_dict[name][:, 1:]
        #     ckpt_pos_embed_shape = state_dict[name].shape

        if self.pos_embed.shape != ckpt_pos_embed_shape:
            # from mmengine.logging import MMLogger
            # logger = MMLogger.get_current_instance()
            # logger.info(
            #     f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
            #     f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - 1)))
            pos_embed_shape = to_2tuple(int(np.sqrt(self.pos_embed.shape[1] - 1)))
            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                num_extra_tokens=1)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}



    def forward_features(self, x):
        B, C, H, W = x.shape
        Hp, Wp = int(H / self.patch_embed.patch_size[0]), int(W / self.patch_embed.patch_size[1])
        x = self.patch_embed(x)
        
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        features = []
        token_select_list = []
        token_logits_list = []
        for i, blk in enumerate(self.blocks):
            x, token_select = blk(x)
            
            if (token_select["sub_token_select"] is not None) and (token_select["token_logits"] is not None):
                token_select_list.append(token_select["sub_token_select"])
                token_logits_list.append(token_select["token_logits"])
                
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features.append(xp.contiguous())
                
        token_select = convert_list_to_tensor(token_select_list)[:, :, 1:, :]
        token_logits = convert_list_to_tensor(token_logits_list)
        
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        loss = self.token_loss(dict(token_select=token_select))
        return tuple(features), dict(token_select=token_select, token_logits=token_logits, loss=loss)

    # def forward_head(self, x, pre_logits: bool = False):
    #     if self.global_pool:
    #         x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    #     x = self.fc_norm(x)
    #     x = self.head_drop(x)
    #     return x if pre_logits else self.head(x)

    def forward(self, x):
        x, token_select = self.forward_features(x)
        # x = self.forward_head(x)
        return x, token_select


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
