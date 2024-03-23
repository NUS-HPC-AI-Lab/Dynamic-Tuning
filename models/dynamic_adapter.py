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

def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret



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
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output