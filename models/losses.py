from numpy.lib.arraysetops import isin
from timm import loss
from timm.data.transforms_factory import transforms_imagenet_train
import torch
from torch.functional import Tensor
import torch.nn as nn

def binaray_entropy(prob, eps=1e-7):
    neg_entro = prob * prob.clamp(min=eps).log() + (1-prob) * (1-prob).clamp(min=eps).log()
    return - neg_entro




class AdaLoss(nn.Module):
    def __init__(self, base_criterion, 
                 
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
        self.base_criterion = base_criterion
        
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




    def forward(self, outputs, y):
        '''
        head_select: (b, num_layers, num_head)
        '''

        x, token_select, _ = outputs["prediction"], outputs["token_select"], outputs["token_logits"]

        base_loss = self.base_criterion(x, y)
        # layer_loss = self._get_layer_loss(x, layer_select, layer_logits)
        token_loss = self._get_token_loss(x, token_select)
        
        loss = base_loss +  self.token_loss_ratio * token_loss

        return loss, dict(base_loss=base_loss, token_loss=self.token_loss_ratio * token_loss)
    
    def _get_token_loss(self, x, token_select):
        """
        token_select : tensor (b, num_layer, l)

        """
        if token_select is not None :
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
        else :
            token_loss = x.new_zeros(1).mean()

        return token_loss

    
    def _get_layer_loss(self, x, layer_select, logits_set):
        if layer_select is not None :
            layer_mean = layer_select.mean()
            layer_flops_loss = (layer_mean - self.layer_target_ratio).abs().mean()

            if self.layer_diverse_ratio > 0 :
                layer_mean = layer_select.mean((0,-1))
                layer_diverse_loss = (layer_mean - self.layer_target_ratio).abs().mean()
            else :
                layer_diverse_loss = 0

            if self.layer_entropy_weight > 0 :
                layer_select_logits = logits_set['layer_select_logits']
                layer_entropy = binaray_entropy(layer_select_logits.sigmoid()).mean()
            else :
                layer_entropy = 0

            if self.layer_minimal_weight > 0 :
                layer_mean = layer_select.mean(0) #(num_layers, 2)
                layer_minimal_loss = (self.layer_minimal - layer_mean).clamp(min=0.).sum()
            else :
                layer_minimal_loss = 0

            layer_loss = layer_flops_loss + self.layer_diverse_ratio * layer_diverse_loss - self.layer_entropy_weight * layer_entropy \
                            + self.layer_minimal_weight * layer_minimal_loss
        else :
            layer_loss = x.new_zeros(1).mean()

        return layer_loss


