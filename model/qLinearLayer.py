import torch
import torch.nn as nn
import sys
sys.path.append('./MixedGemm/build/')
import mixedgemm

import math
import random

def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        if module.enable_quant:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res
    
class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        p8_num, 
        p6_num,
        reorder_index,
        out_reorder_index=None,
    ):
        super().__init__()
      
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
    
        
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
        
        self.p6_num = p6_num
        self.p8_num = p8_num
        self.p4_num = self.in_features - p8_num - p6_num
       
        if self.in_features > 25000:
            self.BN, self.BS, self.BO, self.SFBN, self.SFBS, self.SFBO = mixedgemm.downproj_quantize_w(torch.index_select(originalLayer.weight.data, 1, reorder_index.cuda()), self.p4_num, self.p6_num, self.p8_num)
        else:
            self.BN, self.BS, self.BO, self.SFBN, self.SFBS, self.SFBO = mixedgemm.reorder_quantize_w(originalLayer.weight.data, reorder_index.to(torch.int16).cuda(), self.p4_num, self.p6_num, self.p8_num)
        
        reorder_index.cpu()
        del reorder_index
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        AN, AS, AO, SFAN, SFAS, SFAO, bsz, q_len = x
        y = mixedgemm.matmul(AN, self.BN, AS, self.BS, AO, self.BO, SFAN, self.SFBN, SFAS, self.SFBS, SFAO, self.SFBO)
        if self.bias is not None:
            y = y + self.bias

        y = y.reshape(bsz, q_len, -1)
        return y
    