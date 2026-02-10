import torch
import torch.nn as nn

import sys
sys.path.append('./mgemm/build/')
import mixedgemm

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
        
        self.p6_num = p6_num   #p4_num, p6_num, p8_num需要整除128
        self.p8_num = p8_num
        self.p4_num = self.in_features - p8_num - p6_num
       
       
        out_features, in_features = originalLayer.weight.data.shape
        
        self.reorder_index = reorder_index.to(torch.int16).cuda()

        
        self.BN, self.BS, self.BO, self.SFBN, self.SFBS, self.SFBO = mixedgemm.reorder_quantize_w4(originalLayer.weight.data, self.reorder_index, self.p4_num, self.p6_num, self.p8_num)

        # self.BN_d, self.BS_d, self.BO_d, self.SFBN_d, self.SFBS_d, self.SFBO_d = mixedgemm.reorder_quantize_w4(originalLayer.weight.data, self.reorder_index, 0, 0, self.in_features)
        
        reorder_index.cpu()
        del reorder_index
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        bsz, q_len, _ = x.shape
        x = x.reshape(bsz*q_len, -1).contiguous() 
        # x = hadamard_transform(x).contiguous()
        # if q_len == 1:
        #     AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(x, self.reorder_index, 0, 0, self.in_features)
        #     # y = mixedgemm.matmul(AN, self.BN_d, AS, self.BS_d, AO, self.BO_d, SFAN, self.SFBN_d, SFAS, self.SFBS_d, SFAO, self.SFBO_d)
        # else:
        AN, AS, AO, SFAN, SFAS, SFAO = mixedgemm.reorder_quantize_x(x, self.reorder_index, self.p4_num, self.p6_num, self.p8_num)
        y = mixedgemm.matmul(AN, self.BN, AS, self.BS, AO, self.BO, SFAN, self.SFBN, SFAS, self.SFBS, SFAO, self.SFBO)
            
        if self.bias is not None:
            y = y + self.bias

        y = y.reshape(bsz, q_len, -1)
        return y

