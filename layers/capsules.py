import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg, cfg_from_file
import lib.utils as utils

def squash(inputs, axis=-1):
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs

class Capsule(nn.Module):
    def __init__(self, routings=3):
        super(Capsule, self).__init__()
        self.in_dim_caps = cfg.MODEL.BILINEAR.ATT_DIM
        self.out_dim_caps = cfg.MODEL.BILINEAR.ATT_DIM
        self.routings = routings
    
    def forward(self, x, isLastLayer):
        self.in_num_caps = x.size()[1]
        if isLastLayer:
            self.out_num_caps = 1
        else:
            self.out_num_caps = x.size()[1]
        self.weight = 0.01 * torch.randn(self.out_num_caps, self.in_num_caps, self.out_dim_caps, self.in_dim_caps).cuda()
        
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
        x_hat_detached = x_hat.detach()
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()
        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            c = F.softmax(b, dim=1)
            if i == self.routings - 1:
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else: 
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)
        attn = torch.squeeze(outputs, dim=-2)  
        torch.cuda.empty_cache()    
        return attn
