import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
import layers

class CapsuleLowRank(nn.Module):
    def __init__(self, embed_dim, att_type, att_heads, att_mid_dim, att_mid_drop):
        super(CapsuleLowRank, self).__init__()
        mid_dims = att_mid_dim
        mid_dropout = att_mid_drop
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        output_dim = 2 * embed_dim if cfg.MODEL.BILINEAR.ACT == 'GLU' else embed_dim  
        self.head_dim = output_dim // self.num_heads    
        self.scaling = self.head_dim ** -0.5

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))     
        act = utils.activation(cfg.MODEL.BILINEAR.ACT)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, output_dim))
        self.in_proj_q = nn.Sequential(*sequential)     

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = utils.activation(cfg.MODEL.BILINEAR.ACT)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, output_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = utils.activation(cfg.MODEL.BILINEAR.ACT)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, output_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = utils.activation(cfg.MODEL.BILINEAR.ACT)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, output_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)                     

        self.attn_net = layers.create(att_type) 
        self.clear_buffer() 


        sequential = []
        for i in range(1, len(mid_dims) - 1):  
            sequential.append(nn.Linear(mid_dims[i - 1], mid_dims[i])) 
            sequential.append(nn.ReLU())  
            if mid_dropout > 0:
                sequential.append(nn.Dropout(mid_dropout))
        self.attention_basic = nn.Sequential(*sequential) if len(sequential) > 0 else None   
        self.attention_last = nn.Linear(mid_dims[-2], mid_dims[-1])  
        # copy from sc_att
        self.attention_last1 = nn.Linear(mid_dims[-2], 1)   
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])  
        sequential = []                  
        for i in range(1, len(mid_dims) - 1):                     
            sequential.append(nn.Linear(mid_dims[i - 1], 256))                          
            sequential.append(nn.ReLU())                          
            if mid_dropout > 0:                                  
                sequential.append(nn.Dropout(mid_dropout))                  
        self.attention_basic1 = nn.Sequential(*sequential) if len(sequential) > 0 else None

    def apply_to_states(self, fn):           
        self.buffer_keys = fn(self.buffer_keys)
        self.buffer_value2 = fn(self.buffer_value2)

    def init_buffer(self, batch_size):
        self.buffer_keys = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()    
        self.buffer_value2 = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()  
    def clear_buffer(self):
        self.buffer_keys = None
        self.buffer_value2 = None

    
    def forward(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)            


        q = q.view(batch_size, self.num_heads, self.head_dim)   
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)

        if precompute == False:
            key = key.view(-1, key.size()[-1])   
            value2 = value2.view(-1, value2.size()[-1])   
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            
            k = k.view(batch_size, -1, self.num_heads, self.head_dim)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim)
        else:
            k = key.transpose(1, 2) 
            v2 = value2.transpose(1, 2) 


        attn_map = q.unsqueeze(1) * k 

        if self.attention_basic is not None:   
            att_map1 = self.attention_basic1(attn_map) 
        att_mask = mask
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)   
            att_mask_ext = att_mask.unsqueeze(-1).transpose(1, 2) 
            
            att_map_pool = torch.sum(att_map1 * att_mask_ext, -1) / torch.sum(att_mask_ext, -1)       
        else:
            att_map_pool = att_map1.mean(-2)    
        alpha_channel = torch.sigmoid(att_map_pool)  
        alpha_channel = alpha_channel.unsqueeze(-1)
        attn_map = attn_map * alpha_channel

        attn_map = attn_map.reshape(batch_size, -1, self.embed_dim)
        
        attn = self.attn_net(attn_map)
        attn = attn.squeeze(1)
        return attn

    def forward2(self, query, key, mask, value1, value2, precompute=False): 
        batch_size = query.size()[0]
        query = query.view(-1, query.size()[-1])
        value1 = value1.view(-1, value1.size()[-1])
        
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            if self.buffer_keys is not None and self.buffer_value2 is not None:  
                self.buffer_keys = torch.cat([self.buffer_keys, k], dim=2)  
                self.buffer_value2 = torch.cat([self.buffer_value2, v2], dim=2)
                k = self.buffer_keys
                v2 = self.buffer_value2
        else:
            k = key
            v2 = value2
        
        attn_map = q.unsqueeze(-2) * k.unsqueeze(-3)     
        attn = self.attn_net.forward(attn_map, mask, v1, v2).transpose(1, 2).contiguous()
        attn = attn.view(batch_size, -1, self.num_heads * self.head_dim)
        return attn

    def precompute(self, key, value2):   
        batch_size = value2.size()[0]
        key = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2
