import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter 
import math
class EncoderLayer(nn.Module):
    def __init__(self, 
                dim, 
                heads, 
                dropout,
                ca_channel=False,
                att_distance=1) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.MHA = MultiHeadAttention(dim, heads, dropout,ca_channel,att_distance)
        self.norm1 = nn.LayerNorm(self.dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(self.dim, elementwise_affine=True)
        self.FFN = PositionwiseFeedForward(dim,dropout, 4*dim)

    def forward(self, enc_input, attn_bias=None, attn_mask=None,spd=None):
        enc_output, enc_attn = self.MHA(enc_input, enc_input, enc_input, attn_bias, attn_mask,spd)
        enc_output = self.norm1(enc_input + enc_output) 
        enc_output = self.norm2(enc_output + self.FFN(enc_output)) 
        return enc_output, enc_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout,ca_channel=False,att_distance=1) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = int(self.dim / self.heads)
        assert self.head_dim*heads == self.dim

        self.temperature = self.head_dim **0.5
        self.is_ca = ca_channel
        self.att_distance=att_distance

        if self.is_ca:
            self.w_cf = nn.Linear(self.dim, self.dim, bias=True)
            self.weight = Parameter(torch.ones(size=(1,),dtype=torch.float))

        else:
            self.w_qs = nn.Linear(self.dim, self.dim, bias=True)
            self.w_ks = nn.Linear(self.dim, self.dim, bias=True)
            self.w_vs = nn.Linear(self.dim, self.dim, bias=True)

        self.fc = nn.Linear(self.dim, self.dim, bias=True) 
        self.res_dropout = nn.Dropout(dropout)
        self._init_weight_()
    def _init_weight_(self):
        stdv =  0.5 / math.sqrt(self.dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data,0,stdv)
                if  m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, q, k, v, attn_bias=None,mask=None, spd=None):
        head_dim, n_head = self.head_dim, self.heads
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        if self.is_ca:
            ca_out = self.w_cf(v)
            ca_out = ca_out.view(sz_b, len_v, n_head, head_dim)
            ca_out = ca_out.transpose(1, 2)
            attn = attn_bias.unsqueeze(1) * self.weight
            ca_attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(1)==1, -1e18)
            ca_attn = F.softmax(ca_attn, dim=-1) 
            ca_output = torch.matmul(ca_attn, ca_out)
            output = ca_output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        else:
            q = self.w_qs(q)
            k = self.w_ks(k)
            v = self.w_vs(v)
            q = q.view(sz_b, len_q, n_head, head_dim)
            k = k.view(sz_b, len_k, n_head, head_dim)
            v = v.view(sz_b, len_v, n_head, head_dim)

            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            attn = torch.matmul(q/self.temperature, k.transpose(2,3))
            ga_attn = attn + attn_bias.unsqueeze(1)
            ga_attn = ga_attn.masked_fill(mask.unsqueeze(1).unsqueeze(1)==1, -1e18)
            ga_attn = ga_attn.masked_fill((spd>(self.att_distance+1)).unsqueeze(1), -1e18) # G_MASK

            ga_attn = F.softmax(ga_attn, dim=-1) 
            local_output = torch.matmul(ga_attn, v)
            output = local_output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        output = self.fc(output)
        output = self.res_dropout(output)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, dropout, hidden) -> None:
        super().__init__()
        self.dim = dim
        self.w_1 = nn.Linear(dim, hidden,bias=True) 
        self.w_2 = nn.Linear(hidden, dim,bias=True) 
        self.dropout = nn.Dropout(dropout)
        self._init_weight_()
    def _init_weight_(self):
        stdv =  0.5 / math.sqrt(self.dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data,0,stdv)
                if  m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x):
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        return x