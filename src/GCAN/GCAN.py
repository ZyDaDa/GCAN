import torch
from torch import nn
import numpy as np
from GCAN.layer import EncoderLayer
import math


class GCAN(nn.Module):
    def __init__(self,args, num_item) -> None:
        super().__init__()
        self.dim = args.dim
        self.heads = args.heads
        self.num_item = num_item
        self.dropout = nn.Dropout(args.dropout)
        self.spd_bias = nn.Embedding(100, embedding_dim=1, padding_idx=0)

        # embedding layer 
        self.item_emb = nn.Embedding(self.num_item+1, self.dim, padding_idx=0, max_norm=0.5)
        self.pos_emb = nn.Embedding(args.max_length, self.dim, max_norm=0.5)

        # GA channel 
        self.ga_layer_stack = nn.ModuleList([
                            EncoderLayer(self.dim, 
                            self.heads, 
                            args.dropout,
                            att_distance=args.att_distance) for _ in range(args.layer_num)])

        # CA channel
        self.ca_layer_stack = nn.ModuleList([
                            EncoderLayer(self.dim, 
                            self.heads, 
                            args.dropout,
                            ca_channel=True) for _ in range(1)])
        # fushion layer
        self.merge = nn.Linear(self.dim*2, 1)

        # recommendation layer
        self.w1 = nn.Linear(self.dim,self.dim,bias=False)
        self.w2 = nn.Linear(self.dim,self.dim,bias=False)
        self.w3 = nn.Linear(self.dim,self.dim,bias=True)
        self.w0 = nn.Linear(self.dim,1,bias=False)
        self.w4 = nn.Linear(self.dim*2,self.dim,bias=False)

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.5 / math.sqrt(self.dim)
        nn.init.normal_(self.item_emb.weight, 0, stdv)
        nn.init.normal_(self.pos_emb.weight,0, stdv)
        nn.init.normal_(self.spd_bias.weight,0, stdv)

        nn.init.normal_(self.w1.weight,0,  stdv)
        nn.init.normal_(self.w2.weight,0,  stdv)
        nn.init.normal_(self.w3.weight,0,  stdv)
        nn.init.normal_(self.w0.weight,0,  stdv)
        nn.init.normal_(self.w4.weight,0,  stdv)
        nn.init.normal_(self.merge.weight,0,  stdv)

        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)
            self.spd_bias.weight[0].fill_(0)
            self.w3.bias.data.fill_(0)
            self.merge.bias.data.fill_(0)

    def forward(self, seq, spd, cs, mask):
        # embedding layer
        enc_emb = self.item_emb(seq) + self.pos_emb.weight
        ga_out = self.dropout(enc_emb)
        ga_out[mask] = 0.

        # GA channel
        attn_bias = self.spd_bias(spd).squeeze() 
        for enc_layer in self.ga_layer_stack:
            ga_out, _ = enc_layer(ga_out, attn_bias, mask,spd)
            ga_out[mask] = 0.

        # CA channel
        ca_out = self.dropout(enc_emb)
        for enc_layer in self.ca_layer_stack:
            ca_out, _ = enc_layer(ca_out, cs, mask)
            ca_out[mask] = 0.
        # fushion layer
        score = torch.sigmoid(self.merge(torch.concat([ga_out,ca_out],dim=-1)))
        output = ga_out*score + ca_out*(1-score)

        return output

    def comp_scores(self, embedding, mask):

        # recommendation layer 
        last = embedding[:, -1]  
        
        embedding[mask] = 0.0
        avg = embedding.sum(1) / (~mask).sum(-1).view((-1,1))

        # Eq. 13
        gama = self.w0(torch.sigmoid(self.w1(last).unsqueeze(1) + self.w2(avg).unsqueeze(1) + self.w3(embedding))) 
        gama=gama.masked_fill(mask.unsqueeze(-1)==1, 0)

        # Eq. 14
        final_emb = (gama*embedding).sum(1)
        scores = torch.matmul(final_emb, self.item_emb.weight[1:].T) 

        return scores
