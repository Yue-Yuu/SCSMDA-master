import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Sn_encoder(nn.Module):
    def __init__(self, hidden_dim, nei_num, attn_drop, sc_ngcn):
        super(Sn_encoder, self).__init__()
        self.nei_num = nei_num
        self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(sc_ngcn)])
        self.sc_ngcn = sc_ngcn


    def forward(self, nei_h, nei_index, sc_edge):
        embeds = []

        embeds.append(nei_h)
        for i in range(self.sc_ngcn):
            embeds.append(self.node_level[i](embeds[-1], nei_index[0]))

        return embeds[-1]
