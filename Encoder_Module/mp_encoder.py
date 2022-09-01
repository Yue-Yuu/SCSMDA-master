import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=True)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.001)
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
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        out = F.dropout(out,0.95)
        return self.act(out)


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        # attn_curr = self.attn_drop(self.att)
        attn_curr = self.att
        for embed in embeds:
            sp = self.tanh(F.dropout(self.fc(embed), 0.95)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)

        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp

class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop, mp_ngcn,drug_num, protein_num):
        super(Mp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(5)])
        self.gat_heads = 4
        self.dropout = 0.
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.feat_drop = nn.Dropout(0.2)
        self.mp_ngcn = mp_ngcn

        self.drug_num = drug_num
        self.protein_num = protein_num
        self.att0 = Attention(hidden_dim, attn_drop)
        self.att1 = Attention(hidden_dim, attn_drop)

    def forward(self, h, mps, mp_edge):
        embeds = []
        emb = []
        embeds.append(h)

        embeds.append(self.node_level[0](embeds[0][:self.drug_num,:], mps[0]))
        embeds.append(self.node_level[1](embeds[0][:self.drug_num, :], mps[1]))
        embeds.append(self.node_level[2](embeds[0][self.drug_num:, :], mps[2]))
        embeds.append(self.node_level[3](embeds[0][self.drug_num:, :], mps[3]))
        emb.append(self.att0(embeds[-4:-2]))
        emb.append(self.att1(embeds[-2:]))

        z_mp = torch.cat((emb[-2],emb[-1]))

        return z_mp
