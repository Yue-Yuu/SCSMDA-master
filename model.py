import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Encoder_Module.mp_encoder import Mp_encoder
from Encoder_Module.sc_encoder import Sc_encoder
from Encoder_Module.contrast import Contrast

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=True)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.001)
            # self.bias = self.bias * 1e42
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
        # out = F.dropout(out,0.1)
        return self.act(out)



class Encoder(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P,
                 nei_num, tau, lam, mp_ngcn, sc_ngcn,drug_num, protein_num):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.feats_dim = feats_dim_list
        self.fc_list = nn.Linear(self.feats_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc_list.weight, gain=1.414)
        self.mp_ngcn = mp_ngcn
        self.sc_ngcn = sc_ngcn

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(0.1)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, attn_drop, self.mp_ngcn, drug_num, protein_num)
        self.sc = Sc_encoder(hidden_dim, nei_num, attn_drop, self.sc_ngcn)
        self.contrast_d = Contrast(hidden_dim, tau, lam)
        self.contrast_m = Contrast(hidden_dim, tau, lam)


    def forward(self, feats, pos, mps, nei_index, sc_edge, mp_edge):

        h_all= F.elu(self.feat_drop(self.fc_list(feats)))
        z_mp = self.mp(h_all, mps, mp_edge)
        z_sc = self.sc(h_all, nei_index, sc_edge)
        loss = self.contrast_d(z_mp[:len(pos[0]),:], z_sc[:len(pos[0]),:], pos[0])+ \
               self.contrast_m(z_mp[len(pos[0]):,:], z_sc[len(pos[0]):,:], pos[1])
        return loss

    def get_embeds(self, feats, mps, mp_edge):

        h = F.elu(self.fc_list(feats))
        z_mp = self.mp(h, mps, mp_edge)
        return z_mp

    def get_embeds_sc(self, feats, nei_index, sc_edge):

        h = F.elu(self.fc_list(feats))
        z_sc = self.sc(h, nei_index, sc_edge)
        return z_sc


class NN(nn.Module):
    def __init__(self, ninput, nhidden, noutput, nlayers, dropout=0.5):

        super(NN, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.encode = torch.nn.ModuleList([
            torch.nn.Linear(ninput, noutput) for l in range(self.nlayers)])

    def forward(self, x):
        for l, linear in enumerate(self.encode):
            x = F.relu(linear(x))
        return x

class DTI_Decoder(nn.Module):
    def __init__(self, Protein_num, Drug_num, Nodefeat_size, nhidden, nlayers, dropout=0.3):
        super(DTI_Decoder, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.drug_num = Drug_num
        self.protein_num = Protein_num
        self.decode = torch.nn.ModuleList([
            torch.nn.Linear(Nodefeat_size if l == 0 else Nodefeat_size, Nodefeat_size) for l in
            range(nlayers)])

        self.linear = torch.nn.Linear(Nodefeat_size, 1)
        self.drug_linear = torch.nn.Linear(Nodefeat_size, Nodefeat_size)
        self.proein_linear = torch.nn.Linear(Nodefeat_size, Nodefeat_size)


    def forward(self, nodes_features, drug_index, protein_index):

        protein_features = nodes_features[protein_index]
        drug_features = nodes_features[drug_index]

        protein_features = self.proein_linear(protein_features)
        drug_features = self.drug_linear(drug_features)
        pair_nodes_features = drug_features*protein_features
        for l, dti_nn in enumerate(self.decode):
            pair_nodes_features = F.relu(dti_nn(pair_nodes_features))

        pair_nodes_features = F.dropout(pair_nodes_features, self.dropout)
        output = self.linear(pair_nodes_features)
        return torch.sigmoid(output)

class DTI_Graph(nn.Module):

    def __init__(self,  PNN_hyper, DECO_hyper, Protein_num, Drug_num, dropout, args_pre, feats_dim_list, P, train_type=1):
        super(DTI_Graph, self).__init__()
        self.mp_nn = NN(PNN_hyper[0], PNN_hyper[1], PNN_hyper[2], PNN_hyper[3], dropout)
        self.sc_nn = NN(PNN_hyper[0], PNN_hyper[1], PNN_hyper[2], PNN_hyper[3], dropout)
        self.DTI_Decoder = DTI_Decoder(Protein_num, Drug_num, DECO_hyper[0], DECO_hyper[0], DECO_hyper[2], dropout)
        self.LayerNorm = torch.nn.LayerNorm(DECO_hyper[0])

        self.encoder = Encoder(args_pre.hidden_dim, feats_dim_list, args_pre.feat_drop, args_pre.attn_drop,
                 P, args_pre.nei_num, args_pre.tau, args_pre.lam, args_pre.mp_ngcn,args_pre.sc_ngcn,Drug_num, Protein_num)
        self.encoder.cuda()
        self.drug_num = Drug_num
        self.proein_num = Protein_num





    def forward(self, protein_index, drug_index, feats, pos, mps, nei_index, sc_edge, mp_edge, rate):

        loss = self.encoder(feats, pos, mps, nei_index, sc_edge, mp_edge)

        embs_sc = self.encoder.get_embeds_sc(feats, nei_index, sc_edge)
        embs_sc = self.sc_nn(embs_sc)
        embs_sc = self.LayerNorm(embs_sc)
        Nodes_features = embs_sc


        # Decoder
        output = self.DTI_Decoder(Nodes_features, drug_index, protein_index)
        output = output.view(-1)


        return output, loss, Nodes_features


