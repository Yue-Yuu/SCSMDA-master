import json
import torch
import numpy as np
import random
import torch.nn.functional as F
import functools



def gen_info_data(path):
    ori_data = np.load("./data/pretrain_embeddings.npz")
    protein_tensor = torch.tensor(ori_data['microbefeat'], dtype=torch.float)  #
    drug_tensor = torch.tensor(ori_data['drugfeat'], dtype=torch.float)  #
    protein_num = protein_tensor.shape[0]
    drug_num = drug_tensor.shape[0]
    node_num = protein_num + drug_num
    # protein_tensor = np.random.uniform(0.0, 1.0,(protein_tensor.shape[0], protein_tensor.shape[1]))
    # protein_tensor = torch.tensor(protein_tensor, dtype=torch.float32)
    # drug_tensor = np.random.uniform(0.0, 1.0,(drug_tensor.shape[0], drug_tensor.shape[1]))
    # drug_tensor = torch.tensor(drug_tensor, dtype=torch.float32)
    return protein_tensor, drug_tensor, node_num, protein_num

def load_pre_process(preprocess_path, type):

    # type==0为验证集样本
    a = np.load(preprocess_path, allow_pickle=True)
    adj = torch.FloatTensor(a['adj'])
    dti_inter_mat = torch.FloatTensor(a['dti_inter_mat'])
    train_interact_pos = torch.LongTensor(a['train_interact_pos'])
    val_interact_pos = torch.LongTensor(a['val_interact_pos'])
    return adj, dti_inter_mat, train_interact_pos, val_interact_pos







