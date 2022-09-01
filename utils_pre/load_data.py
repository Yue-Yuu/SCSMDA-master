import os.path
import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+sp.eye(adj.shape[0])
    return adj_normalized.tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_data(dataset, train_times):

    path = os.path.join("./data", dataset)
    path = os.path.join(path,'encoder_'+ str(train_times) + '/')
    # path = "./data/DrugVirus/"
    adj_similiarty = np.load(path + "adj_similiarty.npy", allow_pickle=True)
    adj_dm = np.load(path + "adj_dm.npy", allow_pickle=True)
    dmd = sp.load_npz(path + "dmd.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    dmdmd = sp.load_npz(path + 'dmdmd.npz')
    mdmdm = sp.load_npz(path + 'mdmdm.npz')
    pos_d = sp.load_npz(path + "pos_d.npz")
    pos_m = sp.load_npz(path + "pos_m.npz")
    # val = np.load(path + "split_val.npy")
    node_feature = np.load(path + 'nodefeatures.npy')
    node_feature = torch.FloatTensor(node_feature)

    # dmd.data = np.ones_like(dmd.data)
    # mdm.data = np.ones_like(mdm.data)
    # dmdmd.data = np.ones_like(dmdmd.data)
    # mdmdm.data = np.ones_like(mdmdm.data)


    adj_similiarty = torch.FloatTensor(adj_similiarty)
    adj_dm = torch.FloatTensor(adj_dm)
    dmd = sparse_mx_to_torch_sparse_tensor(normalize_adj(dmd))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    dmdmd = sparse_mx_to_torch_sparse_tensor(normalize_adj(dmdmd))
    mdmdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdmdm))

    pos_d = sparse_mx_to_torch_sparse_tensor(normalize_adj(pos_d))
    pos_m = sparse_mx_to_torch_sparse_tensor(normalize_adj(pos_m))
    # pos_d = sparse_mx_to_torch_sparse_tensor(pos_d)
    # pos_m = sparse_mx_to_torch_sparse_tensor(pos_m)
    adj_similiarty_list = torch.where(adj_similiarty > 0.5)
    adj_similiarty_list = np.array(adj_similiarty_list)
    sc_edge_index = []
    for i in range(len(adj_similiarty_list)):
        sc_edge_index.append(np.array(adj_similiarty_list[i]))
    sc_edge_index = np.array(sc_edge_index)
    sc_edge_weight = adj_similiarty[sc_edge_index]
    sc_edge_index = torch.LongTensor(sc_edge_index)

    dmd_edge_index = dmd.coalesce().indices()
    dmd_edge_weight = dmd.coalesce().values()
    mdm_edge_index = mdm.coalesce().indices()
    mdm_edge_weight = mdm.coalesce().values()



    adj_similiarty = sparse_mx_to_torch_sparse_tensor(preprocess_adj(adj_similiarty))
    adj_dm = sparse_mx_to_torch_sparse_tensor(preprocess_adj(adj_dm))
    adj_similiarty = adj_similiarty.to_dense()
    adj_dm = adj_dm.to_dense()

    return [adj_similiarty, adj_dm], node_feature, [dmd,dmdmd, mdm,mdmdm], [pos_d, pos_m],\
           [sc_edge_index, sc_edge_weight], [dmd_edge_index, dmd_edge_weight, mdm_edge_index, mdm_edge_weight]
