import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

def accuracy(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    threshold_value = (outputs.max() + outputs.min()) /2.
    outputs = outputs.ge(threshold_value).type(torch.int32)
    labels = labels.type(torch.int32)
    corrects = (1 - (outputs ^ labels)).type(torch.int32)
    if labels.size() == 0:
        return np.nan
    return corrects.sum().item() / labels.size()[0]


def precision(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    return precision_score(labels, outputs)


def recall(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    return recall_score(labels, outputs)


def specificity(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.ge(0.5).type(torch.int32).detach().cpu().numpy()
    return recall_score(labels, outputs, pos_label=0)


def f1(outputs, labels):
    return (precision(outputs, labels) + recall(outputs, labels)) / 2


def mcc(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    outputs = outputs.ge(0.5).type(torch.int32)
    labels = labels.type(torch.int32)
    true_pos = (outputs * labels).sum()
    true_neg = ((1 - outputs) * (1 - labels)).sum()
    false_pos = (outputs * (1 - labels)).sum()
    false_neg = ((1 - outputs) * labels).sum()
    numerator = true_pos * true_neg - false_pos * false_neg
    deno_2 = outputs.sum() * (1 - outputs).sum() * labels.sum() * (1 - labels).sum()
    if deno_2 == 0:
        return np.nan
    return (numerator / (deno_2.type(torch.float32).sqrt())).item()


def auc(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return roc_auc_score(labels, outputs)

def aupr(outputs, labels):
    assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return average_precision_score(labels, outputs)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()