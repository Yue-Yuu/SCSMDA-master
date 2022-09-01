import numpy as np
import scipy.io as sio
import torch
import scipy.sparse as sp
import random
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

"""
triple:(num_triple,2)
label: (num_drug,num_virus)
"""

def get_neg_ent(triple, label, neg_num = 20): # label为triple中源节点的所有目标节点集合
    pos_obj = triple[:, 1]
    mask = np.zeros([triple.shape[0], label.shape[1]], dtype=np.bool)
    mask_train = np.zeros((triple.shape[0], neg_num+1))
    label_neg = np.zeros((triple.shape[0], neg_num+1))

    # 每个三元组有1个正样本，20个负样本
    for i in range(triple.shape[0]):
        num = 0
        label_neg[i][0] = triple[i,1]
        mask_train[i][0] = 1
        mask[i, triple[i, 1]] = 1
        while(num < neg_num):
            b = random.randint(0, 94)
            if label[pos_obj[i], b] != 1 and mask[i, b] != 1:
                mask[i, b] = 1
                label_neg[i, num+1] = b
                num += 1

    return mask, label_neg, mask_train

def masked_accuracy(preds, labels):
    """Accuracy with masking."""
    preds = preds.type(torch.float32)
    labels = labels.type(torch.float32) *1.5
    error = torch.square(preds-labels)  # 负样本的标签值为0，正样本的标签值为1
#     return tf.reduce_sum(error)
    return torch.sqrt(torch.mean(error))


def accuracy(outputs, labels):
    # assert labels.dim() == 1 and outputs.dim() == 1
    outputs = outputs.ge(0.3).type(torch.int32)
    labels = labels.type(torch.int32)
    corrects = (1 - (outputs ^ labels)).type(torch.int32)
    TP = (outputs * labels).sum()
    if labels.size() == 0:
        return np.nan
    return [corrects.sum().item() / (labels.size()[0]*labels.size()[1])  , TP.item() / labels.sum()]


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


def auc(outputs, labels):
    # assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return roc_auc_score(labels, outputs, multi_class='ovo')


def aupr(outputs, labels):
    # assert labels.dim() == 1 and outputs.dim() == 1
    labels = labels.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    return average_precision_score(labels, outputs)


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


def compute_f1(outputs, labels):
    return (precision(outputs, labels) + recall(outputs, labels)) / 2
