import torch
import os
import argparse
import random
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils_pre.load_data import preprocess_adj

from data.data_reformat import *

def cross_5_folds(dataset, seed):
    parent_dir = os.path.join("./data", dataset)
    path_drugfeat = os.path.join(parent_dir, "drugfeatures.txt")
    path_microbefeat = os.path.join(parent_dir, "microbefeatures.txt")
    drugfeat = np.loadtxt(path_drugfeat)
    microbefeat = np.loadtxt(path_microbefeat)
    path_adj = os.path.join(parent_dir, "adj.txt")
    adj = np.loadtxt(path_adj)
    for i in adj:
        i[0] -= 1
        i[1] -= 1
    adj = adj.astype("int")
    D = len(drugfeat)
    M = len(microbefeat)
    dm = np.array(adj)
    dm = sp.csr_matrix((dm[:, 2], (dm[:, 0], dm[:, 1])), shape=(D, M)).toarray()
    folds = np.ones(adj.shape[0])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    kf = kf.split(folds)
    a = []

    for i, data in enumerate(kf):
        a.append(data[1])

    for i in range(5):
        for j in range(len(a[i])):
            folds[a[i][j]] = i

    adj = adj.astype(np.int32).tolist()
    num = 0
    drug_id_list = list(range(len(drugfeat)))
    microbe_id_list = list(range(len(microbefeat)))
    pos_num = len(adj)

    """
    get neg sample for final test
    """
    temp_adj = []
    folds1 = []
    while num < pos_num:
        drug_id = random.choice(drug_id_list)
        microbe_id = random.choice(microbe_id_list)
        neg_pos1 = [drug_id, microbe_id, 1]
        neg_pos = [drug_id, microbe_id, 0]
        if (neg_pos1 in adj) or (neg_pos in temp_adj):
            continue
        temp_adj.append(adj[num])
        folds1.append(folds[num % pos_num])
        temp_adj.append(neg_pos)
        folds1.append(folds[num % pos_num])
        num += 1

    folds1 = np.array(folds1)
    temp_adj = np.array(temp_adj)
    nodefeatures = np.vstack((np.hstack((drugfeat, dm)), np.hstack((dm.transpose(), microbefeat))))
    path_data = os.path.join(parent_dir,"data_"+dataset+".npz")
    np.savez(path_data, drugfeat=drugfeat, microbefeat=microbefeat, adj=temp_adj, folds=folds1, nodefeat=nodefeatures)
    return drugfeat, microbefeat, temp_adj, folds1, nodefeatures

def add_dti_info(protein_num, drug_num, ori_dti_inter_mat, positive_sample_num, train_positive_inter_pos, val_positive_inter_pos, refer_val_interact_pos,
                 pred, i_iter, args, rate):


    pred = pred.detach().cpu()
    np.random.seed(args.seed)
    n_epochs = args.epochs
    train_interact_pos = []
    train_label = []
    val_interact_pos = []
    val_label = []

    negative_interact_pos = []
    temp_refer_val_interact_pos = np.array(refer_val_interact_pos[0])
    for i in range(1, 5):
        temp_refer_val_interact_pos = np.concatenate((temp_refer_val_interact_pos, refer_val_interact_pos[i]), axis=0)
    temp_refer_val_interact_pos = temp_refer_val_interact_pos.tolist()


    '''
    based predicted results to generate negative samples
    '''
    rate_n = rate
    alpha = np.tan(np.pi * 0.5 * (i_iter / (max(n_epochs - 1, 1))))

    indexes = np.arange(drug_num * protein_num)

    # Get the desired & actual number of negtive samples
    n_target = positive_sample_num * rate_n
    n_target = int(n_target)
    ori_dti_inter_mat = np.array(ori_dti_inter_mat.cpu().view(-1))
    index_mask = ori_dti_inter_mat==-1
    n_samples = np.count_nonzero(index_mask)

    # Compute the hardness array
    hardness = np.abs(pred[index_mask])

    # index_n: absolute indexes of class C samples
    index_n = indexes[index_mask]
    k_bins = args.k_bins

    if hardness.max() == hardness.min():
        # perform random under-sampling
        index_chosen = np.random.choice(index_n, size=n_target, replace=False)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            # compute population & hardness contribution of each bin
            populations, edges = np.histogram(hardness, bins=k_bins)
            contributions = np.zeros(k_bins)
            index_bins = []
            for i_bin in range(k_bins):
                index_bin = ((hardness >= edges[i_bin]) & (hardness < edges[i_bin + 1]))
                if i_bin == (k_bins - 1):
                    index_bin = index_bin | (hardness == edges[i_bin + 1])
                index_bins.append(index_bin)
                if populations[i_bin] > 0:
                    contributions[i_bin] = hardness[index_bin].mean()


            # compute the expected number of samples to be sampled from each bin
            bin_weights = 1. / (contributions + alpha)

            bin_weights[np.isnan(bin_weights) | np.isinf(bin_weights)] = 0
            # bin_weights = bin_weights * populations
            n_target_samples_bins = n_target * bin_weights / bin_weights.sum()
            # check whether exists empty bins
            n_invalid_samples = sum(n_target_samples_bins[populations == 0])
            if n_invalid_samples > 0:
                n_valid_samples = n_target - n_invalid_samples
                n_target_samples_bins *= n_target / n_valid_samples
                n_target_samples_bins[populations == 0] = 0
            n_target_samples_bins = n_target_samples_bins.astype(int) + 1

            # perform soft (weighted) self-paced under-sampling
            soft_spu_bin_weights = n_target_samples_bins / populations
            soft_spu_bin_weights[~np.isfinite(soft_spu_bin_weights)] = 0
        soft_spu_sample_proba = np.zeros_like(hardness)
        for i_bin in range(k_bins):
            soft_spu_sample_proba[index_bins[i_bin]] = soft_spu_bin_weights[i_bin]

        soft_spu_sample_proba = soft_spu_sample_proba.astype(np.float64)
        soft_spu_sample_proba /= soft_spu_sample_proba.sum()
        # sample with respect to the sampling probabilities
        index_chosen = np.random.choice(index_n, size=n_target, replace=False, p=soft_spu_sample_proba.reshape(-1), )


    drug_id = index_chosen // protein_num
    microbe_id = index_chosen % protein_num

    negative_interact_pos = np.zeros([n_target,2])
    negative_interact_pos[:,0] = drug_id
    negative_interact_pos[:,1] = microbe_id
    negative_interact_pos = negative_interact_pos.astype(np.int32)

    train_negative_inter_pos = []
    val_negative_inter_pos = []

    train_positive_inter_pos = np.array(train_positive_inter_pos)
    val_positive_inter_pos = np.array(val_positive_inter_pos)

    neg_num = len(train_positive_inter_pos) * rate_n
    neg_num = int(neg_num)

    train_negative_inter_pos.append(negative_interact_pos[:neg_num])
    val_negative_inter_pos.append(negative_interact_pos[neg_num:])

    # merge
    for i in range(1):
        train_interact_pos.append(np.concatenate((train_positive_inter_pos, train_negative_inter_pos[i]), axis=0))
        train_positive_label = np.ones(len(train_positive_inter_pos))
        train_negative_label = np.zeros(int(neg_num * rate_n))
        train_label.append(np.concatenate((train_positive_label, train_negative_label), axis=0))  # 1为正样本，0为生成的负样本

        val_interact_pos.append(np.concatenate((val_positive_inter_pos, val_negative_inter_pos[i]), axis=0))
        val_positive_label = np.ones(len(val_positive_inter_pos))
        val_negative_label = np.zeros(int(neg_num * rate_n))
        val_label.append(np.concatenate((val_positive_label, val_negative_label), axis=0))

        """
            shuffle the data
        """
        shuffle_train = np.arange(0, len(train_interact_pos[i]))
        random.shuffle(shuffle_train)
        train_interact_pos[i] = train_interact_pos[i][shuffle_train]
        train_label[i] = train_label[i][shuffle_train]

        shuffle_test = np.arange(0, len(val_interact_pos[i]))
        random.shuffle(shuffle_test)
        val_interact_pos[i] = val_interact_pos[i][shuffle_test]
        val_label[i] = val_label[i][shuffle_test]


    # construct dti
    dti_list = []
    for i in range(1):
        dti_inter_mat = -np.ones((drug_num, protein_num))  # [protein_num, drug_num]
        for j, inter in enumerate(train_interact_pos[i]):
            protein_id = inter[1]
            drug_id = inter[0]
            label = train_label[i][j]
            dti_inter_mat[drug_id][protein_id] = label
        for j, inter in enumerate(val_interact_pos[i]):
            protein_id = inter[1]
            drug_id = inter[0]
            label = val_label[i][j]
            dti_inter_mat[drug_id][protein_id] = label
        # dti_inter_mat = dti_inter_mat.tolist()  # dti_inter_mat包含train和test的正(1)负(-1)样本。
        dti_list.append(dti_inter_mat)

    return dti_list[0], train_interact_pos[0], val_interact_pos[0]


def first_spilt_label(inter, groups, seed, dataset,pos_num):

    np.random.seed(seed)

    inter_folds = [[],[],[],[],[]]
    label_folds = [[],[],[],[],[]]
    pos_inter_folds = [[],[],[],[],[]]
    pos_label_folds = [[],[],[],[],[]]
    neg_inter_folds = [[],[],[],[],[]]
    neg_label_folds = [[],[],[],[],[]]
    for i, inter_k in enumerate(inter):
        inter_type = inter_k[-1]  # 1 positive, 0 negative
        protein_node_id = inter_k[1]
        drug_node_id = inter_k[0]
        fold_id = int(groups[i])
        inter_folds[fold_id].append([drug_node_id, protein_node_id])
        label_folds[fold_id].append(inter_type)
        if inter_type == 1:
            #  positive sample
            pos_inter_folds[fold_id].append([drug_node_id, protein_node_id])
            pos_label_folds[fold_id].append(inter_type)
        elif inter_type == 0:
            # negative sample
            neg_inter_folds[fold_id].append([drug_node_id, protein_node_id])
            neg_label_folds[fold_id].append(inter_type)
        else:
            print("inter type has problem")

    train_positive_inter_pos = [[],[],[],[],[]]
    val_positive_inter_pos = [[],[],[],[],[]]
    train_negative_inter_pos = [[],[],[],[],[]]
    val_negative_inter_pos = [[],[],[],[],[]]

    train_interact_pos = [[],[],[],[],[]]
    val_interact_pos = [[],[],[],[],[]]
    train_label = [[],[],[],[],[]]
    val_label = [[],[],[],[],[]]

    for i in range(5):
        val_fold_id = i
        for j in range(5):
            if j != val_fold_id:
                train_positive_inter_pos[i] += pos_inter_folds[j]
                train_negative_inter_pos[i] += neg_inter_folds[j]

                train_interact_pos[i] += inter_folds[j]
                train_label[i] += label_folds[j]
            else:
                val_positive_inter_pos[i] += pos_inter_folds[j]
                val_negative_inter_pos[i] += neg_inter_folds[j]

                val_interact_pos[i] += inter_folds[j]
                val_label[i] += label_folds[j]
        """
            shuffle the data
        """
        train_interact_pos[i] = np.array(train_interact_pos[i])
        train_label[i] = np.array(train_label[i])
        shuffle_train = np.arange(0,len(train_interact_pos[i]))
        random.shuffle(shuffle_train)
        train_interact_pos[i] = train_interact_pos[i][shuffle_train]
        train_label[i] = train_label[i][shuffle_train]

        val_interact_pos[i] = np.array(val_interact_pos[i])
        val_label[i] = np.array(val_label[i])
        shuffle_test = np.arange(0,len(val_interact_pos[i]))
        random.shuffle(shuffle_test)
        val_interact_pos[i] = val_interact_pos[i][shuffle_test]
        val_label[i] = val_label[i][shuffle_test]




    """
    get the data needed for encoder
    """

    processdata_encoder(dataset, train_positive_inter_pos, pos_num)

    return train_positive_inter_pos, val_positive_inter_pos, train_interact_pos, train_label, val_interact_pos, val_label, train_negative_inter_pos


def load_data(seed, data_root, dataset, start_epoch=0, end_epoch=2000, crossval=True):

    # cv 5
    drugfeat, microbefeat, adj, folds, nodefeatures = cross_5_folds(dataset)

    root_save_dir = os.path.join(data_root, dataset)
    root_save_dir = os.path.join(root_save_dir, 'preprocess')
    if not os.path.exists(root_save_dir):
        os.mkdir(root_save_dir)


    drug_data = drugfeat.astype(np.float32)
    protein_data = microbefeat.astype(np.float32)
    int_label = adj.astype(np.int64)
    groups = folds.astype(np.float32)
    protein_num = len(protein_data)
    drug_num = len(drug_data)
    node_num = protein_num + drug_num
    positive_sample_num = len(int_label)//2
    train_positive_inter_pos, val_positive_inter_pos, test_train_interact_pos, test_train_label, \
    test_val_interact_pos, test_val_label, train_negative_inter_pos = first_spilt_label(int_label, groups, seed)

    test_adj_list = []
    # construct test file
    fold_num = 5 if crossval else 1
    for i in range(fold_num):
        test_dti_inter_mat = -np.ones((drug_num, protein_num))
        for j, inter in enumerate(test_train_interact_pos[i]):
            protein_id = inter[1]
            drug_id = inter[0]
            label = test_train_label[i][j]
            test_dti_inter_mat[drug_id][protein_id] = label
        for j, inter in enumerate(test_val_interact_pos[i]):
            protein_id = inter[1]
            drug_id = inter[0]
            label = test_val_label[i][j]
            test_dti_inter_mat[drug_id][protein_id] = label
        test_dti_inter_mat = test_dti_inter_mat.tolist()

        # construct adj
        test_adj_transform = constr_adj(node_num, train_positive_inter_pos[i], train_negative_inter_pos[i], drug_num)
        test_adj_list.append(test_adj_transform)

        np.savez(os.path.join(root_save_dir, '0_' + str(i) + '.npz'),adj=test_adj_transform, dti_inter_mat=test_dti_inter_mat,
                 train_interact_pos = test_train_interact_pos[i], val_interact_pos=test_val_interact_pos[i])
    for epoch in tqdm(range(start_epoch+1, end_epoch)):
        print("******epoch", epoch, "******")
        # 随机获得与正样本等量的负样本数量，
        dti_list, train_interact_pos, val_interact_pos = \
            add_dti_info(protein_num, drug_num, positive_sample_num, train_positive_inter_pos,
                         val_positive_inter_pos, test_val_interact_pos)
        for i in range(fold_num):
            np.savez(os.path.join(root_save_dir, str(epoch) + '_' + str(i) + '.npz'), adj=test_adj_list[i], dti_inter_mat=dti_list[i],
                     train_interact_pos=train_interact_pos[i].tolist(), val_interact_pos=val_interact_pos[i].tolist())

def constr_adj(node_num, interact_index, neg_inter_index, drug_num):
    """
    DAD
    """
    interact_index = np.array(interact_index)
    adj = sp.csr_matrix((np.ones((len(interact_index))), (interact_index[:, 0], interact_index[:, 1]+drug_num)), shape=(node_num, node_num)).toarray()
    adj = adj + adj.transpose()
    adj = preprocess_adj(adj)
    return adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='drugvirus',  # drugvirus  enzyme
                        help='dataset')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='epoch_num')
    parser.add_argument('--end_epoch', type=int, default=60,
                        help='epoch_num')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='data root')
    parser.add_argument('--crossval', type=int, default=1,
                        help='whether generate 5 fold')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    load_data(seed, args.data_root, dataset=args.dataset, start_epoch=args.start_epoch, end_epoch=args.end_epoch, crossval=args.crossval)
