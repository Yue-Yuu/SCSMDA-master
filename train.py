import datetime
import os
import re
import time
import argparse
import torch.nn as nn
import torch.optim as optim
from model import MDA_Graph
from utils import accuracy, mcc, auc, aupr, f1
import numpy as np
import torch
from utils_pre.load_data import load_data

from processdata import cross_5_folds, first_spilt_label, add_dti_info


###############################################################
# Training settings
parser = argparse.ArgumentParser(description='MDA-GRAPH')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,  # 10000
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005,  #0.0005
                    help='Initial learning rate.')
parser.add_argument("--k_bins", type=int, default=10, help='bin number of negtive sample')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--crossvalidation', type=int, default=1,
                    help='whether use crossvalidation or not')
###############################################################
# Model hyper setting
# Protein_NN
parser.add_argument('--protein_ninput', type=int, default=64,  # 220  95
                    help='microbe vector size')
parser.add_argument('--nn_nlayers', type=int, default=1,
                    help='microbe_nn layers num')
parser.add_argument('--pnn_nhid', type=str, default='[]',
                    help='pnn hidden layer dim, like [200,100] for tow hidden layers')
# Drug_NN
parser.add_argument('--drug_ninput', type=int, default=64,  # 881  175
                    help='Drug fingerprint dimension')
parser.add_argument('--dnn_nlayers', type=int, default=1,
                    help='dnn_nlayers num')
parser.add_argument('--dnn_nhid', type=str, default='[]',
                    help='dnn hidden layer dim, like [200,100] for tow hidden layers')

# Decoder
parser.add_argument('--DTI_nn_nlayers', type=int, default=1,
                    help='Protein_nn layers num')
parser.add_argument('--DTI_nn_nhid', type=str, default='[128,128,128]',
                    help='DTI_nn hidden layer dim, like [200,100] for tow hidden layers')
###############################################################
# data
parser.add_argument('--dataset', type=str, default='drugvirus',  # drugvirus  MDAD aBiofilm
                    help='dataset name')
parser.add_argument('--drug_num', type=int, default=175,  # 175 1373 1720
                    help='drug node num')
parser.add_argument('--microbe_num', type=int, default=95,  # 95 173 140
                    help='microbe node num')
parser.add_argument('--data_path', type=str, default='./data',
                    help='dataset root path')
parser.add_argument('--rate', type=int, default=1,  # 1...10
                    help='neg sample rate')
parser.add_argument('--pos_num', type=int, default=10,  # 2,4...10
                    help='pos_num, the pos sample number for each node in contrast')


###############################################################
# Encoder
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--tau', type=float, default=0.8)
parser.add_argument('--feat_drop', type=float, default=0.3)
parser.add_argument('--attn_drop', type=float, default=0.5)
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--mp_ngcn', type=int, default=1)
parser.add_argument('--sn_ngcn', type=int, default=1)
parser.add_argument("--encoder_loss_rate", type=float,default=0.5)
parser.add_argument('--embs_rate', type=float,default=0.99)

args = parser.parse_args()
args.nei_num = 2  # the number of neighbors' types
args.cuda = torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.set_device(1)

# nn layers
p1 = re.compile(r'[[](.*?)[]]', re.S)
if args.dnn_nhid == '[]':
    args.dnn_nhid = []
else:
    args.dnn_nhid = [int(i) for i in re.findall(p1, args.dnn_nhid)[0].replace(' ', '').split(',')]
if args.pnn_nhid == '[]':
    args.pnn_nhid = []
else:
    args.pnn_nhid = [int(i) for i in re.findall(p1, args.pnn_nhid)[0].replace(' ', '').split(',')]
args.DTI_nn_nhid = [int(i) for i in re.findall(p1, args.DTI_nn_nhid)[0].replace(' ', '').split(',')]

preprocess_path = os.path.join(args.data_path,args.dataset)
preprocess_path = os.path.join(preprocess_path, 'preprocess')

# Hyper Setting
pnn_hyper = [args.protein_ninput, args.pnn_nhid, args.hidden_dim, args.nn_nlayers]
Deco_hyper = [args.hidden_dim, args.DTI_nn_nhid, args.DTI_nn_nlayers]


def train(epoch, link_dti_id_train, train_dti_inter_mat, feats, pos, mps, nei_index, sn_edge, mp_edge,encoder_loss_rate):

    """
    link_dti_id_train: training set of drug-microbe interaction pairs, pos and neg
    train_dti_inter_mat: drug-microbe interaction mat, 1 for pos, 0 for neg used in test, -1 for other negs
    feats: input embeddings
    pos: pos pairs for contrast
    mps: meta-path network constructed previously
    nei_index: normalized sim adj matrix (DAD) for GCN
    sn_edge: similarity network edges
    mp_edge: meta-path network edges
    """

    t = time.time()
    model.train()
    optimizer.zero_grad()
    row_dti_id = link_dti_id_train.permute(1, 0)[0]
    col_dti_id = link_dti_id_train.permute(1, 0)[1]
    drug_index = row_dti_id
    protein_index = col_dti_id + train_dti_inter_mat.shape[0]
    output, loss_encoder, encoder_embeds = model(protein_index, drug_index, feats, pos, mps, nei_index, sn_edge, mp_edge)

    preds = output

    Loss = nn.BCELoss()
    loss_train = Loss(preds, train_dti_inter_mat[row_dti_id, col_dti_id])
    acc_dti_train = accuracy(preds, train_dti_inter_mat[row_dti_id, col_dti_id])
    loss_all = encoder_loss_rate *loss_train + loss_encoder * (1 - encoder_loss_rate)
    loss_all.backward()
    optimizer.step()
    if (epoch+1) % 10 ==0:
        print('Epoch {:04d} Train '.format(epoch + 1),
              'loss_all: {:.4f}'.format(loss_all.item()),
              'acc_dti_train: {:.4f}'.format(acc_dti_train),
              'time: {:.4f}s'.format(time.time() - t))
    optimizer.zero_grad()

def test(link_dti_id_test, test_dti_inter_mat, feats, pos, mps, nei_index, sn_edge, mp_edge,encoder_loss_rate):

    """
    link_dti_id_test: test set of drug-microbe interaction pairs, pos and neg
    train_dti_inter_mat: drug-microbe interaction mat, 1 for pos, 0 for neg used in test, -1 for other negs
    feats: input embeddings
    pos: pos pairs for contrast
    mps: meta-path network constructed previously
    nei_index: normalized sim adj matrix (DAD) for GCN
    sn_edge: similarity network edges
    mp_edge: meta-path network edges
    """

    model.eval()
    with torch.no_grad():

        """
        in test, we pretict all pairs result, and input them to the negative sampling module
        """
        row_dti_id = link_dti_id_test.permute(1, 0)[0]
        col_dti_id = link_dti_id_test.permute(1, 0)[1]
        drug_index = row_dti_id
        protein_index = col_dti_id + test_dti_inter_mat.shape[0]

        drug_index_all = torch.arange(len(pos[0])).reshape([-1,1])
        drug_index_all = drug_index_all.repeat(1,args.microbe_num).cuda()
        protein_index_all = torch.arange(len(feats) - len(pos[0])).repeat(args.drug_num,1).cuda()
        protein_index_all = protein_index_all + test_dti_inter_mat.shape[0]

        output, loss_encoder, encoder_embeds = model(protein_index_all, drug_index_all, feats, pos, mps, nei_index, sn_edge, mp_edge)


        preds = output.reshape(args.drug_num, args.microbe_num)
        preds = preds[drug_index, protein_index - args.drug_num]
        # preds = output

        Loss = nn.BCELoss()
        targets = test_dti_inter_mat[row_dti_id, col_dti_id]
        loss_test = Loss(preds, targets)
        loss_test = 0
        acc_dti_test = accuracy(preds, test_dti_inter_mat[row_dti_id, col_dti_id])
        if (epoch+1) % 10 == 0:
            print("train test acc: {}".format(acc_dti_test))
    return acc_dti_test, loss_test, preds, targets, output, encoder_embeds

pnn_hyper = [args.protein_ninput, args.pnn_nhid, args.hidden_dim, args.nn_nlayers]
Deco_hyper = [args.hidden_dim, args.DTI_nn_nhid, args.DTI_nn_nlayers]
# Train model
t_total = time.time()
acc_score = np.zeros(5)
auc_score = np.zeros(5)
aupr_score = np.zeros(5)
fold_num = 5 if args.crossvalidation else 1
fold_acc_score = np.zeros(5)
fold_auc_score = np.zeros(5)
fold_aupr_score = np.zeros(5)
fold_mcc_score = np.zeros(5)
fold_f1_score = np.zeros(5)


"""
conduct the 5-CV operation on the dataset 

drugfeat：inut drug features
microbefeat：input microbe features
adj：1:1 pos and neg pairs
folds：the 5-CV folds of 1:1 pos neg pairs
nodefeatures：concat drug features and microbe features
"""
drugfeat, microbefeat, adj, folds, nodefeatures = cross_5_folds(args.dataset, args.seed)

drug_data = drugfeat.astype(np.float32)
protein_data = microbefeat.astype(np.float32)
int_label = adj.astype(np.int64)
groups = folds.astype(np.float32)
microbe_num = len(protein_data)
drug_num = len(drug_data)
node_num = microbe_num + drug_num
positive_sample_num = len(int_label)//2


"""
based on the generated 5-CV adj cut the data into train and test
"""
train_positive_inter_pos, val_positive_inter_pos, test_train_interact_pos, test_train_label, \
test_val_interact_pos, test_val_label, train_negative_inter_pos = first_spilt_label(int_label, groups, args.seed, args.dataset,args.pos_num)
test_adj_list = []

t1 = time.time()

for train_times in range(fold_num):

    test_dti_inter_mat = -np.ones((drug_num, microbe_num))  # [microbe_num, drug_num]
    for j, inter in enumerate(test_train_interact_pos[train_times]):
        protein_id = inter[1]
        drug_id = inter[0]
        label = test_train_label[train_times][j]
        test_dti_inter_mat[drug_id][protein_id] = label
    for j, inter in enumerate(test_val_interact_pos[train_times]):
        protein_id = inter[1]
        drug_id = inter[0]
        label = test_val_label[train_times][j]
        test_dti_inter_mat[drug_id][protein_id] = label
    test_dti_inter_mat = test_dti_inter_mat.tolist()

    ori_dti_inter_mat = test_dti_inter_mat
    ori_train_interact_pos = test_train_interact_pos[train_times]
    ori_val_interact_pos = test_val_interact_pos[train_times]

    ori_dti_inter_mat = torch.FloatTensor(ori_dti_inter_mat)
    ori_train_interact_pos = torch.LongTensor(ori_train_interact_pos)
    ori_val_interact_pos = torch.LongTensor(ori_val_interact_pos)

    """
    load encoder data
    """
    nei_index, feats, mps, poss, sn_edge, mp_edge = load_data(args.dataset, train_times)
    feats_dim_list = feats.shape[0]
    P = int(len(mps))

    model = MDA_Graph(PNN_hyper=pnn_hyper, DECO_hyper=Deco_hyper,
                      microbe_num=args.microbe_num, Drug_num=args.drug_num, dropout=args.dropout, args_pre=args,
                      feats_dim_list=feats_dim_list, P=P,)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_test = 0

    output = torch.ones(drug_num * microbe_num)

    if args.cuda:
        model = model.cuda()
        ori_dti_inter_mat = ori_dti_inter_mat.cuda()
        ori_train_interact_pos = ori_train_interact_pos.cuda()
        ori_val_interact_pos = ori_val_interact_pos.cuda()

        feats = feats.cuda()
        mps = [mp.cuda() for mp in mps]
        pos = [pos.cuda() for pos in poss]
        nei_index = [nei.cuda() for nei in nei_index]
        sn_edge = [sc.cuda() for sc in sn_edge]
        mp_edge = [mp.cuda() for mp in mp_edge]


    for epoch in range(args.epochs):


        """
        construct neg samples for each iteration
        """
        dti_list, train_interact_pos, val_interact_pos = \
            add_dti_info(microbe_num, drug_num, ori_dti_inter_mat, positive_sample_num, train_positive_inter_pos[train_times]
                         ,val_positive_inter_pos[train_times], test_val_interact_pos, output, epoch, args, args.rate)
        dti_inter_mat = dti_list
        train_interact_pos = train_interact_pos
        val_interact_pos = val_interact_pos

        dti_inter_mat = torch.FloatTensor(dti_inter_mat)
        train_interact_pos = torch.LongTensor(train_interact_pos)
        val_interact_pos = torch.LongTensor(val_interact_pos)

        if args.cuda:
            dti_inter_mat = dti_inter_mat.cuda()
            train_interact_pos = train_interact_pos.cuda()
            val_interact_pos = val_interact_pos.cuda()
        if (epoch+1) % 10 == 0:
            print('Epoch: {:04d}'.format(epoch + 1), 'Train_times:', train_times)
        train(epoch, train_interact_pos, dti_inter_mat, feats, pos, mps, nei_index, sn_edge, mp_edge, args.encoder_loss_rate)
        test_score, test_loss, predicts, targets, output, encoder_embeds = test(val_interact_pos, dti_inter_mat, feats, pos, mps,
                                                                nei_index, sn_edge, mp_edge,args.encoder_loss_rate)
        if test_score > best_test:
            acc_score[train_times] = round(best_test, 4)


    # ：final test result for each fold
    test_score, test_loss, predicts, targets, output1, encoder_embeds = test(ori_val_interact_pos, ori_dti_inter_mat,
                                                    feats, pos, mps, nei_index, sn_edge, mp_edge, args.encoder_loss_rate)
    fold_test = test_score
    fold_acc_score[train_times] = round(accuracy(predicts,targets), 4)
    fold_auc_score[train_times] = round(auc(predicts, targets), 4)
    fold_aupr_score[train_times] = round(aupr(predicts, targets), 4)
    fold_mcc_score[train_times] = round(mcc(predicts, targets), 4)
    fold_f1_score[train_times] = round(f1(predicts, targets), 4)

    if train_times == 0:
        all_labels = targets.cpu().detach()
        all_preds = predicts.cpu().detach()
    else:
        all_labels = np.append(all_labels, targets.cpu().detach())
        all_preds = np.append(all_preds, predicts.cpu().detach())

    print("-***************************************----")
    print("acc socre:{} avg:{}".format(fold_acc_score,np.mean(fold_acc_score)))
    print("auc socre:{} avg:{}".format(fold_auc_score,np.mean(fold_auc_score)))
    print("aupr score:{} avg:{}".format(fold_aupr_score, np.mean(fold_aupr_score)))

t2 = time.time()
t_all = t2 - t1
print("五折交叉时间：{}".format(t_all))

print("------------------------")


