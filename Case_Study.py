# 27837 ：Yue Yu
# 2022/7/10 10:02
# Case_Study.py
# PyCharm
# -*-coding:utf-8-*-

import numpy as np
import scipy.sparse as sp
import pandas as pd



datasets = ['drugvirus','MDAD','aBiofilm']
i = 0
train_times = 0

# path = 'reds_result_number_{}_{}_folds_{}.txt'.format(i, dataset, train_times)
path_DrugVirus = 'preds_result_number_0_DrugVirus_folds_1.txt'
path_MDAD = 'preds_result_number_5_MDAD_folds_3.txt'
path_aBiofilm = 'preds_result_number_7_aBiofilm_folds_2.txt'


def load_adj_preds(dataset, path_dataset):
    adj = np.loadtxt('./data/{}/adj.txt'.format(dataset))
    preds = np.loadtxt(path_dataset)
    for i in adj:
        i[0] -= 1
        i[1] -= 1
    adj = adj.astype("int")
    D = preds.shape[0]
    M = preds.shape[1]
    dm = np.array(adj)
    adj_matrix = sp.csr_matrix((dm[:, 2], (dm[:, 0], dm[:, 1])), shape=(D, M)).toarray()

    return adj_matrix, preds, adj

def get_ranking(dataset, path, adj_matrix, preds, adj_index):



    preds_sorted_index = (-preds).argsort()
    preds_sorted = -np.sort(-preds)

    preds_ranking = np.zeros((adj_matrix.shape[0] * adj_matrix.shape[1],3))
    temp = np.array(range(adj_matrix.shape[1]))
    temp = temp + 1
    for i in range(adj_matrix.shape[0]):
        preds_ranking[i*adj_matrix.shape[1]:(i+1)*adj_matrix.shape[1], 0] = i
        preds_ranking[i * adj_matrix.shape[1]:(i + 1) * adj_matrix.shape[1], 2] = temp
    preds_ranking[:,1] = preds_sorted_index.reshape(-1)

    for i in adj_index:
        j0 = i[0] * adj_matrix.shape[1]
        for j in range(adj_matrix.shape[1]):
            k = j0 + j
            if preds_ranking[k,0] == i[0] and preds_ranking[k,1] == i[1]:
                preds_ranking[k, 2] = -1
                break
    np.savetxt("{}_preds_ranking.txt".format(dataset), preds_ranking)
    return preds_ranking


def load_name(dataset):
    with open('./data/{}/drugs.txt'.format(dataset)) as file:
        drug_name = file.read().splitlines()
        file.close()
    drug_name = np.array(drug_name)
    with open('./data/{}/microbes.txt'.format(dataset)) as file:
        microbes_name = file.read().splitlines()
        file.close()
    microbes_name = np.array(microbes_name)
    return drug_name, microbes_name

def change_id2name(dataset, preds_ranking, drug_name, microbe_name):
    preds_ranking_id2name = np.empty_like(preds_ranking).astype(np.str)
    with open("{}_preds_ranking2name.txt".format(dataset),"a+") as file:

        for i in range(len(preds_ranking)):
            drug_id = preds_ranking[i,0].astype(np.int)
            microbe_id = preds_ranking[i,1].astype(np.int)
            rank = preds_ranking[i,2].astype(np.int)
            preds_ranking_id2name[i,0] = drug_name[drug_id]
            preds_ranking_id2name[i,1] = microbe_name[microbe_id]
            preds_ranking_id2name[i,2] = rank
            file.write(preds_ranking_id2name[i,0] +'\t' +
                       preds_ranking_id2name[i,1] + '\t' + preds_ranking_id2name[i,2] + '\n')
        file.close()

    return preds_ranking_id2name


"""
get top n preds that not apear in the training data
"""
def get_top_n(dataset, preds_ranking2name, drug_num, microbe_num, top_n = 20):
    with open("{}_preds_top_{}.txt".format(dataset,top_n), "a+") as file:
        for i in range(drug_num):
            count = 0
            for j in range(microbe_num):
                k = i * microbe_num + j
                if count < top_n and preds_ranking2name[k,2].astype(np.int) > 0:
                    file.write(preds_ranking2name[k,0] +'\t' +
                       preds_ranking2name[k,1] + '\t' + preds_ranking2name[k,2] + '\n')
                    count = count+1
                elif count >= 20:
                    break

        file.close()



"""
start process
"""
# adj_matrix, preds, adj_index = load_adj_preds(datasets[0], path_DrugVirus)
# DrugVirus_preds_ranking = get_ranking(datasets[0],path_DrugVirus, adj_matrix, preds, adj_index)
# DrugVirus_drug_name, DrugVirus_microbes_name = load_name(datasets[0])
# DrugVirus_preds_ranking2name = change_id2name(datasets[0], DrugVirus_preds_ranking,DrugVirus_drug_name, DrugVirus_microbes_name)
# get_top_n(datasets[0],DrugVirus_preds_ranking2name,len(DrugVirus_drug_name),len(DrugVirus_microbes_name))
#
# adj_matrix, preds, adj_index = load_adj_preds(datasets[1], path_MDAD)
# MDAD_preds_ranking = get_ranking(datasets[1],path_MDAD, adj_matrix, preds, adj_index)
# MDAD_drug_name, MDAD_microbes_name = load_name(datasets[1])
# MDAD_preds_ranking2name = change_id2name(datasets[1], MDAD_preds_ranking,MDAD_drug_name, MDAD_microbes_name)
# get_top_n(datasets[1],MDAD_preds_ranking2name,len(MDAD_drug_name),len(MDAD_microbes_name))
#
#
# adj_matrix, preds, adj_index = load_adj_preds(datasets[2], path_aBiofilm)
# aBiofilm_preds_ranking = get_ranking(datasets[2],path_aBiofilm, adj_matrix, preds, adj_index)
# aBiofilm_drug_name, aBiofilm_microbes_name = load_name(datasets[2])
# aBiofilm_preds_ranking2name = change_id2name(datasets[2], aBiofilm_preds_ranking,aBiofilm_drug_name, aBiofilm_microbes_name)
# get_top_n(datasets[2],aBiofilm_preds_ranking2name,len(aBiofilm_drug_name),len(aBiofilm_microbes_name))


"""
above uesd to process drug2microbe ranking
below we process microbe2drug, for each microbe we get top n drug
"""
adj_matrix, preds, adj_index = load_adj_preds(datasets[0], path_DrugVirus)
adj_matrix = adj_matrix.transpose()
preds = preds.transpose()
temp = np.zeros((len(adj_index),3))
temp[:,0] = adj_index[:,1]
temp[:,1] = adj_index[:,0]
temp[:,2] = adj_index[:,2]
adj_index = temp
temp_b = adj_index[:,0]
index = np.lexsort((temp_b,))
adj_index = adj_index[index].astype(np.int)


DrugVirus_preds_ranking = get_ranking(datasets[0],path_DrugVirus, adj_matrix, preds, adj_index)
DrugVirus_drug_name, DrugVirus_microbes_name = load_name(datasets[0])
DrugVirus_preds_ranking2name = change_id2name(datasets[0], DrugVirus_preds_ranking,DrugVirus_microbes_name,DrugVirus_drug_name)
get_top_n(datasets[0],DrugVirus_preds_ranking2name,len(DrugVirus_microbes_name),len(DrugVirus_drug_name))


adj_matrix, preds, adj_index = load_adj_preds(datasets[1], path_MDAD)
adj_matrix = adj_matrix.transpose()
preds = preds.transpose()
temp = np.zeros((len(adj_index),3))
temp[:,0] = adj_index[:,1]
temp[:,1] = adj_index[:,0]
temp[:,2] = adj_index[:,2]
adj_index = temp
temp_b = adj_index[:,0]
index = np.lexsort((temp_b,))
adj_index = adj_index[index].astype(np.int)

MDAD_preds_ranking = get_ranking(datasets[1],path_MDAD, adj_matrix, preds, adj_index)
MDAD_drug_name, MDAD_microbes_name = load_name(datasets[1])
MDAD_preds_ranking2name = change_id2name(datasets[1], MDAD_preds_ranking, MDAD_microbes_name,MDAD_drug_name)
get_top_n(datasets[1],MDAD_preds_ranking2name,len(MDAD_microbes_name),len(MDAD_drug_name))


adj_matrix, preds, adj_index = load_adj_preds(datasets[2], path_aBiofilm)
adj_matrix = adj_matrix.transpose()
preds = preds.transpose()
temp = np.zeros((len(adj_index),3))
temp[:,0] = adj_index[:,1]
temp[:,1] = adj_index[:,0]
temp[:,2] = adj_index[:,2]
adj_index = temp
temp_b = adj_index[:,0]
index = np.lexsort((temp_b,))
adj_index = adj_index[index].astype(np.int)

aBiofilm_preds_ranking = get_ranking(datasets[2],path_aBiofilm, adj_matrix, preds, adj_index)
aBiofilm_drug_name, aBiofilm_microbes_name = load_name(datasets[2])
aBiofilm_preds_ranking2name = change_id2name(datasets[2], aBiofilm_preds_ranking, aBiofilm_microbes_name,aBiofilm_drug_name)
get_top_n(datasets[2],aBiofilm_preds_ranking2name,len(aBiofilm_microbes_name),len(aBiofilm_drug_name))






a = 0



















