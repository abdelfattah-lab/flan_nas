import json
import torch
from embedding_generator import NASBench101, NASBench201, NASBench301
from tqdm import tqdm

from sklearn.manifold import TSNE
# from sklearn.datasets import load_iris
from numpy import reshape
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  
import random
import numpy as np
import os

path = './'

skip_adj = False
nb101 = NASBench101()
nb201 = NASBench201()
nb301 = NASBench301(use_nb3_performance_model=True)
'''
Segment each of these into the following 6 classes:
Top 1%    : 5
Top 3%    : 4
Top 5%    : 3
Top 10%   : 2
Top 20%   : 1
Remaining : 0
'''

# def thresholdize(array, divisions=[99, 97, 95, 90, 80]):
#     thresholds = [np.percentile(array, q) for q in [99, 97, 95, 90, 80]]
#     thresholds_map = {0: thresholds[4], 1: thresholds[3], 2: thresholds[2], 3: thresholds[1], 4: thresholds[0]}
#     array = [5 if acc > thresholds_map[4] else 4 if acc > thresholds_map[3] else 3 if acc > thresholds_map[2] else 2 if acc > thresholds_map[1] else 1 if acc > thresholds_map[0] else 0 for acc in array]
#     return array
def thresholdize(array, divisions=[99, 97, 95, 90, 80]):
    thresholds = [np.percentile(array, q) for q in [99, 97, 95, 90, 80, 70, 60, 50]]
    thresholds_map = {0: thresholds[7], 1: thresholds[6], 2: thresholds[5], 3: thresholds[4], 4: thresholds[3], 5: thresholds[2], 6: thresholds[1], 7: thresholds[0]}
    array = [7 if acc > thresholds_map[7] else 6 if acc > thresholds_map[6] else 5 if acc > thresholds_map[5] else 4 if acc > thresholds_map[4] else 3 if acc > thresholds_map[3] else 2 if acc > thresholds_map[2] else 1 if acc > thresholds_map[1] else 0 for acc in array]
    # array = [5 if acc > thresholds_map[4] else 4 if acc > thresholds_map[3] else 3 if acc > thresholds_map[2] else 2 if acc > thresholds_map[1] else 1 if acc > thresholds_map[0] else 0 for acc in array]
    return array

def stackize(dataframe, preserve_names):
    for index, row in tqdm(dataframe.iterrows()):
        try:
            dataframe.at[index, preserve_names] = np.asarray(row[preserve_names]).tolist()
        except:
            dataframe.at[index, preserve_names] = row[preserve_names].detach().numpy().tolist()
    dataframe = pd.DataFrame(dataframe[preserve_names].values.tolist(), index=dataframe.index)
    return dataframe

if True:
    # subsample_adj = 400000
    # subsample_others = 400000
    subsample_adj = 20000
    subsample_others = 20000


    arch2vec_nb101 = pd.DataFrame.from_dict(nb101.arch2vec_nb101).T
    cate_nb101 = pd.DataFrame.from_dict(nb101.cate_nb101)
    zcp_nb101 = pd.DataFrame.from_dict(nb101.zcp_nb101['cifar10']).T
    if subsample_others!=None:
        arch2vec_nb101 = arch2vec_nb101.sample(n=subsample_others)
        cate_nb101 = cate_nb101.sample(n=subsample_others)
        zcp_nb101 = zcp_nb101.sample(n=subsample_others)
    x1 = np.asarray(cate_nb101['valid_accs'])
    x2 = np.asarray(zcp_nb101['val_accuracy'])
    x3 = np.asarray(arch2vec_nb101['valid_accuracy'])
    cate_nb101 = stackize(cate_nb101[['embeddings']], 'embeddings')
    arch2vec_nb101 = stackize(arch2vec_nb101[['feature']], 'feature')
    cate_nb101['tsne_acc'] = thresholdize(x1)
    zcp_nb101['tsne_acc'] = thresholdize(x2)
    arch2vec_nb101['tsne_acc'] = thresholdize(x3)
    zcp_nb101 = zcp_nb101[['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm',
        'nwot', 'params', 'plain', 'snip', 'synflow', 'zen', 'tsne_acc']]
    # subsample_adj = 15620
    # subsample_others = 15620
    subsample_adj = 15620
    subsample_others = 15620
    
    arch2vec_nb201 = pd.DataFrame.from_dict(nb201.arch2vec_nb201).T
    cate_nb201 = pd.DataFrame.from_dict(nb201.cate_nb201)
    zcp_nb201 = pd.DataFrame.from_dict(nb201.zcp_nb201['cifar10']).T
    zcp_nb201_valacc = pd.DataFrame.from_dict(nb201.zcp_nb201_valacc['cifar10']).T
    if subsample_others!=None:
        arch2vec_nb201 = arch2vec_nb201.sample(n=subsample_others)
        cate_nb201 = cate_nb201.sample(n=subsample_others)
        zcp_nb201 = zcp_nb201.sample(n=subsample_others)
    x1 = np.asarray(arch2vec_nb201['valid_accuracy'])
    x2 = np.asarray(cate_nb201['valid_accs'])
    x3 = np.asarray(zcp_nb201_valacc.loc[zcp_nb201.index]['val_accuracy'].tolist())
    cate_nb201 = stackize(cate_nb201[['embeddings']], 'embeddings')
    arch2vec_nb201 = stackize(arch2vec_nb201[['feature']], 'feature')
    arch2vec_nb201['tsne_acc'] = thresholdize(x1)
    cate_nb201['tsne_acc'] = thresholdize(x2)
    zcp_nb201['tsne_acc'] = thresholdize(x3)
    zcp_nb201 = zcp_nb201[['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm',
        'nwot', 'params', 'plain', 'snip', 'synflow', 'zen', 'tsne_acc']]
    # subsample_adj = 500000
    # subsample_others = 500000
    subsample_adj = 20000
    subsample_others = 20000
    
    arch2vec_nb301 = pd.DataFrame.from_dict(nb301.arch2vec_nb301).T
    cate_nb301 = pd.DataFrame.from_dict(nb301.cate_nb301)
    if subsample_others!=None:
        arch2vec_nb301 = arch2vec_nb301.sample(n=subsample_others)
        cate_nb301 = cate_nb301.sample(n=subsample_others)
    x1 = np.asarray(cate_nb301['predicted_accs'])
    x2 = np.asarray(arch2vec_nb301['valid_accuracy'])
    cate_nb301 = stackize(cate_nb301[['embeddings']], 'embeddings')
    arch2vec_nb301 = stackize(arch2vec_nb301[['feature']], 'feature')
    cate_nb301['tsne_acc'] = thresholdize(x1)
    arch2vec_nb301['tsne_acc'] = thresholdize(x2)
    zcp_nb301 = pd.DataFrame.from_dict(nb301.cate_nb301_zcp['cifar10']).T
    zcp_nb301['tsne_acc'] = thresholdize(np.asarray(pd.DataFrame.from_dict(nb301.cate_nb301_zcp_valacc['cifar10']).T['val_accuracy'].tolist()))

if not skip_adj:
    # subsample_adj = 400000
    # subsample_others = 400000
    subsample_adj = 20000
    subsample_others = 20000


    nb1_adj = {}
    acc_list = []
    hash_iterator = list(nb101.nb1_api.hash_iterator())
    # if subsample_adj==None:
    #     for i in tqdm(range(len(nb101.nb1_api.hash_iterator()))):
    if subsample_adj==None:
        for i in tqdm(range(len(nb101.nb1_api.hash_iterator()))):
    
            nb1_adj[str(i)] = nb101.get_adj(i)
            acc_list.append(nb101.nb1_api.get_metrics_from_hash(hash_iterator[i])[1][108][0]['final_validation_accuracy'])
    else:
        # for i in tqdm(random.sample(range(len(nb101.nb1_api.hash_iterator())), subsample_adj)):
        #     nb1_adj[str(i)] = nb101.get_adj(i)
        for i in tqdm(random.sample(range(len(nb101.nb1_api.hash_iterator())), subsample_adj)):
            nb1_adj[str(i)] = nb101.get_adj(i)
        
            acc_list.append(nb101.nb1_api.get_metrics_from_hash(hash_iterator[i])[1][108][0]['final_validation_accuracy'])
    nb1_adj = pd.DataFrame.from_dict(nb1_adj).T.reset_index(drop=True)
    nb1_adj['tsne_acc'] = thresholdize(acc_list)

if not skip_adj:
    # subsample_adj = 15620
    # subsample_others = 15620
    subsample_adj = 15620
    subsample_others = 15620
    nb2_adj = {}
    acc_list = []
    # if subsample_adj==None:
    #     for i in tqdm(range(len(nb201.nb2_api))):
    if subsample_adj==None:
        for i in tqdm(range(len(nb201.nb2_api))):
    
            nb2_adj[str(i)] = nb201.get_adj(i)
            acc_list.append(nb201.get_valacc(i))
    else:
        # for i in tqdm(random.sample(range(len(nb201.nb2_api)), subsample_adj)):
        #     nb2_adj[str(i)] = nb201.get_adj(i)
        for i in tqdm(random.sample(range(len(nb201.nb2_api)), subsample_adj)):
            nb2_adj[str(i)] = nb201.get_adj(i)
        
            acc_list.append(nb201.get_valacc(i))
    nb2_adj = pd.DataFrame.from_dict(nb2_adj).T.reset_index(drop=True)
    nb2_adj['tsne_acc'] = thresholdize(acc_list)

if not skip_adj:
    # subsample_adj = 500000
    # subsample_others = 500000
    subsample_adj = 20000
    subsample_others = 20000


    nb3_adj = {}
    acc_list = []
    # if subsample_adj==None:
    #     for i in tqdm(range(1000000)):
    if subsample_adj==None:
        for i in tqdm(range(1000000)):
    
            nb3_adj[str(i)] = nb301.get_adj(i)
            acc_list.append(nb301.get_valacc(i))
    else:
        # for i in tqdm(random.sample(range(1000000), subsample_adj)):
        #     nb3_adj[str(i)] = nb301.get_adj(i)
        for i in tqdm(random.sample(range(1000000), subsample_adj)):
            nb3_adj[str(i)] = nb301.get_adj(i)
        
            acc_list.append(nb301.get_valacc(i))
    nb3_adj = pd.DataFrame.from_dict(nb3_adj).T.reset_index(drop=True)
    nb3_adj['tsne_acc'] = thresholdize(acc_list)


if not skip_adj:
    nb3_zcp_adj = {}
    acc_list = []
    # if subsample_adj!=None:
    #     for i in tqdm(range(1000000, 1000000+11221, 1)):
    if subsample_adj!=None:
        for i in tqdm(range(1000000, 1000000+11221, 1)):
    
            nb3_zcp_adj[str(i)] = nb301.get_adj(i)
            acc_list.append(nb301.get_valacc(i))
    else:
        # for i in tqdm(random.sample(range(1000000, 1000000+11221, 1), subsample_adj)):
        #     nb3_zcp_adj[str(i)] = nb301.get_adj(i)
        for i in tqdm(random.sample(range(1000000, 1000000+11221, 1), subsample_adj)):
            nb3_zcp_adj[str(i)] = nb301.get_adj(i)
        
            acc_list.append(nb301.get_valacc(i))
    nb3_zcp_adj = pd.DataFrame.from_dict(nb3_zcp_adj).T.reset_index(drop=True)
    nb3_zcp_adj['tsne_acc'] = thresholdize(acc_list)

if skip_adj:
    master_dict = {"nb101": {"cate": cate_nb101, "zcp": zcp_nb101, "arch2vec": arch2vec_nb101}}
    master_dict["nb201"] = {"cate": cate_nb201, "zcp": zcp_nb201, "arch2vec": arch2vec_nb201}
    master_dict["nb301"] = {"cate": cate_nb301, "arch2vec": arch2vec_nb301}
    master_dict["nb301"] = {"zcp": zcp_nb301}
else:
    master_dict = {"nb101": {"cate": cate_nb101, "zcp": zcp_nb101, "arch2vec": arch2vec_nb101, "adj": nb1_adj}}
    master_dict["nb201"] = {"cate": cate_nb201, "zcp": zcp_nb201, "arch2vec": arch2vec_nb201, "adj": nb2_adj}
    master_dict["nb301"] = {"cate": cate_nb301, "arch2vec": arch2vec_nb301, "adj": nb3_adj}
    master_dict["nb301"] = {"zcp": zcp_nb301, "adj": nb3_zcp_adj}


# Make figures folder if it doesnt exist
# master_dict['nb101']['arch2vec'].iloc[:, -1]
if not os.path.exists("./figures"):
    os.makedirs("./figures")

# from cuml.manifold import TSNE
# import plt


        # tsne = TSNE(n_components = 2)

for search_space in master_dict.keys():
    for method in master_dict[search_space].keys():
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(master_dict[search_space][method].iloc[:, :-1])
        df = pd.DataFrame()
        df["y"] = master_dict[search_space][method].iloc[:, -1]
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]
        sns.scatterplot(x="comp-1", y="comp-2", s=3, hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 10),
                        data=df).set(title=f"{search_space} {method} T-SNE projection")
        plt.savefig(f"./figures/15k_{search_space}_{method}_tsne.png")
        plt.cla()
        plt.clf()

