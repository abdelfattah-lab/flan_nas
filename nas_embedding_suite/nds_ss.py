import torch
import json
from tqdm import tqdm
import types
import copy
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing
import random, time
import sys, os

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
NDS_DPATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/NDS/nds_data/'
# nba and nbb are two dictionaries with same keys. for each key in nba, add the value of nbb[key]['feature'] to nba[key]['feature']
# def merge_arch2vec(nba, nbb):
# for key in nba.keys():
#     nba[key]['feature'] = nba[key]['feature'] + nbb[key]['feature']
# return nba
class NDS:
    def __init__(self, path=None, zcp_dict=False, normalize_zcp=True, log_synflow=True, embedding_list = ['adj',
                                                                    'adj_op',
                                                                    'path',
                                                                    'one_hot',
                                                                    'path_indices',
                                                                    'zcp']):
        adj_path = BASE_PATH + 'nds_adj_encoding/'
        # self.spaces = ['Amoeba.json', 'NASNet.json','DARTS.json','ENAS.json','PNAS.json']
        # self.spaces = ["GNPnb101.json", "GNPnb201c10.json", "GNPnb201c100.json", "GNPnb201imgnet.json", "GNPnb301.json", "GNPofa.json", "GNPofa_resnet.json", 'Amoeba.json','PNAS_fix-w-d.json','ENAS_fix-w-d.json','NASNet.json','DARTS.json','ENAS.json','PNAS.json','DARTS_lr-wd.json','DARTS_fix-w-d.json']
        self.spaces = ["GNPnb101.json", "GNPofa.json", "GNPofa_resnet.json", 'Amoeba.json','PNAS_fix-w-d.json','ENAS_fix-w-d.json','NASNet.json','DARTS.json','ENAS.json','PNAS.json','DARTS_lr-wd.json','DARTS_fix-w-d.json']
        self.cate_embeddings = {k.replace(".json", ""): torch.load(BASE_PATH + 'cate_embeddings/cate_nds_{}.pt'.format(k.replace(".json", ""))) for k in self.spaces}
        self.cate_embeddings['nb301'] = torch.load(BASE_PATH + 'cate_embeddings/cate_nb301.pt')
        self.arch2vec_embeddings = {k.replace(".json", ""): torch.load(BASE_PATH + "arch2vec_embeddings/arch2vec-model-dim_32_search_space_{}-nds.pt".format(k.replace(".json", ""))) for k in self.spaces}
        self.arch2vec_embeddings['nb301'] = torch.load(BASE_PATH + "arch2vec_embeddings/arch2vec-model-dim_32_search_space_nb301-nb301.pt")
        self.spaces = ['Amoeba.json','PNAS_fix-w-d.json','ENAS_fix-w-d.json','NASNet.json','DARTS.json','ENAS.json','PNAS.json','DARTS_lr-wd.json','DARTS_fix-w-d.json', 'nb301.json']
        self.space_dicts = {space.replace(".json", ""): json.load(open(NDS_DPATH + space, "r")) for space in self.spaces}
        # self.spaces = ["GNPnb101.json", "GNPnb201c10.json", "GNPnb201c100.json", "GNPnb201imgnet.json", "GNPnb301.json", "GNPofa.json", "GNPofa_resnet.json", 'Amoeba.json','PNAS_fix-w-d.json','ENAS_fix-w-d.json','NASNet.json','DARTS.json','ENAS.json','PNAS.json','DARTS_lr-wd.json','DARTS_fix-w-d.json', 'nb301.json']
        self.spaces = ["GNPnb101.json", "GNPofa.json", "GNPofa_resnet.json", 'Amoeba.json','PNAS_fix-w-d.json','ENAS_fix-w-d.json','NASNet.json','DARTS.json','ENAS.json','PNAS.json','DARTS_lr-wd.json','DARTS_fix-w-d.json', 'nb301.json']
        self.space_adj_mats = {space.replace(".json", ""): json.load(open(adj_path + space, "r")) for space in self.spaces}
        self.gnpaccdict = {space.replace(".json", ""): json.load(open(BASE_PATH + "gnp_accs/{}.json".format(space.replace(".json", "")))) for space in self.spaces if space.__contains__("GNP")}
        self.zcps = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']
        self.normalize_zcp = normalize_zcp
        self.zcp_nds_norm = {}
        if self.normalize_zcp:
            for task_ in self.spaces:
                print("normalizing task: ", task_)
                self.norm_zcp = pd.read_csv(BASE_PATH + "nds_zcps/" + task_.replace(".json", "") + "_zcps.csv", index_col=0)
                self.norm_zcp = self.norm_zcp[self.zcps]
                if task_.__contains__('nb301') == False:
                    minfinite = self.norm_zcp['zen'].replace(-np.inf, 1000).min()
                    self.norm_zcp['zen'] = self.norm_zcp['zen'].replace(-np.inf, minfinite)
                    if log_synflow:
                        self.norm_zcp['synflow'] = self.norm_zcp['synflow'].replace(0, 1e-2)
                        self.norm_zcp['synflow'] = np.log10(self.norm_zcp['synflow'])
                    else:
                        print("WARNING: Not taking log of synflow values for normalization results in very small synflow inputs")
                    minfinite = self.norm_zcp['synflow'].replace(-np.inf, 1000).min()
                    self.norm_zcp['synflow'] = self.norm_zcp['synflow'].replace(-np.inf, minfinite)
                    min_max_scaler = preprocessing.QuantileTransformer()
                    self.norm_zcp = pd.DataFrame(min_max_scaler.fit_transform(self.norm_zcp), columns=self.zcps, index=self.norm_zcp.index)
                self.zcp_nds_norm[task_.replace(".json", "")] = self.norm_zcp.T.to_dict()
        for task_ in self.spaces:
            # normalize cate_embeddings[space]['embeddings']
            min_max_scaler = preprocessing.MinMaxScaler()
            self.cate_embeddings[task_.replace(".json", "")]['embeddings'] = min_max_scaler.fit_transform(self.cate_embeddings[task_.replace(".json", "")]['embeddings'])
        self.all_accs = {}
        self.unnorm_all_accs = {}
        self.minmax_sc = {}
        for space in self.spaces:
            space = space.replace(".json", "")
            self.all_accs[space] = []
            if space.__contains__("GNP"):
                for idx in range(len(self.gnpaccdict[space])):
                    self.all_accs[space].append(self.gnpaccdict[space][str(idx)])
            else:
                for idx in range(len(self.space_dicts[space])):
                    if space == 'nb301':
                        self.all_accs[space].append(float(self.space_dicts[space][str(idx)]['test_ep_top1'])/100.)
                    else:
                        self.all_accs[space].append(float(100.-self.space_dicts[space][idx]['test_ep_top1'][-1])/100.)
            # RobustScaler normalize this space
            min_max_scaler = preprocessing.QuantileTransformer()
            _ = min_max_scaler.fit_transform(np.array(self.all_accs[space]).reshape(-1, 1)).flatten()
            # self.unnorm_all_accs[space] = np.array(self.all_accs[space]).reshape(-1, 1).flatten().tolist()
            # self.all_accs[space] = self.all_accs[space].tolist()
            self.minmax_sc[space] = min_max_scaler
        self.unnorm_all_accs = self.all_accs # need to comment out this line.
        self.all_accs = self.all_accs

    #################### Key Functions Begin ###################
    def get_adjmlp_zcp(self, idx, space="Amoeba"):
        if space=='nds_nb301':
            space = 'nb301'
        adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = self.get_adj_op(idx, space=space).values()
        adj_mat_norm = np.asarray(adj_mat_norm).flatten()
        adj_mat_red = np.asarray(adj_mat_red).flatten()
        op_mat_norm = torch.Tensor(np.asarray(op_mat_norm)).argmax(dim=1).numpy().flatten() # Careful here.
        op_mat_red = torch.Tensor(np.asarray(op_mat_red)).argmax(dim=1).numpy().flatten() # Careful here.
        op_mat_norm = op_mat_norm/np.max(op_mat_norm)
        op_mat_red = op_mat_red/np.max(op_mat_red)
        return np.concatenate((adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red, np.asarray(self.get_zcp(idx, space)).flatten())).tolist()

    def get_a2vcatezcp(self, idx, space="Amoeba", joint=None):
        if space=='nds_nb301':
            space = 'nb301'
        a2v = self.get_arch2vec(idx, joint=joint, space=space)
        if not isinstance(a2v, list):
            a2v = a2v.tolist()
        cate = self.get_cate(idx, joint=joint, space=space)
        if not isinstance(cate, list):
            cate = cate.tolist()
        zcp = self.get_zcp(idx, joint=joint, space=space)
        if not isinstance(zcp, list):
            zcp = zcp.tolist()
        return a2v + cate + zcp
    
    def get_zcp(self, idx, space="Amoeba", joint=None):
        if space=='nds_nb301':
            space = 'nb301'
        return list(self.zcp_nds_norm[space][idx].values())
    
    def get_adj_op(self, idx, space="Amoeba", bin_space=None):
        if space=='nds_nb301':
            space = 'nb301'
        return self.space_adj_mats[space][str(idx)]
    
    def get_cate(self, idx, space="Amoeba", joint=None):
        if space=='nds_nb301':
            space = 'nb301'
        return self.cate_embeddings[space]['embeddings'][idx].tolist()
    
    def get_arch2vec(self, idx, space="Amoeba", joint=None):
        if space=='nds_nb301':
            space = 'nb301'
        return self.arch2vec_embeddings[space][idx]['feature'].tolist()
    
    def get_valacc(self, idx, space="Amoeba"):
        if space=='nds_nb301':
            space = 'nb301'
        return self.unnorm_all_accs[space][idx]
    
    def get_numitems(self, space="Amoeba"):
        if space=='nds_nb301':
            space = 'nb301'
        if space.__contains__("GNP"):
            return len(self.gnpaccdict[space])
        else:
            if space == 'nb301':
                return len(self.space_adj_mats[space])
            else:
                return len(self.space_dicts[space])

    def get_norm_w_d(self, idx, space="Amoeba"):
        if space=='nds_nb301':
            space = 'nb301'
        try:
            if space.__contains__("GNP"):
                return [0,0]
            else:
                if space == 'nb301':
                    return [0,0]
                else:
                    return [self.space_dicts[space][idx]['net']['width']/32., \
                            self.space_dicts[space][idx]['net']['depth']/20.]
        except:
            print("WARNING: No width/depth information found for idx: ", idx, ",", space)
            exit(0)
    ##################### Key Functions End #####################

    def get_flops(self, idx, space="Amoeba"):
        if space.__contains__("GNP"):
            return 0
        else:
            if space=='nds_nb301':
                space = 'nb301'
            return self.space_dicts[space][idx]["flops"]

    def get_params(self, idx, space="Amoeba"):
        if space.__contains__("GNP"):
            return 0
        else:
            if space=='nds_nb301':
                space = 'nb301'
            return self.space_dicts[space][idx]["params"]
    

if __name__ == "__main__":
    nds = NDS()
    print(nds.get_adj_op(0, space="Amoeba"))
    print(nds.get_zcp(0, space="Amoeba"))
    print(nds.get_adj_op(0, space="nb301"))
    print(nds.get_zcp(0, space="nb301"))