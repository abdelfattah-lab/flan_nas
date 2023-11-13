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
class NDSin:
    def __init__(self, path=None, zcp_dict=False, normalize_zcp=True, log_synflow=True, embedding_list = ['adj',
                                                                    'adj_op',
                                                                    'path',
                                                                    'one_hot',
                                                                    'path_indices',
                                                                    'zcp']):
        adj_path = BASE_PATH + 'nds_in_adj_encoding/'
        self.spaces = ["Amoeba_in.json", "DARTS_in.json", "DARTS_lr-wd_in.json", "ENAS_in.json", "NASNet_in.json", "PNAS_in.json"]
        self.space_dicts = {space.replace(".json", ""): json.load(open(NDS_DPATH + space, "r")) for space in self.spaces}
        # import pdb; pdb.set_trace()
        self.space_adj_mats = {space.replace(".json", ""): json.load(open(adj_path + space, "r")) for space in self.spaces}
        self.all_accs = {}
        self.unnorm_all_accs = {}
        self.minmax_sc = {}
        self.maxmetric_map = {}
        for space in self.spaces:
            print(space)
            space = space.replace(".json", "")
            self.maxmetric_map[space] = {}
            self.maxmetric_map[space]['width'] = self.get_maxmetric('width', space=space)
            self.maxmetric_map[space]['depth'] = self.get_maxmetric('depth', space=space)
        for space in self.spaces:
            space = space.replace(".json", "")
            self.all_accs[space] = []
            for idx in range(len(self.space_dicts[space])):
                self.all_accs[space].append(float(100.-self.space_dicts[space][idx]['test_ep_top1'][-1])/100.)
        self.unnorm_all_accs = self.all_accs # need to comment out this line.
        self.all_accs = self.all_accs

    # #################### Key Functions Begin ###################
    def get_adj_op(self, idx, space="Amoeba_in", bin_space=None):
        return self.space_adj_mats[space][str(idx)]
        
    def get_valacc(self, idx, space="Amoeba_in"):
        return self.unnorm_all_accs[space][idx]
    
    def get_numitems(self, space="Amoeba_in"):
        return len(self.space_dicts[space])

    def get_maxmetric(self, metric, space="Amoeba_in"):
        return max([self.space_dicts[space][x]['net'][metric] for x in range(len(self.space_dicts[space]))])
    
    def get_norm_w_d(self, idx, space="Amoeba_in"):
        return [self.space_dicts[space][idx]['net']['width']/self.maxmetric_map[space]['width'], \
                self.space_dicts[space][idx]['net']['depth']/self.maxmetric_map[space]['depth']]
    
    def get_zcp(self, idx, space="Amoeba_in", joint=None):
        return list([0]*13)
        
    def get_cate(self, idx, space="Amoeba_in", joint=None):
        return [0]*32
    
    def get_arch2vec(self, idx, space="Amoeba_in", joint=None):
        return [0]*32
    
    ##################### Key Functions End #####################

    def get_flops(self, idx, space="Amoeba_in"):
        return self.space_dicts[space][idx]["flops"]

    def get_params(self, idx, space="Amoeba_in"):
        return self.space_dicts[space][idx]["params"]
    

if __name__ == "__main__":
    nds = NDSin()