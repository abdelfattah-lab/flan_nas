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
# import pickle
import time
import dill as pickle
import sys

sys.path.append("/home/ya255/projects/nas_embedding_suite/nas_embedding_suite/")

from nb101_ss import NASBench101
from nb201_ss import NASBench201
from nb301_ss import NASBench301
from nds_ss import NDS
from tb101_micro_ss import TransNASBench101Micro
import os
import pickle

CACHE_DIR = '/scratch/ya255/emb_cache'
FILE_MAPPINGS = {
    'nb101': (CACHE_DIR + '/nb101.pkl', NASBench101),
    'nb201': (CACHE_DIR + '/nb201.pkl', NASBench201),
    # 'nb301': ('./cache/nb301.pkl', NASBench301),
    'tb101': (CACHE_DIR + '/tb101.pkl', TransNASBench101Micro),
    'nds': (CACHE_DIR + '/nds.pkl', NDS),
}

class AllSS:
    def __init__(self):
        self.ss_mapper = {"nb101": 0, "nb201": 1, "nb301": 2, "Amoeba": 3, "PNAS_fix-w-d": 4, 
                     "ENAS_fix-w-d": 5, "NASNet": 6, "DARTS": 7, "ENAS": 8, "PNAS": 9, 
                     "DARTS_lr-wd": 10, "DARTS_fix-w-d": 11, "tb101": 12}
        self._ensure_cache_exists()
        self.nb301 = NASBench301()
        self._load_classes()
        self.max_oplen = self.get_max_oplen()

    def get_adj_op(self, idx, space):
        if space in ["nb101", "nb201", "nb301", "tb101"]:
            adj_op_mat = eval("self." + space).get_adj_op(idx)
            opmat = np.asarray(adj_op_mat["module_operations"])
            # opmat will have a shape like 7 x 5
            # Convert it into 7 x max_oplen with leading zero padding
            padded_opmat = np.zeros((opmat.shape[0], self.max_oplen))
            for i in range(opmat.shape[0]):
                padded_opmat[i, -opmat.shape[1]:] = opmat[i]
            # ss_pad will have a 1 x 4 opmat will have a shape 7 x max_oplen
            # replicate ss_pad on each row of opmat to make it 7 x (max_oplen + 4)
            ss_pad = self.ss_to_binary(space) 
            final_mat = np.hstack([padded_opmat, np.tile(ss_pad, (padded_opmat.shape[0], 1))])
            new_adj_op_mat = copy.deepcopy(adj_op_mat)
            new_adj_op_mat["module_operations"] = final_mat.tolist()
        else:
            adj_op_mat = self.nds.get_adj_op(idx, space=space)
            new_adj_op_mat = copy.deepcopy(adj_op_mat)
            for matkey in ["reduce_ops", "normal_ops"]:
                ropmat = np.asarray(new_adj_op_mat[matkey])
                # opmat will have a shape like 7 x 5
                # Convert it into 7 x max_oplen with leading zero padding
                padded_ropmat = np.zeros((ropmat.shape[0], self.max_oplen))
                for i in range(ropmat.shape[0]):
                    padded_ropmat[i, -ropmat.shape[1]:] = ropmat[i]
                # ss_pad will have a 1 x 4 opmat will have a shape 7 x max_oplen
                # replicate ss_pad on each row of opmat to make it 7 x (max_oplen + 4)
                ss_pad = self.ss_to_binary(space) 
                final_mat = np.hstack([padded_ropmat, np.tile(ss_pad, (padded_ropmat.shape[0], 1))])
                new_adj_op_mat[matkey] = final_mat.tolist()
        return new_adj_op_mat
    
    def get_zcp(self, idx, space):
        if space in ["nb101", "nb201", "nb301", "tb101"]:
            zcp = eval("self." + space).get_zcp(idx)
        else:
            zcp = self.nds.get_zcp(idx, space=space)
        return zcp
    
    def get_arch2vec(self, idx, space):
        if space in ["nb101", "nb201", "nb301", "tb101"]:
            arch2vec = eval("self." + space).get_arch2vec(idx)
        else:
            arch2vec = self.nds.get_arch2vec(idx, space=space)
        return arch2vec

    def get_cate(self, idx, space):
        if space in ["nb101", "nb201", "nb301", "tb101"]:
            cate = eval("self." + space).get_cate(idx)
        else:
            cate = self.nds.get_cate(idx, space=space)
        return cate
    
    def get_valacc(self, idx, space):
        if space in ["nb101", "nb201", "nb301", "tb101"]:
            valacc = eval("self." + space).get_valacc(idx)
        else:
            valacc = self.nds.get_valacc(idx, space=space)
        return valacc

    def get_max_oplen(self):
        self.sskeys = list(self.ss_mapper.keys())
        oplens = {}
        for ssk in self.sskeys:
            if ssk in ["nb101", "nb201", "nb301", "tb101"]:
                oplens[ssk] = len(eval("self." + ssk).get_adj_op(0)["module_operations"][0])
            else:
                oplens[ssk + "_n"] = len(self.nds.get_adj_op(0, space=ssk)["normal_ops"][0])
                oplens[ssk + "_r"] = len(self.nds.get_adj_op(0, space=ssk)["reduce_ops"][0])
        self.oplens = oplens
        return max(list(oplens.values()))

    def _ensure_cache_exists(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
    def _load_classes(self):
        for key, (path, cls) in FILE_MAPPINGS.items():
            start_read_time = time.time()
            if os.path.exists(path):
                print("Loading {} from cache".format(key))
                try:
                    self._load_class_from_cache(key, path)
                except:
                    print("Loading {} from source".format(key))
                    self._load_class_from_source_and_save_to_cache(key, path, cls)
            else:
                print("Loading {} from source".format(key))
                self._load_class_from_source_and_save_to_cache(key, path, cls)
            print("Load Time: {}".format(time.time() - start_read_time))

    def _load_class_from_cache(self, key, path):
        with open(path, 'rb') as inp:
            setattr(self, key, pickle.load(inp))

    def _load_class_from_source_and_save_to_cache(self, key, path, cls):
        instance = cls()
        setattr(self, key, instance)
        with open(path, 'wb') as outp:
            pickle.dump(instance, outp, pickle.HIGHEST_PROTOCOL)

    def ss_to_binary(self, space):
        return [int(x) for x in f"{self.ss_mapper[space]:04b}"]

# all_ss = AllSS()