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
from nb123.nas_bench_301.cell_301 import Cell301
from nb123.nb301_arch_tuple_to_nb101 import convert_arch_tuple_to_idx
import nasbench301 as nb3_api

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'

class NASBench301:
    def __init__(self, use_nb3_performance_model=False, zcp_dict=False, normalize_zcp=True, log_synflow=True,
                 path=None, zcp=False, embedding_list = [ 'adj', 
                                                        'adj_op',
                                                        'paths', 
                                                        'path_indices', 
                                                        'genotypes', 
                                                        'cate', 
                                                        'zcp', 
                                                        'arch2vec',
                                                        'valacc']):
        if path==None:
            path = ''
        print("Loading Files for NASBench301...")
        a = time.time()
        self.noisy_nb3 = False

        self.use_nb3_performance_model = use_nb3_performance_model
        self.ensemble_dir_performance = BASE_PATH + "nb_models_0.9/xgb_v0.9"
        self.cate_nb301 = torch.load(BASE_PATH + "cate_embeddings/cate_nasbench301.pt")
        self.nb301_proxy_cate = None
        self.arch2vec_nb301 = torch.load(BASE_PATH + "arch2vec_embeddings/arch2vec-model-dim_32_search_space_nasbench301-nasbench301.pt")
        

        self.zcp_nb301 = json.load(open(BASE_PATH + "zc_nasbench301.json", "r"))
        self.unnorm_zcp_nb301 = json.load(open(BASE_PATH + "zc_nasbench301.json", "r"))
        self.zcp_nb301_valacc = json.load(open(BASE_PATH + "zc_nasbench301.json", "r"))
        self.zcp_nb301_valacc = {k: v['val_accuracy'] for k,v in self.zcp_nb301_valacc['cifar10'].items()}
        valacc_frame = pd.DataFrame(self.zcp_nb301_valacc, index=[0]).T
        self.valacc_frame = valacc_frame
        self.zcp_unnorm_nb301_valacc = pd.DataFrame(valacc_frame, columns=valacc_frame.columns, index=valacc_frame.index).to_dict()[0]
        min_max_scaler = preprocessing.MinMaxScaler()
        self.normalize_zcp = normalize_zcp
        self.normalize_and_process_zcp(normalize_zcp, log_synflow)

        self.nb3_api = nb3_api
        if self.use_nb3_performance_model:
            self.performance_model = nb3_api.load_ensemble(self.ensemble_dir_performance)

        print("Loaded files in: ", time.time() - a, " seconds")

        
        self.op_dict = {
            0: 'max_pool_3x3',
            1: 'avg_pool_3x3',
            2: 'skip_connect',
            3: 'sep_conv_3x3',
            4: 'sep_conv_5x5',
            5: 'dil_conv_3x3',
            6: 'dil_conv_5x5'
            }
        
        self.op_dict_rev = {v: k for k, v in self.op_dict.items()}

                
    def min_max_scaling(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def log_transform(self, data):
        return np.log1p(data)

    def standard_scaling(self, data):
        return (data - np.mean(data)) / np.std(data)

    def normalize_and_process_zcp(self, normalize_zcp, log_synflow):
        if normalize_zcp:
            print("Normalizing ZCP dict")
            self.norm_zcp = pd.DataFrame({k0: {k1: v1["score"] for k1, v1 in v0.items() if v1.__class__() == {}} 
                                          for k0, v0 in self.zcp_nb301['cifar10'].items()}).T

            # Add normalization code here
            self.norm_zcp['epe_nas'] = self.min_max_scaling(self.norm_zcp['epe_nas'])
            self.norm_zcp['fisher'] = self.min_max_scaling(self.log_transform(self.norm_zcp['fisher']))
            self.norm_zcp['flops'] = self.min_max_scaling(self.log_transform(self.norm_zcp['flops']))
            self.norm_zcp['grad_norm'] = self.min_max_scaling(self.log_transform(self.norm_zcp['grad_norm']))
            self.norm_zcp['grasp'] = self.standard_scaling(self.norm_zcp['grasp'])
            self.norm_zcp['jacov'] = self.min_max_scaling(self.norm_zcp['jacov'])
            self.norm_zcp['l2_norm'] = self.min_max_scaling(self.norm_zcp['l2_norm'])
            self.norm_zcp['nwot'] = self.min_max_scaling(self.norm_zcp['nwot'])
            self.norm_zcp['params'] = self.min_max_scaling(self.log_transform(self.norm_zcp['params']))
            self.norm_zcp['plain'] = self.min_max_scaling(self.norm_zcp['plain'])
            self.norm_zcp['snip'] = self.min_max_scaling(self.log_transform(self.norm_zcp['snip']))
            if log_synflow:
                self.norm_zcp['synflow'] = self.min_max_scaling(self.log_transform(self.norm_zcp['synflow']))
            else:
                self.norm_zcp['synflow'] = self.min_max_scaling(self.norm_zcp['synflow'])
            self.norm_zcp['zen'] = self.min_max_scaling(self.norm_zcp['zen'])
            self.zcp_nb301 = {'cifar10': self.norm_zcp.T.to_dict()}

    #################### Key Functions Begin ###################
    def get_adjmlp_zcp(self, idx):
        adj_mat, op_mat = self.get_adj_op(idx)
        adj_mat = np.asarray(adj_mat).flatten()
        op_mat = torch.Tensor(np.asarray(op_mat)).argmax(dim=1).numpy().flatten()
        op_mat = op_mat/np.max(op_mat)
        return np.concatenate([adj_mat, op_mat, np.asarray(self.get_zcp(idx))]).tolist()

    def get_adj_op(self, idx, space=None, bin_space=None):      
        cate_nb301_arch = eval(list(self.zcp_nb301['cifar10'].keys())[idx])
        arch_norm = {'arch': [(int(x[0]), x[1]) for x in cate_nb301_arch[0]]}
        arch_red = {'arch': [(int(x[0]), x[1]) for x in cate_nb301_arch[1]]}
        arch_norm = convert_arch_tuple_to_idx([(str(x[0]), self.op_dict[x[1]]) for x in arch_norm['arch']])
        arch_red = convert_arch_tuple_to_idx([(str(x[0]), self.op_dict[x[1]]) for x in arch_red['arch']])
        adj_dict = {}
        adj_dict['normal_adj'] = arch_norm['module_adjacency']
        adj_dict['normal_ops'] = arch_norm['module_operations']
        adj_dict['reduce_adj'] = arch_red['module_adjacency']
        adj_dict['reduce_ops'] = arch_red['module_operations']
        return adj_dict
        # {'normal_adj': [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 'normal_ops': [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], 'reduce_adj': [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 'reduce_ops': [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]}

        # cate_nb301_arch = self.cate_nb301['genotypes'][idx]
        # arch_desc = {'arch': ([(int(x[0]), self.op_dict_rev[x[1]]) for x in cate_nb301_arch],[(int(x[0]), self.op_dict_rev[x[1]]) for x in cate_nb301_arch])}
        # return convert_arch_tuple_to_idx([(str(x[0]), self.op_dict[x[1]]) for x in arch_desc['arch'][0]])
    
    def get_arch2vec(self, idx, joint=None, space=None):
        return self.arch2vec_nb301[idx]['feature'].tolist()
    
    def get_cate(self, idx, joint=None, space=None):
        return self.cate_nb301['embeddings'][idx].tolist()
    
    def get_zcp(self, idx, joint=None, space=None):
        # import pdb; pdb.set_trace()
        return self.zcp_nb301['cifar10'][list(self.zcp_nb301['cifar10'].keys())[idx]]
    
    def get_valacc(self, idx, space=None, use_nb3_performance_model=False):
        return self.valacc_frame.iloc[idx].item()
        
    def get_norm_w_d(self, idx, space=None):
        return [0, 0]
    
    def get_numitems(self, space=None):
        return 10000
    ##################### Key Functions End #####################
    
    def get_genotype(self, idx):
        return self.index_to_embedding(idx)['genotypes']
    
    def get_params(self, idx):
        if self.nb301_proxy_cate is None:
            with open(BASE_PATH + '/nasbench301_proxy.json', 'r') as f: # load 
                self.nb301_proxy_cate = json.load(f)
        return self.nb301_proxy_cate[str(idx)]["params"]

if __name__=='__main__':
    nb301 = NASBench301(path=os.environ['PROJ_BPATH'] + "/", use_nb3_performance_model=True)
    # for each item, get the adj_op
    norm_adj_op = {}
    red_adj_op = {}
    for i in tqdm(range(len(nb301.zcp_nb301['cifar10'].keys()))):
        adj_op = nb301.get_adj_op(i)
        red_adj_op[i] = {}
        norm_adj_op[i] = {}
        red_adj_op[i]['test_accuracy'] = nb301.get_valacc(i)
        norm_adj_op[i]['test_accuracy'] = nb301.get_valacc(i)
        red_adj_op[i]['validation_accuracy'] = nb301.get_valacc(i)
        norm_adj_op[i]['validation_accuracy'] = nb301.get_valacc(i)
        red_adj_op[i]['validation_accuracy_avg'] = nb301.get_valacc(i)
        norm_adj_op[i]['validation_accuracy_avg'] = nb301.get_valacc(i)
        norm_adj_op[i]['module_adjacency'] = adj_op['normal_adj']
        norm_adj_op[i]['module_operations'] = adj_op['normal_ops']
        red_adj_op[i]['module_adjacency'] = adj_op['reduce_adj']
        red_adj_op[i]['module_operations'] = adj_op['reduce_ops']
        red_adj_op[i]['training_time'] = 0
        norm_adj_op[i]['training_time'] = 0
    # Now, save norm_adj_op as nb301a_arch2vec.json
    with open(BASE_PATH + '/nb301a_arch2vec.json', 'w') as fp:
        json.dump(norm_adj_op, fp)
    # Now, save red_adj_op as nb301b_arch2vec.json
    with open(BASE_PATH + '/nb301b_arch2vec.json', 'w') as fp:
        json.dump(red_adj_op, fp)
        
        


    # import pdb; pdb.set_trace()
    # print(nb301.get_adj_op(0))
    # idx_to_arch = {}
    # for i in tqdm(range(len(nb301.zcp_nb301['cifar10'].keys()))):
    #     idx_to_arch[i] = nb301.get_adj_op(i)
    # # Write idx_to_arch to a json file called 'nb301.json'
    # with open(BASE_PATH + '/nds_adj_encoding/nb301.json', 'w') as fp:
    #     json.dump(idx_to_arch, fp)
    # zcp_valacc_dict = {}
    # for i in tqdm(range(len(nb301.zcp_nb301['cifar10'].keys()))):
    #     zcps = nb301.get_zcp(i).values()
    #     valacc = nb301.get_valacc(i)
    #     zcps = list(zcps)
    #     zcps.insert(0, valacc)
    #     zcp_valacc_dict[i] = zcps

    # if not os.path.exists(BASE_PATH + '/nds_zcps/nb301.csv'):
    #     with open(BASE_PATH + '/nds_zcps/nb301.csv', 'w') as fp:
    #         fp.write(',val_accuracy,' + ','.join(nb301.get_zcp(0).keys()) + '\n')
    # with open(BASE_PATH + '/nds_zcps/nb301.csv', 'a') as fp:
    #     for key, value in zcp_valacc_dict.items():
    #         fp.write(str(key) + ',' + ','.join(str(x) for x in value) + '\n')
    
    # # Also, save a .json file where each index is an element in a list. Each element is a dict with keys 'val_accuracy' and 'flops' and 'params'
    # zcp_valacc_flop_param_dict = {}
    # for i in tqdm(range(len(nb301.zcp_nb301['cifar10'].keys()))):
    #     zcps = nb301.get_zcp(i).values()
    #     valacc = nb301.get_valacc(i)
    #     flops = nb301.unnorm_zcp_nb301['cifar10'][list(nb301.unnorm_zcp_nb301['cifar10'].keys())[i]]['flops']
    #     params = nb301.unnorm_zcp_nb301['cifar10'][list(nb301.unnorm_zcp_nb301['cifar10'].keys())[i]]['params']
    #     zcps = {"test_ep_top1": valacc, "flops": flops, "params": params}
    #     zcp_valacc_flop_param_dict[i] = zcps
    
    # with open(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/NDS/nds_data/' + '/nb301.json', 'w') as fp:
    #     json.dump(zcp_valacc_flop_param_dict, fp)