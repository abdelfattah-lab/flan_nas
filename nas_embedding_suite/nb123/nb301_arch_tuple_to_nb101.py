import json
import random
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
from nb123.darts.cnn.genotypes import Genotype
from nb123.darts.cnn.model import NetworkCIFAR as Network
from thop import profile
from collections import namedtuple
from nasbench.lib import graph_util

OPS = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
    ]
NUM_VERTICES = 4
INPUT_1 = 'c_k-2'
INPUT_2 = 'c_k-1'

def transform_geno_to_nas101_format(geno):
    geno_set = {0: 'c_{k-2}', 1: 'c_{k-1}', 2: 0, 3: 1, 4: 2, 5: 3}
    geno = [ (item[0], geno_set[item[1]]) for item in geno]
    adj = np.zeros([15, 15])
    ops = np.zeros([15, 11])
    # fixed adj indices (hard rule for a valid graph)
    adj[2:4, 4] = 1 # A,B -> 0
    adj[5:7, 7] = 1 # C,D -> 1
    adj[8:10, 10] = 1 # E,F -> 2
    adj[11:13, 13] = 1 # G,H -> 3
    adj[4, -1] = 1 # 0 -> c_k
    adj[7, -1] = 1 # 1 -> c_k
    adj[10, -1] = 1 # 2 -> c_k
    adj[13, -1] = 1 # 3 -> c_k

    # transform geno adj to nas101 format
    adj_set = {'c_{k-2}': 0, 'c_{k-1}': 1, 0: 4, 1: 7, 2: 10, 3: 13}
    adj[adj_set[geno[1][1]]][2] = 1 # connection between x and A
    adj[adj_set[geno[0][1]]][3] = 1 # connection between x and B
    adj[adj_set[geno[3][1]]][5] = 1 # connection between x and C
    adj[adj_set[geno[2][1]]][6] = 1 # connection between x and D
    adj[adj_set[geno[5][1]]][8] = 1 # connection between x and E
    adj[adj_set[geno[4][1]]][9] = 1 # connection between x and F
    adj[adj_set[geno[7][1]]][11] = 1 # connection between x and G
    adj[adj_set[geno[6][1]]][12] = 1 # connection between x and H

    # fixed ops indices (as a valid graph)
    ops[0, 0] = 1 # c_{k-2}
    ops[1, 1] = 1 # c_{k-1}
    ops[4, -2] = 1 # intermediate node 0: sum
    ops[7, -2] = 1 # intermediate node 1: sum
    ops[10, -2] = 1 # intermediate node 2: sum
    ops[13, -2] = 1 # intermediate node 3: sum
    ops[-1, -1] = 1 # c_k

    # transform geno ops to nas101 format
    ops_set = {'c_k-2': 0, 'c_k-1': 1, 'max_pool_3x3': 2, 'avg_pool_3x3': 3, 'skip_connect': 4,
               'sep_conv_3x3': 5, 'sep_conv_5x5': 6, 'dil_conv_3x3': 7, 'dil_conv_5x5': 8, 'sum': 9, 'output': 10}
    ops[2][ops_set[geno[1][0]]] = 1 # A
    ops[3][ops_set[geno[0][0]]] = 1 # B
    ops[5][ops_set[geno[3][0]]] = 1 # C
    ops[6][ops_set[geno[2][0]]] = 1 # D
    ops[8][ops_set[geno[5][0]]] = 1 # E
    ops[9][ops_set[geno[4][0]]] = 1 # F
    ops[11][ops_set[geno[7][0]]] = 1 # G
    ops[12][ops_set[geno[6][0]]] = 1 # H
    return adj, ops

def transform_nas101_format_to_geno(adj, ops):
    """0: c_{k-2}, 1: c_{k-1}, 2: node_0 in row4, 3: node_1 in row7, 4: node_2 in row10, 5: node_3 in row13"""
    adj_set_reverse = {0: 0, 1: 1, 4: 2, 7: 3, 10: 4, 13: 5}
    ops_set_reverse = {0: 'c_k-2', 1: 'c_k-1', 2: 'max_pool_3x3', 3: 'avg_pool_3x3', 4: 'skip_connect',
                       5: 'sep_conv_3x3', 6: 'sep_conv_5x5', 7: 'dil_conv_3x3', 8: 'dil_conv_5x5', 9: 'sum', 10: 'output'}
    geno = []
    geno.append((ops_set_reverse[int(np.where(ops[3,:]==1)[0])], adj_set_reverse[int(np.where(adj[:,3]==1)[0])])) #B
    geno.append((ops_set_reverse[int(np.where(ops[2,:]==1)[0])], adj_set_reverse[int(np.where(adj[:,2]==1)[0])])) #A
    geno.append((ops_set_reverse[int(np.where(ops[6,:]==1)[0])], adj_set_reverse[int(np.where(adj[:,6]==1)[0])])) #D
    geno.append((ops_set_reverse[int(np.where(ops[5,:]==1)[0])], adj_set_reverse[int(np.where(adj[:,5]==1)[0])])) #C
    geno.append((ops_set_reverse[int(np.where(ops[9,:]==1)[0])], adj_set_reverse[int(np.where(adj[:,9]==1)[0])])) #F
    geno.append((ops_set_reverse[int(np.where(ops[8,:]==1)[0])], adj_set_reverse[int(np.where(adj[:,8]==1)[0])])) #E
    geno.append((ops_set_reverse[int(np.where(ops[12,:]==1)[0])], adj_set_reverse[int(np.where(adj[:,12]==1)[0])])) #H
    geno.append((ops_set_reverse[int(np.where(ops[11,:]==1)[0])], adj_set_reverse[int(np.where(adj[:,11]==1)[0])])) #G
    return geno

def sample_arch():
    num_ops = len(OPS)
    normal = []
    for i in range(NUM_VERTICES):
        ops_normal = np.random.choice(range(num_ops), NUM_VERTICES)
        nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
        normal.extend([(str(nodes_in_normal[0]), OPS[ops_normal[0]]),
                       (str(nodes_in_normal[1]), OPS[ops_normal[1]])])
    return normal

def build_mat_encoding(normal):
    normal_cell = [(item[1], int(item[0])) for item in tuple(normal)]
    adj_nas101_format, ops_nas101_format = transform_geno_to_nas101_format(normal_cell)
    return {'module_adjacency': adj_nas101_format.astype(int).tolist(), 'module_operations': ops_nas101_format.astype(int).tolist()}

def convert_arch_tuple_to_idx(arch):
    return build_mat_encoding(arch)