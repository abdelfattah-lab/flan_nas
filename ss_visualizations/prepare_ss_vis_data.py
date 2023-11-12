import torch
import os, sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
sys.path.append("..")
from nas_embedding_suite.nds_nb3 import NDS
from nas_embedding_suite.nb101_ss import NASBench101
from nas_embedding_suite.nb201_ss import NASBench201
from nas_embedding_suite.nb301_ss import NASBench301
from nas_embedding_suite.tb101_micro_ss import TransNASBench101Micro
# initialize each ModelQuery class
nds = NDS()
nb101 = NASBench101()
nb201 = NASBench201()
nb301 = NASBench301()
tb101 = TransNASBench101Micro()

ranges = {                         # ModelQuery
    0: "nb101",                     # nb101
    423624: "nb201",                # nb201
    439249: "nb301",                # nb301
    1439249: "Amoeba",              # nds (space argument is the value)
    1444232: "PNAS_fix-w-d",        # nds (space argument is the value)
    1448791: "ENAS_fix-w-d",        # nds (space argument is the value)
    1453791: "NASNet",              # nds (space argument is the value)
    1458637: "DARTS",               # nds (space argument is the value)
    1463637: "ENAS",                # nds (space argument is the value)
    1468636: "PNAS",                # nds (space argument is the value)
    1473635: "DARTS_lr-wd",         # nds (space argument is the value)
    1478635: "DARTS_fix-w-d",       # nds (space argument is the value)
    1483635: "tb101",               # tb101
}

model_query_map = {              # ModelQuery
    "nb101"         : nb101,        # nb101
    "nb201"         : nb201,   # nb201
    "nb301"         : nb301,# nb301
    "Amoeba"        : nds,  # nds (space argument is the value)
    "PNAS_fix-w-d"  : nds,  # nds (space argument is the value)
    "ENAS_fix-w-d"  : nds,  # nds (space argument is the value)
    "NASNet"        : nds,  # nds (space argument is the value)
    "DARTS"         : nds,  # nds (space argument is the value)
    "ENAS"          : nds,  # nds (space argument is the value)
    "PNAS"          : nds,  # nds (space argument is the value)
    "DARTS_lr-wd"   : nds,  # nds (space argument is the value)
    "DARTS_fix-w-d" : nds,  # nds (space argument is the value)
    "tb101"         : tb101, # tb101
}
marker_styles = {
    "nb101": "o",  # Circle
    "nb201": "s",  # Square
    "nb301": "*",  # Star
    "tb101": "D",  # Diamond
    "Amoeba": "P",  # Plus (filled)
    "NASNet": "X",  # X (filled)
    "DARTS": "d",  # Thin diamond
    "ENAS": "h",  # Hexagon
    "PNAS": "p",  # Pentagon
    "DARTS_lr-wd": "^",  # Upward-pointing triangle
    "DARTS_fix-w-d": "v",  # Downward-pointing triangle
    "PNAS_fix-w-d": "<",  # Left-pointing triangle
    "ENAS_fix-w-d": ">",  # Right-pointing triangle
}


if True:
    def get_model_params(model_query, sampled_indices, space):
        params = []
        for idx in sampled_indices:
            try:
                params.append(model_query.get_params(idx, space=space))
            except:
                params.append(model_query.get_params(idx))
        return params
    def get_model_zcp(model_query, sampled_indices, space):
        zcps = []
        for idx in sampled_indices:
            try:
                zcps.append(model_query.get_zcp(idx, space=space))
            except:
                try:
                    zcps.append(model_query.get_zcp(idx))
                except:
                    zcps.append([0,]*13)
        return zcps
    name_map = {"nb": "NASBench-", "_fix-w-d": "$_{FixWD}$", "_lr-wd": "$_{LRWD}$", "tb": "TransNASBench-"}
    def replace_name(name, name_map):
        for key, value in name_map.items():
            name = name.replace(key, value)
        return name
data_dict1 = torch.load(os.environ["PROJ_BPATH"] + "/" + "/nas_embedding_suite/embedding_datasets/model-dim_32_search_space_all_ss-all_ss.pt")
data_dict2 = torch.load(os.environ["PROJ_BPATH"] + "/" + "/nas_embedding_suite/embedding_datasets/cate_all_ss.pt")
if True:
    included_spaces = list(ranges.values())
    def load_and_prepare_data(data_dict, ranges, included_spaces, num_std_dev=3):
        features = []
        labels = []
        param_map = []
        zcp_map = []
        class_f_l_p = {}
        for class_idx in range(len(included_spaces)):
            class_f_l_p[class_idx] = {"f": [], "l": [], "p": [], "z": []}
        range_start_keys = sorted(ranges.keys())
        if len(data_dict) > 10:
            for key, val in tqdm(data_dict.items()):
                feature_val = val["feature"]
                space_name = None
                local_index = None
                for i, range_start in enumerate(range_start_keys):
                    if key >= range_start:
                        space_name = ranges[range_start]
                        if space_name in included_spaces:
                            next_range_start = range_start_keys[i+1] if i+1 < len(range_start_keys) else None
                            if next_range_start is None or key < next_range_start:
                                local_index = key - range_start
                                break
                if local_index is not None:
                    class_idx = included_spaces.index(space_name)
                    labels.append(class_idx)
                    features.append(feature_val.tolist())
                    model_query = model_query_map[space_name]
                    params = get_model_params(model_query, [local_index], space_name)
                    zcps = get_model_zcp(model_query, [local_index], space_name)
                    zcp_map.extend(zcps)
                    param_map.extend(params)
                    class_f_l_p[class_idx]["f"].append(feature_val.tolist())
                    class_f_l_p[class_idx]["l"].append(class_idx)
                    class_f_l_p[class_idx]["p"].append(params)
                    class_f_l_p[class_idx]["z"].append(zcps)
        else:
            for key, val in enumerate(tqdm(data_dict['embeddings'])):
                feature_val = val
                space_name = None
                local_index = None
                for i, range_start in enumerate(range_start_keys):
                    if key >= range_start:
                        space_name = ranges[range_start]
                        if space_name in included_spaces:
                            next_range_start = range_start_keys[i+1] if i+1 < len(range_start_keys) else None
                            if next_range_start is None or key < next_range_start:
                                local_index = key - range_start
                                break
                if local_index is not None:
                    class_idx = included_spaces.index(space_name)
                    labels.append(class_idx)
                    features.append(feature_val.tolist())
                    model_query = model_query_map[space_name]
                    params = get_model_params(model_query, [local_index], space_name)
                    zcps = get_model_zcp(model_query, [local_index], space_name)
                    zcp_map.extend(zcps)
                    param_map.extend(params)
                    class_f_l_p[class_idx]["f"].append(feature_val.tolist())
                    class_f_l_p[class_idx]["l"].append(class_idx)
                    class_f_l_p[class_idx]["p"].append(params)
                    class_f_l_p[class_idx]["z"].append(zcps)
        return class_f_l_p
    import pickle
    class_f_l_p_1 = load_and_prepare_data(data_dict1, ranges, included_spaces)
    class_f_l_p_2 = load_and_prepare_data(data_dict2, ranges, included_spaces)
    with open('class_f_l_p_z_1.pkl', 'wb') as f:
        pickle.dump(class_f_l_p_1, f)
    with open('class_f_l_p_z_2.pkl', 'wb') as f:
        pickle.dump(class_f_l_p_2, f)