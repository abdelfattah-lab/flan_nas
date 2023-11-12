import torch
import os, sys, time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib.lines import Line2D

if True:
    import seaborn as sns
    import matplotlib.ticker as ticker
    sns.set_palette("tab10")
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.size'] = 16  # Increase font size
    # legend size
    plt.rcParams["legend.fontsize"] = 8

ranges = {                        
    0: "nb101",                   
    423624: "nb201",              
    439249: "nb301",              
    1439249: "Amoeba",            
    1444232: "PNAS_fix-w-d",      
    1448791: "ENAS_fix-w-d",      
    1453791: "NASNet",            
    1458637: "DARTS",             
    1463637: "ENAS",              
    1468636: "PNAS",              
    1473635: "DARTS_lr-wd",       
    1478635: "DARTS_fix-w-d",     
    1483635: "tb101",             
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


name_map = {"nb": "NASBench-", "_fix-w-d": "$_{FixWD}$", "_lr-wd": "$_{LRWD}$", "tb": "TransNASBench-"}
def replace_name(name, name_map):
    for key, value in name_map.items():
        name = name.replace(key, value)
    return name


with open('class_f_l_p_z_1.pkl', 'rb') as f:
    class_f_l_p_1_og = pickle.load(f)
with open('class_f_l_p_z_2.pkl', 'rb') as f:
    class_f_l_p_2_og = pickle.load(f)
if True:
    numq = 5000
    included_spaces = ['nb101', 'nb201', 'DARTS', 'PNAS', "ENAS_fix-w-d", 'tb101']
    included_space_idx = [list(ranges.values()).index(s) for s in included_spaces]
    class_f_l_p_1 = {k: v for k, v in class_f_l_p_1_og.items() if k in included_space_idx}
    class_f_l_p_2 = {k: v for k, v in class_f_l_p_2_og.items() if k in included_space_idx}
    sampled_features1, sampled_labels1, zcp1, params1 = [], [], [], []
    sampled_features2, sampled_labels2, zcp2, params2 = [], [], [], []
    for class_idx in class_f_l_p_1:
        indices = np.arange(len(class_f_l_p_1[class_idx]["f"]))
        sampled_indices = np.random.choice(indices, min(numq, len(indices)), replace=False)
        sampled_features1.extend(np.array(class_f_l_p_1[class_idx]["f"])[sampled_indices])
        sampled_labels1.extend(np.array(class_f_l_p_1[class_idx]["l"])[sampled_indices])
        sampled_params1 = np.array(class_f_l_p_1[class_idx]["p"])[sampled_indices]
        sampled_zcp1 = np.array(class_f_l_p_1[class_idx]["z"])[sampled_indices]
        zcp1.extend(sampled_zcp1)
        sampled_params1 = [p[0] for p in sampled_params1]
        params1.extend(sampled_params1)
    for class_idx in class_f_l_p_2:
        indices = np.arange(len(class_f_l_p_2[class_idx]["f"]))
        sampled_indices = np.random.choice(indices, min(numq, len(indices)), replace=False)
        sampled_features2.extend(np.array(class_f_l_p_2[class_idx]["f"])[sampled_indices])
        sampled_labels2.extend(np.array(class_f_l_p_2[class_idx]["l"])[sampled_indices])
        sampled_params2 = np.array(class_f_l_p_2[class_idx]["p"])[sampled_indices]
        sampled_params2 = [p[0] for p in sampled_params2]
        params2.extend(sampled_params2)
    sampled_features1 = np.array(sampled_features1)
    sampled_labels1 = np.array(sampled_labels1)
    params1 = np.array(params1)
    zcp1 = np.array(zcp1).squeeze()
    sampled_features2 = np.array(sampled_features2)
    sampled_labels2 = np.array(sampled_labels2)
    params2 = np.array(params2)

## TSNE Calculation
if True:
    print("Start TSNE")
    tsne_starT_tm = time.time()
    tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
    X_2d1 = tsne.fit_transform(sampled_features1)
    X_2d2 = tsne.fit_transform(sampled_features2)
    X_2d3 = tsne.fit_transform(zcp1)
    print("TSNE Time Taken:", time.time() - tsne_starT_tm)

## TSNE Visualiztion
if True:
    global_legend_dict = {}
    def visualize_tsne(X_2d, sampled_labels, ranges, ax, title, name_map, param_sizes, marker_styles, global_legend_dict):
        target_ids = range(len(ranges))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        for i, label in zip(target_ids, list(ranges.values())):
            if label in included_spaces:
                idx = sampled_labels == i
                std_param_sizes = (np.array(param_sizes) - np.mean(param_sizes)) / np.std(param_sizes)
                marker_size = (std_param_sizes[idx] - np.min(std_param_sizes)) / (np.max(std_param_sizes) - np.min(std_param_sizes)) * 200
                scatter = ax.scatter(X_2d[idx, 0], X_2d[idx, 1], label=replace_name(label, name_map), s=marker_size, marker=marker_styles[label], alpha=0.2)
                color_tuple = tuple(scatter.get_facecolor()[0])  # Convert to tuple
                global_legend_dict[replace_name(label, name_map)] = (color_tuple, marker_styles[label])
        ax.set_xlim(X_2d[:, 0].min(), X_2d[:, 0].max())
        ax.set_ylim(X_2d[:, 1].min(), X_2d[:, 1].max())
        ax.text(0.98, .05, f"{title}", fontsize=18, horizontalalignment='right', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'), transform=ax.transAxes)
    plt.cla()
    plt.clf()
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    visualize_tsne(X_2d1, sampled_labels1, ranges, axs[0], 'Arch2Vec', name_map, params1, marker_styles, global_legend_dict)
    visualize_tsne(X_2d2, sampled_labels2, ranges, axs[1], 'CATE', name_map, params2, marker_styles, global_legend_dict)
    visualize_tsne(X_2d3, sampled_labels1, ranges, axs[2], 'ZCP', name_map, params1, marker_styles, global_legend_dict)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=style, color='w', label=label, markerfacecolor=color, markersize=10)
                    for label, (color, style) in global_legend_dict.items()]
    lgnd = fig.legend(handles=legend_elements, loc='upper center', ncol=len(included_spaces), bbox_to_anchor=(0.5, 1.02), fontsize=16, framealpha=1)
    for lh in lgnd.legendHandles: 
        lh._legmarker.set_alpha(1)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("tsne_combined_5000_nums_2.png", dpi=500)
    plt.savefig("tsne_combined_5000_nums_2.pdf")
    plt.cla()
    plt.clf()
    import scipy.stats as stats
    def calculate_correlation(X_2d, params):
        log_params = np.log(params + 1)  # Log-transform parameter counts
        correlation_x = stats.pearsonr(X_2d[:, 0], log_params)[0]
        correlation_y = stats.pearsonr(X_2d[:, 1], log_params)[0]
        return correlation_x, correlation_y
    corr_x1, corr_y1 = calculate_correlation(X_2d1, params1)
    corr_x2, corr_y2 = calculate_correlation(X_2d2, params2)
    corr_x2, corr_y2 = calculate_correlation(X_2d3, params1)
    print("Arch2Vec Correlation with X:", corr_x1, "and Y:", corr_y1)
    print("CATE Correlation with X:", corr_x2, "and Y:", corr_y2)
    print("ZCP Correlation with X:", corr_x2, "and Y:", corr_y2)
    plt.cla()
    plt.clf()
    import umap
    def generate_umap(X, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
        return reducer.fit_transform(X)
    umap_stime = time.time()
    X_umap1 = generate_umap(sampled_features1)
    X_umap2 = generate_umap(sampled_features2)
    global_legend_dict = {}
    print("UMAP Time Taken:", time.time() - umap_stime)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    visualize_tsne(X_umap1, sampled_labels1, ranges, axs[0], 'Arch2Vec UMAP', name_map, params1, marker_styles, global_legend_dict)
    visualize_tsne(X_umap2, sampled_labels2, ranges, axs[1], 'CATE UMAP', name_map, params2, marker_styles, global_legend_dict)
    visualize_tsne(zcp1, sampled_labels1, ranges, axs[2], 'ZCP UMAP', name_map, params1, marker_styles, global_legend_dict)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    lgnd = fig.legend(handles, labels, loc='upper center', ncol=len(included_spaces), bbox_to_anchor=(0.5, 1), fontsize=18, framealpha=1)
    plt.savefig("umap_combined_5000_nums.png", dpi=500)
    plt.savefig("umap_combined_5000_nums.pdf")
    plt.cla()
    plt.clf()
    legend_elements = [Line2D([0], [0], marker=style, color='w', label=label, markerfacecolor=color, markersize=10)
                    for label, (color, style) in global_legend_dict.items()]


# ## Correlation calculation
# if True:

# ## UMAP Generation
# if True: