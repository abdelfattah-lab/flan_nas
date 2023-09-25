import torch
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
if True:
    import seaborn as sns
    import matplotlib.ticker as ticker
    sns.set_palette("tab10")
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.size'] = 14  # Increase font size
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

name_map = {"nb": "NASBench-", "_fix-w-d": "$_{FixWD}$", "_lr-wd": "$_{LRWD}$", "tb": "TransNASBench-"}

def replace_name(name, name_map):
    for key, value in name_map.items():
        name = name.replace(key, value)
    return name

def load_and_prepare_data(data_dict, ranges):
    features = []
    labels = []
    if len(data_dict) > 10:
        for key, val in tqdm(data_dict.items()):
            feature_val = val["feature"]
            class_idx = None
            for r in sorted(ranges):
                if key >= r:
                    class_idx = list(ranges.values()).index(ranges[r])
            labels.append(class_idx)
            features.append(feature_val.tolist())
    else:
        for key, val in tqdm(enumerate(data_dict['embeddings'])):
            feature_val = val
            class_idx = None
            for r in sorted(ranges):
                if key >= r:
                    class_idx = list(ranges.values()).index(ranges[r])
            labels.append(class_idx)
            features.append(feature_val.tolist())
        
    features = np.array(features)
    labels = np.array(labels)
    
    # Sample 5000 features per label
    sampled_features = []
    sampled_labels = []
    for unique_label in np.unique(labels):
        indices = np.where(labels == unique_label)[0]
        sampled_indices = np.random.choice(indices, min(5000, len(indices)), replace=False)
        sampled_features.extend(features[sampled_indices])
        sampled_labels.extend(labels[sampled_indices])
    
    sampled_features = np.array(sampled_features)
    sampled_labels = np.array(sampled_labels)
    
    return sampled_features, sampled_labels


def visualize_tsne(X_2d, sampled_labels, ranges, ax, title, name_map, pl):
    target_ids = range(len(ranges))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ranges)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    handles = []
    for i, c, label in zip(target_ids, colors, ranges.values()):
        idx = sampled_labels == i
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], color=c, label=replace_name(label, name_map), s=4)  # s is the marker size
    ax.set_xlim(X_2d[:, 0].min(), X_2d[:, 0].max())
    ax.set_ylim(X_2d[:, 1].min(), X_2d[:, 1].max())
    ax.set_title(title)
    if pl:
        ax.legend(loc='upper right', framealpha=1)


# Load the first dictionary and prepare data
data_dict1 = torch.load(os.environ["PROJ_BPATH"] + "/" + "/nas_embedding_suite/embedding_datasets/model-dim_32_search_space_all_ss-all_ss.pt")
sampled_features1, sampled_labels1 = load_and_prepare_data(data_dict1, ranges)

# Load the second dictionary and prepare data
data_dict2 = torch.load(os.environ["PROJ_BPATH"] + "/" + "/nas_embedding_suite/embedding_datasets/cate_all_ss.pt")
sampled_features2, sampled_labels2 = load_and_prepare_data(data_dict2, ranges)

# Apply the T-SNE transformation
tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
X_2d1 = tsne.fit_transform(sampled_features1)
X_2d2 = tsne.fit_transform(sampled_features2)

if True:
    plt.cla()
    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(21, 7))  # Adjust as needed
    name_map = {"nb": "NASBench-", "_fix-w-d": "$_{FixWD}$", "_lr-wd": "$_{LRWD}$", "tb": "TransNASBench-"}  # Example name_map, replace with your actual mapping
    visualize_tsne(X_2d1, sampled_labels1, ranges, axs[0], 'Unified Arch2Vec Encodings', name_map, pl=False)
    visualize_tsne(X_2d2, sampled_labels2, ranges, axs[1], 'Unified CATE Encodings', name_map, pl=True)
    plt.tight_layout()
    plt.savefig("tsne_combined_5000_nums.png", dpi=500)
    plt.savefig("tsne_combined_5000_nums.pdf")
    # plt.legend(alpha=0.8)
plt.show()
