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

# 1. Load the dictionary
data_dict = torch.load(
    os.environ["PROJ_BPATH"]
    + "/"
    + "/nas_embedding_suite/embedding_datasets/cate_all_ss.pt"
)
# 2. Prepare your data and labels for the T-SNE
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

features = []
labels = []

# for key, val in data_dict.items():
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

# 3. Apply the T-SNE transformation
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(sampled_features)

# 4. Visualize the result
target_ids = range(len(ranges))
colors = plt.cm.rainbow(np.linspace(0, 1, len(ranges)))

fig, ax = plt.subplots(figsize=(10, 10))  # Adjust as needed


for i, c, label in zip(target_ids, colors, ranges.values()):
    idx = sampled_labels == i
    for x, y in zip(X_2d[idx, 0], X_2d[idx, 1]):
        plt.text(x, y, i, fontsize=4, ha="center", va="center", color=c)
plt.xlim(X_2d[:, 0].min(), X_2d[:, 0].max())
plt.ylim(X_2d[:, 1].min(), X_2d[:, 1].max())

plt.legend(
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10)
        for c in colors
    ],
    labels=list(ranges.values()),
)
plt.subplots_adjust(right=0.8)  # Adjust as needed

plt.savefig("tsne_cate_5000_nums.png", dpi=500)
plt.savefig("tsne_cate_5000_nums.pdf")
