from scipy.stats import spearmanr, kendalltau
import torch
from torch.utils.data import DataLoader
from models import FullyConnectedNN, GIN_Model
import argparse, sys, time, random
import numpy as np
from pprint import pprint
from tqdm import tqdm
from configs import configs
from utils import CustomDataset
import os

# Create argparser
parser = argparse.ArgumentParser()
parser.add_argument('--space', type=str, default='Amoeba')
parser.add_argument('--task', type=str, default='class_scene')
parser.add_argument('--representation', type=str, default='cate')
parser.add_argument('--num_samples', type=int, default=4000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

sample_tests = {# NDS
                'Amoeba': 4980,
                'DARTS': 4997,
                'DARTS_fix-w-d': 4997,
                'DARTS_lr-wd': 4997,
                'ENAS': 4996,
                'ENAS_fix-w-d': 4997,
                'NASNet': 4834,
                'PNAS': 4995,
                'PNAS_fix-w-d': 4540,
                # NASBench
                'nb101': 5000,
                'nb201': 5000,
                'nb301': 5000,
                # TransNASBench101
                'tb101': 4090}

if args.space=='nb301' and args.representation=='zcp':
    exit(0)

num_samples = sample_tests[args.space]

device = args.device
space = args.space

sys.path.append("..")
if space in ['Amoeba', 'DARTS', 'DARTS_fix-w-d', 'DARTS_lr-wd', 'ENAS', 'ENAS_fix-w-d', 'NASNet', 'PNAS', 'PNAS_fix-w-d']:
    from nas_embedding_suite.nds_ss import NDS as EmbGenClass
elif space in ['nb101', 'nb201', 'nb301']:
    exec("from nas_embedding_suite.nb{}_ss import NASBench{} as EmbGenClass".format(space[-3:], space[-3:]))
elif space in ['tb101']:
    from nas_embedding_suite.tb101_micro_ss import TransNASBench101Micro as EmbGenClass

embedding_gen = EmbGenClass(normalize_zcp=True, log_synflow=True)

def get_dataloader(args, embedding_gen, space, sample_count, representation, mode, train_indexes=None):
    representations = []
    accs = []
    if mode == 'train':
        if args.space not in ['nb101', 'nb201', 'nb301', 'tb101']:
            sample_indexes = random.sample(range(embedding_gen.get_numitems(space)-1), sample_count)
        else:
            sample_indexes = random.sample(range(embedding_gen.get_numitems()-1), sample_count)
    else:
        if args.space not in ['nb101', 'nb201', 'nb301', 'tb101']:
            remaining_indexes = list(set(range(embedding_gen.get_numitems(space)-1)) - set(train_indexes))
        else:
            remaining_indexes = list(set(range(embedding_gen.get_numitems()-1)) - set(train_indexes))
        sample_indexes = remaining_indexes
    if representation.__contains__("gin")==False:
        if representation == 'adj_mlp':
            for i in sample_indexes:
                if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                    adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=args.space).values()
                    accs.append(embedding_gen.get_valacc(i, space=args.space))
                    adj_mat_norm = np.asarray(adj_mat_norm).flatten()
                    adj_mat_red = np.asarray(adj_mat_red).flatten()
                    op_mat_norm = torch.Tensor(np.asarray(op_mat_norm)).argmax(dim=1).numpy().flatten() # Careful here.
                    op_mat_red = torch.Tensor(np.asarray(op_mat_red)).argmax(dim=1).numpy().flatten() # Careful here.
                    representations.append(np.concatenate((adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red)).tolist())
                else:
                    adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                    if space == 'tb101':
                        accs.append(embedding_gen.get_valacc(i, task=args.task))
                    else:
                        accs.append(embedding_gen.get_valacc(i))
                    adj_mat = np.asarray(adj_mat).flatten()
                    op_mat = torch.Tensor(np.asarray(op_mat)).argmax(dim=1).numpy().flatten() # Careful here.
                    representations.append(np.concatenate((adj_mat, op_mat)).tolist())
        else:
            for i in sample_indexes:
                if space in ['nb101', 'nb201', 'nb301']:
                    exec('representations.append(embedding_gen.get_{}(i))'.format(representation))
                elif space=='tb101':
                    exec('representations.append(embedding_gen.get_{}(i, "{}"}))'.format(representation, args.task))
                else:
                    exec('representations.append(embedding_gen.get_{}(i, "{}"))'.format(representation, args.space))
                if space=='tb101':
                    accs.append(embedding_gen.get_valacc(i, task=args.task))
                elif space not in ['nb101', 'nb201', 'nb301']:
                    accs.append(embedding_gen.get_valacc(i, space=args.space))
                else:
                    accs.append(embedding_gen.get_valacc(i))
        representations = torch.stack([torch.FloatTensor(nxx) for nxx in representations])
    else:
        assert representation == 'adj_gin', "Only adj_gin is supported for GIN"
        representations = []
        for i in sample_indexes:
            if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=args.space).values()
                accs.append(embedding_gen.get_valacc(i, space=args.space))
                representations.append((torch.Tensor(adj_mat_norm), torch.Tensor(op_mat_norm), torch.Tensor(adj_mat_red), torch.Tensor(op_mat_red)))
            else:
                adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                if space == 'tb101':
                    accs.append(embedding_gen.get_valacc(i, task=args.task))
                else:
                    accs.append(embedding_gen.get_valacc(i))
                representations.append((torch.Tensor(adj_mat), torch.Tensor(op_mat)))
    dataset = CustomDataset(representations, accs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True if mode=='train' else False)
    return dataloader, sample_indexes


representation = args.representation
train_indexes = [0]
dataloader, test_indexes = get_dataloader(args, embedding_gen, space, sample_count=num_samples, representation=representation, mode='test', train_indexes=train_indexes)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming your dataloader is named 'dataloader'
inputs_list = []
targets_list = []

# Step 1: Iterate through the dataloader and flatten inputs and targets
for inputs, targets in dataloader:
    inputs_list.extend(inputs.view(inputs.size(0), -1).numpy())
    targets_list.extend(targets.numpy())

inputs_array = np.array(inputs_list)
targets_array = np.array(targets_list)
idx_choices = np.random.choice(len(inputs_array), size=num_samples, replace=False)
inputs_array = inputs_array[idx_choices]
targets_array = targets_array[idx_choices]
# Step 2: Calculate the percentiles
percentiles = [99, 97, 95, 93, 90, 85, 80, 70, 60, 0][::-1]
thresholds = np.percentile(targets_array, percentiles)

# Step 3: Assign integers based on percentiles
integer_targets = np.zeros_like(targets_array)
for i, threshold in enumerate(thresholds):
    integer_targets[targets_array >= threshold] = i

# Step 4: Perform t-SNE visualization
a = time.time()
tsne = TSNE(n_components=2, random_state=42, init='pca')
inputs_embedded = tsne.fit_transform(inputs_array)
print("Fitting time: {}".format(time.time() - a))

plt.figure(figsize=(10, 10))
for i in range(len(percentiles)):
    plt.scatter(
        inputs_embedded[integer_targets == i, 0],
        inputs_embedded[integer_targets == i, 1],
        s=4,
        label=f"Percentile {percentiles[i]}"
    )
plt.legend()
plt.title("t-SNE Visualization of {} {}".format(args.space, args.representation))
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
# If tsne_visualizations folder does not exist, create it
if not os.path.exists("./../tsne_visualizations"):
    os.mkdir("./../tsne_visualizations")
plt.savefig("./../tsne_visualizations/tsne_{}_{}.png".format(args.space.replace("_", "-"), args.representation), dpi=400)