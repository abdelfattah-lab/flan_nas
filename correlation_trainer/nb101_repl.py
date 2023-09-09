from scipy.stats import spearmanr, kendalltau
import torch
from torch.utils.data import DataLoader
from models import FullyConnectedNN, GIN_Model, \
                    GIN_Model_NDS, GIN_ZCP_Model_NDS, \
                    GIN_Emb_Model, \
                    GIN_Emb_Model_NDS, GIN_Emb_ZCP_Model, \
                    GIN_Emb_ZCP_Model_NDS, GAT_ZCP_Model, EAGLE_M, EAGLE_M_GINADJ
from tagates_model import GIN_ZCP_Model
import argparse, sys, time, random, os
import numpy as np
from pprint import pprint
from tqdm import tqdm
from configs import configs
from utils import CustomDataset
from torch.optim.lr_scheduler import StepLR
from pprint import pprint
from torch.optim.lr_scheduler import CosineAnnealingLR
wandb = False
if wandb:
    import wandb

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
'''
If representation is adj, we can either use a GIN or MLP.

This will be indicated as "adj_gin" or "adj_mlp" respectively.

For adj_mlp, convert the ops matrix into its index and append it to flattened adjacency matrix.

'''
# execute bash command: export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

# Create argparser
parser = argparse.ArgumentParser()
####################################################### Search Space Choices #######################################################
parser.add_argument('--space', type=str, default='Amoeba')        # nb101, nb201, nb301, tb101, amoeba, darts, darts_fix-w-d, darts_lr-wd, enas, enas_fix-w-d, nasnet, pnas, pnas_fix-w-d supported
parser.add_argument('--task', type=str, default='class_scene')    # all tb101 tasks supported
parser.add_argument('--representation', type=str, default='cate') # adj_mlp, adj_gin, zcp (except nb301), cate, arch2vec, adjmlp_zcp, adjgin_zcp, adjgat_zcp supported.
parser.add_argument('--test_tagates', action='store_true')        # Currently only supports testing on NB101 networks. Easy to extend.
parser.add_argument('--loss_type', type=str, default='mse')       # mse, pwl supported
parser.add_argument('--op_emb', action='store_true')              # with or without operation embedding table.
parser.add_argument("--num_trials", type=int, default=1)          # Number of trials to run for each sample count.
# Convert adj_tagates, op_tagates, opmat_tagates, zcp_tagates, accs_tagates into parser boolean arguments
for arg in ['adj_tagates', 'op_tagates', 'opmat_tagates', 'opmat_only', 'zcp_tagates', 'accs_tagates']:
    parser.add_argument('--'+arg, action='store_true')
###################################################### Other Hyper-Parameters ######################################################
parser.add_argument('--no-norm_adj_op', action='store_false')
parser.add_argument('--gin_readout', type=str, default='mean')
parser.add_argument('--name_desc', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_hops', type=int, default=4)
parser.add_argument('--num_mlp_layers', type=int, default=3)
parser.add_argument('--num', type=int, default=8)
parser.add_argument('--zcp_gin_dim', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--test_size', type=int, default=None)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--lr_gamma', type=float, default=0.6)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eta_min', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--id', type=int, default=0)
####################################################################################################################################
args = parser.parse_args()

# Set random seeds
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
if args.seed is not None:
    seed_everything(args.seed)

device = args.device

if wandb:
    wandb.login()

if args.space=='tb101':
    run_name = args.space + "_" + args.task + "_" + args.representation
elif args.space in ['nb101', 'nb201', 'nb301']:
    run_name = args.space + "_" + args.representation
else:
    run_name = "nds_" + args.space + "_" + args.representation

if args.name_desc is not None:
    run_name += "_" + args.name_desc

if wandb:
    wandb.init(project="RankCorrelationTests", name=run_name, config=args, allow_val_change=True)

num_trials = args.num_trials
sample_tests = {# NDS
                'Amoeba': [249, 498, 1245, 2491],
                'DARTS': [250, 500, 1250, 2500],
                'DARTS_fix-w-d': [250, 500, 1250, 2500],
                'DARTS_lr-wd': [250, 500, 1250, 2500],
                'ENAS': [25, 50, 125, 250, 2500],
                'ENAS_fix-w-d': [250, 500, 1250, 2500],
                'NASNet': [242, 484, 1211, 2423],
                'PNAS': [249, 499, 1249, 2499],
                'PNAS_fix-w-d': [227, 455, 1139, 2279],
                # NASBench
                # 'nb101': [72, 364, 728, 3645, 7280],
                # 'nb101': [7280],
                # 'nb101': [72],
                'nb101': [364],
                # 'nb201': [7, 39, 78, 390, 781],
                'nb201': [7812],
                'nb301': [29, 58, 294, 589, 2948],
                # TransNASBench101
                'tb101': [40, 204, 409, 2048]}

test_tagates = args.test_tagates

if args.space == 'nb101' and test_tagates:
    print("Explicit TAGATES comparision")
    import sys
    sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
    from nb123.nas_bench_101.cell_101 import Cell101
    from nasbench import api as NB1API
    import pickle
    BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
    nb1_api = NB1API.NASBench(BASE_PATH + 'nasbench_only108_caterec.tfrecord')
    hash_to_idx = {v: idx for idx,v in enumerate(list(nb1_api.hash_iterator()))}
    with open(os.environ['PROJ_BPATH'] + "/" + "/correlation_trainer/tagates_replication/nb101_hash.txt", "rb") as fp:
        nb101_hash = pickle.load(fp)
    nb101_tagates_sample_indices = [hash_to_idx[hash_] for hash_ in nb101_hash]
    with open(os.environ['PROJ_BPATH'] + "/" + "/correlation_trainer/tagates_replication/nb101_hash_train.txt", "rb") as fp:
        nb101_train_hash = pickle.load(fp)
    nb101_train_tagates_sample_indices = [hash_to_idx[hash_] for hash_ in nb101_train_hash]
    with open(os.environ['PROJ_BPATH'] + "/correlation_trainer/tagates_replication/nasbench101_zsall_train.pkl", "rb") as fp:
        tagates_train = pickle.load(fp)
    with open(os.environ['PROJ_BPATH'] + "/correlation_trainer/tagates_replication/nasbench101_zsall_valid.pkl", "rb") as fp:
        tagates_val = pickle.load(fp)


def pwl_train(args, model, dataloader, criterion, optimizer, scheduler, test_dataloader, epoch):
    # model.train()
    model.training = True
    running_loss = 0.0
    for inputs, targets in dataloader:
        if inputs[0].shape[0] == 1 and args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
            continue
        elif inputs[0].shape[0] == 2 and args.space not in ['nb101', 'nb201', 'nb301', 'tb101']:
            continue
        #### Params for PWL Loss
        accs = targets
        max_compare_ratio = 4
        compare_threshold = 0.0
        max_grad_norm = None
        compare_margin = 0.1
        margin = [compare_margin]
        n = targets.shape[0]
        ###### 
        n_max_pairs = int(max_compare_ratio * n)
        acc_diff = np.array(accs)[:, None] - np.array(accs)
        acc_abs_difF_matrix = np.triu(np.abs(acc_diff), 1)
        ex_thresh_inds = np.where(acc_abs_difF_matrix > compare_threshold)
        ex_thresh_nums = len(ex_thresh_inds[0])
        if ex_thresh_nums > n_max_pairs:
            keep_inds = np.random.choice(np.arange(ex_thresh_nums), n_max_pairs, replace=False)
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
        if args.representation == 'adj_gin':
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0])))]
                X_adj_1, X_ops_1 = archs_1[0].to(device), archs_1[1].to(device)
                s_1 = model(X_ops_1, X_adj_1.to(torch.long)).squeeze()
                X_adj_2, X_ops_2 = archs_2[0].to(device), archs_2[1].to(device)
                s_2 = model(X_ops_2, X_adj_2.to(torch.long)).squeeze()
            else:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[0])))]
                X_adj_a_1, X_ops_a_1, X_adj_b_1, X_ops_b_1 = archs_1[0].to(device), archs_1[1].to(device), archs_1[2].to(device), archs_1[3].to(device)
                s_1 = model(X_ops_a_1, X_adj_a_1.to(torch.long), X_ops_b_1, X_adj_b_1.to(torch.long)).squeeze()
                X_adj_a_2, X_ops_a_2, X_adj_b_2, X_ops_b_2 = archs_2[0].to(device), archs_2[1].to(device), archs_2[2].to(device), archs_2[3].to(device)
                s_2 = model(X_ops_a_2, X_adj_a_2.to(torch.long), X_ops_b_2, X_adj_b_2.to(torch.long)).squeeze()
        elif args.representation == 'adjgin_zcp':
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                # import pdb; pdb.set_trace()
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[0])))]
                # import pdb; pdb.set_trace()
                X_adj_1, X_ops_1, zsp_1_, zcp_1_ = archs_1[0].to(device), archs_1[1].to(device), archs_1[2].to(device), archs_1[3].to(device)
                s_1 = model.forward(X_ops_1, X_adj_1.to(torch.long), zsp_1_, zcp_1_).squeeze()
                X_adj_2, X_ops_2, zsp_2_, zcp_2_ = archs_2[0].to(device), archs_2[1].to(device), archs_2[2].to(device), archs_2[3].to(device)
                s_2 = model.forward(X_ops_2, X_adj_2.to(torch.long), zsp_2_, zcp_2_).squeeze()
            else:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1]))),
                           torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[1]))), 
                           torch.stack(list((inputs[4][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0]))),
                           torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[0]))), 
                           torch.stack(list((inputs[4][indx] for indx in ex_thresh_inds[0])))]
                X_adj_a_1, X_ops_a_1, X_adj_b_1, X_ops_b_1, zcp_1_ = archs_1[0].to(device), archs_1[1].to(device), archs_1[2].to(device), archs_1[3].to(device), archs_1[4].to(device)
                s_1 = model(X_ops_a_1, X_adj_a_1.to(torch.long), X_ops_b_1, X_adj_b_1.to(torch.long), zcp_1_).squeeze()
                X_adj_a_2, X_ops_a_2, X_adj_b_2, X_ops_b_2, zcp_2_ = archs_2[0].to(device), archs_2[1].to(device), archs_2[2].to(device), archs_2[3].to(device), archs_2[4].to(device)
                s_2 = model(X_ops_a_2, X_adj_a_2.to(torch.long), X_ops_b_2, X_adj_b_2.to(torch.long), zcp_2_).squeeze()
        else:
            archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))),
                          torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1])))]
            archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0])))]
            X_adj_1, X_ops_1 = archs_1[0].to(device), archs_1[1].to(device)
            s_1 = model(X_ops_1, X_adj_1.to(torch.long)).squeeze()
            X_adj_2, X_ops_2 = archs_2[0].to(device), archs_2[1].to(device)
            s_2 = model(X_ops_2, X_adj_2.to(torch.long)).squeeze()
        better_lst = (acc_diff>0)[ex_thresh_inds]
        better_pm = 2 * s_1.new(np.array(better_lst, dtype=np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = s_1.new(margin)
        margin_l2 = False
        if not margin_l2:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        else:
            pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) \
                    ** 2 / np.maximum(1., margin))
        optimizer.zero_grad()
        pair_loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += pair_loss.item()
    scheduler.step()
    if epoch < epochs - 5:
        return model, running_loss / len(dataloader), 0, 0
    else:
        # model.eval()
        model.training = False
        pred_scores, true_scores = [], []
        for reprs, scores in test_dataloader:
            if args.representation == 'adj_gin':
                if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                    pred_scores.append(model(reprs[1].cuda(), reprs[0].cuda().to(torch.long)).squeeze().detach().cpu().tolist())
                else:
                    pred_scores.append(model(reprs[1].cuda(), reprs[0].cuda().to(torch.long), reprs[3].cuda(), reprs[2].cuda().to(torch.long)).squeeze().detach().cpu().tolist())
            elif args.representation == 'adjgin_zcp':
                if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                    pred_scores.append(model.forward(reprs[1].cuda(), reprs[0].cuda().to(torch.long), reprs[2].cuda(), reprs[3].cuda()).squeeze().detach().cpu().tolist())
                else:
                    pred_scores.append(model(reprs[1].cuda(), reprs[0].cuda().to(torch.long), reprs[3].cuda(), reprs[2].cuda().to(torch.long), reprs[4].cuda()).squeeze().detach().cpu().tolist())
            else:
                pred_scores.append(model(reprs.cuda()).squeeze().detach().cpu().tolist())
            true_scores.append(scores.cpu().tolist())
        try:
            pred_scores = [t for sublist in pred_scores for t in sublist]
        except:
            print("MAJOR ISSUE?")
            pred_scores = [t for sublist in pred_scores[:-1] for t in sublist] + [pred_scores[-1]]
        true_scores = [t for sublist in true_scores for t in sublist]
        return model, running_loss / len(dataloader), spearmanr(true_scores, pred_scores).correlation, kendalltau(true_scores, pred_scores).correlation



def train(args, model, dataloader, criterion, optimizer, scheduler, test_dataloader, epoch):
    # model.train()
    model.training = True
    running_loss = 0.0
    for inputs, targets in dataloader:
        if args.representation not in ['adj_gin', 'adjgin_zcp']:
            if inputs.shape[0]==1:
                continue
        if args.representation == 'adj_gin':
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                X_adj, X_ops, targets = inputs[0].to(device), inputs[1].to(device), targets.float().to(device)
                outputs = model(X_ops, X_adj.to(torch.long))
            else:
                X_adj_1, X_ops_1, X_adj_2, X_ops_2, targets = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device), targets.float().to(device)
                outputs = model(X_ops_1, X_adj_1.to(torch.long), X_ops_2, X_adj_2.to(torch.long))
        elif args.representation == 'adjgin_zcp':
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                X_adj, X_ops, zsp_, zcp_, targets = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device), targets.float().to(device)
                outputs = model.forward(X_ops, X_adj.to(torch.long), zsp_, zcp_).squeeze()
            else:
                X_adj_1, X_ops_1, X_adj_2, X_ops_2, zcp_, targets = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device), inputs[4].to(device), targets.float().to(device)
                outputs = model(X_ops_1, X_adj_1.to(torch.long), X_ops_2, X_adj_2.to(torch.long), zcp_).squeeze()
        else:
            inputs, targets = inputs.to(device), targets.float().to(device)
            outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    scheduler.step()
    if epoch < epochs - 5:
        return model, running_loss / len(dataloader), 0, 0
    else:
        # model.eval()
        model.training = False
        pred_scores, true_scores = [], []
        for reprs, scores in test_dataloader:
            if args.representation == 'adj_gin':
                if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                    pred_scores.append(model(reprs[1].cuda(), reprs[0].cuda().to(torch.long)).squeeze().detach().cpu().tolist())
                else:
                    pred_scores.append(model(reprs[1].cuda(), reprs[0].cuda().to(torch.long), reprs[3].cuda(), reprs[2].cuda().to(torch.long)).squeeze().detach().cpu().tolist())
            elif args.representation == 'adjgin_zcp':
                if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                    pred_scores.append(model.forward(reprs[1].cuda(), reprs[0].cuda().to(torch.long), reprs[2].cuda()).squeeze().detach().cpu().tolist())
                else:
                    pred_scores.append(model(reprs[1].cuda(), reprs[0].cuda().to(torch.long), reprs[3].cuda(), reprs[2].cuda().to(torch.long), reprs[4].cuda()).squeeze().detach().cpu().tolist())
            else:
                pred_scores.append(model(reprs.cuda()).squeeze().detach().cpu().tolist())
            true_scores.append(scores.cpu().tolist())
        try:
            pred_scores = [t for sublist in pred_scores for t in sublist]
        except:
            print("MAJOR ISSUE?")
            pred_scores = [t for sublist in pred_scores[:-1] for t in sublist] + [pred_scores[-1]]
        true_scores = [t for sublist in true_scores for t in sublist]
        return model, running_loss / len(dataloader), spearmanr(true_scores, pred_scores).correlation, kendalltau(true_scores, pred_scores).correlation


def test(args, model, dataloader, criterion):
    # model.eval()
    model.training = False
    running_loss = 0.0
    numitems = 0
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            if idx < 20:
                if args.representation == 'adj_gin':
                    if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                        X_adj, X_ops, targets = inputs[0].to(device), inputs[1].to(device), targets.float().to(device)
                        optimizer.zero_grad()
                        outputs = model(X_ops, X_adj.to(torch.long))
                    else:
                        X_adj_1, X_ops_1, X_adj_2, X_ops_2, targets = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device), targets.float().to(device)
                        optimizer.zero_grad()
                        outputs = model(X_ops_1, X_adj_1.to(torch.long), X_ops_2, X_adj_2.to(torch.long))
                elif args.representation == 'adjgin_zcp':
                    if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                        X_adj, X_ops, zsp_, zcp_, targets = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device), targets.float().to(device)
                        optimizer.zero_grad()
                        outputs = model.forward(X_ops, X_adj.to(torch.long), zsp_, zcp_).squeeze()
                    else:
                        X_adj_1, X_ops_1, X_adj_2, X_ops_2, zcp_, targets = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device), inputs[4].to(device), targets.float().to(device)
                        optimizer.zero_grad()
                        outputs = model(X_ops_1, X_adj_1.to(torch.long), X_ops_2, X_adj_2.to(torch.long), zcp_).squeeze()
                else:
                    inputs, targets = inputs.to(device), targets.float().to(device)
                    outputs = model(inputs)
                # loss = criterion(outputs, targets.unsqueeze(1))
                loss = criterion(outputs.squeeze(), targets)
                running_loss += loss.item()
                numitems += targets.squeeze().shape[0]
    return running_loss / numitems

epochs = args.epochs
space = args.space

sys.path.append("..")
if space in ['Amoeba', 'DARTS', 'DARTS_fix-w-d', 'DARTS_lr-wd', 'ENAS', 'ENAS_fix-w-d', 'NASNet', 'PNAS', 'PNAS_fix-w-d']:
    from nas_embedding_suite.nds_ss import NDS as EmbGenClass
elif space in ['nb101', 'nb201', 'nb301']:
    exec("from nas_embedding_suite.nb{}_ss import NASBench{} as EmbGenClass".format(space[-3:], space[-3:]))
elif space in ['tb101']:
    from nas_embedding_suite.tb101_micro_ss import TransNASBench101Micro as EmbGenClass
embedding_gen = EmbGenClass(normalize_zcp=True, log_synflow=True)

def one_hot_encode(arr, vlmax):
    # Create an array of zeros with shape (len(arr), vlmax)
    one_hot = np.zeros((len(arr), vlmax))
    
    # For each index and value in the list, set the corresponding one-hot position to 1
    for i, val in enumerate(arr):
        one_hot[i, val] = 1
        
    return one_hot

def get_dataloader(args, embedding_gen, space, sample_count, representation, mode, train_indexes=None, test_size=None):
    representations = []
    accs = []
    if space == 'nb101' and test_tagates:
        print("Sampling ONLY for TAGATES NB101 Replication")
        if mode == 'train':
            sample_indexes = random.sample(nb101_train_tagates_sample_indices, sample_count)
        else:
            sample_indexes = nb101_tagates_sample_indices
            # remaining_indexes = list(set(nb101_tagates_sample_indices) - set(train_indexes)) # sample_indexes not needed, implicitly propagated.
            # if test_size is not None:
            #     sample_indexes = random.sample(remaining_indexes, test_size)
            # else:
            #     sample_indexes = remaining_indexes
    else:
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
            if test_size is not None:
                sample_indexes = random.sample(remaining_indexes, test_size)
            else:
                sample_indexes = remaining_indexes
    dtlen = len(embedding_gen.get_adj_op(0).values())
    if dtlen == 2:
        adj_mat, op_mat = embedding_gen.get_adj_op(0).values()
        vlmax = np.asarray(op_mat).shape[1]
    if representation.__contains__("gin")==False:
        if representation == 'adj_mlp':
            for idxlx_, i in tqdm(enumerate(sample_indexes)):
                if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                    adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=args.space).values()
                    accs.append(embedding_gen.get_valacc(i, space=args.space))
                    adj_mat_norm = np.asarray(adj_mat_norm).flatten()
                    adj_mat_red = np.asarray(adj_mat_red).flatten()
                    op_mat_norm = torch.Tensor(np.asarray(op_mat_norm)).argmax(dim=1).numpy().flatten() # Careful here.
                    op_mat_red = torch.Tensor(np.asarray(op_mat_red)).argmax(dim=1).numpy().flatten() # Careful here.
                    if args.op_emb:
                        print("Operation Embedding Data-Type")
                        # Convert list to list of list?
                        print(op_mat_norm)
                    if args.norm_adj_op and not args.op_emb:
                        op_mat_norm = op_mat_norm/np.max(op_mat_norm)
                        op_mat_red = op_mat_red/np.max(op_mat_red)
                    representations.append(np.concatenate((adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red)).tolist())
                else:
                    # adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                    if mode == 'train':
                        adj_mat = tagates_train[idxlx_][0][0]
                        op_mat = one_hot_encode(tagates_train[idxlx_][0][1], vlmax)
                        # op_mat = tagates_train[idxlx_][0][1]
                    else:
                        adj_mat = tagates_val[idxlx_][0][0]
                        op_mat = one_hot_encode(tagates_val[idxlx_][0][1], vlmax)
                        # op_mat = tagates_train[idxlx_][0][1]
                    # import pdb; pdb.set_trace()
                    if space == 'tb101':
                        accs.append(embedding_gen.get_valacc(i, task=args.task))
                    else:
                        accs.append(embedding_gen.get_valacc(i))
                    adj_mat = np.asarray(adj_mat).flatten()
                    op_mat = torch.Tensor(np.asarray(op_mat)).argmax(dim=1).numpy().flatten() # Careful here.
                    if args.op_emb:
                        print("Operation Embedding Data-Type")
                        # Convert list to list of list?
                        print(op_mat)
                    if args.norm_adj_op and not args.op_emb:
                        op_mat = op_mat/np.max(op_mat)
                    representations.append(np.concatenate((adj_mat, op_mat)).tolist())
        else:
            for idxlx_, i in tqdm(enumerate(sample_indexes)):
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
        assert representation in ['adj_gin', 'adjgin_zcp'],  "Only adj_gin, adjgin_zcp is supported for GIN"
        representations = []
        if args.representation == 'adj_gin':
            for idxlx_, i in tqdm(enumerate(sample_indexes)):
                if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                    adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=args.space).values()
                    accs.append(embedding_gen.get_valacc(i, space=args.space))
                    representations.append((torch.Tensor(adj_mat_norm), torch.Tensor(op_mat_norm), torch.Tensor(adj_mat_red), torch.Tensor(op_mat_red)))
                else:
                    if mode == 'train':
                        tagates_samps = tagates_train
                    else:
                        tagates_samps = tagates_val
                    adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                    # adj_mat = tagates_samps[idxlx_][0][0]
                    # op_mat = one_hot_encode(tagates_train[idxlx_][0][1], vlmax)
                    # op_mat = one_hot_encode([0] + [xlz + 2 for xlz in tagates_samps[idxlx_][0][1]] + [1], 5)
                    # op_mat = tagates_samps[idxlx_][1]
                    ## If adding ZCPs, add them here.
                    if space == 'tb101':
                        accs.append(embedding_gen.get_valacc(i, task=args.task))
                    else:
                        accs.append(embedding_gen.get_valacc(i))
                        # accs.append(tagates_samps[idxlx_][-3])
                    representations.append((torch.Tensor(adj_mat), torch.Tensor(op_mat)))
        else:
            for idxlx_, i in tqdm(enumerate(sample_indexes)):
                if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                    adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=args.space).values()
                    zcp_ = embedding_gen.get_zcp(i, space=args.space)
                    accs.append(embedding_gen.get_valacc(i, space=args.space))
                    representations.append((torch.Tensor(adj_mat_norm), torch.Tensor(op_mat_norm), torch.Tensor(adj_mat_red), torch.Tensor(op_mat_red), torch.Tensor(zcp_)))
                else:
                    if mode == 'train':
                        tagates_samps = tagates_train
                        nb101_idx_converter = nb101_train_tagates_sample_indices
                    else:
                        tagates_samps = tagates_val
                        nb101_idx_converter = nb101_tagates_sample_indices
                    if args.adj_tagates:
                        print("Using TAGATES Adjacency Matrix")
                        adj_mat = tagates_samps[idxlx_][0][0]
                    else:
                        print("Using NASBench Adjacency Matrix")
                        adj_mat, _ = embedding_gen.get_adj_op(nb101_idx_converter[idxlx_]).values()
                    if args.op_tagates:
                        print("Using TAGATES Operation Matrix")
                        # op_mat = one_hot_encode([0] + [xlz + 2 for xlz in tagates_samps[idxlx_][0][1]] + [1], 5)
                        op_mat = tagates_samps[idxlx_][0][1]
                    else:
                        print("Using NASBench Operation Matrix")
                        _, op_mat = embedding_gen.get_adj_op(nb101_idx_converter[idxlx_]).values()
                    # import pdb; pdb.set_trace()
                    if args.opmat_tagates:
                        print("Using TAGATES OPMAT")
                        if args.opmat_only:
                            print("Using TAGATES OPMAT ONLY")
                            zsp_ = np.asarray(tagates_samps[idxlx_][1])
                        else:
                            if args.op_tagates:
                                print("Using TAGATES OPMAT CONCAT TAGATES Operation Matrix")
                            else:
                                print("Using TAGATES OPMAT CONCAT NASBench101 Operation Matrix")
                            # op_mat = np.concatenate((op_mat, np.asarray(tagates_samps[idxlx_][1])), axis=1)
                            zsp_ = np.asarray(tagates_samps[idxlx_][1])
                    if args.zcp_tagates:
                        print("Using TAGATES ZCP (7 ZCPs)")
                        zcp_ = tagates_samps[idxlx_][-1]
                    else:
                        print("Using NASBench ZCP (13 ZCPs)")
                        zcp_ = embedding_gen.get_zcp(nb101_idx_converter[idxlx_])
                    if args.op_emb:
                        # op_mat = torch.Tensor(np.array(op_mat)).argmax(dim=1)
                        op_mat = tagates_samps[idxlx_][0][1]
                        # op_mat = [0] + [xlz + 2 for xlz in tagates_samps[idxlx_][0][1]] + [1]
                        # op_mat = [0] +[xlz + 2 for xlz in tagates_samps[idxlx_][0][1]]
                    if space == 'tb101':
                        accs.append(embedding_gen.get_valacc(i, task=args.task))
                    else:
                        if args.accs_tagates:
                            print("Using TAGATES ACCS")
                            accs.append(tagates_samps[idxlx_][-3])
                        else:
                            print("Using NASBench ACCS")
                            accs.append(embedding_gen.get_valacc(nb101_idx_converter[idxlx_], normalized=False))
                            # accs.append(embedding_gen.get_valacc(nb101_idx_converter[idxlx_]))
                        
                    if args.op_emb:
                        representations.append((torch.Tensor(adj_mat), torch.LongTensor(op_mat), torch.Tensor(zsp_), torch.Tensor(zcp_)))
                    else:
                        representations.append((torch.Tensor(adj_mat), torch.Tensor(op_mat), torch.Tensor(zcp_)))

    dataset = CustomDataset(representations, accs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True if mode=='train' else False)
    return dataloader, sample_indexes

representation = args.representation
sample_counts = sample_tests[space]
samp_eff = {}

if wandb:
    config = wandb.config

across_trials = {sample_count: [] for sample_count in sample_counts}
for tr_ in range(num_trials):
    for sample_count in sample_counts:
        # config.sample_count = sample_count
        # config.update({'sample_count': sample_count}, allow_val_change=True)
        # Initialize data get_dataloader(args, space, sample_count, representation, mode, train_indexes=None):
        train_dataloader, train_indexes = get_dataloader(args, embedding_gen, space, sample_count, representation, mode='train')
        test_dataloader, test_indexes = get_dataloader(args, embedding_gen, space, sample_count=None, representation=representation, mode='test', train_indexes=train_indexes, test_size=args.test_size)

        # Initialize MLP
        if representation == 'adj_gin':
            cfg = configs[4]
            input_dim = next(iter(train_dataloader))[0][1].shape[2]
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                # model = EAGLE_M_GINADJ(num_features=input_dim, num_layers=5, num_hidden=512, dropout_ratio=0.1).to(device)
                model = GIN_Model(op_emb=args.op_emb, input_dim=input_dim, hidden_dim=args.hidden_size, latent_dim=1, readout=args.gin_readout,
                                num_hops=args.num_hops, num_mlp_layers=args.num_mlp_layers, dropout=0.3, **cfg['GAE']).to(device)
            else:
                model = GIN_Model_NDS(op_emb=args.op_emb, input_dim=input_dim, hidden_dim=args.hidden_size, latent_dim=1, readout=args.gin_readout,
                                num_hops=args.num_hops, num_mlp_layers=args.num_mlp_layers, dropout=0.3, **cfg['GAE']).to(device)
        elif representation == 'adjgin_zcp':
            cfg = configs[4]
            if args.op_emb:
                input_dim = len(list(embedding_gen.get_adj_op(0).values())[1][0])
            else:
                input_dim = next(iter(train_dataloader))[0][1].shape[2]
            print("INPUT DIM: ", input_dim)
            if args.op_emb:
                input_dim = next(iter(train_dataloader))[0][1].shape[1]
            else:
                input_dim = next(iter(train_dataloader))[0][1].shape[2]
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                # model = EAGLE_M(num_features=input_dim, num_layers=5, num_hidden=512, dropout_ratio=0.1).to(device)
                model = GIN_ZCP_Model()
                # model = GIN_ZCP_Model(op_emb=args.op_emb, input_dim=input_dim, hidden_dim=args.hidden_size, latent_dim=1, readout=args.gin_readout,
                #                 zcp_dim=next(iter(train_dataloader))[0][-1].shape[1], zcp_gin_dim=args.zcp_gin_dim,
                #                 num_hops=args.num_hops, num_mlp_layers=args.num_mlp_layers, dropout=0.3, **cfg['GAE']).to(device)
            else:
                model = GIN_ZCP_Model_NDS(op_emb=args.op_emb, input_dim=input_dim, hidden_dim=args.hidden_size, latent_dim=1, readout=args.gin_readout,
                                zcp_dim=next(iter(train_dataloader))[0][-1].shape[1], zcp_gin_dim=args.zcp_gin_dim,
                                num_hops=args.num_hops, num_mlp_layers=args.num_mlp_layers, dropout=0.3, **cfg['GAE']).to(device)
        elif representation == 'adjgat_zcp':
            cfg - configs[4]
            input_dim = next(iter(train_dataloader))[0][1].shape[2]
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                model = GAT_ZCP_Model(op_emb=args.op_emb, input_dim=input_dim, hidden_dim=args.hidden_size, latent_dim=1, readout=args.gin_readout,
                                zcp_dim=next(iter(train_dataloader))[0][-1].shape[1], zcp_gin_dim=args.zcp_gin_dim,
                                num_hops=args.num_hops, num_mlp_layers=args.num_mlp_layers, dropout=0.3, **cfg['GAE']).to(device)
            else:
                raise NotImplementedError
        else:
            representation_size = next(iter(train_dataloader))[0].shape[1]
            model = FullyConnectedNN(layer_sizes = [representation_size] + [args.hidden_size] * args.num_layers + [1]).to(device)
        # Initialize criterion and optimizer
        criterion = torch.nn.MSELoss()
        prms = list(model.mlp.parameters()) + list(model.gcns.parameters()) + list(model.b_gcns.parameters()) + list(model.fb_conversion.parameters()) + list(model.op_emb.parameters()) + list(model.param_zs_embedder.parameters())
        optimizer = torch.optim.AdamW(prms, lr=args.lr, weight_decay=args.weight_decay)
        # scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
        # Train model
        kdt_l5, spr_l5 = [], []
        for epoch in range(epochs):
            start_time = time.time()
            if args.loss_type == 'mse':
                model, mse_loss, spr, kdt = train(args, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader, epoch)
            else:
                model, mse_loss, spr, kdt = pwl_train(args, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader, epoch)
            test_loss = test(args, model, test_dataloader, criterion)
            end_time = time.time()
            if epoch > epochs - 5:
                kdt_l5.append(kdt)
                spr_l5.append(spr)
                print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {mse_loss:.4f} | Test Loss: {test_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman: {spr:.4f} | Kendall: {kdt:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {mse_loss:.4f} | Test Loss: {test_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s')
            # samp_eff[sample_count] = (max(spr_l5), max(kdt_l5))
            if wandb:
                wandb.log({"epoch": epoch + 1, "train_loss": mse_loss, "test_loss": test_loss})
        samp_eff[sample_count] = (sum(spr_l5)/len(spr_l5), sum(kdt_l5)/len(kdt_l5))
        # Generate and compare codes for all points in test_dataloader
        # samp_eff[sample_count] = (spearmanr(true_scores, pred_scores).correlation, kendalltau(true_scores, pred_scores).correlation)
        pprint(samp_eff)
        across_trials[sample_count].append(samp_eff[sample_count])
        if wandb:
            wandb.log({"sample_count": sample_count, "spearman_corr": samp_eff[sample_count][0], "kendall_corr": samp_eff[sample_count][1]})

# print average across trials for each sample count
for sample_count in sample_counts:
    print("Average KDT: ", sum([across_trials[sample_count][i][1] for i in range(len(across_trials[sample_count]))])/len(across_trials[sample_count]))
    # Print variance of KDT across tests
    print("Variance KDT: ", np.var([across_trials[sample_count][i][1] for i in range(len(across_trials[sample_count]))]))
    # print SPR
    print("Average SPR: ", sum([across_trials[sample_count][i][0] for i in range(len(across_trials[sample_count]))])/len(across_trials[sample_count]))
    # Print variance of SPR across tests
    print("Variance SPR: ", np.var([across_trials[sample_count][i][0] for i in range(len(across_trials[sample_count]))]))

sample_count = sample_counts[-1]
avkdt = str(sum([across_trials[sample_count][i][1] for i in range(len(across_trials[sample_count]))])/len(across_trials[sample_count]))
fstring = ",".join([str(z) for z in [args.adj_tagates, args.op_tagates, args.opmat_tagates, args.opmat_only, args.zcp_tagates, args.accs_tagates]]) + "," + avkdt
# append fstring to a new line in a file
with open('typetests.csv', 'a') as f:
    f.write(fstring + "\n")

import os

if not os.path.exists('correlation_results'):
    os.makedirs('correlation_results')
    
filename = f'correlation_results/{args.space}_samp_eff.csv'

header = "seed,batch_size,hidden_size,num_layers,epochs,space,representation,pwl_mse,test_tagates,key,spr,kdt"

if not os.path.isfile(filename):
    with open(filename, 'w') as f:
        f.write(header + "\n")

with open(filename, 'a') as f:
    for key in samp_eff.keys():
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % 
                (
                str(args.seed),
                str(args.batch_size),
                str(args.hidden_size),
                str(args.num_layers),
                str(args.epochs),
                str(args.space),
                str(args.representation),
                str(args.loss_type),
                str(args.test_tagates),
                str(key),
                str(samp_eff[key][0]),
                str(samp_eff[key][1]))
                )
        