import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from utils import preprocessing, normalize_adj
import time
from mlp import MLP

from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool


class FullyConnectedNN(nn.Module):
    def __init__(self, layer_sizes):
        super(FullyConnectedNN, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))  # Add batch normalization
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GIN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout, dropout, **kwargs):
        super(GIN_Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(self.num_layers - 1):
            mlp_input_dim = input_dim if layer == 0 else hidden_dim
            self.mlps.append(MLP(num_mlp_layers, mlp_input_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.fc1 = nn.Linear(self.hidden_dim, 3*self.latent_dim)
        self.fc2 = nn.Linear(3*self.latent_dim, self.latent_dim)

    def _encoder(self, ops, adj):
        batch_size, node_num, _ = ops.shape
        x = ops
        
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        
        mu = self.fc2(F.relu(self.fc1(x)))
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.latent_dim))
        
        return mu
    
    def forward(self, ops, adj):
        return self._encoder(ops, adj)

class GIN_Model_NDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout, dropout, **kwargs):
        super(GIN_Model_NDS, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        
        self.eps_1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.eps_2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        
        self.mlps_1 = self._build_mlp_layers(num_mlp_layers)
        self.mlps_2 = self._build_mlp_layers(num_mlp_layers)
        
        self.batch_norms_1 = self._build_batch_norm_layers()
        self.batch_norms_2 = self._build_batch_norm_layers()
        
        self.fc1_1 = nn.Linear(self.hidden_dim, 64)
        self.fc1_2 = nn.Linear(self.hidden_dim, 64)
        
        self.fc_comb_a = nn.Linear(128, 128)
        self.fc_comb_b = nn.Linear(128, 3*self.latent_dim)
        self.fc_oz = nn.Linear(3*self.latent_dim, self.latent_dim)

    def _build_mlp_layers(self, num_mlp_layers):
        mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            input_dim = self.input_dim if layer == 0 else self.hidden_dim
            mlps.append(MLP(num_mlp_layers, input_dim, self.hidden_dim, self.hidden_dim))
        return mlps

    def _build_batch_norm_layers(self):
        return torch.nn.ModuleList([nn.BatchNorm1d(self.hidden_dim) for _ in range(self.num_layers - 1)])

    def _encoder_helper(self, ops, adj, eps, mlps, batch_norms, fc):
        batch_size, node_num, _ = ops.shape
        x = ops
        
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(batch_norms[l](mlps[l](agg)).view(batch_size, node_num, -1))
        
        return fc(x)

    def _encoder(self, ops1, adj1, ops2, adj2):
        mu_1 = self._encoder_helper(ops1, adj1, self.eps_1, self.mlps_1, self.batch_norms_1, self.fc1_1)
        mu_2 = self._encoder_helper(ops2, adj2, self.eps_2, self.mlps_2, self.batch_norms_2, self.fc1_2)
        
        mu = torch.cat((mu_1, mu_2), dim=2)
        mu = F.relu(self.fc_comb_a(mu))
        mu = self.fc_comb_b(mu)
        mu = self.fc_oz(F.relu(mu))
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.latent_dim))
        
        return mu
    
    def forward(self, ops1, adj1, ops2, adj2):
        return self._encoder(ops1, adj1, ops2, adj2)



class Reconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        return loss


class VAEReconstructed_Loss(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu, logvar):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        loss_ops = self.loss_ops(ops_recon, ops)
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * loss_ops + self.w_adj * loss_adj
        KLD = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return loss + KLD


class WeightedLoss(nn.MSELoss):
    def __init__(self, weight=None):
        super(WeightedLoss, self).__init__()
        self.weight = weight


    def forward(self, inputs, targets):
        res = (torch.exp(inputs)-1.0) * F.mse_loss(inputs, targets, size_average=False)
        return torch.mean(res, dim=0) / (self.weight - 1)



class LinearModel(nn.Module):
    def __init__(self, input_dim, hid_dim, activation=F.relu):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)
        self.activation = activation

    def forward(self, x):
        h = self.activation(self.fc1(x))
        y = torch.sigmoid(self.fc2(h))
        return y

class DecoderNN(object):
    def __init__(self, model, ops, adj, cfg):
        print('Initializing NN decoder')
        t_s = time.time()
        self.model = model
        self.ops = ops
        self.adj = adj
        self.cfg = cfg
        with torch.no_grad():
            adj_prep, ops_prep, _ = preprocessing(self.adj, self.ops, **self.cfg['prep'])
            self.embedding = self.model.encoder(ops_prep, adj_prep)
        assert len(self.embedding.shape) == 3
        print('Using {} seconds to initialize NN decoder'.format(time.time()-t_s))

    def find_NN(self, ops, adj, ind, k = 10):
        assert len(ops.shape)==3
        ind_t1_list = []
        ind_tk_list = []
        with torch.no_grad():
            adj_prep, ops_prep, _ = preprocessing(adj, ops, **self.cfg['prep'])
            embeddings = self.model.encoder(ops_prep, adj_prep)
            for e in embeddings:
                dist = torch.sum( (self.embedding - e) ** 2, dim=[1,2])
                _, ind_t1 = torch.topk(dist, 1, largest=False)
                _, ind_tk = torch.topk(dist, k, largest=False)
                ind_t1_list.append(ind_t1.item())
                ind_tk_list.append(ind_tk.tolist())
        op_recon, adj_recon = self.ops[ind_t1_list], self.adj[ind_t1_list]
        op_recon_tk, adj_recon_tk = self.ops[ind_t1_list], self.adj[ind_t1_list]
        return op_recon, adj_recon, op_recon_tk, adj_recon_tk, ind_t1_list, ind_tk_list
