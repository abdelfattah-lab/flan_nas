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

class GIN_Emb_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout, emb_dim,
                 dropout, **kwargs):
        super(GIN_Emb_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.emb_dim = emb_dim
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.op_emb = nn.Embedding(
            self.input_dim,
            self.emb_dim
            )
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, self.emb_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)

    def _encoder(self, ops, adj):
        batch_size, node_num, opt_num = ops.shape
        x = self.dev_emb(ops).squeeze(1)
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        # print(x.shape) # [32, 7, 256]
        mu = self.fc1(x)
        # print(mu.shape) # [32, 7, 1]
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, 1))
        return mu
    
    def forward(self, ops, adj):
        mu = self._encoder(ops, adj)
        return mu


class GIN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout,
                 dropout, **kwargs):
        super(GIN_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)

    def _encoder(self, ops, adj):
        batch_size, node_num, opt_num = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        # print(x.shape) # [32, 7, 256]
        mu = self.fc1(x)
        # print(mu.shape) # [32, 7, 1]
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, 1))
        return mu
    
    def forward(self, ops, adj):
        mu = self._encoder(ops, adj)
        return mu

class GIN_Emb_Model_NDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout, emb_dim,
                 dropout, **kwargs):
        super(GIN_Emb_Model_NDS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.emb_dim = emb_dim
        self.op_emb_1 = nn.Embedding(input_dim, self.emb_dim)
        self.eps_1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps_1 = torch.nn.ModuleList()
        self.batch_norms_1 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_1.append(MLP(num_mlp_layers, self.emb_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_1.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_1.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1_1 = nn.Linear(self.hidden_dim, 64)

        self.mlps_2 = torch.nn.ModuleList()
        self.op_emb_2 = nn.Embedding(input_dim, self.emb_dim)
        self.eps_2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.batch_norms_2 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_2.append(MLP(num_mlp_layers, self.emb_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_2.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_2.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1_2 = nn.Linear(self.hidden_dim, 64)

        self.fc_comb_a = nn.Linear(128, 128)
        self.fc_comb_b = nn.Linear(128, self.latent_dim)

    def _encoder(self, ops1, adj1, ops2, adj2):
        batch_size_1, node_num_1, opt_num_1 = ops1.shape
        x_1 = self.op_emb_1(ops1).squeeze(1)
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj1.float(), x_1)
            agg = (1 + self.eps_2[l]) * x_1.view(batch_size_1 * node_num_1, -1) \
                  + neighbor.view(batch_size_1 * node_num_1, -1)
            x_1 = F.relu(self.batch_norms_1[l](self.mlps_1[l](agg)).view(batch_size_1, node_num_1, -1))
        mu_1 = self.fc1_1(x_1)
        batch_size_2, node_num_2, opt_num_2 = ops2.shape
        x_2 = self.op_emb_2(ops2).squeeze(1)
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj2.float(), x_2)
            agg = (1 + self.eps_2[l]) * x_2.view(batch_size_2 * node_num_2, -1) \
                  + neighbor.view(batch_size_2 * node_num_2, -1)
            x_2 = F.relu(self.batch_norms_2[l](self.mlps_2[l](agg)).view(batch_size_2, node_num_2, -1))
        mu_2 = self.fc1_2(x_2)
        
        mu = torch.cat((mu_1, mu_2), dim=2)
        # print(mu.shape)
        mu = F.relu(self.fc_comb_a(mu))
        mu = self.fc_comb_b(mu)
        # print(mu.shape)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, 1))
        return mu
    
    def forward(self, ops1, adj1, ops2, adj2):
        mu = self._encoder(ops1, adj1, ops2, adj2)
        return mu
    
class GIN_Model_NDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers, readout,
                 dropout, **kwargs):
        super(GIN_Model_NDS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.eps_1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps_1 = torch.nn.ModuleList()
        self.batch_norms_1 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_1.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_1.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_1.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1_1 = nn.Linear(self.hidden_dim, 64)

        self.mlps_2 = torch.nn.ModuleList()
        self.eps_2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.batch_norms_2 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_2.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_2.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_2.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1_2 = nn.Linear(self.hidden_dim, 64)

        self.fc_comb_a = nn.Linear(128, 128)
        self.fc_comb_b = nn.Linear(128, self.latent_dim)

    def _encoder(self, ops1, adj1, ops2, adj2):
        batch_size_1, node_num_1, opt_num_1 = ops1.shape
        x_1 = ops1
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj1.float(), x_1)
            agg = (1 + self.eps_2[l]) * x_1.view(batch_size_1 * node_num_1, -1) \
                  + neighbor.view(batch_size_1 * node_num_1, -1)
            x_1 = F.relu(self.batch_norms_1[l](self.mlps_1[l](agg)).view(batch_size_1, node_num_1, -1))
        mu_1 = self.fc1_1(x_1)
        batch_size_2, node_num_2, opt_num_2 = ops2.shape
        x_2 = ops2
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj2.float(), x_2)
            agg = (1 + self.eps_2[l]) * x_2.view(batch_size_2 * node_num_2, -1) \
                  + neighbor.view(batch_size_2 * node_num_2, -1)
            x_2 = F.relu(self.batch_norms_2[l](self.mlps_2[l](agg)).view(batch_size_2, node_num_2, -1))
        mu_2 = self.fc1_2(x_2)
        
        mu = torch.cat((mu_1, mu_2), dim=2)
        # print(mu.shape)
        mu = F.relu(self.fc_comb_a(mu))
        mu = self.fc_comb_b(mu)
        # print(mu.shape)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, 1))
        return mu
    
    def forward(self, ops1, adj1, ops2, adj2):
        mu = self._encoder(ops1, adj1, ops2, adj2)
        return mu

class GIN_ZCP_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, num_mlp_layers, readout,
                 dropout, **kwargs):
        super(GIN_ZCP_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zcp_dim = zcp_dim
        self.zcp_gin_dim = zcp_gin_dim
        self.emb_dim = 16
        self.num_layers = num_hops
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        # self.op_emb = nn.Embedding(input_dim, self.emb_dim)
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1 = nn.Linear(self.hidden_dim, self.zcp_gin_dim)

        self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
        self.bn1_zcp = nn.BatchNorm1d(64)
        self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
        self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)

        self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)


    def _encoder(self, ops, adj, zcp):
        batch_size, node_num, opt_num = ops.shape
        # batch_size, node_num = ops.shape
        x = ops
        # x = self.op_emb(ops)
        # print(ops)
        # print(x.shape)
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        mu = self.fc1(x)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.zcp_gin_dim))
        zcp = F.relu(self.bn1_zcp(self.fc1_zcp(zcp)))
        # print(mu.shape, zcp.shape)
        zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
        # print(zcp.shape)
        zcp = self.fcout(F.relu(self.bn2_zcp(self.fc2_zcp(zcp))))
        # print(zcp.shape)
        # print("*"*30)
        return zcp
    
    def forward(self, ops, adj, zcp):
        mu = self._encoder(ops, adj, zcp)
        return mu

class GIN_Emb_ZCP_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, num_mlp_layers, readout, emb_dim,
                 dropout, **kwargs):
        super(GIN_Emb_ZCP_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zcp_dim = zcp_dim
        self.zcp_gin_dim = zcp_gin_dim
        self.emb_dim = emb_dim
        self.op_emb = nn.Embedding(input_dim, self.emb_dim)
        self.num_layers = num_hops
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, self.emb_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1 = nn.Linear(self.hidden_dim, self.zcp_gin_dim)

        self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
        self.bn1_zcp = nn.BatchNorm1d(64)
        self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
        self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)

        self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)


    def _encoder(self, ops, adj, zcp):
        batch_size, node_num, opt_num = ops.shape
        x = self.op_emb(ops).squeeze(1)
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        mu = self.fc1(x)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.zcp_gin_dim))
        zcp = F.relu(self.bn1_zcp(self.fc1_zcp(zcp)))
        # print(mu.shape, zcp.shape)
        zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
        # print(zcp.shape)
        zcp = self.fcout(F.relu(self.bn2_zcp(self.fc2_zcp(zcp))))
        # print(zcp.shape)
        # print("*"*30)
        return zcp
    
    def forward(self, ops, adj, zcp):
        mu = self._encoder(ops, adj, zcp)
        return mu
    

class GIN_ZCP_Model_NDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, num_mlp_layers, readout,
                 dropout, **kwargs):
        super(GIN_ZCP_Model_NDS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zcp_dim = zcp_dim
        self.zcp_gin_dim = zcp_gin_dim
        self.num_layers = num_hops
        self.eps_1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps_1 = torch.nn.ModuleList()
        self.batch_norms_1 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_1.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_1.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_1.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1_1 = nn.Linear(self.hidden_dim, 64)

        self.mlps_2 = torch.nn.ModuleList()
        self.eps_2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.batch_norms_2 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_2.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_2.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_2.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1_2 = nn.Linear(self.hidden_dim, 64)

        self.fc_comb_a = nn.Linear(192, 128)
        self.fc_comb_b = nn.Linear(128, self.zcp_gin_dim)

        self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
        self.bn1_zcp = nn.BatchNorm1d(64)
        self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
        self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)
        
        self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)

    def _encoder(self, ops1, adj1, ops2, adj2, zcp):
        batch_size_1, node_num_1, opt_num_1 = ops1.shape
        x_1 = ops1
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj1.float(), x_1)
            agg = (1 + self.eps_2[l]) * x_1.view(batch_size_1 * node_num_1, -1) \
                  + neighbor.view(batch_size_1 * node_num_1, -1)
            x_1 = F.relu(self.batch_norms_1[l](self.mlps_1[l](agg)).view(batch_size_1, node_num_1, -1))
        mu_1 = self.fc1_1(x_1)
        batch_size_2, node_num_2, opt_num_2 = ops2.shape
        x_2 = ops2
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj2.float(), x_2)
            agg = (1 + self.eps_2[l]) * x_2.view(batch_size_2 * node_num_2, -1) \
                  + neighbor.view(batch_size_2 * node_num_2, -1)
            x_2 = F.relu(self.batch_norms_2[l](self.mlps_2[l](agg)).view(batch_size_2, node_num_2, -1))
        mu_2 = self.fc1_2(x_2)
        
        mu = torch.cat((mu_1, mu_2, zcp.unsqueeze(1)), dim=2)
        # print(mu.shape)
        mu = F.relu(self.fc_comb_a(mu))
        mu = self.fc_comb_b(mu)
        # print(mu.shape)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.zcp_gin_dim))

        zcp = F.relu(self.bn1_zcp(self.fc1_zcp(zcp)))
        zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
        zcp = self.fcout(F.relu(self.bn2_zcp(self.fc2_zcp(zcp))))
        return zcp
    
    def forward(self, ops1, adj1, ops2, adj2, zcp):
        mu = self._encoder(ops1, adj1, ops2, adj2, zcp)
        return mu

class GIN_Emb_ZCP_Model_NDS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, zcp_dim, zcp_gin_dim, num_hops, num_mlp_layers, readout, emb_dim,
                 dropout, **kwargs):
        super(GIN_Emb_ZCP_Model_NDS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.zcp_dim = zcp_dim
        self.zcp_gin_dim = zcp_gin_dim
        self.num_layers = num_hops
        self.emb_dim = emb_dim
        self.op_emb_1 = nn.Embedding(input_dim, self.emb_dim)
        self.eps_1 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps_1 = torch.nn.ModuleList()
        self.batch_norms_1 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_1.append(MLP(num_mlp_layers, self.emb_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_1.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_1.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1_1 = nn.Linear(self.hidden_dim, 64)
        self.op_emb_2 = nn.Embedding(input_dim, self.emb_dim)
        self.mlps_2 = torch.nn.ModuleList()
        self.eps_2 = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.batch_norms_2 = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps_2.append(MLP(num_mlp_layers, self.emb_dim, hidden_dim, hidden_dim))
            else:
                self.mlps_2.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms_2.append(nn.BatchNorm1d(hidden_dim))
        # convert to latent space
        self.fc1_2 = nn.Linear(self.hidden_dim, 64)

        self.fc_comb_a = nn.Linear(192, 128)
        self.fc_comb_b = nn.Linear(128, self.zcp_gin_dim)

        self.fc1_zcp = nn.Linear(self.zcp_dim, 64)
        self.bn1_zcp = nn.BatchNorm1d(64)
        self.fc2_zcp = nn.Linear(64 + self.zcp_gin_dim, 64 + self.zcp_gin_dim)
        self.bn2_zcp = nn.BatchNorm1d(64 + self.zcp_gin_dim)
        
        self.fcout = nn.Linear(self.zcp_gin_dim + 64, self.latent_dim)

    def _encoder(self, ops1, adj1, ops2, adj2, zcp):
        batch_size_1, node_num_1, opt_num_1 = ops1.shape
        x_1 = self.op_emb_1(ops1).squeeze(1)
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj1.float(), x_1)
            agg = (1 + self.eps_2[l]) * x_1.view(batch_size_1 * node_num_1, -1) \
                  + neighbor.view(batch_size_1 * node_num_1, -1)
            x_1 = F.relu(self.batch_norms_1[l](self.mlps_1[l](agg)).view(batch_size_1, node_num_1, -1))
        mu_1 = self.fc1_1(x_1)
        batch_size_2, node_num_2, opt_num_2 = ops2.shape
        x_2 = self.op_emb_2(ops2).squeeze(1)
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj2.float(), x_2)
            agg = (1 + self.eps_2[l]) * x_2.view(batch_size_2 * node_num_2, -1) \
                  + neighbor.view(batch_size_2 * node_num_2, -1)
            x_2 = F.relu(self.batch_norms_2[l](self.mlps_2[l](agg)).view(batch_size_2, node_num_2, -1))
        mu_2 = self.fc1_2(x_2)
        
        mu = torch.cat((mu_1, mu_2, zcp.unsqueeze(1)), dim=2)
        # print(mu.shape)
        mu = F.relu(self.fc_comb_a(mu))
        mu = self.fc_comb_b(mu)
        # print(mu.shape)
        mu = torch.nn.functional.adaptive_avg_pool2d(mu, (1, self.zcp_gin_dim))

        zcp = F.relu(self.bn1_zcp(self.fc1_zcp(zcp)))
        zcp = torch.cat((mu, zcp.unsqueeze(1)), dim=2).squeeze(1)
        zcp = self.fcout(F.relu(self.bn2_zcp(self.fc2_zcp(zcp))))
        return zcp
    
    def forward(self, ops1, adj1, ops2, adj2, zcp):
        mu = self._encoder(ops1, adj1, ops2, adj2, zcp)
        return mu

class GAE(nn.Module):
    def __init__(self, dims, normalize, reg_emb, reg_dec_l2, reg_dec_gp, dropout, **kwargs):
        super(GAE, self).__init__()
        self.encoder = Encoder(dims, normalize, reg_emb, dropout)
        self.decoder = Decoder(dims[-1], dims[0], dropout, **kwargs)
        self.reg_dec_l2 = reg_dec_l2
        self.reg_dec_gp = reg_dec_gp

    def forward(self, ops, adj):
        x, emb_loss = self.encoder(ops, adj)
        ops_recon, adj_recon = self.decoder(x)
        if self.reg_dec_l2:
            dec_loss_l2 = 0
            for p in self.decoder.parameters():
                dec_loss_l2 += torch.norm(p, 2)
            return ops_recon, adj_recon, emb_loss, dec_loss_l2, None
        if self.reg_dec_gp:
            return ops_recon, adj_recon, emb_loss, torch.FloatTensor([0.]).cuda(), x
        return ops_recon, adj_recon, emb_loss, torch.FloatTensor([0.]).cuda(), None

class GVAE(nn.Module):
    def __init__(self, dims, normalize, dropout, **kwargs):
        super(GVAE, self).__init__()
        self.encoder = VAEncoder(dims, normalize, dropout)
        self.decoder = Decoder(dims[-1], dims[0], dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, ops, adj):
        mu, logvar = self.encoder(ops, adj)
        z = self.reparameterize(mu, logvar)
        ops_recon, adj_recon = self.decoder(z)
        return ops_recon, adj_recon, mu, logvar

class Encoder(nn.Module):
    def __init__(self, dims, normalize, reg_emb, dropout):
        super(Encoder, self).__init__()
        self.gcs = nn.ModuleList(self.get_gcs(dims, dropout))
        self.normalize = normalize
        self.reg_emb = reg_emb

    def get_gcs(self,dims,dropout):
        gcs = []
        for k in range(len(dims)-1):
            gcs.append(GraphConvolution(dims[k],dims[k+1], dropout))
        return gcs

    def forward(self, ops, adj):
        if self.normalize:
            adj = normalize_adj(adj)
        x = ops
        for gc in self.gcs:
            x = gc(x, adj)
        if self.reg_emb:
            emb = x.mean(dim=1).squeeze()
            emb_loss = torch.mean(torch.norm(emb, p=2, dim=1))
            return x, emb_loss
        return x, torch.FloatTensor([0.]).cuda()

class VAEncoder(nn.Module):
    def __init__(self, dims, normalize, dropout):
        super(VAEncoder, self).__init__()
        self.gcs = nn.ModuleList(self.get_gcs(dims, dropout))
        self.gc_mu = GraphConvolution(dims[-2], dims[-1], dropout)
        self.gc_logvar = GraphConvolution(dims[-2], dims[-1], dropout)
        self.normalize = normalize

    def get_gcs(self,dims,dropout):
        gcs = []
        for k in range(len(dims)-1):
            gcs.append(GraphConvolution(dims[k],dims[k+1], dropout))
        return gcs

    def forward(self, ops, adj):
        if self.normalize:
            adj = normalize_adj(adj)
        x = ops
        for gc in self.gcs[:-1]:
            x = gc(x, adj)
        mu = self.gc_mu(x, adj)
        logvar = self.gc_logvar(x, adj)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_dim, dropout, activation_adj=torch.sigmoid, activation_ops=torch.sigmoid, adj_hidden_dim=None, ops_hidden_dim=None):
        super(Decoder, self).__init__()
        if adj_hidden_dim == None:
            self.adj_hidden_dim = embedding_dim
        if ops_hidden_dim == None:
            self.ops_hidden_dim = embedding_dim
        self.activation_adj = activation_adj
        self.activation_ops = activation_ops
        self.weight = torch.nn.Linear(embedding_dim, input_dim)
        self.dropout = dropout

    def forward(self, embedding):
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        ops = self.weight(embedding)
        adj = torch.matmul(embedding, embedding.permute(0, 2, 1))
        return self.activation_adj(ops), self.activation_adj(adj)

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
